"""
MLflow trainer — same training pipeline as manual_trainer.py,
wrapped in MLflow 3.12 experiment tracking.

Logs per run:
  - All hyperparameters
  - Accuracy, macro F1, weighted F1, per-class precision/recall
  - Confusion matrix heatmap (PNG artifact)
  - SHAP feature importance plot (PNG artifact)
  - Trained model artifact via mlflow.<flavour>.log_model()

Registers best ensemble to MLflow Model Registry as:
  <SYMBOL>-stock-predictor  (Staging → Production via MLflow UI)
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

load_dotenv()
logger = logging.getLogger(__name__)

SIGNAL_NAMES = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]


def _load_config() -> dict:
    cfg_path = Path(__file__).parents[2] / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _setup_mlflow(cfg: dict):
    import mlflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    return mlflow


def _log_confusion_matrix(y_true, y_pred, run) -> None:
    """Log a confusion matrix heatmap as MLflow artifact."""
    import os
    import mlflow as _mlflow
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SIGNAL_NAMES, yticklabels=SIGNAL_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    with tempfile.NamedTemporaryFile(suffix="_confusion_matrix.png", delete=False) as f:
        tmp_path = f.name
    fig.savefig(tmp_path, bbox_inches="tight")
    _mlflow.log_artifact(tmp_path)
    os.unlink(tmp_path)
    plt.close(fig)


def _log_shap_plot(model, X: np.ndarray, feature_names: list) -> None:
    """Log SHAP feature importance bar chart as MLflow artifact."""
    import os, mlflow as _mlflow
    try:
        import shap
        explainer = shap.TreeExplainer(model.model)
        shap_vals = explainer.shap_values(X[:min(200, len(X))])
        if isinstance(shap_vals, list):
            importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        else:
            importance = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(importance)[::-1][:20]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([feature_names[i] for i in reversed(top_idx)],
                [importance[i] for i in reversed(top_idx)])
        ax.set_title("Top-20 Feature Importance (SHAP)")
        ax.set_xlabel("Mean |SHAP value|")
        with tempfile.NamedTemporaryFile(suffix="_shap_importance.png", delete=False) as f:
            tmp_path = f.name
        fig.savefig(tmp_path, bbox_inches="tight")
        _mlflow.log_artifact(tmp_path)
        os.unlink(tmp_path)
        plt.close(fig)
    except Exception as e:
        logger.warning("SHAP plot skipped: %s", e)


def _compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":       float(accuracy_score(y_true, y_pred)),
        "macro_f1":       float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "weighted_f1":    float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "buy_f1":         float(f1_score(y_true, y_pred, labels=[3, 4], average="macro", zero_division=0)),
        "sell_f1":        float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
    }


def train_symbol_mlflow(
    symbol: str,
    cfg: Optional[dict] = None,
    skip_autogluon: bool = False,
) -> Dict[str, dict]:
    """Full MLflow-tracked training for one symbol."""
    import mlflow
    import mlflow.lightgbm
    import mlflow.sklearn

    from src.data.features import load_processed
    from src.models.lgbm_model import LGBMModel
    from src.models.xgboost_model import XGBoostModel
    from src.models.lstm_model import LSTMModel
    from src.models.autogluon_model import AutoGluonModel
    from src.models.ensemble_model import StackingEnsemble

    cfg = cfg or _load_config()
    tr_cfg = cfg["training"]
    _setup_mlflow(cfg)

    df = load_processed(symbol)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    # Load raw close prices for Chronos-2 (processed parquet only has indicators)
    raw_path = Path(cfg["paths"]["raw_data"]) / f"{symbol}.csv"
    close = None
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path, index_col="date", parse_dates=True)
        raw_df = raw_df.reindex(df.index).ffill()
        close_col = next((c for c in ["close", "Close"] if c in raw_df.columns), None)
        if close_col:
            close = raw_df[close_col].values.astype(np.float32)

    split_idx = int(len(X) * (1 - tr_cfg["test_size"]))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    close_train = close[:split_idx] if close is not None else None
    close_test  = close[split_idx:] if close is not None else None
    val_split = int(len(X_train) * 0.8)

    all_metrics = {}

    # ── LightGBM ──────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"lgbm_{symbol}"):
        mlflow.log_params({"symbol": symbol, "model": "LightGBM", "optuna_trials": tr_cfg["optuna_trials"]})
        lgbm = LGBMModel(symbol, n_trials=tr_cfg["optuna_trials"], random_state=tr_cfg["random_state"])
        lgbm.fit(X_train[:val_split], y_train[:val_split],
                 X_train[val_split:], y_train[val_split:],
                 feature_names=feature_cols)
        preds = lgbm.predict(X_test)
        m = _compute_metrics(y_test, preds)
        mlflow.log_metrics(m)
        mlflow.log_params(lgbm.best_params)
        _log_confusion_matrix(y_test, preds, None)
        _log_shap_plot(lgbm, X_test, feature_cols)
        mlflow.lightgbm.log_model(lgbm.model, "lgbm_model")
        all_metrics["lgbm"] = m
        logger.info("%s LightGBM — Accuracy=%.4f Macro-F1=%.4f", symbol, m["accuracy"], m["macro_f1"])

    # ── XGBoost ───────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"xgb_{symbol}"):
        mlflow.log_params({"symbol": symbol, "model": "XGBoost"})
        xgb_m = XGBoostModel(symbol, random_state=tr_cfg["random_state"])
        xgb_m.fit(X_train[:val_split], y_train[:val_split],
                   X_train[val_split:], y_train[val_split:],
                   feature_names=feature_cols)
        preds = xgb_m.predict(X_test)
        m = _compute_metrics(y_test, preds)
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(xgb_m.model, "xgb_model")
        all_metrics["xgb"] = m
        logger.info("%s XGBoost — Accuracy=%.4f Macro-F1=%.4f", symbol, m["accuracy"], m["macro_f1"])

    # ── PyTorch LSTM ──────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"lstm_{symbol}"):
        mlflow.log_params({"symbol": symbol, "model": "PyTorch-LSTM", "epochs": 50})
        lstm_m = LSTMModel(symbol, epochs=50, random_state=tr_cfg["random_state"])
        lstm_m.fit(X_train[:val_split], y_train[:val_split],
                   X_train[val_split:], y_train[val_split:],
                   feature_names=feature_cols)
        preds = lstm_m.predict(X_test)
        m = _compute_metrics(y_test, preds)
        mlflow.log_metrics(m)
        all_metrics["lstm"] = m
        logger.info("%s LSTM — Accuracy=%.4f Macro-F1=%.4f", symbol, m["accuracy"], m["macro_f1"])

    # ── AutoGluon ─────────────────────────────────────────────────────
    if not skip_autogluon:
        with mlflow.start_run(run_name=f"autogluon_{symbol}"):
            mlflow.log_params({"symbol": symbol, "model": "AutoGluon", "time_limit": 300})
            ag_m = AutoGluonModel(symbol, time_limit=300, random_state=tr_cfg["random_state"])
            ag_m.fit(X_train[:val_split], y_train[:val_split],
                     X_train[val_split:], y_train[val_split:],
                     feature_names=feature_cols)
            preds = ag_m.predict(X_test, feature_names=feature_cols)
            m = _compute_metrics(y_test, preds)
            mlflow.log_metrics(m)
            all_metrics["autogluon"] = m

    # ── Stacking Ensemble (OOF) — main run ───────────────────────────
    with mlflow.start_run(run_name=f"ensemble_{symbol}"):
        mlflow.log_params({
            "symbol": symbol, "model": "StackingEnsemble-OOF",
            "cv_folds": tr_cfg["cv_folds"], "use_smote": tr_cfg["use_smote"],
        })
        ensemble = StackingEnsemble(
            symbol,
            cv_folds=tr_cfg["cv_folds"],
            random_state=tr_cfg["random_state"],
            use_smote=tr_cfg["use_smote"],
            lgbm_trials=min(tr_cfg["optuna_trials"], 20),
            lstm_epochs=30,
        )
        ensemble.fit(X_train, y_train, feature_names=feature_cols, close_train=close_train)
        preds = ensemble.predict(X_test, close=close_test)
        m = _compute_metrics(y_test, preds)
        mlflow.log_metrics(m)

        # Log OOF per-fold F1 scores
        for model_name, fold_scores in ensemble.oof_f1_scores.items():
            for fold_i, score in enumerate(fold_scores):
                mlflow.log_metric(f"oof_{model_name}_fold{fold_i+1}", score)

        _log_confusion_matrix(y_test, preds, None)
        all_metrics["ensemble"] = m

        # Register to Model Registry
        registry_name = cfg["mlflow"]["registry_model_name"].format(symbol=symbol)
        mlflow.lightgbm.log_model(
            ensemble.meta_learner,
            "ensemble_meta_learner",
            registered_model_name=registry_name,
        )
        logger.info("%s Ensemble registered as '%s'", symbol, registry_name)
        logger.info("%s Ensemble — Accuracy=%.4f Macro-F1=%.4f", symbol, m["accuracy"], m["macro_f1"])

    return all_metrics


def train_all_mlflow(
    symbols: Optional[List[str]] = None,
    skip_autogluon: bool = False,
) -> dict:
    """Train all configured symbols with MLflow tracking."""
    cfg = _load_config()
    symbols = symbols or cfg["stocks"]["symbols"]
    all_metrics = {}

    for symbol in symbols:
        try:
            all_metrics[symbol] = train_symbol_mlflow(symbol, cfg=cfg, skip_autogluon=skip_autogluon)
        except Exception as e:
            logger.error("MLflow training failed for %s: %s", symbol, e, exc_info=True)

    logger.info("MLflow training complete for %d symbols", len(all_metrics))
    return all_metrics
