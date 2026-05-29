"""
Evaluator — loads trained artifacts and produces a side-by-side comparison table
for all models × all symbols. Outputs terminal table + CSV + confusion matrix PNGs.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

logger = logging.getLogger(__name__)
SIGNAL_NAMES = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]


def _load_config() -> dict:
    cfg_path = Path(__file__).parents[2] / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _load_test_data(symbol: str, cfg: dict):
    """Load processed features and return test split."""
    from src.data.features import load_processed
    df = load_processed(symbol)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)
    split_idx = int(len(X) * (1 - cfg["training"]["test_size"]))
    return X[split_idx:], y[split_idx:], feature_cols


def _metrics_row(y_true, y_pred) -> dict:
    return {
        "Accuracy":       round(accuracy_score(y_true, y_pred), 4),
        "Macro F1":       round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "Weighted F1":    round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Buy F1":         round(f1_score(y_true, y_pred, labels=[3, 4], average="macro", zero_division=0), 4),
        "Sell F1":        round(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0), 4),
    }


def _save_confusion_matrix(y_true, y_pred, symbol: str, model_name: str, reports_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SIGNAL_NAMES, yticklabels=SIGNAL_NAMES, ax=ax)
    ax.set_title(f"{symbol} — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    out = reports_dir / f"{symbol}_{model_name}_cm.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Confusion matrix saved: %s", out)


def evaluate_symbol(
    symbol: str,
    cfg: Optional[dict] = None,
    include_mlflow: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all available trained models for one symbol.
    Returns a DataFrame with one row per model, columns = metrics.
    """
    cfg = cfg or _load_config()
    model_dir = Path(cfg["paths"]["manual_models"])
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test, feature_cols = _load_test_data(symbol, cfg)
    rows = []

    # ── Manual models ─────────────────────────────────────────────────
    model_files = {
        "LightGBM (manual)":   (model_dir / f"{symbol}_lgbm.pkl",     "pkl"),
        "XGBoost (manual)":    (model_dir / f"{symbol}_xgb.pkl",      "pkl"),
        "LSTM (manual)":       (model_dir / f"{symbol}_lstm.pt",      "pt"),
        "AutoGluon (manual)":  (model_dir / f"{symbol}_autogluon.pkl","ag"),
        "Ensemble (manual)":   (model_dir / f"{symbol}_ensemble.pkl", "pkl"),
    }

    for model_name, (path, kind) in model_files.items():
        if not path.exists():
            logger.debug("%s: %s not found — skipping", symbol, path.name)
            continue
        try:
            if kind == "pkl":
                import pickle
                with open(path, "rb") as f:
                    model = pickle.load(f)
                preds = model.predict(X_test)
            elif kind == "pt":
                from src.models.lstm_model import LSTMModel
                model = LSTMModel.load(str(path))
                preds = model.predict(X_test)
            elif kind == "ag":
                from src.models.autogluon_model import AutoGluonModel
                model = AutoGluonModel.load(str(path))
                preds = model.predict(X_test, feature_names=feature_cols)

            m = _metrics_row(y_test, preds)
            m["Symbol"] = symbol
            m["Model"] = model_name
            rows.append(m)
            _save_confusion_matrix(y_test, preds, symbol, model_name.replace(" ", "_"), reports_dir)
        except Exception as e:
            logger.warning("%s %s evaluation failed: %s", symbol, model_name, e)

    # ── MLflow production model ───────────────────────────────────────
    if include_mlflow:
        try:
            import mlflow
            registry_name = cfg["mlflow"]["registry_model_name"].format(symbol=symbol)
            mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
            model = mlflow.lightgbm.load_model(f"models:/{registry_name}/Production")
            preds = model.predict(X_test)
            m = _metrics_row(y_test, preds)
            m["Symbol"] = symbol
            m["Model"] = "Ensemble (MLflow Prod)"
            rows.append(m)
        except Exception as e:
            logger.debug("MLflow production model not available for %s: %s", symbol, e)

    return pd.DataFrame(rows)


def compare_all(
    symbols: Optional[List[str]] = None,
    include_mlflow: bool = True,
) -> pd.DataFrame:
    """
    Full comparison table for all symbols and all models.
    Saves comparison CSV to reports/.
    """
    cfg = _load_config()
    symbols = symbols or cfg["stocks"]["symbols"]

    all_rows = []
    for symbol in symbols:
        try:
            df = evaluate_symbol(symbol, cfg=cfg, include_mlflow=include_mlflow)
            all_rows.append(df)
        except Exception as e:
            logger.error("Evaluation failed for %s: %s", symbol, e)

    if not all_rows:
        logger.warning("No evaluation results available.")
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    result = result[["Symbol", "Model", "Accuracy", "Macro F1", "Weighted F1", "Buy F1", "Sell F1"]]
    result = result.sort_values(["Symbol", "Macro F1"], ascending=[True, False])

    # Save CSV
    reports_dir = Path(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result.to_csv(csv_path, index=False)
    logger.info("Comparison saved to %s", csv_path)

    # Print to terminal
    _print_comparison(result)
    return result


def _print_comparison(df: pd.DataFrame) -> None:
    print(f"\n{'='*80}")
    print("  MODEL COMPARISON — ALL SYMBOLS")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")
