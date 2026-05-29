"""
Manual trainer — trains all models and saves artifacts locally.
No MLflow logging. Purely for comparison against MLflow-tracked results.

Artifact locations (one file per stock per model):
  models/manual/<SYMBOL>_lgbm.pkl
  models/manual/<SYMBOL>_xgb.pkl
  models/manual/<SYMBOL>_lstm.pt
  models/manual/<SYMBOL>_autogluon.pkl  (pointer to AutoGluon's own save dir)
  models/manual/<SYMBOL>_ensemble.pkl
  models/manual/metadata.json

After saving, DVC tracks each file:
  dvc add models/manual/<SYMBOL>_lgbm.pkl  → creates .dvc pointer file
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = Path(__file__).parents[2] / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _dvc_track(file_path: str) -> None:
    """Run dvc add on a file. Silently skips if DVC is not initialised."""
    try:
        result = subprocess.run(
            ["dvc", "add", file_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            logger.info("DVC tracked: %s", file_path)
        else:
            logger.debug("dvc add skipped for %s: %s", file_path, result.stderr.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("DVC not available — skipping tracking for %s", file_path)


def _time_split(df: pd.DataFrame, test_size: float = 0.2):
    """Chronological train/test split — no shuffle."""
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_symbol(
    symbol: str,
    cfg: Optional[dict] = None,
    skip_autogluon: bool = False,
) -> Dict[str, dict]:
    """
    Full training run for a single symbol.
    Returns metrics dict keyed by model name.
    """
    from src.data.features import load_processed
    from src.models.lgbm_model import LGBMModel
    from src.models.xgboost_model import XGBoostModel
    from src.models.lstm_model import LSTMModel
    from src.models.autogluon_model import AutoGluonModel
    from src.models.ensemble_model import StackingEnsemble

    cfg = cfg or _load_config()
    tr_cfg = cfg["training"]
    model_dir = Path(cfg["paths"]["manual_models"])
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load processed features
    logger.info("=" * 60)
    logger.info("Training: %s", symbol)
    df = load_processed(symbol)

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    # Load raw close prices for Chronos-2 (not in processed features parquet)
    raw_path = Path(cfg["paths"]["raw_data"]) / f"{symbol}.csv"
    close = None
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path, index_col="date", parse_dates=True)
        raw_df = raw_df.reindex(df.index).ffill()
        close_col = next((c for c in ["close", "Close"] if c in raw_df.columns), None)
        if close_col:
            close = raw_df[close_col].values.astype(np.float32)

    # Chronological split
    split_idx = int(len(X) * (1 - tr_cfg["test_size"]))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    close_train = close[:split_idx] if close is not None else None
    close_test  = close[split_idx:] if close is not None else None

    metrics = {}

    # ── LightGBM ──────────────────────────────────────────────────────
    lgbm = LGBMModel(symbol, n_trials=tr_cfg["optuna_trials"], random_state=tr_cfg["random_state"])
    val_split = int(len(X_train) * 0.8)
    lgbm.fit(X_train[:val_split], y_train[:val_split],
             X_train[val_split:], y_train[val_split:],
             feature_names=feature_cols)
    preds = lgbm.predict(X_test)
    metrics["lgbm"] = _compute_metrics(y_test, preds, lgbm.predict_proba(X_test))
    lgbm_path = str(model_dir / f"{symbol}_lgbm.pkl")
    lgbm.save(lgbm_path)
    _dvc_track(lgbm_path)

    # ── XGBoost ───────────────────────────────────────────────────────
    xgb_m = XGBoostModel(symbol, random_state=tr_cfg["random_state"])
    xgb_m.fit(X_train[:val_split], y_train[:val_split],
               X_train[val_split:], y_train[val_split:],
               feature_names=feature_cols)
    preds = xgb_m.predict(X_test)
    metrics["xgb"] = _compute_metrics(y_test, preds, xgb_m.predict_proba(X_test))
    xgb_path = str(model_dir / f"{symbol}_xgb.pkl")
    xgb_m.save(xgb_path)
    _dvc_track(xgb_path)

    # ── PyTorch LSTM ──────────────────────────────────────────────────
    lstm_m = LSTMModel(symbol, epochs=50, random_state=tr_cfg["random_state"])
    lstm_m.fit(X_train[:val_split], y_train[:val_split],
               X_train[val_split:], y_train[val_split:],
               feature_names=feature_cols)
    preds = lstm_m.predict(X_test)
    metrics["lstm"] = _compute_metrics(y_test, preds, lstm_m.predict_proba(X_test))
    lstm_path = str(model_dir / f"{symbol}_lstm.pt")
    lstm_m.save(lstm_path)
    _dvc_track(lstm_path)

    # ── AutoGluon (optional) ──────────────────────────────────────────
    if not skip_autogluon:
        ag_m = AutoGluonModel(symbol, time_limit=300, random_state=tr_cfg["random_state"])
        ag_save = str(model_dir / f"{symbol}_autogluon")
        ag_m.fit(X_train[:val_split], y_train[:val_split],
                 X_train[val_split:], y_train[val_split:],
                 feature_names=feature_cols, save_path=ag_save)
        preds = ag_m.predict(X_test, feature_names=feature_cols)
        metrics["autogluon"] = _compute_metrics(y_test, preds, ag_m.predict_proba(X_test, feature_cols))
        ag_path = str(model_dir / f"{symbol}_autogluon.pkl")
        ag_m.save(ag_path)
        _dvc_track(ag_path)

    # ── Stacking Ensemble (OOF) ───────────────────────────────────────
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
    metrics["ensemble"] = _compute_metrics(y_test, preds, ensemble.predict_proba(X_test, close=close_test))
    ens_path = str(model_dir / f"{symbol}_ensemble.pkl")
    ensemble.save(ens_path)
    _dvc_track(ens_path)

    _print_metrics_table(symbol, metrics)
    return metrics


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy":       float(accuracy_score(y_true, y_pred)),
        "macro_f1":       float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "weighted_f1":    float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "buy_precision":  float(f1_score(y_true, y_pred, labels=[3, 4], average="macro", zero_division=0)),
        "sell_precision": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "report":         classification_report(y_true, y_pred,
                              target_names=["Strong Sell","Sell","Hold","Buy","Strong Buy"],
                              zero_division=0),
    }


def _print_metrics_table(symbol: str, metrics: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  {symbol} — Manual Training Results")
    print(f"{'='*70}")
    print(f"  {'Model':<18} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12} {'Buy F1':>8} {'Sell F1':>8}")
    print(f"  {'-'*68}")
    for name, m in metrics.items():
        print(f"  {name:<18} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} "
              f"{m['weighted_f1']:>12.4f} {m['buy_precision']:>8.4f} {m['sell_precision']:>8.4f}")
    print(f"{'='*70}\n")


def _save_metadata(symbols: List[str], metrics: dict, cfg: dict) -> None:
    model_dir = Path(cfg["paths"]["manual_models"])
    meta = {
        "trained_at":  datetime.now().isoformat(),
        "symbols":     symbols,
        "metrics":     {s: {m: {k: v for k, v in mv.items() if k != "report"}
                           for m, mv in sm.items()}
                       for s, sm in metrics.items()},
    }
    meta_path = model_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)


def train_all(
    symbols: Optional[List[str]] = None,
    skip_autogluon: bool = False,
) -> dict:
    """Train all configured symbols and return combined metrics."""
    cfg = _load_config()
    symbols = symbols or cfg["stocks"]["symbols"]
    all_metrics = {}

    for symbol in symbols:
        try:
            all_metrics[symbol] = train_symbol(symbol, cfg=cfg, skip_autogluon=skip_autogluon)
        except Exception as e:
            logger.error("Training failed for %s: %s", symbol, e, exc_info=True)

    _save_metadata(symbols, all_metrics, cfg)
    logger.info("Manual training complete for %d symbols", len(all_metrics))
    return all_metrics
