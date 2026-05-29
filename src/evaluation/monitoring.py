"""
Evidently AI drift monitoring — detects data, target, and prediction drift.

Runs nightly (configurable via cron in config.yaml).
Saves an interactive HTML report to reports/drift/<date>_drift_report.html.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    cfg_path = Path(__file__).parents[2] / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _load_reference_and_current(symbol: str, cfg: dict):
    """
    Returns (reference_df, current_df) from processed features.
    Reference = training split (80%), Current = test split (20%).
    """
    from src.data.features import load_processed
    df = load_processed(symbol)
    split_idx = int(len(df) * (1 - cfg["training"]["test_size"]))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def _get_predictions(symbol: str, X: np.ndarray, feature_cols: list, cfg: dict) -> Optional[np.ndarray]:
    """Load best manual ensemble and generate predictions."""
    model_path = Path(cfg["paths"]["manual_models"]) / f"{symbol}_ensemble.pkl"
    if not model_path.exists():
        return None
    try:
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model.predict(X)
    except Exception as e:
        logger.warning("%s: could not load ensemble for drift check: %s", symbol, e)
        return None


def run_drift_report(
    symbols: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Generate Evidently AI drift report for all symbols.
    Returns {symbol: report_path}.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.metrics import ColumnDriftMetric
    except ImportError:
        logger.error("evidently not installed. Run: pip install evidently==0.7.21")
        return {}

    cfg = _load_config()
    symbols = symbols or cfg["stocks"]["symbols"]
    drift_dir = Path(cfg["paths"]["reports"]) / "drift"
    drift_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_paths = {}

    for symbol in symbols:
        try:
            ref_df, curr_df = _load_reference_and_current(symbol, cfg)
            feature_cols = [c for c in ref_df.columns if c != "label"]

            # Add prediction columns — both current AND reference must use model outputs
            curr_preds = _get_predictions(symbol, curr_df[feature_cols].values.astype(np.float32), feature_cols, cfg)
            ref_preds  = _get_predictions(symbol, ref_df[feature_cols].values.astype(np.float32),  feature_cols, cfg)
            if curr_preds is not None:
                curr_df = curr_df.copy()
                curr_df["prediction"] = curr_preds
                ref_df = ref_df.copy()
                # Use actual model predictions on reference data so drift compares
                # prediction distributions — not predictions vs ground-truth labels
                ref_df["prediction"] = ref_preds if ref_preds is not None else ref_df["label"]

            # Build Evidently report
            report = Report(metrics=[
                DataDriftPreset(),
                TargetDriftPreset(),
            ])

            # Select feature subset (top 30 for performance)
            selected_cols = feature_cols[:30] + ["label"]
            if "prediction" in curr_df.columns:
                selected_cols += ["prediction"]

            ref_subset  = ref_df[[c for c in selected_cols if c in ref_df.columns]].rename(columns={"label": "target"})
            curr_subset = curr_df[[c for c in selected_cols if c in curr_df.columns]].rename(columns={"label": "target"})

            report.run(reference_data=ref_subset, current_data=curr_subset)

            out_path = drift_dir / f"{symbol}_{date_str}_drift_report.html"
            report.save_html(str(out_path))
            report_paths[symbol] = str(out_path)
            logger.info("%s drift report saved: %s", symbol, out_path)

            # Check thresholds
            result = report.as_dict()
            _check_drift_thresholds(symbol, result, cfg)

        except Exception as e:
            logger.error("Drift monitoring failed for %s: %s", symbol, e, exc_info=True)

    return report_paths


def _check_drift_thresholds(symbol: str, report_dict: dict, cfg: dict) -> None:
    """Log alerts if any drift metric exceeds configured thresholds."""
    threshold = cfg["monitoring"]["drift_threshold_psi"]
    try:
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})
            drift_score = result.get("drift_score", 0)
            feature = result.get("column_name", "unknown")
            if drift_score and drift_score > threshold:
                logger.warning(
                    "DRIFT ALERT: %s | feature=%s | drift_score=%.4f > threshold=%.2f",
                    symbol, feature, drift_score, threshold,
                )
    except Exception:
        pass


def get_latest_drift_status(symbol: str) -> dict:
    """Return summary of the latest drift report for a symbol (for API use)."""
    cfg = _load_config()
    drift_dir = Path(cfg["paths"]["reports"]) / "drift"
    if not drift_dir.exists():
        return {"status": "no_reports", "symbol": symbol}

    reports = sorted(drift_dir.glob(f"{symbol}_*_drift_report.html"))
    if not reports:
        return {"status": "no_reports", "symbol": symbol}

    latest = reports[-1]
    return {
        "status": "ok",
        "symbol": symbol,
        "latest_report": str(latest),
        "report_date": (m.group(1) if (m := re.search(r"\d{8}_\d{6}", latest.stem)) else latest.stem),
    }
