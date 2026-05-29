"""
AutoGluon tabular baseline — independent benchmark (NOT part of the stacking ensemble).
Trains AutoGluon on the same feature set and reports its performance for comparison.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AutoGluonModel:
    """
    AutoGluon TabularPredictor wrapper.
    Used only as a standalone baseline to validate the manual ensemble quality.
    """

    def __init__(self, symbol: str, time_limit: int = 300, random_state: int = 42):
        self.symbol = symbol
        self.time_limit = time_limit
        self.random_state = random_state
        self.predictor = None
        self._save_path: Optional[str] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        save_path: Optional[str] = None,
    ) -> "AutoGluonModel":
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            logger.warning("AutoGluon not installed. Skipping baseline. pip install autogluon")
            return self

        cols = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        train_df = pd.DataFrame(X_train, columns=cols)
        train_df["label"] = y_train.astype(int)
        val_df = pd.DataFrame(X_val, columns=cols)
        val_df["label"] = y_val.astype(int)

        self._save_path = save_path or f"/tmp/autogluon_{self.symbol}"
        if Path(self._save_path).exists():
            shutil.rmtree(self._save_path)

        logger.info("%s AutoGluon: fitting (time_limit=%ds) ...", self.symbol, self.time_limit)
        self.predictor = TabularPredictor(
            label="label",
            path=self._save_path,
            eval_metric="f1_macro",
            verbosity=0,
        ).fit(
            train_df,
            tuning_data=val_df,   # val kept separate — same split as all other models
            time_limit=self.time_limit,
            presets="medium_quality",
            excluded_model_types=["KNN"],  # KNN very slow on large datasets
        )
        logger.info("%s AutoGluon: fitted", self.symbol)
        return self

    def predict(self, X: np.ndarray, feature_names: Optional[list] = None) -> np.ndarray:
        if self.predictor is None:
            return np.full(len(X), 2)
        cols = feature_names or [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        return self.predictor.predict(df).to_numpy().astype(int)

    def predict_proba(self, X: np.ndarray, feature_names: Optional[list] = None) -> np.ndarray:
        if self.predictor is None:
            return np.full((len(X), 5), 0.2)
        cols = feature_names or [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        proba = self.predictor.predict_proba(df)
        # Ensure columns are ordered 0–4
        for c in range(5):
            if c not in proba.columns:
                proba[c] = 0.0
        return proba[[0, 1, 2, 3, 4]].to_numpy().astype(float)

    def leaderboard(self) -> Optional[pd.DataFrame]:
        if self.predictor is None:
            return None
        return self.predictor.leaderboard(silent=True)

    def save(self, path: str) -> None:
        """AutoGluon saves its own artifacts in self._save_path; we just record the path."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"save_path": self._save_path, "symbol": self.symbol}, f)
        logger.info("AutoGluonModel pointer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "AutoGluonModel":
        import pickle
        with open(path, "rb") as f:
            meta = pickle.load(f)
        obj = cls(symbol=meta["symbol"])
        obj._save_path = meta["save_path"]
        try:
            from autogluon.tabular import TabularPredictor
            obj.predictor = TabularPredictor.load(meta["save_path"])
        except Exception as e:
            logger.warning("Could not load AutoGluon predictor: %s", e)
        return obj
