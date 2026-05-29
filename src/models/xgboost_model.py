"""
XGBoost multi-class model with early stopping.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost 5-class softprob classifier with early stopping."""

    def __init__(self, symbol: str, random_state: int = 42):
        self.symbol = symbol
        self.random_state = random_state
        self.model = None
        self.feature_names: list = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> "XGBoostModel":
        import xgboost as xgb

        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=50,
            eval_metric="mlogloss",
            random_state=self.random_state,
            verbosity=0,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info(
            "%s XGBoost fitted | best_iteration=%d",
            self.symbol,
            self.model.best_iteration,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def top_features(self, n: int = 10) -> Dict[str, float]:
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:n]
        return {self.feature_names[i]: float(importance[i]) for i in indices}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("XGBoostModel saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        with open(path, "rb") as f:
            return pickle.load(f)
