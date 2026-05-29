"""
LightGBM multi-class model with Optuna hyperparameter tuning and SHAP explainability.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class LGBMModel:
    """LightGBM 5-class classifier with Optuna tuning."""

    def __init__(self, symbol: str, n_trials: int = 50, random_state: int = 42):
        self.symbol = symbol
        self.n_trials = n_trials
        self.random_state = random_state
        self.model = None
        self.feature_names: list = []
        self.best_params: dict = {}

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        import lightgbm as lgb
        params = {
            "objective":        "multiclass",
            "num_class":        5,
            "metric":           "multi_logloss",
            "verbosity":        -1,
            "boosting_type":    "gbdt",
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
            "max_depth":        trial.suggest_int("max_depth", 4, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state":     self.random_state,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average="macro")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> "LGBMModel":
        import lightgbm as lgb
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        self.best_params = study.best_params
        self.best_params.update({
            "objective": "multiclass",
            "num_class": 5,
            "metric": "multi_logloss",
            "verbosity": -1,
            "random_state": self.random_state,
        })
        logger.info("%s LGBM best macro-F1=%.4f | params=%s", self.symbol, study.best_value, self.best_params)

        self.model = lgb.LGBMClassifier(**self.best_params)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            return explainer.shap_values(X)
        except Exception as e:
            logger.warning("SHAP failed: %s", e)
            return None

    def top_features(self, X: np.ndarray, n: int = 10) -> Dict[str, float]:
        """Return top-N features by mean absolute SHAP value."""
        shap_vals = self.shap_values(X)
        if shap_vals is None:
            return {}
        if isinstance(shap_vals, list):
            importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        else:
            importance = np.abs(shap_vals).mean(axis=0)
        indices = np.argsort(importance)[::-1][:n]
        return {self.feature_names[i]: float(importance[i]) for i in indices}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("LGBMModel saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "LGBMModel":
        with open(path, "rb") as f:
            return pickle.load(f)
