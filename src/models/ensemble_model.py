"""
OOF Stacking Ensemble — LightGBM + XGBoost + PyTorch LSTM + Chronos-2 → LightGBM meta-learner.

Critical: meta-learner is trained ONLY on Out-of-Fold predictions from TimeSeriesSplit.
          No in-sample predictions are used — prevents meta-learner overfitting.

Procedure:
  Phase A — OOF loop (4 iterations of TimeSeriesSplit):
    For each fold k in [1..4]:
      1. Apply SMOTE to train_fold only
      2. Fit StandardScaler on train_fold only, transform val_fold
      3. Train LightGBM, XGBoost, LSTM on train_fold
      4. Run Chronos-2 (zero-shot — no training)
      5. Predict val_fold → store as OOF[k+1]
    → OOF Pool: rows from folds 2–5 (~80% of training data, never seen by any model)

  Phase B — Meta-learner:
    6. Concatenate OOF predictions [lgbm_proba, xgb_proba, lstm_proba, chronos_proba] (20 features)
    7. Train LightGBM meta-learner on OOF Pool

  Phase C — Full retrain:
    8. Retrain all Level-0 models on FULL training set

  Inference:
    Level-0 (full retrain) → 20-feature meta input → Meta-learner → Final prediction
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def _apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """SMOTE on training portion only. Falls back gracefully if class has < 6 samples."""
    try:
        from imblearn.over_sampling import SMOTE
        min_samples = min(np.bincount(y))
        k = min(5, min_samples - 1)
        if k < 1:
            return X, y
        sm = SMOTE(k_neighbors=k, random_state=random_state)
        return sm.fit_resample(X, y)
    except Exception as e:
        logger.warning("SMOTE skipped: %s", e)
        return X, y


class StackingEnsemble:
    """OOF stacking ensemble with LightGBM meta-learner."""

    def __init__(
        self,
        symbol: str,
        cv_folds: int = 5,
        random_state: int = 42,
        use_smote: bool = True,
        lgbm_trials: int = 30,
        lstm_epochs: int = 30,
    ):
        self.symbol = symbol
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.use_smote = use_smote
        self.lgbm_trials = lgbm_trials
        self.lstm_epochs = lstm_epochs

        # Level-0 models (full-train versions for inference)
        self.lgbm = None
        self.xgb = None
        self.lstm = None
        self.chronos = None

        # Level-1 meta-learner
        self.meta_learner = None
        self.meta_scaler = StandardScaler()

        self.feature_names: List[str] = []
        self.oof_f1_scores: dict = {}

    def _fit_fold_models(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        close_tr: Optional[np.ndarray] = None,
        close_val: Optional[np.ndarray] = None,
        seq_len: int = 20,
    ) -> np.ndarray:
        """
        Train Level-0 models on one fold's train portion and predict val portion.
        Returns meta-features array of shape (len(X_val), 20).
        """
        from src.models.lgbm_model import LGBMModel
        from src.models.xgboost_model import XGBoostModel
        from src.models.lstm_model import LSTMModel
        from src.models.chronos_model import ChronosModel

        # Per-fold scaler (fit on train only)
        scaler = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        # Per-fold SMOTE (on scaled train only)
        if self.use_smote:
            X_tr_sm, y_tr_sm = _apply_smote(X_tr_sc, y_tr, self.random_state)
        else:
            X_tr_sm, y_tr_sm = X_tr_sc, y_tr

        n_val = len(X_val)
        meta = np.zeros((n_val, 20))

        # LightGBM (Optuna — fewer trials per fold for speed)
        lgbm = LGBMModel(self.symbol, n_trials=self.lgbm_trials, random_state=self.random_state)
        lgbm.fit(X_tr_sm, y_tr_sm, X_val_sc, y_val, feature_names=self.feature_names)
        meta[:, 0:5] = lgbm.predict_proba(X_val_sc)

        # XGBoost
        xgb_m = XGBoostModel(self.symbol, random_state=self.random_state)
        xgb_m.fit(X_tr_sm, y_tr_sm, X_val_sc, y_val, feature_names=self.feature_names)
        meta[:, 5:10] = xgb_m.predict_proba(X_val_sc)

        # PyTorch LSTM
        lstm_m = LSTMModel(self.symbol, epochs=self.lstm_epochs, random_state=self.random_state)
        lstm_m.fit(X_tr_sm, y_tr_sm, X_val_sc, y_val, feature_names=self.feature_names)
        meta[:, 10:15] = lstm_m.predict_proba(X_val_sc)

        # Chronos-2 (zero-shot — no training required; uses raw close prices)
        if close_tr is not None and close_val is not None:
            chronos_m = ChronosModel(self.symbol)
            close_combined = np.concatenate([close_tr, close_val])
            full_proba = chronos_m.predict_proba(close_combined)
            # Align: last len(X_val) rows correspond to val portion
            meta[:, 15:20] = full_proba[-n_val:]
        else:
            meta[:, 15:20] = 0.2  # uniform if no close prices

        return meta

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        close_train: Optional[np.ndarray] = None,
        seq_len: int = 20,
    ) -> "StackingEnsemble":
        """
        Full OOF training procedure.
        X_train / y_train: training split (80% of full data, chronological).
        close_train: raw Close prices for Chronos-2 (aligned with X_train).
        """
        from src.models.lgbm_model import LGBMModel
        from src.models.xgboost_model import XGBoostModel
        from src.models.lstm_model import LSTMModel
        from src.models.chronos_model import ChronosModel

        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        oof_meta  = np.zeros((len(X_train), 20))
        oof_valid = np.zeros(len(X_train), dtype=bool)

        logger.info("%s Ensemble: starting OOF loop (%d folds)", self.symbol, self.cv_folds)

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            if fold_idx == 0:
                # Skip first fold — it's used only as training context (no OOF output)
                continue

            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            close_tr = close_train[tr_idx] if close_train is not None else None
            close_val = close_train[val_idx] if close_train is not None else None

            logger.info("  Fold %d | train=%d val=%d", fold_idx + 1, len(X_tr), len(X_val))
            fold_meta = self._fit_fold_models(X_tr, y_tr, X_val, y_val, close_tr, close_val, seq_len)

            oof_meta[val_idx]  = fold_meta
            oof_valid[val_idx] = True

            # Per-fold OOF F1 for each base learner
            for k, (start, end, name) in enumerate([(0,5,"lgbm"), (5,10,"xgb"), (10,15,"lstm"), (15,20,"chronos")]):
                preds = np.argmax(fold_meta[:, start:end], axis=1)
                f1 = f1_score(y_val, preds, average="macro", zero_division=0)
                self.oof_f1_scores.setdefault(name, []).append(f1)
                logger.info("    Fold %d %s OOF F1=%.4f", fold_idx + 1, name, f1)

        # Train meta-learner on OOF pool (rows predicted independently)
        X_oof = oof_meta[oof_valid]
        y_oof = y_train[oof_valid]
        logger.info("%s Meta-learner: training on %d OOF rows", self.symbol, len(X_oof))

        import lightgbm as lgb
        self.meta_learner = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=5,
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            random_state=self.random_state,
            verbosity=-1,
        )
        self.meta_learner.fit(X_oof, y_oof)

        # Phase C — Retrain all Level-0 on FULL training set
        logger.info("%s Ensemble: retraining Level-0 on full training set", self.symbol)
        scaler_full = StandardScaler()
        X_full_sc = scaler_full.fit_transform(X_train)
        if self.use_smote:
            X_full_sm, y_full_sm = _apply_smote(X_full_sc, y_train, self.random_state)
        else:
            X_full_sm, y_full_sm = X_full_sc, y_train

        self.lgbm = LGBMModel(self.symbol, n_trials=self.lgbm_trials, random_state=self.random_state)
        self.lgbm.fit(X_full_sm, y_full_sm, X_full_sc[-100:], y_train[-100:], feature_names=self.feature_names)

        self.xgb = XGBoostModel(self.symbol, random_state=self.random_state)
        self.xgb.fit(X_full_sm, y_full_sm, X_full_sc[-100:], y_train[-100:], feature_names=self.feature_names)

        self.lstm = LSTMModel(self.symbol, epochs=self.lstm_epochs, random_state=self.random_state)
        self.lstm.fit(X_full_sm, y_full_sm, X_full_sc[-100:], y_train[-100:], feature_names=self.feature_names)

        if close_train is not None:
            self.chronos = ChronosModel(self.symbol)

        self._full_scaler = scaler_full
        logger.info("%s Ensemble: training complete", self.symbol)
        return self

    def _build_meta_input(self, X: np.ndarray, close: Optional[np.ndarray] = None) -> np.ndarray:
        X_sc = self._full_scaler.transform(X)
        meta = np.zeros((len(X), 20))
        meta[:, 0:5]  = self.lgbm.predict_proba(X_sc)
        meta[:, 5:10] = self.xgb.predict_proba(X_sc)
        meta[:, 10:15]= self.lstm.predict_proba(X_sc)
        if close is not None and self.chronos is not None:
            meta[:, 15:20] = self.chronos.predict_proba(close)
        else:
            meta[:, 15:20] = 0.2
        return meta

    def predict_proba(self, X: np.ndarray, close: Optional[np.ndarray] = None) -> np.ndarray:
        meta = self._build_meta_input(X, close)
        return self.meta_learner.predict_proba(meta)

    def predict(self, X: np.ndarray, close: Optional[np.ndarray] = None) -> np.ndarray:
        return np.argmax(self.predict_proba(X, close), axis=1)

    def top_features(self, X: np.ndarray, n: int = 5) -> dict:
        """Top features from the LightGBM Level-0 model via SHAP."""
        X_sc = self._full_scaler.transform(X)
        return self.lgbm.top_features(X_sc, n=n)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("StackingEnsemble saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "StackingEnsemble":
        with open(path, "rb") as f:
            return pickle.load(f)
