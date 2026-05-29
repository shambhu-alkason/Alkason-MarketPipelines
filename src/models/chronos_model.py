"""
Amazon Chronos-2 zero-shot time series forecaster.
Model: amazon/chronos-bolt-small (October 2025, Chronos-2 family).

Usage: feed raw Close price history → get next-day probabilistic forecast
       → convert to 5-class signal based on predicted price change.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

CHRONOS_MODEL_ID = "amazon/chronos-bolt-small"


class ChronosModel:
    """
    Wraps Amazon Chronos-2 for zero-shot stock price forecasting.
    No training required — pretrained on 700B+ data points.
    """

    def __init__(self, symbol: str, device: str = "cpu"):
        self.symbol = symbol
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from chronos import BaseChronosPipeline
            logger.info("Loading %s ...", CHRONOS_MODEL_ID)
            self._pipeline = BaseChronosPipeline.from_pretrained(
                CHRONOS_MODEL_ID,
                device_map=self.device,
                torch_dtype="bfloat16",
            )
            logger.info("Chronos-2 loaded on %s", self.device)
        except Exception as e:
            logger.warning("Chronos-2 load failed: %s — using fallback uniform probs", e)
            self._pipeline = None
        return self._pipeline

    def _return_to_signal(
        self,
        predicted_return: float,
        strong_buy: float = 0.02,
        buy: float = 0.005,
        sell: float = -0.005,
        strong_sell: float = -0.02,
    ) -> int:
        if predicted_return > strong_buy:   return 4
        if predicted_return > buy:          return 3
        if predicted_return > sell:         return 2
        if predicted_return > strong_sell:  return 1
        return 0

    def forecast_next(
        self,
        close_series: np.ndarray,
        context_length: int = 64,
        n_samples: int = 20,
    ) -> np.ndarray:
        """
        Forecast next-day price using Chronos-2.

        Returns probability array of shape (5,) — one prob per signal class.
        """
        pipeline = self._load_pipeline()
        if pipeline is None:
            return np.full(5, 0.2)

        try:
            import torch
            context = torch.tensor(close_series[-context_length:], dtype=torch.float32).unsqueeze(0)
            forecast = pipeline.predict(
                context=context,
                prediction_length=1,
                num_samples=n_samples,
            )
            # forecast shape: (1, n_samples, prediction_length)
            samples = forecast[0, :, 0].float().cpu().numpy()  # shape: (n_samples,)
            current_price = float(close_series[-1])
            returns = (samples - current_price) / current_price

            # Convert each sample return to a signal class
            classes = np.array([self._return_to_signal(r) for r in returns])
            probs = np.bincount(classes, minlength=5).astype(float)
            probs /= probs.sum()
            return probs
        except Exception as e:
            logger.warning("%s Chronos-2 forecast failed: %s", self.symbol, e)
            return np.full(5, 0.2)

    def predict_proba_matrix(
        self,
        close_values: np.ndarray,
        context_length: int = 64,
    ) -> np.ndarray:
        """
        Generate probability matrix for all rows in close_values.
        For each row i, uses close_values[:i+1] as context.
        Rows with insufficient context get uniform probs.

        Returns: np.ndarray of shape (len(close_values), 5)
        """
        n = len(close_values)
        result = np.full((n, 5), 0.2)

        for i in range(context_length, n):
            context = close_values[:i + 1]
            result[i] = self.forecast_next(context, context_length=context_length)

        return result

    def predict(self, close_values: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba_matrix(close_values), axis=1)

    def predict_proba(self, close_values: np.ndarray) -> np.ndarray:
        return self.predict_proba_matrix(close_values)
