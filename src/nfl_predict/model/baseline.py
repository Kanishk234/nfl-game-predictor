"""The Vegas baseline, held to the same no-peeking rule as the model.

The closing spread is already a margin prediction, so spread metrics need no conversion. For win
probability it has to be mapped from points to a probability, and that mapping is *fitted*: a
logistic regression of home-win on spread. Fitting it on the full dataset and then scoring
held-out seasons would hand the baseline a small look at the future, so it is fitted per fold on
training rows only, exactly as the model is.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class SpreadToWinProb:
    """P(home win) as a logistic function of the closing spread."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e6)  # effectively unregularised, one feature

    def fit(self, spread: np.ndarray, home_win: np.ndarray) -> SpreadToWinProb:
        self._lr.fit(np.asarray(spread, float).reshape(-1, 1), np.asarray(home_win, int))
        return self

    def predict(self, spread: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(np.asarray(spread, float).reshape(-1, 1))[:, 1]

    @property
    def points_per_logit(self) -> float:
        """Points of spread per unit of log-odds: a sanity check, ~7 in the literature."""
        return float(1.0 / self._lr.coef_[0, 0])
