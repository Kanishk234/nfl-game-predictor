"""Scoring. Decided up front (per PLAN.md Phase 5) so the backtest, the grader and the site all
report the same things the same way.

Win-probability metrics take probabilities of a *home* win. Spread metrics take predicted home
margin, on the same scale and sign as `spread_line` (positive = home favoured).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

#: Breakeven hit rate against the spread at standard -110 juice. Anything below this loses money
#: even when it is above 50%; worth keeping in view so ATS numbers are read honestly.
ATS_BREAKEVEN = 0.5238


@dataclass(frozen=True)
class WinProbMetrics:
    n: int
    accuracy: float
    brier: float
    log_loss: float
    auc: float
    ece: float  # expected calibration error, 10 equal-width bins

    def row(self) -> dict:
        return {
            "n": self.n, "accuracy": self.accuracy, "brier": self.brier,
            "log_loss": self.log_loss, "auc": self.auc, "ece": self.ece,
        }


@dataclass(frozen=True)
class SpreadMetrics:
    n: int
    mae: float
    rmse: float
    ats_record: tuple[int, int, int]  # wins, losses, pushes against the closing line
    ats_pct: float  # of non-push games

    def row(self) -> dict:
        w, l, p = self.ats_record
        return {
            "n": self.n, "mae": self.mae, "rmse": self.rmse,
            "ats_w": w, "ats_l": l, "ats_push": p, "ats_pct": self.ats_pct,
        }


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1 - 1e-6)


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(prob[mask].mean() - y[mask].mean())
    return float(ece)


def calibration_table(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Reliability bins for the site's calibration chart."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        rows.append({
            "bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
            "n": int(mask.sum()),
            "mean_predicted": float(prob[mask].mean()) if mask.any() else None,
            "observed": float(y[mask].mean()) if mask.any() else None,
        })
    return rows


def win_prob_metrics(prob: np.ndarray, y: np.ndarray) -> WinProbMetrics:
    prob, y = _clip(np.asarray(prob, float)), np.asarray(y, float)
    # Degenerate folds (all one class) have no AUC; report NaN rather than crash.
    auc = float(roc_auc_score(y, prob)) if 0 < y.mean() < 1 else float("nan")
    return WinProbMetrics(
        n=len(y),
        accuracy=float(((prob > 0.5) == (y == 1)).mean()),
        brier=float(np.mean((prob - y) ** 2)),
        log_loss=float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))),
        auc=auc,
        ece=expected_calibration_error(prob, y),
    )


def spread_metrics(pred_margin: np.ndarray, margin: np.ndarray, line: np.ndarray) -> SpreadMetrics:
    """Spread accuracy plus the against-the-spread record versus the closing line.

    ATS: we take the home side when our margin beats the line, the away side otherwise. The bet
    wins if the actual result lands on the same side of the line, pushes if it lands exactly on
    it. Vegas cannot have an ATS record against itself, so this is ours alone.
    """
    pred, margin, line = (np.asarray(a, float) for a in (pred_margin, margin, line))
    err = pred - margin
    take_home = pred > line
    covered_home = margin > line
    push = margin == line
    win = ~push & (take_home == covered_home)
    loss = ~push & ~win
    w, l, p = int(win.sum()), int(loss.sum()), int(push.sum())
    return SpreadMetrics(
        n=len(margin),
        mae=float(np.abs(err).mean()),
        rmse=float(np.sqrt(np.mean(err**2))),
        ats_record=(w, l, p),
        ats_pct=float(w / (w + l)) if (w + l) else float("nan"),
    )
