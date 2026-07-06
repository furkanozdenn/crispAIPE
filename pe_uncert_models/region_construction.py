"""Confidence-region constructions on the Dirichlet 2-simplex.

Three constructions, sharing a common interface:

- ``PCABaseline``  — the legacy PCA ellipse on samples projected to (edited, unedited).
  Kept here only as the ablation baseline. Coverage is correct only under the
  unrealistic assumption that posterior samples are approximately Gaussian in the
  2D projection.

- ``HyndmanHDR`` — Bayesian highest-density region of Dir(alpha_hat) via Hyndman's
  density-quantile algorithm (Hyndman 1996, *The American Statistician*).
  For each test input, draw S samples from the Dirichlet, evaluate log density at
  each, and take the (1 - gamma) sample quantile as the threshold tau_hat.
  Region = {y in Delta^2 : log Dir(y; alpha_hat) >= tau_hat}. Stays on the simplex
  by construction. Coverage correct only if Dir(alpha_hat) is well-calibrated.

- ``SplitConformalHDR`` — same level-set geometry as HyndmanHDR, but with the
  threshold computed once over a held-out calibration fold using
  s_i = -log Dir(y_i_true; alpha_hat(x_i)) (Amaral et al. 2026; Vovk-Gammerman-
  Shafer 2005; Lei et al. 2018; Angelopoulos & Bates 2023). Gives finite-sample
  marginal coverage under exchangeability regardless of model calibration.

All three expose:
- ``threshold(alpha_hat)`` returning the per-input log-density threshold tau (i.e.
  the level value of log Dir defining the region). For conformal, this is
  -q_hat (constant across inputs).
- ``contains(y, alpha_hat)`` membership test, vectorized.
- ``area_simplex(alpha_hat)`` Monte Carlo area on the (edited, unedited) projection
  of Delta^2. The full triangle has area 0.5 in this projection, so the maximum
  value of ``area_simplex`` is 0.5.

Geometry note: ``Delta^2`` is the 2-simplex {y in R^3 : y >= 0, sum(y) = 1}, with
the (edited, unedited) projection mapping y -> (y_0, y_1) onto the right triangle
T = {(e, u) : e >= 0, u >= 0, e + u <= 1} of area 0.5. Area numbers reported in
the manuscript are in this projection so that the PCA baseline (originally defined
in this projection) and the new constructions are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2


# ---------- low-level Dirichlet log-density on the simplex ----------


def log_dirichlet_pdf(y: np.ndarray, alpha: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Log Dir(y; alpha), broadcasting over leading dims.

    y : (..., 3) points on the simplex (sum to 1, non-negative).
    alpha : (..., 3) concentration parameters (positive).

    Returns log density of shape broadcast(y.shape[:-1], alpha.shape[:-1]).
    The (eps) floor on y avoids log(0) at the simplex boundary.
    """
    y = np.clip(y, eps, 1.0)
    log_y = np.log(y)
    log_B = np.sum(gammaln(alpha), axis=-1) - gammaln(np.sum(alpha, axis=-1))
    return np.sum((alpha - 1.0) * log_y, axis=-1) - log_B


# ---------- shared uniform-on-simplex sampler for area estimation ----------


def uniform_simplex_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n iid points uniformly on the 3-component simplex.

    Returns array of shape (n, 3). Distribution is Dirichlet(1, 1, 1).
    """
    return rng.dirichlet(alpha=np.ones(3), size=n)


# ---------- base construction class ----------


@dataclass
class RegionConstruction:
    """Shared interface and helpers."""

    confidence_level: float = 0.95
    n_posterior_samples: int = 1000
    n_uniform_samples: int = 8000
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        # Pre-draw uniform samples shared across all area-on-simplex evaluations.
        self._uniform = uniform_simplex_samples(self.n_uniform_samples, self._rng)
        # Precompute log y for the uniform samples (used in every area call).
        self._uniform_log = np.log(np.clip(self._uniform, 1e-12, 1.0))

    # threshold semantics:
    #   region = {y in Delta^2 : log Dir(y; alpha) >= tau(alpha)}
    def threshold(self, alpha: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def contains(self, y: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Vectorized membership test. y, alpha both shape (N, 3)."""
        log_pdf = log_dirichlet_pdf(y, alpha)
        tau = self.threshold(alpha)
        return log_pdf >= tau

    def area_simplex(self, alpha: np.ndarray) -> np.ndarray:
        """Monte Carlo area on the (e, u) projection of Delta^2 (max value 0.5).

        For each row of alpha, returns the area of {y in Delta^2 : log Dir(y; alpha) >= tau(alpha)}.
        """
        alpha = np.atleast_2d(alpha)
        tau = self.threshold(alpha)                           # (N,)
        # Compute log Dir(y_uniform; alpha) for every (uniform, alpha) pair.
        # log_pdf shape (N, M):
        #   sum over k of (alpha_k - 1) * log y_uniform_k  - log B(alpha)
        # alpha has shape (N, 3); _uniform_log has shape (M, 3).
        contribs = (alpha[:, None, :] - 1.0) * self._uniform_log[None, :, :]   # (N, M, 3)
        log_pdf = contribs.sum(axis=-1)                                        # (N, M)
        log_B = np.sum(gammaln(alpha), axis=-1) - gammaln(np.sum(alpha, axis=-1))  # (N,)
        log_pdf = log_pdf - log_B[:, None]
        inside = log_pdf >= tau[:, None]
        # Triangle area in the (edited, unedited) projection is 0.5.
        return 0.5 * inside.mean(axis=-1)


# ---------- PCA baseline (legacy) ----------


class PCABaseline(RegionConstruction):
    """Legacy PCA ellipse on 2D-projected posterior samples.

    Kept for ablation. The membership test uses the closed-form Mahalanobis
    distance under the closed-form 2D Dirichlet marginal covariance (the
    analytic formula already used by the manuscript's projected_area95 function).
    """

    def threshold(self, alpha: np.ndarray) -> np.ndarray:
        # PCA membership does not map cleanly onto a log-density level set
        # (the region is a Gaussian Mahalanobis ball, not a Dirichlet level set).
        # The method below overrides contains() and area_simplex() directly, so
        # threshold() is unused. Return a placeholder.
        return np.full(alpha.shape[0] if alpha.ndim == 2 else 1, np.nan)

    @staticmethod
    def _cov_2d(alpha: np.ndarray) -> np.ndarray:
        """Closed-form 2D Dirichlet marginal covariance on (edited, unedited)."""
        a0 = alpha.sum(axis=-1, keepdims=True)
        a01 = alpha[..., :2]
        c00 = a01[..., 0] * (a0[..., 0] - a01[..., 0]) / (a0[..., 0] ** 2 * (a0[..., 0] + 1))
        c11 = a01[..., 1] * (a0[..., 0] - a01[..., 1]) / (a0[..., 0] ** 2 * (a0[..., 0] + 1))
        c01 = -(a01[..., 0] * a01[..., 1]) / (a0[..., 0] ** 2 * (a0[..., 0] + 1))
        cov = np.stack([
            np.stack([c00, c01], axis=-1),
            np.stack([c01, c11], axis=-1),
        ], axis=-2)
        return cov

    @staticmethod
    def _mean_2d(alpha: np.ndarray) -> np.ndarray:
        a0 = alpha.sum(axis=-1, keepdims=True)
        return alpha[..., :2] / a0

    def contains(self, y: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        mu = self._mean_2d(alpha)
        cov = self._cov_2d(alpha)
        # Mahalanobis distance squared in 2D.
        delta = y[..., :2] - mu
        # solve cov @ x = delta
        inv = np.linalg.inv(cov)
        m2 = np.einsum("...i,...ij,...j->...", delta, inv, delta)
        thresh = chi2.ppf(self.confidence_level, df=2)
        return m2 <= thresh

    def area_simplex(self, alpha: np.ndarray) -> np.ndarray:
        # Analytic 2D ellipse area. Equivalent to projected_area95 in the audit code.
        alpha = np.atleast_2d(alpha)
        cov = self._cov_2d(alpha)
        det = np.maximum(cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] ** 2, 0)
        return np.pi * chi2.ppf(self.confidence_level, df=2) * np.sqrt(det)


# ---------- Option 1: Bayesian HDR via Hyndman density-quantile ----------


class HyndmanHDR(RegionConstruction):
    """Hyndman 1996 density-quantile HDR on Dir(alpha_hat).

    For each test input, draw S samples from Dir(alpha_hat), evaluate log density
    at each, and take the (1 - gamma) sample quantile as the threshold tau_hat.
    """

    def threshold(self, alpha: np.ndarray) -> np.ndarray:
        alpha = np.atleast_2d(alpha)
        tau_quantile = 1.0 - self.confidence_level
        n = alpha.shape[0]
        out = np.empty(n)
        # Per-row Dirichlet sampling. np.random.Generator.dirichlet does not
        # broadcast across alpha, so loop. This is the bottleneck.
        S = self.n_posterior_samples
        for i in range(n):
            samples = self._rng.dirichlet(alpha[i], size=S)
            log_pdf = log_dirichlet_pdf(samples, alpha[i])
            out[i] = np.quantile(log_pdf, tau_quantile)
        return out


# ---------- Option 3: Split conformal HDR ----------


class SplitConformalHDR(RegionConstruction):
    """Split conformal prediction on the Dirichlet level set.

    Calibration: for each (x_i, y_i^true) in the calibration fold, compute
        s_i = -log Dir(y_i^true; alpha_hat(x_i))
    and take q_hat = the ceil((n + 1) * (1 - gamma)) / n empirical quantile of {s_i}.
    Region = {y : -log Dir(y; alpha_hat) <= q_hat}  =  {y : log Dir(y; alpha_hat) >= -q_hat}.
    Threshold is a single scalar shared across all test inputs.

    Use ``calibrate(alpha_cal, y_cal)`` once to fit q_hat before calling
    ``contains`` or ``area_simplex``.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._q_hat: float | None = None
        self._n_cal: int | None = None
        self._raw_scores: np.ndarray | None = None

    def calibrate(self, alpha_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalHDR":
        scores = -log_dirichlet_pdf(y_cal, alpha_cal)
        n = scores.shape[0]
        level = (np.ceil((n + 1) * self.confidence_level)) / n
        level = float(np.minimum(level, 1.0))
        self._q_hat = float(np.quantile(scores, level))
        self._n_cal = n
        self._raw_scores = scores
        return self

    @property
    def q_hat(self) -> float:
        if self._q_hat is None:
            raise RuntimeError("SplitConformalHDR is not calibrated. Call calibrate() first.")
        return self._q_hat

    def threshold(self, alpha: np.ndarray) -> np.ndarray:
        n = alpha.shape[0] if alpha.ndim == 2 else 1
        return np.full(n, -self.q_hat)


__all__ = [
    "RegionConstruction",
    "PCABaseline",
    "HyndmanHDR",
    "SplitConformalHDR",
    "log_dirichlet_pdf",
    "uniform_simplex_samples",
]
