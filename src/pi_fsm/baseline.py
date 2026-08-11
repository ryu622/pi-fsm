"""Fujimura-Sugihara model: analytic solution and Narizuka et al. (2023)'s
kinetic-parameter estimation method from tracking-data heat maps.

Reference: Narizuka, Takizawa & Yamazaki (2023), Scientific Reports 13:865.
Eq. numbers in comments refer to that paper.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

# --- Analytic solution (Eq. 4-6) -------------------------------------------


def A(alpha: float, t: float) -> float:
    """Eq. (5): first coefficient of the arrival-circle solution."""
    if alpha <= 0:
        return t
    return (1 - np.exp(-alpha * t)) / alpha


def B(alpha: float, vmax: float, t: float) -> float:
    """Eq. (6): second coefficient (= arrival-circle radius)."""
    return vmax * (t - A(alpha, t))


def arrival_circle(alpha: float, vmax: float, v0: float, t: float) -> tuple[float, float]:
    """Center-x (center-y is 0 in the v0-aligned frame) and radius of the
    arrival circle (Eq. 4), for a player with initial speed v0."""
    center_x = A(alpha, t) * v0
    radius = B(alpha, vmax, t)
    return center_x, radius


# --- Heat map construction & circle fit (Investigation method, Eq. 7-8) ----


def outlier_threshold(delta_t: float) -> int:
    """c: min count of nonblank Moore-neighborhood cells (of 8) to keep a
    heat map cell. Paper states c=8 for dt<1, 6 for 1<=dt<=3, 4 for dt>3,
    but does not specify the exact neighborhood definition; we use the
    literal 8-cell Moore neighborhood and treat c as a ">=" threshold."""
    if delta_t < 1:
        return 8
    if delta_t <= 3:
        return 6
    return 4


def _neighbor_counts(nonblank: np.ndarray) -> np.ndarray:
    """Count of nonblank cells in the 8-neighborhood of each cell."""
    padded = np.pad(nonblank.astype(int), 1)
    total = np.zeros_like(nonblank, dtype=int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            total += padded[1 + dy : 1 + dy + nonblank.shape[0], 1 + dx : 1 + dx + nonblank.shape[1]]
    return total


def fit_arrival_circle(
    x_arr: np.ndarray, y_arr: np.ndarray, delta_t: float
) -> tuple[float, float, float] | None:
    """Eq. (7)-(8): estimate (xc, yc, rc) of a v0-bin's arrival heat map.

    Returns None if too few points remain after outlier filtering.
    """
    if len(x_arr) < 10:
        return None

    cell = 0.2 * delta_t
    x_edges = np.arange(x_arr.min() - cell, x_arr.max() + 2 * cell, cell)
    y_edges = np.arange(y_arr.min() - cell, y_arr.max() + 2 * cell, cell)
    counts, x_edges, y_edges = np.histogram2d(x_arr, y_arr, bins=[x_edges, y_edges])

    nonblank = counts > 0
    c = outlier_threshold(delta_t)
    keep = nonblank & (_neighbor_counts(nonblank) >= c)
    if not keep.any():
        return None

    ix, iy = np.nonzero(keep)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xs, ys = x_centers[ix], y_centers[iy]

    x_max, x_min, y_max, y_min = xs.max(), xs.min(), ys.max(), ys.min()
    xc = (x_max + x_min) / 2  # Eq. (7)
    yc = (y_max + y_min) / 2
    rc = (x_max - x_min + y_max - y_min) / 4  # Eq. (8)
    return xc, yc, rc


# --- Kinetic parameter estimation ------------------------------------------


@dataclass(frozen=True)
class KineticParams:
    delta_t: float
    alpha: float
    vmax: float
    slope: float  # regression slope of xc vs v0 (estimate of A(alpha, t))
    rc_mean: float  # mean radius (estimate of B(alpha, vmax, t))
    bins: pd.DataFrame  # per-v0-bin (v0, xc, yc, rc) used in the fit


def estimate_kinetic_parameters(
    arrival_df: pd.DataFrame,
    delta_t: float,
    v0_bin_width: float = 0.3,
    v0_fit_max: float = 6.0,
    v0_hist_max: float = 8.0,
) -> KineticParams | None:
    """Estimate alpha, V_max for one delta_t from a set of arrival points.

    `arrival_df` must have columns v0, x_arr, y_arr (see
    preprocessing.arrival_points), already restricted to a single delta_t.
    """
    edges = np.arange(0, v0_hist_max + v0_bin_width, v0_bin_width)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = arrival_df[(arrival_df["v0"] >= lo) & (arrival_df["v0"] < hi)]
        fit = fit_arrival_circle(sub["x_arr"].to_numpy(), sub["y_arr"].to_numpy(), delta_t)
        if fit is None:
            continue
        xc, yc, rc = fit
        rows.append({"v0": (lo + hi) / 2, "xc": xc, "yc": yc, "rc": rc})

    bins = pd.DataFrame(rows)
    if bins.empty:
        return None

    fit_bins = bins[bins["v0"] <= v0_fit_max]
    if len(fit_bins) < 3:
        return None

    # xc = A(alpha,t) * v0, proportional (no intercept), per Eq. (4)-(5).
    slope = np.sum(fit_bins["v0"] * fit_bins["xc"]) / np.sum(fit_bins["v0"] ** 2)
    rc_mean = fit_bins["rc"].mean()

    if not (0 < slope < delta_t):
        return None  # slope must lie in A(alpha,t)'s range (0, t)

    def objective(alpha: float) -> float:
        return A(alpha, delta_t) - slope

    # A(alpha,t) is monotonically decreasing in alpha, from t (alpha->0) to 0 (alpha->inf).
    alpha = brentq(objective, 1e-6, 1e4)
    vmax = rc_mean / (delta_t - A(alpha, delta_t))

    return KineticParams(
        delta_t=delta_t, alpha=alpha, vmax=vmax, slope=slope, rc_mean=rc_mean, bins=bins
    )
