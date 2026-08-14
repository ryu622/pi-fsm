"""Phase 4: synthetic recovery test -- measurement error vs model limitation.

Answers the central open question (documents/research_plan.md 1.3, goal 2):
is the delta_t-instability found in real data (phase 1) better explained by
(a) TRACAB measurement noise (~1m) alone, given the model's constant-
    parameter assumption is actually TRUE, or
(b) the constant-parameter assumption being wrong -- alpha, Vmax genuinely
    vary with time, as PINN's F(t), k(t) suggest?

Method: generate synthetic populations of "arrival points" under several
conditions, run the exact Phase-1 Baseline-1 estimator (baseline.py,
unchanged) on each at the same delta_t values used in phase 1, and compare
the resulting Vmax(delta_t) curves against the REAL curve (R):

  constant_no_noise      -- sanity check: pipeline should recover flat Vmax(dt)
  constant_noise, sigma=s -- "measurement-error fingerprint" at noise level s
  timevarying_no_noise (T') -- "pure model-limitation fingerprint"
  timevarying_noise    (T) -- both effects combined (representative noise level)

NOISE MODELING CAVEAT (see documents/phase4_synthetic_recovery_results.md,
2026-08-14 entry): the exact magnitude of TRACAB's per-frame position
noise is not something we can pin down precisely -- "+/-1m" (as cited in
research_plan.md, following the source paper) is most plausibly a bound/
percentile rather than a Gaussian standard deviation, and if real TRACAB
error is temporally correlated (a slowly-drifting bias rather than
independent per-frame jitter, plausible for camera-based tracking), the
noise relevant to a short-delta_t DISPLACEMENT would be considerably
smaller than the raw positional accuracy. Rather than build a more
elaborate correlated-noise model now, this script runs condition C across
a range of plausible independent-noise sigmas (NOISE_SIGMAS) as a
sensitivity analysis, and notes the correlated-noise refinement as a
limitation for future work.

Ground truth for the constant case is J03WPY's own delta_t=1s Baseline-1
fit (documents/phase1_baseline_results.md). Ground truth for the
time-varying case is the PINN's F(t), k(t) trained on real J03WPY data
(documents/phase2_pilot_results.md; consistency confirmed across 3
matches in the 2026-08-14 entry of documents/phase4_synthetic_recovery_results.md),
integrated forward under the ORIGINAL Fujimura-Sugihara assumption of an
arbitrary driving-force direction n per instance (this is what makes
Baseline 1's circular arrival region meaningful -- PINN's own training
fixed n=(1,0) for a different, simpler purpose, but the underlying
F(t),k(t) magnitudes are direction-agnostic physical quantities and can
be reused here).

Usage: uv run python scripts/phase4_synthetic_recovery.py
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scipy.optimize import brentq

from pi_fsm.baseline import A, KineticParams
from pi_fsm.cache import get_sprint_segments
from pi_fsm.pinn.train import TrainConfig, train_pinn

DELTA_TS = [0.2, 0.48, 1.0, 2.0, 3.0]  # matches phase 1's *actual* dt values (frame-snapped)
N_SAMPLES = 200_000
V0_MIN, V0_MAX = 0.05, 8.0
DT_INTEGRATE = 0.002
NOISE_SIGMAS = [0.1, 0.3, 0.5, 1.0]  # meters -- sensitivity range, see module docstring
NOISE_STD_FOR_T = 0.3  # representative "middle" noise level used for the combined T condition
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# J03WPY's own Baseline-1 fit at dt=1.0s (outputs/phase1_baseline/summary.json) -- the
# constant ground truth used for the "constant_*" conditions.
ALPHA_CONST = 0.9472464302470589
VMAX_CONST = 12.340496300571179

# Real Vmax(delta_t) curve to compare against (outputs/phase1_baseline/summary.json).
REAL_VMAX = {0.2: 15.582, 0.48: 13.937, 1.0: 12.340, 2.0: 10.535, 3.0: 10.492}


def simulate(F_fn, k_fn, noise_std: float, seed: int = 0) -> dict[float, pd.DataFrame]:
    """Vectorized Euler integration of m*x''=F(t)n-k(t)x' for N_SAMPLES instances,
    each with its own random v0 (initial speed) and n (driving-force direction,
    arbitrary per instance -- this is what produces Baseline 1's arrival circle).
    Returns {delta_t: DataFrame(v0, x_arr, y_arr)} in the same format
    preprocessing.arrival_points produces, ready for baseline.estimate_kinetic_parameters.
    """
    rng = np.random.default_rng(seed)
    v0_np = rng.uniform(V0_MIN, V0_MAX, size=N_SAMPLES).astype(np.float32)
    angle_np = rng.uniform(0, 2 * np.pi, size=N_SAMPLES).astype(np.float32)

    v0 = torch.tensor(v0_np, device=DEVICE)
    nx = torch.tensor(np.cos(angle_np), device=DEVICE)
    ny = torch.tensor(np.sin(angle_np), device=DEVICE)

    x = torch.zeros(N_SAMPLES, device=DEVICE)
    y = torch.zeros(N_SAMPLES, device=DEVICE)
    vx = v0.clone()
    vy = torch.zeros(N_SAMPLES, device=DEVICE)

    t_max = max(DELTA_TS)
    n_steps = int(round(t_max / DT_INTEGRATE))
    checkpoint_steps = {round(dt / DT_INTEGRATE): dt for dt in DELTA_TS}
    snapshots = {}

    with torch.no_grad():
        for step in range(1, n_steps + 1):
            t = torch.full((1, 1), step * DT_INTEGRATE, device=DEVICE)
            F_t = F_fn(t).reshape(())
            k_t = k_fn(t).reshape(())
            ax = F_t * nx - k_t * vx
            ay = F_t * ny - k_t * vy
            vx = vx + ax * DT_INTEGRATE
            vy = vy + ay * DT_INTEGRATE
            x = x + vx * DT_INTEGRATE
            y = y + vy * DT_INTEGRATE
            if step in checkpoint_steps:
                snapshots[checkpoint_steps[step]] = (x.clone().cpu().numpy(), y.clone().cpu().numpy())

    noise_rng = np.random.default_rng(seed + 1)
    out = {}
    for dt, (xs, ys) in snapshots.items():
        if noise_std > 0:
            xs = xs + noise_rng.normal(0, noise_std, size=xs.shape)
            ys = ys + noise_rng.normal(0, noise_std, size=ys.shape)
        out[dt] = pd.DataFrame({"v0": v0_np, "x_arr": xs, "y_arr": ys})
    return out


def estimate_kinetic_parameters_meanbased(
    arrival_df: pd.DataFrame,
    delta_t: float,
    v0_bin_width: float = 0.3,
    v0_fit_max: float = 6.0,
    v0_hist_max: float = 8.0,
    min_bin_count: int = 20,
) -> KineticParams | None:
    """Same regression framework as baseline.estimate_kinetic_parameters
    (proportional xc-vs-v0 slope -> alpha, mean rc -> Vmax), but using a
    per-v0-bin MEAN (xc, rc) instead of baseline.fit_arrival_circle's
    heatmap + neighbor-connectivity outlier filter.

    Needed because that heatmap approach assumes noise fills the arrival
    circle into a dense blob (true for real TRACAB data and for the
    noisy synthetic conditions here) -- with NO noise, points from a
    uniform-angle-n population land exactly on the circle's rim (a hollow
    ring), which the density-based outlier filter rejects almost entirely.
    A simple mean is robust to both the noiseless (ring) and noisy
    (filled disk) cases, since it doesn't depend on cell occupancy.
    """
    edges = np.arange(0, v0_hist_max + v0_bin_width, v0_bin_width)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = arrival_df[(arrival_df["v0"] >= lo) & (arrival_df["v0"] < hi)]
        if len(sub) < min_bin_count:
            continue
        xc = sub["x_arr"].mean()
        yc = sub["y_arr"].mean()
        rc = np.hypot(sub["x_arr"] - xc, sub["y_arr"] - yc).mean()
        rows.append({"v0": (lo + hi) / 2, "xc": xc, "yc": yc, "rc": rc})

    bins = pd.DataFrame(rows)
    if bins.empty:
        return None
    fit_bins = bins[bins["v0"] <= v0_fit_max]
    if len(fit_bins) < 3:
        return None

    slope = np.sum(fit_bins["v0"] * fit_bins["xc"]) / np.sum(fit_bins["v0"] ** 2)
    rc_mean = fit_bins["rc"].mean()
    if not (0 < slope < delta_t):
        return None

    def objective(alpha: float) -> float:
        return A(alpha, delta_t) - slope

    alpha = brentq(objective, 1e-6, 1e4)
    vmax = rc_mean / (delta_t - A(alpha, delta_t))
    return KineticParams(delta_t=delta_t, alpha=alpha, vmax=vmax, slope=slope, rc_mean=rc_mean, bins=bins)


def run_condition(name: str, F_fn, k_fn, noise_std: float) -> pd.DataFrame:
    print(f"--- {name} ---")
    t0 = time.time()
    sims = simulate(F_fn, k_fn, noise_std)
    rows = []
    for dt in DELTA_TS:
        res = estimate_kinetic_parameters_meanbased(sims[dt], delta_t=dt)
        alpha = res.alpha if res else None
        vmax = res.vmax if res else None
        rows.append({"condition": name, "delta_t": dt, "alpha": alpha, "vmax": vmax})
        print(f"  dt={dt}: alpha={alpha}, vmax={vmax}")
    print(f"  ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def main() -> None:
    print("training PINN on real J03WPY data (t<=2.5s) for the time-varying ground truth...")
    segs_full = get_sprint_segments("J03WPY")
    segs_train = segs_full[segs_full["t"] <= 2.5].copy()
    v0_max_train = float(segs_train["v0"].max())
    config = TrainConfig(
        epochs=10000, lr=3e-3, lr_min_factor=0.05, gamma_final=5.0, gamma_warmup_epochs=2000,
        beta=0.01, n_colloc=1500, t_max=2.5, v0_min=0.0, v0_max=v0_max_train,
        traj_hidden=128, traj_n_hidden=5, seed=0, device=DEVICE,
    )
    t0 = time.time()
    pinn_res = train_pinn(segs_train, config)
    print(f"trained in {time.time()-t0:.0f}s, L_data={pinn_res.history['data'].iloc[-1]:.3f}")
    F_net, k_net = pinn_res.force_net, pinn_res.resistance_net

    def F_const(t):
        return torch.full_like(t, ALPHA_CONST * VMAX_CONST)

    def k_const(t):
        return torch.full_like(t, ALPHA_CONST)

    results = [run_condition("constant_no_noise", F_const, k_const, noise_std=0.0)]
    for sigma in NOISE_SIGMAS:
        results.append(run_condition(f"constant_noise_sigma{sigma}", F_const, k_const, noise_std=sigma))
    results.append(run_condition("timevarying_no_noise", F_net, k_net, noise_std=0.0))
    results.append(run_condition("timevarying_noise", F_net, k_net, noise_std=NOISE_STD_FOR_T))

    combined = pd.concat(results, ignore_index=True)
    combined.to_csv("outputs/phase4/synthetic_recovery.csv", index=False)
    print("\n", combined.to_string())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    noise_cmap = plt.get_cmap("Oranges")
    for i, sigma in enumerate(NOISE_SIGMAS):
        sub = combined[combined["condition"] == f"constant_noise_sigma{sigma}"].sort_values("delta_t")
        color = noise_cmap(0.35 + 0.55 * i / max(1, len(NOISE_SIGMAS) - 1))
        ax.plot(sub["delta_t"], sub["vmax"], "o-", color=color, lw=1.3, ms=4,
                label=f"C: 定数+ノイズ (σ={sigma}m)")
    sub = combined[combined["condition"] == "constant_no_noise"].sort_values("delta_t")
    ax.plot(sub["delta_t"], sub["vmax"], "o--", color="#898781", label="定数・ノイズなし(確認用)")
    sub = combined[combined["condition"] == "timevarying_no_noise"].sort_values("delta_t")
    ax.plot(sub["delta_t"], sub["vmax"], "o-", color="#1baf7a", lw=2, label="T': 時間変化のみ・ノイズなし")
    sub = combined[combined["condition"] == "timevarying_noise"].sort_values("delta_t")
    ax.plot(sub["delta_t"], sub["vmax"], "o-", color="#2a78d6", lw=2,
            label=f"T: 時間変化+ノイズ(σ={NOISE_STD_FOR_T}m)")
    real_dt = sorted(REAL_VMAX.keys())
    ax.plot(real_dt, [REAL_VMAX[d] for d in real_dt], "ks--", lw=2.5, ms=9, label="R: 実データ(J03WPY, フェーズ1)")
    ax.set_xlabel("Δt [s]")
    ax.set_ylabel("$V_{max}$")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("フェーズ4: 実データの不安定性はどちらの指紋に近いか(ノイズ感度分析込み)")
    fig.tight_layout()
    fig.savefig("outputs/phase4/synthetic_recovery.png", dpi=150, facecolor="white", bbox_inches="tight")
    print("saved outputs/phase4/synthetic_recovery.png")


if __name__ == "__main__":
    main()
