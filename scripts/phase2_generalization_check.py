"""Phase 2 (reframed): does a SINGLE PINN fit (time-dependent F(t),k(t))
predict sprint-onset trajectories more accurately ACROSS MULTIPLE tau than
a SINGLE Baseline-1 constant-parameter fit does?

This is the methodologically correct comparison (research_plan.md 5-2):
compare model *predictions at matched time*, not raw parameter values at
mismatched axes (Baseline 1's delta_t is a prediction horizon averaged over
the whole match; PINN's t is elapsed time since one sprint's onset -- they
are not the same axis, so comparing their fitted numbers directly, as
earlier experiments today did, was a category error).

Ground truth and PINN predictions both use segments.py's own rotation
convention (net heading over the first second) throughout, avoiding a
second, different rotation convention (preprocessing.arrival_points' more
instantaneous-velocity-based one) that would confound the comparison.

Usage: uv run python scripts/phase2_generalization_check.py
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pi_fsm.baseline import A, B
from pi_fsm.cache import get_sprint_segments
from pi_fsm.pinn.train import TrainConfig, train_pinn

MATCH_ID = "J03WPY"
T_TRAIN_CUTOFF = 2.5  # PINN is trained only on segment data up to this tau
EVAL_TAUS = [0.5, 1.0, 1.5, 2.0, 2.5]
V0_BIN_WIDTH = 0.3
MIN_BIN_COUNT = 20

# Baseline 1's phase-1 fit at delta_t=1s (documents/phase1_baseline_results.md), used AS-IS
# (not refit) at every tau below -- this is the "single constant model" being tested.
ALPHA_BASELINE1 = 0.9472464302470589
VMAX_BASELINE1 = 12.340496300571179


def bin_ground_truth(segs_full: pd.DataFrame, tau: float, v0_max: float) -> pd.DataFrame:
    """Mean actual x(tau) per v0 bin, from segments whose own duration covers tau."""
    window = 1.0 / 25.0  # frame spacing; tolerate one frame of snapping
    sub = segs_full[(segs_full["t"] >= tau - window) & (segs_full["t"] <= tau + window)]
    sub = sub.copy()
    sub["t_diff"] = (sub["t"] - tau).abs()
    sub = sub.sort_values("t_diff").drop_duplicates("segment_id")  # closest sample per segment

    bins = np.arange(0, v0_max + V0_BIN_WIDTH, V0_BIN_WIDTH)
    sub["v0_bin"] = pd.cut(sub["v0"], bins)
    g = sub.groupby("v0_bin", observed=True).agg(v0=("v0", "mean"), x=("x", "mean"), n=("x", "size"))
    return g[g["n"] >= MIN_BIN_COUNT].reset_index(drop=True)


def main() -> None:
    segs_full = get_sprint_segments(MATCH_ID)  # full natural duration, default v_low=2.0
    segs_train = segs_full[segs_full["t"] <= T_TRAIN_CUTOFF].copy()
    v0_max = float(segs_train["v0"].max())
    print(f"{MATCH_ID}: {segs_full['segment_id'].nunique()} segments total, "
          f"{segs_train['segment_id'].nunique()} used for PINN training (t<={T_TRAIN_CUTOFF}s)")

    config = TrainConfig(
        epochs=10000, lr=3e-3, lr_min_factor=0.05, gamma_final=5.0, gamma_warmup_epochs=2000,
        beta=0.01, n_colloc=1500, t_max=T_TRAIN_CUTOFF, v0_min=0.0, v0_max=v0_max,
        traj_hidden=128, traj_n_hidden=5, seed=0,
    )
    t0 = time.time()
    res = train_pinn(segs_train, config)
    print(f"trained in {time.time()-t0:.0f}s, L_data={res.history['data'].iloc[-1]:.3f}")

    device = config.device
    rows = []
    for tau in EVAL_TAUS:
        gt = bin_ground_truth(segs_full, tau, v0_max)
        if len(gt) < 3:
            print(f"tau={tau}: too few bins with data ({len(gt)}), skipping")
            continue

        v0_t = torch.tensor(gt["v0"].to_numpy(), dtype=torch.float32, device=device).unsqueeze(-1)
        t_t = torch.full_like(v0_t, tau)
        with torch.no_grad():
            pinn_pred = res.traj_net(t_t, v0_t).cpu().numpy()[:, 0]

        baseline_pred = A(ALPHA_BASELINE1, tau) * gt["v0"].to_numpy() + B(ALPHA_BASELINE1, VMAX_BASELINE1, tau)

        err_pinn = float(np.sqrt(np.mean((pinn_pred - gt["x"].to_numpy()) ** 2)))
        err_baseline = float(np.sqrt(np.mean((baseline_pred - gt["x"].to_numpy()) ** 2)))
        rows.append({"tau": tau, "n_bins": len(gt), "rmse_pinn": err_pinn, "rmse_baseline1": err_baseline})
        print(f"tau={tau}: n_bins={len(gt)}, RMSE PINN={err_pinn:.3f}m, RMSE Baseline1={err_baseline:.3f}m")

    result = pd.DataFrame(rows)
    print()
    print(result)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result["tau"], result["rmse_pinn"], "o-", color="#2a78d6", label="PINN (single time-dependent fit)")
    ax.plot(result["tau"], result["rmse_baseline1"], "o-", color="#eb6834", label="Baseline1 (single constant fit, dt=1s)")
    ax.axvline(1.0, color="gray", ls=":", label="Baseline1 was fit here")
    ax.set_xlabel("tau [s]")
    ax.set_ylabel("RMSE of predicted center x(tau) [m]")
    ax.set_title(f"{MATCH_ID}: does a single time-dependent fit generalize across tau better?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("outputs/phase2_pinn/generalization_check.png", dpi=150, facecolor="white", bbox_inches="tight")
    print("saved outputs/phase2_pinn/generalization_check.png")


if __name__ == "__main__":
    main()
