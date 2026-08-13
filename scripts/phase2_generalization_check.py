"""Phase 2 (reframed): does a SINGLE PINN fit (time-dependent F(t),k(t))
predict sprint-onset trajectories more accurately ACROSS MULTIPLE tau than
a SINGLE Baseline-1 constant-parameter fit does?

This is the methodologically correct comparison (research_plan.md 5-2):
compare model *predictions at matched time*, not raw parameter values at
mismatched axes (Baseline 1's delta_t is a prediction horizon averaged over
the whole match; PINN's t is elapsed time since one sprint's onset -- they
are not the same axis, so comparing their fitted numbers directly, as
earlier experiments did, was a category error).

Ground truth and PINN predictions both use segments.py's own rotation
convention (net heading over the first second) throughout, avoiding a
second, different rotation convention (preprocessing.arrival_points' more
instantaneous-velocity-based one) that would confound the comparison.

Originally run on J03WPY only (documents/phase2_pilot_results.md); this
version loops over multiple matches to check whether that result
replicates -- see notebooks/colab_generalization_check.ipynb for the
Colab entry point.

Usage: uv run python scripts/phase2_generalization_check.py [--match-ids J03WPY,J03WMX,...] [--out-dir DIR]
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pi_fsm.baseline import A, B, estimate_kinetic_parameters
from pi_fsm.cache import get_players_with_velocity, get_sprint_segments
from pi_fsm.data import MATCH_IDS
from pi_fsm.pinn.train import TrainConfig, train_pinn
from pi_fsm.preprocessing import arrival_points, exclude_goalkeepers

T_TRAIN_CUTOFF = 2.5  # PINN is trained only on segment data up to this tau
EVAL_TAUS = [0.5, 1.0, 1.5, 2.0, 2.5]
V0_BIN_WIDTH = 0.3
MIN_BIN_COUNT = 20
BASELINE1_FIT_DT = 1.0  # the single delta_t Baseline1 is calibrated on, per match


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


def baseline1_fit(match_id: str) -> tuple[float, float]:
    """This match's own Baseline1 (alpha, Vmax) at delta_t=1s (full population, not sprint-restricted)."""
    df, frame_rate = get_players_with_velocity(match_id)
    df = exclude_goalkeepers(df)
    ap = arrival_points(df, frame_rate, delta_t=BASELINE1_FIT_DT)
    res = estimate_kinetic_parameters(ap, delta_t=BASELINE1_FIT_DT)
    if res is None:
        raise RuntimeError(f"Baseline1 fit failed for {match_id}")
    return res.alpha, res.vmax


def run_one_match(match_id: str, out_dir: Path) -> pd.DataFrame:
    segs_full = get_sprint_segments(match_id)  # full natural duration, default v_low=2.0
    segs_train = segs_full[segs_full["t"] <= T_TRAIN_CUTOFF].copy()
    v0_max = float(segs_train["v0"].max())
    print(f"{match_id}: {segs_full['segment_id'].nunique()} segments total, "
          f"{segs_train['segment_id'].nunique()} used for PINN training (t<={T_TRAIN_CUTOFF}s)")

    alpha1, vmax1 = baseline1_fit(match_id)
    print(f"{match_id}: Baseline1 (dt={BASELINE1_FIT_DT}s) alpha={alpha1:.3f}, Vmax={vmax1:.3f}")

    config = TrainConfig(
        epochs=10000, lr=3e-3, lr_min_factor=0.05, gamma_final=5.0, gamma_warmup_epochs=2000,
        beta=0.01, n_colloc=1500, t_max=T_TRAIN_CUTOFF, v0_min=0.0, v0_max=v0_max,
        traj_hidden=128, traj_n_hidden=5, seed=0,
    )
    t0 = time.time()
    res = train_pinn(segs_train, config)
    print(f"{match_id}: trained in {time.time()-t0:.0f}s, L_data={res.history['data'].iloc[-1]:.3f}")

    device = config.device
    rows = []
    for tau in EVAL_TAUS:
        gt = bin_ground_truth(segs_full, tau, v0_max)
        if len(gt) < 3:
            print(f"{match_id}: tau={tau}: too few bins with data ({len(gt)}), skipping")
            continue

        v0_t = torch.tensor(gt["v0"].to_numpy(), dtype=torch.float32, device=device).unsqueeze(-1)
        t_t = torch.full_like(v0_t, tau)
        with torch.no_grad():
            pinn_pred = res.traj_net(t_t, v0_t).cpu().numpy()[:, 0]

        baseline_pred = A(alpha1, tau) * gt["v0"].to_numpy() + B(alpha1, vmax1, tau)

        err_pinn = float(np.sqrt(np.mean((pinn_pred - gt["x"].to_numpy()) ** 2)))
        err_baseline = float(np.sqrt(np.mean((baseline_pred - gt["x"].to_numpy()) ** 2)))
        rows.append({
            "match_id": match_id, "tau": tau, "n_bins": len(gt),
            "rmse_pinn": err_pinn, "rmse_baseline1": err_baseline,
        })
        print(f"{match_id}: tau={tau}: n_bins={len(gt)}, RMSE PINN={err_pinn:.3f}m, RMSE Baseline1={err_baseline:.3f}m")

    result = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_json(out_dir / f"{match_id}.json", orient="records", indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-ids", default=",".join(MATCH_IDS))
    parser.add_argument("--out-dir", default="outputs/phase2_pinn/generalization_multi")
    args = parser.parse_args()
    match_ids = args.match_ids.split(",")
    out_dir = Path(args.out_dir)

    done_ids = {p.stem for p in out_dir.glob("*.json")} if out_dir.exists() else set()
    if done_ids:
        print(f"already done: {sorted(done_ids)}")

    all_results = []
    for match_id in match_ids:
        if match_id in done_ids:
            all_results.append(pd.read_json(out_dir / f"{match_id}.json"))
            continue
        all_results.append(run_one_match(match_id, out_dir))

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(out_dir / "all_matches_summary.csv", index=False)
    print(f"\nsaved {out_dir / 'all_matches_summary.csv'}")

    pivot_pinn = combined.pivot(index="tau", columns="match_id", values="rmse_pinn")
    pivot_baseline = combined.pivot(index="tau", columns="match_id", values="rmse_baseline1")
    print("\nRMSE PINN:\n", pivot_pinn.round(3))
    print("\nRMSE Baseline1:\n", pivot_baseline.round(3))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cmap = plt.get_cmap("viridis")
    for i, match_id in enumerate(match_ids):
        color = cmap(i / max(1, len(match_ids) - 1))
        sub = combined[combined["match_id"] == match_id].sort_values("tau")
        axes[0].plot(sub["tau"], sub["rmse_pinn"], "o-", color=color, label=match_id)
        axes[1].plot(sub["tau"], sub["rmse_baseline1"], "o-", color=color, label=match_id)
    for ax, title in zip(axes, ["PINN (time-dependent, single fit)", "Baseline1 (constant, single fit at dt=1s)"]):
        ax.set_xlabel("tau [s]")
        ax.set_title(title)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("RMSE of predicted center x(tau) [m]")
    fig.suptitle("Generalization across tau: all matches")
    fig.tight_layout()
    fig_path = out_dir / "generalization_multi.png"
    fig.savefig(fig_path, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
