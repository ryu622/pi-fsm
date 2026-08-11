"""Phase 2: systematic seed sweep at a FIXED training schedule.

Local pilot runs (phase2_pilot_train.py, documents/phase2_pilot_results.md)
found that the recovered Vmax(t)=F(t)/k(t) plateau varied a lot (~30 to ~40)
between runs — but those runs also differed in epoch count/warmup, so we
couldn't tell whether the instability was due to (a) not having converged
yet, or (b) genuine non-identifiability (multiple F,k splits fitting the
data equally well). This script isolates the seed effect: same schedule
(the one that gave the lowest, most reproducible L_data locally), varying
only the random seed.

- Low variance across seeds (e.g. coefficient of variation < ~10%) => (a):
  the schedule was fine, earlier disagreement was under-convergence.
- Still-high variance => (b): a genuine identifiability gap that a fixed
  schedule alone won't fix (would need richer conditioning / more data /
  explicit regularization on F,k).

Designed to run on Colab (GPU) via notebooks/colab_seed_sweep.ipynb, but
works anywhere `pi_fsm` is installed — just slower without a GPU.

Usage: uv run python scripts/phase2_seed_sweep.py [--match-id J03WPY] [--seeds 0,1,2,3,4] [--out-dir DIR]
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pi_fsm.cache import get_sprint_segments
from pi_fsm.pinn.train import TrainConfig, train_pinn
from pi_fsm.segments import SegmentParams

T_CUTOFF = 2.5  # s: same segment-length restriction that fixed L_data locally
EARLY_WINDOW = (0.3, 2.0)
PLATEAU_WINDOW_START = 1.5

# The schedule that gave the lowest, most reproducible L_data in the local
# pilot (documents/phase2_pilot_results.md, "seed=0" run).
FIXED_SCHEDULE = dict(
    epochs=10000,
    lr=3e-3,
    lr_min_factor=0.05,
    gamma_final=5.0,
    gamma_warmup_epochs=2000,
    n_colloc=1500,
    traj_hidden=128,
    traj_n_hidden=5,
)


def run_one_seed(segs, v0_max: float, seed: int, out_dir: Path) -> dict:
    config = TrainConfig(seed=seed, t_max=T_CUTOFF, v0_min=0.0, v0_max=v0_max, **FIXED_SCHEDULE)
    t0 = time.time()
    res = train_pinn(segs, config)
    dt = time.time() - t0

    device = config.device
    t_eval = torch.linspace(0.05, T_CUTOFF, 200, device=device).unsqueeze(-1)
    with torch.no_grad():
        F_pred = res.force_net(t_eval).cpu().numpy().flatten()
        k_pred = res.resistance_net(t_eval).cpu().numpy().flatten()
    tt = t_eval.cpu().numpy().flatten()
    vmax_pred = F_pred / k_pred

    mask_early = (tt > EARLY_WINDOW[0]) & (tt < EARLY_WINDOW[1])
    mask_plateau = tt > PLATEAU_WINDOW_START

    result = {
        "seed": seed,
        "train_seconds": dt,
        "device": device,
        "final_data_loss": float(res.history["data"].iloc[-1]),
        "final_physics_loss": float(res.history["physics"].iloc[-1]),
        "F_early_mean": float(F_pred[mask_early].mean()),
        "k_early_mean": float(k_pred[mask_early].mean()),
        "vmax_plateau_mean": float(vmax_pred[mask_plateau].mean()),
        "t_eval": tt.tolist(),
        "F_pred": F_pred.tolist(),
        "k_pred": k_pred.tolist(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"seed_{seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"seed={seed}: {dt:.0f}s on {device}, L_data={result['final_data_loss']:.4f}, "
        f"Vmax_plateau={result['vmax_plateau_mean']:.2f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", default="J03WPY")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out-dir", default="outputs/phase2_pinn/seed_sweep")
    parser.add_argument(
        "--v-low", type=float, default=2.0,
        help="segments.SegmentParams.v_low — raise this (e.g. 6.0) to widen the v0 range "
        "captured at segment onset, to test whether v0 diversity was limiting identifiability",
    )
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)

    seg_params = SegmentParams(v_low=args.v_low)
    segs_full = get_sprint_segments(args.match_id, params=seg_params)
    segs = segs_full[segs_full["t"] <= T_CUTOFF].copy()
    v0_max = float(segs["v0"].max())
    print(
        f"{args.match_id} (v_low={args.v_low}): {segs['segment_id'].nunique()} segments (t<={T_CUTOFF}s), "
        f"{len(segs)} rows, v0_max={v0_max:.2f}"
    )
    print(f"schedule: {FIXED_SCHEDULE}")
    print(f"seeds: {seeds}")

    results = [run_one_seed(segs, v0_max, seed, out_dir) for seed in seeds]

    plateaus = np.array([r["vmax_plateau_mean"] for r in results])
    data_losses = np.array([r["final_data_loss"] for r in results])
    cv = float(plateaus.std() / plateaus.mean())
    print()
    print(f"Vmax plateau across seeds: mean={plateaus.mean():.2f}, std={plateaus.std():.2f}, cv={cv:.1%}")
    print(f"L_data across seeds: mean={data_losses.mean():.4f}, std={data_losses.std():.4f}")
    print("VERDICT:", "converged (schedule was the issue)" if cv < 0.10 else "still unstable (likely a genuine identifiability gap)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(results):
        color = cmap(i / max(1, len(results) - 1))
        tt = np.array(r["t_eval"])
        F_pred = np.array(r["F_pred"])
        k_pred = np.array(r["k_pred"])
        axes[0].plot(tt, F_pred / k_pred, color=color, label=f"seed={r['seed']}")
    axes[0].axhline(12.34, color="gray", ls="--", label="Baseline1 (dt=1s)")
    axes[0].set_ylim(0, 50)
    axes[0].set_xlabel("t [s]")
    axes[0].set_title("Vmax(t)=F/k across seeds")
    axes[0].legend(fontsize=8)

    axes[1].bar([str(r["seed"]) for r in results], plateaus, color="#2a78d6")
    axes[1].axhline(12.34, color="gray", ls="--")
    axes[1].set_xlabel("seed")
    axes[1].set_ylabel("Vmax plateau (t>1.5s mean)")
    axes[1].set_title(f"cv={cv:.1%}")

    fig.suptitle(f"Seed sweep, fixed schedule, {args.match_id} (t<={T_CUTOFF}s segments)")
    fig.tight_layout()
    fig_path = out_dir / "seed_sweep_summary.png"
    fig.savefig(fig_path, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"saved {fig_path}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "match_id": args.match_id,
                "schedule": FIXED_SCHEDULE,
                "seeds": seeds,
                "vmax_plateau_mean": float(plateaus.mean()),
                "vmax_plateau_std": float(plateaus.std()),
                "vmax_plateau_cv": cv,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"saved {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
