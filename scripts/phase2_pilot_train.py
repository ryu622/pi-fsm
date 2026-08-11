"""Phase 2 pilot: train the PINN (gamma>0) and Baseline 2 (gamma=0, plain
trajectory-regression NN) on real sprint segments from one match, and
compare. Small-scale local run — see research_plan.md 4.4 for the
local-vs-Colab division of labor; full multi-match/multi-gamma runs belong
on Colab, not here.

Usage: uv run python scripts/phase2_pilot_train.py [match_id]
Writes: outputs/phase2_pinn/pilot_{match_id}.png, pilot_{match_id}_summary.json
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pi_fsm.baseline import estimate_kinetic_parameters
from pi_fsm.cache import get_sprint_segments
from pi_fsm.data import load_match
from pi_fsm.pinn.train import TrainConfig, train_pinn
from pi_fsm.preprocessing import add_velocity, arrival_points, exclude_goalkeepers

EARLY_WINDOW = (0.3, 2.0)  # s — see phase2_sanity_check.py docstring for why F,k are only meaningful here
OUT_DIR = "outputs/phase2_pinn"


def baseline1_reference(match_id: str) -> tuple[float, float] | None:
    """Baseline 1's alpha, Vmax at delta_t=1s, for a plausibility comparison."""
    m = load_match(match_id)
    df = exclude_goalkeepers(m.players)
    df = add_velocity(df, m.frame_rate)
    ap = arrival_points(df, m.frame_rate, delta_t=1.0)
    res = estimate_kinetic_parameters(ap, delta_t=1.0)
    return (res.alpha, res.vmax) if res else None


def main(match_id: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    segs = get_sprint_segments(match_id)
    v0_max = float(segs["v0"].max())
    t_max = float(segs.groupby("segment_id")["t"].max().max())
    print(f"{match_id}: {segs['segment_id'].nunique()} segments, {len(segs)} rows, "
          f"v0_max={v0_max:.2f}, t_max={t_max:.2f}")

    common = dict(
        epochs=8000, lr=3e-3, lr_min_factor=0.1, n_colloc=1500, t_max=t_max,
        v0_min=0.0, v0_max=v0_max, gamma_warmup_epochs=1500,
    )

    print("training Baseline 2 (gamma=0, no physics constraint)...")
    t0 = time.time()
    baseline2 = train_pinn(segs, TrainConfig(gamma_final=0.0, **common))
    print(f"  done in {time.time()-t0:.0f}s, final data loss={baseline2.history['data'].iloc[-1]:.4f}")

    print("training proposed PINN (gamma=5)...")
    t0 = time.time()
    proposed = train_pinn(segs, TrainConfig(gamma_final=5.0, **common))
    print(f"  done in {time.time()-t0:.0f}s, final data loss={proposed.history['data'].iloc[-1]:.4f}, "
          f"physics loss={proposed.history['physics'].iloc[-1]:.4f}")

    device = proposed.traj_net.mlp.net[0].weight.device
    t_eval = torch.linspace(0.05, t_max, 200, device=device).unsqueeze(-1)
    with torch.no_grad():
        F_pred = proposed.force_net(t_eval).cpu().numpy().flatten()
        k_pred = proposed.resistance_net(t_eval).cpu().numpy().flatten()
    t_eval_np = t_eval.cpu().numpy().flatten()
    vmax_pred = F_pred / k_pred

    mask = (t_eval_np > EARLY_WINDOW[0]) & (t_eval_np < EARLY_WINDOW[1])
    F_early = float(F_pred[mask].mean())
    k_early = float(k_pred[mask].mean())
    vmax_median = float(np.median(vmax_pred))

    ref = baseline1_reference(match_id)
    print(f"PINN early-window (t in {EARLY_WINDOW}): F={F_early:.2f}, k(=alpha)={k_early:.2f}, "
          f"Vmax=F/k median={vmax_median:.2f}")
    if ref:
        print(f"Baseline 1 (dt=1s) for reference: alpha={ref[0]:.2f}, Vmax={ref[1]:.2f}")

    # --- plots -----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].plot(baseline2.history["epoch"], baseline2.history["data"], color="#898781", label="Baseline2 (γ=0)")
    axes[0, 0].plot(proposed.history["epoch"], proposed.history["data"], color="#2a78d6", label="Proposed L_data")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("L_data")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(proposed.history["epoch"], proposed.history["physics"], color="#eb6834")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Proposed: L_physics")
    axes[0, 1].set_xlabel("epoch")

    axes[0, 2].hist(segs.groupby("segment_id")["v0"].first(), bins=30, color="#2a78d6")
    axes[0, 2].set_title("segment v0 distribution")
    axes[0, 2].set_xlabel("v0 [m/s]")

    axes[1, 0].axvspan(*EARLY_WINDOW, color="#2a78d6", alpha=0.08)
    axes[1, 0].plot(t_eval_np, F_pred, color="#2a78d6")
    axes[1, 0].set_title("recovered F(t)")
    axes[1, 0].set_xlabel("t [s]")

    axes[1, 1].axvspan(*EARLY_WINDOW, color="#2a78d6", alpha=0.08)
    axes[1, 1].plot(t_eval_np, k_pred, color="#2a78d6")
    axes[1, 1].set_title("recovered k(t)")
    axes[1, 1].set_xlabel("t [s]")

    axes[1, 2].plot(t_eval_np, vmax_pred, color="#2a78d6", label="PINN Vmax=F/k")
    if ref:
        axes[1, 2].axhline(ref[1], color="#eb6834", ls="--", label=f"Baseline1 Vmax(Δt=1s)={ref[1]:.1f}")
    axes[1, 2].set_ylim(0, max(vmax_pred[mask].max() * 1.5, (ref[1] if ref else 15) * 1.5))
    axes[1, 2].set_title("Vmax(t) = F(t)/k(t)")
    axes[1, 2].set_xlabel("t [s]")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle(f"{match_id}: Phase 2 pilot (Baseline2 vs proposed PINN)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/pilot_{match_id}.png", dpi=150, facecolor="white", bbox_inches="tight")
    print(f"saved {OUT_DIR}/pilot_{match_id}.png")

    summary = {
        "match_id": match_id,
        "n_segments": int(segs["segment_id"].nunique()),
        "n_rows": int(len(segs)),
        "baseline2_final_data_loss": float(baseline2.history["data"].iloc[-1]),
        "proposed_final_data_loss": float(proposed.history["data"].iloc[-1]),
        "proposed_final_physics_loss": float(proposed.history["physics"].iloc[-1]),
        "pinn_F_early": F_early,
        "pinn_k_early": k_early,
        "pinn_vmax_median": vmax_median,
        "baseline1_alpha_dt1": ref[0] if ref else None,
        "baseline1_vmax_dt1": ref[1] if ref else None,
    }
    with open(f"{OUT_DIR}/pilot_{match_id}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    match_id = sys.argv[1] if len(sys.argv) > 1 else "J03WPY"
    main(match_id)
