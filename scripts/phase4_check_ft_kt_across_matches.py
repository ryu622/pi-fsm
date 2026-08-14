"""Phase 4 diagnostic: is PINN's k(t) sharp rise around t~0.5-1.5s
(documents/phase4_synthetic_recovery_results.md) specific to J03WPY, or
does it appear consistently across other matches?

Trains the same PINN recipe (t<=2.5s, beta=0.01, seed=0) used for the
phase-4 synthetic ground truth on a few more matches and compares F(t),
k(t) curves directly.

Usage: uv run python scripts/phase4_check_ft_kt_across_matches.py [match_id ...]
"""

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pi_fsm.cache import get_sprint_segments
from pi_fsm.pinn.train import TrainConfig, train_pinn

T_CUTOFF = 2.5
CHECK_TS = [0.05, 0.5, 1.0, 1.5, 2.0, 2.5]


def train_one(match_id: str):
    segs_full = get_sprint_segments(match_id)
    segs_train = segs_full[segs_full["t"] <= T_CUTOFF].copy()
    v0_max = float(segs_train["v0"].max())
    config = TrainConfig(
        epochs=10000, lr=3e-3, lr_min_factor=0.05, gamma_final=5.0, gamma_warmup_epochs=2000,
        beta=0.01, n_colloc=1500, t_max=T_CUTOFF, v0_min=0.0, v0_max=v0_max,
        traj_hidden=128, traj_n_hidden=5, seed=0,
    )
    t0 = time.time()
    res = train_pinn(segs_train, config)
    print(f"{match_id}: trained in {time.time()-t0:.0f}s, L_data={res.history['data'].iloc[-1]:.3f}")
    return res, config.device


def main() -> None:
    match_ids = sys.argv[1:] or ["J03WMX", "J03WN1"]

    results = {}
    for match_id in match_ids:
        res, device = train_one(match_id)
        with torch.no_grad():
            row = []
            for t_val in CHECK_TS:
                t_t = torch.full((1, 1), t_val, device=device)
                row.append((res.force_net(t_t).item(), res.resistance_net(t_t).item()))
        results[match_id] = row
        print(f"{match_id}: F,k at {CHECK_TS} =")
        for t_val, (F_val, k_val) in zip(CHECK_TS, row):
            print(f"  t={t_val}: F={F_val:.4f}, k={k_val:.4f}")

    # J03WPY reference (documents/phase4_synthetic_recovery_results.md)
    results["J03WPY (reference)"] = [
        (4.4030, 0.0256), (4.0182, 0.0446), (3.5364, 0.5029),
        (3.3073, 0.7450), (3.2243, 0.7500), (3.1918, 0.7313),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, (match_id, row) in enumerate(results.items()):
        color = "black" if "reference" in match_id else cmap(i / max(1, len(results) - 1))
        ls = "--" if "reference" in match_id else "-"
        F_vals = [r[0] for r in row]
        k_vals = [r[1] for r in row]
        axes[0].plot(CHECK_TS, F_vals, "o-", color=color, ls=ls, label=match_id)
        axes[1].plot(CHECK_TS, k_vals, "o-", color=color, ls=ls, label=match_id)
    axes[0].set_title("F(t)")
    axes[1].set_title("k(t)")
    for ax in axes:
        ax.set_xlabel("t [s]")
        ax.legend(fontsize=8)
    fig.suptitle("F(t), k(t) across matches: is the k(t) rise consistent?")
    fig.tight_layout()
    fig.savefig("outputs/phase4/ft_kt_across_matches.png", dpi=150, facecolor="white", bbox_inches="tight")
    print("saved outputs/phase4/ft_kt_across_matches.png")


if __name__ == "__main__":
    main()
