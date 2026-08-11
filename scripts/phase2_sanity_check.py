"""Phase 2 sanity check: train the PINN on SYNTHETIC data generated from
the Fujimura-Sugihara model with known constant alpha, Vmax, and verify
that the learned F(t), k(t) converge to (approximately) the constant true
values. This must pass before trusting any real-data PINN result.

Important finding from developing this check: F(t), k(t) are only jointly
identifiable from trajectory data near t=0, where different segments (with
different v0) still have visibly different velocities. As t grows, all
segments' velocity converges toward the same Vmax regardless of v0 (that's
the whole point of the model), so the "spread across v0" signal that pins
down F and k *individually* vanishes — only the ratio Vmax=F(t)/k(t)
remains identifiable at late t. We therefore score recovery on an early
time window (where the check is actually meaningful) and separately report
the F/k ratio across the full range as a secondary diagnostic. This isn't
a bug — it also happens to line up with where Phase 1 found the Delta-t
instability to be strongest (Delta t <~ 1s), so it's the regime the
research cares about most anyway.

Usage: uv run python scripts/phase2_sanity_check.py
Writes: outputs/phase2_pinn/sanity_check.png
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pi_fsm.baseline import A, B
from pi_fsm.pinn.train import TrainConfig, train_pinn

ALPHA_TRUE = 1.0  # 1/s
VMAX_TRUE = 12.0  # m/s
F_TRUE = ALPHA_TRUE * VMAX_TRUE  # since m=1: F = k*Vmax = alpha*Vmax
K_TRUE = ALPHA_TRUE  # since m=1: k = alpha*m

T_MAX = 8.0
FRAME_RATE = 25.0
N_SEGMENTS = 200
V0_MAX = 6.0
NOISE_STD = 0.03  # meters, small TRACAB-like per-frame jitter
EARLY_WINDOW = (0.3, 2.5)  # s: where F(t), k(t) are actually identifiable (see module docstring)


def make_synthetic_segments(seed: int = 0, noise_std: float = NOISE_STD) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v0s = rng.uniform(0.2, V0_MAX, size=N_SEGMENTS)
    t_grid = np.arange(0, T_MAX, 1.0 / FRAME_RATE)

    rows = []
    for seg_id, v0 in enumerate(v0s):
        x = np.array([A(ALPHA_TRUE, t) * v0 + B(ALPHA_TRUE, VMAX_TRUE, t) for t in t_grid])
        y = np.zeros_like(x)
        if noise_std > 0:
            x = x + rng.normal(0, noise_std, size=x.shape)
            y = y + rng.normal(0, noise_std, size=y.shape)
        rows.append(pd.DataFrame({"segment_id": seg_id, "t": t_grid, "x": x, "y": y, "v0": v0}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    print("generating synthetic data...")
    segs = make_synthetic_segments()
    print(f"segments: {segs['segment_id'].nunique()}, rows: {len(segs)}")

    config = TrainConfig(
        epochs=8000,
        lr=2e-3,
        gamma_final=5.0,
        gamma_warmup_epochs=1200,
        n_colloc=2000,
        t_max=T_MAX,
        v0_min=0.0,
        v0_max=V0_MAX,
    )
    print(f"device: {config.device}")

    t0 = time.time()
    result = train_pinn(segs, config)
    print(f"trained in {time.time()-t0:.0f}s")

    hist = result.history
    print(hist.tail(5))

    t_eval = torch.linspace(0.1, T_MAX, 200, device=config.device).unsqueeze(-1)
    with torch.no_grad():
        F_pred = result.force_net(t_eval).cpu().numpy().flatten()
        k_pred = result.resistance_net(t_eval).cpu().numpy().flatten()
    t_eval_np = t_eval.cpu().numpy().flatten()
    vmax_pred = F_pred / k_pred

    # F, k are only identifiable early (see module docstring); score there.
    mask = (t_eval_np > EARLY_WINDOW[0]) & (t_eval_np < EARLY_WINDOW[1])
    F_err = np.abs(F_pred[mask].mean() - F_TRUE) / F_TRUE
    k_err = np.abs(k_pred[mask].mean() - K_TRUE) / K_TRUE
    vmax_err_full = np.abs(np.median(vmax_pred) - VMAX_TRUE) / VMAX_TRUE
    print(f"[early window {EARLY_WINDOW}] F: true={F_TRUE:.2f}, recovered={F_pred[mask].mean():.2f}, rel_err={F_err:.1%}")
    print(f"[early window {EARLY_WINDOW}] k: true={K_TRUE:.2f}, recovered={k_pred[mask].mean():.2f}, rel_err={k_err:.1%}")
    print(f"[full range] Vmax=F/k: true={VMAX_TRUE:.2f}, recovered_median={np.median(vmax_pred):.2f}, rel_err={vmax_err_full:.1%}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(hist["epoch"], hist["data"], label="L_data", color="#2a78d6")
    axes[0].plot(hist["epoch"], hist["physics"], label="L_physics", color="#eb6834")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("loss curves")
    axes[0].legend()

    for ax in axes[1:3]:
        ax.axvspan(*EARLY_WINDOW, color="#2a78d6", alpha=0.08, label="scored window")

    axes[1].plot(t_eval_np, F_pred, color="#2a78d6", label="recovered F(t)")
    axes[1].axhline(F_TRUE, color="#eb6834", ls="--", label=f"true F={F_TRUE:.1f}")
    axes[1].set_xlabel("t [s]")
    axes[1].set_title("F(t)")
    axes[1].legend(fontsize=8)

    axes[2].plot(t_eval_np, k_pred, color="#2a78d6", label="recovered k(t)")
    axes[2].axhline(K_TRUE, color="#eb6834", ls="--", label=f"true k={K_TRUE:.1f}")
    axes[2].set_xlabel("t [s]")
    axes[2].set_title("k(t)")
    axes[2].legend(fontsize=8)

    axes[3].plot(t_eval_np, vmax_pred, color="#2a78d6", label="recovered F/k")
    axes[3].axhline(VMAX_TRUE, color="#eb6834", ls="--", label=f"true Vmax={VMAX_TRUE:.1f}")
    axes[3].set_ylim(0, VMAX_TRUE * 2)
    axes[3].set_xlabel("t [s]")
    axes[3].set_title("Vmax = F(t)/k(t)\n(identifiable at all t)")
    axes[3].legend(fontsize=8)

    fig.suptitle(
        f"Sanity check: synthetic const. F,k recovery "
        f"(early-window F err={F_err:.1%}, k err={k_err:.1%}, Vmax ratio err={vmax_err_full:.1%})"
    )
    fig.tight_layout()
    import os
    os.makedirs("outputs/phase2_pinn", exist_ok=True)
    fig.savefig("outputs/phase2_pinn/sanity_check.png", dpi=150, facecolor="white", bbox_inches="tight")
    print("saved outputs/phase2_pinn/sanity_check.png")

    ok = F_err < 0.15 and k_err < 0.15 and vmax_err_full < 0.20
    print("SANITY CHECK:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
