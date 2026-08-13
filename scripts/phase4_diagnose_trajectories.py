"""Phase 4 diagnostic: visualize individual synthetic time-varying
trajectories (PINN F(t),k(t), random driving-force direction n) to sanity
check the ODE integration used by phase4_synthetic_recovery.py.

Usage: uv run python scripts/phase4_diagnose_trajectories.py
"""

import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pi_fsm.cache import get_sprint_segments
from pi_fsm.pinn.train import TrainConfig, train_pinn

import os

OUT_DIR = "outputs/phase4"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DT_INTEGRATE = 0.002
T_MAX = 3.0

print("training PINN...")
segs_full = get_sprint_segments("J03WPY")
segs_train = segs_full[segs_full["t"] <= 2.5].copy()
v0_max_train = float(segs_train["v0"].max())
config = TrainConfig(
    epochs=10000, lr=3e-3, lr_min_factor=0.05, gamma_final=5.0, gamma_warmup_epochs=2000,
    beta=0.01, n_colloc=1500, t_max=2.5, v0_min=0.0, v0_max=v0_max_train,
    traj_hidden=128, traj_n_hidden=5, seed=0, device=DEVICE,
)
t0 = time.time()
res = train_pinn(segs_train, config)
print(f"trained in {time.time()-t0:.0f}s, L_data={res.history['data'].iloc[-1]:.3f}")
F_net, k_net = res.force_net, res.resistance_net

# print raw F(t), k(t) values at a few t to see the actual magnitudes
with torch.no_grad():
    for t_val in [0.05, 0.5, 1.0, 1.5, 2.0, 2.5]:
        t_t = torch.full((1, 1), t_val, device=DEVICE)
        print(f"t={t_val}: F={F_net(t_t).item():.4f}, k={k_net(t_t).item():.4f}")

# integrate a handful of individual trajectories with different v0 and n angle
N = 8
rng = np.random.default_rng(0)
v0_np = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
angle_np = rng.uniform(0, 2 * np.pi, size=N).astype(np.float32)
print("v0:", v0_np)
print("angle (deg):", np.degrees(angle_np))

v0 = torch.tensor(v0_np, device=DEVICE)
nx = torch.tensor(np.cos(angle_np), device=DEVICE)
ny = torch.tensor(np.sin(angle_np), device=DEVICE)
x = torch.zeros(N, device=DEVICE)
y = torch.zeros(N, device=DEVICE)
vx = v0.clone()
vy = torch.zeros(N, device=DEVICE)

n_steps = int(round(T_MAX / DT_INTEGRATE))
traj_x = np.zeros((n_steps + 1, N))
traj_y = np.zeros((n_steps + 1, N))
traj_speed = np.zeros((n_steps + 1, N))
traj_x[0] = x.cpu().numpy()
traj_y[0] = y.cpu().numpy()
traj_speed[0] = v0_np

with torch.no_grad():
    for step in range(1, n_steps + 1):
        t = torch.full((1, 1), step * DT_INTEGRATE, device=DEVICE)
        F_t = F_net(t).reshape(())
        k_t = k_net(t).reshape(())
        ax = F_t * nx - k_t * vx
        ay = F_t * ny - k_t * vy
        vx = vx + ax * DT_INTEGRATE
        vy = vy + ay * DT_INTEGRATE
        x = x + vx * DT_INTEGRATE
        y = y + vy * DT_INTEGRATE
        traj_x[step] = x.cpu().numpy()
        traj_y[step] = y.cpu().numpy()
        traj_speed[step] = torch.hypot(vx, vy).cpu().numpy()

t_axis = np.arange(n_steps + 1) * DT_INTEGRATE

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for i, ax in enumerate(axes.flat):
    ax.plot(t_axis, traj_speed[:, i], color="#2a78d6")
    ax.axhline(v0_np[i], color="#eb6834", ls="--", lw=1)
    ax.set_title(f"v0={v0_np[i]:.1f}, angle={np.degrees(angle_np[i]):.0f}deg", fontsize=9)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("speed [m/s]")
fig.suptitle("Synthetic time-varying trajectories: speed(t) (PINN F(t),k(t), random n)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/phase4_traj_speed.png", dpi=150, facecolor="white", bbox_inches="tight")
print("saved speed plot")

fig2, ax2 = plt.subplots(figsize=(6, 6))
for i in range(N):
    ax2.plot(traj_x[:, i], traj_y[:, i], label=f"v0={v0_np[i]:.1f}")
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.legend(fontsize=7)
ax2.set_aspect("equal")
ax2.set_title("Synthetic trajectory paths (x,y)")
fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/phase4_traj_path.png", dpi=150, facecolor="white", bbox_inches="tight")
print("saved path plot")
