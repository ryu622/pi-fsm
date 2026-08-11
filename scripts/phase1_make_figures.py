"""Generate Phase-1 baseline figures for the pi-fsm report artifact."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np

from pi_fsm.data import load_match
from pi_fsm.preprocessing import add_velocity, arrival_points, exclude_goalkeepers
from pi_fsm.baseline import A, estimate_kinetic_parameters

import os

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1_baseline")
os.makedirs(OUT, exist_ok=True)

BLUE_SEQ = LinearSegmentedColormap.from_list(
    "blue_seq",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"],
)
SERIES_BLUE = "#2a78d6"
ACCENT_ORANGE = "#eb6834"
INK = "#14171c"
MUTED = "#6b7280"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "font.size": 11,
})

DELTA_TS = [0.2, 0.5, 1.0, 2.0, 3.0]
V0_PANELS = [1, 3, 5, 7]

print("loading match...")
m = load_match("J03WPY")
df = exclude_goalkeepers(m.players)
df = add_velocity(df, m.frame_rate)
print("loaded & velocity computed:", df.shape)

results = {}
for dt in DELTA_TS:
    lag = round(dt * m.frame_rate)
    actual_dt = lag / m.frame_rate
    ap = arrival_points(df, m.frame_rate, delta_t=actual_dt)
    res = estimate_kinetic_parameters(ap, delta_t=actual_dt)
    results[dt] = (actual_dt, ap, res)
    print(dt, actual_dt, res.alpha if res else None, res.vmax if res else None)

# --- Figure 1: arrival heat maps (dt=1,2 rows x v0=1,3,5,7 cols) -----------
fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
for row, dt in enumerate([1.0, 2.0]):
    actual_dt, ap, res = results[dt]
    for col, v0 in enumerate(V0_PANELS):
        ax = axes[row, col]
        sub = ap[(ap["v0"] >= v0 - 0.15) & (ap["v0"] < v0 + 0.15)]
        extent = 12 if dt == 1.0 else 20
        bins = np.linspace(-extent * 0.6, extent, 60)
        ybins = np.linspace(-extent * 0.8, extent * 0.8, 60)
        h = ax.hist2d(
            sub["x_arr"], sub["y_arr"], bins=[bins, ybins], cmap=BLUE_SEQ,
            norm=LogNorm(vmin=1, vmax=max(sub.shape[0] // 20, 10)),
        )
        if res is not None:
            cx = A(res.alpha, actual_dt) * v0
            r = res.rc_mean if False else None
        # overlay fitted circle from per-bin fit if available
        row_bin = res.bins[np.isclose(res.bins["v0"], v0, atol=0.2)] if res is not None else None
        if res is not None and row_bin is not None and len(row_bin):
            from pi_fsm.baseline import B
            cx = A(res.alpha, actual_dt) * v0
            r = B(res.alpha, res.vmax, actual_dt)
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(cx + r * np.cos(theta), r * np.sin(theta), color=ACCENT_ORANGE, lw=1.6)
        ax.set_title(f"$v_0$={v0} m/s", fontsize=10, color=INK)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(ybins[0], ybins[-1])
        ax.set_aspect("equal")
        if col == 0:
            ax.set_ylabel(f"$\\Delta t$={dt:g}s\ny [m]", fontsize=10)
        if row == 1:
            ax.set_xlabel("x [m]", fontsize=10)
fig.suptitle("Arrival-point heat maps with fitted arrival circle (Eq. 4)", fontsize=13, y=1.00)
fig.tight_layout()
fig.savefig(f"{OUT}/heatmaps.png", dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved heatmaps.png")

# --- Figure 2: xc, yc vs v0 regression --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
for ax, dt in zip(axes, [1.0, 2.0]):
    actual_dt, ap, res = results[dt]
    bins = res.bins
    ax.scatter(bins["v0"], bins["xc"], s=22, color=SERIES_BLUE, label="$x_c$", zorder=3)
    ax.scatter(bins["v0"], bins["yc"], s=22, facecolors="none", edgecolors=MUTED, label="$y_c$", zorder=3)
    v0_line = np.linspace(0, 6, 50)
    ax.plot(v0_line, res.slope * v0_line, color=ACCENT_ORANGE, lw=1.8, label=f"fit: slope={res.slope:.2f}")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_title(f"$\\Delta t$={dt:g}s   →  $\\alpha$={res.alpha:.2f}/s", fontsize=11)
    ax.set_xlabel("$v_0$ [m/s]")
    ax.legend(fontsize=8, frameon=False)
axes[0].set_ylabel("center coordinate [m]")
fig.suptitle("Circle-center $v_0$ dependence (Eq. 7) — proportional fit gives $A(\\alpha,\\Delta t)$", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/regression_center.png", dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved regression_center.png")

# --- Figure 3: rc vs v0 ------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
for ax, dt in zip(axes, [1.0, 2.0]):
    actual_dt, ap, res = results[dt]
    bins = res.bins
    ax.scatter(bins["v0"], bins["rc"], s=22, color=SERIES_BLUE, zorder=3)
    ax.axhline(res.rc_mean, color=ACCENT_ORANGE, lw=1.8, label=f"mean={res.rc_mean:.2f} m")
    ax.axvline(6, color="#c3c2b7", lw=0.8, ls="--")
    ax.set_title(f"$\\Delta t$={dt:g}s   →  $V_{{max}}$={res.vmax:.2f} m/s", fontsize=11)
    ax.set_xlabel("$v_0$ [m/s]")
    ax.legend(fontsize=8, frameon=False)
axes[0].set_ylabel("radius $r_c$ [m]")
fig.suptitle("Circle-radius $v_0$ independence (Eq. 8) — mean gives $B(\\alpha,V_{max},\\Delta t)$", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/regression_radius.png", dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved regression_radius.png")

# --- Figure 4: alpha, Vmax vs delta_t (two small multiples, no dual axis) ---
dts_actual = [results[dt][0] for dt in DELTA_TS]
alphas = [results[dt][2].alpha if results[dt][2] else np.nan for dt in DELTA_TS]
vmaxs = [results[dt][2].vmax if results[dt][2] else np.nan for dt in DELTA_TS]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
axes[0].plot(dts_actual, alphas, "o-", color=SERIES_BLUE, lw=1.8, ms=6)
axes[0].set_title("$\\alpha$ vs $\\Delta t$", fontsize=11)
axes[0].set_xlabel("$\\Delta t$ [s]")
axes[0].set_ylabel("$\\alpha$ [1/s]")
axes[1].plot(dts_actual, vmaxs, "o-", color=ACCENT_ORANGE, lw=1.8, ms=6)
axes[1].set_title("$V_{max}$ vs $\\Delta t$", fontsize=11)
axes[1].set_xlabel("$\\Delta t$ [s]")
axes[1].set_ylabel("$V_{max}$ [m/s]")
fig.suptitle("Kinetic-parameter instability across $\\Delta t$ (cf. paper Fig. 7)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/dt_dependence.png", dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved dt_dependence.png")

# dump summary numbers
import json
summary = {
    str(dt): {
        "actual_dt": results[dt][0],
        "alpha": results[dt][2].alpha if results[dt][2] else None,
        "vmax": results[dt][2].vmax if results[dt][2] else None,
        "slope": results[dt][2].slope if results[dt][2] else None,
        "rc_mean": results[dt][2].rc_mean if results[dt][2] else None,
    }
    for dt in DELTA_TS
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
