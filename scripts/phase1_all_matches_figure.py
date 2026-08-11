import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("outputs/phase1_baseline/all_matches_summary.csv")
match_ids = sorted(df["match_id"].unique())
cmap = plt.get_cmap("viridis")
colors = {mid: cmap(i / (len(match_ids) - 1)) for i, mid in enumerate(match_ids)}

plt.rcParams.update({
    "font.family": "Hiragino Sans",
    "axes.edgecolor": "#c3c2b7",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, col, title, ylabel in zip(
    axes, ["alpha", "vmax"],
    ["α vs Δt(7試合)", "V_max vs Δt(7試合)"],
    ["α [1/s]", "V_max [m/s]"],
):
    for mid in match_ids:
        sub = df[df["match_id"] == mid].sort_values("delta_t_actual")
        ax.plot(sub["delta_t_actual"], sub[col], "o-", color=colors[mid], lw=1.5, ms=4)
        last = sub.iloc[-1]
        ax.annotate(mid, (last["delta_t_actual"], last[col]), textcoords="offset points",
                    xytext=(6, 0), fontsize=7.5, color=colors[mid], va="center")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Δt [s]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.6)

fig.suptitle("idsse-data 全7試合でのΔt依存パラメータ不安定性", fontsize=12)
fig.tight_layout()
fig.savefig("outputs/phase1_baseline/all_matches_dt_dependence.png", dpi=170, facecolor="white", bbox_inches="tight")
print("saved")
