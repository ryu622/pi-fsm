"""Phase 4: summarize the multi-match synthetic recovery run (all 7 matches).

Reads outputs/phase4/synthetic_recovery_all_matches.csv (produced by
scripts/phase4_synthetic_recovery.py --match-ids ... run on Colab, see
notebooks/colab_phase4_synthetic_recovery.ipynb) and outputs/phase1_baseline/
all_matches_summary.csv (the real Vmax(delta_t) curves), and checks whether
the two-regime pattern found on J03WPY alone
(documents/phase4_synthetic_recovery_results.md, 2026-08-14 entry) replicates:
short delta_t explained by measurement noise, long delta_t explained by
PINN's time-varying F(t),k(t).

Usage: uv run python scripts/phase4_multi_match_summary.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"

SYNTH_CSV = "outputs/phase4/synthetic_recovery_all_matches.csv"
BASELINE1_CSV = "outputs/phase1_baseline/all_matches_summary.csv"
OUT_PNG = "outputs/phase4/synthetic_recovery_multi_match_summary.png"


def main() -> None:
    synth = pd.read_csv(SYNTH_CSV)
    base1 = pd.read_csv(BASELINE1_CSV)
    real = base1.set_index(["match_id", "delta_t_actual"])["vmax"]

    match_ids = sorted(synth["match_id"].unique())

    # dt=3.0: does the time-varying-no-noise condition beat the constant condition?
    print("--- delta_t=3.0: |real - const| vs |real - timevarying_no_noise| ---")
    rows = []
    for m in match_ids:
        r = real[(m, 3.0)]
        c = synth[(synth.match_id == m) & (synth.delta_t == 3.0) & (synth.condition == "constant_no_noise")]["vmax"].iloc[0]
        t = synth[(synth.match_id == m) & (synth.delta_t == 3.0) & (synth.condition == "timevarying_no_noise")]["vmax"].iloc[0]
        rows.append({"match_id": m, "real": r, "const": c, "timevarying_no_noise": t,
                      "diff_const": abs(r - c), "diff_tv": abs(r - t), "tv_wins": abs(r - t) < abs(r - c)})
    summary_long = pd.DataFrame(rows)
    print(summary_long.round(2).to_string(index=False))
    print(f"\ntimevarying_no_noise closer to real in {summary_long['tv_wins'].sum()}/{len(summary_long)} matches")

    # dt=0.2: where does real fall relative to the noise-sigma sweep?
    print("\n--- delta_t=0.2: real vs noise-sigma sweep ---")
    rows = []
    for m in match_ids:
        r = real[(m, 0.2)]
        vals = {sigma: synth[(synth.match_id == m) & (synth.delta_t == 0.2) &
                              (synth.condition == f"constant_noise_sigma{sigma}")]["vmax"].iloc[0]
                for sigma in [0.1, 0.3, 0.5, 1.0]}
        rows.append({"match_id": m, "real": r, **{f"sigma={s}": v for s, v in vals.items()}})
    summary_short = pd.DataFrame(rows)
    print(summary_short.round(2).to_string(index=False))

    # small-multiples figure: one panel per match
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    axes = axes.flatten()
    for i, m in enumerate(match_ids):
        ax = axes[i]
        sub = synth[synth.match_id == m]
        for cond, color, lw, label in [
            ("constant_no_noise", "#898781", 1.3, "定数・ノイズなし"),
            ("constant_noise_sigma0.1", "#f0a35a", 1.3, "C: σ=0.1"),
            ("constant_noise_sigma0.3", "#d9720a", 1.3, "C: σ=0.3"),
            ("timevarying_no_noise", "#1baf7a", 2.0, "T': 時間変化"),
        ]:
            c = sub[sub.condition == cond].sort_values("delta_t")
            ax.plot(c["delta_t"], c["vmax"], "o-", color=color, lw=lw, ms=3.5, label=label)
        r = real[m].sort_index()
        ax.plot(r.index, r.values, "ks--", lw=2, ms=6, label="R: 実データ")
        ax.set_title(m, fontsize=10)
        ax.set_ylim(0, 40)
        ax.set_xlabel("Δt [s]", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel("$V_{max}$", fontsize=9)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", fontsize=9)
    fig.suptitle("フェーズ4: 二重メカニズム仮説は全7試合で再現するか(Vmax≤40に軸を制限)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"\nsaved {OUT_PNG}")


if __name__ == "__main__":
    main()
