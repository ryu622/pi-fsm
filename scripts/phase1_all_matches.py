"""Phase 1 baseline: check the Delta-t-dependent instability across all
7 idsse-data matches, not just J03WPY.

Usage: uv run python scripts/phase1_all_matches.py
Writes: outputs/phase1_baseline/all_matches_summary.csv
"""

import os
import time

import pandas as pd

from pi_fsm.baseline import estimate_kinetic_parameters
from pi_fsm.data import MATCH_IDS, load_match
from pi_fsm.preprocessing import add_velocity, arrival_points, exclude_goalkeepers

DELTA_TS = [0.2, 0.5, 1.0, 2.0, 3.0]
OUT_PATH = "outputs/phase1_baseline/all_matches_summary.csv"


def run_match(match_id: str) -> list[dict]:
    t0 = time.time()
    m = load_match(match_id)
    df = exclude_goalkeepers(m.players)
    df = add_velocity(df, m.frame_rate)

    rows = []
    for dt in DELTA_TS:
        lag = round(dt * m.frame_rate)
        actual_dt = lag / m.frame_rate
        ap = arrival_points(df, m.frame_rate, delta_t=actual_dt)
        res = estimate_kinetic_parameters(ap, delta_t=actual_dt)
        rows.append(
            {
                "match_id": match_id,
                "delta_t_nominal": dt,
                "delta_t_actual": actual_dt,
                "alpha": res.alpha if res else None,
                "vmax": res.vmax if res else None,
                "n_bins": len(res.bins) if res else 0,
            }
        )
    print(f"{match_id}: done in {time.time()-t0:.0f}s")
    return rows


def main() -> None:
    done_ids = set()
    if os.path.exists(OUT_PATH):
        done_ids = set(pd.read_csv(OUT_PATH)["match_id"].unique())
        print(f"already done: {sorted(done_ids)}")

    for match_id in MATCH_IDS:
        if match_id in done_ids:
            continue
        rows = run_match(match_id)
        df = pd.DataFrame(rows)
        header = not os.path.exists(OUT_PATH)
        df.to_csv(OUT_PATH, mode="a", header=header, index=False)

    summary = pd.read_csv(OUT_PATH)
    print(f"\nwrote {OUT_PATH}")
    print(summary.pivot(index="delta_t_actual", columns="match_id", values="alpha").round(2))
    print()
    print(summary.pivot(index="delta_t_actual", columns="match_id", values="vmax").round(2))


if __name__ == "__main__":
    main()
