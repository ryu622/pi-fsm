"""Phase 1 baseline: check whether the Delta-t-dependent instability that
Narizuka et al. (2023) found on J1-league data also appears on idsse-data.

Usage: uv run python scripts/phase1_dt_dependence.py [match_id]
"""

import sys

from pi_fsm.baseline import estimate_kinetic_parameters
from pi_fsm.data import load_match
from pi_fsm.preprocessing import add_velocity, arrival_points, exclude_goalkeepers

DELTA_TS = [0.2, 0.5, 1.0, 2.0, 3.0]


def main(match_id: str) -> None:
    m = load_match(match_id)
    print(f"match {match_id}: {m.players['player_id'].nunique()} players, frame_rate={m.frame_rate}")

    df = exclude_goalkeepers(m.players)
    df = add_velocity(df, m.frame_rate)

    print(f"{'dt_nominal':>10} {'dt_actual':>10} {'alpha':>8} {'vmax':>8} {'n_bins':>7}")
    for dt in DELTA_TS:
        lag = round(dt * m.frame_rate)
        actual_dt = lag / m.frame_rate
        ap = arrival_points(df, m.frame_rate, delta_t=actual_dt)
        res = estimate_kinetic_parameters(ap, delta_t=actual_dt)
        if res is None:
            print(f"{dt:>10.2f} {actual_dt:>10.2f} {'fit failed':>8}")
            continue
        print(f"{dt:>10.2f} {actual_dt:>10.2f} {res.alpha:>8.3f} {res.vmax:>8.3f} {len(res.bins):>7}")


if __name__ == "__main__":
    match_id = sys.argv[1] if len(sys.argv) > 1 else "J03WPY"
    main(match_id)
