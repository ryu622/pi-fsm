"""Phase 2 follow-up: run Baseline 1's population-wide arrival-circle
method (Narizuka et al. 2023's estimation, src/pi_fsm/baseline.py) but
restricted to frames within a detected sprint segment (segments.py) --
i.e. the same "accelerating from near rest" population that the PINN's
segments isolate -- and compare against the full-population Baseline 1
numbers (phase1) and the PINN's Vmax~4.5 (phase2).

Motivation (documents/phase2_pilot_results.md): the PINN's reproducible
Vmax~4.3-4.5 doesn't match full-population Baseline 1 (12.34 at dt=1s).
One hypothesis is that PINN and Baseline 1 are measuring different
populations, not that either is wrong. This script tests that directly,
and additionally checks whether the dt-instability itself persists within
this narrower, matched population.

Usage: uv run python scripts/phase2_matched_population_baseline.py
"""

from pi_fsm.baseline import estimate_kinetic_parameters
from pi_fsm.cache import get_players_with_velocity
from pi_fsm.preprocessing import arrival_points, exclude_goalkeepers
from pi_fsm.segments import SegmentParams, sprint_frame_mask

MATCH_ID = "J03WPY"
DELTA_TS = [0.5, 1.0, 2.0, 3.0]


def main() -> None:
    df, frame_rate = get_players_with_velocity(MATCH_ID)
    df = exclude_goalkeepers(df)

    mask = sprint_frame_mask(df, frame_rate, SegmentParams())
    print(f"full population: {len(df)} rows")
    print(f"sprint-onset population: {mask.sum()} rows ({mask.mean():.1%})")
    print(f"v0 in sprint-onset population: min={df.loc[mask,'v0'].min():.2f}, "
          f"max={df.loc[mask,'v0'].max():.2f}, mean={df.loc[mask,'v0'].mean():.2f}")

    print()
    print(f"{'dt':>6} {'alpha_full':>12} {'vmax_full':>11} {'alpha_sprint':>14} {'vmax_sprint':>13} {'n_bins_sprint':>14}")
    for dt in DELTA_TS:
        lag = round(dt * frame_rate)
        actual_dt = lag / frame_rate

        ap_full = arrival_points(df, frame_rate, delta_t=actual_dt)
        res_full = estimate_kinetic_parameters(ap_full, delta_t=actual_dt)

        ap_sprint = arrival_points(df, frame_rate, delta_t=actual_dt, current_frame_mask=mask)
        res_sprint = estimate_kinetic_parameters(ap_sprint, delta_t=actual_dt)

        af = f"{res_full.alpha:.2f}" if res_full else "fit failed"
        vf = f"{res_full.vmax:.2f}" if res_full else "-"
        as_ = f"{res_sprint.alpha:.2f}" if res_sprint else "fit failed"
        vs = f"{res_sprint.vmax:.2f}" if res_sprint else "-"
        nb = len(res_sprint.bins) if res_sprint else 0
        print(f"{actual_dt:>6.2f} {af:>12} {vf:>11} {as_:>14} {vs:>13} {nb:>14}")


if __name__ == "__main__":
    main()
