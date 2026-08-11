"""Sprint segment extraction for PINN training (Phase 2).

A "sprint" is a contiguous stretch of frames starting from a slow speed and
showing a sustained speed increase, per research_plan.md 4.2 step 4. Each
extracted segment is rotated into the frame where the player's *net heading
over the first `window_s` seconds* points along +x, matching the fixed
n=(1,0) driving-force-direction assumption used in the PINN physics loss.

Note this deliberately does NOT use the instantaneous velocity direction at
t=0: since every segment starts near v_low (slow, by construction), a
single 1s-backward-difference velocity sample there is dominated by TRACAB
position noise (~1m) and gives a near-random angle. Averaging the heading
over the first second (a much larger displacement than the noise floor)
gives a far more stable rotation reference. An early version of this module
used the instantaneous v(0) direction and produced segments with 50m+ of
apparent "lateral drift" — an artifact of rotating by a noisy angle, not
real movement (see notebooks/02_segment_exploration.ipynb).

A segment is cut at the first sustained deceleration after its peak speed
(rather than always running the full t_max), based on notebooks/
02_segment_exploration.ipynb showing that fixed-length windows often
contain a second, unrelated acceleration burst. Segments that never
decelerate and instead run out the full t_max are dropped — in practice
these are not clean sprints but sustained low-effort wandering (e.g. a
player jogging/repositioning) that happened to cross v_low. Segments
containing a single-frame speed spike above `max_frame_speed`
(tracking-error artifacts, e.g. an ID swap) are also dropped entirely.

Thresholds below are a first-pass default — inspect segment count/length/v0
distributions before trusting downstream results, and adjust here if
needed.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

V_LOW_DEFAULT = 2.0  # m/s: must be below this to be considered "not already sprinting"
DELTA_V_DEFAULT = 2.0  # m/s: required speed increase within `window_s` to trigger a start
WINDOW_S_DEFAULT = 1.0  # s: window over which the speed increase is measured
T_MAX_DEFAULT = 10.0  # s: max segment length (MSD-analysis-justified straight-line range)
DECEL_MARGIN_DEFAULT = 1.0  # m/s: drop from running-peak v0 that counts as "decelerating"
DECEL_HOLD_S_DEFAULT = 0.5  # s: how long the drop must persist to cut the segment there
MAX_FRAME_SPEED_DEFAULT = 12.0  # m/s: frame-to-frame speed above this => tracking-error, drop segment
MIN_DURATION_S_DEFAULT = 1.0  # s: minimum segment length to keep
MIN_PEAK_SPEED_DEFAULT = 4.0  # m/s: segment must reach at least this speed to count as a "sprint"


@dataclass(frozen=True)
class SegmentParams:
    v_low: float = V_LOW_DEFAULT
    delta_v: float = DELTA_V_DEFAULT
    window_s: float = WINDOW_S_DEFAULT
    t_max: float = T_MAX_DEFAULT
    decel_margin: float = DECEL_MARGIN_DEFAULT
    decel_hold_s: float = DECEL_HOLD_S_DEFAULT
    max_frame_speed: float = MAX_FRAME_SPEED_DEFAULT
    min_duration_s: float = MIN_DURATION_S_DEFAULT
    min_peak_speed: float = MIN_PEAK_SPEED_DEFAULT


def _find_segment_end(
    v0: np.ndarray, frame_id: np.ndarray, start: int, frame_rate: float, params: SegmentParams
) -> tuple[int, bool]:
    """Returns (end_index, clean_end). end_index is inclusive: the first
    sustained deceleration after the running-peak v0, or a frame gap.
    clean_end is False if neither happened and the segment just ran out at
    t_max or the end of available data (see module docstring: these are
    dropped as likely non-sprint wandering, not truncated)."""
    t_max_frames = round(params.t_max * frame_rate)
    decel_hold_frames = max(1, round(params.decel_hold_s * frame_rate))
    limit = min(start + t_max_frames, len(v0) - 1)

    peak_val = v0[start]
    peak_idx = start
    below_since = None
    for j in range(start + 1, limit + 1):
        if frame_id[j] - frame_id[j - 1] != 1:
            return j - 1, False
        if v0[j] > peak_val:
            peak_val = v0[j]
            peak_idx = j
            below_since = None
        elif peak_val - v0[j] >= params.decel_margin:
            if below_since is None:
                below_since = j
            elif j - below_since + 1 >= decel_hold_frames:
                return peak_idx, True
        else:
            below_since = None
    return limit, False


def _extract_for_group(
    frame_id: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    v0: np.ndarray,
    frame_rate: float,
    match_id: str,
    player_id: str,
    period_id: int,
    params: SegmentParams,
) -> list[tuple]:
    n = len(v0)
    window = round(params.window_s * frame_rate)
    min_duration_frames = round(params.min_duration_s * frame_rate)
    rows = []
    i = 0
    while i < n - window:
        if np.isnan(v0[i]) or np.isnan(vx[i]) or frame_id[i + window] - frame_id[i] != window:
            i += 1
            continue
        is_start = v0[i] < params.v_low and (v0[i + window] - v0[i]) >= params.delta_v
        if not is_start:
            i += 1
            continue

        end, clean_end = _find_segment_end(v0, frame_id, i, frame_rate, params)
        if not clean_end:
            i = end + 1
            continue
        if end - i < min_duration_frames:
            i += 1
            continue

        if v0[i : end + 1].max() < params.min_peak_speed:
            i = end + 1
            continue

        frame_speed = np.hypot(np.diff(x[i : end + 1]), np.diff(y[i : end + 1])) / (1.0 / frame_rate)
        if len(frame_speed) and frame_speed.max() > params.max_frame_speed:
            i = end + 1  # still skip past it, just don't emit
            continue

        # Rotation reference: net heading over the first `window` frames,
        # not the noisy instantaneous v(0) (see module docstring).
        theta = np.arctan2(y[i + window] - y[i], x[i + window] - x[i])
        cos_t, sin_t = np.cos(-theta), np.sin(-theta)
        dx = x[i : end + 1] - x[i]
        dy = y[i : end + 1] - y[i]
        x_rot = dx * cos_t - dy * sin_t
        y_rot = dx * sin_t + dy * cos_t
        t_rel = (frame_id[i : end + 1] - frame_id[i]) / frame_rate

        seg_id = f"{match_id}_{player_id}_{period_id}_{frame_id[i]}"
        v0_start = v0[i]
        rows.extend(
            (seg_id, match_id, player_id, tt, xx, yy, v0_start)
            for tt, xx, yy in zip(t_rel, x_rot, y_rot)
        )
        i = end + 1  # skip past this segment (avoids overlapping starts)

    return rows


def extract_sprints(
    players_with_velocity: pd.DataFrame,
    frame_rate: float,
    match_id: str,
    params: SegmentParams | None = None,
) -> pd.DataFrame:
    """Extract sprint segments, rotated so v(t=0) points along +x.

    Expects GK-excluded, velocity-added input (preprocessing.exclude_goalkeepers
    + add_velocity). Returns long format: segment_id, match_id, player_id,
    t (seconds from segment start), x, y (rotated frame, meters), v0
    (segment's initial speed, constant within a segment).
    """
    params = params or SegmentParams()
    all_rows = []
    for (player_id, period_id), g in players_with_velocity.sort_values(
        ["player_id", "period_id", "frame_id"]
    ).groupby(["player_id", "period_id"]):
        all_rows.extend(
            _extract_for_group(
                g["frame_id"].to_numpy(),
                g["x"].to_numpy(),
                g["y"].to_numpy(),
                g["vx"].to_numpy(),
                g["vy"].to_numpy(),
                g["v0"].to_numpy(),
                frame_rate,
                match_id,
                player_id,
                period_id,
                params,
            )
        )
    return pd.DataFrame(
        all_rows, columns=["segment_id", "match_id", "player_id", "t", "x", "y", "v0"]
    )
