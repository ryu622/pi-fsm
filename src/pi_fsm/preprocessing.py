"""Preprocessing matching Narizuka et al. (2023)'s "Investigation method".

Velocity v(t) is the 1-second backward difference (per the paper's
Preliminary analysis section), not a per-frame finite difference. Arrival
points at t+dt are expressed in the rotated frame where the player is at
the origin at time t, facing +x (Fig. 3 of the paper).
"""

import numpy as np
import pandas as pd


def exclude_goalkeepers(players: pd.DataFrame) -> pd.DataFrame:
    return players.loc[~players["is_gk"]].reset_index(drop=True)


def add_velocity(
    players: pd.DataFrame, frame_rate: float, lag_seconds: float = 1.0
) -> pd.DataFrame:
    """Add vx, vy, v0 columns using the 1s backward-difference velocity."""
    lag = round(lag_seconds * frame_rate)
    df = players.sort_values(["player_id", "period_id", "frame_id"]).reset_index(
        drop=True
    )
    g = df.groupby(["player_id", "period_id"])
    x_prev = g["x"].shift(lag)
    y_prev = g["y"].shift(lag)
    frame_prev = g["frame_id"].shift(lag)

    valid = (df["frame_id"] - frame_prev) == lag
    df["vx"] = np.where(valid, (df["x"] - x_prev) / lag_seconds, np.nan)
    df["vy"] = np.where(valid, (df["y"] - y_prev) / lag_seconds, np.nan)
    df["v0"] = np.hypot(df["vx"], df["vy"])
    return df


def arrival_points(
    players_with_velocity: pd.DataFrame, frame_rate: float, delta_t: float
) -> pd.DataFrame:
    """Arrival points at t+delta_t, in the frame where v(t) points along +x.

    Returns a DataFrame with columns: v0, x_arr, y_arr — one row per
    (player, frame t) pair where both v(t) and the position at t+delta_t
    are available.
    """
    lag = round(delta_t * frame_rate)
    df = players_with_velocity.sort_values(
        ["player_id", "period_id", "frame_id"]
    ).reset_index(drop=True)
    g = df.groupby(["player_id", "period_id"])
    x_fut = g["x"].shift(-lag)
    y_fut = g["y"].shift(-lag)
    frame_fut = g["frame_id"].shift(-lag)

    valid = ((frame_fut - df["frame_id"]) == lag) & df["vx"].notna()

    dx = x_fut - df["x"]
    dy = y_fut - df["y"]
    theta = np.arctan2(df["vy"], df["vx"])
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    x_rot = dx * cos_t - dy * sin_t
    y_rot = dx * sin_t + dy * cos_t

    out = pd.DataFrame({"v0": df["v0"], "x_arr": x_rot, "y_arr": y_rot})
    return out.loc[valid].reset_index(drop=True)
