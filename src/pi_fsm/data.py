"""Load idsse-data tracking data via kloppy into a tidy long-format DataFrame.

Coordinates are transformed to the TRACAB system (meters, origin at pitch
center), matching the coordinate convention used by Narizuka et al. (2023),
who also used TRACAB tracking data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from kloppy import sportec
from kloppy.domain import PositionType, Provider, TrackingDataset

MATCH_IDS = [
    "J03WPY",
    "J03WMX",
    "J03WN1",
    "J03WOH",
    "J03WOY",
    "J03WQQ",
    "J03WR9",
]

CM_PER_M = 100.0


@dataclass(frozen=True)
class MatchTrackingData:
    match_id: str
    frame_rate: float
    pitch_length: float
    pitch_width: float
    players: pd.DataFrame  # long format, see load_match docstring


def _player_is_gk(player) -> bool:
    position = player.starting_position or (
        player.positions.last() if player.positions else None
    )
    return position == PositionType.Goalkeeper


def load_match(match_id: str, limit: int | None = None) -> MatchTrackingData:
    """Load one match's tracking data in long format.

    Returned `players` DataFrame columns:
        period_id, frame_id, t (seconds, period-relative), player_id,
        team_id, is_gk, x, y (meters, TRACAB coordinate system)
    """
    dataset: TrackingDataset = sportec.load_open_tracking_data(
        match_id=match_id, limit=limit
    )
    dataset = dataset.transform(to_coordinate_system=Provider.TRACAB)

    player_meta = {}
    for team in dataset.metadata.teams:
        for player in team.players:
            player_meta[player.player_id] = (team.team_id, _player_is_gk(player))

    wide = dataset.to_df()
    wide["t"] = wide["timestamp"].dt.total_seconds()

    player_ids = sorted(
        {col[: -len("_x")] for col in wide.columns if col.endswith("_x") and col != "ball_x"}
    )

    frames = []
    for pid in player_ids:
        x_col, y_col = f"{pid}_x", f"{pid}_y"
        if x_col not in wide.columns:
            continue
        team_id, is_gk = player_meta.get(pid, (None, False))
        sub = wide[["period_id", "frame_id", "t", x_col, y_col]].rename(
            columns={x_col: "x", y_col: "y"}
        )
        sub["player_id"] = pid
        sub["team_id"] = team_id
        sub["is_gk"] = is_gk
        frames.append(sub)

    players = pd.concat(frames, ignore_index=True)
    players["x"] = players["x"] / CM_PER_M
    players["y"] = players["y"] / CM_PER_M
    players = players.dropna(subset=["x", "y"]).reset_index(drop=True)

    pitch = dataset.metadata.pitch_dimensions
    return MatchTrackingData(
        match_id=match_id,
        frame_rate=dataset.metadata.frame_rate,
        pitch_length=pitch.pitch_length,
        pitch_width=pitch.pitch_width,
        players=players,
    )
