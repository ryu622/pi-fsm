"""Parquet caching for preprocessed tracking data and extracted sprint
segments, per research_plan.md 4.2 step 6.

Loading+preprocessing a match takes ~70-120s (mostly the kloppy/Hugging
Face fetch); segment extraction adds a few seconds. Both are cached to disk
keyed by match_id (and, for segments, the SegmentParams used) so repeated
runs / notebook restarts don't pay this cost again.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from pi_fsm.data import load_match
from pi_fsm.preprocessing import add_velocity, exclude_goalkeepers
from pi_fsm.segments import SegmentParams, extract_sprints

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"


def _params_key(params: SegmentParams) -> str:
    digest = hashlib.sha1(json.dumps(asdict(params), sort_keys=True).encode()).hexdigest()
    return digest[:10]


def get_players_with_velocity(
    match_id: str, cache_dir: Path = DEFAULT_CACHE_DIR, force: bool = False
) -> tuple[pd.DataFrame, float]:
    """GK-excluded, velocity-added player frames for a match. Cached."""
    cache_dir = Path(cache_dir)
    data_path = cache_dir / f"{match_id}_players_v.parquet"
    meta_path = cache_dir / f"{match_id}_meta.json"

    if not force and data_path.exists() and meta_path.exists():
        df = pd.read_parquet(data_path)
        meta = json.loads(meta_path.read_text())
        return df, meta["frame_rate"]

    m = load_match(match_id)
    df = exclude_goalkeepers(m.players)
    df = add_velocity(df, m.frame_rate)

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_path)
    meta_path.write_text(
        json.dumps(
            {
                "frame_rate": m.frame_rate,
                "pitch_length": m.pitch_length,
                "pitch_width": m.pitch_width,
            }
        )
    )
    return df, m.frame_rate


def get_sprint_segments(
    match_id: str,
    params: SegmentParams | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Sprint segments (see segments.extract_sprints) for a match. Cached
    per (match_id, params) — different SegmentParams get different cache files."""
    params = params or SegmentParams()
    cache_dir = Path(cache_dir)
    key = _params_key(params)
    seg_path = cache_dir / f"{match_id}_segments_{key}.parquet"

    if not force and seg_path.exists():
        return pd.read_parquet(seg_path)

    df, frame_rate = get_players_with_velocity(match_id, cache_dir, force=force)
    segs = extract_sprints(df, frame_rate, match_id=match_id, params=params)

    cache_dir.mkdir(parents=True, exist_ok=True)
    segs.to_parquet(seg_path)
    return segs


def get_sprint_segments_multi(
    match_ids: list[str],
    params: SegmentParams | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Sprint segments concatenated across multiple matches."""
    return pd.concat(
        [get_sprint_segments(mid, params, cache_dir, force) for mid in match_ids],
        ignore_index=True,
    )
