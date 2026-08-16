"""Detect fixed-threshold PICO MoTracker trajectory jumps in LeRobot data.

The detector is intentionally Qt-free.  It scans a local dataset's Parquet
files and reports frame-to-frame TCP position jumps for each configured hand.
Thresholds are supplied by the caller (normally config.json), so changing the
standard does not require changing this module.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULTS = {
    "source": "observation.state",
    "hands": ["left", "right"],
    "axis_step_threshold": {"x": 0.20, "y": 0.20, "z": 0.20},
    "xyz_step_threshold": 0.35,
    # Protect the UI from an unexpectedly noisy dataset while still counting
    # every event.  Only the first N event details are retained for rendering.
    "max_event_details": 1000,
}


@dataclass(frozen=True)
class TrajectoryEvent:
    episode_index: int
    hand: str
    previous_frame: int
    frame_index: int
    previous_time: float
    timestamp: float
    deltas: dict
    axis_hits: tuple
    xyz_step: float
    xyz_hit: bool


@dataclass
class DetectionResult:
    events: list = field(default_factory=list)
    total_events: int = 0
    affected_episodes: set = field(default_factory=set)
    scanned_transitions: int = 0
    source: str = ""
    thresholds: dict = field(default_factory=dict)
    truncated: bool = False


def resolve_config(cfg=None):
    """Validate and normalize a ``checks.pico_motracker`` configuration."""
    raw = cfg or {}
    source = str(raw.get("source", DEFAULTS["source"]))
    hands = list(raw.get("hands", DEFAULTS["hands"]) or [])
    if not hands or any(hand not in ("left", "right") for hand in hands):
        raise ValueError("hands must contain left and/or right")

    supplied_axis = raw.get("axis_step_threshold", {}) or {}
    axis = {
        name: float(supplied_axis.get(name, DEFAULTS["axis_step_threshold"][name]))
        for name in ("x", "y", "z")
    }
    if any(value <= 0 for value in axis.values()):
        raise ValueError("axis step thresholds must be greater than zero")

    xyz = float(raw.get("xyz_step_threshold", DEFAULTS["xyz_step_threshold"]))
    if xyz <= 0:
        raise ValueError("xyz_step_threshold must be greater than zero")
    max_details = int(raw.get("max_event_details", DEFAULTS["max_event_details"]))
    if max_details < 1:
        raise ValueError("max_event_details must be at least 1")
    return {
        "source": source,
        "hands": hands,
        "axis_step_threshold": axis,
        "xyz_step_threshold": xyz,
        "max_event_details": max_details,
    }


def _feature_indices(dataset_dir, source, hands):
    info_path = Path(dataset_dir) / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {info_path}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid JSON in {info_path}: {exc}") from exc

    feature = (info.get("features") or {}).get(source) or {}
    names = feature.get("names")
    if not isinstance(names, list):
        raise ValueError(f"feature {source!r} has no component names")
    name_to_index = {name: index for index, name in enumerate(names)}
    indices = {}
    for hand in hands:
        required = [f"{hand}_tcp.{axis}" for axis in ("x", "y", "z")]
        missing = [name for name in required if name not in name_to_index]
        if missing:
            raise ValueError(f"feature {source!r} is missing: {', '.join(missing)}")
        indices[hand] = [name_to_index[name] for name in required]
    return indices


def _parquet_files(dataset_dir):
    files = sorted((Path(dataset_dir) / "data").rglob("*.parquet"))
    if not files:
        raise ValueError("no Parquet data files found under data/")
    return files


def detect(dataset_dir, cfg=None):
    """Scan a local dataset and return fixed-threshold trajectory jump events.

    Only consecutive frames inside the same episode are compared.  This avoids
    treating episode boundaries or missing frame ranges as tracker jumps.
    """
    config = resolve_config(cfg)
    source = config["source"]
    hand_indices = _feature_indices(dataset_dir, source, config["hands"])
    result = DetectionResult(source=source, thresholds=config)
    columns = ["episode_index", "frame_index", "timestamp", source]
    previous = None

    for path in _parquet_files(dataset_dir):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=65536, columns=columns):
            episode = np.asarray(batch.column("episode_index"), dtype=np.int64)
            frame = np.asarray(batch.column("frame_index"), dtype=np.int64)
            timestamp = np.asarray(batch.column("timestamp"), dtype=np.float64)
            values = np.asarray(batch.column(source).to_pylist(), dtype=np.float64)
            if not len(episode):
                continue

            if previous is not None:
                episode = np.concatenate(([previous[0]], episode))
                frame = np.concatenate(([previous[1]], frame))
                timestamp = np.concatenate(([previous[2]], timestamp))
                values = np.vstack((previous[3], values))

            consecutive = (episode[1:] == episode[:-1]) & (frame[1:] == frame[:-1] + 1)
            result.scanned_transitions += int(np.count_nonzero(consecutive))
            candidate_rows = np.flatnonzero(consecutive) + 1

            for hand, indices in hand_indices.items():
                xyz = values[:, indices]
                steps = xyz[1:] - xyz[:-1]
                for row in candidate_rows:
                    delta = steps[row - 1]
                    axis_hits = tuple(
                        axis for axis, value in zip(("x", "y", "z"), delta)
                        if abs(value) >= config["axis_step_threshold"][axis]
                    )
                    xyz_step = float(np.linalg.norm(delta))
                    xyz_hit = xyz_step >= config["xyz_step_threshold"]
                    if not axis_hits and not xyz_hit:
                        continue

                    result.total_events += 1
                    result.affected_episodes.add(int(episode[row]))
                    if len(result.events) >= config["max_event_details"]:
                        result.truncated = True
                        continue
                    result.events.append(TrajectoryEvent(
                        episode_index=int(episode[row]),
                        hand=hand,
                        previous_frame=int(frame[row - 1]),
                        frame_index=int(frame[row]),
                        previous_time=float(timestamp[row - 1]),
                        timestamp=float(timestamp[row]),
                        deltas={axis: float(value)
                                for axis, value in zip(("x", "y", "z"), delta)},
                        axis_hits=axis_hits,
                        xyz_step=xyz_step,
                        xyz_hit=xyz_hit,
                    ))

            previous = (int(episode[-1]), int(frame[-1]),
                        float(timestamp[-1]), values[-1].copy())

    result.events.sort(
        key=lambda event: (event.episode_index, event.frame_index, event.hand))
    return result
