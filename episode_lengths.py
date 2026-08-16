"""Read and group LeRobot v3 episode lengths for the Workbench UI.

The grouping intentionally mirrors xense-lerobot-viewer's episode-length
histogram rules, but returns ordinary rows rather than chart data.  Keeping the
logic Qt-free makes it easy to test and reuse from the desktop application.
"""

import json
import math
from pathlib import Path


def _js_round(value, digits=0):
    """Round a non-negative value like JavaScript's ``Math.round``."""
    scale = 10 ** digits
    return math.floor(value * scale + 0.5) / scale


def load_episode_lengths(dataset_dir):
    """Return ``(episodes, error)`` for a local LeRobot v3 dataset.

    Each episode is ``{"episode_index", "frames", "length_seconds"}``.
    Only the two small metadata columns required by this view are loaded.
    """
    root = Path(dataset_dir)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return [], "未找到 meta/info.json。"

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        fps = float(info.get("fps") or 0)
    except Exception as exc:
        return [], f"info.json 解析失败: {exc}"
    if not math.isfinite(fps) or fps <= 0:
        return [], "数据集 fps 无效，无法计算 episode 时长。"

    parquet_files = sorted((root / "meta" / "episodes").glob(
        "chunk-*/file-*.parquet"))
    if not parquet_files:
        return [], "未找到 meta/episodes 元数据（当前仅支持 LeRobot v3 数据集）。"

    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        return [], f"缺少 pyarrow，无法读取 episode 时长: {exc}"

    episodes = []
    try:
        for path in parquet_files:
            table = pq.read_table(path, columns=["episode_index", "length"])
            cols = table.to_pydict()
            indices = cols.get("episode_index") or []
            lengths = cols.get("length") or []
            if len(indices) != len(lengths):
                return [], f"{path.name} 的 episode_index/length 列长度不一致。"
            for episode_index, frames in zip(indices, lengths):
                if episode_index is None or frames is None:
                    continue
                frame_count = int(frames)
                episodes.append({
                    "episode_index": int(episode_index),
                    "frames": frame_count,
                    "length_seconds": _js_round(frame_count / fps, 2),
                })
    except Exception as exc:
        return [], f"episode 元数据解析失败: {exc}"

    if not episodes:
        return [], "episode 元数据为空。"
    episodes.sort(key=lambda row: row["episode_index"])
    return episodes, None


def group_episode_lengths(episodes):
    """Group episodes using the same bin geometry as lerobot-viewer.

    The p1/p99 range determines a human-friendly bin width. Values outside that
    range are clamped into the first/last bin, matching the viewer. Empty bins
    are retained in the return value so callers can choose whether to display
    them.
    """
    if not episodes:
        return []

    lengths = [float(row["length_seconds"]) for row in episodes]
    sorted_lengths = sorted(lengths)
    hist_min = min(lengths)
    hist_max = max(lengths)

    if hist_max == hist_min:
        return [{
            "label": f"{hist_min:.1f}s",
            "episodes": sorted(episodes, key=lambda row: row["episode_index"]),
        }]

    count = len(sorted_lengths)
    p1 = sorted_lengths[math.floor(count * 0.01)]
    p99 = sorted_lengths[math.ceil(count * 0.99) - 1]
    value_range = p99 - p1 or 1
    target_bins = max(10, min(50, math.ceil(math.log2(count) + 1)))
    raw_width = value_range / target_bins
    magnitude = 10 ** math.floor(math.log10(raw_width))
    nice_width = next(
        (step * magnitude for step in (1, 2, 2.5, 5, 10)
         if step * magnitude >= raw_width),
        raw_width,
    )
    nice_min = math.floor(p1 / nice_width) * nice_width
    nice_max = math.ceil(p99 / nice_width) * nice_width
    bin_count = max(1, int(_js_round((nice_max - nice_min) / nice_width)))

    bins = [[] for _ in range(bin_count)]
    for episode in episodes:
        bin_index = math.floor(
            (float(episode["length_seconds"]) - nice_min) / nice_width)
        bin_index = max(0, min(bin_count - 1, bin_index))
        bins[bin_index].append(episode)

    groups = []
    for index, members in enumerate(bins):
        low = nice_min + index * nice_width
        high = low + nice_width
        groups.append({
            "label": f"{low:.1f}–{high:.1f}s",
            "episodes": sorted(members, key=lambda row: row["episode_index"]),
        })
    return groups
