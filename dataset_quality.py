"""Local episode-level quality checks for downloaded LeRobot datasets.

These checks intentionally run only when a dataset is available on disk. They
look for issues that the dataset-level summary cannot localize:

* short one-frame/short-window flicker in the six visual streams
* abnormal start/end robot state compared with sibling episodes
* action/state jumps that indicate overly fast or discontinuous motion

The module is Qt-free and dependency-light. Video decoding uses OpenCV only
when it is installed; state/action checks use pyarrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import csv
import html
import math
import os
import re
import shutil
import json
from pathlib import Path
import datetime as _dt


class QualityCancelled(RuntimeError):
    """Raised when the GUI asks a long-running quality scan to stop."""


STREAM_FIELDS = (
    "left_wrist",
    "right_wrist",
    "left_tactile_left",
    "left_tactile_right",
    "right_tactile_left",
    "right_tactile_right",
)

VECTOR_COLUMNS = (
    "observation.state",
    "state",
    "action",
)

# Tactile gel cameras: a static, low-texture image is their NORMAL state (no
# contact = no change), so freeze/blur heuristics built for wrist views would
# fire constantly on them.
TACTILE_FIELDS = tuple(f for f in STREAM_FIELDS if "tactile" in f)

DEFAULT_CFG = {
    "start_end_window": 10,
    "boundary_mad_factor": 6.0,
    "boundary_abs_threshold": 0.20,
    "jump_mad_factor": 14.0,
    "jump_abs_threshold": 0.35,
    "flicker_sample_step": 1,
    "flicker_luma_threshold": 45.0,
    "flicker_recover_ratio": 0.45,
    "max_video_frames": 12000,
    # UMI-style handheld capture degrades in known ways (see UMI/TacUMI and
    # score_lerobot_episodes): motion blur from fast motion, bad exposure in
    # dim/backlit scenes, frozen camera streams, padded idle trajectories and
    # outlier episode lengths. Thresholds below gate those checks.
    "blur_laplacian_threshold": 45.0,   # absolute floor: below = blurry
    "blur_rel_ratio": 0.12,             # ...or below this share of the video's
                                        # own median sharpness (camera-agnostic)
    "blur_min_sec": 0.8,                # sustained this long before flagging
    "exposure_low_luma": 25.0,          # mean gray below = 欠曝
    "exposure_high_luma": 230.0,        # mean gray above = 过曝
    "exposure_min_sec": 0.8,
    "freeze_diff_threshold": 0.4,       # mean abs frame diff below = frozen
    "freeze_min_sec": 1.0,
    "idle_epsilon": 0.004,              # per-frame state L2 delta ~ standstill
    "idle_max_sec": 5.0,                # longest tolerated standstill
    "length_iqr_factor": 2.5,           # Tukey fence for episode durations
    "clip_margin_sec": 1.0,
    "clip_lead_sec": 4.0,
    "clip_max_sec": 10.0,
    "remote_enabled": True,
    "remote_cache_dir": ".quality_cache",
    "report_dir": ".quality_reports",
    "remote_max_episodes": 0,
    "max_issues": 80,
}


@dataclass
class Issue:
    severity: str
    rule: str
    message: str
    episode_index: int | None = None
    frame_index: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    field: str | None = None
    path: str | None = None
    clip_path: str | None = None
    clip_paths: dict | None = None
    plot_path: str | None = None  # per-DOF line chart (trajectory issues)


def _cfg(cfg):
    merged = dict(DEFAULT_CFG)
    if isinstance(cfg, dict):
        merged.update(cfg or {})
    return merged


def dataset_dir(dataset, out_dir="datasets/TacVerse"):
    local_dir = (dataset or {}).get("local_dir")
    if local_dir and (Path(local_dir) / "meta" / "info.json").is_file():
        return Path(local_dir)

    leaf = ((dataset or {}).get("dataset_name") or "").split("/")[-1]
    if not leaf:
        return None
    base = Path(out_dir)
    candidates = []
    flat = base / leaf
    if (flat / "meta" / "info.json").is_file():
        candidates.append(flat)
    candidates.extend(
        p.parent.parent
        for p in base.glob(f"*/{leaf}/meta/info.json")
        if p.is_file()
    )
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _repo_leaf(repo_id):
    return _safe_name((repo_id or "").split("/")[-1])


def _hub_local_dir(local_dir):
    """Return an extended Win32 path for Hub cache files over MAX_PATH."""
    path = str(Path(local_dir).resolve())
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path.lstrip("\\")
    return "\\\\?\\" + path


def _episode_key(path):
    ep = _episode_from_name(path)
    return ep if ep is not None else -1


def _remote_allow_patterns(files, cfg):
    patterns = {
        "meta/info.json", "meta/tasks.parquet", "meta/episodes/**",
        "*.log", "**/*.log", "logs/**", "**/*log*.txt", "**/*log*.json",
    }
    data_files = [f for f in files if f.startswith("data/") and f.endswith(".parquet")]
    video_files = [
        f for f in files
        if f.startswith("videos/")
        and f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
        and any(field in f for field in STREAM_FIELDS)
    ]

    max_eps = int(cfg.get("remote_max_episodes") or 0)
    if max_eps > 0:
        eps = sorted({_episode_key(f) for f in data_files + video_files if _episode_key(f) >= 0})[:max_eps]
        keep = set(eps)
        data_files = [f for f in data_files if _episode_key(f) in keep]
        video_files = [f for f in video_files if _episode_key(f) in keep]

    patterns.update(data_files)
    patterns.update(video_files)
    return sorted(patterns)


def remote_dataset_dir(dataset, cfg):
    """Materialize only files needed by the quality scanner into .quality_cache."""
    if not cfg.get("remote_enabled", True):
        return None, "远程检查未启用"
    repo_id = (dataset or {}).get("dataset_name")
    if not repo_id or "/" not in repo_id:
        return None, "缺少 Hugging Face repo id，无法远程检查"

    token = cfg.get("token")
    cache_root = Path(cfg.get("remote_cache_dir") or ".quality_cache")
    local_dir = cache_root / _repo_leaf(repo_id)
    try:
        from huggingface_hub import HfApi, snapshot_download
        files = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
        allow_patterns = _remote_allow_patterns(files, cfg)
        if not any(str(p).startswith("data/") for p in allow_patterns):
            return None, "远程数据集中未发现 data/*.parquet，无法做轨迹检查"
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            local_dir=_hub_local_dir(local_dir),
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        return None, f"远程按需缓存失败: {exc}"

    if not (local_dir / "meta" / "info.json").is_file():
        return None, "远程缓存缺少 meta/info.json"
    return local_dir, None


def _episode_from_name(path):
    stem = Path(path).stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    return int(digits[-1]) if digits else None


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("_") or "unknown"


def _round_time(value):
    return round(float(value), 3) if value is not None else None


def _issue(rule, message, episode_index=None, frame_index=None, field=None,
           path=None, severity="fail", fps=None, start_sec=None, end_sec=None):
    if start_sec is None and frame_index is not None and fps:
        center = frame_index / fps
        start_sec = max(0.0, center - 1.0)
        end_sec = center + 1.0
    return Issue(
        severity, rule, message,
        episode_index=episode_index,
        frame_index=frame_index,
        start_sec=_round_time(start_sec),
        end_sec=_round_time(end_sec),
        field=field,
        path=str(path) if path else None,
    )


def _to_vectors(values):
    rows = []
    for value in values or []:
        if value is None:
            rows.append(None)
            continue
        if isinstance(value, (int, float)):
            rows.append([float(value)])
            continue
        if isinstance(value, (list, tuple)):
            flat = []
            stack = list(value)
            while stack:
                item = stack.pop(0)
                if isinstance(item, (list, tuple)):
                    stack = list(item) + stack
                elif item is None:
                    flat.append(float("nan"))
                else:
                    try:
                        flat.append(float(item))
                    except (TypeError, ValueError):
                        pass
            rows.append(flat or None)
            continue
        rows.append(None)
    return rows


def _finite_vec(vec):
    if not vec:
        return None
    vals = [x for x in vec if isinstance(x, float) and math.isfinite(x)]
    return vals if vals else None


def _mean_vec(vectors):
    vectors = [_finite_vec(v) for v in vectors if _finite_vec(v)]
    if not vectors:
        return None
    width = min(len(v) for v in vectors)
    if width <= 0:
        return None
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(width)]


def _l2(a, b):
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n <= 0:
        return None
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))


def _median(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _mad_threshold(values, factor, abs_threshold):
    med = _median(values)
    if med is None:
        return abs_threshold
    mad = _median([abs(v - med) for v in values if v is not None])
    if not mad:
        return max(abs_threshold, med * 3)
    return max(abs_threshold, med + factor * mad)


def _read_data_tables(root):
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        return {}, [Issue("skip", "local_dependency", f"缺少 pyarrow，无法做轨迹检查: {exc}")]

    episodes = {}
    issues = []
    for path in sorted((root / "data").glob("**/*.parquet")):
        try:
            table = pq.read_table(path)
        except Exception as exc:
            issues.append(Issue("warn", "parquet_read", f"数据 parquet 读取失败: {exc}", path=str(path)))
            continue
        cols = set(table.column_names)
        ep_col = "episode_index" if "episode_index" in cols else None
        frame_col = "frame_index" if "frame_index" in cols else None
        available = [c for c in VECTOR_COLUMNS if c in cols]
        if not available:
            continue
        data = table.select([c for c in [ep_col, frame_col, *available] if c]).to_pydict()
        row_count = table.num_rows
        for i in range(row_count):
            ep = data.get(ep_col, [None] * row_count)[i] if ep_col else _episode_from_name(path)
            if ep is None:
                continue
            rec = episodes.setdefault(int(ep), {"frames": [], "columns": {}})
            frame = data.get(frame_col, [i] * row_count)[i] if frame_col else i
            rec["frames"].append(int(frame) if frame is not None else i)
        for col in available:
            values = _to_vectors(data.get(col))
            ep_values = data.get(ep_col, [None] * row_count) if ep_col else [_episode_from_name(path)] * row_count
            frame_values = data.get(frame_col, list(range(row_count))) if frame_col else list(range(row_count))
            for ep, frame, vec in zip(ep_values, frame_values, values):
                if ep is None:
                    continue
                rec = episodes.setdefault(int(ep), {"frames": [], "columns": {}})
                rec["columns"].setdefault(col, []).append((int(frame), vec))
    return episodes, issues


def _check_boundaries(episodes, cfg, fps):
    issues = []
    window = max(1, int(cfg["start_end_window"]))
    for col in VECTOR_COLUMNS:
        starts, ends = {}, {}
        for ep, rec in episodes.items():
            seq = sorted(rec.get("columns", {}).get(col, []), key=lambda x: x[0])
            if not seq:
                continue
            starts[ep] = _mean_vec([v for _, v in seq[:window]])
            ends[ep] = _mean_vec([v for _, v in seq[-window:]])

        for label, points, rule in (
            ("起始位置", starts, "boundary_start"),
            ("结束位置", ends, "boundary_end"),
        ):
            valid = {ep: v for ep, v in points.items() if v}
            if len(valid) < 3:
                continue
            center = []
            width = min(len(v) for v in valid.values())
            for i in range(width):
                center.append(_median([v[i] for v in valid.values()]))
            distances = {ep: _l2(v, center) for ep, v in valid.items()}
            threshold = _mad_threshold(
                list(distances.values()),
                float(cfg["boundary_mad_factor"]),
                float(cfg["boundary_abs_threshold"]),
            )
            for ep, dist in sorted(distances.items(), key=lambda x: x[1] or 0, reverse=True):
                if dist is not None and dist > threshold:
                    rec = episodes.get(ep, {})
                    frames = sorted(rec.get("frames") or [])
                    if rule == "boundary_start":
                        start_sec = 0.0
                        end_sec = min(float(cfg["clip_max_sec"]), window / fps)
                        frame = frames[0] if frames else 0
                    else:
                        last_frame = frames[-1] if frames else None
                        end_sec = (last_frame / fps) if last_frame is not None else None
                        start_sec = max(0.0, (end_sec or float(cfg["clip_max_sec"])) - float(cfg["clip_max_sec"]))
                        frame = last_frame
                    issues.append(_issue(
                        rule,
                        f"{label}偏离同数据集中位轨迹，{col} L2={dist:.4g} > {threshold:.4g}",
                        episode_index=ep, frame_index=frame, field=col,
                        fps=fps, start_sec=start_sec, end_sec=end_sec))
    return issues


def _check_jumps(episodes, cfg, fps):
    issues = []
    for ep, rec in episodes.items():
        for col, seq in rec.get("columns", {}).items():
            seq = sorted(seq, key=lambda x: x[0])
            jumps = []
            for (prev_frame, prev), (frame, cur) in zip(seq, seq[1:]):
                dist = _l2(_finite_vec(prev), _finite_vec(cur))
                if dist is not None:
                    jumps.append((frame, dist))
            if len(jumps) < 5:
                continue
            threshold = _mad_threshold(
                [d for _, d in jumps],
                float(cfg["jump_mad_factor"]),
                float(cfg["jump_abs_threshold"]),
            )
            for frame, dist in jumps:
                if dist > threshold:
                    center = frame / fps
                    margin = float(cfg["clip_margin_sec"])
                    issues.append(_issue(
                        "trajectory_jump",
                        f"{col} 相邻帧突变 L2={dist:.4g} > {threshold:.4g}",
                        episode_index=ep, frame_index=frame, field=col, fps=fps,
                        start_sec=max(0.0, center - margin),
                        end_sec=center + margin))
                    break
    return issues


def _check_idle(episodes, cfg, fps):
    """Flag long standstills inside an episode (padded/hesitant trajectories).

    score_lerobot_episodes calls these "idle periods and inflated
    trajectories" — dead time that adds training cost without signal."""
    issues = []
    epsilon = float(cfg["idle_epsilon"])
    max_idle = float(cfg["idle_max_sec"])
    for ep, rec in episodes.items():
        cols = rec.get("columns", {})
        col = next((c for c in VECTOR_COLUMNS if cols.get(c)), None)
        if not col:
            continue
        seq = sorted(cols[col], key=lambda x: x[0])
        run_start = None
        worst = None  # (length_sec, start_frame, end_frame)
        for (prev_frame, prev), (frame, cur) in zip(seq, seq[1:]):
            dist = _l2(_finite_vec(prev), _finite_vec(cur))
            if dist is not None and dist < epsilon:
                if run_start is None:
                    run_start = prev_frame
                length = (frame - run_start) / fps
                if worst is None or length > worst[0]:
                    worst = (length, run_start, frame)
            else:
                run_start = None
        if worst and worst[0] > max_idle:
            length, f0, f1 = worst
            issues.append(_issue(
                "idle_period",
                f"{col} 连续静止 {length:.1f}s (> {max_idle:g}s)，疑似长时间停顿/注水轨迹",
                episode_index=ep, frame_index=f0, field=col, fps=fps,
                start_sec=f0 / fps, end_sec=f1 / fps))
    return issues


def _check_episode_length(episodes, cfg, fps):
    """Flag episodes whose duration falls outside the Tukey IQR fence of the
    dataset — usually aborted takes that should have been deleted, or runaway
    recordings."""
    durations = {}
    for ep, rec in episodes.items():
        frames = sorted(rec.get("frames") or [])
        if frames:
            durations[ep] = (frames[-1] - frames[0] + 1) / fps
    if len(durations) < 5:
        return []
    values = sorted(durations.values())
    q1 = values[int(0.25 * (len(values) - 1))]
    q3 = values[int(0.75 * (len(values) - 1))]
    iqr = max(q3 - q1, 1e-9)
    factor = float(cfg["length_iqr_factor"])
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    med = _median(values)
    issues = []
    for ep, sec in sorted(durations.items(), key=lambda x: x[1]):
        if sec < lo or sec > hi:
            kind = "偏短" if sec < lo else "偏长"
            issues.append(_issue(
                "episode_length",
                f"episode 时长 {sec:.1f}s 明显{kind}(数据集中位 {med:.1f}s)，疑似失败重录未删或超时录制",
                episode_index=ep, frame_index=0, fps=fps,
                start_sec=0.0, end_sec=min(sec, float(cfg["clip_max_sec"]))))
    return issues


def _video_paths(root):
    paths = []
    for field in STREAM_FIELDS:
        matched = sorted((root / "videos").glob(f"**/*{field}*/**/*.mp4"))
        matched += sorted((root / "videos").glob(f"**/*{field}*.mp4"))
        for path in matched:
            paths.append((field, path))
    seen = set()
    out = []
    for field, path in paths:
        key = (field, str(path))
        if key not in seen:
            seen.add(key)
            out.append((field, path))
    return out


def _videos_by_episode(root):
    mapped = _videos_by_episode_meta(root)
    if mapped:
        return mapped
    by_ep = {}
    for field, path in _video_paths(root):
        ep = _episode_from_name(path)
        if ep is None:
            continue
        by_ep.setdefault(ep, {})[field] = {"path": path, "offset": 0.0}
    return by_ep


def _videos_by_episode_meta(root):
    try:
        import pyarrow.parquet as pq
    except Exception:
        return {}
    root = Path(root)
    info = {}
    try:
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    template = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    by_ep = {}
    for ep_file in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        try:
            table = pq.read_table(ep_file)
        except Exception:
            continue
        cols = table.to_pydict()
        episodes = cols.get("episode_index") or []
        for i, ep in enumerate(episodes):
            if ep is None:
                continue
            ep = int(ep)
            for field in STREAM_FIELDS:
                video_key = f"observation.images.{field}"
                prefix = f"videos/{video_key}"
                chunks = cols.get(f"{prefix}/chunk_index")
                files = cols.get(f"{prefix}/file_index")
                starts = cols.get(f"{prefix}/from_timestamp")
                if not chunks or not files:
                    continue
                chunk_index = chunks[i]
                file_index = files[i]
                if chunk_index is None or file_index is None:
                    continue
                rel = template.format(
                    video_key=video_key,
                    chunk_index=int(chunk_index),
                    file_index=int(file_index),
                )
                path = root / rel
                if path.is_file():
                    offset = float(starts[i] or 0.0) if starts else 0.0
                    by_ep.setdefault(ep, {})[field] = {"path": path, "offset": offset}
    return by_ep


def _choose_video(issue, videos):
    if issue.episode_index is None:
        return None
    fields = videos.get(issue.episode_index) or {}
    if issue.field in fields:
        return issue.field, fields[issue.field]
    for field in ("left_wrist", "right_wrist", *STREAM_FIELDS):
        if field in fields:
            return field, fields[field]
    if fields:
        field, info = next(iter(fields.items()))
        return field, info
    return None


def _video_fps(path):
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    cap.release()
    return float(fps) if fps and fps > 0 else None


@lru_cache(maxsize=1)
def _ffmpeg_exe():
    """An H.264-capable ffmpeg: PATH first, then imageio-ffmpeg's bundled one."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _is_h264(path):
    try:
        import cv2
        import struct
        cap = cv2.VideoCapture(str(path))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()
        codec = struct.pack("<I", fourcc).decode(errors="replace").lower()
        return codec in ("h264", "avc1")
    except Exception:
        return True  # can't probe -> don't churn the file


def _transcode_h264(path):
    """Re-encode a clip to H.264/yuv420p in place so browsers can play it.

    OpenCV's VideoWriter only has MPEG-4 Part 2 ("mp4v") available in this
    env, which HTML5 <video> cannot decode — the report showed every clip as
    0:00. Best effort: without ffmpeg the mp4v file is kept (still playable
    in desktop players)."""
    exe = _ffmpeg_exe()
    if not exe:
        return
    import subprocess
    path = Path(path)
    tmp = path.with_suffix(".h264.mp4")
    cmd = [exe, "-y", "-v", "error", "-i", str(path),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=180)
        if r.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            path.unlink()
            tmp.rename(path)
        else:
            tmp.unlink(missing_ok=True)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _write_clip(src, dst, start_sec, end_sec, *, episode_start_sec=None, camera_name=None):
    try:
        import cv2
    except Exception:
        return None
    src, dst = Path(src), Path(dst)
    if dst.is_file():
        # Clips cached by an older run may still be browser-unplayable mp4v;
        # upgrade them in place instead of serving them forever.
        if not _is_h264(dst):
            _transcode_h264(dst)
        return dst
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        return None
    start_frame = max(0, int(float(start_sec or 0) * fps))
    end_frame = int(float(end_sec or (start_sec or 0) + 1) * fps)
    if total:
        end_frame = min(max(start_frame + 1, end_frame), total)
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur = start_frame
    while cur < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        t_abs = cur / fps
        t_ep = t_abs - float(episode_start_sec or 0.0)
        label = f"{camera_name or src.stem}  video {t_abs:.2f}s"
        if episode_start_sec is not None:
            label += f"  episode {max(0.0, t_ep):.2f}s"
        cv2.rectangle(frame, (8, 8), (min(width - 1, 620), 48), (0, 0, 0), -1)
        cv2.putText(
            frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        cur += 1
    writer.release()
    cap.release()
    if not dst.is_file():
        return None
    _transcode_h264(dst)
    return dst


def _clip_targets(issue, fields):
    per_stream_rules = ("camera_flicker", "jpeg_log_error", "motion_blur",
                        "exposure", "camera_freeze")
    if issue.rule in per_stream_rules and issue.field in fields:
        chosen = issue.field if issue.field in fields else None
        return [chosen] if chosen else []
    wrists = [field for field in ("left_wrist", "right_wrist") if field in fields]
    return wrists or ([next(iter(fields))] if fields else [])


def _attach_clips(root, issues, cfg, default_fps, progress=None, cancel=None):
    videos = _videos_by_episode(root)
    clip_root = root / "quality_clips"
    actionable = sum(1 for x in issues if x.severity in ("warn", "fail"))
    done = 0
    for issue in issues:
        if issue.severity not in ("warn", "fail"):
            continue
        _check_cancel(cancel)
        done += 1
        pct = 88 + int(7 * (done - 1) / actionable) if actionable else 88
        _emit(progress, f"生成问题视频切片 {done}/{actionable}...", pct)
        if issue.episode_index is None:
            continue
        fields = videos.get(issue.episode_index) or {}
        targets = _clip_targets(issue, fields)
        if not targets:
            continue
        clip_paths = {}
        first_source = None
        base_fps = default_fps
        first_info = fields.get(targets[0]) if targets else None
        if isinstance(first_info, dict) and first_info.get("path"):
            base_fps = _video_fps(first_info["path"]) or default_fps
        if issue.start_sec is None or issue.end_sec is None:
            if issue.frame_index is None:
                continue
            center = issue.frame_index / base_fps
            margin = float(cfg["clip_margin_sec"])
            issue.start_sec = _round_time(max(0.0, center - margin))
            issue.end_sec = _round_time(center + margin)
        if issue.end_sec <= issue.start_sec:
            issue.end_sec = _round_time(issue.start_sec + 1.0)
        # The issue window [start_sec, end_sec] must survive intact; the lead
        # is expendable context. So when the total exceeds clip_max_sec, shrink
        # the lead first instead of trimming the tail (the old behaviour cut
        # the event itself off end-of-episode clips).
        lead = float(cfg.get("clip_lead_sec", 0.0) or 0.0)
        max_len = float(cfg["clip_max_sec"])
        clip_end = issue.end_sec
        window = clip_end - issue.start_sec
        if window > max_len:
            clip_end = _round_time(issue.start_sec + max_len)
            window = max_len
        lead = min(lead, max_len - window)
        clip_start = _round_time(max(0.0, issue.start_sec - lead))
        for video_field in targets:
            video_info = fields.get(video_field)
            if isinstance(video_info, dict):
                video_path = video_info.get("path")
                video_offset = float(video_info.get("offset") or 0.0)
            else:
                video_path = video_info
                video_offset = 0.0
            if not video_path:
                continue
            source_start = clip_start + video_offset
            source_end = clip_end + video_offset
            filename = (
                f"ep{issue.episode_index:06d}_"
                f"{_safe_name(issue.rule)}_"
                f"{_safe_name(video_field)}_"
                f"{clip_start:.3f}-{clip_end:.3f}.mp4"
            )
            clip = _write_clip(
                video_path, clip_root / filename, source_start, source_end,
                episode_start_sec=video_offset, camera_name=video_field)
            if clip:
                clip_paths[video_field] = str(clip)
                first_source = first_source or str(video_path)
        if clip_paths:
            issue.clip_paths = clip_paths
            issue.clip_path = next(iter(clip_paths.values()))
            issue.path = first_source
    return issues


def _attach_dof_plots(root, issues, episodes, cfg, default_fps):
    """Render a per-DOF line chart around each trajectory jump.

    The clip shows WHAT happened; this shows WHICH joint did it — one line
    per state/action dimension over the same window as the clip, with the
    jump instant marked. Skipped silently when matplotlib is missing."""
    targets = [x for x in issues
               if x.rule == "trajectory_jump" and x.episode_index is not None
               and x.field and x.start_sec is not None]
    if not targets or not episodes:
        return issues
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return issues
    plot_root = root / "quality_clips"
    lead = float(cfg.get("clip_lead_sec", 0.0) or 0.0)
    for issue in targets:
        rec = episodes.get(issue.episode_index) or {}
        seq = sorted(rec.get("columns", {}).get(issue.field) or [], key=lambda x: x[0])
        if not seq:
            continue
        fps = default_fps
        f_lo = int(max(0.0, issue.start_sec - lead) * fps)
        f_hi = int(issue.end_sec * fps) + 1
        window = [(f, v) for f, v in seq if f_lo <= f <= f_hi and v]
        if len(window) < 3:
            continue
        width = min(len(v) for _, v in window)
        times = [f / fps for f, _ in window]
        fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=110)
        for dim in range(width):
            ax.plot(times, [v[dim] for _, v in window], lw=1.0, label=f"dof{dim}")
        if issue.frame_index is not None:
            ax.axvline(issue.frame_index / fps, color="red", ls="--", lw=1.2)
        ax.set_xlabel("episode time (s)")
        ax.set_ylabel(issue.field)
        # English title: matplotlib's default font has no CJK glyphs.
        ax.set_title(f"ep{issue.episode_index} {issue.field} jump @ "
                     f"{(issue.frame_index or 0) / fps:.2f}s", fontsize=10)
        if width <= 16:
            ax.legend(fontsize=6, ncol=4, loc="upper right", framealpha=0.6)
        fig.tight_layout()
        plot_root.mkdir(parents=True, exist_ok=True)
        png = plot_root / (
            f"ep{issue.episode_index:06d}_trajectory_jump_"
            f"{_safe_name(issue.field)}_{issue.start_sec:.3f}_dof.png")
        try:
            fig.savefig(png)
            issue.plot_path = str(png)
        except Exception:
            pass
        finally:
            plt.close(fig)
    return issues


def _check_flicker(root, cfg, progress=None, cancel=None):
    try:
        import cv2
    except Exception as exc:
        return [Issue("skip", "local_dependency", f"缺少 opencv-python，无法做六路画面闪烁检查: {exc}")]

    issues = []
    step = max(1, int(cfg["flicker_sample_step"]))
    luma_threshold = float(cfg["flicker_luma_threshold"])
    recover_ratio = float(cfg["flicker_recover_ratio"])
    max_frames = int(cfg["max_video_frames"])
    blur_threshold = float(cfg["blur_laplacian_threshold"])
    low_luma = float(cfg["exposure_low_luma"])
    high_luma = float(cfg["exposure_high_luma"])
    freeze_diff = float(cfg["freeze_diff_threshold"])
    margin = float(cfg["clip_margin_sec"])
    videos = _video_paths(root)
    total = len(videos)
    for vid_no, (field, path) in enumerate(videos, 1):
        _check_cancel(cancel)
        # This is by far the longest stage (full decode of every stream), so
        # report per-video progress across the 60-75% band instead of sitting
        # silently at 60%. Blur/exposure/freeze checks piggyback on the same
        # decode pass, so they cost no extra decoding.
        pct = 60 + int(15 * (vid_no - 1) / total) if total else 60
        _emit(progress, f"检查视频闪烁/模糊/曝光/冻结 {vid_no}/{total}: {path.name}", pct)
        ep = _episode_from_name(path)
        is_tactile = field in TACTILE_FIELDS
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            issues.append(Issue("warn", "video_read", f"{field}: 视频无法打开", ep, field=field, path=str(path)))
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # sustained-run lengths in samples (durations are configured in secs)
        need = lambda key: max(3, int(float(cfg[key]) * fps / step))
        blur_need, expo_need, freeze_need = (
            need("blur_min_sec"), need("exposure_min_sec"), need("freeze_min_sec"))
        prev = None
        prev_diff = None
        prev_gray = None
        runs = {"dark": 0, "bright": 0, "freeze": 0}
        sharp_series = []  # (frame_idx, laplacian_var) on well-exposed frames
        reported = set()  # at most one issue per rule per video
        frame_idx = -1
        sampled = 0

        def _video_issue(rule, message, at_frame):
            center = at_frame / fps
            issues.append(_issue(
                rule, message, episode_index=ep, frame_index=at_frame,
                field=field, path=path, fps=fps,
                start_sec=max(0.0, center - margin), end_sec=center + margin))
            reported.add(rule)

        while sampled < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % step:
                continue
            sampled += 1
            if sampled % 600 == 0 and cancel and cancel():
                cap.release()  # keep 取消 responsive mid-video
                raise QualityCancelled("检查已取消")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean = float(gray.mean())

            # -- flicker (single-frame luma spike that recovers) --------------
            if prev is not None:
                diff = abs(mean - prev)
                if prev_diff is not None and prev_diff > luma_threshold and diff > luma_threshold * recover_ratio:
                    if "camera_flicker" not in reported:
                        _video_issue(
                            "camera_flicker",
                            f"{field} 疑似短时闪烁，亮度突变 {prev_diff:.1f}",
                            frame_idx - step)
                prev_diff = diff
            prev = mean

            # -- motion blur (UMI: fast motion smears the wrist views) --------
            # Absolute Laplacian thresholds don't transfer between cameras, so
            # sharpness is collected here (well-exposed frames only, to avoid
            # double-reporting dark scenes) and judged against the video's own
            # median after decoding. Tactile gel views are inherently smooth
            # and motionless — blur/freeze do not apply to them.
            if not is_tactile and low_luma <= mean <= high_luma:
                sharp_series.append(
                    (frame_idx, float(cv2.Laplacian(gray, cv2.CV_64F).var())))

            # -- exposure (dim / backlit scenes break perception) --------------
            if "exposure" not in reported:
                runs["dark"] = runs["dark"] + 1 if mean < low_luma else 0
                runs["bright"] = runs["bright"] + 1 if mean > high_luma else 0
                if runs["dark"] >= expo_need:
                    _video_issue("exposure", f"{field} 画面持续欠曝(平均亮度 {mean:.0f} < {low_luma:g})", frame_idx)
                elif runs["bright"] >= expo_need:
                    _video_issue("exposure", f"{field} 画面持续过曝(平均亮度 {mean:.0f} > {high_luma:g})", frame_idx)

            # -- frozen stream (encoder/driver stall repeats frames) -----------
            # Skipped for tactile cameras: no contact = no change is normal.
            if not is_tactile and prev_gray is not None and "camera_freeze" not in reported:
                frame_delta = float(cv2.absdiff(gray, prev_gray).mean())
                runs["freeze"] = runs["freeze"] + 1 if frame_delta < freeze_diff else 0
                if runs["freeze"] >= freeze_need:
                    _video_issue(
                        "camera_freeze",
                        f"{field} 画面疑似冻结/重复帧({runs['freeze'] * step / fps:.1f}s 无变化)",
                        frame_idx)
            prev_gray = gray

            if len(issues) >= int(cfg["max_issues"]):
                cap.release()
                return issues
        cap.release()
        if sharp_series and "motion_blur" not in reported:
            med = _median([s for _, s in sharp_series]) or 0.0
            thr = max(blur_threshold, float(cfg["blur_rel_ratio"]) * med)
            run = 0
            for fidx, sharp in sharp_series:
                run = run + 1 if sharp < thr else 0
                if run >= blur_need:
                    _video_issue(
                        "motion_blur",
                        f"{field} 画面持续模糊(清晰度 {sharp:.0f}，低于本视频基准 {med:.0f} 的 {100 * float(cfg['blur_rel_ratio']):.0f}%)，疑似运动过快或镜头脏污/失焦",
                        fidx)
                    break
    return issues


def _first_number(patterns, text):
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except (TypeError, ValueError):
                pass
    return None


def _infer_stream(text):
    for field in STREAM_FIELDS:
        if field in text:
            return field
    for field in STREAM_FIELDS:
        dotted = f"observation.images.{field}"
        if dotted in text:
            return field
    return None


def _check_jpeg_logs(root, cfg, fps):
    issues = []
    seen = set()
    candidates = []
    for pattern in ("*.log", "**/*.log", "**/*log*.txt", "**/*log*.json"):
        candidates.extend(Path(root).glob(pattern))
    for path in sorted(set(candidates)):
        if ".cache" in path.parts:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, 1):
            if not re.search(r"jpe?g", line, flags=re.IGNORECASE):
                continue
            text = f"{path} {line}"
            ep_val = _first_number((r"\bepisode[_\s:=#-]*(\d+)", r"\bep[_\s:=#-]*(\d+)"), text)
            frame_val = _first_number((r"\bframe[_\s:=#-]*(\d+)", r"\bframe_index[_\s:=#-]*(\d+)"), text)
            time_val = _first_number((r"\btime(?:stamp)?[_\s:=#-]*([0-9]+(?:\.[0-9]+)?)", r"\bt[_\s:=#-]*([0-9]+(?:\.[0-9]+)?)"), text)
            field = _infer_stream(text)
            ep = int(ep_val) if ep_val is not None else None
            frame = int(frame_val) if frame_val is not None else None
            if time_val is None and frame is not None:
                time_val = frame / fps
            key = (ep, field, path.name)
            if key in seen:
                continue
            seen.add(key)
            start = max(0.0, time_val - float(cfg["clip_margin_sec"])) if time_val is not None else None
            end = time_val + float(cfg["clip_margin_sec"]) if time_val is not None else None
            msg = f"日志检测到 JPEG 异常，可能对应画面闪烁/白屏: {path.name}:{line_no} {line.strip()[:180]}"
            issues.append(_issue(
                "jpeg_log_error", msg,
                episode_index=ep, frame_index=frame, field=field,
                fps=fps, start_sec=start, end_sec=end))
    return issues


def _emit(progress, text, pct=None):
    if progress:
        progress(text, pct)


def _check_cancel(cancel):
    if cancel and cancel():
        raise QualityCancelled("检查已取消")


def _scan_path_uncached(root, cfg, progress=None, cancel=None):
    root = Path(root)
    issues = []
    _check_cancel(cancel)
    _emit(progress, "读取 parquet 轨迹数据...", 10)
    episodes, read_issues = _read_data_tables(root)
    issues.extend(read_issues)
    default_fps = float(cfg.get("fps") or 30.0)
    _check_cancel(cancel)
    if episodes:
        _emit(progress, f"检查首尾位置偏差，共 {len(episodes)} 条 episode...", 25)
        issues.extend(_check_boundaries(episodes, cfg, default_fps))
        _check_cancel(cancel)
        _emit(progress, "检查轨迹突变/动作过快...", 40)
        issues.extend(_check_jumps(episodes, cfg, default_fps))
        _check_cancel(cancel)
        _emit(progress, "检查空闲停顿与 episode 时长分布...", 50)
        issues.extend(_check_idle(episodes, cfg, default_fps))
        issues.extend(_check_episode_length(episodes, cfg, default_fps))
    else:
        issues.append(Issue("skip", "trajectory_data", "未找到可检查的 data/*.parquet 轨迹列"))
    _check_cancel(cancel)
    _emit(progress, "检查视频闪烁/白屏线索...", 60)
    issues.extend(_check_flicker(root, cfg, progress=progress, cancel=cancel))
    _check_cancel(cancel)
    _emit(progress, "扫描 JPEG/相机异常日志...", 75)
    issues.extend(_check_jpeg_logs(root, cfg, default_fps))
    _check_cancel(cancel)
    _emit(progress, "生成问题视频切片...", 88)
    issues = _attach_clips(root, issues, cfg, default_fps,
                           progress=progress, cancel=cancel)
    _check_cancel(cancel)
    _emit(progress, "绘制突变问题的自由度折线图...", 94)
    issues = _attach_dof_plots(root, issues, episodes, cfg, default_fps)
    max_issues = int(cfg["max_issues"])
    _emit(progress, "检查规则完成。", 95)
    return tuple(issues[:max_issues])


@lru_cache(maxsize=32)
def scan_path(root_str, cfg_items):
    root = Path(root_str)
    cfg = _cfg(dict(cfg_items))
    return _scan_path_uncached(root, cfg)


def scan_dataset(dataset, out_dir="datasets/TacVerse", cfg=None, progress=None, cancel=None):
    root = dataset_dir(dataset, out_dir)
    merged = _cfg(cfg)
    if (dataset or {}).get("fps"):
        merged["fps"] = dataset.get("fps")
    if not root:
        _emit(progress, "远程按需缓存检查文件...", 5)
        root, error = remote_dataset_dir(dataset, merged)
        if not root:
            return [Issue("skip", "remote_dataset", error or "无法获取远程检查所需文件")]
    _check_cancel(cancel)
    if progress or cancel:
        return list(_scan_path_uncached(root.resolve(), merged, progress=progress, cancel=cancel))
    return list(scan_path(str(root.resolve()), tuple(sorted(merged.items()))))


def _issue_line(issue):
    parts = []
    if issue.episode_index is not None:
        parts.append(f"episode {issue.episode_index}")
    if issue.start_sec is not None and issue.end_sec is not None:
        parts.append(f"{issue.start_sec:.3f}s-{issue.end_sec:.3f}s")
    elif issue.frame_index is not None:
        parts.append(f"frame {issue.frame_index}")
    if issue.field:
        parts.append(issue.field)
    prefix = " / ".join(parts)
    return f"{prefix}: {issue.message}" if prefix else issue.message


def _issue_record(issue, clips=None, plot=None):
    return {
        "episode": issue.episode_index,
        "rule": issue.rule,
        "severity": issue.severity,
        "field": issue.field,
        "start_sec": issue.start_sec,
        "end_sec": issue.end_sec,
        "frame": issue.frame_index,
        "message": issue.message,
        "source": issue.path,
        "clips": clips or {},
        "plot": plot,
        "review_status": "未确认",
    }


def _issue_id(rec, fallback):
    ep = "unknown" if rec.get("episode") is None else f"{int(rec['episode']):06d}"
    start = rec.get("start_sec")
    start_text = "frame" if start is None else f"{float(start):.3f}"
    return f"{fallback:04d}_{ep}_{_safe_name(rec.get('rule'))}_{_safe_name(rec.get('field') or 'all')}_{start_text}"


def summarize_records(records):
    actionable = list(records or [])
    by_rule = {}
    by_field = {}
    by_ep = {}
    unconfirmed = 0
    for rec in actionable:
        by_rule[rec.get("rule") or "unknown"] = by_rule.get(rec.get("rule") or "unknown", 0) + 1
        by_field[rec.get("field") or "unknown"] = by_field.get(rec.get("field") or "unknown", 0) + 1
        ep = rec.get("episode")
        ep_key = "unknown" if ep is None else str(ep)
        by_ep[ep_key] = by_ep.get(ep_key, 0) + 1
        if rec.get("review_status", "未确认") == "未确认":
            unconfirmed += 1
    return {
        "total_issues": len(actionable),
        "episode_count": len(by_ep),
        "by_rule": dict(sorted(by_rule.items(), key=lambda x: (-x[1], x[0]))),
        "by_field": dict(sorted(by_field.items(), key=lambda x: (-x[1], x[0]))),
        "by_episode": dict(sorted(by_ep.items(), key=lambda x: (x[0] == "unknown", x[0]))),
        "unconfirmed": unconfirmed,
    }


def load_report_records(report_dir):
    try:
        rows = json.loads((Path(report_dir) / "issues.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(rows, list):
        return []
    review = {}
    try:
        review = json.loads((Path(report_dir) / "review_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        review = {}
    for i, rec in enumerate(rows, 1):
        issue_id = rec.get("issue_id") or _issue_id(rec, i)
        rec["issue_id"] = issue_id
        status_rec = review.get(issue_id) or review.get(str(i)) or {}
        if isinstance(status_rec, dict):
            rec["review_status"] = status_rec.get("status", rec.get("review_status", "未确认"))
            rec["review_note"] = status_rec.get("note", rec.get("review_note", ""))
    return rows


def save_review_status(report_dir, records):
    review = {}
    for i, rec in enumerate(records or [], 1):
        issue_id = rec.get("issue_id") or _issue_id(rec, i)
        review[issue_id] = {
            "episode": rec.get("episode"),
            "rule": rec.get("rule"),
            "field": rec.get("field"),
            "start_sec": rec.get("start_sec"),
            "end_sec": rec.get("end_sec"),
            "status": rec.get("review_status", "未确认"),
            "note": rec.get("review_note", ""),
        }
    (Path(report_dir) / "review_status.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def _clip_camera_name(issue):
    """Best camera/sensor name for report clip filenames."""
    path_text = str(issue.path or "")
    for field in STREAM_FIELDS:
        if field in path_text:
            return field
    if issue.field in STREAM_FIELDS:
        return issue.field
    return _safe_name(issue.field or "clip")


def write_report(dataset, issues, cfg=None):
    """Write an episode-grouped quality report and copy issue clips beside it."""
    cfg = _cfg(cfg)
    repo_id = (dataset or {}).get("dataset_name") or "unknown_dataset"
    leaf = _repo_leaf(repo_id)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_root = Path(cfg.get("report_dir") or ".quality_reports") / leaf / stamp
    actionable = [x for x in issues if x.severity in ("warn", "fail")]
    report_root.mkdir(parents=True, exist_ok=True)

    by_ep = {}
    all_records = []
    for issue in actionable:
        ep = issue.episode_index if issue.episode_index is not None else "unknown"
        by_ep.setdefault(ep, []).append(issue)

    index_lines = [
        f"# Quality report: {repo_id}",
        "",
        f"- generated_at: {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"- total_issues: {len(actionable)}",
        "",
    ]
    if not actionable:
        index_lines.append("No actionable issues found.")

    for ep in sorted(by_ep, key=lambda x: (x == "unknown", x)):
        ep_dir = report_root / (f"ep_{ep:06d}" if isinstance(ep, int) else "ep_unknown")
        ep_dir.mkdir(parents=True, exist_ok=True)
        index_lines.append(f"- [{ep_dir.name}]({ep_dir.name}/report.md): {len(by_ep[ep])} issues")
        lines = [f"# {repo_id} / {ep_dir.name}", ""]
        clip_counts = {}
        ep_records = []
        for i, issue in enumerate(by_ep[ep], 1):
            clip_names = []
            clip_links = {}
            source_clips = issue.clip_paths or {}
            if not source_clips and issue.clip_path:
                source_clips = {_clip_camera_name(issue): issue.clip_path}
            issue_global_index = len(all_records) + 1
            issue_id = _issue_id(_issue_record(issue), issue_global_index)
            start_text = "na" if issue.start_sec is None else f"{issue.start_sec:.3f}"
            end_text = "na" if issue.end_sec is None else f"{issue.end_sec:.3f}"
            for camera, clip_path in source_clips.items():
                if not clip_path or not Path(clip_path).is_file():
                    continue
                src = Path(clip_path)
                camera = _safe_name(camera)
                clip_counts[camera] = clip_counts.get(camera, 0) + 1
                suffix = "" if clip_counts[camera] == 1 else f"_{clip_counts[camera]:02d}"
                ep_text = f"ep{int(ep):06d}" if isinstance(ep, int) else "ep_unknown"
                clip_name = (
                    f"{ep_text}_{_safe_name(issue.rule)}_{camera}_"
                    f"{start_text}-{end_text}{suffix}{src.suffix or '.mp4'}"
                )
                shutil.copy2(src, ep_dir / clip_name)
                clip_names.append(clip_name)
                # Forward slashes: this relative path lands in <video src>,
                # where a Windows backslash breaks some browsers' URL parsing.
                clip_links[camera] = f"{ep_dir.name}/{clip_name}"
            plot_link = None
            if issue.plot_path and Path(issue.plot_path).is_file():
                plot_src = Path(issue.plot_path)
                shutil.copy2(plot_src, ep_dir / plot_src.name)
                plot_link = f"{ep_dir.name}/{plot_src.name}"
            rec = _issue_record(issue, clip_links, plot_link)
            rec["issue_id"] = issue_id
            ep_records.append(rec)
            all_records.append(rec)
            lines.extend([
                f"## {i}. {issue.rule}",
                "",
                f"- issue_id: {issue_id}",
                f"- severity: {issue.severity}",
                f"- field: {issue.field or '-'}",
                f"- time: {issue.start_sec:.3f}s-{issue.end_sec:.3f}s" if issue.start_sec is not None and issue.end_sec is not None else f"- frame: {issue.frame_index}",
                f"- detail: {issue.message}",
            ])
            for clip_name in clip_names:
                lines.append(f"- clip: {clip_name}")
            if plot_link:
                lines.append(f"- dof_plot: {Path(plot_link).name}")
            if issue.path:
                lines.append(f"- source: {issue.path}")
            lines.append("")
        (ep_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
        (ep_dir / "issues.json").write_text(
            json.dumps(ep_records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

    (report_root / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (report_root / "issues.json").write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    summary = summarize_records(all_records)
    (report_root / "overview.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    with (report_root / "issues.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "issue_id", "episode", "rule", "severity", "field", "start_sec", "end_sec",
                "frame", "message", "source", "clips", "plot", "review_status",
            ])
        writer.writeheader()
        for rec in all_records:
            row = dict(rec)
            row["clips"] = json.dumps(row["clips"], ensure_ascii=False)
            writer.writerow(row)
    save_review_status(report_root, all_records)
    html_rows = []
    for i, rec in enumerate(all_records, 1):
        clips = []
        for camera, rel in (rec.get("clips") or {}).items():
            rel_esc = html.escape(rel)
            camera_esc = html.escape(camera)
            clips.append(
                f'<div><a href="{rel_esc}">{camera_esc}</a><br>'
                f'<video src="{rel_esc}" controls width="260"></video></div>')
        if rec.get("plot"):
            plot_esc = html.escape(rec["plot"])
            clips.append(
                f'<div><a href="{plot_esc}">DOF curves</a><br>'
                f'<img src="{plot_esc}" width="380" loading="lazy"></div>')
        start_text = "" if rec["start_sec"] is None else f"{rec['start_sec']:.3f}"
        end_text = "" if rec["end_sec"] is None else f"{rec['end_sec']:.3f}"
        html_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{html.escape(str(rec['episode']))}</td>"
            f"<td>{html.escape(rec['rule'])}</td>"
            f"<td>{html.escape(rec['severity'])}</td>"
            f"<td>{html.escape(str(rec.get('field') or '-'))}</td>"
            f"<td>{start_text}</td>"
            f"<td>{end_text}</td>"
            f"<td>{html.escape(rec['message'])}</td>"
            f"<td>{', '.join(clips) or '-'}</td>"
            f"<td>{html.escape(rec['review_status'])}</td>"
            "</tr>")
    summary_html = f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>Quality report - {html.escape(repo_id)}</title>
<style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #ddd;padding:6px 8px;vertical-align:top}}
th{{background:#f5f5f5;text-align:left}}
.meta{{color:#666;margin-bottom:16px}}
</style>
<h1>{html.escape(repo_id)}</h1>
<div class="meta">generated_at: {_dt.datetime.now().isoformat(timespec='seconds')} | total_issues: {len(actionable)}</div>
<h2>Overview</h2>
<ul>
<li>problem episodes: {summary['episode_count']}</li>
<li>unconfirmed: {summary['unconfirmed']}</li>
<li>by rule: {html.escape(json.dumps(summary['by_rule'], ensure_ascii=False))}</li>
<li>by field: {html.escape(json.dumps(summary['by_field'], ensure_ascii=False))}</li>
</ul>
<p><a href="index.md">Markdown index</a> | <a href="issues.csv">CSV</a> | <a href="issues.json">JSON</a></p>
<table>
<thead><tr><th>#</th><th>ep</th><th>rule</th><th>severity</th><th>field</th><th>start(s)</th><th>end(s)</th><th>detail</th><th>clips</th><th>确认状态</th></tr></thead>
<tbody>{''.join(html_rows) or '<tr><td colspan="10">No actionable issues found.</td></tr>'}</tbody>
</table>
</html>
"""
    (report_root / "summary.html").write_text(summary_html, encoding="utf-8")
    return str(report_root.resolve())


def report_repo_id(report_dir):
    """Read the repo id from a generated quality report's index.md."""
    try:
        first = (Path(report_dir) / "index.md").read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    prefix = "# Quality report: "
    return first[len(prefix):].strip() if first.startswith(prefix) else None


def latest_reports(report_dir=".quality_reports"):
    """Return {repo_id: latest_report_dir} from generated report folders."""
    root = Path(report_dir)
    found = {}
    if not root.is_dir():
        return found
    for index in root.glob("*/*/index.md"):
        report = index.parent
        repo_id = report_repo_id(report)
        if not repo_id:
            continue
        prev = found.get(repo_id)
        if prev is None or report.stat().st_mtime > Path(prev).stat().st_mtime:
            found[repo_id] = str(report.resolve())
    return found


def load_quality_status(path="quality_status.local.json", report_dir=".quality_reports"):
    """Load persisted GUI quality status and repair it from existing reports."""
    data = {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = raw
    except (OSError, ValueError):
        data = {}

    reports = latest_reports(report_dir)
    for repo_id, report in reports.items():
        data[repo_id] = {
            "status": "已检查",
            "report_dir": report,
            "checked_at": data.get(repo_id, {}).get("checked_at"),
        }

    clean = {}
    for repo_id, rec in data.items():
        if not isinstance(rec, dict):
            continue
        report = rec.get("report_dir")
        if rec.get("status") == "已检查" and report and Path(report).is_dir():
            clean[repo_id] = rec
    return clean


def save_quality_status(data, path="quality_status.local.json"):
    Path(path).write_text(json.dumps(data or {}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def scan_dataset_with_report(dataset, out_dir="datasets/TacVerse", cfg=None, progress=None, cancel=None):
    issues = scan_dataset(dataset, out_dir=out_dir, cfg=cfg, progress=progress, cancel=cancel)
    _check_cancel(cancel)
    _emit(progress, "写入 Markdown/HTML/CSV/JSON 报告...", 97)
    report_dir = write_report(dataset, issues, cfg=cfg)
    _emit(progress, "深度检查完成。", 100)
    return issues, report_dir
