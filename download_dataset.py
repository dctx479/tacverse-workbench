#!/usr/bin/env python3
"""Pull one or more Hugging Face dataset repos into an organization folder.

Datasets are re-pulled on every run (`snapshot_download` syncs incrementally,
so newly merged files are fetched and unchanged ones are skipped). Each run
stores each dataset directly under the configured organization folder:

    <out-dir>/<dataset-name>/...                       # dataset files
    <out-dir>/pull_result_<YYMMDD>_<HHMMSS>.json      # aggregate summary

The list of datasets lives in DATASETS and can be overridden with repeated
`--repo-id` flags. The fields lifted from each dataset's meta/info.json are
declared in INFO_FIELDS, so extending the report is a one-line change.
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

# By default every dataset under this org is discovered and pulled. Override
# with --org, or pass explicit --repo-id flags to pull a specific subset.
ORG = "TacVerse"

HF_DATASET_URL = "https://huggingface.co/datasets/{repo_id}"

# Fields copied verbatim from meta/info.json into each dataset's summary.
# Extend this list to surface more of info.json (e.g. "fps", "total_tasks",
# "robot_type") with no other change. `key` is the output name, `source` the
# info.json key; `required=False` skips the field for datasets that lack it.
INFO_FIELDS = [
    {"key": "total_episodes", "source": "total_episodes"},
    {"key": "total_frames", "source": "total_frames"},
    {"key": "fps", "source": "fps", "required": False},
    {"key": "robot_type", "source": "robot_type", "required": False},
    {"key": "total_tasks", "source": "total_tasks", "required": False},
]

# Assumed capture rate (frames per second) when a dataset's info.json omits fps.
DEFAULT_FPS = 30


def normalize_proxy_env() -> None:
    """Make the shell proxy vars parseable by httpx (huggingface_hub 1.x).

    httpx rejects a schemeless `socks://` proxy URL. The http(s)_proxy vars
    already cover HTTPS traffic to the Hub, so drop the offending ALL_PROXY
    vars and normalize any remaining socks:// value to socks5://.
    """
    for var in ("ALL_PROXY", "all_proxy"):
        os.environ.pop(var, None)
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        val = os.environ.get(var)
        if val and val.startswith("socks://"):
            os.environ[var] = "socks5://" + val[len("socks://"):]


def _apply_info(summary: dict, info: dict) -> dict:
    """Fill INFO_FIELDS + derived duration_hours into `summary` from an info dict."""
    for field in INFO_FIELDS:
        src = field["source"]
        if src in info:
            summary[field["key"]] = info[src]
        elif field.get("required", True):
            summary[field["key"]] = None
    # Recording duration in hours = frames / fps / 3600.
    frames = summary.get("total_frames")
    if frames is not None:
        fps = summary.get("fps") or DEFAULT_FPS
        summary["duration_hours"] = round(frames / fps / 3600, 3)
    return summary


def build_summary(repo_id: str, local_dir: str) -> dict:
    """Assemble a per-dataset summary from a *downloaded* dataset directory.

    Derived fields (name, link, local_dir) plus every entry in INFO_FIELDS read
    from meta/info.json. Missing info.json or missing keys degrade gracefully.
    """
    summary = {
        "dataset_name": repo_id,
        "link": HF_DATASET_URL.format(repo_id=repo_id),
        "local_dir": str(local_dir),
    }
    info_path = Path(local_dir) / "meta" / "info.json"
    if info_path.is_file():
        _apply_info(summary, json.loads(info_path.read_text()))
    else:
        # Not a LeRobot-style dataset (no meta/info.json); leave the
        # info-derived fields absent rather than guessing.
        print(f"Note: {info_path} not found; summary limited to name and link.")
    tasks_path = Path(local_dir) / "meta" / "tasks.parquet"
    if tasks_path.is_file():
        import tasks_reader

        rows, _ = tasks_reader.load(tasks_path)
        summary["tasks"] = rows
    return summary


def fetch_tasks(repo_id: str, token=None) -> list:
    """Fetch a dataset's task instructions from meta/tasks.parquet.

    tasks.parquet is a tiny file (a few KB) carrying the natural-language task
    string(s) the dataset was recorded against — the base "prompt". Fetched on
    the stats-only path so the dashboard can show prompts without a full pull.
    Returns [{"index", "task"}] (sorted), or [] if absent/unreadable.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="meta/tasks.parquet",
            repo_type="dataset",
            token=token,
        )
    except Exception:
        return []  # dataset has no tasks.parquet (or no access)
    import tasks_reader

    rows, _ = tasks_reader.load(path)
    return rows


def fetch_summary(repo_id: str, token=None) -> dict:
    """Summarize a dataset by fetching only small meta files (no full download).

    Used for the stats-only path: downloads meta/info.json (+ meta/tasks.parquet
    for the task prompt) instead of the whole (potentially huge) dataset. Falls
    back to name+link if info.json is absent.
    """
    from huggingface_hub import hf_hub_download

    summary = {
        "dataset_name": repo_id,
        "link": HF_DATASET_URL.format(repo_id=repo_id),
    }
    try:
        info_path = hf_hub_download(
            repo_id=repo_id,
            filename="meta/info.json",
            repo_type="dataset",
            token=token,
        )
    except Exception:
        return summary  # no info.json -> name+link only
    _apply_info(summary, json.loads(Path(info_path).read_text()))
    summary["tasks"] = fetch_tasks(repo_id, token)
    return summary


def discover_datasets_meta(org, token):
    """Return [{"id", "last_modified"}] for every dataset under an org/user.

    Ordered most-recently-updated first, matching the Hugging Face org page's
    default "Recently updated" sort. Datasets missing a timestamp sort last.
    """
    from huggingface_hub import list_datasets

    # Ask the Hub for its own "Recently updated" ranking when available; older
    # huggingface_hub versions (e.g. 1.23.x) do not expose a `direction` kwarg,
    # so we sort client-side as the source of truth and also pin timestamp-less
    # repos last.
    ds = list(list_datasets(author=org, token=token, sort="lastModified"))
    ds.sort(key=lambda d: (d.last_modified is not None, d.last_modified), reverse=True)
    out = []
    for d in ds:
        lm = d.last_modified
        out.append({"id": d.id, "last_modified": lm.isoformat() if lm else None})
    return out


def discover_datasets(org, token):
    """Return every dataset repo id under an org/user (recently-updated first)."""
    return [d["id"] for d in discover_datasets_meta(org, token)]


def fetch_uploader(repo_id, token=None):
    """Return uploader info from the dataset's HF commit history.

    `uploader` is the author of the earliest (initial) commit — i.e. who created
    the dataset. `uploaders` lists every distinct commit author. Degrades to an
    empty dict on any error (private repo, network, etc.).
    """
    from huggingface_hub import HfApi

    try:
        commits = HfApi().list_repo_commits(repo_id, repo_type="dataset", token=token)
    except Exception:
        return {}
    if not commits:
        return {}
    # Commits come newest-first; the last one is the initial commit.
    authors, seen = [], set()
    for c in commits:
        for a in (getattr(c, "authors", None) or []):
            if a not in seen:
                seen.add(a)
                authors.append(a)
    initial = commits[-1]
    creator = (getattr(initial, "authors", None) or [None])[0]
    last_at = getattr(commits[0], "created_at", None)
    return {
        "uploader": creator,
        "uploaders": authors,
        "last_commit_at": last_at.isoformat() if last_at else None,
    }


def pull_dataset(repo_id, dataset_dir, revision, token):
    """Download one dataset into <dataset_dir>/<dataset-name> and summarize it."""
    from huggingface_hub import snapshot_download

    local_dir = Path(dataset_dir) / repo_id.split("/")[-1]
    print(f"Downloading {repo_id} -> {local_dir}")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        token=token,
    )
    return build_summary(repo_id, path)


def build_report(summaries, failures, now, org, requested):
    """Build the aggregate report dict (totals first, then per-dataset list)."""
    agg_episodes = sum(x.get("total_episodes") or 0 for x in summaries)
    agg_frames = sum(x.get("total_frames") or 0 for x in summaries)
    agg_hours = round(sum(x.get("duration_hours") or 0 for x in summaries), 3)
    report = {
        "total_datasets": len(summaries),
        "total_episodes": agg_episodes,
        "total_frames": agg_frames,
        "total_hours": agg_hours,
        "pulled_at": now.isoformat(timespec="seconds"),
        "date": now.strftime("%y%m%d"),
        "org": org,
        "requested": requested,
        "count": len(summaries),
        "datasets": summaries,
    }
    if failures:
        report["failures"] = failures
    return report


def run_pull(repo_ids, out_dir, org, revision=None, token=None, now=None,
             log=print, progress=None, write_summary=True,
             meta_map=None, with_uploader=True):
    """Pull every repo in `repo_ids` into one organization folder and write a report.

    `log(msg)` receives human-readable progress lines (same text as the CLI).
    `progress(done, total)` is called before and after each dataset so a UI can
    drive a progress bar. Returns (report_dict, out_path_or_None).
    """
    now = now or dt.datetime.now()
    dataset_dir = Path(out_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    summaries, failures = [], []
    total = len(repo_ids)
    if progress:
        progress(0, total)
    for i, repo_id in enumerate(repo_ids, 1):
        log(f"[{i}/{total}] {repo_id}")
        try:
            s = pull_dataset(repo_id, dataset_dir, revision, token)
            _enrich(s, repo_id, meta_map, with_uploader, token)
            summaries.append(s)
        except Exception as exc:  # keep pulling the rest if one fails
            log(f"ERROR pulling {repo_id}: {exc}")
            failures.append({"dataset_name": repo_id, "error": str(exc)})
        if progress:
            progress(i, total)

    report = build_report(summaries, failures, now, org, total)
    out_path = None
    if write_summary:
        out_path = dataset_dir / f"pull_result_{now.strftime('%y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        log(f"Wrote summary -> {out_path}")
    try:
        append_pull(report)  # git-committed change-log; survives datasets/ being ignored
        log(f"Updated history -> {DATASET_LOG_FILE}")
    except OSError as exc:
        log(f"WARN: could not update {DATASET_LOG_FILE}: {exc}")
    return report, out_path


def _enrich(summary, repo_id, meta_map, with_uploader, token):
    """Attach last_modified (from meta_map) and uploader fields to a summary."""
    if meta_map and repo_id in meta_map:
        summary["last_modified"] = meta_map[repo_id]
    if with_uploader:
        summary.update(fetch_uploader(repo_id, token))
    return summary


def collect_stats(repo_ids, org, token=None, now=None, log=print, progress=None,
                  meta_map=None, with_uploader=True):
    """Build a report from meta/info.json only — no dataset files downloaded.

    Same report shape as run_pull (totals + per-dataset list), so a UI can show
    it in the exact same dashboard. Per-dataset entries have no local_dir but do
    carry last_modified + uploader when meta_map/with_uploader are supplied.
    """
    now = now or dt.datetime.now()
    summaries, failures = [], []
    total = len(repo_ids)
    if progress:
        progress(0, total)
    for i, repo_id in enumerate(repo_ids, 1):
        log(f"[{i}/{total}] {repo_id}")
        try:
            s = fetch_summary(repo_id, token)
            _enrich(s, repo_id, meta_map, with_uploader, token)
            summaries.append(s)
        except Exception as exc:
            log(f"ERROR reading {repo_id}: {exc}")
            failures.append({"dataset_name": repo_id, "error": str(exc)})
        if progress:
            progress(i, total)
    return build_report(summaries, failures, now, org, total)


def find_latest_report(out_dir):
    """Return the newest pull_result_*.json directly under an org directory."""
    files = sorted(Path(out_dir).glob("pull_result_*.json"))
    if not files:
        files = sorted(Path(out_dir).glob("*/pull_result_*.json"))
    return files[-1] if files else None


# --------------------------------------------------------------------------- #
# Analytics helpers (pure functions over report dicts — used by the GUI)
# --------------------------------------------------------------------------- #
# Two git-committed json files at the repo root, both travelling with the code so
# a fresh clone gets the collection trend / 每日新增 WITHOUT syncing the multi-GB
# datasets/ folder (which is .gitignore'd):
#
#   config.json       — hand-edited only:
#                        { "checks": {...}, "uploader_names": {"<hf_id>": "<中文名>"} }
#   dataset_log.json  — auto-appended per-DATASET change log (no time-major dupes):
#     { "dataset_index": [ "<name>", ... ],                     # names stored once
#       "daily_totals": [ {pulled_at,date,org,total_*,present:[idx],
#                            source?:"manual"} ],  # 1 row/pull/manual snapshot
#       "datasets": { "<name>": { <meta>, "changes": [ {date,pulled_at,
#                                total_*,d_*} ] } } }  # a row ONLY when totals moved
CONFIG_FILE = str(Path(__file__).parent / "config.json")
DATASET_LOG_FILE = str(Path(__file__).parent / "dataset_log.json")

# Per-dataset fields exposed to the GUI in a reconstructed snapshot row.
_HISTORY_DS_FIELDS = (
    "dataset_name", "total_episodes", "total_frames", "duration_hours",
    "fps", "robot_type", "total_tasks", "uploader", "last_modified",
)
# Slow-changing metadata stored once per dataset (latest value wins).
_DS_META_FIELDS = ("fps", "robot_type", "total_tasks", "uploader", "last_modified")
# Cumulative totals whose movement triggers a new `changes` entry.
_DS_TOTAL_FIELDS = ("total_episodes", "total_frames", "duration_hours")


def load_config(path=CONFIG_FILE):
    """Read the unified config; returns {} (never raises) if missing/corrupt."""
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError):
        return {}


def load_uploader_names(path=CONFIG_FILE):
    """The hand-edited HF id -> Chinese name map from the config file."""
    return load_config(path).get("uploader_names", {}) or {}


# --------------------------------------------------------------------------- #
# Dataset change-log (dataset_log.json) — read / append / reconstruct
# --------------------------------------------------------------------------- #
def load_dataset_log(path=DATASET_LOG_FILE):
    """Read the dataset change-log; returns an empty skeleton on missing/corrupt."""
    try:
        log = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log = None
    if not isinstance(log, dict):
        log = {}
    log.setdefault("dataset_index", [])
    log.setdefault("daily_totals", [])
    log.setdefault("datasets", {})
    return log


def _dataset_log_text(log, present_per_line=16):
    """Serialize the change-log in a compact but still reviewable layout.

    Normal ``json.dumps(..., indent=2)`` places every integer on its own line,
    and expands every small dataset/change object across many lines.  Keep the
    top-level collections readable, group ``present`` indices, render each
    change object on one line, and render each dataset's metadata on one line.
    The result remains plain JSON and round-trips through any standard parser.
    """
    def compact(value):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))

    def append_regular_property(lines, key, value, trailing_comma):
        dumped = json.dumps(value, indent=2, ensure_ascii=False).splitlines()
        key_text = json.dumps(key, ensure_ascii=False)
        if len(dumped) == 1:
            lines.append(f"  {key_text}: {dumped[0]}" + ("," if trailing_comma else ""))
            return
        lines.append(f"  {key_text}: {dumped[0]}")
        lines.extend("  " + line for line in dumped[1:-1])
        lines.append("  " + dumped[-1] + ("," if trailing_comma else ""))

    def append_datasets(lines, datasets, trailing_comma):
        lines.append('  "datasets": {')
        items = list(datasets.items())
        for dataset_i, (name, section) in enumerate(items):
            dataset_comma = dataset_i < len(items) - 1
            lines.append(f"    {json.dumps(name, ensure_ascii=False)}: {{")
            changes = section.get("changes", []) or []
            metadata = [(key, value) for key, value in section.items()
                        if key != "changes"]
            if changes:
                lines.append('      "changes": [')
                for change_i, change in enumerate(changes):
                    if change_i < len(changes) - 1:
                        suffix = ","
                    else:
                        suffix = "]" + ("," if metadata else "")
                    lines.append("        " + compact(change) + suffix)
            else:
                lines.append('      "changes": []' + ("," if metadata else ""))

            if metadata:
                meta_text = ", ".join(
                    f"{json.dumps(key, ensure_ascii=False)}: {compact(value)}"
                    for key, value in metadata)
                lines.append("      " + meta_text + "}" + ("," if dataset_comma else ""))
            else:
                lines.append("    }" + ("," if dataset_comma else ""))
        lines.append("  }" + ("," if trailing_comma else ""))

    lines = ["{"]
    properties = list(log.items())
    for prop_i, (key, value) in enumerate(properties):
        trailing_comma = prop_i < len(properties) - 1
        if key == "datasets" and isinstance(value, dict):
            append_datasets(lines, value, trailing_comma)
        else:
            append_regular_property(lines, key, value, trailing_comma)
    lines.append("}")

    # Compact only the already-rendered multi-line present arrays; empty arrays
    # stay as ``[]`` and all other arrays retain their dedicated layouts above.
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() != '"present": [':
            out.append(line)
            i += 1
            continue

        values = []
        j = i + 1
        while j < len(lines) and lines[j].strip() not in ("]", "],"):
            values.append(lines[j].strip().rstrip(","))
            j += 1
        if not values or j >= len(lines):
            out.append(line)
            i += 1
            continue

        out.append(line)
        value_indent = line[:len(line) - len(line.lstrip())] + "  "
        for start in range(0, len(values), present_per_line):
            chunk = values[start:start + present_per_line]
            suffix = "," if start + present_per_line < len(values) else ""
            out.append(value_indent + ", ".join(chunk) + suffix)
        out.append(lines[j])
        i = j + 1
    return "\n".join(out) + "\n"


def write_dataset_log(log, path=DATASET_LOG_FILE):
    """Write ``log`` using the repository's readable compact JSON format."""
    Path(path).write_text(_dataset_log_text(log), encoding="utf-8")
    return path


def _fold_report_into_log(log, report):
    """Merge one report into `log` in place. Appends the per-pull aggregate row
    and, per dataset, a change entry ONLY when its totals moved — recording both
    the resulting totals and the +delta vs the previous entry.

    Dataset names are stored once in the shared `dataset_index`; each daily_totals
    row lists the datasets present at that pull as integer indices into it
    (`present`), so names aren't re-listed in full on every pull. Reports must be
    folded oldest-first so deltas chain correctly."""
    at = report.get("pulled_at")
    date = report.get("date")
    org = report.get("org")
    totals = log.setdefault("daily_totals", [])

    # One authoritative snapshot per organisation and calendar day.  Ignore an
    # out-of-order older report instead of allowing it to replace newer data.
    same_day = [t for t in totals
                if t.get("date") == date and t.get("org") == org]
    if same_day and max((t.get("pulled_at") or "") for t in same_day) > (at or ""):
        return log

    # Remove the earlier aggregate row and its per-dataset changes before
    # calculating this pull's deltas.  The remaining previous change is then
    # from an earlier day, so d_* represents the whole day's growth.
    totals[:] = [t for t in totals
                 if not (t.get("date") == date and t.get("org") == org)]
    report_names = {d.get("dataset_name") for d in report.get("datasets", [])
                    if d.get("dataset_name")}
    prefix = f"{org}/" if org else None
    for name, sect in log.setdefault("datasets", {}).items():
        if name not in report_names and not (prefix and name.startswith(prefix)):
            continue
        changes = sect.setdefault("changes", [])
        changes[:] = [c for c in changes if c.get("date") != date]

    index = log.setdefault("dataset_index", [])
    pos = {n: i for i, n in enumerate(index)}
    dsets = log.setdefault("datasets", {})

    present = []
    for d in report.get("datasets", []):
        name = d.get("dataset_name")
        if not name:
            continue
        if name not in pos:                    # register the name once
            pos[name] = len(index)
            index.append(name)
        present.append(pos[name])              # present even if unchanged below
        sect = dsets.setdefault(name, {"changes": []})
        for k in _DS_META_FIELDS:              # keep newest slow-changing metadata
            sect[k] = d.get(k)
        changes = sect.setdefault("changes", [])
        changes[:] = [c for c in changes if c.get("pulled_at") != at]  # idempotent
        cur = {k: d.get(k) for k in _DS_TOTAL_FIELDS}
        prev = changes[-1] if changes else None
        if prev and all(prev.get(k) == cur.get(k) for k in _DS_TOTAL_FIELDS):
            continue                            # unchanged -> write no change row
        base = prev or {}
        changes.append({
            "date": report.get("date"), "pulled_at": at, **cur,
            "d_episodes": (cur.get("total_episodes") or 0) - (base.get("total_episodes") or 0),
            "d_frames": (cur.get("total_frames") or 0) - (base.get("total_frames") or 0),
            "d_hours": round((cur.get("duration_hours") or 0) - (base.get("duration_hours") or 0), 3),
        })

    row = {
        "pulled_at": at,
        "date": report.get("date"),
        "org": report.get("org"),
        "total_datasets": report.get("total_datasets"),
        "total_episodes": report.get("total_episodes"),
        "total_frames": report.get("total_frames"),
        "total_hours": report.get("total_hours"),
        "present": sorted(present),
    }
    if report.get("source"):
        row["source"] = report["source"]
    totals.append(row)
    totals.sort(key=lambda t: t.get("pulled_at") or "")
    return log


def append_pull(report, path=DATASET_LOG_FILE):
    """Fold `report` into the git-committed dataset change-log at `path`.

    Unchanged datasets are NOT re-stored on every pull (unlike the old
    pull_history), so the file stays small. Re-reads first; safe to re-run.
    Returns the path written.
    """
    log = load_dataset_log(path)
    # A real stats/pull on the same day supersedes an aggregate-only manual
    # placeholder.  This prevents the synthetic 23:59:59 manual timestamp from
    # continuing to win after fresh detailed data becomes available.
    totals = log.setdefault("daily_totals", [])
    totals[:] = [t for t in totals if not (
        t.get("source") == "manual"
        and t.get("date") == report.get("date")
        and t.get("org") == report.get("org")
    )]
    _fold_report_into_log(log, report)
    return write_dataset_log(log, path)


def upsert_manual_totals(date, org, total_datasets, total_episodes,
                         total_frames, total_hours, path=DATASET_LOG_FILE,
                         today=None):
    """Insert or replace one aggregate-only manual snapshot.

    ``date`` uses the same YYMMDD format as regular reports.  The synthetic
    end-of-day timestamp makes the manual snapshot authoritative for that day;
    :func:`append_pull` removes it if real detailed statistics are later pulled
    for the same organisation and date.
    """
    try:
        day = dt.datetime.strptime(date, "%y%m%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError("date must use YYMMDD format") from exc
    today = today or dt.date.today()
    if day > today:
        raise ValueError("manual snapshot date cannot be in the future")
    if not isinstance(org, str) or not org.strip():
        raise ValueError("org cannot be empty")

    integer_fields = {
        "total_datasets": total_datasets,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
    }
    normalized = {}
    for key, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
        normalized[key] = value
    if isinstance(total_hours, bool) or not isinstance(total_hours, (int, float)) \
            or total_hours < 0:
        raise ValueError("total_hours must be a non-negative number")

    report = {
        "pulled_at": day.strftime("%Y-%m-%dT23:59:59"),
        "date": date,
        "org": org.strip(),
        **normalized,
        "total_hours": round(float(total_hours), 3),
        "datasets": [],
        "source": "manual",
    }
    log = load_dataset_log(path)
    totals = log.setdefault("daily_totals", [])
    totals[:] = [t for t in totals if not (
        t.get("source") == "manual"
        and t.get("date") == date
        and t.get("org") == report["org"]
    )]
    _fold_report_into_log(log, report)
    write_dataset_log(log, path)
    return report


def _state_as_of(log, pulled_at, names):
    """Per-dataset snapshot rows for `names` as they stood at `pulled_at`.

    For each name, take the last change entry with pulled_at <= the target and
    merge in the dataset's metadata, rebuilding a row shaped like the old
    pull_history datasets[] entries (_HISTORY_DS_FIELDS)."""
    dsets = log.get("datasets", {})
    target = pulled_at or ""
    rows = []
    for name in names:
        sect = dsets.get(name)
        if not sect:
            continue
        latest = None
        for c in sect.get("changes", []):
            if (c.get("pulled_at") or "") <= target:
                latest = c
        if latest is None:
            continue
        row = {"dataset_name": name}
        for k in _DS_TOTAL_FIELDS:
            row[k] = latest.get(k)
        for k in _DS_META_FIELDS:
            row[k] = sect.get(k)
        rows.append(row)
    return rows


def _reconstruct_history(log):
    """Rebuild the oldest-first list of full pull snapshots from the change-log,
    equivalent to the old config['pull_history'] shape so downstream analytics
    (daily_series / find_baseline / compute_deltas) need no changes."""
    index = log.get("dataset_index", [])
    snaps = []
    for row in sorted(log.get("daily_totals", []),
                      key=lambda t: t.get("pulled_at") or ""):
        names = [index[i] for i in row.get("present", []) if 0 <= i < len(index)]
        snap = {k: row.get(k) for k in
                ("pulled_at", "date", "org", "total_datasets",
                 "total_episodes", "total_frames", "total_hours")}
        if row.get("source"):
            snap["source"] = row["source"]
        snap["datasets"] = _state_as_of(log, row.get("pulled_at"), names)
        snaps.append(snap)
    return snaps


def compact_dataset_log(path=DATASET_LOG_FILE):
    """Rewrite an existing log with only its latest rows per calendar day.

    The cleanup is deliberately in-place: dataset metadata and orphaned index
    entries are retained, while aggregate rows and each dataset's change rows
    are collapsed independently.  Deltas are then recomputed against the prior
    retained day.
    """
    log = load_dataset_log(path)
    before = len(log.get("daily_totals", []))
    latest = {}
    for row in log.get("daily_totals", []):
        key = (row.get("date") or row.get("pulled_at"), row.get("org"))
        previous = latest.get(key)
        if previous is None or (row.get("pulled_at") or "") \
                >= (previous.get("pulled_at") or ""):
            latest[key] = row
    log["daily_totals"] = sorted(
        latest.values(), key=lambda r: r.get("pulled_at") or "")

    for sect in log.get("datasets", {}).values():
        by_day = {}
        for change in sect.get("changes", []):
            key = change.get("date") or change.get("pulled_at")
            previous = by_day.get(key)
            if previous is None or (change.get("pulled_at") or "") \
                    >= (previous.get("pulled_at") or ""):
                by_day[key] = change
        changes = sorted(by_day.values(), key=lambda c: c.get("pulled_at") or "")
        previous = None
        for change in changes:
            base = previous or {}
            change["d_episodes"] = ((change.get("total_episodes") or 0)
                                     - (base.get("total_episodes") or 0))
            change["d_frames"] = ((change.get("total_frames") or 0)
                                   - (base.get("total_frames") or 0))
            change["d_hours"] = round(
                (change.get("duration_hours") or 0)
                - (base.get("duration_hours") or 0), 3)
            previous = change
        sect["changes"] = changes

    write_dataset_log(log, path)
    return before, len(log["daily_totals"])


def load_history(out_dir, log_file=DATASET_LOG_FILE):
    """Load pull snapshots oldest-first for trends / deltas.

    Reconstructs the committed snapshots from the dataset change-log and merges
    them with any local pull_result_*.json still on disk, deduping by
    pulled_at so both sources contribute but neither double-counts.
    """
    by_at = {}
    for r in _reconstruct_history(load_dataset_log(log_file)):
        key = (r.get("pulled_at") or id(r), r.get("org"))
        by_at[key] = r
    files = sorted(Path(out_dir).glob("pull_result_*.json"))
    if not files:
        files = sorted(Path(out_dir).glob("*/pull_result_*.json"))
    for f in files:
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        key = (r.get("pulled_at") or str(f), r.get("org"))
        by_at.setdefault(key, r)  # log wins on ties
    history = list(by_at.values())
    history.sort(key=lambda r: r.get("pulled_at", ""))
    return history


def load_latest_local_report(out_dir, org=ORG):
    """Return the newest locally available report without network access.

    Priority: explicit pull_result JSON, then the committed dataset log, then a
    best-effort scan of downloaded <dataset>/meta/info.json directories.
    """
    latest = find_latest_report(out_dir)
    if latest:
        data = _load_json(latest)
        if isinstance(data, dict) and data.get("datasets"):
            return data, str(latest)

    history = load_history(out_dir)
    if history:
        report = history[-1]
        if isinstance(report, dict) and report.get("datasets"):
            return report, DATASET_LOG_FILE

    summaries = []
    for info in sorted(Path(out_dir).glob("*/meta/info.json")):
        dataset_dir = info.parent.parent
        summaries.append(build_summary(f"{org}/{dataset_dir.name}", str(dataset_dir)))
    if summaries:
        latest_time = max((Path(s["local_dir"]).stat().st_mtime for s in summaries), default=None)
        now = dt.datetime.fromtimestamp(latest_time) if latest_time else dt.datetime.now()
        return build_report(summaries, [], now, org, len(summaries)), str(Path(out_dir))
    return None, None


def migrate_pull_history_to_log(config_path=CONFIG_FILE, log_path=DATASET_LOG_FILE):
    """One-time: convert config['pull_history'] into dataset_log.json, then drop
    the pull_history key from config (keeping checks + uploader_names). Rebuilds
    the log from scratch out of config's history, so it is safe to re-run."""
    cfg = load_config(config_path)
    hist = cfg.get("pull_history", []) or []
    log = {"dataset_index": [], "daily_totals": [], "datasets": {}}
    for snap in sorted(hist, key=lambda r: r.get("pulled_at") or ""):
        _fold_report_into_log(log, snap)
    write_dataset_log(log, log_path)
    cfg.pop("pull_history", None)
    Path(config_path).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    return log_path


def daily_series(history):
    """Collapse history to one snapshot per day (the day's last pull).

    `total_hours` is the absolute library total at that snapshot (already
    cumulative). `new_hours` is the day-over-day increase (this day's total
    minus the previous pulled day's total); the first day's `new_hours` equals
    its total. Returns a date-sorted list of {date, total_hours, new_hours,
    total_episodes, total_frames, total_datasets} for trend charts.
    """
    by_day = {}
    for r in history:  # history is oldest-first, so later pulls overwrite
        by_day[r.get("date", "")] = r
    series = []
    prev_total = None
    for date in sorted(k for k in by_day if k):
        r = by_day[date]
        total = r.get("total_hours", 0) or 0
        new_hours = total if prev_total is None else round(total - prev_total, 3)
        prev_total = total
        series.append({
            "date": date,
            "total_hours": total,
            "new_hours": new_hours,
            "total_episodes": r.get("total_episodes", 0) or 0,
            "total_frames": r.get("total_frames", 0) or 0,
            "total_datasets": r.get("total_datasets", 0) or 0,
        })
    return series


def daily_group_series(history, key_fn):
    """Per-group daily positive growth from the last snapshot of each day.

    Returns rows sorted by date oldest-first and hours descending within each day:
    {date, group, hours, episodes, datasets}. The first detailed day counts
    each dataset's full duration as that day's contribution. If the previous day
    has only aggregate totals and no dataset details, attribution for the next
    day is skipped because per-group growth cannot be derived safely.
    """
    by_day = {}
    for r in history:
        by_day[r.get("date", "")] = r
    rows = []
    prev_report = None
    prev = {}
    for date in sorted(k for k in by_day if k):
        report = by_day[date]
        datasets = report.get("datasets", []) or []
        aggregate_only_prior = bool(prev_report) and not prev
        if not aggregate_only_prior:
            groups = {}
            for dataset in datasets:
                name = dataset.get("dataset_name")
                if not name:
                    continue
                prior = prev.get(name)
                d_hours = round((dataset.get("duration_hours") or 0)
                                - (prior.get("duration_hours") or 0 if prior else 0), 3)
                d_episodes = (dataset.get("total_episodes") or 0) \
                    - (prior.get("total_episodes") or 0 if prior else 0)
                hours = max(0, d_hours)
                episodes = max(0, d_episodes)
                if hours <= 0 and episodes <= 0:
                    continue
                key = key_fn(dataset) or "—"
                group = groups.setdefault(
                    key, {"date": date, "group": key, "hours": 0.0,
                          "episodes": 0, "datasets": 0})
                group["hours"] += hours
                group["episodes"] += episodes
                group["datasets"] += 1
            day_rows = sorted(groups.values(), key=lambda g: g["hours"], reverse=True)
            for row in day_rows:
                row["hours"] = round(row["hours"], 3)
            rows.extend(day_rows)
        prev_report = report
        prev = {d.get("dataset_name"): d for d in datasets if d.get("dataset_name")}
    return rows


def daily_uploader_series(history):
    """Backward-compatible per-uploader daily growth helper."""
    return daily_group_series(history, lambda d: d.get("uploader") or "")


def find_baseline(current_report, history):
    """Return the snapshot to diff `current_report` against: the last pull of the
    most recent *earlier day*.

    "今日新增" is measured against the previous pull DAY, not merely the previous
    pull — so multiple pulls on the same day all compare back to that earlier day.
    `history` is oldest-first, so the last entry whose date precedes the current
    report's date is that day's final pull. Returns None when no earlier day
    exists (the current report is the first ever).
    """
    cur_date = current_report.get("date") or ""
    prior = None
    for r in history:  # oldest-first; last match = newest earlier-day pull
        rd = r.get("date") or ""
        if rd and rd < cur_date:
            prior = r
    return prior


def aggregate_deltas(current_report, history):
    """Return aggregate (new_hours, new_episodes) vs the prior recorded day."""
    base = find_baseline(current_report, history)
    base_hours = (base.get("total_hours") or 0) if base else 0
    base_eps = (base.get("total_episodes") or 0) if base else 0
    new_hours = round((current_report.get("total_hours") or 0) - base_hours, 2)
    new_episodes = (current_report.get("total_episodes") or 0) - base_eps
    return new_hours, new_episodes


def compute_deltas(current_report, history):
    """Per-dataset growth of current_report vs the previous pull day's snapshot.

    Baseline = find_baseline(current_report, history). Returns
    {dataset_name: {d_episodes, d_frames, d_hours, is_new}}. With no earlier-day
    snapshot every dataset is marked is_new with its full totals as the delta.
    """
    prior = find_baseline(current_report, history)
    prev = {d["dataset_name"]: d for d in (prior.get("datasets", []) if prior else [])}
    # An aggregate-only baseline (backfilled history that has totals but no
    # per-dataset detail) can't attribute growth to individual datasets. Report
    # zero per-dataset deltas there instead of pretending everything is new; the
    # KPI falls back to the aggregate total difference (see main_app._new_totals).
    aggregate_only = bool(prior) and not prev
    deltas = {}
    for d in current_report.get("datasets", []):
        name = d["dataset_name"]
        if aggregate_only:
            deltas[name] = {"d_episodes": 0, "d_frames": 0, "d_hours": 0,
                            "is_new": False}
            continue
        p = prev.get(name)
        deltas[name] = {
            "d_episodes": (d.get("total_episodes") or 0) - (p.get("total_episodes") or 0 if p else 0),
            "d_frames": (d.get("total_frames") or 0) - (p.get("total_frames") or 0 if p else 0),
            "d_hours": round((d.get("duration_hours") or 0) - (p.get("duration_hours") or 0 if p else 0), 3),
            "is_new": p is None,
        }
    return deltas


def task_prefix(dataset_name):
    """Derive a task label: drop the owner and a trailing -MMDD date suffix.

    e.g. 'TacVerse/taccap-g1-pepper-0703' -> 'taccap-g1-pepper'.
    """
    name = dataset_name.split("/")[-1]
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        return parts[0]
    return name


def rollup(datasets, key_fn):
    """Group datasets by key_fn(d) and sum count/episodes/frames/hours.

    Returns a list of {group, count, episodes, frames, hours, pct_hours} sorted
    by hours descending. pct_hours is each group's share of total hours.
    """
    groups = {}
    for d in datasets:
        key = key_fn(d) or "—"
        g = groups.setdefault(
            key, {"group": key, "count": 0, "episodes": 0, "frames": 0, "hours": 0.0})
        g["count"] += 1
        g["episodes"] += d.get("total_episodes") or 0
        g["frames"] += d.get("total_frames") or 0
        g["hours"] += d.get("duration_hours") or 0
    total_hours = sum(g["hours"] for g in groups.values()) or 1
    rows = sorted(groups.values(), key=lambda g: g["hours"], reverse=True)
    for g in rows:
        g["hours"] = round(g["hours"], 3)
        g["pct_hours"] = round(100 * g["hours"] / total_hours, 1)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull HF datasets into one organization dataset folder."
    )
    parser.add_argument(
        "--org",
        default=ORG,
        help=f"Pull every dataset under this org/user (default: {ORG})",
    )
    parser.add_argument(
        "--repo-id",
        action="append",
        dest="repo_ids",
        metavar="OWNER/NAME",
        help="Pull only these datasets (repeat); overrides --org discovery",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/TacVerse",
        help="Organization dataset directory; datasets are stored directly inside",
    )
    parser.add_argument("--revision", default=None, help="Branch, tag, or commit")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF access token (defaults to $HF_TOKEN or the cached login token)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Download only; skip writing the summary file",
    )
    parser.add_argument(
        "--migrate-log",
        action="store_true",
        help="One-time: convert config.json's legacy pull_history into "
             "dataset_log.json and drop pull_history from config.json, then exit",
    )
    args = parser.parse_args()

    if args.migrate_log:
        path = migrate_pull_history_to_log()
        print(f"Migrated pull_history -> {path}; removed pull_history from {CONFIG_FILE}")
        return 0

    normalize_proxy_env()

    try:
        import huggingface_hub  # noqa: F401  (imported for the clear error below)
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface_hub"
        )

    meta_map = None
    if args.repo_ids:
        repo_ids = args.repo_ids
    else:
        print(f"Discovering datasets under '{args.org}' ...")
        meta = discover_datasets_meta(args.org, args.token)
        repo_ids = [m["id"] for m in meta]
        meta_map = {m["id"]: m["last_modified"] for m in meta}
        print(f"Found {len(repo_ids)} datasets.")
    if not repo_ids:
        sys.exit(f"No datasets to pull (org '{args.org}' returned nothing).")

    report, _ = run_pull(
        repo_ids,
        out_dir=args.out_dir,
        org=args.org,
        revision=args.revision,
        token=args.token,
        write_summary=not args.no_summary,
        meta_map=meta_map,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
