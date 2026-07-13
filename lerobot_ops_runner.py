#!/usr/bin/env python
"""Subprocess entry point that runs lerobot's REAL dataset_tools operations.

Why a subprocess (not an in-process import)? Importing lerobot pulls in torch +
a GPU video encoder and, in this env, occasionally segfaults or trips a flaky
pyarrow issue during heavy ops. Running it in a child process means a crash
fails one operation instead of taking down the whole PySide6 GUI. It also keeps
the workbench itself free of a hard lerobot dependency.

This must be launched with the lerobot-xense env's python (the same interpreter
the app runs under). Protocol:
  * stdin:  one JSON spec object (see SPEC below)
  * stdout: exactly one line "RESULT_JSON:<json>" with the outcome
  * stderr: human-readable progress/logs (streamed to the GUI status line)

SPEC = {
  "op": "delete" | "split" | "merge" | "add_feature" | "remove_feature",
  "vcodec": "libx264",                         # codec for video re-encode ops
  "sources": [{"repo_id": str, "root": abs}],  # 1 source, or N for merge
  "out_dir": abs,      # single-output ops (delete/remove/add/merge): dataset dir
  "out_parent": abs,   # split: parent dir; outputs = out_parent/<out_leaf>_<split>
  "out_leaf": str,     # split: base leaf name for the per-split output dirs
  "out_repo_id": str,  # repo id for the produced dataset(s)
  "params": {...}      # op-specific (see each handler)
}

RESULT = {"ok": bool, "op": str, "outputs": [{repo_id, root, episodes, frames}],
          "error": str|null}
"""

import json
import os
import shutil
import sys
import traceback
from pathlib import Path


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def _install_vcodec_shim(vcodec):
    """lerobot 0.4.1 passes vcodec='auto' straight to PyAV's add_stream(), which
    raises UnknownCodecError. Substitute a real codec without editing lerobot's
    source (so the dependency stays pristine / upgradable)."""
    import lerobot.datasets.dataset_tools as dts

    orig = dts._keep_episodes_from_video_with_av

    def patched(input_path, output_path, ranges, fps, vc="auto", pix_fmt="yuv420p"):
        if vc in (None, "auto"):
            vc = vcodec
        return orig(input_path, output_path, ranges, fps, vc, pix_fmt)

    dts._keep_episodes_from_video_with_av = patched


def _load(repo_id, root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id, root=str(root))


def _out(ds):
    return {
        "repo_id": ds.repo_id,
        "root": str(ds.root),
        "episodes": int(ds.meta.total_episodes),
        "frames": int(ds.meta.total_frames),
    }


def handle_delete(spec):
    from lerobot.datasets.dataset_tools import delete_episodes

    src = spec["sources"][0]
    ds = _load(src["repo_id"], src["root"])
    idx = list(spec["params"]["episode_indices"])
    log(f"delete_episodes: removing {len(idx)} of {ds.meta.total_episodes} episodes")
    new = delete_episodes(ds, episode_indices=idx,
                          output_dir=spec["out_dir"], repo_id=spec["out_repo_id"])
    return [_out(new)]


def handle_remove_feature(spec):
    from lerobot.datasets.dataset_tools import remove_feature

    src = spec["sources"][0]
    ds = _load(src["repo_id"], src["root"])
    names = list(spec["params"]["feature_names"])
    log(f"remove_feature: {names}")
    new = remove_feature(ds, feature_names=names,
                         output_dir=spec["out_dir"], repo_id=spec["out_repo_id"])
    return [_out(new)]


def handle_add_feature(spec):
    import numpy as np

    from lerobot.datasets.dataset_tools import add_features

    src = spec["sources"][0]
    ds = _load(src["repo_id"], src["root"])
    p = spec["params"]
    name = p["name"]
    dtype = p["dtype"]
    # shape MUST be a tuple: lerobot maps shape==(1,) (tuple) to a scalar Value,
    # but a list [1] falls through to Sequence(len=1) and mismatches the data.
    shape = tuple(p["shape"])
    fill = p["fill"]
    total = ds.meta.total_frames
    # Constant-valued feature over every frame: a dense (total, *shape) array,
    # exactly the array path add_features supports (values[frame:end]).
    values = np.full((total, *shape), fill, dtype=dtype)
    feature_info = {"dtype": dtype, "shape": shape, "names": None}
    log(f"add_features: {name} dtype={dtype} shape={shape} fill={fill} over {total} frames")
    new = add_features(ds, features={name: (values, feature_info)},
                       output_dir=spec["out_dir"], repo_id=spec["out_repo_id"])
    return [_out(new)]


def handle_merge(spec):
    from lerobot.datasets.dataset_tools import merge_datasets

    datasets = [_load(s["repo_id"], s["root"]) for s in spec["sources"]]
    log(f"merge_datasets: {len(datasets)} datasets -> {spec['out_repo_id']}")
    new = merge_datasets(datasets, output_repo_id=spec["out_repo_id"],
                         output_dir=spec["out_dir"])
    return [_out(new)]


def handle_split(spec):
    from lerobot.datasets.dataset_tools import split_dataset

    src = spec["sources"][0]
    ds = _load(src["repo_id"], src["root"])
    splits = spec["params"]["splits"]  # {name: fraction} or {name: [indices]}
    parent = Path(spec["out_parent"])
    leaf = spec["out_leaf"]
    # split_dataset writes each split to <tmp>/<split_name>; relocate each to a
    # top-level dataset dir <parent>/<leaf>_<split_name> so the workbench lists
    # them like any other pulled dataset.
    tmp = parent / f".{leaf}__split_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    log(f"split_dataset: {splits}")
    result = split_dataset(ds, splits=splits, output_dir=str(tmp))
    outputs = []
    for split_name in result:
        final = parent / f"{leaf}_{split_name}"
        if final.exists():
            raise FileExistsError(f"目标已存在: {final}")
        shutil.move(str(tmp / split_name), str(final))
        chk = _load(f"{spec['out_repo_id']}_{split_name}", final)
        outputs.append(_out(chk))
    shutil.rmtree(tmp, ignore_errors=True)
    return outputs


HANDLERS = {
    "delete": handle_delete,
    "remove_feature": handle_remove_feature,
    "add_feature": handle_add_feature,
    "merge": handle_merge,
    "split": handle_split,
}


def main():
    # Avoid SSL hangs behind a flaky proxy; loads are fully local so the Hub is
    # never contacted (LeRobotDataset only phones home if local load fails).
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)
    os.environ.setdefault("HF_LEROBOT_HOME", "/tmp/tacverse_lerobot_home")

    try:
        spec = json.loads(sys.stdin.read())
    except Exception as exc:
        print("RESULT_JSON:" + json.dumps({"ok": False, "error": f"bad spec: {exc}"}))
        return

    op = spec.get("op")
    handler = HANDLERS.get(op)
    if handler is None:
        print("RESULT_JSON:" + json.dumps({"ok": False, "error": f"unknown op: {op}"}))
        return

    try:
        _install_vcodec_shim(spec.get("vcodec", "libx264"))
        outputs = handler(spec)
        print("RESULT_JSON:" + json.dumps({"ok": True, "op": op, "outputs": outputs, "error": None}))
    except Exception as exc:
        log(traceback.format_exc())
        print("RESULT_JSON:" + json.dumps({"ok": False, "op": op, "outputs": [], "error": str(exc)}))


if __name__ == "__main__":
    main()
