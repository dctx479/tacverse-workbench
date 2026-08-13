"""Manage the vendored xense_lerobot_viewer as a black-box web service.

workbench talks to the viewer ONLY through its three stable contracts, so the
viewer's source is never modified and stays upgradable from upstream:

  ① LOCAL_DATASET_ROOT env  → the shared dataset root the viewer scans
  ② HTTP on PORT            → home `/` and `/_local/<encodedPath>` deep links
  ③ side-effect files       → meta/xense_tags.json, meta/lerobot_annotations.json

This module owns the viewer subprocess lifecycle (start / health / stop) and
builds deep-link URLs. It is Qt-free so it can be unit-tested and reused.
"""

import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
import webbrowser
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent / "vendor" / "lerobot_viewer"
DEFAULT_PORT = 3000


def encode_dataset_path(rel_path: str) -> str:
    """base64url(rel_path) without padding.

    Mirrors the viewer's encodeLocalDatasetPath so `/_local/<enc>` resolves to
    the same dataset the viewer discovered.
    """
    rel = rel_path.replace("\\", "/").strip("/")
    raw = base64.urlsafe_b64encode(rel.encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


def find_bun():
    """Locate the bun executable (PATH or the default ~/.bun install)."""
    found = shutil.which("bun")
    if found:
        return found
    bin_dir = Path.home() / ".bun" / "bin"
    candidates = [bin_dir / "bun.exe", bin_dir / "bun"] if os.name == "nt" \
        else [bin_dir / "bun"]
    return next((str(path) for path in candidates if path.is_file()), None)


def _port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _http_ok(url, timeout=1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def _same_path(left, right):
    """Compare resolved paths with the host platform's case rules."""
    if not left or not right:
        return False
    return os.path.normcase(str(Path(left).resolve())) == \
        os.path.normcase(str(Path(right).resolve()))


def _listener_pids(port):
    """Return listening process IDs for a TCP port on Windows."""
    if os.name != "nt":
        return []
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=5,
            creationflags=flags, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    pids = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0].upper() != "TCP":
            continue
        if fields[-2].upper() != "LISTENING":
            continue
        if fields[1].rsplit(":", 1)[-1] != str(port):
            continue
        try:
            pids.add(int(fields[-1]))
        except ValueError:
            pass
    return sorted(pids)


def _windows_process_table():
    """Return a minimal Windows process table keyed by PID."""
    if os.name != "nt":
        return {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    command = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=10,
            creationflags=flags, check=False,
        )
        rows = json.loads(result.stdout.lstrip("\ufeff") or "[]")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}
    if isinstance(rows, dict):
        rows = [rows]
    table = {}
    for row in rows:
        try:
            table[int(row["ProcessId"])] = row
        except (KeyError, TypeError, ValueError):
            pass
    return table


class ViewerService:
    """Supervises one `bun run dev` viewer process bound to a dataset root."""

    def __init__(self, viewer_dir=VIEWER_DIR, port=DEFAULT_PORT):
        self.viewer_dir = Path(viewer_dir)
        self.port = int(port)
        self.root = None
        self.proc = None
        self._log_path = None
        self._attached = False
        self._last_error = None
        self._info_cache = None
        self._info_cache_at = 0.0
        state_name = f"tacverse-viewer-{self.port}.json"
        self._state_path = Path(tempfile.gettempdir()) / state_name

    # --- URLs (contract ②) ------------------------------------------------
    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def home_url(self):
        return self.base_url + "/"

    def dataset_url(self, rel_path, episode=None):
        url = f"{self.base_url}/_local/{encode_dataset_path(rel_path)}"
        if episode is not None:
            url += f"/{episode}"
        return url

    def dataset_rel_path(self, dataset, root=None):
        """Path of `dataset` relative to the dataset root, or None if not local.

        Uses the record's local_dir when present, else globs the root for a
        matching leaf directory (the stats-only case). Returns None when the
        dataset isn't under the root (so the caller can disable the open link).
        """
        root = Path(root or self.root or "").resolve()
        if not root:
            return None
        local_dir = (dataset or {}).get("local_dir")
        if local_dir:
            p = Path(local_dir).resolve()
            try:
                rel = p.relative_to(root)
                return str(rel)
            except ValueError:
                pass
        leaf = ((dataset or {}).get("dataset_name") or "").split("/")[-1]
        if not leaf:
            return None
        matches = [d for d in root.glob(f"*/{leaf}") if (d / "meta").is_dir()]
        matches += [d for d in root.glob(leaf) if (d / "meta").is_dir()]
        if not matches:
            return None
        best = max(matches, key=lambda d: d.stat().st_mtime)
        return str(best.relative_to(root))

    # --- lifecycle (contract ①②) -----------------------------------------
    def available(self):
        """viewer directory + installed deps present?"""
        return (self.viewer_dir / "package.json").is_file() and \
               (self.viewer_dir / "node_modules").is_dir()

    def _invalidate_info(self):
        self._info_cache = None
        self._info_cache_at = 0.0

    def service_info(self, timeout=1.5, max_age=1.0):
        """Return the Viewer discovery response, or None for another service."""
        now = time.monotonic()
        if self._info_cache is not None and now - self._info_cache_at <= max_age:
            return self._info_cache
        try:
            with urllib.request.urlopen(
                    self.base_url + "/api/local-datasets", timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("root"), str):
                return None
            if not isinstance(data.get("datasets"), list):
                return None
        except Exception:
            return None
        self._info_cache = data
        self._info_cache_at = now
        return data

    def _root_matches(self, info=None):
        info = info if info is not None else self.service_info()
        return bool(info and _same_path(info.get("root"), self.root))

    def is_running(self):
        return self._root_matches()

    def is_ready(self):
        # The discovery endpoint proves both that this is our Viewer and that
        # it has completed startup with the expected dataset root.
        return self._root_matches()

    def _write_state(self):
        if not self.proc:
            return
        try:
            self._state_path.write_text(json.dumps({
                "pid": self.proc.pid,
                "root": self.root,
                "viewer_dir": str(self.viewer_dir.resolve()),
            }), encoding="utf-8")
        except OSError:
            pass

    def _state_pid(self, expected_root=None):
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            root = self.root if expected_root is None else expected_root
            if _same_path(state.get("root"), root) and \
                    _same_path(state.get("viewer_dir"), self.viewer_dir):
                return int(state["pid"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return None

    @staticmethod
    def _kill_windows_tree(pid):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10, creationflags=flags, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    def _windows_viewer_tree_roots(self, listener_pids):
        """Find the highest vendored Viewer ancestor for each listener PID."""
        table = _windows_process_table()
        viewer_marker = os.path.normcase(str(self.viewer_dir.resolve()))
        roots = set()
        for listener_pid in listener_pids:
            current = listener_pid
            highest = listener_pid
            visited = set()
            while current in table and current not in visited:
                visited.add(current)
                row = table[current]
                name = str(row.get("Name") or "").lower()
                command = os.path.normcase(str(row.get("CommandLine") or ""))
                if name not in {"node.exe", "node", "bun.exe", "bun"} or \
                        viewer_marker not in command:
                    break
                highest = current
                try:
                    current = int(row.get("ParentProcessId") or 0)
                except (TypeError, ValueError):
                    break
            roots.add(highest)
        return sorted(roots)

    def _stop_processes(self, actual_root=None):
        """Stop the launch process and any server still listening on our port."""
        parent_pid = self.proc.pid if self.proc else (
            self._state_pid(actual_root) if _port_in_use(self.port) else None)
        if os.name == "nt":
            if parent_pid:
                self._kill_windows_tree(parent_pid)
            listener_pids = _listener_pids(self.port)
            for pid in self._windows_viewer_tree_roots(listener_pids):
                self._kill_windows_tree(pid)
        elif self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except OSError:
                    pass
        elif parent_pid:
            try:
                os.kill(parent_pid, signal.SIGTERM)
            except OSError:
                pass

        deadline = time.monotonic() + 5
        while _port_in_use(self.port) and time.monotonic() < deadline:
            time.sleep(0.1)

    def start(self, root, wait=False, timeout=60, log_path=None):
        """Launch (or reuse) the viewer bound to `root`. Returns (ok, message).

        `wait=False` returns as soon as the process is spawned; poll status()
        for readiness. `wait=True` blocks until the discovery API answers or
        times out. A compatible Viewer on this dedicated port is reused when it
        has the requested root, or replaced when it has a stale root.
        """
        root_path = Path(root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        self.root = str(root_path)
        self._last_error = None
        self._invalidate_info()

        if self.proc and self.proc.poll() is None:
            info = self.service_info(max_age=0)
            if not info or self._root_matches(info):
                if wait:
                    return self._wait_until_ready(timeout)
                return True, "启动中…" if not info else \
                    f"已在运行 · {self.base_url}"
            self._stop_processes(info.get("root"))
            self.proc = None
            self._invalidate_info()

        if _port_in_use(self.port):
            info = self.service_info(max_age=0)
            if not info:
                self._last_error = f"端口 {self.port} 已被其他程序占用"
                return False, self._last_error
            self._attached = True
            if self._root_matches(info):
                return True, f"已连接现有 Viewer · {self.base_url}"
            # Port 3001 is reserved for Workbench's Viewer. Replace a verified
            # Viewer left behind with an obsolete dataset root.
            old_root = info.get("root") or "未知"
            self._stop_processes(info.get("root"))
            self._attached = False
            self._invalidate_info()
            if _port_in_use(self.port):
                self._last_error = f"无法关闭旧 Viewer（数据根: {old_root}）"
                return False, self._last_error
        if not self.available():
            self._last_error = f"Viewer 未就绪（缺 node_modules）: {self.viewer_dir}"
            return False, self._last_error
        bun = find_bun()
        if not bun:
            self._last_error = "未找到 Bun，请先安装 Bun"
            return False, self._last_error

        env = dict(os.environ)
        env["LOCAL_DATASET_ROOT"] = self.root
        env["PORT"] = str(self.port)
        self._log_path = str(log_path or (
            Path(tempfile.gettempdir()) / f"tacverse-viewer-{self.port}.log"))
        out = open(self._log_path, "w", encoding="utf-8")
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen(
                [bun, "run", "dev"],
                cwd=str(self.viewer_dir),
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
                creationflags=creationflags,
            )
        except Exception as exc:
            self.proc = None
            self._last_error = f"启动失败: {exc}"
            return False, self._last_error
        finally:
            out.close()

        self._attached = False
        self._write_state()

        if not wait:
            return True, "启动中…"
        return self._wait_until_ready(timeout)

    def _wait_until_ready(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                self._last_error = f"Viewer 进程意外退出（日志: {self._log_path}）"
                return False, self._last_error
            self._invalidate_info()
            if self.is_ready():
                return True, f"运行中 · {self.base_url}"
            time.sleep(0.5)
        self._last_error = f"启动超时（{timeout}s，日志: {self._log_path}）"
        return False, self._last_error

    def stop(self):
        """Terminate this Workbench Viewer and its full process tree."""
        port_in_use = _port_in_use(self.port)
        info = self.service_info(max_age=0) if port_in_use else None
        if port_in_use and not info:
            return False, f"端口 {self.port} 不是可识别的 Viewer，未关闭外部服务"
        owns_process = bool(self.proc or self._attached or
                            (port_in_use and self._state_pid(info.get("root"))))
        if info and not self._root_matches(info):
            return False, "端口上的 Viewer 数据根不匹配，未关闭外部服务"
        if not owns_process:
            try:
                self._state_path.unlink(missing_ok=True)
            except OSError:
                pass
            return True, "Viewer 已停止"

        self._stop_processes(info.get("root") if info else self.root)
        self.proc = None
        self._attached = False
        self._invalidate_info()
        try:
            self._state_path.unlink(missing_ok=True)
        except OSError:
            pass
        if _port_in_use(self.port):
            self._last_error = f"Viewer 未能完全停止，端口 {self.port} 仍被占用"
            return False, self._last_error
        self._last_error = None
        return True, "Viewer 已停止"

    # --- introspection (contract ②) --------------------------------------
    def dataset_count(self):
        """How many datasets the viewer currently sees (JSON API), or None."""
        data = self.service_info(timeout=3)
        return len(data["datasets"]) if data else None

    def report(self, rel_path, include=None, timeout=180):
        """Fetch the viewer's /report analysis JSON for a dataset.

        Returns (report_dict, None) on success, or (None, error_message). The
        analysis is computed server-side and can take tens of seconds — call
        this off the UI thread.
        """
        url = f"{self.base_url}/api/local-datasets/{encode_dataset_path(rel_path)}/report"
        if include:
            url += "?include=" + ",".join(include)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return None, str(exc)
        if isinstance(data, dict) and data.get("ok") is False:
            return None, str(data.get("error") or "analysis failed")
        return data, None

    def status(self):
        port_in_use = _port_in_use(self.port)
        info = self.service_info() if port_in_use else None
        ready = bool(info and self._root_matches(info))
        process_alive = bool(self.proc and self.proc.poll() is None)
        if self.proc and not process_alive and not port_in_use and not self._last_error:
            self._last_error = f"Viewer 进程已退出（日志: {self._log_path}）"
        if ready:
            state = "ready"
        elif process_alive:
            state = "starting"
        elif port_in_use:
            state = "conflict"
        elif self._last_error:
            state = "error"
        else:
            state = "stopped"
        return {
            "state": state,
            "running": ready,
            "ready": ready,
            "managed": bool(process_alive or self._attached),
            "port_in_use": port_in_use,
            "port": self.port,
            "root": self.root,
            "actual_root": info.get("root") if info else None,
            "url": self.base_url,
            "error": self._last_error,
            "log_path": self._log_path,
        }

    # --- convenience openers ----------------------------------------------
    def open_home(self):
        webbrowser.open(self.home_url())

    def open_dataset(self, rel_path, episode=None):
        webbrowser.open(self.dataset_url(rel_path, episode))
