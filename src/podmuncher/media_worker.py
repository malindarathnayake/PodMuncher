# ruff: noqa: BLE001
"""In-image media worker: typed metadata, download and probe operations.

The worker speaks the version 1 worker JSONL protocol: one request line on
stdin, JSONL events on stdout, nothing else. It runs inside the media image
only (``python -I -m podmuncher.media_worker``); the host never imports yt-dlp.

Operations (assigned slot sets are exact, anything else is rejected before any
network or process I/O):

* metadata / youtube: yt-dlp ``extract_info(<id>, download=False)``, forced
  YouTube extractor, finite stream/duration metadata only.
* metadata / direct: no I/O; every stream and duration field is null.
* download / direct: ``direct_download.download`` (reused, not copied).
* download / youtube: yt-dlp streams to the assigned basenames, then an FFmpeg
  stream-copy remux; a single ``merged_video`` artifact for video outputs, or
  ``downloaded_audio`` for transcript-only.
* probe: ffprobe over the read-only ``/input/<basename>`` mount.

This module never claims that outbound worker network access is a sandbox; the
host grants a bridge network to metadata and download operations only, and
remote hosts are reached by yt-dlp (YouTube) or ``direct_download`` (public HTTPS).
Events never carry titles, URLs, raw downloader text or host paths.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO

from . import direct_download
from .domains import Capability, DomainValidationError, ReasonCode
from .protocol import WorkerEventRecord, WorkerRequest, emit_event

__all__ = (
    "YDL_OPTION_NAMES",
    "ffmpeg_merge_arguments",
    "ffprobe_arguments",
    "format_selector",
    "main",
    "run",
    "verify_installed_yt_dlp",
)

_YOUTUBE_URL = re.compile(r"https://www\.youtube\.com/watch\?v=([A-Za-z0-9_-]{11})")
_FORMAT_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_OPERATIONS = frozenset(("metadata", "download", "probe"))
_PROBE_SLOT = frozenset(("probe_metadata",))
_MEDIA_SLOT = frozenset(("downloaded_media",))
_AUDIO_SLOT = frozenset(("downloaded_audio",))
_VIDEO_SLOTS = frozenset(("downloaded_video", "downloaded_audio", "merged_video"))
_TMP_NAME = ".media-tmp"
_MAX_REQUEST_BYTES = 65537
_MAX_PROBE_STDOUT = 4 * 1024 * 1024
_MAX_METADATA_BYTES = 64 * 1024
_HASH_CHUNK = 1024 * 1024
_YTDLP_SOCKET_TIMEOUT = 30
_LIVE_STATUSES = frozenset(("is_live", "is_upcoming", "was_live", "post_live"))
_PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "FTP_PROXY")
DEFAULT_DENO_PATH = "/usr/local/bin/deno"

# Every key placed in the yt-dlp parameter dict. verify_installed_yt_dlp() proves
# each name still exists in the installed yt-dlp source at image build time.
YDL_OPTION_NAMES = (
    "quiet", "no_warnings", "noprogress", "logger", "noplaylist", "cookiefile", "cachedir",
    "allowed_extractors", "socket_timeout", "retries", "fragment_retries", "extractor_retries",
    "max_filesize", "js_runtimes", "remote_components", "nocheckcertificate", "proxy", "format",
    "progress_hooks", "fixup", "updatetime", "ignoreerrors", "overwrites",
    "external_downloader", "external_downloader_args",
)


class _Fail(Exception):
    """A closed worker failure. Never carries downloader or filesystem text."""

    def __init__(self, reason: ReasonCode) -> None:
        super().__init__(reason.value)
        self.reason = reason


class _Abort(Exception):
    """Raised from a yt-dlp progress hook to stop a download; the reason is recorded."""


class _NullLogger:
    """yt-dlp logger that drops every message (stdout carries only JSONL events)."""

    def debug(self, message: object) -> None:
        pass

    def info(self, message: object) -> None:
        pass

    def warning(self, message: object) -> None:
        pass

    def error(self, message: object) -> None:
        pass


# --------------------------------------------------------------------------
# Public argument builders
# --------------------------------------------------------------------------

def ffprobe_arguments(path: str) -> tuple[str, ...]:
    """Fixed ffprobe argument array; only the local file protocol is allowed."""
    _checked_path(path)
    return ("ffprobe", "-v", "error", "-hide_banner", "-protocol_whitelist", "file",
            "-print_format", "json", "-show_format", "-show_streams", "-i", path)


def ffmpeg_merge_arguments(video: str, audio: str | None, output: str,
                           byte_limit: int) -> tuple[str, ...]:
    """Stream-copy remux; ``-n`` (never ``-y``) so the output must not pre-exist."""
    _checked_path(video)
    _checked_path(output)
    if audio is not None:
        _checked_path(audio)
    if type(byte_limit) is not int or byte_limit <= 0:
        raise ValueError("invalid byte limit")
    arguments = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                 "-protocol_whitelist", "file", "-i", video]
    if audio is None:
        # A progressive file: the audio stream is optional.
        maps = ["-map", "0:v:0", "-map", "0:a:0?"]
    else:
        arguments += ["-protocol_whitelist", "file", "-i", audio]
        maps = ["-map", "0:v:0", "-map", "1:a:0"]
    # Matroska accepts any stream-copied codec, and the muxer is explicit because
    # host-generated basenames carry no reliable extension.
    return (*arguments, *maps, "-c", "copy", "-fs", str(byte_limit), "-f", "matroska", output)


def format_selector(video: bool, video_quality: str) -> str:
    """yt-dlp format expression: height cap only, never a re-encode."""
    if not video:
        return "ba/b"
    if video_quality == "1080p":
        return "bv*[height<=1080]+ba/b[height<=1080]"
    return "bv*+ba/b"


def _checked_path(value: object) -> str:
    if (type(value) is not str or not value or value.startswith("-")
            or any(ord(char) < 32 for char in value)):
        raise ValueError("invalid path")
    return value


# --------------------------------------------------------------------------
# Small pure helpers
# --------------------------------------------------------------------------

def _finite_number(value: object) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return float(value)


def _nonnegative_int(value: object) -> int | None:
    number = _finite_number(value)
    if number is None or number < 0 or number != int(number):
        return None
    return int(number)


def _metadata_record(operation: str, source_kind: str | None, **values: object) -> dict:
    record = {
        "schema_version": 1, "operation": operation, "source_kind": source_kind,
        "duration_seconds": None, "has_audio": None, "has_video": None, "video_height": None,
        "audio_streams": None, "video_streams": None, "estimated_bytes": None,
    }
    for key, value in values.items():
        if key not in record or key in ("schema_version", "operation", "source_kind"):
            raise ValueError("unknown metadata key")
        record[key] = value
    return record


def _youtube_id(source: object) -> str:
    """The canonical id, strictly re-validated; anything else is input_rejected."""
    value = source.get("input") if isinstance(source, Mapping) else None
    match = _YOUTUBE_URL.fullmatch(value) if type(value) is str else None
    if match is None:
        raise _Fail(ReasonCode.INPUT_REJECTED)
    return match.group(1)


def _error_chain(error: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    stack: list[object] = [error]
    while stack and len(seen) < 16:
        item = stack.pop()
        if not isinstance(item, BaseException) or any(item is known for known in seen):
            continue
        seen.append(item)
        stack.extend((item.__cause__, item.__context__))
        info = getattr(item, "exc_info", None)
        if isinstance(info, tuple) and len(info) > 1:
            stack.append(info[1])
    return seen


def _reason_for(error: BaseException) -> ReasonCode:
    """Map any failure to a closed reason. Raw text is inspected, never forwarded."""
    chain = _error_chain(error)
    for item in chain:
        if isinstance(item, _Fail):
            return item.reason
    for item in chain:
        if isinstance(item, OSError) and item.errno == errno.ENOSPC:
            return ReasonCode.STORAGE_GUARD_TRIPPED
    for item in chain:
        if isinstance(item, DomainValidationError):
            return (item.reason_code if item.reason_code is not ReasonCode.OK
                    else ReasonCode.WORKER_PROTOCOL_MISMATCH)
    if type(error).__module__.split(".")[0] == "yt_dlp" or any(
            type(item).__module__.split(".")[0] == "yt_dlp" for item in chain):
        return _classify_yt_dlp(chain)
    return ReasonCode.WORKER_EXIT_UNEXPECTED


def _classify_yt_dlp(chain: Sequence[BaseException]) -> ReasonCode:
    text = " ".join(str(item).lower() for item in chain)
    if "certificate verify failed" in text:
        # TLS verification failure is a transport problem, not a content denial.
        return ReasonCode.UPSTREAM_UNAVAILABLE
    if "format is not available" in text or "no video formats" in text:
        return ReasonCode.INPUT_REJECTED
    if any(word in text for word in ("playlist", "live event", "premiere", "is live", "will begin")):
        return ReasonCode.INPUT_REJECTED
    if "max-filesize" in text or "larger than" in text:
        return ReasonCode.MEDIA_LIMIT_REACHED
    if any(word in text for word in (
            "private video", "sign in", "members-only", "age-restricted", "confirm your age",
            "not available in your country", "geo restrict", "geo-restrict", "geoblock", "copyright", "video unavailable",
            "has been removed", "terminated")):
        return ReasonCode.CONTENT_DENIED
    return ReasonCode.UPSTREAM_UNAVAILABLE


def _has_codec(item: Mapping, key: str) -> bool | None:
    value = item.get(key)
    if value is None:
        return None
    return isinstance(value, str) and value not in ("", "none")


def _sha256(path: Path) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


# --------------------------------------------------------------------------
# Event emission
# --------------------------------------------------------------------------

class _Emitter:
    def __init__(self, request: WorkerRequest, sink: Callable[[bytes], object],
                 clock: Callable[[], float]) -> None:
        self._ids = (request.job_id, request.attempt_id, request.stage_id)
        self._sink, self._clock = sink, clock
        self._sequence = 0
        self._last_progress: float | None = None

    def emit(self, event: str, payload: Mapping[str, object] | None = None) -> None:
        record = WorkerEventRecord(1, *self._ids, self._sequence, event, dict(payload or {}))
        emit_event(record, self._sink)
        self._sequence += 1

    def progress(self, percent: float) -> None:
        """At most one progress event per second (the host allows ten)."""
        now = self._clock()
        if self._last_progress is not None and now - self._last_progress < 1.0:
            return
        self._last_progress = now
        self.emit("progress", {"percent": round(min(100.0, max(0.0, percent)), 2)})

    def error(self, reason: ReasonCode) -> None:
        self.emit("error", {"reason_code": reason.value,
                            "summary": reason.name.replace("_", " ").capitalize()})


# --------------------------------------------------------------------------
# Production defaults
# --------------------------------------------------------------------------

def _default_run_process(arguments: Sequence[str], timeout: float) -> subprocess.CompletedProcess:
    """shell=False argument array; stderr is discarded, never forwarded."""
    process = subprocess.Popen(list(arguments), shell=False, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        with contextlib.suppress(OSError):
            process.kill()

    timer = threading.Timer(timeout, expire)
    timer.start()
    try:
        # Never buffer more than the cap (plus one byte, to detect overflow).
        data = process.stdout.read(_MAX_PROBE_STDOUT + 1)  # type: ignore[union-attr]
        if len(data) > _MAX_PROBE_STDOUT:
            process.kill()
        returncode = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        timer.cancel()
        if process.stdout is not None:
            process.stdout.close()
    if expired.is_set():
        raise subprocess.TimeoutExpired(list(arguments), timeout)
    return subprocess.CompletedProcess(list(arguments), returncode, data, None)


def deno_path() -> str:
    """The deno binary installed by the pinned ``deno`` wheel."""
    with contextlib.suppress(Exception):
        import deno  # type: ignore[import-not-found]
        found = deno.find_deno_bin()
        if found:
            return str(found)
    return shutil.which("deno") or DEFAULT_DENO_PATH


def _prepare_environment(tmp: Path) -> None:
    """Writable caches live under the worker's own scratch subdirectory; no proxies."""
    for name in _PROXY_ENV:
        os.environ.pop(name, None)
        os.environ.pop(name.lower(), None)
    for name, sub in (("HOME", "home"), ("XDG_CACHE_HOME", "cache"), ("XDG_CONFIG_HOME", "config"),
                      ("DENO_DIR", "deno")):
        target = tmp / sub
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    os.environ["TMPDIR"] = str(tmp)
    os.environ["DENO_NO_UPDATE_CHECK"] = "1"
    tempfile.tempdir = str(tmp)


def _production_ydl_factory(tmp: Path) -> Callable[[dict], object]:
    _prepare_environment(tmp)
    import yt_dlp  # lazy: the host venv has no yt-dlp.
    # Plugin off-switch: yt_dlp/globals.py ``plugin_dirs`` (an Indirect holding the
    # search list; ['default'] by default). yt_dlp/plugins.py load_plugins() returns early
    # when ``not plugin_dirs.value``, and yt_dlp/YoutubeDL.py YoutubeDL.__init__ calls
    # load_all_plugins() (not import time), so this assignment must precede construction.
    # Verified in the built image: with [] no plugin_ies/plugin_pps are registered.
    yt_dlp.globals.plugin_dirs.value = []
    return yt_dlp.YoutubeDL


def verify_installed_yt_dlp() -> dict[str, str]:
    """Build-time proof that every option and switch used here exists in yt-dlp.

    Returns option name -> yt_dlp source file where the name appears, and raises
    AssertionError for anything missing.
    """
    import yt_dlp

    root = Path(yt_dlp.__file__).parent
    files = [path for pattern in ("*.py", "downloader/*.py", "networking/*.py", "postprocessor/*.py")
             for path in sorted(root.glob(pattern))]
    sources = {path: path.read_text(encoding="utf-8", errors="replace") for path in files}
    found: dict[str, str] = {}
    for name in YDL_OPTION_NAMES:
        pattern = re.compile(r"""(?<![A-Za-z0-9_])['"]?""" + re.escape(name) + r"""['"]?(?![A-Za-z0-9_])""")
        for path, text in sources.items():
            if pattern.search(text):
                found[name] = "yt_dlp/" + path.relative_to(root).as_posix()
                break
        else:
            raise AssertionError("yt-dlp option not found: " + name)
    # The Python API never loads configuration: only options.py reads config files.
    api = sources[root / "YoutubeDL.py"]
    assert "load_configs" not in api and "read_config" not in api
    assert hasattr(yt_dlp.globals, "plugin_dirs")
    assert callable(yt_dlp.YoutubeDL.dl) and callable(yt_dlp.YoutubeDL.extract_info)
    import yt_dlp_ejs  # noqa: F401
    return found


# --------------------------------------------------------------------------
# Job execution
# --------------------------------------------------------------------------

class _Bytes:
    """Cumulative downloaded bytes across all yt-dlp streams, merged by filename."""

    def __init__(self) -> None:
        self.by_file: dict[str, int] = {}
        self.abort: ReasonCode | None = None

    @property
    def total(self) -> int:
        return sum(self.by_file.values())


class _Job:
    def __init__(self, request: WorkerRequest, emitter: _Emitter, *, scratch: Path,
                 input_root: Path, ydl_factory: Callable[[dict], object] | None,
                 run_process: Callable[[Sequence[str], float], subprocess.CompletedProcess] | None,
                 direct: Callable[..., Awaitable[direct_download.DownloadResult]],
                 free_space: Callable[[], int] | None, clock: Callable[[], float],
                 js_runtime_path: str | None) -> None:
        self.request, self.emitter = request, emitter
        self.scratch, self.input_root = scratch, input_root
        self._ydl_factory, self._run_process = ydl_factory, run_process or _default_run_process
        self._ydl_injected = ydl_factory is not None
        self.direct = direct
        self.free_space = free_space or (lambda: shutil.disk_usage(scratch).free)
        self.clock = clock
        self._js_runtime_path = js_runtime_path
        self.limits = request.limits
        self.tmp = scratch / _TMP_NAME
        self._tmp_created = False

    # ---- envelope -------------------------------------------------------

    def _expected_slots(self) -> frozenset[str]:
        request = self.request
        if request.capability != Capability.MEDIA.value or request.operation not in _OPERATIONS:
            raise _Fail(ReasonCode.WORKER_PROTOCOL_MISMATCH)
        source = request.source
        if request.operation == "probe":
            if source is not None or request.input_basename is None:
                raise _Fail(ReasonCode.WORKER_PROTOCOL_MISMATCH)
            return _PROBE_SLOT
        if source is None or request.input_basename is not None:
            raise _Fail(ReasonCode.WORKER_PROTOCOL_MISMATCH)
        kind = source.get("kind")
        if kind not in ("direct", "youtube"):
            raise _Fail(ReasonCode.INPUT_REJECTED)
        if request.operation == "metadata":
            return _PROBE_SLOT
        if kind == "direct":
            return _MEDIA_SLOT
        return _VIDEO_SLOTS if request.outputs.video else _AUDIO_SLOT

    def execute(self) -> None:
        expected = self._expected_slots()
        if set(self.request.assigned_slots) != expected:
            raise _Fail(ReasonCode.WORKER_PROTOCOL_MISMATCH)
        operation, source = self.request.operation, self.request.source
        if operation == "probe":
            self._probe()
        elif source is not None and source["kind"] == "direct":
            if operation == "metadata":
                self._metadata_direct()
            else:
                self._download_direct()
        else:
            video_id = _youtube_id(source)  # validated before any I/O
            if operation == "metadata":
                self._metadata_youtube(video_id)
            else:
                self._download_youtube(video_id)

    def cleanup(self, *, failed: bool) -> None:
        """Remove the temp dir; on failure also this operation's partial outputs."""
        with contextlib.suppress(Exception):
            if self._tmp_created:
                shutil.rmtree(self.tmp, ignore_errors=True)
        if failed:
            for basename in self.request.assigned_slots.values():
                for path in self.scratch.glob(_escape_glob(basename) + "*"):
                    with contextlib.suppress(OSError):
                        if path.is_file() or path.is_symlink():
                            path.unlink()

    # ---- artifacts ------------------------------------------------------

    def _slot_path(self, slot: str) -> Path:
        return self.scratch / self.request.assigned_slots[slot]

    def _announce(self, slot: str) -> None:
        size, digest = _sha256(self._slot_path(slot))
        self.emitter.emit("artifact", {"slot": slot, "bytes": size, "sha256": digest})

    def _write_metadata(self, record: dict) -> None:
        data = json.dumps(record, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(data) > _MAX_METADATA_BYTES:
            raise _Fail(ReasonCode.ARTIFACT_REJECTED)
        try:
            with self._slot_path("probe_metadata").open("xb") as handle:
                handle.write(data)
        except FileExistsError:
            raise _Fail(ReasonCode.DESTINATION_CONFLICT) from None
        self._announce("probe_metadata")

    # ---- probe ----------------------------------------------------------

    def _run(self, arguments: Sequence[str]) -> subprocess.CompletedProcess:
        try:
            return self._run_process(arguments, float(self.limits.decode_timeout_seconds))
        except OSError as error:
            if error.errno == errno.ENOSPC:
                raise _Fail(ReasonCode.STORAGE_GUARD_TRIPPED) from None
            raise _Fail(ReasonCode.MEDIA_PROBE_FAILED) from None
        except (subprocess.SubprocessError, ValueError):
            raise _Fail(ReasonCode.MEDIA_PROBE_FAILED) from None

    def _probe(self) -> None:
        path = self.input_root / str(self.request.input_basename)
        if not _regular_file(path):
            raise _Fail(ReasonCode.MEDIA_PROBE_FAILED)
        result = self._run(ffprobe_arguments(str(path)))
        stdout = result.stdout
        if isinstance(stdout, str):
            stdout = stdout.encode("utf-8", "replace")
        if result.returncode != 0 or not isinstance(stdout, bytes) or len(stdout) > _MAX_PROBE_STDOUT:
            raise _Fail(ReasonCode.MEDIA_PROBE_FAILED)
        record = _parse_ffprobe(stdout)
        duration = record["duration_seconds"]
        if duration is not None and duration > self.limits.duration_seconds:
            raise _Fail(ReasonCode.DURATION_EXCEEDED)
        self._write_metadata(record)

    # ---- metadata -------------------------------------------------------

    def _metadata_direct(self) -> None:
        self._write_metadata(_metadata_record("metadata", "direct"))

    def _metadata_youtube(self, video_id: str) -> None:
        selector = format_selector(self.request.outputs.video, self.request.outputs.video_quality)
        started = self.clock()
        with self._session(selector) as ydl:
            info = self._extract(ydl, video_id)
        self._check_deadline(started)
        selected = _selected_formats(info)
        duration = self._checked_duration(info)
        audio = [_has_codec(item, "acodec") for item in selected]
        video = [_has_codec(item, "vcodec") for item in selected]
        known_audio, known_video = any(v is not None for v in audio), any(v is not None for v in video)
        heights = [h for item, v in zip(selected, video) if v
                   for h in (_nonnegative_int(item.get("height")),) if h]
        self._write_metadata(_metadata_record(
            "metadata", "youtube", duration_seconds=duration,
            has_audio=any(audio) if known_audio else None,
            has_video=any(video) if known_video else None,
            video_height=max(heights) if heights else None,
            audio_streams=sum(1 for v in audio if v) if known_audio else None,
            video_streams=sum(1 for v in video if v) if known_video else None,
            estimated_bytes=_estimate(selected)))

    def _checked_duration(self, info: Mapping) -> float | None:
        """Finite duration (negative is unknown); above the limit is DURATION_EXCEEDED."""
        duration = _finite_number(info.get("duration"))
        if duration is not None and duration < 0:
            duration = None
        if duration is not None and duration > self.limits.duration_seconds:
            raise _Fail(ReasonCode.DURATION_EXCEEDED)
        return duration

    def _check_deadline(self, started: float) -> None:
        """Operation time limit outside the progress hook (extraction does not call it)."""
        if self.clock() - started > float(self.limits.download_timeout_seconds):
            raise _Fail(ReasonCode.DOWNLOAD_TIMEOUT)

    # ---- yt-dlp ---------------------------------------------------------

    def _ensure_tmp(self) -> None:
        if not self._tmp_created:
            self.tmp.mkdir(mode=0o700)
            self._tmp_created = True
            if self._ydl_factory is None:
                self._ydl_factory = _production_ydl_factory(self.tmp)

    def _ydl_options(self, selector: str, *,
                     hooks: Sequence[Callable[[Mapping], None]] = ()) -> dict:
        attempts = self.limits.download_attempts
        runtime = self._js_runtime_path or (DEFAULT_DENO_PATH if self._ydl_injected else deno_path())
        options: dict[str, object] = {
            # yt_dlp/YoutubeDL.py (YoutubeDL.__doc__ parameters) for every key below
            # unless a different file is named; verify_installed_yt_dlp() re-proves
            # each name against the installed source when the image is built.
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "logger": _NullLogger(),
            "noplaylist": True,
            "cookiefile": None,            # no cookie jar; cookiesfrombrowser never set
            "cachedir": False,             # yt_dlp/cache.py: Cache.enabled is `cachedir is not False`
            # yt_dlp/YoutubeDL.py (allowed_extractors, matched via get_info_extractor
            # setup at ~line 936): regexes over IE_NAME; ['^youtube$'] leaves only the
            # youtube extractor registered (verified in the image: _ies == ['youtube']).
            "allowed_extractors": ["^youtube$"],
            "socket_timeout": _YTDLP_SOCKET_TIMEOUT,
            "retries": attempts,           # yt_dlp/downloader/common.py
            "fragment_retries": attempts,  # yt_dlp/downloader/fragment.py
            "extractor_retries": attempts,
            "max_filesize": self.limits.media_bytes,   # yt_dlp/YoutubeDL.py _match_entry
            # yt_dlp/YoutubeDL.py _clean_js_runtimes(): keys must be in
            # yt_dlp/globals.py supported_js_runtimes (deno, node, bun, quickjs).
            "js_runtimes": {"deno": {"path": runtime}},
            # yt_dlp/YoutubeDL.py: stored as a set; supported values are ejs:github and
            # ejs:npm (globals.supported_remote_components); [] allows no remote fetch.
            "remote_components": [],
            "nocheckcertificate": False,   # TLS verification stays enabled
            "proxy": "",                   # yt_dlp/YoutubeDL.py: empty string ignores proxy env
            "format": selector,
            "fixup": "never",              # no hidden FFmpeg fixup passes
            "updatetime": False,
            "ignoreerrors": False,
            "overwrites": False,
            "external_downloader": None,
            "external_downloader_args": None,
        }
        if hooks:
            options["progress_hooks"] = list(hooks)
        return options

    @contextlib.contextmanager
    def _session(self, selector: str, *, hooks: Sequence[Callable[[Mapping], None]] = (),
                 state: _Bytes | None = None):
        """One YoutubeDL instance; every failure leaves as a closed ``_Fail`` reason."""
        self._ensure_tmp()
        options = self._ydl_options(selector, hooks=hooks)
        try:
            with self._ydl_factory(options) as ydl:  # type: ignore[misc]
                yield ydl
        except _Fail:
            raise
        except _Abort:
            raise _Fail(state.abort if state and state.abort else ReasonCode.WORKER_EXIT_UNEXPECTED) from None
        except Exception as error:
            if state is not None and state.abort is not None:
                raise _Fail(state.abort) from None
            raise _Fail(_reason_for(error)) from None

    def _extract(self, ydl: object, video_id: str) -> Mapping:
        """The single ``extract_info(<id>, download=False)`` of an operation, validated."""
        try:
            info = ydl.extract_info(video_id, download=False)  # type: ignore[attr-defined]
        except _Abort:
            raise
        except Exception as error:
            raise _Fail(_reason_for(error)) from None
        if not isinstance(info, Mapping):
            raise _Fail(ReasonCode.UPSTREAM_UNAVAILABLE)
        if info.get("_type") in ("playlist", "multi_video") or "entries" in info:
            raise _Fail(ReasonCode.INPUT_REJECTED)
        if (info.get("is_live") or info.get("was_live")
                or info.get("live_status") in _LIVE_STATUSES):
            raise _Fail(ReasonCode.INPUT_REJECTED)
        if info.get("id") is not None and info.get("id") != video_id:
            raise _Fail(ReasonCode.SOURCE_CHANGED)
        return info

    @staticmethod
    def _stream_info(info: Mapping, chosen: Mapping) -> dict:
        """The per-format info dict handed to ``YoutubeDL.dl``.

        Mirrors yt_dlp/YoutubeDL.py process_info() (the requested_formats loop): copy the
        info dict, drop ``requested_formats`` and overlay the chosen format.
        """
        merged = {key: value for key, value in info.items() if key != "requested_formats"}
        merged.update(chosen)
        merged.pop("requested_formats", None)  # a progressive format is its own info dict
        identifier = merged.get("format_id")
        if type(identifier) is not str or _FORMAT_ID.fullmatch(identifier) is None:
            raise _Fail(ReasonCode.INPUT_REJECTED)
        return merged

    def _download_youtube(self, video_id: str) -> None:
        request, limits = self.request, self.limits
        selector = format_selector(request.outputs.video, request.outputs.video_quality)
        state, started = _Bytes(), self.clock()
        timeout = float(limits.download_timeout_seconds)
        estimate: int | None = None

        def hook(update: Mapping) -> None:
            if not isinstance(update, Mapping):
                return
            name, got = update.get("filename"), update.get("downloaded_bytes")
            if type(name) is str and _finite_number(got) is not None and got >= 0:
                state.by_file[name] = max(state.by_file.get(name, 0), int(got))
            if state.total > limits.media_bytes:
                state.abort = ReasonCode.MEDIA_LIMIT_REACHED
                raise _Abort
            if self.clock() - started > timeout:
                state.abort = ReasonCode.DOWNLOAD_TIMEOUT
                raise _Abort
            if estimate:
                self.emitter.progress(state.total * 100.0 / max(estimate, 1))

        video_slot, audio_slot = "downloaded_video", "downloaded_audio"
        with self._session(selector, hooks=(hook,), state=state) as ydl:
            # Exactly one extraction per operation: YouTube serves a different format set
            # on a second request, so the formats chosen here are downloaded from this very
            # result, inside this same YoutubeDL instance.
            info = self._extract(ydl, video_id)
            self._check_deadline(started)
            self._checked_duration(info)
            selected = _selected_formats(info)
            estimate = _estimate(selected)
            if estimate is not None and estimate > limits.media_bytes:
                raise _Fail(ReasonCode.MEDIA_LIMIT_REACHED)

            if not request.outputs.video:
                if len(selected) != 1:
                    raise _Fail(ReasonCode.INPUT_REJECTED)
                plan = [(audio_slot, self._stream_info(info, selected[0]))]
            elif len(selected) >= 2:
                video_format = next((i for i in selected if _has_codec(i, "vcodec")), None)
                audio_format = next((i for i in selected if i is not video_format
                                     and _has_codec(i, "acodec")), None)
                if video_format is None or audio_format is None:
                    raise _Fail(ReasonCode.INPUT_REJECTED)
                plan = [(video_slot, self._stream_info(info, video_format)),
                        (audio_slot, self._stream_info(info, audio_format))]
            else:
                if _has_codec(selected[0], "vcodec") is False:
                    raise _Fail(ReasonCode.INPUT_REJECTED)
                plan = [(video_slot, self._stream_info(info, selected[0]))]

            for slot, stream in plan:
                target = self._slot_path(slot)
                if target.exists() or target.is_symlink():
                    raise _Fail(ReasonCode.DESTINATION_CONFLICT)
                self._check_deadline(started)
                # yt_dlp/YoutubeDL.py YoutubeDL.dl(name, info): the public method that
                # downloads one already-selected format dict to an explicit filename. It
                # runs only the downloader (progress hooks included); the merger and the
                # fixup postprocessors live in process_info(), which is never called.
                try:
                    outcome = ydl.dl(str(target), stream)  # type: ignore[attr-defined]
                except _Abort:
                    raise
                except Exception as error:
                    if state.abort is not None:
                        raise _Fail(state.abort) from None
                    raise _Fail(_reason_for(error)) from None
                success = outcome[0] if isinstance(outcome, tuple) else bool(outcome)
                if not success or not _regular_file(target):
                    # yt-dlp declines (without raising) a stream above max_filesize.
                    raise _Fail(ReasonCode.MEDIA_LIMIT_REACHED)
                if self.clock() - started > timeout:
                    raise _Fail(ReasonCode.DOWNLOAD_TIMEOUT)
        if sum(self._slot_path(slot).stat().st_size for slot, _ in plan) > limits.media_bytes:
            raise _Fail(ReasonCode.MEDIA_LIMIT_REACHED)

        if not request.outputs.video:
            self._announce(audio_slot)
            return
        has_audio_file = any(slot == audio_slot for slot, _ in plan)
        self._merge(self._slot_path(video_slot),
                    self._slot_path(audio_slot) if has_audio_file else None)

    def _merge(self, video: Path, audio: Path | None) -> None:
        output = self._slot_path("merged_video")
        limit = self.limits.media_bytes
        if output.exists() or output.is_symlink():
            raise _Fail(ReasonCode.DESTINATION_CONFLICT)
        # ffmpeg exits 0 when -fs stops the output, so the cap is limit + 1: a merged
        # file above the limit proves truncation, and one at or below it proves -fs
        # never triggered.
        result = self._run(ffmpeg_merge_arguments(
            str(video), None if audio is None else str(audio), str(output), limit + 1))
        if result.returncode != 0 or not _regular_file(output):
            raise _Fail(ReasonCode.MEDIA_PROBE_FAILED)
        if output.stat().st_size > limit:
            output.unlink()
            raise _Fail(ReasonCode.MEDIA_LIMIT_REACHED)
        for source in (video, audio):
            if source is not None:
                source.unlink(missing_ok=True)
        self._announce("merged_video")

    # ---- direct ---------------------------------------------------------

    def _download_direct(self) -> None:
        limits = self.limits
        url = str(self.request.source["input"])  # type: ignore[index]
        target = self._slot_path("downloaded_media")
        policy = direct_download.DownloadPolicy(
            byte_limit=limits.media_bytes, admitted_bytes=limits.media_bytes,
            overall_timeout_seconds=limits.download_timeout_seconds,
            read_timeout_seconds=min(60, limits.download_timeout_seconds),
            attempts=limits.download_attempts)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        try:
            descriptor = os.open(target, flags, 0o600)
        except FileExistsError:
            raise _Fail(ReasonCode.DESTINATION_CONFLICT) from None
        output = io.FileIO(descriptor, "w+b", closefd=True)
        try:
            result = asyncio.run(self._direct_with_progress(url, output, policy))
        finally:
            output.close()
        size = target.stat().st_size
        if size > limits.media_bytes:
            raise _Fail(ReasonCode.MEDIA_LIMIT_REACHED)
        if size != result.bytes_written:
            raise _Fail(ReasonCode.ARTIFACT_REJECTED)
        self._announce("downloaded_media")

    async def _direct_with_progress(self, url: str, output: io.FileIO,
                                    policy: direct_download.DownloadPolicy):
        limit = self.limits.media_bytes

        async def poll() -> None:
            # direct_download has no progress callback: poll the output size, <= 1 Hz.
            while True:
                await asyncio.sleep(1.0)
                with contextlib.suppress(OSError, ValueError):
                    self.emitter.progress(os.fstat(output.fileno()).st_size * 100.0 / max(limit, 1))

        poller = asyncio.create_task(poll())
        try:
            return await self.direct(url, output, policy, free_space=self.free_space)
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller


def _escape_glob(value: str) -> str:
    return "".join("[" + char + "]" if char in "*?[" else char for char in value)


def _selected_formats(info: Mapping) -> list[Mapping]:
    formats = info.get("requested_formats")
    if isinstance(formats, list) and formats:
        return [item for item in formats if isinstance(item, Mapping)]
    return [info]


def _estimate(selected: Sequence[Mapping]) -> int | None:
    total, known = 0, False
    for item in selected:
        size = _nonnegative_int(item.get("filesize"))
        if size is None:
            size = _nonnegative_int(item.get("filesize_approx"))
        if size is not None:
            total, known = total + size, True
    return total if known else None


def _parse_ffprobe(stdout: bytes) -> dict:
    try:
        document = json.loads(stdout.decode("utf-8", "strict"))
    except (ValueError, RecursionError):
        raise _Fail(ReasonCode.MEDIA_PROBE_FAILED) from None
    if not isinstance(document, dict):
        raise _Fail(ReasonCode.MEDIA_PROBE_FAILED)
    streams, container = document.get("streams"), document.get("format", {})
    if (not isinstance(streams, list) or not streams or not isinstance(container, dict)
            or any(not isinstance(item, dict) for item in streams)):
        raise _Fail(ReasonCode.MEDIA_PROBE_FAILED)

    duration = _probe_number(container.get("duration"))
    size = _probe_number(container.get("size"))
    audio_streams = sum(1 for item in streams if item.get("codec_type") == "audio")
    video_items = [item for item in streams if item.get("codec_type") == "video"
                   and not _attached_picture(item)]
    heights = [h for item in video_items for h in (_nonnegative_int(item.get("height")),) if h]
    return _metadata_record(
        "probe", None, duration_seconds=duration, has_audio=audio_streams > 0,
        has_video=bool(video_items), video_height=max(heights) if heights else None,
        audio_streams=audio_streams, video_streams=len(video_items),
        estimated_bytes=None if size is None else int(size))


def _attached_picture(stream: Mapping) -> bool:
    disposition = stream.get("disposition")
    return isinstance(disposition, Mapping) and disposition.get("attached_pic") == 1


def _probe_number(value: object) -> float | None:
    """None when absent; media_probe_failed for NaN, infinite or negative values."""
    if value is None or value == "N/A":
        return None
    if type(value) is str:
        try:
            value = float(value)
        except ValueError:
            raise _Fail(ReasonCode.MEDIA_PROBE_FAILED) from None
    number = _finite_number(value)
    if number is None or number < 0:
        raise _Fail(ReasonCode.MEDIA_PROBE_FAILED)
    return number


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

def run(request: WorkerRequest, sink: Callable[[bytes], object], *,
        scratch: Path = Path("/scratch"), input_root: Path = Path("/input"),
        ydl_factory: Callable[[dict], object] | None = None,
        run_process: Callable[[Sequence[str], float], subprocess.CompletedProcess] | None = None,
        direct: Callable[..., Awaitable[direct_download.DownloadResult]] = direct_download.download,
        free_space: Callable[[], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
        js_runtime_path: str | None = None) -> int:
    """Execute one request, emitting JSONL events to ``sink``; returns the exit code."""
    emitter = _Emitter(request, sink, clock)
    try:
        emitter.emit("started")
    except Exception:
        return 1
    job = _Job(request, emitter, scratch=scratch, input_root=input_root, ydl_factory=ydl_factory,
               run_process=run_process, direct=direct, free_space=free_space, clock=clock,
               js_runtime_path=js_runtime_path)
    failure: ReasonCode | None = None
    try:
        job.execute()
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        failure = _reason_for(error)
    # A pre-existing destination is host-owned state: never delete it.
    job.cleanup(failed=failure not in (None, ReasonCode.DESTINATION_CONFLICT))
    if failure is not None:
        with contextlib.suppress(Exception):
            emitter.error(failure)
        return 1
    try:
        emitter.emit("completed")
    except Exception:
        return 1
    return 0


def main(argv: Sequence[str] | None = None, *, stdin: BinaryIO | None = None,
         stdout: BinaryIO | None = None) -> int:
    """Read one request line, write JSONL events. Exit 0 only after ``completed``."""
    production = stdout is None
    source = sys.stdin.buffer if stdin is None else stdin
    line = source.readline(_MAX_REQUEST_BYTES)
    if len(line) >= _MAX_REQUEST_BYTES and not line.endswith(b"\n"):
        return 2
    try:
        request = WorkerRequest.from_json(line.rstrip(b"\r\n") if line.endswith(b"\n") else line)
    except DomainValidationError:
        return 2  # no trusted identity to attach events to
    if production:
        # Events go to a private duplicate of fd 1; stray writes to fd 1 are discarded.
        events = os.fdopen(os.dup(1), "wb", buffering=0)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.close(null)
    else:
        events = stdout

    def sink(data: bytes) -> None:
        events.write(data)
        events.flush()

    return run(request, sink)


if __name__ == "__main__":
    raise SystemExit(main())
