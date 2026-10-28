# ruff: noqa: BLE001
"""In-image offline ASR worker: whole-file faster-whisper transcription.

The worker speaks the version 1 worker JSONL protocol: one request line on
stdin, JSONL events on stdout, nothing else. It runs inside the ASR image only
(``python -I -m podmuncher.asr_worker``); the host never imports faster-whisper,
PyAV or ctranslate2, and this module imports them lazily inside production
defaults only.

Two operations are served. ``transcribe`` runs over the read-only
``/input/<basename>`` mount with the verified CTranslate2 snapshot at ``/model``.
The whole file is decoded once, transcribed once and the full result is written
as a single ``transcript_segments`` JSON artifact; nothing is streamed to the host
and no partial result is ever kept. The worker never fabricates or certifies a
measured envelope: it only records the measurements of this run. Diarization is
out of scope. Events never carry transcript text, URLs or host paths.

``probe`` takes no input and no model. It reports the in-image runtime (the
CTranslate2 version and the compute types it supports on this worker's device)
as one bounded ``probe_metadata`` JSON artifact, then stays alive for a bounded
hold so the host can sample the running container's memory before ``completed``.
It never touches the network, a model or an input file.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .domains import DomainValidationError, ReasonCode
from .protocol import WorkerEventRecord, WorkerRequest, emit_event

__all__ = (
    "MediaProbe",
    "MediaProbeError",
    "compute_type_for",
    "main",
    "probe_hold_seconds_for",
    "run",
    "transcribe_options",
)

SAMPLE_RATE = 16000
# float32 mono PCM at 16 kHz, plus the host's declared decode working-set rate
# (model_policy.DECODE_BYTES_PER_SECOND; not importable inside the image).
PCM_BYTES_PER_SECOND = 4 * SAMPLE_RATE
DECODE_BYTES_PER_SECOND = 256000
SCHEMA_VERSION = 1
_SLOT = "transcript_segments"
_TMP_NAME = ".asr-tmp"
_MAX_REQUEST_BYTES = 65537
_HASH_CHUNK = 1024 * 1024
_MAX_CPU_THREADS = 32
_REQUIRED_MODEL_FILES = ("model.bin", "config.json", "tokenizer.json")
_LANGUAGE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
_OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "TRANSFORMERS_OFFLINE": "1"}
_CGROUP_MEMORY_MAX = "/sys/fs/cgroup/memory.max"
_CGROUP_MEMORY_PEAK = "/sys/fs/cgroup/memory.peak"
_DEVICES = {"asr_cpu": "cpu", "asr_cuda": "cuda"}
_COMPUTE_TYPES = {"cpu": "int8", "cuda": "int8_float16"}
_PROBE_SLOT = "probe_metadata"
_PROBE_MAX_BYTES = 4096
_PROBE_HOLD_DEFAULT = 12.0
_PROBE_HOLD_MAX = 15.0
_PROBE_HOLD_STEP = 0.25
_MAX_COMPUTE_TYPES = 32
_COMPUTE_TYPE = re.compile(r"[a-z0-9_]{1,32}\Z")
_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}\Z")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")
_NVIDIA_GPUS = "/proc/driver/nvidia/gpus"


class _Fail(Exception):
    """A closed worker failure. Never carries model, media or filesystem text."""

    def __init__(self, reason: ReasonCode) -> None:
        super().__init__(reason.value)
        self.reason = reason


class MediaProbeError(Exception):
    """The input container cannot be opened or is corrupt."""


@dataclass(frozen=True, slots=True)
class MediaProbe:
    """Container facts read before decoding: audio stream presence and declared duration."""

    has_audio: bool
    duration_seconds: float | None


# --------------------------------------------------------------------------
# Public pure helpers
# --------------------------------------------------------------------------

def compute_type_for(device: str) -> str:
    """The fixed compute type: int8 on CPU, int8_float16 on CUDA."""
    try:
        return _COMPUTE_TYPES[device]
    except (KeyError, TypeError):
        raise ValueError("unsupported device") from None


def probe_hold_seconds_for(value: object) -> float:
    """The probe hold, clamped to [0, 15] seconds; a non-finite or non-number value is the default."""
    number = _finite_number(value)
    if number is None:
        return _PROBE_HOLD_DEFAULT
    return min(_PROBE_HOLD_MAX, max(0.0, number))


def transcribe_options(language: str) -> dict[str, object]:
    """Keyword arguments for ``WhisperModel.transcribe`` (auto language means detect)."""
    return {
        "language": None if language == "auto" else language,
        "task": "transcribe",
        "beam_size": 5,
        "vad_filter": True,
        "word_timestamps": False,
        "condition_on_previous_text": False,
    }


def _finite_number(value: object) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return float(value)


def _error_chain(error: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    stack: list[object] = [error]
    while stack and len(seen) < 16:
        item = stack.pop()
        if not isinstance(item, BaseException) or any(item is known for known in seen):
            continue
        seen.append(item)
        stack.extend((item.__cause__, item.__context__))
    return seen


def _is_oom(error: BaseException) -> bool:
    """MemoryError or CUDA/host out-of-memory text anywhere in the chain (text never forwarded)."""
    chain = _error_chain(error)
    return (any(isinstance(item, MemoryError) for item in chain)
            or any("out of memory" in str(item).lower() for item in chain))


def _reason_for(error: BaseException) -> ReasonCode:
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
    return ReasonCode.WORKER_EXIT_UNEXPECTED


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
        """At most one progress event per second (the host allows ten); never 100."""
        now = self._clock()
        if self._last_progress is not None and now - self._last_progress < 1.0:
            return
        self._last_progress = now
        self.emit("progress", {"percent": round(min(99.0, max(0.0, percent)), 2)})

    def error(self, reason: ReasonCode) -> None:
        self.emit("error", {"reason_code": reason.value,
                            "summary": reason.name.replace("_", " ").capitalize()})


# --------------------------------------------------------------------------
# Production defaults (heavy imports are lazy: the host venv has none of them)
# --------------------------------------------------------------------------

def _cpu_threads() -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(available, _MAX_CPU_THREADS))


def _default_model_factory(path: str, **options: object) -> object:
    from faster_whisper import WhisperModel  # lazy; env is already forced offline.
    return WhisperModel(path, **options)


def _default_decode(path: str, check: Callable[[int], None]) -> Sequence[float]:
    # Not faster_whisper.decode_audio: it calls av.open(metadata_errors=...), a keyword
    # PyAV 19 removed, and faster-whisper 1.2.1 (latest) has no fix. Same output contract:
    # 1-D float32, 16 kHz mono, int16 PCM scaled by 1/32768. ``check(samples_so_far)`` runs
    # after every resampled chunk so the caller can stop the decode at its caps.
    import av  # lazy: PyAV wheel bundles its codecs.
    import numpy as np

    container = av.open(path, mode="r")
    try:
        streams = [stream for stream in container.streams if stream.type == "audio"]
        if not streams:
            raise MediaProbeError
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        chunks: list = []
        total = 0
        for frame in container.decode(streams[0]):
            for item in resampler.resample(frame):
                chunk = item.to_ndarray().reshape(-1)
                chunks.append(chunk)
                total += len(chunk)
                check(total)
        for item in resampler.resample(None):
            chunk = item.to_ndarray().reshape(-1)
            chunks.append(chunk)
            total += len(chunk)
            check(total)
    finally:
        with contextlib.suppress(Exception):
            container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _default_probe(path: Path) -> MediaProbe:
    import av  # lazy: PyAV wheel bundles its codecs.

    try:
        container = av.open(str(path), mode="r")
    except Exception:
        raise MediaProbeError from None
    try:
        audio = [stream for stream in container.streams if stream.type == "audio"]
        duration: float | None = None
        if container.duration is not None:
            duration = container.duration / av.time_base
        elif audio and audio[0].duration is not None and audio[0].time_base is not None:
            duration = float(audio[0].duration * audio[0].time_base)
        return MediaProbe(bool(audio), duration)
    except Exception:
        raise MediaProbeError from None
    finally:
        with contextlib.suppress(Exception):
            container.close()


def _read_int(path: str) -> int | None:
    try:
        text = Path(path).read_text(encoding="ascii").strip()
        return int(text) if text.isdigit() else None
    except (OSError, ValueError, UnicodeError):
        return None


def _default_memory_limit() -> int | None:
    """cgroup v2 ``memory.max`` in bytes; ``max`` or unreadable means unknown."""
    return _read_int(_CGROUP_MEMORY_MAX)


def _default_peak_rss() -> int | None:
    """Largest of getrusage ru_maxrss and cgroup memory.peak; None when neither is readable."""
    peaks: list[int] = []
    with contextlib.suppress(Exception):
        import resource
        peaks.append(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)
    cgroup = _read_int(_CGROUP_MEMORY_PEAK)
    if cgroup is not None:
        peaks.append(cgroup)
    return max(peaks) if peaks else None


def _default_runtime_module() -> object:
    import ctranslate2  # lazy: the host venv has no ctranslate2.
    return ctranslate2


def _default_package_version(name: str) -> str | None:
    """Installed distribution version from metadata only; the package itself is never imported."""
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _default_cuda_uuid() -> str | None:
    """GPU UUID from the NVIDIA driver procfs (no NVML); None unless exactly one GPU is visible."""
    try:
        entries = sorted(Path(_NVIDIA_GPUS).iterdir())
        if len(entries) != 1:
            return None
        text = (entries[0] / "information").read_text(encoding="ascii", errors="strict")
    except (OSError, UnicodeError):
        return None
    for line in text.splitlines()[:64]:
        name, _, value = line.partition(":")
        if name.strip() == "GPU UUID":
            value = value.strip()
            return value if _GPU_UUID.fullmatch(value) else None
    return None


def _prepare_environment(tmp: Path) -> None:
    """Force offline mode and worker-owned writable dirs before faster_whisper is imported."""
    tmp.mkdir(mode=0o700)
    for name, value in _OFFLINE_ENV.items():
        os.environ[name] = value
    for name in ("HOME", "XDG_CACHE_HOME", "TMPDIR"):
        os.environ[name] = str(tmp)
    tempfile.tempdir = str(tmp)


# --------------------------------------------------------------------------
# Job execution
# --------------------------------------------------------------------------

class _Job:
    def __init__(self, request: WorkerRequest, emitter: _Emitter, *, scratch: Path,
                 input_root: Path, model_root: Path, model_factory, decode, probe,
                 memory_limit, peak_rss, clock: Callable[[], float], runtime_module=None,
                 package_version=None, cuda_uuid=None, sleep: Callable[[float], object] = time.sleep,
                 probe_hold_seconds: float = _PROBE_HOLD_DEFAULT) -> None:
        self.request, self.emitter = request, emitter
        self.scratch, self.input_root, self.model_root = scratch, input_root, model_root
        self.model_factory = model_factory or _default_model_factory
        self.decode = decode or _default_decode
        self.probe = probe or _default_probe
        self.memory_limit = memory_limit or _default_memory_limit
        self.peak_rss = peak_rss or _default_peak_rss
        self.clock = clock
        self.runtime_module = runtime_module or _default_runtime_module
        self.package_version = package_version or _default_package_version
        self.cuda_uuid = cuda_uuid or _default_cuda_uuid
        self.sleep = sleep
        self.probe_hold_seconds = probe_hold_seconds_for(probe_hold_seconds)
        self.limits = request.limits
        self.tmp = scratch / _TMP_NAME
        self.tmp_created = False
        self.output: Path | None = None
        self.written = False
        self.load_seconds = 0.0

    # ---- envelope -------------------------------------------------------

    def _check_envelope(self) -> str:
        request = self.request
        device = _DEVICES.get(request.capability)
        if request.operation == "probe":
            # No input, no model and no source; the device comes from the capability alone.
            if (device is None or request.selection.device not in ("auto", device)
                    or set(request.assigned_slots) != {_PROBE_SLOT}
                    or request.input_basename is not None
                    or request.source is not None):
                raise _Fail(ReasonCode.WORKER_PROTOCOL_MISMATCH)
            return device
        if (request.operation != "transcribe" or device is None
                or request.selection.device != device
                or set(request.assigned_slots) != {_SLOT}
                or request.model_snapshot_id is None
                or request.selection.model == "auto"
                or request.input_basename is None
                or request.source is not None):
            raise _Fail(ReasonCode.WORKER_PROTOCOL_MISMATCH)
        return device

    # ---- steps ----------------------------------------------------------

    def _verify_model(self) -> None:
        root = self.model_root
        if not root.is_dir() or root.is_symlink():
            raise _Fail(ReasonCode.ASSET_VERIFICATION_FAILED)
        for name in _REQUIRED_MODEL_FILES:
            if not _regular_file(root / name):
                raise _Fail(ReasonCode.ASSET_VERIFICATION_FAILED)
        vocabularies = sorted(root.glob("vocabulary.*"))
        if not vocabularies or not all(_regular_file(item) for item in vocabularies):
            raise _Fail(ReasonCode.ASSET_VERIFICATION_FAILED)

    def _load(self, device: str, compute_type: str) -> object:
        if not self.tmp_created:
            _prepare_environment(self.tmp)
            self.tmp_created = True
        started = self.clock()
        try:
            model = self.model_factory(
                str(self.model_root), device=device, compute_type=compute_type,
                local_files_only=True, cpu_threads=_cpu_threads(), num_workers=1)
        except Exception as error:
            if _is_oom(error):
                raise _Fail(ReasonCode.MODEL_OOM) from None
            raise _Fail(ReasonCode.MODEL_LOAD_FAILED) from None
        self.load_seconds = self.clock() - started
        return model

    def _check_memory(self, seconds: float) -> None:
        limit = self.memory_limit()
        if limit is None:
            return
        if seconds * PCM_BYTES_PER_SECOND + seconds * DECODE_BYTES_PER_SECOND > limit:
            raise _Fail(ReasonCode.RAM_INSUFFICIENT)

    def _inspect_input(self, path: Path) -> float | None:
        """Container probe before decoding; returns the declared duration or None."""
        if not _regular_file(path):
            raise _Fail(ReasonCode.MEDIA_PROBE_FAILED)
        try:
            info = self.probe(path)
        except MediaProbeError:
            raise _Fail(ReasonCode.MEDIA_PROBE_FAILED) from None
        except Exception:
            raise _Fail(ReasonCode.MEDIA_PROBE_FAILED) from None
        if not info.has_audio:
            raise _Fail(ReasonCode.NO_AUDIO)
        declared = _finite_number(info.duration_seconds)
        if declared is not None and declared <= 0:
            declared = None
        if declared is not None:
            if declared > self.limits.duration_seconds:
                raise _Fail(ReasonCode.DURATION_EXCEEDED)
            self._check_memory(declared)
        return declared

    def _decode(self, path: Path, declared: float | None,
                deadline: float) -> tuple[object, float]:
        def check(samples: int) -> None:
            if samples > self.limits.duration_seconds * SAMPLE_RATE:
                raise _Fail(ReasonCode.DURATION_EXCEEDED)
            if declared is None or samples / SAMPLE_RATE > declared:
                self._check_memory(samples / SAMPLE_RATE)
            if self.clock() > deadline:
                raise _Fail(ReasonCode.REQUEST_TIMEOUT)

        try:
            audio = self.decode(str(path), check)
            samples = len(audio)
        except MemoryError:
            raise _Fail(ReasonCode.MODEL_OOM) from None
        except _Fail:
            raise
        except Exception as error:
            if _is_oom(error):
                raise _Fail(ReasonCode.MODEL_OOM) from None
            raise _Fail(ReasonCode.MEDIA_PROBE_FAILED) from None
        if samples <= 0:
            raise _Fail(ReasonCode.NO_AUDIO)
        duration = samples / SAMPLE_RATE
        if duration > self.limits.duration_seconds:
            raise _Fail(ReasonCode.DURATION_EXCEEDED)
        if declared is None or duration > declared:
            self._check_memory(duration)
        return audio, duration

    def _transcribe(self, model: object, audio: object, duration: float,
                    deadline: float) -> tuple[list[tuple[float, float, object]], object]:
        if self.clock() > deadline:
            raise _Fail(ReasonCode.REQUEST_TIMEOUT)
        options = transcribe_options(self.request.selection.language)
        collected: list[tuple[float, float, object]] = []
        try:
            segments, info = model.transcribe(audio, **options)  # type: ignore[attr-defined]
            for segment in segments:
                collected.append((segment.start, segment.end, segment.text))
                end = _finite_number(segment.end)
                if end is not None and duration > 0:
                    self.emitter.progress(min(99.0, end / duration * 100.0))
                if self.clock() > deadline:
                    raise _Fail(ReasonCode.REQUEST_TIMEOUT)
        except _Fail:
            raise
        except Exception as error:
            collected.clear()  # never reuse a partial transcript
            if _is_oom(error):
                raise _Fail(ReasonCode.MODEL_OOM) from None
            raise _Fail(ReasonCode.WORKER_EXIT_UNEXPECTED) from None
        return collected, info

    @staticmethod
    def _validate(raw: Sequence[tuple[float, float, object]],
                  duration: float) -> list[dict[str, object]]:
        # Whisper can place a final segment slightly past the decoded audio (300.96 s on
        # 300.0 s). Times past the duration are clamped to it rather than discarding the
        # whole transcript; a segment wholly past the end is kept at [duration, duration].
        # Non-finite, negative, reversed or non-monotonic times are still rejected.
        valid: list[dict[str, object]] = []
        previous = 0.0
        for start, end, text in raw:
            start, end = _finite_number(start), _finite_number(end)
            if (start is None or end is None or type(text) is not str
                    or not 0 <= start <= end or start < previous):
                raise _Fail(ReasonCode.ARTIFACT_REJECTED)
            previous = start
            text = text.strip()
            if text:
                valid.append({"start_seconds": min(start, duration),
                              "end_seconds": min(end, duration), "text": text})
        return valid

    def _language(self, info: object) -> tuple[str, float | None]:
        detected = getattr(info, "language", None)
        requested = self.request.selection.language
        language = detected if detected is not None else (None if requested == "auto" else requested)
        if type(language) is not str or _LANGUAGE.fullmatch(language) is None:
            raise _Fail(ReasonCode.ARTIFACT_REJECTED)
        probability = _finite_number(getattr(info, "language_probability", None))
        return language, probability

    def _write(self, record: dict[str, object], slot: str = _SLOT,
               max_bytes: int | None = None) -> None:
        try:
            data = json.dumps(record, separators=(",", ":"), allow_nan=False,
                              ensure_ascii=False).encode("utf-8")
        except (ValueError, UnicodeError):
            raise _Fail(ReasonCode.ARTIFACT_REJECTED) from None
        if len(data) > self.limits.export_bytes or (max_bytes is not None and len(data) > max_bytes):
            raise _Fail(ReasonCode.ARTIFACT_REJECTED)
        target = self.scratch / self.request.assigned_slots[slot]
        try:
            handle = target.open("xb")
        except FileExistsError:
            raise _Fail(ReasonCode.DESTINATION_CONFLICT) from None
        self.output = target
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.written = True
        # Completion is only emitted after this flush: claim the artifact from the bytes on disk.
        size, digest = _sha256(target)
        self.emitter.emit("artifact", {"slot": slot, "bytes": size, "sha256": digest})

    # ---- probe ----------------------------------------------------------

    def _runtime_facts(self, device: str) -> tuple[str, list[str]]:
        """CTranslate2 version and supported compute types; any doubt is envelope_unvalidated."""
        try:
            module = self.runtime_module()
            version = getattr(module, "__version__", None)
            if device == "cuda":
                # Never claim CUDA unless the runtime itself sees a device.
                count = module.get_cuda_device_count()  # type: ignore[attr-defined]
                if type(count) is not int or count < 1:
                    raise _Fail(ReasonCode.ENVELOPE_UNVALIDATED)
            reported = list(module.get_supported_compute_types(device))  # type: ignore[attr-defined]
        except _Fail:
            raise
        except Exception:
            raise _Fail(ReasonCode.ENVELOPE_UNVALIDATED) from None
        if type(version) is not str or _VERSION.fullmatch(version) is None:
            raise _Fail(ReasonCode.ENVELOPE_UNVALIDATED)
        if (not reported or len(reported) > _MAX_COMPUTE_TYPES
                or any(type(item) is not str or _COMPUTE_TYPE.fullmatch(item) is None
                       for item in reported)):
            raise _Fail(ReasonCode.ENVELOPE_UNVALIDATED)
        return version, sorted(set(reported))

    def _optional_version(self, name: str) -> str | None:
        try:
            version = self.package_version(name)
        except Exception:
            return None
        return version if type(version) is str and _VERSION.fullmatch(version) else None

    def _optional_cuda_uuid(self) -> str | None:
        try:
            value = self.cuda_uuid()
        except Exception:
            return None
        return value if type(value) is str and _GPU_UUID.fullmatch(value) else None

    def _hold(self) -> None:
        """Stay alive for the bounded hold in short sleeps; the step count is capped too."""
        end = self.clock() + self.probe_hold_seconds
        for _ in range(math.ceil(_PROBE_HOLD_MAX / _PROBE_HOLD_STEP) + 1):
            remaining = end - self.clock()
            if remaining <= 0:
                return
            self.sleep(min(_PROBE_HOLD_STEP, remaining))

    def probe_runtime(self, device: str) -> None:
        if not self.tmp_created:
            _prepare_environment(self.tmp)
            self.tmp_created = True
        version, compute_types = self._runtime_facts(device)
        record = {
            "schema_version": SCHEMA_VERSION,
            "runtime": {"name": "ctranslate2", "version": version},
            "compute_types": compute_types,
            "device": device,
            "cuda_device_uuid": self._optional_cuda_uuid() if device == "cuda" else None,
            "faster_whisper_version": self._optional_version("faster-whisper"),
        }
        self._write(record, _PROBE_SLOT, _PROBE_MAX_BYTES)
        self._hold()

    # ---- transcribe -----------------------------------------------------

    def execute(self) -> None:
        request = self.request
        device = self._check_envelope()
        if request.operation == "probe":
            self.probe_runtime(device)
            return
        compute_type = compute_type_for(device)
        self._verify_model()
        path = self.input_root / str(request.input_basename)
        declared = self._inspect_input(path)
        _, source_sha256 = _sha256(path)
        model = self._load(device, compute_type)
        try:
            deadline = self.clock() + float(self.limits.decode_timeout_seconds)
            audio, duration = self._decode(path, declared, deadline)
            began = self.clock()
            raw, info = self._transcribe(model, audio, duration, deadline)
            transcribe_seconds = self.clock() - began
        finally:
            del model
        segments = self._validate(raw, duration)
        language, probability = self._language(info)
        peak_rss = self.peak_rss()
        record = {
            "schema_version": SCHEMA_VERSION,
            "language": language,
            "language_probability": probability,
            "duration_seconds": duration,
            "no_speech_detected": not segments,
            "segments": segments,
            "provenance": {
                "model_id": request.selection.model,
                "model_snapshot_id": request.model_snapshot_id,
                "device": device,
                "compute_type": compute_type,
                "source_sha256": source_sha256,
                "attempt_id": request.attempt_id,
                "image_digest": None,  # stamped by the host
            },
            "measurements": {
                "load_seconds": self.load_seconds,
                "transcribe_wall_seconds": transcribe_seconds,
                "audio_seconds": duration,
                "peak_rss_bytes": None if peak_rss is None else int(peak_rss),
                "peak_vram_bytes": None,
            },
        }
        self._write(record)

    def cleanup(self, *, failed: bool) -> None:
        if self.tmp_created:
            shutil.rmtree(self.tmp, ignore_errors=True)
        # Only a file this worker created is removed; a pre-existing destination is host state.
        if failed and self.output is not None:
            with contextlib.suppress(OSError):
                self.output.unlink()


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

def run(request: WorkerRequest, sink: Callable[[bytes], object], *,
        scratch: Path = Path("/scratch"), input_root: Path = Path("/input"),
        model_root: Path = Path("/model"),
        model_factory: Callable[..., object] | None = None,
        decode: Callable[[str, Callable[[int], None]], Sequence[float]] | None = None,
        probe: Callable[[Path], MediaProbe] | None = None,
        memory_limit: Callable[[], int | None] | None = None,
        peak_rss: Callable[[], int | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        runtime_module: Callable[[], object] | None = None,
        package_version: Callable[[str], str | None] | None = None,
        cuda_uuid: Callable[[], str | None] | None = None,
        sleep: Callable[[float], object] = time.sleep,
        probe_hold_seconds: float = _PROBE_HOLD_DEFAULT) -> int:
    """Execute one request, emitting JSONL events to ``sink``; returns the exit code."""
    emitter = _Emitter(request, sink, clock)
    try:
        emitter.emit("started")
    except Exception:
        return 1
    job = _Job(request, emitter, scratch=scratch, input_root=input_root, model_root=model_root,
               model_factory=model_factory, decode=decode, probe=probe, memory_limit=memory_limit,
               peak_rss=peak_rss, clock=clock, runtime_module=runtime_module,
               package_version=package_version, cuda_uuid=cuda_uuid, sleep=sleep,
               probe_hold_seconds=probe_hold_seconds)
    failure: ReasonCode | None = None
    try:
        job.execute()
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        failure = ReasonCode.MODEL_OOM if isinstance(error, MemoryError) else _reason_for(error)
    job.cleanup(failed=failure is not None and failure is not ReasonCode.DESTINATION_CONFLICT)
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
