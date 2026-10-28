"""Real end-to-end product runs (P6-U3): the actual CLI, composed daemon and Docker workers.

Every case here is ``@pytest.mark.live``.  The cases share ONE ``podmuncher`` service (the
real console script as a subprocess) with tiny temporary roots, ONE new empty cache root, an
explicit development runtime manifest (real local image digests, the real in-image runtime
version, the real pinned tiny model with HTTPS sources) and short generated fixtures; the
CPU jobs are submitted back to back and each case asserts its own job.  No user recording
is ever read.  An unmet prerequisite fails the case with a clear message; only
the opt-in NVIDIA case skips, and it is never counted by the CPU smoke summary.

Helpers are test-local (stdlib plus already-created ``podmuncher`` modules only):

* ``make_media_fixture(kind, seconds, directory)`` writes a generated WAV with stdlib or
  an MP4 (tone + test pattern, or silent) with ffmpeg inside the local media image.
* ``LiveBatch`` / ``controlled_run(...)`` start the service through the CLI and always
  stop it, kill a stray daemon, prove no owned container or process remains and remove
  every temporary root.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The repo venv's console script: ``Scripts\podmuncher.exe`` on Windows, ``bin/podmuncher`` on
# POSIX (the Mac test venv is installed with ``uv pip install -e .``).
CONSOLE = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
    "podmuncher.exe" if os.name == "nt" else "podmuncher")
ENV_NAMES = ("PODMUNCHER_TEST_MEDIA_IMAGE", "PODMUNCHER_TEST_ASR_IMAGE",
             "PODMUNCHER_TEST_MODEL_ROOT", "PODMUNCHER_TEST_DIRECT_URL",
             "PODMUNCHER_TEST_YOUTUBE_URL")
MIB, GIB = 1024**2, 1024**3
# The whole shared run (fixtures, every job, cancel) must end well inside the 600 s smoke.
BATCH_BUDGET_SECONDS = 540.0
TERMINAL = frozenset({"completed", "partial", "failed", "cancelled", "blocked", "interrupted"})
TRANSCRIPT_FILES = {
    "transcript_txt": "transcript.txt",
    "transcript_timestamped_txt": "transcript.timestamped.txt",
    "transcript_readable_txt": "transcript.readable.txt",
    "transcript_srt": "transcript.srt",
    "transcript_json": "transcript.json",
}
SIDECAR = "podmuncher-manifest.json"
# The pinned fixture/smoke model (SPEC_GAP-4 d): Systran/faster-whisper-tiny, MIT.
FIXTURE_MODEL_ID = "fixture-faster-whisper-tiny"
FIXTURE_REVISION = "d90ca5fe260221311c53c58e660288d3deb8d356"
FIXTURE_FILES = (
    ("config.json", 2249, "a73a28cdfe1c43ccc7202fa333d1f89c202477271407ae9a7f19afa52039cac8"),
    ("model.bin", 75538270, "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1"),
    ("tokenizer.json", 2203239, "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
    ("vocabulary.txt", 459861, "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"),
)
MODEL_ADMISSION_BYTES = 160 * MIB
# The development manifest names the real (synthetic: false) pinned model "tiny".
MANIFEST_MODEL_ID = "tiny"
# Three real transcriptions only (600 s contract smoke): local audio (which also proves the
# new cache, offline inference and the real model download), direct HTTPS video+transcript
# (the remote-combined copy path) and YouTube.  Local video+transcript is covered
# deterministically by tests/test_failure_matrix.py::test_outcome_matrix[combined_ok].
CPU_CASES = ("local_audio_transcript", "new_cache_offline", "direct_https_combined",
             "youtube_transcript", "local_video_only", "silent_video_combined",
             "detach_cancel")
_PASSED: dict[str, dict[str, Any]] = {}


# --------------------------------------------------------------------------- prerequisites


def require_env() -> dict[str, str]:
    """The five declared live variables; a missing one fails (names only, never values)."""
    missing = [name for name in ENV_NAMES if not os.environ.get(name)]
    assert not missing, ("live prerequisite missing: set " + ", ".join(missing)
                         + " (load only these names from ~/.foreman-mcp/.env); the live "
                         "end-to-end cases never skip")
    assert CONSOLE.exists(), f"live prerequisite missing: console script {CONSOLE.name}"
    return {name: os.environ[name].strip() for name in ENV_NAMES}


def source_kind(value: str) -> str | None:
    """The CLI's own classification; failures never echo the (secret-free but private) URL."""
    from podmuncher.requests import classify_source

    try:
        return classify_source(value).kind
    except Exception:  # noqa: BLE001 - only the kind matters, never the message
        return None


@dataclass
class Prerequisites:
    """Admitted prerequisites resolved once, outside the per-case timings."""

    endpoint: str
    media_image: str
    media_bytes: int
    asr_image: str
    asr_bytes: int
    runtime_version: str
    model_root: Path
    snapshot_id: str
    snapshot_path: Path
    direct_url: str = field(repr=False)
    youtube_url: str = field(repr=False)
    seconds: float = 0.0


def docker(endpoint: str, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(("docker", "--host", endpoint, *args), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, check=False,
                          stdin=subprocess.DEVNULL)


def fixture_model() -> Any:
    """The pinned fixture model as ``ModelCache`` acquires it (tests/test_asr_worker shape)."""
    from podmuncher.domains import ModelTier
    from podmuncher.manifests import ManifestFile, ManifestModel
    from podmuncher.model_policy import DECODE_BYTES_PER_SECOND, ModelArtifact

    base = ("https://huggingface.co/Systran/faster-whisper-tiny/resolve/"
            + FIXTURE_REVISION + "/")
    return ManifestModel(
        ModelArtifact(FIXTURE_MODEL_ID, ModelTier.TINY, FIXTURE_REVISION, FIXTURE_FILES[1][2]),
        tuple(base + name for name, _, _ in FIXTURE_FILES),
        tuple(ManifestFile(name, size, digest) for name, size, digest in FIXTURE_FILES),
        "MIT", "fixture-smoke-asset-not-a-production-envelope", ("int8",),
        1, 0, 1, DECODE_BYTES_PER_SECOND, 1, ("wav",), "2" * 64, False, False)


def acquire_fixture_snapshot(model_root: Path) -> Any:
    """Admitted ``ModelCache`` acquisition; a cached snapshot is re-verified, not refetched."""
    from podmuncher.model_cache import ModelCache

    class _Audit:
        def record(self, **values: Any) -> None:
            pass

    cache = ModelCache(model_root, audit=_Audit(),
                       admission=lambda **v: v["declared_bytes"] <= MODEL_ADMISSION_BYTES
                       and v["allocated_bytes"] <= MODEL_ADMISSION_BYTES)
    try:
        return cache.acquire(fixture_model())
    finally:
        cache.close()


def resolve_prerequisites() -> Prerequisites:
    """Images present locally (never pulled or built here), runtime version, model snapshot."""
    began = time.monotonic()
    values = require_env()
    from podmuncher import bootstrap
    from podmuncher.requests import classify_source

    assert classify_source(values["PODMUNCHER_TEST_DIRECT_URL"]).kind == "direct", \
        "PODMUNCHER_TEST_DIRECT_URL is not a direct HTTPS source"
    assert classify_source(values["PODMUNCHER_TEST_YOUTUBE_URL"]).kind == "youtube", \
        "PODMUNCHER_TEST_YOUTUBE_URL is not a YouTube source"
    target = bootstrap.resolve_docker_target()
    images = {}
    for name in ("PODMUNCHER_TEST_MEDIA_IMAGE", "PODMUNCHER_TEST_ASR_IMAGE"):
        inspected = docker(target.endpoint, "image", "inspect", "--format", "{{.Id}} {{.Size}}",
                           values[name])
        assert inspected.returncode == 0, (
            f"live prerequisite missing: the {name} image is not present locally; build it "
            "with the tests/test_media_worker.py or tests/test_asr_worker.py live smoke first "
            "(nothing is pulled or built by this smoke)")
        image_id, size = inspected.stdout.split()
        images[name] = (image_id, int(size))
    version = docker(target.endpoint, "run", "--rm", "--pull", "never", "--network", "none",
                     "--entrypoint", "python", images["PODMUNCHER_TEST_ASR_IMAGE"][0], "-I", "-c",
                     "import ctranslate2; print(ctranslate2.__version__)")
    assert version.returncode == 0 and version.stdout.strip(), \
        "could not read the ctranslate2 runtime version from the ASR image"
    model_root = Path(values["PODMUNCHER_TEST_MODEL_ROOT"]).resolve()
    assert ROOT not in (model_root, *model_root.parents), "the model root must be outside the repo"
    model_root.mkdir(parents=True, exist_ok=True)
    snapshot = acquire_fixture_snapshot(model_root)
    assert sorted(p.name for p in snapshot.path.iterdir()) == sorted(n for n, _, _ in FIXTURE_FILES)
    return Prerequisites(
        target.endpoint, *images["PODMUNCHER_TEST_MEDIA_IMAGE"],
        *images["PODMUNCHER_TEST_ASR_IMAGE"], version.stdout.strip(), model_root,
        snapshot.snapshot_id, Path(snapshot.path), values["PODMUNCHER_TEST_DIRECT_URL"],
        values["PODMUNCHER_TEST_YOUTUBE_URL"], time.monotonic() - began)


@pytest.fixture(scope="module")
def prereq() -> Prerequisites:
    return resolve_prerequisites()


# ------------------------------------------------------------------------------- fixtures


def _write_wav(path: Path, seconds: float, *, rate: int = 16000) -> None:
    """Speech-like generated signal: a voiced harmonic stack with syllable-rate envelope."""
    frames = bytearray()
    total = int(seconds * rate)
    for index in range(total):
        t = index / rate
        pitch = 120 + 30 * math.sin(2 * math.pi * 0.7 * t)
        envelope = max(0.0, math.sin(2 * math.pi * 3.0 * t)) ** 2 if 0.3 < t < seconds - 0.3 else 0.0
        voiced = sum(math.sin(2 * math.pi * pitch * k * t) / k for k in range(1, 6))
        value = int(6000 * envelope * voiced / 2.3)
        frames += max(-32768, min(32767, value)).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))


def make_media_fixture(kind: str, seconds: float, directory: Path, *,
                       prereq: Prerequisites | None = None) -> Path:
    """Generate one short fixture: ``wav`` (stdlib), ``mp4`` (tone + pattern) or ``silent_mp4``.

    MP4 fixtures are encoded by ffmpeg inside the local media image (network none, no pull,
    removed on exit); the only mount is the fixture directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if kind == "wav":
        path = directory / "fixture.wav"
        _write_wav(path, seconds)
        return path
    assert kind in ("mp4", "silent_mp4"), kind
    assert prereq is not None, "MP4 fixtures need the media image"
    name = "fixture.mp4" if kind == "mp4" else "silent.mp4"
    args = ["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
            f"testsrc=size=160x120:rate=10:duration={seconds}"]
    if kind == "mp4":
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={seconds}",
                 "-c:a", "aac", "-shortest"]
    else:
        args += ["-an"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", f"/out/{name}"]
    made = docker(prereq.endpoint, "run", "--rm", "--pull", "never", "--network", "none",
                  "--entrypoint", "ffmpeg", "-v", f"{directory}:/out", prereq.media_image, *args,
                  timeout=120)
    path = directory / name
    assert made.returncode == 0 and path.is_file() and path.stat().st_size > 0, \
        "ffmpeg fixture generation failed: " + made.stderr[-400:]
    return path


# --------------------------------------------------------------------- manifest and config


def _capability(cap: str, image: str, size: int, runtime_version: str, memory: int,
                module: str) -> dict:
    return {
        "id": cap, "platform": "fixture-platform", "image_reference": "fixture-local",
        "digest": image, "entrypoint": ["python", "-I", "-m", module],
        "compressed_bytes": size, "unpacked_peak_bytes": size,
        "worker_overhead_bytes": 256 * MIB + 2 * 10 * MIB, "worker_memory_bytes": memory,
        "dependency_lock_sha256": "2" * 64, "licenses": ["fixture-license"],
        "provenance": "fixture:p6u3-local-image", "runtime_version": runtime_version,
        "driver_version": None, "hardware_fingerprint": "fixture-hardware",
    }


def make_manifest(directory: Path, prereq: Prerequisites) -> Path:
    """Development manifest naming the real local image digests, runtime and a real model.

    The model is the pinned tiny snapshot of ``fixture_model()`` declared as a real
    (``synthetic: false``) development model: one pinned HTTPS source per file with its
    byte size and sha256, so the composed daemon acquires it through the real ModelCache
    into the run's new, empty cache root.  No model file is copied next to the manifest.
    """
    directory.mkdir(parents=True, exist_ok=True)
    model = fixture_model()
    declared = [{"relative_path": f.relative_path, "bytes": f.bytes, "sha256": f.sha256}
                for f in model.files]
    doc = {
        "schema_version": 1, "release_version": "fixture",
        "source": {"kind": "development_fixture", "label": "fixture",
                   "provenance": "fixture:test_end_to_end"},
        "platforms": ["fixture-platform"],
        "capabilities": [
            _capability("media", prereq.media_image, prereq.media_bytes, "media-dev-local",
                        GIB, "podmuncher.media_worker"),
            _capability("asr_cpu", prereq.asr_image, prereq.asr_bytes, prereq.runtime_version,
                        3 * GIB, "podmuncher.asr_worker"),
        ],
        "models": [{
            "id": MANIFEST_MODEL_ID, "tier": "tiny", "revision": FIXTURE_REVISION,
            "digest": FIXTURE_FILES[1][2],
            "source_urls": list(model.source_urls),
            "files": declared, "license": "MIT", "provenance": "fixture:faster-whisper-tiny",
            "compute_types": ["int8"], "worker_peak_ram_bytes": 512 * MIB,
            "worker_peak_vram_bytes": 0, "validated_max_duration_seconds": 3600,
            "decode_bytes_per_second": 256000, "decode_timeout_seconds": 300,
            "codec_matrix": ["wav", "aac", "opus"], "evidence_sha256": "4" * 64,
            "release_tested": True, "synthetic": False,
        }],
        "languages": ["en"],
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _toml(path: Path) -> str:
    return "'" + str(path) + "'"


@dataclass
class Roots:
    base: Path
    data: Path
    output: Path
    cache: Path
    scratch: Path
    state: Path
    manifest: Path
    config: Path
    inputs: Path
    env: dict[str, str] = field(default_factory=dict, repr=False)


def make_roots(base: Path, prereq: Prerequisites, *, cache: Path | None = None) -> Roots:
    """Tiny temporary roots, the fixture manifest and a config TOML with bounded limits."""
    paths = {name: base / name for name in ("data", "output", "cache", "scratch", "state",
                                            "inputs")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    cache_root = cache or paths["cache"]
    manifest = make_manifest(base / "manifest", prereq)
    config = base / "config.toml"
    config.write_text("\n".join([
        "schema_version = 1", "[paths]",
        f"data = {_toml(paths['data'])}", f"output = {_toml(paths['output'])}",
        f"cache = {_toml(cache_root)}", f"scratch = {_toml(paths['scratch'])}",
        "[limits]", f"media_bytes = {256 * MIB}", "duration_seconds = 3600",
        f"export_bytes = {8 * MIB}", "download_timeout_seconds = 300",
        "decode_timeout_seconds = 300", "",
    ]), encoding="utf-8")
    env = os.environ.copy()
    for name in ENV_NAMES:
        env.pop(name, None)  # the daemon never needs the fixture URLs or roots
    env.update(PODMUNCHER_STATE_DIR=str(paths["state"]), PODMUNCHER_CONFIG=str(config),
               PODMUNCHER_RUNTIME_MANIFEST=str(manifest), PYTHONDONTWRITEBYTECODE="1",
               TERM="dumb")
    return Roots(base, paths["data"], paths["output"], cache_root, paths["scratch"],
                 paths["state"], manifest, config, paths["inputs"], env)


# ------------------------------------------------------------------------- container spy


@dataclass
class SeenContainer:
    container_id: str
    image: str
    network: str
    readonly: bool
    instance: str
    job: str


class ContainerSpy:
    """Polls owned (``podmuncher.instance``-labelled) containers and records their config."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.seen: dict[str, SeenContainer] = {}
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            # A transient host failure (e.g. WinError 1455 commit exhaustion) must not end
            # the spy: count it and keep polling.
            try:
                self._poll()
            except (OSError, subprocess.SubprocessError):
                self.errors += 1
            self._stop.wait(0.5)

    def _poll(self) -> None:
        listed = docker(self.endpoint, "ps", "-a", "--no-trunc", "--format", "{{.ID}}",
                        "--filter", "label=podmuncher.instance", timeout=30)
        for cid in listed.stdout.split():
            if cid in self.seen:
                continue
            raw = docker(self.endpoint, "container", "inspect", cid, timeout=30)
            if raw.returncode != 0:
                continue
            with contextlib.suppress(ValueError, KeyError, IndexError):
                data = json.loads(raw.stdout)[0]
                self.seen[cid] = SeenContainer(
                    cid, data["Image"], data["HostConfig"]["NetworkMode"],
                    bool(data["HostConfig"]["ReadonlyRootfs"]),
                    (data["Config"].get("Labels") or {}).get("podmuncher.instance", ""),
                    (data["Config"].get("Labels") or {}).get("podmuncher.job", ""))

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(30)

    def images(self, job_id: str | None = None) -> list[str]:
        return [c.image for c in list(self.seen.values()) if job_id in (None, c.job)]


def owned_containers(endpoint: str) -> list[str]:
    listed = docker(endpoint, "ps", "-a", "--no-trunc", "--format", "{{.ID}}", "--filter",
                    "label=podmuncher.instance", timeout=60)
    assert listed.returncode == 0, "docker ps failed"
    return listed.stdout.split()


# ---------------------------------------------------------------------------- controlled run


@dataclass
class Result:
    code: int
    doc: dict
    seconds: float


class Runner:
    """One controlled service run: every command is the real console script."""

    def __init__(self, roots: Roots, prereq: Prerequisites, spy: ContainerSpy) -> None:
        self.roots, self.prereq, self.spy = roots, prereq, spy
        self.pid: int | None = None
        self.timings: dict[str, float] = {}

    def cli(self, *args: str, timeout: float = 120) -> Result:
        began = time.monotonic()
        done = subprocess.run([str(CONSOLE), *args, "--json"], cwd=self.roots.base,
                              env=self.roots.env, capture_output=True, timeout=timeout,
                              check=False, stdin=subprocess.DEVNULL)
        text = done.stdout.decode("utf-8", "replace").strip()
        try:
            doc = json.loads(text.splitlines()[-1]) if text else {}
        except ValueError:
            doc = {"unparsed": text[-400:]}
        return Result(done.returncode, doc, time.monotonic() - began)

    def submit(self, source: str, *flags: str) -> str:
        self.timings["submit_at"] = time.monotonic()
        reply = self.cli(source, *flags, "--detach")
        assert reply.code == 0 and reply.doc.get("job_id"), reply.doc
        return reply.doc["job_id"]

    def wait(self, job_id: str, timeout: float = 240.0,
             states: frozenset[str] = TERMINAL) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            job = self.cli("job", job_id).doc.get("job") or {}
            if job.get("state") in states:
                self.timings["job"] = round(time.monotonic() - self.timings.get("submit_at", 0), 1)
                return job
            assert time.monotonic() < deadline, (
                f"job {job_id} stuck in {job.get('state')} ({job.get('reason_code')})")
            time.sleep(0.5)

    def job_dir(self, job_id: str) -> Path:
        return self.roots.output / f"podmuncher-{job_id}"


def _kill(pid: int | None) -> None:
    if pid is None:
        return
    import psutil

    with contextlib.suppress(psutil.Error):
        process = psutil.Process(pid)
        process.kill()
        process.wait(15)


# Ids, closed event/action names and reason codes only: never messages, paths or URLs.
_DIAGNOSTIC_KEYS = ("ts", "event_name", "event", "name", "action", "component", "stage",
                    "reason_code", "job_id", "attempt_id")
_POLLING_ACTIONS = frozenset({"jobs_get", "access", "identity", "snapshot", "jobs_list"})


def _flatten(item: dict) -> dict:
    """One flat view of a log/trace/audit line (attributes and audit ``ids`` merged in)."""
    merged: dict = {}
    for nested in ("attributes", "ids"):
        value = item.get(nested)
        if isinstance(value, dict):
            merged.update({str(k).removeprefix("fields."): v for k, v in value.items()})
    merged.update({k: v for k, v in item.items() if k not in ("attributes", "ids")})
    return merged


def print_diagnostics(state: Path, job_id: str | None = None, limit: int = 40) -> None:
    """The service's own audit and diagnostic records for a failing job.

    With ``job_id`` only records carrying that job id (or one of its attempt ids) are
    printed; each record is reduced to ids, closed event/action names and reason codes.
    """
    for name in ("audit.jsonl", "logs.jsonl", "traces.jsonl"):
        path = state / "diagnostics" / name
        if not path.is_file():
            continue
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-4000:]:
            with contextlib.suppress(ValueError, AttributeError, TypeError):
                rows.append(_flatten(json.loads(line)))
        attempts = {str(r.get("attempt_id")) for r in rows
                    if job_id and r.get("job_id") == job_id and r.get("attempt_id")}
        kept = []
        for row in rows:
            if row.get("action") in _POLLING_ACTIONS:
                continue
            if job_id and row.get("job_id") != job_id \
                    and str(row.get("attempt_id")) not in attempts:
                continue
            entry = {k: row[k] for k in _DIAGNOSTIC_KEYS
                     if k in row and isinstance(row[k], (str, int, float, type(None)))}
            entry.update({k: v for k, v in row.items()
                          if k.endswith("_id") and k not in entry and isinstance(v, str)})
            kept.append(entry)
        print(f"DIAG {name} job={job_id or '*'} records={len(kept)}")
        for entry in kept[-limit:]:
            print("DIAG", name, entry)


@contextlib.contextmanager
def controlled_run(base: Path, prereq: Prerequisites, *,
                   cache: Path | None = None) -> Iterator[Runner]:
    """Start the composed daemon via the CLI; always stop it and clean everything up."""
    import psutil

    before = set(owned_containers(prereq.endpoint))
    assert not before, "owned podmuncher containers already exist before the case started"
    roots = make_roots(base, prereq, cache=cache)
    with ContainerSpy(prereq.endpoint) as spy:
        runner = Runner(roots, prereq, spy)
        try:
            started = runner.cli("service", "start", timeout=180)
            runner.timings["start"] = round(started.seconds, 1)
            assert started.code == 0 and started.doc.get("pid"), started.doc
            runner.pid = int(started.doc["pid"])
            yield runner
        finally:
            if sys.exc_info()[0] is not None:
                print_diagnostics(roots.state)
            stopped = runner.cli("service", "stop", timeout=180) if runner.pid else None
            runner.timings["stop"] = round(stopped.seconds, 1) if stopped else -1.0
            runner.timings.pop("submit_at", None)
            print(f" timings={runner.timings}")
            pid, runner.pid = runner.pid, None
            deadline = time.monotonic() + 20
            while pid and psutil.pid_exists(pid) and time.monotonic() < deadline:
                time.sleep(0.2)
            alive = bool(pid) and psutil.pid_exists(pid) and \
                psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
            _kill(pid if alive else None)
            left = owned_containers(prereq.endpoint)
            for cid in left:  # only ever our own labelled workers
                docker(prereq.endpoint, "rm", "-f", cid, timeout=60)
            shutil.rmtree(base, ignore_errors=True)
    assert stopped is None or (stopped.code == 0 and stopped.doc.get("outcome") == "stopped"), \
        stopped.doc
    assert not alive, "the service process survived service stop"
    assert left == [], f"owned containers remained after the case: {len(left)}"
    assert not base.exists(), "temporary roots were not removed"


# ----------------------------------------------------------------------------- assertions


def summary(job: dict) -> dict:
    """A compact, path-free view of a job for assertion messages."""
    keep = ("state", "reason_code", "next_action", "stage", "stages", "attempt_id")
    view = {k: job.get(k) for k in keep if k in job}
    view["artifacts"] = [(a.get("slot"), a.get("available")) for a in job.get("artifacts", ())]
    view["keys"] = sorted(job)
    return view


def published_names(job_dir: Path) -> list[str]:
    return sorted(p.name for p in job_dir.iterdir()) if job_dir.is_dir() else []


def assert_transcripts(runner: Runner, job: dict, source: Path | None) -> dict:
    """Five exports plus sidecar; JSON provenance names image digest, revision, source sha."""
    job_dir = runner.job_dir(job["job_id"])
    names = published_names(job_dir)
    for name in TRANSCRIPT_FILES.values():
        assert name in names, f"{name} missing: {names}"
    available = sorted(a["slot"] for a in job["artifacts"] if a.get("available"))
    assert set(TRANSCRIPT_FILES) <= set(available), available
    document = json.loads((job_dir / "transcript.json").read_text(encoding="utf-8"))
    sidecar = json.loads((job_dir / SIDECAR).read_text(encoding="utf-8"))
    provenance = sidecar["provenance"]
    assert provenance["image_digest"] == runner.prereq.asr_image, provenance
    assert provenance["model_revision"], provenance
    text = json.dumps(document)
    assert runner.prereq.asr_image in text, "transcript JSON lacks the image digest"
    if source is not None:
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        assert sha in text, "transcript JSON lacks the source sha256"
    return {"files": names, "provenance": provenance}


KNOWN_DEFECTS = {
    "asset_verification_failed": (
        "real development model acquisition failed: DEFECT-1 was fixed in P3-U2 (development "
        "manifests may declare a real synthetic=false model with pinned HTTPS sources and "
        "per-file bytes/sha256), so this is a regression or a source/pin mismatch"),
    "upstream_unavailable": (
        "DEFECT-5 when the trail ends in preparing (model acquisition): the composed daemon's "
        "ModelCache has a 30 s deadline for the WHOLE acquisition (model_cache.py:1128, "
        ":1731; bootstrap.py _open_cache passes no timeout) and calls admission per read1 "
        "chunk (model_cache.py:1912, ~5000 calls for the 78 MB tiny model), each admission "
        "taking two VolumeProvider samples (bootstrap.py _host_plan, ~10 ms per call here), "
        "so the real download is quarantined at exactly 30 s"),
    "envelope_unvalidated": (
        "DEFECT-3 recurrence: the probe hold was raised to 12 s in P3-U6 (asr_worker.py); a "
        "hardware re-sample still landed after the probe container exited"),
    "storage_unknown": (
        "DEFECT-4 (intermittent, cause unverified): a stage-boundary storage preflight on "
        "this Docker Desktop host returned storage_unknown"),
    "writer_unresolved": (
        "attempt-1 intermittent: a job stuck in preparing with writer_unresolved"),
    "artifact_rejected": (
        "attempt-1 intermittent: artifact_rejected after transcription"),
}


class JobWatcher:
    """One poller for every job of the shared run: latest doc, terminal time, state trail."""

    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self.latest: dict[str, dict] = {}
        self.terminal_at: dict[str, float] = {}
        self.trail: dict[str, list[tuple[str, str, str]]] = {}
        self.errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                reply = self.runner.cli("jobs", "--limit", "50", timeout=60)
            except (OSError, subprocess.SubprocessError):
                self.errors += 1  # transient host failure; keep polling
            else:
                now = time.monotonic()
                for job in reply.doc.get("jobs") or ():
                    self._observe(job, now)
            self._stop.wait(0.5)

    def _observe(self, job: dict, now: float) -> None:
        job_id = job.get("job_id")
        if not isinstance(job_id, str):
            return
        self.latest[job_id] = job
        step = (str(job.get("state")), str(job.get("stage")), str(job.get("reason_code")))
        trail = self.trail.setdefault(job_id, [])
        if not trail or trail[-1] != step:
            trail.append(step)
        if job.get("state") in TERMINAL:
            self.terminal_at.setdefault(job_id, now)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(90)


@dataclass
class Submitted:
    case: str
    job_id: str
    source: Path | None
    submitted_at: float


class LiveBatch:
    """ONE composed service, ONE new empty cache root, the CPU jobs submitted back to back.

    The serial scheduler runs them in submission order; the first transcript job acquires
    the real pinned model into the new cache and every later job reuses it.  Each case test
    then waits for and asserts its own job; the detach/cancel case runs last on the same
    service.  ``close()`` stops the service and proves cleanup (no service process, no owned
    container, temporary roots removed).
    """

    def __init__(self, base: Path, prereq: Prerequisites) -> None:
        self.prereq = prereq
        self.began = time.monotonic()
        self.deadline = self.began + BATCH_BUDGET_SECONDS
        self._stack = contextlib.ExitStack()
        self.closed = False
        self.cleanup_ok = False
        self.jobs: dict[str, Submitted] = {}
        self.order: list[str] = []
        try:
            self.run = self._stack.enter_context(controlled_run(base, prereq))
            self.watcher = self._stack.enter_context(JobWatcher(self.run))
            self.cache_before = self.run.cli("cache", "list").doc
            inputs = self.run.roots.inputs
            self.sources = {
                "wav": make_media_fixture("wav", 3, inputs / "audio"),
                "mp4_only": make_media_fixture("mp4", 2.5, inputs / "video-only",
                                               prereq=prereq),
                "silent": make_media_fixture("silent_mp4", 3, inputs / "silent", prereq=prereq),
                "cancel": make_media_fixture("wav", 2.5, inputs / "cancel"),
            }
        except BaseException:
            self.close()
            raise

    def submit(self, case: str, source: str, local: Path | None, *flags: str) -> Submitted:
        at = time.monotonic()
        job_id = self.run.submit(source, *flags)
        self.jobs[case] = Submitted(case, job_id, local, at)
        self.order.append(case)
        return self.jobs[case]

    def submit_cpu_cases(self) -> None:
        s = self.sources
        self.submit("local_audio_transcript", str(s["wav"]), s["wav"])
        self.submit("local_video_only", str(s["mp4_only"]), s["mp4_only"], "--video",
                    "--no-transcript")
        self.submit("silent_video_combined", str(s["silent"]), s["silent"], "--video")
        self.submit("direct_https_combined", self.prereq.direct_url, None, "--video")
        self.submit("youtube_transcript", self.prereq.youtube_url, None)

    def wait(self, case: str) -> dict:
        """Wait for the case's job to be terminal; return its final doc via the real CLI."""
        job_id = self.jobs[case].job_id
        while True:
            job = self.watcher.latest.get(job_id) or {}
            if job.get("state") in TERMINAL:
                break
            if time.monotonic() >= self.deadline:
                trail = self.watcher.trail.get(job_id)
                print(f"\nSTUCK job={job_id} state={job.get('state')} "
                      f"reason={job.get('reason_code')} trail={trail}")
                print_diagnostics(self.run.roots.state, job_id)
                pytest.fail(f"{case}: job {job_id} stuck in {job.get('state')}/"
                            f"{job.get('stage')} ({job.get('reason_code')}); trail={trail}")
            time.sleep(0.25)
        return self.run.cli("job", job_id).doc.get("job") or job

    def timing(self, case: str) -> dict[str, float]:
        """Submit-to-terminal and own service time (from the previous job's terminal)."""
        item = self.jobs[case]
        done = self.watcher.terminal_at.get(item.job_id, time.monotonic())
        index = self.order.index(case)
        previous = [self.watcher.terminal_at.get(self.jobs[c].job_id, item.submitted_at)
                    for c in self.order[:index]]
        started = max([item.submitted_at, *previous])
        return {"total_s": round(done - item.submitted_at, 1), "own_s": round(done - started, 1)}

    def expect(self, job: dict, state: str, reason: str | None = None) -> None:
        expect_state(job, state, reason, trail=self.watcher.trail.get(job.get("job_id", "")),
                     state_dir=self.run.roots.state)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._stack.close()  # watcher stops, then controlled_run stops and proves cleanup
        self.cleanup_ok = True


@pytest.fixture(scope="module")
def batch(prereq: Prerequisites, tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveBatch]:
    live = LiveBatch(tmp_path_factory.mktemp("live") / "run", prereq)
    try:
        assert live.cache_before.get("entries") == [], live.cache_before  # new and empty
        live.submit_cpu_cases()
        yield live
    finally:
        live.close()


def expect_state(job: dict, state: str, reason: str | None = None, *,
                 trail: list | None = None, state_dir: Path | None = None) -> None:
    """Assert the terminal outcome; a known or watched defect is named and diagnosed.

    On a mismatch the job's state trail and reason, and the service's own audit/diagnostic
    records for that job (ids and reason codes only) are printed before failing.
    """
    got = (job.get("state"), job.get("reason_code"))
    if got[0] == state and (reason is None or got[1] == reason):
        return
    known = KNOWN_DEFECTS.get(str(got[1]))
    detail = f" -- {known}" if known else ""
    print(f"\nFAILED job={job.get('job_id')} state={got[0]} reason={got[1]} "
          f"stage={job.get('stage')} trail={trail}")
    if state_dir is not None:
        print_diagnostics(state_dir, job.get("job_id"))
    pytest.fail(f"expected {state}/{reason or '*'} but got {got}{detail}; job_id="
                f"{job.get('job_id')} trail={trail}; {summary(job)}")


def record(case: str, **evidence: Any) -> None:
    _PASSED[case] = evidence
    print(f"\nCASE {case} ok {json.dumps(evidence, sort_keys=True, default=str)}")


def asr_containers(batch: LiveBatch, job_id: str | None = None) -> list[SeenContainer]:
    return [c for c in list(batch.run.spy.seen.values())
            if c.image == batch.prereq.asr_image and job_id in (None, c.job)]


# --------------------------------------------------------------------------------- cases


@pytest.mark.live
def test_local_audio_transcript_new_cache_offline(batch: LiveBatch) -> None:
    """Cases 1 and 7 share one transcription: a local WAV into the run's new empty cache root.

    It is the first job, so the composed daemon acquires the real pinned model through the
    real ModelCache into the empty cache; transcript-only completes with all five exports
    and provenance; every ASR container (probe and transcribe) runs network-none with a
    read-only root.
    """
    case = "local_audio_transcript"
    job = batch.wait(case)
    batch.expect(job, "completed")
    evidence = assert_transcripts(batch.run, job, batch.jobs[case].source)
    asr = asr_containers(batch)
    assert len(asr) >= 2, "expected the probe and the transcription containers"
    assert all(c.network == "none" and c.readonly for c in asr), asr
    listed = batch.run.cli("cache", "list")
    entries = listed.doc.get("entries")
    assert listed.code == 0 and entries, listed.doc
    record(case, job_id=job["job_id"], **batch.timing(case), **evidence)
    record("new_cache_offline", job_id=job["job_id"], cache_entries_before=0,
           cache_entries_after=len(entries), asr_containers=len(asr),
           networks=sorted({c.network for c in asr}), acquisition="real ModelCache download")


@pytest.mark.live
def test_local_video_only_never_provisions_asr(batch: LiveBatch) -> None:
    case = "local_video_only"
    job = batch.wait(case)
    batch.expect(job, "completed")
    names = published_names(batch.run.job_dir(job["job_id"]))
    assert [n for n in names if n.endswith(".mp4")] and SIDECAR in names, names
    assert not [n for n in names if n.startswith("transcript")], names
    assert not asr_containers(batch, job["job_id"]), "an ASR container ran for a video-only job"
    record(case, job_id=job["job_id"], **batch.timing(case), files=names,
           containers=len(batch.run.spy.images(job["job_id"])))


@pytest.mark.live
def test_silent_video_with_transcript_is_partial_no_audio(batch: LiveBatch) -> None:
    case = "silent_video_combined"
    job = batch.wait(case)
    batch.expect(job, "partial", "no_audio")
    names = published_names(batch.run.job_dir(job["job_id"]))
    assert [n for n in names if n.endswith(".mp4")], names
    assert not [n for n in names if n.startswith("transcript")], names
    assert not asr_containers(batch, job["job_id"]), "ASR provisioned for silent media"
    record(case, job_id=job["job_id"], **batch.timing(case), files=names)


@pytest.mark.live
def test_direct_https_video_and_transcript(batch: LiveBatch) -> None:
    """Remote video+transcript: the video is published by copy, so the ASR input survives.

    ``video.mp4`` and all five transcript exports are published and available.
    """
    case = "direct_https_combined"
    job = batch.wait(case)
    batch.expect(job, "completed")
    evidence = assert_transcripts(batch.run, job, None)
    video = [a for a in job["artifacts"] if a["slot"] == "original_video"]
    assert video and video[0]["available"], summary(job)
    job_dir = batch.run.job_dir(job["job_id"])
    assert "video.mp4" in evidence["files"], evidence["files"]
    assert (job_dir / "video.mp4").stat().st_size > 0
    assert asr_containers(batch, job["job_id"]), "no ASR container ran for the combined job"
    record(case, job_id=job["job_id"], **batch.timing(case), **evidence)


@pytest.mark.live
def test_youtube_transcript(batch: LiveBatch) -> None:
    case = "youtube_transcript"
    job = batch.wait(case)
    batch.expect(job, "completed")
    evidence = assert_transcripts(batch.run, job, None)
    record(case, job_id=job["job_id"], **batch.timing(case), **evidence)


@pytest.mark.live
def test_detach_then_cancel(batch: LiveBatch) -> None:
    """Last on the same service: detach a transcript job, cancel it once its worker runs."""
    case = "detach_cancel"
    for earlier in list(batch.order):  # the serial queue is drained first
        batch.wait(earlier)
    source = batch.sources["cancel"]
    item = batch.submit(case, str(source), source)
    deadline = time.monotonic() + 60
    while not batch.run.spy.images(item.job_id):
        job = batch.watcher.latest.get(item.job_id) or {}
        assert job.get("state") not in TERMINAL, \
            f"job ended before any worker was observed: {summary(job)}"
        assert time.monotonic() < deadline, "no worker container started for the cancel job"
        time.sleep(0.1)
    cancelled = batch.run.cli("cancel", item.job_id)
    assert cancelled.code == 0 and cancelled.doc.get("status") in ("cancelled", "intent"), \
        cancelled.doc
    job = batch.wait(case)
    batch.expect(job, "cancelled")
    deadline = time.monotonic() + 30
    while owned_containers(batch.prereq.endpoint):
        assert time.monotonic() < deadline, "owned containers remain after cancel"
        time.sleep(0.5)
    record(case, job_id=item.job_id, **batch.timing(case), status=cancelled.doc.get("status"))


@pytest.mark.live
def test_optional_nvidia_gpu(prereq: Prerequisites) -> None:
    """Separate GPU matrix job: explicit ``PODMUNCHER_TEST_GPU=1`` only; never in the CPU smoke."""
    if os.environ.get("PODMUNCHER_TEST_GPU") != "1":
        pytest.skip("GPU matrix not selected (PODMUNCHER_TEST_GPU=1); NOT run, NOT passed")
    smi = shutil.which("nvidia-smi")
    assert smi, "PODMUNCHER_TEST_GPU=1 but nvidia-smi is not available: no NVIDIA GPU"
    probe = subprocess.run([smi, "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=30, check=False)
    assert probe.returncode == 0 and probe.stdout.strip().startswith("GPU-"), \
        "PODMUNCHER_TEST_GPU=1 but no usable NVIDIA GPU was reported"
    cuda_image = os.environ.get("PODMUNCHER_TEST_CUDA_IMAGE")
    assert cuda_image, "PODMUNCHER_TEST_GPU=1 needs PODMUNCHER_TEST_CUDA_IMAGE (asr cuda target)"
    pytest.fail("asr_cuda end-to-end is not wired in this unit (SPEC_GAP-4 e: untested)")


@pytest.mark.live
def test_cpu_smoke_summary(batch: LiveBatch) -> None:
    """Stop the shared service and prove cleanup; print the marker only if every CPU case
    passed in this session."""
    batch.close()
    assert batch.cleanup_ok
    wall = round(time.monotonic() - batch.began, 1)
    timings = {case: batch.timing(case) for case in batch.order}
    print(f"\nCPU smoke timings={json.dumps(timings, sort_keys=True)} "
          f"service={batch.run.timings} batch_wall_s={wall} "
          f"prerequisites_s={round(batch.prereq.seconds, 1)} "
          f"poll_errors={batch.run.spy.errors + batch.watcher.errors}")
    missing = [case for case in CPU_CASES if case not in _PASSED]
    assert not missing, f"CPU cases did not pass: {missing}"
    print(f"CPU smoke cases={len(_PASSED)} wall_s={wall} at={datetime.now(UTC).isoformat()}")
    print("PODMUNCHER_SMOKE_OK")
