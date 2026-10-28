"""Hardware-proxy ASR model policy and one bounded, owned load/retry lifecycle.

Model memory requirements come from the shipped table
``assets/model_hardware.json`` (published, estimated or measured figures, each
row with its source). An optional measured envelope that matches the artifact,
backend, runtime fingerprint and compute type exactly takes precedence over the
table row. Automatic selection picks the highest-rank verified model whose
known requirement plus headroom fits the live worker evidence, trying the GPU
first. Speed is never a selection criterion. An explicit operator model skips
the fit check; it must still be a verified registry artifact on a usable
backend. A trusted worker adapter joins hardware and runtime evidence to the
same admitted identity. Selection alone never authorizes inference.

Adapters run synchronously, must honor absolute monotonic deadlines, and must
address the supplied immutable worker identity at every OS use. No detached
thread is cancelled and reported as reaped. Storage admission retains the job's
exclusive lease through loading, cleanup and retry. The caller owns that lease
and the successfully loaded worker after this single-use coordinator returns.
"""

import functools
import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from fractions import Fraction
from importlib import resources
from typing import Protocol

from .admission import AdmissionPlan
from .domains import (
    Backend,
    Device,
    DomainValidationError,
    FieldStatus,
    ModelTier,
    OutputMode,
    Profile,
    ReasonCode,
    validate_utc_timestamp,
)
from .hardware import AdmittedWorkerRef, HardwareEvidence, RuntimeComputeEvidence
from .snapshots import Measurement

GIB = 1024**3
DECODE_BYTES_PER_SECOND = 256000
_MAX_BYTES = 2**63 - 1
_TIERS = tuple(ModelTier)
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_COMPUTE = {Backend.CUDA: 'int8_float16', Backend.CPU: 'int8'}
_BASES = frozenset({'published', 'estimated', 'measured'})


class ModelPolicyError(DomainValidationError):
    def __init__(self, reason_code=ReasonCode.INVALID_INPUT):
        super().__init__(ReasonCode.parse(reason_code))


def _bytes(value, *, positive=False):
    if type(value) is not int or not (1 if positive else 0) <= value <= _MAX_BYTES:
        raise ModelPolicyError()
    return value


def _seconds(value):
    if type(value) not in (int, float):
        raise ModelPolicyError()
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ModelPolicyError() from None
    if not math.isfinite(result) or not 0 < result <= _MAX_BYTES / DECODE_BYTES_PER_SECOND:
        raise ModelPolicyError()
    return result


def _label(value):
    if type(value) is not str or not value or len(value) > 512 or any(ord(x) < 32 for x in value):
        raise ModelPolicyError()
    return value


def _hash(value):
    if type(value) is not str or not _SHA.fullmatch(value):
        raise ModelPolicyError()


def _stamp(value, now, max_age, reason):
    validate_utc_timestamp(value)
    age = (now - datetime.fromisoformat(value)).total_seconds()
    if not 0 <= age <= max_age:
        raise ModelPolicyError(reason)


def _reading(field, now, max_age, reason):
    if type(field) is not Measurement or field.status is not FieldStatus.OK:
        raise ModelPolicyError(reason)
    _stamp(field.sample_time, now, max_age, reason)
    return field.value


@dataclass(frozen=True, slots=True)
class RuntimeFingerprint:
    image_digest: str
    runtime_version: str
    driver_version: str | None
    hardware_fingerprint: str

    def __post_init__(self):
        if type(self.image_digest) is not str or not self.image_digest.startswith('sha256:'):
            raise ModelPolicyError()
        _hash(self.image_digest[7:])
        _label(self.runtime_version)
        _label(self.hardware_fingerprint)
        if self.driver_version is not None:
            _label(self.driver_version)


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    model_id: str
    tier: ModelTier
    revision: str
    sha256: str

    def __post_init__(self):
        _label(self.model_id)
        if self.model_id == 'auto':
            raise ModelPolicyError()
        object.__setattr__(self, 'tier', ModelTier.parse(self.tier))
        _label(self.revision)
        _hash(self.sha256)


@dataclass(frozen=True, slots=True)
class MeasuredEnvelope:
    """Optional verified whole-file release measurement, never a short smoke estimate.

    When one matches a candidate exactly it overrides the hardware table row.

    RAM is the separately measured peak model + runtime cost, excluding decode.
    ``synthetic`` labels development fixtures and is retained in the decision.
    Verification of evidence_sha256 and artifact bytes belongs to the supplied
    evidence loader; this pure policy performs exact joins, not file I/O.
    """
    artifact: ModelArtifact
    runtime: RuntimeFingerprint
    backend: Backend
    compute_type: str
    worker_peak_ram_bytes: int
    worker_peak_vram_bytes: int
    validated_max_duration_seconds: float
    decode_bytes_per_second: int
    decode_timeout_seconds: float
    evidence_sha256: str
    release_tested: bool
    synthetic: bool = False

    def __post_init__(self):
        if type(self.artifact) is not ModelArtifact or type(self.runtime) is not RuntimeFingerprint:
            raise ModelPolicyError()
        object.__setattr__(self, 'backend', Backend.parse(self.backend))
        _label(self.compute_type)
        _bytes(self.worker_peak_ram_bytes, positive=True)
        _bytes(self.worker_peak_vram_bytes)
        _bytes(self.decode_bytes_per_second, positive=True)
        _seconds(self.validated_max_duration_seconds)
        _seconds(self.decode_timeout_seconds)
        _hash(self.evidence_sha256)
        if self.release_tested is not True:
            raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)
        if type(self.synthetic) is not bool or self.decode_bytes_per_second < DECODE_BYTES_PER_SECOND:
            raise ModelPolicyError()
        if self.backend is Backend.CPU and self.worker_peak_vram_bytes != 0:
            raise ModelPolicyError()
        if self.backend is Backend.CUDA and (not self.worker_peak_vram_bytes or self.runtime.driver_version is None):
            raise ModelPolicyError()


@dataclass(frozen=True, slots=True)
class WorkerEvidence:
    """Trusted sampler's same-worker join; preserve every Measurement timestamp."""
    identity: AdmittedWorkerRef
    hardware: HardwareEvidence
    runtime: RuntimeComputeEvidence
    fingerprint: RuntimeFingerprint
    runtime_sample_time: str

    def __post_init__(self):
        if (type(self.identity) is not AdmittedWorkerRef or type(self.hardware) is not HardwareEvidence
                or type(self.runtime) is not RuntimeComputeEvidence
                or type(self.fingerprint) is not RuntimeFingerprint):
            raise ModelPolicyError()
        if self.runtime.identity != self.identity:
            raise ModelPolicyError(ReasonCode.WORKER_PROTOCOL_MISMATCH)
        validate_utc_timestamp(self.runtime_sample_time)


@dataclass(frozen=True, slots=True)
class PolicyEvidence:
    workers: tuple[WorkerEvidence, ...]
    artifacts: tuple[ModelArtifact, ...]

    def __post_init__(self):
        if (type(self.workers) is not tuple or len(self.workers) > 2
                or any(type(x) is not WorkerEvidence for x in self.workers)
                or len({x.runtime.backend for x in self.workers}) != len(self.workers)
                or type(self.artifacts) is not tuple
                or any(type(x) is not ModelArtifact for x in self.artifacts)
                or len({x.model_id for x in self.artifacts}) != len(self.artifacts)):
            raise ModelPolicyError()


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    duration_seconds: float | None
    worker_ram_cap_bytes: int
    device: Device = Device.AUTO
    profile: Profile = Profile.BALANCED
    model: str = 'auto'
    output_mode: OutputMode = OutputMode.TRANSCRIPT
    duration_limit_seconds: float = 21600
    decode_timeout_seconds: float = 300

    def __post_init__(self):
        for key, kind in (('device', Device), ('profile', Profile), ('output_mode', OutputMode)):
            object.__setattr__(self, key, kind.parse(getattr(self, key)))
        _label(self.model)
        _bytes(self.worker_ram_cap_bytes, positive=True)
        _seconds(self.duration_limit_seconds)
        _seconds(self.decode_timeout_seconds)
        if self.duration_seconds is not None:
            _seconds(self.duration_seconds)


@dataclass(frozen=True, slots=True)
class BackendRequirement:
    """One backend's table row. ``None`` peaks mean "unknown" (never auto-picked).

    ``peak_ram_bytes`` is the worker host RAM peak excluding decode (CUDA rows:
    host RAM next to the GPU, often unpublished); ``peak_vram_bytes`` is CUDA only.
    """
    compute_type: str
    peak_ram_bytes: int | None
    peak_vram_bytes: int | None
    basis: str
    source: str
    conditions: str


@dataclass(frozen=True, slots=True)
class ModelHardware:
    model_id: str
    tier: ModelTier
    rank: int
    cuda: BackendRequirement
    cpu: BackendRequirement

    def requirement(self, backend):
        return self.cuda if backend is Backend.CUDA else self.cpu


@dataclass(frozen=True, slots=True)
class HardwareTable:
    """Parsed ``model_hardware.json``: per-model proxy requirements and headroom."""
    schema_version: int
    updated: str
    ram_fraction: Fraction
    vram_fraction: Fraction
    vram_min_bytes: int
    models: tuple[ModelHardware, ...]

    def row(self, model_id):
        return next((x for x in self.models if x.model_id == model_id), None)


def _keys(value, expected):
    if type(value) is not dict or set(value) != expected:
        raise ModelPolicyError()
    return value


def _optional_bytes(value):
    return None if value is None else _bytes(value, positive=True)


def _fraction(value):
    if type(value) not in (int, float) or not 0 < value < 1:
        raise ModelPolicyError()
    return Fraction(str(value))


def _requirement(value, backend):
    keys = {'compute_type', 'basis', 'source', 'conditions'}
    keys |= {'peak_vram_bytes', 'peak_host_ram_bytes'} if backend is Backend.CUDA else {'peak_ram_bytes'}
    value = _keys(value, keys)
    basis, source = value['basis'], _label(value['source'])
    if type(basis) is not str or basis not in _BASES or (basis != 'measured' and not source.startswith('https://')):
        raise ModelPolicyError()
    cuda = backend is Backend.CUDA
    return BackendRequirement(
        _label(value['compute_type']),
        _optional_bytes(value['peak_host_ram_bytes' if cuda else 'peak_ram_bytes']),
        _optional_bytes(value['peak_vram_bytes']) if cuda else None,
        basis, source, _label(value['conditions']))


def parse_hardware_table(data) -> HardwareTable:
    """Simple structural validation: required keys, positive ints or null, known basis."""
    data = _keys(data, {'schema_version', 'updated', 'headroom', 'models'})
    headroom = _keys(data['headroom'], {'ram_fraction', 'vram_fraction', 'vram_min_bytes'})
    if data['schema_version'] != 1 or type(data['schema_version']) is not int or type(data['models']) is not list:
        raise ModelPolicyError()
    models = []
    for row in data['models']:
        row = _keys(row, {'model_id', 'tier', 'rank', 'cuda', 'cpu'})
        models.append(ModelHardware(_label(row['model_id']), ModelTier.parse(row['tier']),
                                    _bytes(row['rank'], positive=True),
                                    _requirement(row['cuda'], Backend.CUDA),
                                    _requirement(row['cpu'], Backend.CPU)))
    if len({x.model_id for x in models}) != len(models) or len({x.rank for x in models}) != len(models):
        raise ModelPolicyError()
    return HardwareTable(1, _label(data['updated']), _fraction(headroom['ram_fraction']),
                         _fraction(headroom['vram_fraction']),
                         _bytes(headroom['vram_min_bytes'], positive=True), tuple(models))


@functools.cache
def load_hardware_table() -> HardwareTable:
    """The packaged table (read once per process)."""
    text = resources.files('podmuncher').joinpath('assets', 'model_hardware.json').read_text('utf-8')
    return parse_hardware_table(json.loads(text))


@dataclass(frozen=True, slots=True)
class EnvelopeView:
    """Compatibility shim for callers that still read ``choice.envelope.*``.

    bootstrap.py reads ``.backend``, ``.artifact`` and ``.compute_type``; the
    peaks are 0 when unknown. P6-U2 migrates callers to the ModelChoice fields.
    """
    artifact: ModelArtifact
    backend: Backend
    compute_type: str
    worker_peak_ram_bytes: int
    worker_peak_vram_bytes: int


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """An un-loaded candidate plus the requirement it was judged by.

    ``basis`` is measured|published|estimated for automatic picks and
    ``explicit`` for an operator choice (fit not checked). Peaks are ``None``
    when unknown. ``measured_envelope`` is set when an exact envelope was used.
    """
    artifact: ModelArtifact
    backend: Backend
    compute_type: str
    identity: AdmittedWorkerRef
    reason: str
    basis: str
    source: str
    peak_ram_bytes: int | None
    peak_vram_bytes: int | None
    decode_bytes: int
    worker_memory_cap_bytes: int
    worker_reserve_bytes: int
    device_reserve_bytes: int
    decode_timeout_seconds: float
    measured_envelope: MeasuredEnvelope | None = None

    @property
    def required_ram_bytes(self):
        if self.peak_ram_bytes is None:
            return None
        return self.decode_bytes + self.peak_ram_bytes + self.worker_reserve_bytes

    @property
    def required_vram_bytes(self):
        if self.backend is Backend.CPU:
            return 0
        return None if self.peak_vram_bytes is None else self.peak_vram_bytes + self.device_reserve_bytes

    @property
    def envelope(self):
        """Shim: the old envelope-shaped view (see EnvelopeView)."""
        return EnvelopeView(self.artifact, self.backend, self.compute_type,
                            self.peak_ram_bytes or 0, self.peak_vram_bytes or 0)

    def to_dict(self):
        """Protected record: includes provenance and the admitted worker identity."""
        return asdict(self)


def _capacity(worker, request, now, max_age):
    hw = worker.hardware
    values = [_reading(getattr(hw, key), now, max_age, ReasonCode.RAM_INSUFFICIENT)
              for key in ('worker_ram_limit', 'worker_ram_current', 'worker_ram_available')]
    limit, current, available = map(_bytes, values)
    if not limit or current > limit or available > limit - current:
        raise ModelPolicyError(ReasonCode.RAM_INSUFFICIENT)
    effective = min(limit, request.worker_ram_cap_bytes)
    if current > effective:
        raise ModelPolicyError(ReasonCode.RAM_INSUFFICIENT)
    available = min(available, effective - current)
    _stamp(worker.runtime_sample_time, now, max_age, ReasonCode.ENVELOPE_UNVALIDATED)
    compute = _reading(hw.runtime_compute_types, now, max_age, ReasonCode.ENVELOPE_UNVALIDATED)
    if (type(compute) is not tuple or compute != worker.runtime.compute_types
            or not compute or any(type(x) is not str for x in compute)):
        raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)
    visible = _reading(hw.worker_gpu_visible, now, max_age, ReasonCode.GPU_UNAVAILABLE)
    if type(visible) is not bool or visible is not worker.identity.gpu_visible:
        raise ModelPolicyError(ReasonCode.WORKER_PROTOCOL_MISMATCH)
    total = free = 0
    if worker.runtime.backend is Backend.CUDA:
        if not visible or worker.runtime.gpu_uuid != worker.identity.gpu_uuid:
            raise ModelPolicyError(ReasonCode.GPU_UNAVAILABLE)
        total, free = [_bytes(_reading(getattr(hw, name), now, max_age, ReasonCode.GPU_UNAVAILABLE))
                       for name in ('gpu_vram_total', 'gpu_vram_free')]
        # A fresh live driver reading is required. The fingerprint's
        # driver_version is the release's "tested with" evidence, not a gate:
        # it only decides whether a measured envelope matches exactly. A
        # driver the runtime cannot use surfaces as a probe or load failure.
        _reading(hw.gpu_driver, now, max_age, ReasonCode.GPU_UNAVAILABLE)
        if not total or free > total:
            raise ModelPolicyError(ReasonCode.GPU_UNAVAILABLE)
    return effective, available, total, free


@dataclass(frozen=True, slots=True)
class _Requirement:
    ram: int | None
    vram: int | None
    basis: str
    source: str
    envelope: MeasuredEnvelope | None


def _rank(artifact, table):
    """Table rank (larger = better); untabled artifacts rank by tier."""
    row = table.row(artifact.model_id)
    return row.rank if row is not None else (_TIERS.index(artifact.tier) + 1) * 10


def _table_requirement(artifact, backend, table):
    """The table row for this backend; None when absent, other compute type, or unknown peak."""
    row = table.row(artifact.model_id)
    if row is None:
        return None
    need = row.requirement(backend)
    if need.compute_type != _COMPUTE[backend]:
        return None
    if (need.peak_vram_bytes if backend is Backend.CUDA else need.peak_ram_bytes) is None:
        return None
    return _Requirement(need.peak_ram_bytes, need.peak_vram_bytes, need.basis, need.source, None)


def _resolve(artifact, backend, worker, envelopes, table, duration=None):
    """Exact measured envelope first, else the table row; None when unknown.

    Identical duplicate envelopes collapse to one. Conflicting envelopes for
    the same exact join make the requirement unknown (never auto-picked; an
    explicit operator choice may still proceed without a fit check).

    With a ``duration`` (automatic selection only) an envelope whose validated
    duration is shorter is not used: the table row applies as if no envelope
    existed, so a long request is never certified from a short measurement.
    """
    compute = _COMPUTE[backend]
    matches = {x for x in envelopes if x.artifact == artifact and x.backend is backend
               and x.runtime == worker.fingerprint and x.compute_type == compute}
    if len(matches) > 1:
        return None
    if matches:
        found = next(iter(matches))
        if duration is None or duration <= found.validated_max_duration_seconds:
            return _Requirement(found.worker_peak_ram_bytes, found.worker_peak_vram_bytes
                                if backend is Backend.CUDA else None, 'measured',
                                'measured envelope sha256:' + found.evidence_sha256, found)
    return _table_requirement(artifact, backend, table)


def _choice(request, artifact, backend, need, worker, capacity, table, reason, *, enforce_fit):
    """Build the choice; with ``enforce_fit`` reject a known requirement that does not fit."""
    effective, available, total, free = capacity
    envelope = need.envelope if need is not None else None
    rate = envelope.decode_bytes_per_second if envelope is not None else DECODE_BYTES_PER_SECOND
    # Use the exact decimal input ratio and integer ceiling; no float byte loss.
    numerator, denominator = request.duration_seconds.as_integer_ratio()
    decode = (numerator * rate + denominator - 1) // denominator
    reserve = -(-effective * table.ram_fraction.numerator // table.ram_fraction.denominator)
    device_reserve = 0
    if backend is Backend.CUDA:
        share = -(-total * table.vram_fraction.numerator // table.vram_fraction.denominator)
        device_reserve = max(share, table.vram_min_bytes)
    ram = need.ram if need is not None else None
    vram = need.vram if need is not None else None
    if enforce_fit:
        if decode + (ram or 0) + reserve > available:
            raise ModelPolicyError(ReasonCode.RAM_INSUFFICIENT)
        if backend is Backend.CUDA and vram + device_reserve > free:
            raise ModelPolicyError(ReasonCode.GPU_UNAVAILABLE)
    timeout = request.decode_timeout_seconds
    if envelope is not None:
        timeout = min(timeout, envelope.decode_timeout_seconds)
    if enforce_fit:
        basis, source = need.basis, need.source
    else:
        basis, source = 'explicit', need.source if need is not None else 'operator choice; requirement unknown'
    return ModelChoice(artifact, backend, _COMPUTE[backend], worker.identity, reason, basis, source,
                       ram, vram, decode, effective, reserve, device_reserve, timeout, envelope)


def _select(request, evidence, envelopes, table, now, max_age_seconds, lower_than, enforce_fit):
    if type(request) is not PolicyRequest:
        raise ModelPolicyError()
    if request.output_mode is OutputMode.VIDEO:
        return None
    if request.duration_seconds is None:
        raise ModelPolicyError(ReasonCode.DURATION_UNKNOWN)
    if request.duration_seconds > request.duration_limit_seconds:
        raise ModelPolicyError(ReasonCode.DURATION_EXCEEDED)
    if (type(evidence) is not PolicyEvidence or type(envelopes) is not tuple
            or any(type(x) is not MeasuredEnvelope for x in envelopes)):
        raise ModelPolicyError()
    table = load_hardware_table() if table is None else table
    if type(table) is not HardwareTable:
        raise ModelPolicyError()
    if lower_than is not None and (type(lower_than) is not ModelChoice or request.model != 'auto'):
        raise ModelPolicyError(ReasonCode.MODEL_OOM)
    _seconds(max_age_seconds)
    if max_age_seconds > 3:
        raise ModelPolicyError()
    now = datetime.now(UTC) if now is None else now
    if type(now) is not datetime or now.tzinfo is None or now.utcoffset().total_seconds() != 0:
        raise ModelPolicyError()
    backends = (Backend.CUDA, Backend.CPU) if request.device is Device.AUTO else (Backend(request.device.value),)
    artifacts = {x.model_id: x for x in evidence.artifacts}
    if request.model != 'auto' and request.model not in artifacts:
        raise ModelPolicyError(ReasonCode.ASSET_VERIFICATION_FAILED)
    if request.model != 'auto':
        candidates = (artifacts[request.model],)
    else:
        candidates = tuple(sorted(evidence.artifacts, key=lambda x: (_rank(x, table), x.model_id), reverse=True))
    if lower_than is not None:
        floor = _rank(lower_than.artifact, table)
        candidates = tuple(x for x in candidates if _rank(x, table) < floor)
        if not candidates:
            # The single OOM retry has nowhere lower to go: the load ends as OOM.
            raise ModelPolicyError(ReasonCode.MODEL_OOM)
    reason = ('oom_lower_retry' if lower_than is not None else 'explicit_model' if request.model != 'auto'
              else 'auto_largest_fit')
    failure = ReasonCode.ENVELOPE_UNVALIDATED
    for backend in backends:
        worker = next((x for x in evidence.workers if x.runtime.backend is backend), None)
        if worker is None:
            failure = ReasonCode.GPU_UNAVAILABLE if backend is Backend.CUDA else failure
            continue
        try:
            capacity = _capacity(worker, request, now, max_age_seconds)
        except ModelPolicyError as exc:
            failure = exc.reason_code
            continue
        failure = ReasonCode.ENVELOPE_UNVALIDATED
        if _COMPUTE[backend] not in worker.runtime.compute_types:
            continue
        for artifact in candidates:
            need = _resolve(artifact, backend, worker, envelopes, table,
                            request.duration_seconds if enforce_fit else None)
            if enforce_fit and need is None:
                failure = ReasonCode.ENVELOPE_UNVALIDATED
                continue
            try:
                return _choice(request, artifact, backend, need, worker, capacity, table, reason,
                               enforce_fit=enforce_fit)
            except ModelPolicyError as exc:
                failure = exc.reason_code
    raise ModelPolicyError(failure)


def select_model(request: PolicyRequest, evidence: PolicyEvidence,
                 envelopes: tuple[MeasuredEnvelope, ...] = (), *, now=None, max_age_seconds=3,
                 lower_than: ModelChoice | None = None, table: HardwareTable | None = None) -> ModelChoice | None:
    """Return an un-loaded candidate; raise a closed reason when none is usable.

    ``auto`` picks the highest-rank verified artifact whose known requirement
    fits (GPU first on device auto). An explicit model keeps its exact registry
    ID and skips the fit check; explicit devices keep their backend.
    ``lower_than`` is only for the single automatic OOM retry (next lower rank).
    ``table`` defaults to the packaged ``model_hardware.json``.
    """
    enforce_fit = type(request) is not PolicyRequest or request.model == 'auto'
    return _select(request, evidence, envelopes, table, now, max_age_seconds, lower_than, enforce_fit)


@dataclass(frozen=True, slots=True)
class LoadAdmission:
    """Fresh, candidate-specific full storage + worker re-admission under lease."""
    choice: ModelChoice
    evidence: PolicyEvidence
    storage: AdmissionPlan
    lease_held: bool


@dataclass(frozen=True, slots=True)
class LoadProbeResult:
    choice: ModelChoice
    reason: ReasonCode
    peak_ram_bytes: int
    peak_vram_bytes: int
    enforced_memory_cap_bytes: int
    enforced_decode_timeout_seconds: float

    def __post_init__(self):
        if type(self.choice) is not ModelChoice:
            raise ModelPolicyError()
        object.__setattr__(self, 'reason', ReasonCode.parse(self.reason))
        if self.reason not in (ReasonCode.OK, ReasonCode.MODEL_OOM, ReasonCode.MODEL_LOAD_FAILED, ReasonCode.REQUEST_TIMEOUT):
            raise ModelPolicyError()
        for name in ('peak_ram_bytes', 'peak_vram_bytes', 'enforced_memory_cap_bytes'):
            _bytes(getattr(self, name))
        _seconds(self.enforced_decode_timeout_seconds)
        if self.reason is ReasonCode.OK and (not self.peak_ram_bytes or
                self.choice.backend is Backend.CUDA and not self.peak_vram_bytes):
            raise ModelPolicyError()


@dataclass(frozen=True, slots=True)
class CleanupResult:
    identity: AdmittedWorkerRef
    fragments_discarded: bool
    model_unloaded: bool
    writers_reaped: bool


@dataclass(frozen=True, slots=True)
class LoadedModel:
    choice: ModelChoice
    observations: LoadProbeResult
    attempts: int

    def to_dict(self):
        return asdict(self)


class ModelLifecycle(Protocol):
    """Trusted bounded adapters; exceptions may not contain public diagnostics.

    ``sample`` performs read-only verified sampling. ``readmit`` recomputes all
    storage growth/reservations and memory under the exclusive job lease, and
    returns timestamps from actual sampling, never refreshes cached timestamps.
    ``load`` enforces choice's RAM cap and decode timeout BEFORE decode/load,
    then retains the loaded model on success. ``cleanup`` returns true only
    after fragment disposal, unload, and all owned writers actually exit.
    """
    def sample(self, *, deadline: float) -> PolicyEvidence: ...
    def readmit(self, choice: ModelChoice, *, deadline: float) -> LoadAdmission: ...
    def load(self, choice: ModelChoice, *, deadline: float) -> LoadProbeResult: ...
    def cleanup(self, choice: ModelChoice, *, deadline: float) -> CleanupResult: ...


class ModelPolicy:
    """Single-use load coordinator; at most two owned load invocations total.

    ``envelopes`` may be empty: requirements then come from the hardware table.
    Observed load peaks are checked only against a measured envelope; table
    figures are proxies, not claims.
    """
    def __init__(self, envelopes: tuple[MeasuredEnvelope, ...], lifecycle: ModelLifecycle, *,
                 clock=None, monotonic=time.monotonic, callback_timeout_seconds=30,
                 max_age_seconds=3, table: HardwareTable | None = None):
        if type(envelopes) is not tuple or any(type(x) is not MeasuredEnvelope for x in envelopes):
            raise ModelPolicyError()
        if table is not None and type(table) is not HardwareTable:
            raise ModelPolicyError()
        _seconds(callback_timeout_seconds)
        _seconds(max_age_seconds)
        if callback_timeout_seconds > 300 or max_age_seconds > 3:
            raise ModelPolicyError()
        self.envelopes, self.lifecycle, self.table = envelopes, lifecycle, table
        self.clock = clock if clock is not None else lambda: datetime.now(UTC)
        self.monotonic, self.timeout, self.max_age = monotonic, callback_timeout_seconds, max_age_seconds
        self._lock, self._used = threading.Lock(), False

    def _call(self, callback, *args):
        deadline = self.monotonic() + self.timeout
        try:
            result = callback(*args, deadline=deadline)
        except ModelPolicyError:
            raise
        except MemoryError:
            raise ModelPolicyError(ReasonCode.MODEL_OOM) from None
        except Exception:  # noqa: BLE001 - adapter detail never leaves; closed reason only
            raise ModelPolicyError(ReasonCode.MODEL_LOAD_FAILED) from None
        if self.monotonic() >= deadline:
            raise ModelPolicyError(ReasonCode.REQUEST_TIMEOUT)
        return result

    def _choose(self, request, evidence, lower_than=None, *, enforce_fit=None):
        enforce_fit = request.model == 'auto' if enforce_fit is None else enforce_fit
        return _select(request, evidence, self.envelopes, self.table, self.clock(), self.max_age,
                       lower_than, enforce_fit)

    def _admit(self, request, choice):
        admission = self._call(self.lifecycle.readmit, choice)
        if (type(admission) is not LoadAdmission or admission.choice != choice
                or type(admission.storage) is not AdmissionPlan or admission.lease_held is not True):
            raise ModelPolicyError(ReasonCode.CONFLICT)
        storage_reason = ReasonCode.STORAGE_STALE if admission.storage.admitted else admission.storage.reason
        if _reading(admission.storage.admission, self.clock(), self.max_age, storage_reason) is not True:
            raise ModelPolicyError(ReasonCode.STORAGE_UNKNOWN)
        # Revalidate precisely the chosen artifact/backend, not a substitute; an
        # automatic choice must still fit, an explicit one keeps skipping fit.
        explicit = replace(request, model=choice.artifact.model_id, device=Device(choice.backend.value))
        current = self._choose(explicit, admission.evidence, enforce_fit=request.model == 'auto')
        if (current.artifact != choice.artifact or current.backend is not choice.backend
                or current.compute_type != choice.compute_type
                or current.measured_envelope != choice.measured_envelope
                or current.peak_vram_bytes != choice.peak_vram_bytes or current.identity != choice.identity
                or current.worker_memory_cap_bytes != choice.worker_memory_cap_bytes
                or current.required_ram_bytes != choice.required_ram_bytes
                or current.device_reserve_bytes != choice.device_reserve_bytes):
            raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)

    def _cleanup(self, choice):
        try:
            result = self._call(self.lifecycle.cleanup, choice)
        except ModelPolicyError:
            raise ModelPolicyError(ReasonCode.WRITER_UNRESOLVED) from None
        if (type(result) is not CleanupResult or result.identity != choice.identity
                or result.fragments_discarded is not True or result.model_unloaded is not True
                or result.writers_reaped is not True):
            raise ModelPolicyError(ReasonCode.WRITER_UNRESOLVED)

    def load(self, request: PolicyRequest) -> LoadedModel | None:
        if type(request) is not PolicyRequest:
            raise ModelPolicyError()
        if request.output_mode is OutputMode.VIDEO:
            return None
        if not self._lock.acquire(blocking=False):
            raise ModelPolicyError(ReasonCode.CONFLICT)
        try:
            if self._used:
                raise ModelPolicyError(ReasonCode.CONFLICT)
            self._used = True
            choice = self._choose(request, self._call(self.lifecycle.sample))
            for attempt in (1, 2):
                self._admit(request, choice)
                try:
                    result = self._call(self.lifecycle.load, choice)
                    if type(result) is not LoadProbeResult or result.choice != choice:
                        raise ModelPolicyError(ReasonCode.MODEL_LOAD_FAILED)
                    if result.reason is not ReasonCode.OK:
                        raise ModelPolicyError(result.reason)
                    if (result.enforced_memory_cap_bytes != choice.worker_memory_cap_bytes
                            or result.enforced_decode_timeout_seconds != choice.decode_timeout_seconds):
                        raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)
                    measured = choice.measured_envelope
                    if measured is not None and (
                            result.peak_ram_bytes > choice.required_ram_bytes - choice.worker_reserve_bytes
                            or result.peak_vram_bytes > measured.worker_peak_vram_bytes):
                        raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)
                    return LoadedModel(choice, result, attempt)
                except ModelPolicyError as exc:
                    self._cleanup(choice)
                    if exc.reason_code is not ReasonCode.MODEL_OOM or request.model != 'auto' or attempt == 2:
                        raise
                choice = self._choose(request, self._call(self.lifecycle.sample), lower_than=choice)
            raise ModelPolicyError(ReasonCode.MODEL_OOM)
        finally:
            self._lock.release()
