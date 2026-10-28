"""Pure, conservative storage admission for one active operation.

Callers describe the bytes an operation can add during half-open stage
intervals.  This module neither samples storage nor acquires a lease.  The
caller must take a fresh sample and atomically acquire the single execution
lease before starting writers, then retain it until every writer is reaped.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
import math
from typing import Mapping, Protocol

from .config import Config, Limits
from .docker_storage import Constraint, DockerStorageEvidence
from .domains import DomainValidationError, FieldStatus, JobState, ReasonCode
from .snapshots import Measurement
from .volumes import VolumeEvidence


MIB = 1024 * 1024
GIB = 1024 * MIB
WORKER_FLOOR = 256 * MIB
LOG_FLOOR = 20 * MIB
CONTROL_BOUND = 512 * MIB


class AdmissionError(DomainValidationError):
    """Invalid planner input with a closed reason and no supplied value."""

    def __init__(self, reason_code: ReasonCode = ReasonCode.INVALID_INPUT):
        super().__init__(ReasonCode.parse(reason_code))


def _bytes(value: object) -> int:
    if type(value) is not int or value < 0:
        raise AdmissionError()
    return value


def _identity(value: object) -> str:
    if type(value) is not str or not value or len(value) > 8192:
        raise AdmissionError()
    return value


@dataclass(frozen=True, slots=True)
class GrowthInterval:
    """An asset's total size and already allocated bytes during [start, end).

    Equal ``asset_key`` values on one physical volume describe one reusable
    allocation.  Its remaining bytes are charged once at any instant.
    Stage indices are internal ordering labels, not persisted domain values.
    """

    asset_key: str
    volume_identity: str
    total_bytes: int | None
    allocated_bytes: int
    start: int
    end: int
    media_cap_bytes: int | None = None

    def __post_init__(self) -> None:
        _identity(self.asset_key)
        _identity(self.volume_identity)
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start < self.end:
            raise AdmissionError()
        _bytes(self.allocated_bytes)
        if self.total_bytes is None:
            if self.media_cap_bytes is None:
                raise AdmissionError(ReasonCode.STORAGE_UNKNOWN)
            total = _bytes(self.media_cap_bytes)
        else:
            total = _bytes(self.total_bytes)
            if self.media_cap_bytes is not None and total > _bytes(self.media_cap_bytes):
                raise AdmissionError(ReasonCode.MEDIA_LIMIT_REACHED)
        if self.allocated_bytes > total:
            raise AdmissionError()
        if self.media_cap_bytes is not None:
            _bytes(self.media_cap_bytes)

    @property
    def remaining_bytes(self) -> int:
        return (self.media_cap_bytes if self.total_bytes is None else self.total_bytes) - self.allocated_bytes


@dataclass(frozen=True, slots=True)
class StoragePlacement:
    scratch: str
    cache: str
    output: str
    state: str
    docker_vm: str
    docker_backing: str
    image_store: str | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name == 'image_store' and self.image_store is None:
                continue
            _identity(getattr(self, name))


@dataclass(frozen=True, slots=True)
class WorkloadGrowth:
    """Upper bounds for every normal allocation in a worker operation.

    The caller supplies measured or verified sizes.  ``None`` for a source
    stream uses the configured media cap.  Existing allocated bytes belong in
    ``allocated`` under the corresponding interval key. Each credit is bound
    to that asset's primary placement volume, not other discovered Docker
    stores. A verified reused asset can use its allocated full size there.
    """

    image_compressed: int
    image_unpacked: int
    model_partial: int
    model_unpacked: int
    source_streams: tuple[int | None, ...]
    merge: int
    copied_source: int
    copied_video: int
    publication_staging: int
    transcript_outputs: tuple[int, ...]
    sidecars: int
    state: int
    worker: int = WORKER_FLOOR
    container_metadata_and_logs: int = LOG_FLOOR

    def __post_init__(self) -> None:
        for name in ('image_compressed', 'image_unpacked', 'model_partial', 'model_unpacked',
                     'merge', 'copied_source', 'copied_video', 'publication_staging',
                     'sidecars', 'state', 'worker', 'container_metadata_and_logs'):
            _bytes(getattr(self, name))
        if type(self.source_streams) is not tuple or not self.source_streams:
            raise AdmissionError()
        for size in self.source_streams:
            if size is not None:
                _bytes(size)
        if type(self.transcript_outputs) is not tuple:
            raise AdmissionError()
        for size in self.transcript_outputs:
            _bytes(size)
        if self.state > CONTROL_BOUND:
            raise AdmissionError()


def standard_growth_intervals(
    growth: WorkloadGrowth, placement: StoragePlacement, config: Config | Limits, *,
    allocated: Mapping[str, int] | None = None,
    host_growth_fraction: float = 0.10,
) -> tuple[GrowthInterval, ...]:
    """Build conservative overlapping stage lifetimes and VHD host growth.

    Stage 0 starts image/model acquisition; 3 starts source acquisition; 5
    starts export; 6 starts publication.  Retained outputs remain live through
    publication.  VM writes also grow the host VHD budget with a rounded-up
    filesystem overhead. Host growth stays allocated through operation end:
    deleting a temporary VM file does not prove that its VHD shrinks.
    State and diagnostics share one bounded control pool. The ``control``
    allocation credit covers the whole pool, including ``state``; overlapping
    credits use their maximum, never their sum.
    """
    if type(growth) is not WorkloadGrowth or type(placement) is not StoragePlacement:
        raise AdmissionError()
    limits = _limits(config)
    if (type(host_growth_fraction) not in (int, float) or
            (type(host_growth_fraction) is float and not math.isfinite(host_growth_fraction))
            or host_growth_fraction < 0.10):
        raise AdmissionError()
    allocated = {} if allocated is None else allocated
    if not isinstance(allocated, Mapping) or any(type(k) is not str or type(v) is not int or v < 0
                                                  for k, v in allocated.items()):
        raise AdmissionError()
    result: list[GrowthInterval] = []
    known_keys: set[str] = set()

    def add(key: str, volume: str, total: int | None, start: int, end: int,
            *, media: bool = False) -> None:
        known_keys.add(key)
        cap = limits.media_bytes if media else None
        part = allocated.get(key, 0)
        item = GrowthInterval(key, volume, total, part, start, end, cap)
        result.append(item)
        if volume == placement.docker_vm and placement.docker_backing != placement.docker_vm:
            # VM allocation does not prove physical VHD allocation.  Charge
            # the entire remaining VM growth plus host filesystem overhead.
            upper = limits.media_bytes if total is None else total
            numerator, denominator = _ratio(host_growth_fraction)
            remaining = upper - part
            host_total = remaining + (remaining * numerator + denominator - 1) // denominator
            result.append(GrowthInterval('host:' + key, placement.docker_backing,
                                         host_total, 0, start, 8))

    image_store = placement.image_store or placement.docker_vm
    add('image_compressed', image_store, growth.image_compressed, 0, 2)
    add('image_unpacked', image_store, growth.image_unpacked, 1, 8)
    add('model_partial', placement.cache, growth.model_partial, 1, 3)
    add('model_unpacked', placement.cache, growth.model_unpacked, 2, 8)
    for index, size in enumerate(growth.source_streams):
        add('stream:' + str(index), placement.scratch, size, 3, 6, media=True)
    add('merge', placement.scratch, growth.merge, 4, 6)
    add('copied_source', placement.scratch, growth.copied_source, 3, 7)
    add('copied_video', placement.output, growth.copied_video, 4, 8)
    add('publication_staging', placement.output, growth.publication_staging, 6, 8)
    for index, size in enumerate(growth.transcript_outputs):
        add('transcript:' + str(index), placement.output, size, 5, 8)
    add('sidecars', placement.output, growth.sidecars, 5, 8)
    if allocated.get('state', 0) > growth.state:
        raise AdmissionError()
    allocated = dict(allocated)
    allocated['control'] = max(allocated.get('control', 0), allocated.get('state', 0))
    known_keys.add('state')
    add('control', placement.state, CONTROL_BOUND, 0, 8)
    add('worker', placement.docker_vm, max(WORKER_FLOOR, growth.worker), 0, 8)
    add('container_metadata_and_logs', placement.docker_vm,
        max(LOG_FLOOR, growth.container_metadata_and_logs), 0, 8)
    if allocated.keys() - known_keys:
        raise AdmissionError()
    return tuple(result)


@dataclass(frozen=True, slots=True)
class VolumeRequirement:
    volume_identity: str
    capacity: Measurement[int]
    free: Measurement[int]
    reserve: Measurement[int]
    reserved: Measurement[int]
    required: Measurement[int]
    available_after_job: Measurement[int]
    reason: ReasonCode


@dataclass(frozen=True, slots=True)
class AdmissionPlan:
    admission: Measurement[bool]
    volumes: tuple[VolumeRequirement, ...]
    intervals: tuple[GrowthInterval, ...]

    @property
    def admitted(self) -> bool:
        return self.admission.status is FieldStatus.OK and self.admission.value is True

    @property
    def reason(self) -> ReasonCode:
        return ReasonCode.OK if self.admitted else self.admission.reason or ReasonCode.STORAGE_UNKNOWN


class ExecutionLease(Protocol):
    """Atomic single-operation lease supplied by the state owner.

    ``try_acquire`` must atomically reject another active holder.  ``release``
    must refuse release while writers remain; admission does not call either.
    Queued and blocked jobs never call ``try_acquire``.
    """

    def try_acquire(self, operation_id: str, plan: AdmissionPlan) -> bool: ...
    def release(self, operation_id: str, *, writers_reaped: bool) -> None: ...


def may_acquire_lease(state: JobState, plan: AdmissionPlan) -> bool:
    """Only a freshly admitted checking operation can reserve the lease."""
    if type(plan) is not AdmissionPlan:
        raise AdmissionError()
    try:
        state = JobState.parse(state)
    except DomainValidationError:
        raise AdmissionError() from None
    return state is JobState.CHECKING and plan.admitted


def _limits(value: Config | Limits) -> Limits:
    limits = value.limits if type(value) is Config else value
    if type(limits) is not Limits:
        raise AdmissionError(ReasonCode.INVALID_CONFIG)
    fraction = limits.reserve_fraction
    if (type(fraction) not in (int, float)
            or (type(fraction) is float and not math.isfinite(fraction))
            or fraction < 0 or type(limits.reserve_floor_bytes) is not int
            or type(limits.reserve_ceiling_bytes) is not int
            or limits.reserve_floor_bytes < 0
            or limits.reserve_floor_bytes > limits.reserve_ceiling_bytes):
        raise AdmissionError(ReasonCode.INVALID_CONFIG)
    return limits


def _ratio(value: int | float) -> tuple[int, int]:
    # Never coerce an arbitrary-sized integer to a float or decimal string.
    return (value, 1) if type(value) is int else Decimal(str(value)).as_integer_ratio()


def reserve_bytes(capacity: int, config: Config | Limits) -> int:
    """Exact ceiling for the configured decimal fraction of capacity."""
    _bytes(capacity)
    limits = _limits(config)
    numerator, denominator = _ratio(limits.reserve_fraction)
    fraction = (capacity * numerator + denominator - 1) // denominator
    return max(limits.reserve_floor_bytes, min(limits.reserve_ceiling_bytes, fraction))


def peak_remaining(intervals: tuple[GrowthInterval, ...] | list[GrowthInterval]) -> int:
    """Peak simultaneous remaining growth, deduplicating reusable assets."""
    if type(intervals) not in (tuple, list) or any(type(x) is not GrowthInterval for x in intervals):
        raise AdmissionError()
    assets: dict[tuple[str, str], tuple[int | None, int, int | None]] = {}
    for item in intervals:
        asset = (item.volume_identity, item.asset_key)
        signature = (item.total_bytes, item.allocated_bytes, item.media_cap_bytes)
        if asset in assets and assets[asset] != signature:
            raise AdmissionError()
        assets[asset] = signature
    edges = sorted({edge for x in intervals for edge in (x.start, x.end)})
    peak = 0
    for point in edges[:-1]:
        active: dict[str, int] = {}
        for item in intervals:
            if item.start <= point < item.end:
                active[item.asset_key] = max(active.get(item.asset_key, 0), item.remaining_bytes)
        peak = max(peak, sum(active.values()))
    return peak


def _status(field: Measurement, now: datetime, max_age_seconds: int) -> ReasonCode | None:
    if type(field) is not Measurement:
        return ReasonCode.STORAGE_UNKNOWN
    if field.status is not FieldStatus.OK:
        return ReasonCode.STORAGE_STALE if field.status is FieldStatus.STALE else field.reason or ReasonCode.STORAGE_UNKNOWN
    try:
        stamp = datetime.fromisoformat(field.sample_time.replace('Z', '+00:00'))
        age = (now - stamp).total_seconds()
    except (TypeError, ValueError):
        return ReasonCode.STORAGE_UNKNOWN
    return None if 0 <= age <= max_age_seconds else ReasonCode.STORAGE_STALE


def _volume_reason(volume: VolumeEvidence, now: datetime, max_age: int, *, require_inodes: bool) -> ReasonCode | None:
    for name in ('identity', 'capacity', 'free', 'mounted', 'writable', 'network'):
        reason = _status(getattr(volume, name), now, max_age)
        if reason is not None:
            return reason
    if require_inodes or volume.free_inodes.status is FieldStatus.OK:
        reason = _status(volume.free_inodes, now, max_age)
        if reason is not None:
            return reason
    if (type(volume.identity.value) is not str or not volume.identity.value
            or type(volume.capacity.value) is not int or volume.capacity.value <= 0
            or type(volume.free.value) is not int or not 0 <= volume.free.value <= volume.capacity.value
            or volume.mounted.value is not True or volume.writable.value is not True
            or volume.network.value is not False
            or (require_inodes and (type(volume.free_inodes.value) is not int or volume.free_inodes.value <= 0))
            or (volume.free_inodes.status is FieldStatus.OK and
                (type(volume.free_inodes.value) is not int or volume.free_inodes.value <= 0))):
        return ReasonCode.STORAGE_UNKNOWN
    return None


def _measure(value: int | bool | None, now: str, reason: ReasonCode | None = None) -> Measurement:
    return Measurement(value, 'admission', FieldStatus.OK if reason is None else FieldStatus.ERROR,
                       now, reason)


def make_plan(
    volumes: tuple[VolumeEvidence, ...], intervals: tuple[GrowthInterval, ...],
    config: Config | Limits, *, now: datetime, docker: DockerStorageEvidence | None = None,
    other_active_reservations: Mapping[str, int] | None = None,
    inode_optional_identities: frozenset[str] = frozenset(),
    max_age_seconds: int = 3,
    state_evidence: VolumeEvidence | None = None,
) -> AdmissionPlan:
    """Plan one operation against each independent physical volume constraint.

    All aliases of a physical identity share one budget, using the tightest
    observed capacity/free.  Docker VM and its backing host volume remain two
    separate constraints; their free counters are never summed. This low-level
    API accepts custom intervals; use make_workload_plan for worker floors and
    placement validation. Explicit state evidence proves path eligibility;
    ordinary aliases still constrain space but need not prove SQLite safety.
    """
    if (type(volumes) is not tuple or not volumes or any(type(v) is not VolumeEvidence for v in volumes)
            or type(intervals) is not tuple or not intervals or any(type(x) is not GrowthInterval for x in intervals)
            or type(now) is not datetime or now.tzinfo is None
            or type(max_age_seconds) is not int or max_age_seconds <= 0
            or type(inode_optional_identities) is not frozenset):
        raise AdmissionError()
    limits = _limits(config)
    reservations = {} if other_active_reservations is None else other_active_reservations
    if (not isinstance(reservations, Mapping) or any(type(k) is not str or not k or
            type(v) is not int or v < 0 for k, v in reservations.items())):
        raise AdmissionError()
    stamp = now.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    if state_evidence is not None:
        if type(state_evidence) is not VolumeEvidence:
            raise AdmissionError()
        volumes = (*volumes, state_evidence)
    if docker is not None:
        if (type(docker) is not DockerStorageEvidence or type(docker.constraints) is not tuple
                or any(type(c) is not Constraint or type(c.volume) is not VolumeEvidence
                       for c in docker.constraints)):
            raise AdmissionError()
        volumes = (*volumes, *(c.volume for c in docker.constraints))
    if docker is None and any(x.asset_key.startswith('host:') for x in intervals):
        return AdmissionPlan(_measure(None, stamp, ReasonCode.BACKING_UNRESOLVED), (), intervals)
    grouped: dict[str, list[VolumeEvidence]] = {}
    for volume in volumes:
        identity = volume.identity.value
        if type(identity) is not str or not identity:
            # A missing identity cannot safely coalesce aliases.
            return AdmissionPlan(_measure(None, stamp, ReasonCode.STORAGE_UNKNOWN), (), intervals)
        grouped.setdefault(identity, []).append(volume)
    if any(x.volume_identity not in grouped for x in intervals) or any(k not in grouped for k in reservations):
        return AdmissionPlan(_measure(None, stamp, ReasonCode.STORAGE_UNKNOWN), (), intervals)
    docker_reason = None
    if docker is not None:
        if type(docker) is not DockerStorageEvidence:
            raise AdmissionError()
        docker_reason = _status(docker.admission, now, max_age_seconds)
        if docker_reason is None and docker.admission.value is not True:
            docker_reason = ReasonCode.STORAGE_UNKNOWN
        for link in docker.backing:
            for field in (link.vm_identity, link.host_identity, link.verified):
                docker_reason = docker_reason or _status(field, now, max_age_seconds)
            if docker_reason is None and (link.verified.value is not True or
                    link.vm_identity.value not in grouped or link.host_identity.value not in grouped):
                docker_reason = ReasonCode.BACKING_UNRESOLVED
        for host_item in (x for x in intervals if x.asset_key.startswith('host:')):
            source_key = host_item.asset_key.removeprefix('host:')
            vm_items = (x for x in intervals if x.asset_key == source_key and
                        x.start == host_item.start and x.end <= host_item.end)
            if not any(any(link.vm_identity.value == vm_item.volume_identity and
                           link.host_identity.value == host_item.volume_identity and
                           link.verified.value is True for link in docker.backing)
                       for vm_item in vm_items):
                docker_reason = docker_reason or ReasonCode.BACKING_UNRESOLVED
        for constraint in docker.constraints:
            if constraint.volume.identity.value not in grouped:
                docker_reason = docker_reason or ReasonCode.STORAGE_UNKNOWN
    requirements = []
    failure = docker_reason
    for identity, aliases in grouped.items():
        reason = next((r for v in aliases if (r := _volume_reason(v, now, max_age_seconds,
                       require_inodes=identity not in inode_optional_identities)) is not None), None)
        if reason is None and any(x.volume_identity == identity and x.asset_key in ('state', 'control')
                                  for x in intervals):
            state_proofs = ((state_evidence.state_allowed,)
                            if state_evidence is not None and state_evidence.identity.value == identity
                            else tuple(v.state_allowed for v in aliases
                                       if v.state_allowed.status is not FieldStatus.NOT_APPLICABLE))
            if not state_proofs:
                reason = ReasonCode.STORAGE_UNKNOWN
            else:
                reason = next((r for proof in state_proofs if (r := _status(proof, now, max_age_seconds))
                               is not None), None)
                if reason is None and any(proof.value is not True for proof in state_proofs):
                    reason = ReasonCode.STORAGE_UNKNOWN
        def tightest(name: str) -> Measurement:
            fields = tuple(getattr(v, name) for v in aliases)
            # Keep missing evidence as missing, and retain original provenance.
            return next((f for f in fields if type(f.value) is not int),
                        min(fields, key=lambda f: f.value if type(f.value) is int else 0))

        capacity_field, free_field = tightest('capacity'), tightest('free')
        capacity = capacity_field.value if type(capacity_field.value) is int else None
        free = free_field.value if type(free_field.value) is int else None
        required = peak_remaining(tuple(x for x in intervals if x.volume_identity == identity))
        reserved = reservations.get(identity, 0)
        reserve = reserve_bytes(capacity, limits) if capacity is not None and capacity >= 0 else None
        available = free - reserved - required if free is not None else None
        if reason is None and available is not None and reserve is not None and available < reserve:
            reason = ReasonCode.STORAGE_INSUFFICIENT
        failure = failure or reason
        unknown = reason or ReasonCode.STORAGE_UNKNOWN
        capacity_reason = next((r for v in aliases if (r := _status(v.capacity, now, max_age_seconds))
                                is not None), None)
        available_reason = next((r for v in aliases if (r := _status(v.free, now, max_age_seconds))
                                 is not None), None)
        requirements.append(VolumeRequirement(identity,
            capacity_field, free_field,
            _measure(reserve, stamp) if reserve is not None and capacity_reason is None
            else _measure(None, stamp, capacity_reason or unknown),
            _measure(reserved, stamp), _measure(required, stamp),
            _measure(max(0, available), stamp) if available is not None and available_reason is None
            else _measure(None, stamp, available_reason or unknown),
            reason or ReasonCode.OK))
    return AdmissionPlan(_measure(True, stamp) if failure is None else _measure(None, stamp, failure),
                         tuple(requirements), intervals)


def make_workload_plan(
    volumes: tuple[VolumeEvidence, ...], growth: WorkloadGrowth,
    placement: StoragePlacement, config: Config | Limits, *, now: datetime,
    docker: DockerStorageEvidence,
    state_evidence: VolumeEvidence,
    allocated: Mapping[str, int] | None = None,
    other_active_reservations: Mapping[str, int] | None = None,
    inode_optional_identities: frozenset[str] = frozenset(),
    host_growth_fraction: float = 0.10,
    max_age_seconds: int = 3,
) -> AdmissionPlan:
    """Complete worker plan: fixed floors, stages, Docker proof, and budgets.

    Generic native Docker discovery cannot distinguish image, snapshot, and
    runtime writes. Charge every Docker asset on every distinct discovered
    store. Flat allocation credits remain bound to their primary placements:
    image_store (or docker_vm) for images, docker_vm for worker/log allocations.
    """
    if (type(placement) is not StoragePlacement or type(docker) is not DockerStorageEvidence
            or type(state_evidence) is not VolumeEvidence
            or state_evidence.identity.value != placement.state
            or type(docker.constraints) is not tuple
            or any(type(c) is not Constraint or type(c.volume) is not VolumeEvidence
                   for c in docker.constraints)):
        raise AdmissionError()
    intervals = standard_growth_intervals(growth, placement, config,
                                          allocated=allocated,
                                          host_growth_fraction=host_growth_fraction)
    # A truthy discovery verdict alone is not an allocation map. Every actual
    # store must join the modeled placement. Generic containerd roles do not
    # prove a narrower allocation map, even when an image_store is supplied.
    roles: dict[str, set[str]] = {}
    for constraint in docker.constraints:
        if (type(constraint.roles) is not tuple or not constraint.roles
                or type(constraint.paths) is not tuple or not constraint.paths
                or any(type(p) is not str or not p for p in constraint.paths)):
            raise AdmissionError()
        for role in constraint.roles:
            if type(role) is not str or role not in ('docker_vm', 'docker_backing', 'docker_data', 'containerd'):
                raise AdmissionError()
            roles.setdefault(role, set()).add(constraint.volume.identity.value)
    if placement.docker_vm == placement.docker_backing:
        stores = roles.get('docker_data', set()) | roles.get('containerd', set())
        verified = (not docker.backing and roles.get('docker_data') == {placement.docker_vm}
                    and roles.keys() <= {'docker_data', 'containerd'}
                    and all(type(identity) is str and identity for identity in stores)
                    and (placement.image_store is None or placement.image_store in stores))
        if verified:
            # Each primary interval already carries its own volume-bound
            # credit. Replicated upper bounds have no allocation proof; a
            # matching asset key must never transfer credit to another store.
            docker_assets = {'image_compressed', 'image_unpacked', 'worker',
                             'container_metadata_and_logs'}
            intervals += tuple(replace(item, volume_identity=identity, allocated_bytes=0)
                               for identity in sorted(stores) for item in intervals
                               if item.asset_key in docker_assets and item.volume_identity != identity)
    else:
        verified = ((placement.image_store or placement.docker_vm) == placement.docker_vm
                    and roles == {'docker_vm': {placement.docker_vm},
                              'docker_backing': {placement.docker_backing}}
                    and any(link.vm_identity.value == placement.docker_vm
                            and link.host_identity.value == placement.docker_backing
                            and link.verified.value is True for link in docker.backing))
    plan = make_plan(volumes, intervals, config, now=now, docker=docker,
                     state_evidence=state_evidence,
                     other_active_reservations=other_active_reservations,
                     inode_optional_identities=inode_optional_identities,
                     max_age_seconds=max_age_seconds)
    if not verified:
        stamp = now.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
        return AdmissionPlan(_measure(None, stamp, ReasonCode.BACKING_UNRESOLVED),
                             plan.volumes, intervals)
    return plan


__all__ = ('AdmissionError', 'GrowthInterval', 'VolumeRequirement', 'AdmissionPlan',
           'StoragePlacement', 'WorkloadGrowth', 'standard_growth_intervals',
           'ExecutionLease', 'may_acquire_lease', 'reserve_bytes', 'peak_remaining',
           'make_plan', 'make_workload_plan',
           'WORKER_FLOOR', 'LOG_FLOOR', 'CONTROL_BOUND')
