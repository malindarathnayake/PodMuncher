"""Synthetic policy fixtures: no fit/support claims about real models.

Every model hash, memory peak, runtime and hardware fingerprint below is invented
test evidence, and selection tests use a synthetic hardware table (``TABLE``).
Only the ``model_hardware.json`` tests read the shipped table. Load/readmission
adapters are deterministic in-process fakes.
"""

import json
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from importlib import resources
from pathlib import Path

import pytest

from podmuncher.admission import AdmissionPlan
from podmuncher.domains import (
    Backend,
    Device,
    DomainValidationError,
    FieldStatus,
    ModelTier,
    Profile,
    ReasonCode,
)
from podmuncher.hardware import (
    AdmittedWorkerRef,
    HardwareEvidence,
    RuntimeComputeEvidence,
)
from podmuncher.model_policy import (
    DECODE_BYTES_PER_SECOND,
    GIB,
    CleanupResult,
    EnvelopeView,
    LoadAdmission,
    LoadProbeResult,
    MeasuredEnvelope,
    ModelArtifact,
    ModelPolicy,
    ModelPolicyError,
    PolicyEvidence,
    PolicyRequest,
    RuntimeFingerprint,
    WorkerEvidence,
    load_hardware_table,
    parse_hardware_table,
    select_model,
)
from podmuncher.snapshots import Measurement

NOW = datetime(2026, 10, 9, tzinfo=UTC)
STAMP = NOW.isoformat().replace('+00:00', 'Z')
GPU_UUID = 'GPU-12345678-1234-1234-1234-123456789012'
ARTIFACTS = tuple(ModelArtifact(name, tier, 'synthetic-revision', f'{index + 1:064x}')
                  for index, (name, tier) in enumerate(zip(
                      ('tiny', 'base', 'small', 'medium', 'large-v3'), ModelTier)))


def measure(value, *, stamp=STAMP, status=FieldStatus.OK):
    return Measurement(value if status is FieldStatus.OK else None, 'synthetic_fixture', status,
                       stamp, None if status is FieldStatus.OK else ReasonCode.STORAGE_UNKNOWN)


def worker(backend=Backend.CPU, *, limit=16 * GIB, available=8 * GIB,
           vram_total=16 * GIB, vram_free=10 * GIB, stamp=STAMP):
    gpu = backend is Backend.CUDA
    identity = AdmittedWorkerRef('synthetic', 'npipe:////./pipe/docker_engine',
                                ('b' if gpu else 'a') * 64, 'synthetic', 'job', 'attempt',
                                gpu, GPU_UUID if gpu else None)
    unknown = measure(None, status=FieldStatus.UNKNOWN)
    fields = {name: unknown for name in HardwareEvidence.__dataclass_fields__}
    compute = ('int8_float16',) if gpu else ('int8',)
    fields.update(worker_ram_limit=measure(limit, stamp=stamp),
                  worker_ram_current=measure(limit - available, stamp=stamp),
                  worker_ram_available=measure(available, stamp=stamp),
                  worker_gpu_visible=measure(gpu, stamp=stamp),
                  runtime_compute_types=measure(list(compute), stamp=stamp),
                  gpu_vram_total=measure(vram_total, stamp=stamp),
                  gpu_vram_free=measure(vram_free, stamp=stamp),
                  gpu_driver=measure('synthetic-driver', stamp=stamp))
    runtime = RuntimeComputeEvidence(identity, backend, compute, GPU_UUID if gpu else None)
    fingerprint = RuntimeFingerprint('sha256:' + 'd' * 64, 'synthetic-runtime',
                                     'synthetic-driver' if gpu else None, 'synthetic-hardware')
    return WorkerEvidence(identity, HardwareEvidence(**fields), runtime, fingerprint, stamp)


def envelope(artifact=ARTIFACTS[0], *, on=None, ram=GIB // 4, vram=None, duration=21600):
    on = worker() if on is None else on
    gpu = on.runtime.backend is Backend.CUDA
    return MeasuredEnvelope(artifact, on.fingerprint, on.runtime.backend,
                            'int8_float16' if gpu else 'int8', ram,
                            (GIB if gpu else 0) if vram is None else vram,
                            duration, DECODE_BYTES_PER_SECOND, 180,
                            'e' * 64, release_tested=True, synthetic=True)


def evidence(*workers, artifacts=ARTIFACTS):
    return PolicyEvidence(workers or (worker(),), artifacts)


def envelopes_for(sample):
    return tuple(envelope(artifact, on=on) for on in sample.workers for artifact in sample.artifacts)


def side(backend, peak, *, host=None, basis='published', source='https://example.com/synthetic'):
    if backend is Backend.CUDA:
        return {'compute_type': 'int8_float16', 'peak_vram_bytes': peak, 'peak_host_ram_bytes': host,
                'basis': basis, 'source': source, 'conditions': 'synthetic'}
    return {'compute_type': 'int8', 'peak_ram_bytes': peak, 'basis': basis, 'source': source,
            'conditions': 'synthetic'}


def table_doc(rows=None):
    """Synthetic table: rank 10..50, CPU RAM and CUDA VRAM peaks 1..5 GiB."""
    if rows is None:
        rows = [{'model_id': a.model_id, 'tier': a.tier.value, 'rank': (i + 1) * 10,
                 'cuda': side(Backend.CUDA, (i + 1) * GIB), 'cpu': side(Backend.CPU, (i + 1) * GIB)}
                for i, a in enumerate(ARTIFACTS)]
    return {'schema_version': 1, 'updated': '2026-10-14',
            'headroom': {'ram_fraction': 0.2, 'vram_fraction': 0.2, 'vram_min_bytes': GIB},
            'models': rows}


TABLE = parse_hardware_table(table_doc())


def table_with(model_id, backend, peak):
    doc = table_doc()
    for row in doc['models']:
        if row['model_id'] == model_id:
            row[backend.value]['peak_vram_bytes' if backend is Backend.CUDA else 'peak_ram_bytes'] = peak
    return parse_hardware_table(doc)


def request(**kwargs):
    return PolicyRequest(**({'duration_seconds': 10, 'worker_ram_cap_bytes': 32 * GIB} | kwargs))


def choose(sample=None, envelopes=(), table=TABLE, lower_than=None, **kwargs):
    sample = evidence() if sample is None else sample
    return select_model(request(**kwargs), sample, envelopes, now=NOW, table=table, lower_than=lower_than)


def reason(expected, callback):
    with pytest.raises(ModelPolicyError) as error:
        callback()
    assert error.value.reason_code is expected


DECODE_10S = 10 * DECODE_BYTES_PER_SECOND
RESERVE_16G = -(-16 * GIB // 5)


# ------------------------------------------------------------------ automatic selection


@pytest.mark.parametrize(('extra', 'expected'), [
    (0, 'medium'), (-1, 'small'), (GIB - 1, 'medium'), (GIB, 'large-v3'),
])
def test_cpu_auto_picks_highest_rank_that_fits_exact_and_one_byte_short(extra, expected):
    available = 4 * GIB + DECODE_10S + RESERVE_16G + extra
    choice = choose(evidence(worker(limit=16 * GIB, available=available)))
    assert choice.artifact.model_id == expected and choice.backend is Backend.CPU
    assert choice.reason == 'auto_largest_fit' and choice.basis == 'published'
    assert choice.required_ram_bytes <= available


@pytest.mark.parametrize(('free', 'expected'), [
    (6 * GIB, 'medium'), (6 * GIB - 1, 'small'), (7 * GIB, 'large-v3'), (3 * GIB, 'tiny'),
])
def test_cuda_auto_picks_highest_rank_that_fits_vram_with_twenty_percent_reserve(free, expected):
    choice = choose(evidence(worker(Backend.CUDA, vram_total=10 * GIB, vram_free=free)))
    assert choice.artifact.model_id == expected and choice.backend is Backend.CUDA
    assert choice.compute_type == 'int8_float16' and choice.device_reserve_bytes == 2 * GIB
    assert choice.required_vram_bytes == choice.peak_vram_bytes + 2 * GIB <= free


@pytest.mark.parametrize(('total', 'free', 'reserve'), [(10 * GIB, 3 * GIB, 2 * GIB), (4 * GIB, 2 * GIB, GIB)])
def test_vram_reserve_is_max_of_twenty_percent_or_one_gib(total, free, reserve):
    on = worker(Backend.CUDA, vram_total=total, vram_free=free)
    choice = choose(evidence(on))
    assert choice.device_reserve_bytes == reserve and choice.artifact.model_id == 'tiny'
    short = replace(on, hardware=replace(on.hardware, gpu_vram_free=measure(free - 1)))
    reason(ReasonCode.GPU_UNAVAILABLE, lambda: choose(evidence(short)))


def test_auto_device_tries_gpu_first_then_cpu_but_explicit_cuda_never_uses_cpu():
    roomy = evidence(worker(Backend.CUDA, vram_free=10 * GIB), worker())
    assert choose(roomy).backend is Backend.CUDA
    sample = evidence(worker(Backend.CUDA, vram_free=3 * GIB - 1), worker())
    assert choose(sample).backend is Backend.CPU
    reason(ReasonCode.GPU_UNAVAILABLE, lambda: choose(sample, device='cuda'))
    reason(ReasonCode.GPU_UNAVAILABLE, lambda: choose(evidence(worker()), device='cuda'))


def test_speed_and_profile_never_change_the_memory_based_pick():
    sample = evidence(worker(limit=16 * GIB, available=8 * GIB))
    assert choose(sample, profile='quality') == choose(sample, profile='balanced')


def test_unknown_requirement_is_never_auto_picked():
    roomy = evidence(worker(limit=64 * GIB, available=64 * GIB))
    assert choose(roomy, table=table_with('large-v3', Backend.CPU, None)).artifact.model_id == 'medium'
    gpu = evidence(worker(Backend.CUDA, vram_free=12 * GIB))
    assert choose(gpu, table=table_with('large-v3', Backend.CUDA, None)).artifact.model_id == 'medium'
    # A verified artifact with no table row and no measured envelope is unknown.
    custom = ModelArtifact('verified-custom-id', 'large', 'synthetic-revision', 'f' * 64)
    reason(ReasonCode.ENVELOPE_UNVALIDATED, lambda: choose(evidence(worker(), artifacts=(custom,))))
    empty = parse_hardware_table(table_doc([]))
    reason(ReasonCode.ENVELOPE_UNVALIDATED, lambda: choose(roomy, table=empty))


def test_cuda_host_ram_requirement_and_decode_still_checked():
    rows = table_doc()
    for row in rows['models']:
        row['cuda']['peak_host_ram_bytes'] = 20 * GIB
    reason(ReasonCode.RAM_INSUFFICIENT,
           lambda: choose(evidence(worker(Backend.CUDA)), table=parse_hardware_table(rows)))


def test_reserves_round_up_and_decode_is_ceil_at_least_256000_bytes_per_second():
    on = worker(limit=5 * GIB + 1, available=5 * GIB + 1)
    choice = choose(evidence(on), duration_seconds=1 / 512000)
    assert choice.worker_reserve_bytes == GIB + 1
    assert choice.decode_bytes == 1
    reason(ReasonCode.INVALID_INPUT, lambda: replace(envelope(), decode_bytes_per_second=255999))


def test_whole_file_decode_blocks_long_media_for_auto():
    on = worker(limit=4 * GIB, available=4 * GIB)
    reason(ReasonCode.RAM_INSUFFICIENT, lambda: choose(evidence(on), duration_seconds=21600))
    large = worker(limit=32 * GIB, available=32 * GIB)
    assert choose(evidence(large), duration_seconds=21600).decode_bytes == 21600 * 256000


def test_unknown_and_excessive_duration():
    reason(ReasonCode.DURATION_UNKNOWN, lambda: choose(duration_seconds=None))
    reason(ReasonCode.DURATION_EXCEEDED, lambda: choose(duration_seconds=21601))
    sample = evidence(worker(limit=32 * GIB, available=32 * GIB))
    # Table requirements carry no validated duration: only the request limit applies.
    assert choose(sample, duration_seconds=21601, duration_limit_seconds=30000)


@pytest.mark.parametrize('value', [True, False, 0, -1, float('nan'), float('inf'), '21600', 10**500])
@pytest.mark.parametrize('field', ['duration_seconds', 'duration_limit_seconds', 'decode_timeout_seconds'])
def test_invalid_finite_duration_inputs(field, value):
    with pytest.raises(DomainValidationError):
        request(**{field: value})


def test_request_cap_bounds_capacity():
    sample = evidence(worker(limit=64 * GIB, available=64 * GIB))
    choice = choose(sample, worker_ram_cap_bytes=4 * GIB, duration_seconds=1)
    assert choice.artifact.model_id == 'small'
    assert choice.worker_memory_cap_bytes == 4 * GIB
    assert choice.decode_timeout_seconds == 300
    reason(ReasonCode.RAM_INSUFFICIENT, lambda: choose(sample, worker_ram_cap_bytes=GIB))


# ------------------------------------------------------------------ OOM fallback


def test_lower_than_picks_the_next_lower_rank():
    sample = evidence(worker(limit=64 * GIB, available=64 * GIB))
    first = choose(sample)
    assert first.artifact.model_id == 'large-v3'
    second = choose(sample, lower_than=first)
    assert second.artifact.model_id == 'medium' and second.reason == 'oom_lower_retry'
    tiny = choose(sample, lower_than=choose(sample, lower_than=choose(sample, lower_than=second)))
    assert tiny.artifact.model_id == 'tiny'
    reason(ReasonCode.MODEL_OOM, lambda: choose(sample, lower_than=tiny))
    reason(ReasonCode.MODEL_OOM, lambda: choose(sample, lower_than=first, model='small'))


def test_oom_retry_with_no_lower_candidate_ends_as_model_oom():
    sample = evidence(worker(limit=64 * GIB, available=64 * GIB))
    only_tiny = evidence(worker(limit=64 * GIB, available=64 * GIB), artifacts=ARTIFACTS[:1])
    tiny = choose(only_tiny)
    assert tiny.artifact.model_id == 'tiny'
    reason(ReasonCode.MODEL_OOM, lambda: choose(only_tiny, lower_than=tiny))
    reason(ReasonCode.MODEL_OOM, lambda: choose(sample, lower_than=tiny))
    reason(ReasonCode.MODEL_OOM, lambda: choose(sample, lower_than=tiny, device='cpu'))


# ------------------------------------------------------------------ explicit operator choice


def test_explicit_model_skips_fit_check():
    tight = evidence(worker(limit=GIB, available=GIB))
    reason(ReasonCode.RAM_INSUFFICIENT, lambda: choose(tight))
    choice = choose(tight, model='large-v3', duration_seconds=21600)
    assert choice.artifact.model_id == 'large-v3' and choice.basis == 'explicit'
    assert choice.reason == 'explicit_model' and choice.backend is Backend.CPU
    assert choice.compute_type == 'int8' and choice.peak_ram_bytes == 5 * GIB
    gpu = evidence(worker(Backend.CUDA, vram_total=4 * GIB, vram_free=GIB))
    assert choose(gpu, model='large-v3', device='cuda').compute_type == 'int8_float16'


def test_explicit_unknown_requirement_is_allowed_and_reported_unknown():
    custom = ModelArtifact('verified-custom-id', 'large', 'synthetic-revision', 'f' * 64)
    choice = choose(evidence(worker(), artifacts=(custom,)), model=custom.model_id)
    assert choice.artifact == custom and choice.basis == 'explicit'
    assert choice.peak_ram_bytes is None and choice.required_ram_bytes is None
    assert choice.source.startswith('operator choice')


def test_explicit_model_must_be_verified_and_is_never_substituted():
    reason(ReasonCode.ASSET_VERIFICATION_FAILED, lambda: choose(model='unknown-model'))
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: choose(evidence(worker(), artifacts=ARTIFACTS[:2]), model='large-v3'))


def test_explicit_model_with_auto_device_prefers_visible_gpu_else_cpu():
    both = evidence(worker(Backend.CUDA, vram_total=4 * GIB, vram_free=GIB), worker())
    assert choose(both, model='large-v3').backend is Backend.CUDA
    assert choose(evidence(worker()), model='large-v3').backend is Backend.CPU
    on = worker(Backend.CUDA)
    hidden = replace(on, hardware=replace(on.hardware, gpu_vram_free=measure(None, status=FieldStatus.UNKNOWN)))
    assert choose(evidence(hidden, worker()), model='large-v3').backend is Backend.CPU


def test_explicit_model_still_needs_a_usable_backend():
    on = worker()
    stale = replace(on, hardware=replace(on.hardware, worker_ram_available=measure(
        GIB, stamp=(NOW - timedelta(seconds=11)).isoformat())))
    reason(ReasonCode.RAM_INSUFFICIENT, lambda: choose(evidence(stale), model='tiny'))
    reason(ReasonCode.GPU_UNAVAILABLE, lambda: choose(evidence(worker()), model='tiny', device='cuda'))


# ------------------------------------------------------------------ measured envelopes


def test_measured_envelope_beats_the_table():
    on = worker(limit=16 * GIB, available=8 * GIB)
    sample = evidence(on)
    assert choose(sample).artifact.model_id == 'medium'
    big = envelope(ARTIFACTS[3], on=on, ram=6 * GIB)
    choice = choose(sample, envelopes=(big,))
    assert choice.artifact.model_id == 'small' and choice.basis == 'published'
    small = envelope(ARTIFACTS[4], on=on, ram=GIB)
    choice = choose(sample, envelopes=(small,))
    assert choice.artifact.model_id == 'large-v3' and choice.basis == 'measured'
    assert choice.measured_envelope == small and choice.decode_timeout_seconds == 180
    assert choice.source == 'measured envelope sha256:' + 'e' * 64


def test_measured_envelope_validated_duration_applies_to_auto_only():
    # P2-U6 release correction: a too-short envelope no longer refuses the
    # model for auto; the table row applies instead (basis never 'measured').
    on = worker(limit=64 * GIB, available=64 * GIB)
    short = envelope(ARTIFACTS[4], on=on, ram=GIB, duration=60)
    choice = choose(evidence(on), envelopes=(short,), duration_seconds=61)
    assert choice.artifact.model_id == 'large-v3' and choice.basis == 'published'
    assert choice.measured_envelope is None and choice.peak_ram_bytes == 5 * GIB
    explicit = choose(evidence(on), envelopes=(short,), duration_seconds=61, model='large-v3')
    assert explicit.measured_envelope == short and explicit.basis == 'explicit'


def test_duration_within_envelope_uses_measured_else_table_requirement_decides_fit():
    on = worker(limit=16 * GIB, available=8 * GIB)
    short = envelope(ARTIFACTS[4], on=on, ram=GIB, duration=600)
    within = choose(evidence(on), envelopes=(short,), duration_seconds=600)
    assert within.artifact.model_id == 'large-v3' and within.basis == 'measured'
    assert within.measured_envelope == short and within.decode_timeout_seconds == 180
    # Above the envelope: the table's 5 GiB large-v3 row no longer fits 8 GiB
    # with decode and reserve, so auto drops to medium on the table basis.
    beyond = choose(evidence(on), envelopes=(short,), duration_seconds=601)
    assert beyond.artifact.model_id == 'medium' and beyond.basis == 'published'
    assert beyond.measured_envelope is None and beyond.decode_timeout_seconds == 300


def test_duration_above_envelope_without_table_row_keeps_the_refusal():
    on = worker(limit=64 * GIB, available=64 * GIB)
    custom = ModelArtifact('verified-custom-id', 'large', 'synthetic-revision', 'f' * 64)
    short = envelope(custom, on=on, ram=GIB, duration=60)
    assert choose(evidence(on, artifacts=(custom,)), envelopes=(short,), duration_seconds=60).basis == 'measured'
    reason(ReasonCode.ENVELOPE_UNVALIDATED,
           lambda: choose(evidence(on, artifacts=(custom,)), envelopes=(short,), duration_seconds=61))
    unknown_row = table_with('large-v3', Backend.CPU, None)
    long_large = envelope(ARTIFACTS[4], on=on, ram=GIB, duration=60)
    reason(ReasonCode.ENVELOPE_UNVALIDATED,
           lambda: choose(evidence(on, artifacts=ARTIFACTS[4:]), envelopes=(long_large,),
                          duration_seconds=61, table=unknown_row))
    explicit = choose(evidence(on, artifacts=(custom,)), envelopes=(short,), duration_seconds=61,
                      model=custom.model_id)
    assert explicit.basis == 'explicit' and explicit.measured_envelope == short


@pytest.mark.parametrize('field', ['image_digest', 'runtime_version', 'driver_version', 'hardware_fingerprint'])
def test_runtime_fingerprint_change_drops_envelope_back_to_table(field):
    on = worker(Backend.CUDA)
    original = envelopes_for(evidence(on))
    value = 'sha256:' + 'f' * 64 if field == 'image_digest' else 'synthetic-changed'
    changed = replace(on, fingerprint=replace(on.fingerprint, **{field: value}))
    if field == 'driver_version':
        changed = replace(changed, hardware=replace(changed.hardware, gpu_driver=measure(value)))
    assert choose(evidence(on), envelopes=original).basis == 'measured'
    choice = choose(evidence(changed), envelopes=original, device='cuda')
    assert choice.basis == 'published' and choice.measured_envelope is None


@pytest.mark.parametrize('field', ['revision', 'sha256'])
def test_model_revision_or_hash_change_drops_envelope(field):
    value = 'f' * 64 if field == 'sha256' else 'synthetic-new-revision'
    changed = replace(ARTIFACTS[0], **{field: value})
    choice = choose(evidence(artifacts=(changed,)), envelopes=(envelope(),), model='tiny')
    assert choice.measured_envelope is None and choice.artifact == changed


def test_duplicate_envelopes_dedupe_but_conflicting_ones_make_requirement_unknown():
    on = worker(limit=64 * GIB, available=64 * GIB)
    sample = evidence(on)
    small = envelope(ARTIFACTS[4], on=on, ram=GIB)
    twin = envelope(ARTIFACTS[4], on=on, ram=GIB)
    assert small == twin and small is not twin
    choice = choose(sample, envelopes=(small, twin))
    assert choice.artifact.model_id == 'large-v3' and choice.basis == 'measured'
    assert choice.measured_envelope == small
    conflicting = envelope(ARTIFACTS[4], on=on, ram=2 * GIB)
    choice = choose(sample, envelopes=(small, conflicting))
    assert choice.artifact.model_id == 'medium' and choice.basis == 'published'
    reason(ReasonCode.ENVELOPE_UNVALIDATED,
           lambda: choose(evidence(on, artifacts=ARTIFACTS[4:]), envelopes=(small, conflicting)))
    explicit = choose(sample, envelopes=(small, conflicting), model='large-v3')
    assert explicit.artifact.model_id == 'large-v3' and explicit.basis == 'explicit'
    assert explicit.measured_envelope is None and explicit.peak_ram_bytes is None
    assert explicit.source.startswith('operator choice')


def test_envelope_construction_still_validated():
    reason(ReasonCode.ENVELOPE_UNVALIDATED, lambda: replace(envelope(), release_tested=False))
    reason(ReasonCode.INVALID_INPUT, lambda: replace(envelope(), worker_peak_vram_bytes=1))


# ------------------------------------------------------------------ compatibility shim


def test_envelope_view_shim_and_protected_record():
    choice = choose(evidence(worker(Backend.CUDA)), device='cuda')
    view = choice.envelope
    assert type(view) is EnvelopeView
    assert (view.backend, view.artifact, view.compute_type) == (choice.backend, choice.artifact, 'int8_float16')
    assert view.worker_peak_vram_bytes == choice.peak_vram_bytes and view.worker_peak_ram_bytes == 0
    data = choice.to_dict()
    assert data['basis'] == 'published' and data['measured_envelope'] is None
    json.dumps(data)


# ------------------------------------------------------------------ evidence joins (unchanged)


@pytest.mark.parametrize('field', ['worker_ram_limit', 'worker_ram_current', 'worker_ram_available', 'runtime_compute_types', 'worker_gpu_visible'])
def test_unknown_and_stale_required_hardware_block_asr(field):
    on = worker()
    original = getattr(on.hardware, field).to_dict()
    for replacement in (measure(None, status=FieldStatus.UNKNOWN),
                        Measurement.from_dict(original | {'sample_time': (NOW - timedelta(seconds=11)).isoformat()}),
                        Measurement.from_dict(original | {'sample_time': (NOW + timedelta(seconds=1)).isoformat()})):
        sample = evidence(replace(on, hardware=replace(on.hardware, **{field: replacement})))
        with pytest.raises(ModelPolicyError):
            choose(sample)
        with pytest.raises(ModelPolicyError):
            choose(sample, model='tiny')


@pytest.mark.parametrize('bad', [True, -1, 1.5, '5000'])
def test_invalid_worker_measurement_values_never_borrow_host_ram(bad):
    on = worker()
    hardware = replace(on.hardware, worker_ram_available=measure(bad), host_ram_available=measure(1000 * GIB))
    with pytest.raises(ModelPolicyError):
        choose(evidence(replace(on, hardware=hardware)))


def test_inconsistent_cgroup_available_and_over_limit_current_block():
    on = worker(limit=4 * GIB, available=2 * GIB)
    for values in ({'worker_ram_available': measure(3 * GIB)}, {'worker_ram_current': measure(5 * GIB)}):
        reason(ReasonCode.RAM_INSUFFICIENT,
               lambda values=values: choose(evidence(replace(on, hardware=replace(on.hardware, **values)))))


def test_runtime_compute_identity_backend_and_timestamp_are_strictly_joined():
    on = worker()
    alien = replace(on.identity, job='another-job')
    reason(ReasonCode.WORKER_PROTOCOL_MISMATCH,
           lambda: replace(on, runtime=replace(on.runtime, identity=alien)))
    wrong_types = replace(on, hardware=replace(on.hardware, runtime_compute_types=measure(['float32'])))
    reason(ReasonCode.ENVELOPE_UNVALIDATED, lambda: choose(evidence(wrong_types)))
    stale = replace(on, runtime_sample_time=(NOW - timedelta(seconds=11)).isoformat())
    reason(ReasonCode.ENVELOPE_UNVALIDATED, lambda: choose(evidence(stale)))
    cuda = worker(Backend.CUDA)
    with pytest.raises(ValueError):
        replace(cuda.runtime, gpu_uuid=GPU_UUID.replace('1234', '5678'))


def test_unsupported_compute_type_makes_backend_unusable():
    on = worker()
    floats = ('float32',)
    on = replace(on, runtime=replace(on.runtime, compute_types=floats),
                 hardware=replace(on.hardware, runtime_compute_types=measure(list(floats))))
    reason(ReasonCode.ENVELOPE_UNVALIDATED, lambda: choose(evidence(on)))


def test_cuda_accepts_any_live_driver_but_envelopes_still_join_on_the_tested_driver():
    on = worker(Backend.CUDA, vram_total=16 * GIB, vram_free=10 * GIB)
    assert on.fingerprint.driver_version == 'synthetic-driver'
    other = replace(on, hardware=replace(on.hardware, gpu_driver=measure('synthetic-other-driver')))
    choice = choose(evidence(other), device='cuda')
    assert choice.backend is Backend.CUDA and choice.basis == 'published'
    assert choose(evidence(other), model='tiny', device='cuda').basis == 'explicit'
    # The manifest driver is "tested with" evidence: the envelope keyed on the
    # fingerprint still applies when only the live reading differs.
    measured = choose(evidence(other), envelopes=envelopes_for(evidence(on)), device='cuda')
    assert measured.basis == 'measured' and measured.measured_envelope is not None


@pytest.mark.parametrize('bad', ['unknown', 'stale', 'future'])
def test_missing_or_stale_driver_reading_still_blocks_cuda(bad):
    on = worker(Backend.CUDA)
    if bad == 'unknown':
        field = measure(None, status=FieldStatus.UNKNOWN)
    else:
        offset = timedelta(seconds=-11) if bad == 'stale' else timedelta(seconds=1)
        field = measure('synthetic-driver', stamp=(NOW + offset).isoformat())
    sample = evidence(replace(on, hardware=replace(on.hardware, gpu_driver=field)))
    reason(ReasonCode.GPU_UNAVAILABLE, lambda: choose(sample, device='cuda'))
    reason(ReasonCode.GPU_UNAVAILABLE, lambda: choose(sample, model='tiny', device='cuda'))


def test_unknown_host_fields_are_irrelevant_and_driver_reserved_vram_is_allowed():
    on = worker(Backend.CUDA, vram_total=16 * GIB, vram_free=10 * GIB)
    on = replace(on, hardware=replace(on.hardware, gpu_vram_used=measure(4 * GIB)))
    assert choose(evidence(on)).backend is Backend.CUDA


def test_contract_domains_reject_negative_examples_and_non_string_coercion():
    domains = json.loads((Path(__file__).parents[1] / 'Docs' / 'contract-domains.json').read_text())
    for key in ('device', 'profile', 'output_mode'):
        for bad in domains['negative_examples'] + [True, 1, None, 'CPU']:
            with pytest.raises(DomainValidationError):
                request(**{key: bad})
    for bad in domains['negative_examples'] + [True, 1, None]:
        with pytest.raises(DomainValidationError):
            replace(ARTIFACTS[0], tier=bad)
        doc = table_doc()
        doc['models'][0]['tier'] = bad
        with pytest.raises(DomainValidationError):
            parse_hardware_table(doc)
    assert request(device='cpu', profile='quality').device is Device.CPU
    assert request(device='cpu', profile='quality').profile is Profile.QUALITY


# ------------------------------------------------------------------ model_hardware.json


def test_shipped_table_is_well_formed_and_sourced():
    raw = json.loads(resources.files('podmuncher').joinpath('assets', 'model_hardware.json').read_text('utf-8'))
    table = load_hardware_table()
    assert table == parse_hardware_table(raw)
    assert {x.model_id for x in table.models} == {
        'tiny', 'base', 'small', 'medium', 'large-v3', 'large-v3-turbo', 'distil-large-v3', 'large-v2'}
    assert table.ram_fraction == Fraction(1, 5) and table.vram_fraction == Fraction(1, 5)
    assert table.vram_min_bytes == GIB
    ranks = {x.model_id: x.rank for x in table.models}
    assert ranks['tiny'] < ranks['base'] < ranks['small'] < ranks['medium'] < ranks['large-v3']
    assert max(ranks, key=ranks.get) == 'large-v3'
    for row in table.models:
        for need in (row.cuda, row.cpu):
            assert need.basis in {'published', 'estimated', 'measured'}
            if need.basis == 'measured':
                assert need.source == 'podmuncher measured 2026-10-13'
            else:
                assert need.source.startswith('https://')
        assert row.cuda.peak_vram_bytes and row.cpu.peak_ram_bytes  # every shipped row is known
        assert row.cuda.compute_type == 'int8_float16' and row.cpu.compute_type == 'int8'


def test_shipped_table_drives_default_selection():
    choice = select_model(request(), evidence(worker(limit=16 * GIB, available=16 * GIB)), now=NOW)
    assert choice.artifact.model_id == 'large-v3' and choice.basis == 'estimated'
    assert choice.source.startswith('https://')


def _mutate(path, value):
    doc = table_doc()
    target = doc
    for key in path[:-1]:
        target = target[key]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return doc


_DELETE = object()


@pytest.mark.parametrize(('path', 'value'), [
    (('schema_version',), 2), (('schema_version',), True), (('updated',), _DELETE), (('extra',), 1),
    (('headroom', 'ram_fraction'), 0), (('headroom', 'vram_fraction'), 1.5),
    (('headroom', 'vram_min_bytes'), 0), (('models',), {}),
    (('models', 0, 'rank'), 0), (('models', 0, 'rank'), True), (('models', 0, 'model_id'), ''),
    (('models', 0, 'cuda'), _DELETE), (('models', 0, 'cpu', 'peak_ram_bytes'), 0),
    (('models', 0, 'cpu', 'peak_ram_bytes'), -1), (('models', 0, 'cpu', 'peak_ram_bytes'), 1.5),
    (('models', 0, 'cpu', 'peak_ram_bytes'), True), (('models', 0, 'cuda', 'peak_vram_bytes'), '1'),
    (('models', 0, 'cpu', 'basis'), 'guessed'), (('models', 0, 'cpu', 'source'), 'http://example.com'),
    (('models', 0, 'cpu', 'source'), 'podmuncher measured 2026-10-13'),
    (('models', 0, 'cpu', 'conditions'), _DELETE), (('models', 0, 'cpu', 'peak_vram_bytes'), 1),
    (('models', 1, 'model_id'), 'tiny'), (('models', 1, 'rank'), 10),
])
def test_table_validation_rejects_malformed_rows(path, value):
    with pytest.raises(DomainValidationError):
        parse_hardware_table(_mutate(path, value))


@pytest.mark.parametrize('value', [['published'], {'basis': 'published'}, ('measured',), 1, None])
def test_table_rejects_non_string_basis_with_the_table_error(value):
    with pytest.raises(ModelPolicyError) as error:
        parse_hardware_table(_mutate(('models', 0, 'cpu', 'basis'), value))
    assert error.value.reason_code is ReasonCode.INVALID_INPUT


def test_table_row_with_other_compute_type_is_unknown_for_that_backend():
    doc = table_doc()
    for row in doc['models']:
        if row['model_id'] == 'large-v3':
            row['cpu']['compute_type'] = 'float32'
            row['cuda']['compute_type'] = 'float16'
    table = parse_hardware_table(doc)
    assert table.row('large-v3').cpu.compute_type == 'float32'
    roomy = evidence(worker(limit=64 * GIB, available=64 * GIB))
    assert choose(roomy, table=table).artifact.model_id == 'medium'
    gpu = evidence(worker(Backend.CUDA, vram_total=64 * GIB, vram_free=60 * GIB))
    assert choose(gpu, table=table, device='cuda').artifact.model_id == 'medium'
    reason(ReasonCode.ENVELOPE_UNVALIDATED,
           lambda: choose(evidence(worker(limit=64 * GIB, available=64 * GIB), artifacts=ARTIFACTS[4:]), table=table))
    explicit = choose(roomy, table=table, model='large-v3')
    assert explicit.basis == 'explicit' and explicit.compute_type == 'int8'
    assert explicit.peak_ram_bytes is None and explicit.source.startswith('operator choice')
    on = worker(limit=64 * GIB, available=64 * GIB)
    measured = choose(evidence(on), envelopes=(envelope(ARTIFACTS[4], on=on, ram=GIB),), table=table)
    assert measured.artifact.model_id == 'large-v3' and measured.basis == 'measured'


def test_table_accepts_null_peaks_and_measured_non_url_source():
    doc = _mutate(('models', 0, 'cpu', 'peak_ram_bytes'), None)
    doc['models'][1]['cpu'].update(basis='measured', source='podmuncher measured 2026-10-13')
    table = parse_hardware_table(doc)
    assert table.models[0].cpu.peak_ram_bytes is None
    assert table.models[1].cpu.basis == 'measured'
    assert table.row('missing') is None


class Lifecycle:
    """Synthetic bounded adapters; no subprocess, storage or inference occurs."""
    def __init__(self, sample=None, outcomes=(ReasonCode.OK,)):
        self.evidence = evidence() if sample is None else sample
        self.outcomes = list(outcomes)
        self.events = []
        self.admissions = []
        self.loads = []
        self.cleanup_transform = lambda value: value
        self.readmit_transform = lambda value: value
        self.result_transform = lambda value: value
        self.sample_hook = lambda: None

    def sample(self, *, deadline):
        assert deadline > 100
        self.events.append('sample')
        self.sample_hook()
        return self.evidence

    def readmit(self, choice, *, deadline):
        assert deadline > 100
        self.events.append('readmit')
        self.admissions.append(choice)
        # Synthetic trusted planner result. Storage arithmetic is tested in U3.
        storage = AdmissionPlan(measure(True), (), ())
        return self.readmit_transform(LoadAdmission(choice, self.evidence, storage, True))

    def load(self, choice, *, deadline):
        assert deadline > 100
        self.events.append('load')
        self.loads.append(choice)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return self.result_transform(LoadProbeResult(
            choice, outcome, choice.envelope.worker_peak_ram_bytes,
            choice.envelope.worker_peak_vram_bytes, choice.worker_memory_cap_bytes,
            choice.decode_timeout_seconds))

    def cleanup(self, choice, *, deadline):
        assert deadline > 100
        self.events.append('cleanup')
        return self.cleanup_transform(CleanupResult(choice.identity, True, True, True))


def coordinator(lifecycle, **kwargs):
    envelopes = kwargs.pop('envelopes', envelopes_for(lifecycle.evidence))
    return ModelPolicy(envelopes, lifecycle,
                       **({'clock': lambda: NOW, 'monotonic': lambda: 100, 'table': TABLE} | kwargs))


def test_bounded_load_is_required_and_persists_provenance():
    lifecycle = Lifecycle()
    loaded = coordinator(lifecycle).load(request(profile='quality'))
    assert lifecycle.events == ['sample', 'readmit', 'load']
    assert loaded.choice.artifact.model_id == 'large-v3'
    data = loaded.to_dict()
    assert data['attempts'] == 1
    assert data['choice']['reason'] == 'auto_largest_fit' and data['choice']['basis'] == 'measured'
    assert data['choice']['measured_envelope']['synthetic'] is True
    assert data['choice']['artifact']['revision'] == 'synthetic-revision'
    assert data['choice']['artifact']['sha256'] == ARTIFACTS[4].sha256
    assert data['choice']['measured_envelope']['runtime']['image_digest'] == 'sha256:' + 'd' * 64
    assert data['observations']['peak_ram_bytes'] > 0
    json.dumps(data)


def test_competing_allocation_between_select_and_load_blocks_without_substitution():
    lifecycle = Lifecycle()
    on = lifecycle.evidence.workers[0]
    busy = replace(on, hardware=replace(on.hardware, worker_ram_current=measure(15 * GIB),
                                       worker_ram_available=measure(GIB)))
    lifecycle.readmit_transform = lambda adm: replace(adm, evidence=evidence(busy))
    reason(ReasonCode.RAM_INSUFFICIENT, lambda: coordinator(lifecycle).load(request()))
    assert lifecycle.events == ['sample', 'readmit']


def test_changed_worker_identity_between_selection_and_load_blocks():
    lifecycle = Lifecycle()
    on = lifecycle.evidence.workers[0]
    identity = replace(on.identity, container_id='c' * 64)
    changed = replace(on, identity=identity, runtime=replace(on.runtime, identity=identity))
    lifecycle.readmit_transform = lambda adm: replace(adm, evidence=evidence(changed))
    reason(ReasonCode.ENVELOPE_UNVALIDATED, lambda: coordinator(lifecycle).load(request()))
    assert not lifecycle.loads


@pytest.mark.parametrize('failure', ['stale-storage', 'denied-storage', 'lease', 'stale-memory', 'runtime-change'])
def test_fresh_complete_readmission_is_required(failure):
    lifecycle = Lifecycle()
    def change(adm):
        if failure == 'stale-storage':
            return replace(adm, storage=replace(adm.storage, admission=measure(True, stamp=(NOW - timedelta(seconds=11)).isoformat())))
        if failure == 'denied-storage':
            field = Measurement(None, 'synthetic', FieldStatus.ERROR, STAMP, ReasonCode.STORAGE_INSUFFICIENT)
            return replace(adm, storage=replace(adm.storage, admission=field))
        if failure == 'lease':
            return replace(adm, lease_held=1)
        on = adm.evidence.workers[0]
        if failure == 'stale-memory':
            on = replace(on, hardware=replace(on.hardware, worker_ram_current=measure(0, stamp=(NOW - timedelta(seconds=11)).isoformat())))
        else:
            on = replace(on, fingerprint=replace(on.fingerprint, runtime_version='synthetic-new-runtime'))
        return replace(adm, evidence=evidence(on))
    lifecycle.readmit_transform = change
    with pytest.raises(ModelPolicyError):
        coordinator(lifecycle).load(request())
    assert lifecycle.events == ['sample', 'readmit']


def test_fragmentation_oom_retries_once_only_after_confirmed_cleanup_and_readmission():
    lifecycle = Lifecycle(outcomes=(ReasonCode.MODEL_OOM, ReasonCode.OK))
    loaded = coordinator(lifecycle).load(request())
    assert loaded.attempts == 2
    assert [x.artifact.model_id for x in lifecycle.loads] == ['large-v3', 'medium']
    assert loaded.choice.reason == 'oom_lower_retry'
    assert lifecycle.events == ['sample', 'readmit', 'load', 'cleanup', 'sample', 'readmit', 'load']


def test_second_oom_cleans_up_then_fails_without_another_sample_or_fallback():
    lifecycle = Lifecycle(outcomes=(MemoryError('private adapter detail'), ReasonCode.MODEL_OOM))
    policy = coordinator(lifecycle)
    reason(ReasonCode.MODEL_OOM, lambda: policy.load(request()))
    assert len(lifecycle.loads) == 2
    assert lifecycle.events == ['sample', 'readmit', 'load', 'cleanup', 'sample', 'readmit', 'load', 'cleanup']
    reason(ReasonCode.CONFLICT, lambda: policy.load(request()))


@pytest.mark.parametrize('field', ['fragments_discarded', 'model_unloaded', 'writers_reaped'])
@pytest.mark.parametrize('value', [False, 1, None])
def test_cleanup_requires_every_exact_confirmation_before_retry(field, value):
    lifecycle = Lifecycle(outcomes=(ReasonCode.MODEL_OOM,))
    lifecycle.cleanup_transform = lambda result: replace(result, **{field: value})
    policy = coordinator(lifecycle)
    reason(ReasonCode.WRITER_UNRESOLVED, lambda: policy.load(request()))
    assert lifecycle.events == ['sample', 'readmit', 'load', 'cleanup']
    reason(ReasonCode.CONFLICT, lambda: policy.load(request()))


def test_cleanup_must_confirm_the_same_owned_worker():
    lifecycle = Lifecycle(outcomes=(ReasonCode.MODEL_OOM,))
    lifecycle.cleanup_transform = lambda result: replace(result, identity=replace(result.identity, job='alien'))
    reason(ReasonCode.WRITER_UNRESOLVED, lambda: coordinator(lifecycle).load(request()))
    assert len(lifecycle.loads) == 1


def test_explicit_model_oom_never_substitutes():
    lifecycle = Lifecycle(outcomes=(ReasonCode.MODEL_OOM,))
    reason(ReasonCode.MODEL_OOM, lambda: coordinator(lifecycle).load(request(model='small')))
    assert lifecycle.events == ['sample', 'readmit', 'load', 'cleanup']


def test_explicit_cuda_retry_keeps_cuda_and_never_silently_uses_cpu():
    sample = evidence(worker(Backend.CUDA), worker())
    lifecycle = Lifecycle(sample, outcomes=(ReasonCode.MODEL_OOM, ReasonCode.OK))
    coordinator(lifecycle).load(request(device='cuda'))
    assert [x.artifact.model_id for x in lifecycle.loads] == ['large-v3', 'medium']
    assert all(x.backend is Backend.CUDA for x in lifecycle.loads)


def test_retry_rechecks_storage_after_cleanup_before_second_load():
    lifecycle = Lifecycle(outcomes=(ReasonCode.MODEL_OOM,))
    def readmit(adm):
        if len(lifecycle.admissions) == 2:
            field = Measurement(None, 'synthetic', FieldStatus.ERROR, STAMP, ReasonCode.STORAGE_INSUFFICIENT)
            return replace(adm, storage=replace(adm.storage, admission=field))
        return adm
    lifecycle.readmit_transform = readmit
    reason(ReasonCode.STORAGE_INSUFFICIENT, lambda: coordinator(lifecycle).load(request()))
    assert lifecycle.events[-2:] == ['sample', 'readmit']
    assert len(lifecycle.loads) == 1


@pytest.mark.parametrize('field', ['enforced_memory_cap_bytes', 'enforced_decode_timeout_seconds', 'peak_ram_bytes', 'peak_vram_bytes'])
def test_unenforced_bounds_or_exceeded_envelope_cleans_up_and_invalidates(field):
    lifecycle = Lifecycle()
    lifecycle.result_transform = lambda result: replace(result, **{field: 99 * GIB})
    reason(ReasonCode.ENVELOPE_UNVALIDATED, lambda: coordinator(lifecycle).load(request(duration_seconds=1)))
    assert lifecycle.events[-1] == 'cleanup'
    assert len(lifecycle.loads) == 1


def test_load_failure_never_retries_and_does_not_expose_adapter_exception():
    lifecycle = Lifecycle(outcomes=(RuntimeError('secret internal path'),))
    with pytest.raises(ModelPolicyError) as error:
        coordinator(lifecycle).load(request())
    assert error.value.reason_code is ReasonCode.MODEL_LOAD_FAILED
    assert 'secret' not in str(error.value)
    assert lifecycle.events == ['sample', 'readmit', 'load', 'cleanup']


def test_late_load_return_does_not_count_as_success_or_reap_proof():
    lifecycle = Lifecycle()
    tick = [100]
    def result(value):
        tick[0] += 31
        return value
    lifecycle.result_transform = result
    policy = coordinator(lifecycle, monotonic=lambda: tick[0])
    reason(ReasonCode.REQUEST_TIMEOUT, lambda: policy.load(request()))
    assert lifecycle.events[-1] == 'cleanup'


def test_late_cleanup_does_not_authorize_a_new_candidate():
    lifecycle = Lifecycle(outcomes=(ReasonCode.MODEL_OOM,))
    tick = [100]
    def cleanup(value):
        tick[0] += 31
        return value
    lifecycle.cleanup_transform = cleanup
    reason(ReasonCode.WRITER_UNRESOLVED,
           lambda: coordinator(lifecycle, monotonic=lambda: tick[0]).load(request()))
    assert lifecycle.events == ['sample', 'readmit', 'load', 'cleanup']


def test_concurrent_calls_cannot_create_overlapping_owned_loads():
    lifecycle = Lifecycle()
    started, finish = threading.Event(), threading.Event()
    def hold():
        started.set()
        assert finish.wait(5)
    lifecycle.sample_hook = hold
    policy = coordinator(lifecycle)
    outcomes = []
    def run():
        try:
            outcomes.append(policy.load(request()))
        except Exception as exc:  # noqa: BLE001 - the outcome itself is under test
            outcomes.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert started.wait(5)
        reason(ReasonCode.CONFLICT, lambda: policy.load(request()))
        assert lifecycle.events == ['sample']
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(outcomes) == 1 and not isinstance(outcomes[0], Exception)
    assert len(lifecycle.loads) == 1


def test_video_only_skips_model_policy_and_every_callback():
    lifecycle = Lifecycle()
    result = coordinator(lifecycle).load(request(output_mode='video', duration_seconds=None))
    assert result is None and lifecycle.events == []
    assert select_model(request(output_mode='video', duration_seconds=None), None, None) is None


def test_table_only_coordinator_loads_without_envelopes_and_skips_peak_claims():
    lifecycle = Lifecycle()
    lifecycle.result_transform = lambda result: replace(result, peak_ram_bytes=99 * GIB)
    loaded = coordinator(lifecycle, envelopes=()).load(request())
    assert loaded.choice.measured_envelope is None and loaded.choice.basis == 'published'
    assert loaded.choice.artifact.model_id == 'medium'


def test_explicit_choice_is_readmitted_without_fit_but_auto_is_refit():
    lifecycle = Lifecycle()
    on = lifecycle.evidence.workers[0]
    busy = replace(on, hardware=replace(on.hardware, worker_ram_current=measure(15 * GIB),
                                       worker_ram_available=measure(GIB)))
    lifecycle.readmit_transform = lambda adm: replace(adm, evidence=evidence(busy))
    loaded = coordinator(lifecycle, envelopes=()).load(request(model='large-v3'))
    assert loaded.choice.basis == 'explicit' and lifecycle.events == ['sample', 'readmit', 'load']
