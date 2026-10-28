"""Runtime trust boundaries and a separately admitted native fixture smoke."""
import asyncio
import base64
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest
import psutil

from podmuncher.admission import AdmissionPlan, WorkloadGrowth
from podmuncher.config import Limits
from podmuncher.domains import DomainValidationError, FieldStatus, ReasonCode
from podmuncher.docker_storage import (BoundedRunner, DockerStorageSession,
                                      make_docker_storage_provider)
from podmuncher.guards import CancellationCoordinator, OperationBinding, StorageSample, WriteGuard
from podmuncher.hardware import DockerCgroupMemory, HardwareSampler
from podmuncher.protocol import WorkerEventRecord, WorkerRequest
import podmuncher.runtime as _runtime_module
from podmuncher.runtime import (DockerRuntime, DockerTarget, NativeDockerCLI, NativeProcess,
    OwnedContainer, RuntimeFailure, RuntimeStorage, WorkerMounts, WorkerPolicy)
from podmuncher.snapshots import Measurement

IMAGE = 'python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f'
# Unit-test target for fake factories. The live smoke derives its target from
# the current Docker context instead (see _live_target).
TARGET = DockerTarget('desktop-linux', 'npipe:////./pipe/dockerDesktopLinuxEngine')
CID = 'a' * 64
GPU_UUID = 'GPU-3a23c669-1f69-c64e-cf85-44e9b07e7a2a'
LOCAL_IMAGE = 'sha256:' + 'b' * 64


def request(**changes):
    data = dict(schema_version=1, job_id=str(uuid4()), attempt_id=str(uuid4()), stage_id=str(uuid4()),
        capability='asr_cpu', operation='probe', assigned_slots={'probe_metadata': 'tiny.bin'}, source=None,
        outputs=dict(transcript=True, video=False, video_quality='source_best'), input_basename=None,
        model_snapshot_id=None, limits=asdict(Limits()),
        selection=dict(device='cpu', profile='balanced', model='auto', language='auto'))
    data.update(changes)
    return WorkerRequest.from_dict(data)


def event(req, sequence, kind, payload=None):
    return WorkerEventRecord(1, req.job_id, req.attempt_id, req.stage_id,
                             sequence, kind, payload or {}).to_json() + b'\n'


def admission(reason=ReasonCode.OK):
    return AdmissionPlan(Measurement(True if reason is ReasonCode.OK else None,
        'fixture', FieldStatus.OK if reason is ReasonCode.OK else FieldStatus.ERROR,
        datetime.now(timezone.utc).isoformat(), None if reason is ReasonCode.OK else reason), (), ())


def make_runtime_storage(tmp_path, **kwargs):
    evidence = SimpleNamespace(can_pull=True, can_create_cached=True)
    session = DockerStorageSession(SimpleNamespace(), evidence, object())
    paths = {role: tmp_path / role for role in ('scratch', 'cache', 'output', 'state')}
    growth = WorkloadGrowth(0, 0, 0, 0, (8,), 0, 0, 0, 0, (), 0, 0)
    return RuntimeStorage(session, SimpleNamespace(), paths=paths, growth=growth,
                          limits=Limits(), **kwargs)


def test_runtime_storage_sample_budget_is_validated(tmp_path):
    assert make_runtime_storage(tmp_path).sample_budget_seconds == 0.95
    assert make_runtime_storage(tmp_path, sample_budget_seconds=1.45).sample_budget_seconds == 1.45
    for invalid in (0, -1, 3.01, float('nan'), float('inf'), True, '1', None):
        with pytest.raises(RuntimeFailure) as error:
            make_runtime_storage(tmp_path, sample_budget_seconds=invalid)
        assert error.value.reason_code is ReasonCode.INVALID_INPUT


def test_runtime_storage_sample_uses_configured_budget(tmp_path, monkeypatch):
    recorded = []
    volume = SimpleNamespace(identity=SimpleNamespace(value='volume'))
    evidence = SimpleNamespace(can_create_cached=True, oldest_observed_monotonic=time.monotonic(),
        constraints=(SimpleNamespace(roles=('docker_data',), volume=volume),))
    session = SimpleNamespace(refresh=lambda *, deadline: recorded.append(deadline) or evidence)
    volumes = SimpleNamespace(sample=lambda path, *, sqlite_state: volume)
    monkeypatch.setitem(RuntimeStorage._sample.__globals__, 'make_workload_plan',
                        lambda *args, **kwargs: admission())

    storage = make_runtime_storage(tmp_path, sample_budget_seconds=1.45)
    storage.session, storage.volumes = session, volumes
    started = time.monotonic()
    storage._sample()
    finished = time.monotonic()
    # deadline = monotonic() + budget, read between started and finished. A
    # fine-grained clock (macOS) makes recorded - started strictly exceed budget.
    budget = storage.sample_budget_seconds
    assert budget == 1.45
    assert started + budget <= recorded[0] <= finished + budget

    def slow_refresh(*, deadline):
        time.sleep(0.05)
        return evidence

    storage = make_runtime_storage(tmp_path, sample_budget_seconds=0.02)
    storage.session, storage.volumes = SimpleNamespace(refresh=slow_refresh), volumes
    with pytest.raises(RuntimeFailure) as error:
        storage._sample()
    assert error.value.reason_code is ReasonCode.STORAGE_STALE


class Lease:
    def __init__(self):
        self.held = False
        self.released = 0

    def try_acquire(self, owner, plan):
        assert plan.admitted
        if self.held:
            return False
        self.held = True
        return True

    def release(self, owner, *, writers_reaped):
        assert writers_reaped and self.held
        self.held = False
        self.released += 1


class FakeProcess:
    def __init__(self, factory, out, code=0, stderr=b''):
        self.pid = len(factory.processes) + 100
        self.returncode = code
        self.stdout, self.stderr, self.stdin = io.BytesIO(out), io.BytesIO(stderr), io.BytesIO()
        self.terminated = self.killed = False
        factory.processes.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class FakeProcessFactory:
    """Exact subprocess seam, including the daemon surviving lost CLI output."""
    def __init__(self, req):
        self.req, self.calls, self.processes = req, [], []
        self.labels = {}
        self.exists = self.running = False
        self.payload = event(req, 0, 'started') + event(req, 1, 'completed')
        self.stderr = b''
        self.create_output = (CID + '\n').encode()
        self.failed_start = False
        self.fail_remove = False
        self.foreign = False
        self.endpoint = TARGET.endpoint
        self.device_requests = None
        self.image_id = LOCAL_IMAGE
        self.exit_code = 0

    def identity(self, pid):
        return SimpleNamespace(create_time=lambda: float(pid))

    def __call__(self, argv, **kwargs):
        assert kwargs['shell'] is False
        assert not any(k in kwargs['env'] for k in ('DOCKER_HOST', 'DOCKER_CONTEXT'))
        assert type(argv) is tuple
        self.calls.append(argv)
        command = argv[3:]
        out, code, stderr = b'', 0, b''
        if command[:2] == ('context', 'inspect'):
            assert argv[:3] == ('docker', '--context', TARGET.context)
            out = json.dumps([dict(Name=TARGET.context, Endpoints=dict(docker=dict(Host=self.endpoint)))]).encode()
        else:
            assert argv[:3] == ('docker', '--host', TARGET.endpoint)
            if command[0] == 'create':
                self.exists = True
                self.labels = dict(command[i + 1].split('=', 1) for i, v in enumerate(command) if v == '--label')
                out = self.create_output
            elif command[:2] == ('image', 'inspect'):
                out = json.dumps([{'Id': self.image_id}]).encode()
            elif command[0] == 'ps':
                out = (CID + '\n').encode() if self.exists else b''
            elif command[:2] == ('container', 'inspect'):
                labels = dict(self.labels)
                if self.foreign:
                    labels['podmuncher.instance'] = 'other'
                out = json.dumps([dict(Id=CID, Config=dict(Labels=labels),
                                      HostConfig=dict(DeviceRequests=self.device_requests),
                                      State=dict(Running=self.running, Restarting=False))]).encode()
                code = 0 if self.exists else 1
            elif command[0] == 'start':
                code = 1 if self.failed_start else 0
                out, stderr = self.payload, self.stderr
            elif command[0] == 'wait':
                out = f'{self.exit_code}\n'.encode()
            elif command[0] in ('stop', 'kill'):
                self.running = False
            elif command[0] == 'rm':
                code = int(self.fail_remove)
                self.exists = self.fail_remove
            elif command[0] == 'pull':
                out = b'pulled\n'
            else:
                raise AssertionError(command)
        return FakeProcess(self, out, code, stderr)


def make_runtime(tmp_path, req=None, *, sampler=None, policy=None, timeout=10):
    req = request() if req is None else req
    factory = FakeProcessFactory(req)
    lease = Lease()
    async def sample():
        return StorageSample(admission(), time.monotonic())
    guard = WriteGuard(CancellationCoordinator('owned-operation', lease, stop_grace_seconds=0.2,
        reap_timeout_seconds=0.2, callback_timeout_seconds=0.2), sampler or sample,
        byte_limit=65536, timeout_seconds=timeout)
    scratch = tmp_path / 'scratch'
    scratch.mkdir(exist_ok=True)
    policy = policy or WorkerPolicy('asr_cpu', IMAGE, ('python', '-I', '/probe.py'), 0, 0, lambda: True)
    cli = NativeDockerCLI(TARGET, factory=factory, identity=factory.identity, environ={})
    runtime = DockerRuntime(cli, guard, instance='test-instance', policy=policy, mounts=WorkerMounts(scratch))
    return runtime, factory, lease, req


def test_hardened_exact_array_and_success(tmp_path):
    runtime, factory, lease, req = make_runtime(tmp_path)
    result = asyncio.run(runtime.run(req))
    create = next(c[3:] for c in factory.calls if c[3] == 'create')
    assert create[:2] == ('create', '--name')
    for flag, value in (('--pull', 'never'), ('--user', _runtime_module._worker_user()), ('--cap-drop', 'ALL'),
        ('--security-opt', 'no-new-privileges'), ('--network', 'none'), ('--pids-limit', '64'),
        ('--memory', str(512 * 1024**2)), ('--memory-swap', str(512 * 1024**2)), ('--cpus', '1.0')):
        assert create[create.index(flag) + 1] == value
    assert '--read-only' in create and '--privileged' not in create
    assert create.count('--mount') == 1
    assert 'max-size=10m' in create and 'max-file=2' in create
    assert create[-4:] == ('python', IMAGE, '-I', '/probe.py')
    assert factory.labels['podmuncher.job'] == req.job_id
    assert factory.labels['podmuncher.attempt'] == req.attempt_id
    assert result.completed and result.exit_code == 0 and result.artifact_claims == ()
    assert not lease.held and lease.released == 1 and not factory.exists
    assert all(p.stdout.closed and p.stderr.closed for p in factory.processes)


def gpu_requests(uuid=GPU_UUID):
    return [dict(Driver='', Count=0, DeviceIDs=[uuid], Capabilities=[['gpu']], Options={})]


@pytest.mark.parametrize('capability,device,allowed', [
    ('asr_cpu', 'cpu', True), ('asr_cpu', 'auto', True), ('asr_cpu', 'cuda', False),
    ('asr_cuda', 'cuda', True), ('asr_cuda', 'auto', True), ('asr_cuda', 'cpu', False),
    ('media', 'cuda', True), ('media', 'cpu', True), ('media', 'auto', True)])
def test_explicit_device_and_trusted_dispatch_matrix(tmp_path, capability, device, allowed):
    req = request(capability=capability,
        input_basename='media.bin' if capability == 'media' else None,
        selection=dict(device=device, profile='balanced', model='auto', language='auto'),
        outputs=dict(transcript=capability != 'media', video=capability == 'media', video_quality='source_best'))
    policy = WorkerPolicy(capability, IMAGE, ('python', '/probe.py'), 0, 0, lambda: True,
                          selected_gpu_uuid=GPU_UUID if capability == 'asr_cuda' else None)
    runtime, factory, lease, _ = make_runtime(tmp_path, req, policy=policy)
    if capability == 'media':
        media = tmp_path / 'media.bin'
        media.write_bytes(b'fixture')
        runtime.mounts = replace(runtime.mounts, input_file=media)
    if capability == 'asr_cuda':
        factory.device_requests = gpu_requests()
    if not allowed:
        with pytest.raises(RuntimeFailure) as error:
            asyncio.run(runtime.run(req))
        assert error.value.reason_code is ReasonCode.WORKER_PROTOCOL_MISMATCH
        assert not factory.calls and not lease.held
        return
    result = asyncio.run(runtime.run(req))
    create = next(c[3:] for c in factory.calls if c[3] == 'create')
    if capability == 'asr_cuda':
        assert create.count('--gpus') == 1
        assert create[create.index('--gpus') + 1] == 'device=' + GPU_UUID
        assert result.reference.gpu_visible and result.reference.gpu_uuid == GPU_UUID
    else:
        assert '--gpus' not in create
        assert not result.reference.gpu_visible and result.reference.gpu_uuid is None
        if capability == 'media':
            assert create.count('--mount') == 2
            assert any(value.endswith('dst=/input/media.bin,readonly') for value in create)
    assert lease.released == 1


@pytest.mark.parametrize('uuid', [None, 'all', '0', 'GPU-unknown', GPU_UUID + ',device=all'])
def test_cuda_requires_independent_valid_trusted_uuid(tmp_path, uuid):
    req = request(capability='asr_cuda',
        selection=dict(device='auto', profile='balanced', model='auto', language='auto'))
    with pytest.raises(RuntimeFailure) as error:
        policy = WorkerPolicy('asr_cuda', IMAGE, ('python',), 1, 1, selected_gpu_uuid=uuid)
        runtime, factory, lease, _ = make_runtime(tmp_path, req, policy=policy)
        asyncio.run(runtime.run(req))
    assert error.value.reason_code is ReasonCode.GPU_UNAVAILABLE
    if uuid is None:
        assert not factory.calls and not lease.held


@pytest.mark.parametrize('requests', [None, [], gpu_requests('GPU-' + 'f' * 8 + '-ffff-ffff-ffff-' + 'f' * 12),
    [dict(Driver='', Count=-1, DeviceIDs=[], Capabilities=[['gpu']], Options={})],
    gpu_requests() * 2, [dict(Driver='', Count=0, DeviceIDs=[GPU_UUID], Capabilities=[['utility']], Options={})]])
def test_gpu_inspect_join_rejects_unintended_visibility(tmp_path, requests):
    req = request(capability='asr_cuda',
        selection=dict(device='cuda', profile='balanced', model='auto', language='auto'))
    policy = WorkerPolicy('asr_cuda', IMAGE, ('python',), 1, 1, selected_gpu_uuid=GPU_UUID)
    runtime, factory, lease, _ = make_runtime(tmp_path, req, policy=policy)
    factory.device_requests = requests
    with pytest.raises(RuntimeFailure) as error:
        asyncio.run(runtime.run(req))
    assert error.value.reason_code is ReasonCode.GPU_UNAVAILABLE
    assert runtime.reference is None and not any(c[3] == 'start' for c in factory.calls)
    assert not factory.exists and lease.released == 1


@pytest.mark.parametrize('proof_ok,image_matches', [(True, True), (False, True), (True, False)])
def test_development_image_requires_fresh_exact_inspect_and_cached_proof(tmp_path, proof_ok, image_matches):
    proofs = []
    policy = WorkerPolicy('asr_cpu', LOCAL_IMAGE, ('python',), 0, 0,
        lambda: proofs.append('fresh') or proof_ok, development_image=True)
    runtime, factory, lease, req = make_runtime(tmp_path, policy=policy)
    factory.image_id = LOCAL_IMAGE if image_matches else 'sha256:' + 'c' * 64
    assert not proofs and not factory.calls
    if proof_ok and image_matches:
        result = asyncio.run(runtime.run(req))
        assert result.completed
        commands = [c[3:] for c in factory.calls]
        inspected = commands.index(('image', 'inspect', LOCAL_IMAGE))
        created = next(i for i, c in enumerate(commands) if c[0] == 'create')
        assert inspected < created
        assert commands[created][commands[created].index('--pull') + 1] == 'never'
        assert LOCAL_IMAGE in commands[created]
    else:
        with pytest.raises(RuntimeFailure) as error:
            asyncio.run(runtime.run(req))
        assert error.value.reason_code is (ReasonCode.STORAGE_UNKNOWN if not proof_ok else ReasonCode.DOCKER_UNAVAILABLE)
        assert not any(c[3] == 'create' for c in factory.calls)
    assert proofs == ['fresh'] and not any(c[3] == 'pull' for c in factory.calls)
    assert lease.released == (1 if proof_ok else 0) and not lease.held


@pytest.mark.parametrize('image,development,pull', [(LOCAL_IMAGE, False, False),
    (LOCAL_IMAGE, True, True), (IMAGE, True, False), ('python:latest', True, False)])
def test_development_ids_cannot_cross_policy_branches(image, development, pull):
    with pytest.raises(RuntimeFailure):
        WorkerPolicy('asr_cpu', image, ('python',), 1, 1,
                     development_image=development, allow_pull=pull)


@pytest.mark.parametrize('complete,exit_code,start_code,expected', [
    (True, 1, False, ReasonCode.WORKER_EXIT_UNEXPECTED),
    (True, 0, True, ReasonCode.WORKER_EXIT_UNEXPECTED),
    (False, 1, False, ReasonCode.WORKER_PROTOCOL_MISMATCH)])
def test_worker_exit_and_protocol_reasons_are_distinct(tmp_path, complete, exit_code, start_code, expected):
    runtime, factory, lease, req = make_runtime(tmp_path)
    factory.exit_code, factory.failed_start = exit_code, start_code
    if not complete:
        factory.payload = event(req, 0, 'started')
    with pytest.raises(RuntimeFailure) as error:
        asyncio.run(runtime.run(req))
    assert error.value.reason_code is expected
    assert lease.released == 1 and not factory.exists


@pytest.mark.parametrize('kind', ['missing_completed', 'duplicate_started', 'after_completed',
    'wrong_identity', 'sequence', 'oversize', 'stderr', 'error', 'failed_start', 'artifact_claim'])
def test_untrusted_worker_boundaries(tmp_path, kind):
    runtime, factory, lease, req = make_runtime(tmp_path)
    if kind == 'missing_completed':
        factory.payload = event(req, 0, 'started')
    elif kind == 'duplicate_started':
        factory.payload = event(req, 0, 'started') + event(req, 1, 'started')
    elif kind == 'after_completed':
        factory.payload += event(req, 2, 'progress', {'percent': 100})
    elif kind == 'wrong_identity':
        factory.payload = event(request(), 0, 'started')
    elif kind == 'sequence':
        factory.payload = event(req, 1, 'started') + event(req, 0, 'completed')
    elif kind == 'oversize':
        factory.payload = b'x' * 65537
    elif kind == 'stderr':
        factory.stderr = b'C:\\private\\password=secret\n' * 4000
    elif kind == 'error':
        factory.payload = event(req, 0, 'started') + event(req, 1, 'error',
            {'reason_code': 'docker_unavailable', 'summary': 'Worker failed'})
    elif kind == 'failed_start':
        factory.failed_start = True
    elif kind == 'artifact_claim':
        factory.payload = event(req, 0, 'started') + event(req, 1, 'artifact',
            {'slot': 'probe_metadata', 'bytes': 3, 'sha256': 'a' * 64}) + event(req, 2, 'completed')
        result = asyncio.run(runtime.run(req))
        assert len(result.artifact_claims) == 1
        assert not hasattr(result, 'published')
        return
    with pytest.raises(RuntimeFailure) as error:
        asyncio.run(runtime.run(req))
    assert 'secret' not in str(error.value) and 'private' not in str(error.value)
    assert not factory.exists and not lease.held
    if kind == 'error':
        assert error.value.reason_code is ReasonCode.DOCKER_UNAVAILABLE


def test_error_event_with_ok_reason_is_protocol_mismatch(tmp_path):
    runtime, factory, lease, req = make_runtime(tmp_path)
    factory.payload = event(req, 0, 'started') + event(req, 1, 'error',
        {'reason_code': 'ok', 'summary': 'worker claims ok'})
    with pytest.raises(RuntimeFailure) as error:
        asyncio.run(runtime.run(req))
    assert error.value.reason_code is ReasonCode.WORKER_PROTOCOL_MISMATCH
    assert not factory.exists and not lease.held and lease.released == 1


def test_lost_create_stdout_recovers_exact_labels(tmp_path):
    runtime, factory, lease, req = make_runtime(tmp_path)
    factory.create_output = b''
    with pytest.raises(RuntimeFailure):
        asyncio.run(runtime.run(req))
    query = next(c for c in factory.calls if c[3] == 'ps')
    assert sum('podmuncher.' in a for a in query) == 4
    assert not factory.exists and not lease.held
    assert not any(c[3] == 'start' for c in factory.calls)


def test_failed_reap_retains_lease(tmp_path):
    runtime, factory, lease, req = make_runtime(tmp_path)
    factory.fail_remove = True
    with pytest.raises(RuntimeFailure) as error:
        asyncio.run(runtime.run(req))
    assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert lease.held and not lease.released and factory.exists


def test_foreign_labels_are_never_signalled(tmp_path):
    runtime, factory, lease, req = make_runtime(tmp_path)
    factory.foreign = True
    with pytest.raises(RuntimeFailure) as error:
        asyncio.run(runtime.run(req))
    assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert lease.held
    assert not any(c[3] in ('start', 'stop', 'kill', 'rm') for c in factory.calls)


def test_cached_creation_requires_fresh_admission_and_materialization(tmp_path):
    calls = 0
    async def sample():
        nonlocal calls
        calls += 1
        return StorageSample(admission(ReasonCode.OK if calls == 1 else ReasonCode.STORAGE_INSUFFICIENT), time.monotonic())
    runtime, factory, lease, req = make_runtime(tmp_path, sampler=sample)
    with pytest.raises(RuntimeFailure):
        asyncio.run(runtime.run(req))
    assert not factory.calls and not lease.held and calls == 2
    runtime, factory, lease, req = make_runtime(tmp_path,
        policy=WorkerPolicy('asr_cpu', IMAGE, ('python', '/probe.py'), 0, 0, lambda: False))
    with pytest.raises(RuntimeFailure) as error:
        asyncio.run(runtime.run(req))
    assert error.value.reason_code is ReasonCode.STORAGE_UNKNOWN
    assert not factory.calls and not lease.held


def test_cached_proof_completes_before_admission_and_watchdog(tmp_path):
    samples, proofs, docker_calls, holder = [], [], [], {}
    async def sample():
        samples.append(time.monotonic())
        return StorageSample(admission(), time.monotonic())
    def proof():
        proofs.append((time.monotonic(), len(samples), len(holder['factory'].calls)))
        return True
    policy = WorkerPolicy('asr_cpu', IMAGE, ('python', '/probe.py'), 0, 0, proof)
    runtime, factory, lease, req = make_runtime(tmp_path, sampler=sample, policy=policy)
    holder['factory'] = factory
    def spawn(argv, **kwargs):
        docker_calls.append(time.monotonic())
        return factory(argv, **kwargs)
    runtime.cli.factory = spawn
    result = asyncio.run(runtime.run(req))
    assert result.completed
    assert len(proofs) == 1
    proved_at, samples_seen, docker_seen = proofs[0]
    assert samples_seen == 0 and docker_seen == 0
    assert samples and samples[0] >= proved_at
    assert docker_calls and docker_calls[0] >= proved_at
    assert lease.released == 1 and not lease.held


def test_cancelled_during_cached_proof_reports_cancelled(tmp_path):
    holder = {}
    def proof():
        holder['runtime'].obsolete()
        return True
    policy = WorkerPolicy('asr_cpu', IMAGE, ('python', '/probe.py'), 0, 0, proof)
    runtime, factory, lease, req = make_runtime(tmp_path, policy=policy)
    holder['runtime'] = runtime
    with pytest.raises(RuntimeFailure) as error:
        asyncio.run(runtime.run(req))
    assert error.value.reason_code is ReasonCode.CANCELLED
    assert not factory.calls and not lease.held and lease.released == 0


def test_unfinished_cached_proof_thread_is_unresolved(tmp_path):
    release = threading.Event()
    def proof():
        # Blocks longer than reap_timeout_seconds (0.2 s) until the test releases it.
        assert release.wait(timeout=5), 'Fixture proof thread was not released'
        return True
    policy = WorkerPolicy('asr_cpu', IMAGE, ('python', '/probe.py'), 0, 0, proof)
    runtime, factory, lease, req = make_runtime(tmp_path, policy=policy)
    outcomes = []
    async def drive():
        try:
            # Python 3.12 wait_for runs the coroutine in this task, so the
            # cancellation surfaces as RuntimeFailure(WRITER_UNRESOLVED).
            await asyncio.wait_for(runtime.run(req), timeout=0.05)
        except RuntimeFailure as error:
            outcomes.append(error.reason_code)
        except (asyncio.TimeoutError, asyncio.CancelledError) as error:
            outcomes.append(type(error))
        assert not factory.calls and not lease.held and lease.released == 0
        task = runtime._proof_task
        assert task is not None and not task.done() and not task.cancelled()
        release.set()
        await asyncio.wait_for(asyncio.shield(task), timeout=2)
        assert task.done() and not task.cancelled() and task.result() is True
    try:
        asyncio.run(drive())
    finally:
        release.set()
    assert len(outcomes) == 1
    if isinstance(outcomes[0], ReasonCode):
        assert outcomes[0] is ReasonCode.WRITER_UNRESOLVED
    else:
        assert outcomes[0] in (asyncio.TimeoutError, asyncio.CancelledError)


def test_obsolete_discards_late_completed_event(tmp_path):
    runtime, factory, lease, req = make_runtime(tmp_path)
    seen = []
    def consume(item):
        seen.append(item.event)
        runtime.obsolete()
    with pytest.raises(RuntimeFailure):
        asyncio.run(runtime.run(req, on_event=consume))
    assert seen == ['started']
    assert not lease.held and not factory.exists


def test_context_drift_and_environment_routing_rejected(tmp_path):
    runtime, factory, lease, req = make_runtime(tmp_path)
    factory.endpoint = 'tcp://remote:2375'
    with pytest.raises(RuntimeFailure):
        asyncio.run(runtime.run(req))
    assert not any(c[3] == 'create' for c in factory.calls)
    for key in ('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_API_VERSION'):
        with pytest.raises(RuntimeFailure):
            NativeDockerCLI(TARGET, environ={key: 'untrusted'})


def test_mount_separator_network_and_command_injection(tmp_path):
    runtime, _, _, req = make_runtime(tmp_path)
    bad = tmp_path / 'unsafe,readonly=false'
    bad.mkdir()
    with pytest.raises(RuntimeFailure):
        WorkerMounts(bad).arguments(req)
    remote = request(capability='media', operation='download', source={'kind': 'direct', 'input': 'https://example.com/movie'})
    probe = tmp_path / 'probe.py'
    probe.write_text('pass')
    with pytest.raises(RuntimeFailure):
        WorkerMounts(runtime.mounts.scratch, probe_file=probe).arguments(remote)
    args, _ = runtime.create_arguments(replace(req, operation='echo.secret'), 'unique')
    assert 'echo.secret' not in args
    args, _ = DockerRuntime(runtime.cli, runtime.guard, instance='instance',
        policy=replace(runtime.policy, capability='media'), mounts=runtime.mounts).create_arguments(remote, 'unique')
    assert args[args.index('--network') + 1] == 'bridge'


def test_native_process_revalidates_original_binding_and_creation_identity():
    factory = SimpleNamespace(processes=[])
    process = FakeProcess(factory, b'')
    process.returncode = None
    created = [1.0]
    operation = NativeProcess(process, 'owner', identity=lambda _: SimpleNamespace(create_time=lambda: created[0]))
    expected = operation.binding
    async def check():
        assert operation.binding is expected
        with pytest.raises(RuntimeFailure):
            await operation.kill(OperationBinding('owner', expected.target_id))
        created[0] = 2.0
        for callback in (operation.terminate, operation.kill, operation.reap):
            with pytest.raises(RuntimeFailure):
                await callback(expected)
    asyncio.run(check())
    assert not process.killed and not process.terminated


class _OwnedPopen(subprocess.Popen):
    """A real Popen type (for the isinstance gate) whose OS calls are recorded."""
    def __init__(self, pid=4242):
        self._child_created = False  # Popen.__del__ must not reap anything.
        self.pid, self.returncode, self.signals = pid, None, []
        self.stdin = self.stdout = self.stderr = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.signals.append('TERM')

    def kill(self):
        self.signals.append('KILL')


def test_posix_without_pidfd_signals_owned_unreaped_popen():
    # macOS: no os.pidfd_open. The exact owned Popen is signalled, not a bare PID.
    process = _OwnedPopen()
    operation = NativeProcess(process, 'owner', identity=lambda _: SimpleNamespace(create_time=lambda: 1.0),
                              posix=True, pidfd=False)
    async def run():
        await operation.terminate(operation.binding)
        await operation.kill(operation.binding)
        assert process.signals == ['TERM', 'KILL']
        process.returncode = -9
        # Once reaped, the PID may be reused: no further signal is sent.
        await operation.terminate(operation.binding)
        await operation.kill(operation.binding)
        assert process.signals == ['TERM', 'KILL']
        assert await operation.reap(operation.binding)
    asyncio.run(run())


def test_posix_without_pidfd_still_revalidates_creation_identity():
    process = _OwnedPopen()
    created = [1.0]
    operation = NativeProcess(process, 'owner', identity=lambda _: SimpleNamespace(create_time=lambda: created[0]),
                              posix=True, pidfd=False)
    created[0] = 2.0
    async def run():
        for callback in (operation.terminate, operation.kill):
            with pytest.raises(RuntimeFailure) as error:
                await callback(operation.binding)
            assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED
    asyncio.run(run())
    assert process.signals == []


def test_linux_pidfd_route_is_unchanged(monkeypatch):
    sent, opened = [], []
    monkeypatch.setattr(os, 'pidfd_open', lambda pid, *_flags: opened.append(pid) or 77, raising=False)
    monkeypatch.setattr(_runtime_module.signal, 'pidfd_send_signal',
                        lambda fd, sig: sent.append((fd, sig)), raising=False)
    process = _OwnedPopen()
    operation = NativeProcess(process, 'owner', identity=lambda _: SimpleNamespace(create_time=lambda: 1.0),
                              posix=True, pidfd=True)
    asyncio.run(operation.terminate(operation.binding))
    assert opened == [4242] and len(sent) == 1 and sent[0][0] == 77 and process.signals == []


def test_linux_pidfd_route_without_pidfd_fails_closed():
    process = _OwnedPopen()
    operation = NativeProcess(process, 'owner', identity=lambda _: SimpleNamespace(create_time=lambda: 1.0),
                              capture=False, posix=True, pidfd=True)
    process.returncode = 0  # an exited child gets no pidfd at capture
    operation.capture_identity()
    process.returncode = None  # a pidfd-less live child must never fall back to a PID signal
    with pytest.raises(RuntimeFailure) as error:
        asyncio.run(operation.kill(operation.binding))
    assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED and process.signals == []


def test_darwin_cli_path_gains_docker_desktop_dir_only_on_darwin():
    darwin = NativeDockerCLI(TARGET, environ={'PATH': '/usr/bin:/bin'}, platform='darwin')
    assert darwin.env['PATH'] == '/usr/bin:/bin:/usr/local/bin'
    assert NativeDockerCLI(TARGET, environ={}, platform='darwin').env['PATH'] == '/usr/local/bin'
    kept = NativeDockerCLI(TARGET, environ={'PATH': '/usr/local/bin:/usr/bin'}, platform='darwin')
    assert kept.env['PATH'] == '/usr/local/bin:/usr/bin'
    for platform in ('win32', 'linux'):
        assert dict(NativeDockerCLI(TARGET, environ={'PATH': 'x'}, platform=platform).env) == {'PATH': 'x'}
        assert dict(NativeDockerCLI(TARGET, environ={}, platform=platform).env) == {}


def test_failed_identity_retains_exact_child_when_kill_and_wait_fail(monkeypatch):
    children, failures = [], []
    saved = {}
    def factory(argv, **kwargs):
        child = subprocess.Popen((sys.executable, '-c', 'import time; time.sleep(30)'), **kwargs)
        children.append(child)
        saved.update(kill=child.kill, wait=child.wait)
        def fail_kill():
            failures.append('kill')
            raise OSError('injected kill failure')
        def fail_wait(timeout=None):
            failures.append('wait')
            raise OSError('injected wait failure')
        monkeypatch.setattr(child, 'kill', fail_kill)
        monkeypatch.setattr(child, 'wait', fail_wait)
        return child
    def identity(pid):
        raise psutil.AccessDenied(pid)
    cli = NativeDockerCLI(TARGET, factory=factory, identity=identity, environ={})
    lease = Lease()
    coordinator = CancellationCoordinator('identity-failure', lease,
        stop_grace_seconds=0.05, reap_timeout_seconds=0.05, callback_timeout_seconds=0.05)
    async def run():
        coordinator.acquire(admission())
        with pytest.raises(RuntimeFailure) as error:
            await cli._raw(('fixture',), owner=coordinator.operation_id, coordinator=coordinator, timeout=1)
        assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED
        owner = cli._owners[coordinator.operation_id]
        assert owner.children[0].process is children[0] and owner.failed
        assert not children[0].stdout.closed and not children[0].stderr.closed
        stopped = await coordinator.stop()
        assert stopped.reason is ReasonCode.WRITER_UNRESOLVED and lease.held and not lease.released
        before = len(children)
        with pytest.raises(RuntimeFailure):
            await cli._raw(('later-cleanup',), owner=coordinator.operation_id, timeout=1)
        assert len(children) == before
    try:
        asyncio.run(run())
        # Only Windows has an exact Popen native handle fallback. On POSIX the
        # missing creation identity must cause zero blind PID signal attempts.
        assert failures == (['kill', 'wait'] if os.name == 'nt' else [])
    finally:
        for child in children:
            saved['kill']()
            saved['wait'](timeout=5)
            child.stdout.close()
            child.stderr.close()


def test_cleanup_child_failure_cannot_escape_earlier_owner_reap_or_retry(tmp_path, monkeypatch):
    import podmuncher.runtime as module
    runtime, factory, lease, req = make_runtime(tmp_path)
    original_reap = module._NativeOwner.reap
    early_reaps, broken = [], []
    async def observe_reap(self, expected):
        result = await original_reap(self, expected)
        if result and not broken:
            early_reaps.append(True)
        return result
    monkeypatch.setattr(module._NativeOwner, 'reap', observe_reap)
    def faulty(argv, **kwargs):
        process = factory(argv, **kwargs)
        if argv[3] == 'rm':
            process.returncode = None
            broken.append(process)
        return process
    def identity(pid):
        if broken and pid == broken[0].pid:
            raise psutil.AccessDenied(pid)
        return factory.identity(pid)
    runtime.cli.factory, runtime.cli.identity = faulty, identity
    async def run():
        with pytest.raises(RuntimeFailure) as error:
            await runtime.run(req)
        assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED
        owner = runtime.cli._owners[runtime.guard.coordinator.operation_id]
        assert early_reaps and len(broken) == 1
        assert owner.failed and any(c.process is broken[0] for c in owner.children)
        assert not broken[0].stdout.closed and not broken[0].killed
        before = len(factory.calls)
        recovery = next(item.operation for item in runtime.guard.coordinator._operations
                        if item.binding.target_id.startswith('create:'))
        with pytest.raises(RuntimeFailure):
            await recovery.reap(recovery.binding)
        assert len(factory.calls) == before
        stopped = await runtime.guard.finish()
        assert stopped.reason is ReasonCode.WRITER_UNRESOLVED and lease.held and not lease.released
    asyncio.run(run())


def _capture_owners(cli):
    seen, original = [], cli.bind_owner
    def bind(owner, coordinator=None):
        lifetime = original(owner, coordinator)
        if lifetime not in seen:
            seen.append(lifetime)
        return lifetime
    cli.bind_owner = bind
    return seen


def test_final_owner_barrier_seals_even_after_successful_early_reap(tmp_path):
    runtime, factory, lease, req = make_runtime(tmp_path)
    seen = _capture_owners(runtime.cli)
    async def run():
        await runtime.run(req)
        owner = seen[0]
        assert runtime.guard.coordinator.operation_id not in runtime.cli._owners
        assert owner.sealed and not owner.failed and await owner.reap(owner.binding)
        before = len(factory.calls)
        with pytest.raises(RuntimeFailure) as error:
            await runtime.cli.run('ps', owner=runtime.guard.coordinator.operation_id)
        assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED
        assert len(factory.calls) == before and lease.released == 1
    asyncio.run(run())


@pytest.mark.parametrize('reason', [ReasonCode.WORKER_EXIT_UNEXPECTED, ReasonCode.CANCELLED])
def test_bounded_cleanup_io_settles_before_native_reap_and_lease_release(tmp_path, monkeypatch, reason):
    import podmuncher.runtime as module
    runtime, factory, lease, req = make_runtime(tmp_path)
    coordinator = runtime.guard.coordinator
    factory.exit_code = 1 if reason is ReasonCode.WORKER_EXIT_UNEXPECTED else 0
    entered, reaping, returned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release = threading.Event()
    delayed, kills = [], []
    original_reap, original_kill = module._NativeOwner.reap, module._NativeOwner.kill

    async def observe_reap(self, expected):
        await entered.wait()
        reaping.set()
        result = await original_reap(self, expected)
        returned.set()
        return result

    async def observe_kill(self, expected):
        kills.append(expected)
        await original_kill(self, expected)

    def spawn(argv, **kwargs):
        process = factory(argv, **kwargs)
        if coordinator.stopped.is_set() and not delayed:
            loop = asyncio.get_running_loop()
            class DelayedPipe(io.BytesIO):
                def read1(self, size):
                    loop.call_soon_threadsafe(entered.set)
                    assert release.wait(timeout=1), 'Fixture cleanup IO was not released'
                    return super().read1(size)
            original = process.stdout
            process.stdout = DelayedPipe(original.getvalue())
            original.close()
            delayed.append(process)
        return process

    monkeypatch.setattr(module._NativeOwner, 'reap', observe_reap)
    monkeypatch.setattr(module._NativeOwner, 'kill', observe_kill)
    runtime.cli.factory = spawn
    seen = _capture_owners(runtime.cli)

    async def run():
        task = asyncio.create_task(runtime.run(req, on_event=(
            lambda item: runtime.obsolete() if item.event == 'started' else None)
            if reason is ReasonCode.CANCELLED else None))
        try:
            await asyncio.wait_for(reaping.wait(), timeout=1)
            assert not returned.is_set() and lease.held and not lease.released
            assert len(delayed) == 1 and not delayed[0].stdout.closed
            release.set()
            with pytest.raises(RuntimeFailure) as error:
                await task
            assert error.value.reason_code is reason
            owner = seen[0]
            assert coordinator.operation_id not in runtime.cli._owners
            assert owner.sealed and not owner.failed and not kills
            assert all(child.closed for child in owner.children)
            assert all(task.done() and not task.cancelled() for child in owner.children for task in child.tasks)
            assert all(task.done() for task in coordinator._controls)
            assert not factory.exists and lease.released == 1 and not lease.held
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_cancelled_cleanup_waiter_retains_child_io_and_lease():
    factory = SimpleNamespace(processes=[])
    entered, release = asyncio.Event(), asyncio.Event()
    async def consume(chunk):
        entered.set()
        await release.wait()
    def spawn(argv, **kwargs):
        return FakeProcess(factory, b'owned IO')
    cli = NativeDockerCLI(TARGET, factory=spawn,
        identity=lambda pid: SimpleNamespace(create_time=lambda: float(pid)), environ={})
    lease = Lease()
    coordinator = CancellationCoordinator('cancelled-cleanup', lease,
        stop_grace_seconds=0.05, reap_timeout_seconds=0.05, callback_timeout_seconds=0.05)
    async def run():
        coordinator.acquire(admission())
        owner = cli.bind_owner(coordinator.operation_id, coordinator)
        # A cleanup command has no active coordinator argument, but keeps the
        # same lifetime and can never disappear on cancellation of its waiter.
        command = asyncio.create_task(cli._raw(('cleanup',), owner=coordinator.operation_id,
                                               timeout=5, on_stdout=consume))
        await entered.wait()
        command.cancel()
        while not owner.cleanups:
            await asyncio.sleep(0)
        command.cancel()
        with pytest.raises(RuntimeFailure) as error:
            await command
        assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED
        assert owner.failed and any(not t.done() for c in owner.children for t in c.tasks)
        stopped = await coordinator.stop()
        assert stopped.reason is ReasonCode.WRITER_UNRESOLVED and lease.held
        release.set()
        await asyncio.gather(*owner.cleanups, return_exceptions=True)
        assert (await coordinator.stop()).reason is ReasonCode.WRITER_UNRESOLVED
        assert lease.held and not lease.released
    asyncio.run(run())


@pytest.mark.parametrize('behavior', ['timeout', 'oversize', 'stderr'])
def test_actual_native_child_is_bounded_and_reaped(behavior):
    children = []
    script = {'timeout': 'import time; time.sleep(30)',
              'oversize': 'import sys; sys.stdout.buffer.write(b"x"*100000); sys.stdout.flush()',
              'stderr': 'import sys; sys.stderr.buffer.write(b"private=secret"*10000); sys.stderr.flush()'}[behavior]
    def factory(argv, **kwargs):
        child = subprocess.Popen((sys.executable, '-c', script), **kwargs)
        children.append(child)
        return child
    cli = NativeDockerCLI(TARGET, factory=factory, environ={}, max_output=8192, max_stderr=8192)
    async def run():
        with pytest.raises(RuntimeFailure) as error:
            await cli._raw(('native-fixture',), owner='owner', timeout=0.05 if behavior == 'timeout' else 2)
        assert error.value.reason_code is (ReasonCode.REQUEST_TIMEOUT if behavior == 'timeout' else ReasonCode.EVENT_LIMIT_REACHED)
    asyncio.run(run())
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdout.closed and children[0].stderr.closed


_NEGATIVE = json.loads((Path(__file__).parents[1] / 'Docs/contract-domains.json').read_text())['negative_examples']


@pytest.mark.parametrize('bad', _NEGATIVE)
@pytest.mark.parametrize('domain', ['capability', 'job_state', 'reason_code', 'worker_event'])
def test_both_registry_negative_examples(tmp_path, bad, domain):
    with pytest.raises(DomainValidationError):
        if domain == 'capability':
            WorkerPolicy(bad, IMAGE, ('python',), 1, 1)
        elif domain == 'reason_code':
            RuntimeFailure(bad)
        elif domain == 'worker_event':
            event(request(), 0, bad)
        else:
            runtime, _, _, req = make_runtime(tmp_path)
            asyncio.run(runtime.run(req, state=bad))


# Test-only cached fixture proof. These hashes pin the independent public probe
# client/schema already provisioned by the host. There is no download fallback.
_PROBE_HASHES = {
    'grpcurl': '32a5cc701c4d2c84980ed348c6383926a8b397d96a4cd9303769ee868cac2251',
    'proto/services/content/v1/content.proto': '70aadb447ff3157da276afecc355a5a58ef2e322eb7530d5c2868734d626eea8',
    'proto/services/images/v1/images.proto': 'f7943198b308917555040e524251f06d9262b917f856a771159253c51635b056',
    'proto/services/snapshots/v1/snapshots.proto': '6e23ec19b971e1088f3984b7a07ee4238d6ad56b85173750c38b0c911803bc23',
    'proto/types/descriptor.proto': 'f98bec409b5e535e2a8d81dacc549180e1dadb31bd6ac46f7aedc6ab0020e447',
    'proto/types/mount.proto': 'c7a31fa848f002cead05a23adfdd9e29a459cd01ae573b7b61b060b3056754d7',
}
_MANIFEST = 'sha256:44ff437bba879d4941b710a369a8f19266aea34b29002807f0c487fabc9eec9b'
_CONFIG = 'sha256:9e87977b867847e186d066f531ef783b006d582a985c341c269446088d90f2c4'

_ASSOCIATION_SCRIPT = r'''
set -eu
export LC_ALL=C
d=''; c=''; count=0
for path in /proc/[0-9]*; do
    count=$((count+1)); [ "$count" -le 4096 ] || exit 1
    IFS= read -r name < "$path/comm" 2>/dev/null || continue
    case "$name" in
      dockerd) [ -z "$d" ] || exit 1; d=${path##*/};;
      containerd) [ -z "$c" ] || exit 1; c=${path##*/};;
    esac
done
[ -n "$d" ] && [ -n "$c" ]
witness() {
    for p in "$d" "$c"; do
        printf 'process\t%s\t' "$p"
        awk '{printf "%s\t", $22}' "/proc/$p/stat"
        readlink "/proc/$p/exe"
        stat -L -c '%d:%i' "/proc/$p/exe" "/proc/$p/root"
        for namespace in mnt net user; do readlink "/proc/$p/ns/$namespace"; done
        sha256sum "/proc/$p/cmdline" | cut -d ' ' -f 1
    done
    stat -c '%d:%i:%F' "/proc/$d/root$1"
    inode=$(awk -v path="$1" '$8==path && $4=="00010000" && $5=="0001" && $6=="01" {print $7}' "/proc/$d/net/unix")
    case "$inode" in ''|*[!0-9]*) exit 1;; esac
    owned=''; n=0
    for fd in /proc/"$c"/fd/*; do
        n=$((n+1)); [ "$n" -le 4096 ] || exit 1
        link=$(readlink "$fd" 2>/dev/null || true)
        if [ "$link" = "socket:[$inode]" ]; then owned=$inode; fi
    done
    [ "$owned" = "$inode" ]
    printf 'listener\t%s\n' "$owned"
    sha256sum "/proc/$d/root$2" | cut -d ' ' -f 1
}
first=$(witness "$1" "$2")
printf '%s\n@@config\n' "$first"
head -c 65537 "/proc/$d/root$2" | base64
printf '@@argv\n'
head -c 8193 "/proc/$d/cmdline" | base64
printf '@@after\n'
witness "$1" "$2"
'''


def _fresh_containerd_association(call, target, engine):
    assert target.endpoint == TARGET.endpoint
    result = call(('wsl.exe', '--distribution', 'docker-desktop', '--user', 'root', '--exec',
        'timeout', '-s', 'KILL', '5', '/bin/sh', '-c', _ASSOCIATION_SCRIPT,
        'runtime-association', '/run/containerd/containerd.sock', '/run/config/docker/daemon.json'))
    first, rest = result.stdout.decode('ascii').split('@@config\n', 1)
    config_encoded, rest = rest.split('@@argv\n', 1)
    argv_encoded, last = rest.split('@@after\n', 1)
    assert first == last, 'Daemon/containerd association changed during observation'
    config_raw = base64.b64decode(config_encoded)
    argv_raw = base64.b64decode(argv_encoded)
    assert len(config_raw) <= 65536 and len(argv_raw) <= 8192
    config = json.loads(config_raw)
    assert config.get('containerd') == '/run/containerd/containerd.sock'
    assert config.get('containerd-namespace', 'moby') == 'moby'
    argv = argv_raw.decode('utf-8').rstrip('\0').split('\0')
    assert '--config-file' in argv
    assert argv[argv.index('--config-file') + 1] == '/run/config/docker/daemon.json'
    lines = first.splitlines()
    assert len(lines) == 17
    daemon = lines[0].split('\t')
    container = lines[7].split('\t')
    assert daemon[0] == container[0] == 'process'
    assert daemon[1].isdecimal() and daemon[2].isdecimal() and daemon[3] == '/usr/bin/dockerd'
    assert container[1].isdecimal() and container[2].isdecimal() and container[3] == '/usr/bin/containerd'
    assert lines[3:6] == lines[10:13]
    assert lines[14].endswith(':socket') and lines[15].startswith('listener\t')
    assert hashlib.sha256(config_raw).hexdigest() == lines[16]
    return dict(container_pid=container[1], witness=first, engine=(engine['ID'],
        engine['DockerRootDir'], engine['ServerVersion'], engine['Driver']))


def _fixture_tool():
    root = Path(__file__).resolve().parents[1] / '.venv'
    candidates = list(root.glob('fixture-probe-*'))
    matches = []
    for candidate in candidates:
        if not candidate.is_dir() or candidate.is_symlink() or candidate.is_junction():
            continue
        valid = True
        for relative, digest in _PROBE_HASHES.items():
            path = candidate / relative
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 32 * 1024**2:
                valid = False
                break
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                valid = False
                break
        if valid:
            matches.append(candidate)
    assert len(matches) == 1, 'Pinned readonly grpcurl fixture client/schema missing or ambiguous'
    selected = matches[0].resolve()
    assert selected.drive and selected.drive[1:] == ':'
    return '/mnt/host/' + selected.drive[0].lower() + selected.as_posix()[2:]


def _live_env():
    env = _runtime_module._child_path(dict(os.environ), sys.platform)
    assert not any(env.get(k) for k in ('DOCKER_HOST', 'DOCKER_CONTEXT'))
    return env


def _live_target(context):
    """The live endpoint of the context docker_storage resolved, from the CLI."""
    result = BoundedRunner(max_output=262144)(('docker', '--context', context, 'context', 'inspect', context),
                                              deadline=time.monotonic() + 15, env=_live_env())
    inspected = json.loads(result.stdout)
    assert len(inspected) == 1 and inspected[0]['Name'] == context
    return DockerTarget(context, inspected[0]['Endpoints']['docker']['Host'])


def _portable_cached_proof(docker, image, engine, deadline):
    """Non-Windows cached-image proof through the engine API only.

    The containerd content/snapshot proof below needs root WSL access to the
    Docker Desktop VM, which exists only on Windows. Elsewhere the exact index
    image must be present locally for the engine's native platform.
    """
    index_digest = image.split('@', 1)[1]
    inspected, = docker('image', 'inspect', image)
    assert inspected['Id'] == index_digest and inspected['Descriptor']['digest'] == index_digest
    arch = {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(engine['Architecture'], engine['Architecture'])
    assert inspected['Os'] == 'linux' and inspected['Architecture'] == arch
    layers = inspected['RootFS']['Layers']
    assert layers and all(type(layer) is str and layer.startswith('sha256:') for layer in layers)
    assert docker('info', '--format', '{{json .}}')['ID'] == engine['ID']
    assert time.monotonic() < deadline
    return True


def _cached_materialized_proof(target, image):
    """Fresh bounded immutable content plus committed snapshot evidence.

    This proves zero *future* acquisition/unpack for --pull never. It is not
    zero disk usage or protection against an external Docker administrator.
    """
    assert image == IMAGE
    deadline = time.monotonic() + 35
    runner = BoundedRunner(max_output=262144)
    env = _live_env()

    def call(argv):
        result = runner(tuple(argv), deadline=deadline, env=env)
        assert result.returncode == 0, 'Cached fixture readonly proof failed'
        return result

    def docker(*args):
        context = json.loads(call(('docker', '--context', target.context, 'context', 'inspect', target.context)).stdout)
        assert len(context) == 1 and context[0]['Name'] == target.context
        assert context[0]['Endpoints']['docker']['Host'] == target.endpoint
        return json.loads(call(('docker', '--host', target.endpoint, *args)).stdout)

    engine = docker('info', '--format', '{{json .}}')
    assert engine['Driver'] == 'overlayfs'
    assert ['driver-type', 'io.containerd.snapshotter.v1'] in engine['DriverStatus']
    if sys.platform != 'win32':
        return _portable_cached_proof(docker, image, engine, deadline)
    tool = _fixture_tool()
    before = _fresh_containerd_association(call, target, engine)
    pid = before['container_pid']

    def rpc(service, method, data):
        assert (service, method) in {('images', 'Get'), ('content', 'Info'),
                                    ('content', 'Read'), ('snapshots', 'Stat')}
        typename = {'images': 'Images', 'content': 'Content', 'snapshots': 'Snapshots'}[service]
        raw = call(('wsl.exe', '--distribution', 'docker-desktop', '--user', 'root', '--exec',
            tool + '/grpcurl', '-plaintext', '-unix', '-max-time', '5',
            '-H', 'containerd-namespace: moby', '-import-path', tool + '/proto',
            '-proto', f'services/{service}/v1/{service}.proto', '-d', json.dumps(data),
            f'/proc/{pid}/root/run/containerd/containerd.sock',
            f'containerd.services.{service}.v1.{typename}/{method}')).stdout.decode('utf-8')
        decoder = json.JSONDecoder()
        records = []
        while raw.strip():
            item, end = decoder.raw_decode(raw.lstrip())
            records.append(item)
            raw = raw.lstrip()[end:]
        assert records
        return records if method == 'Read' else records[0]

    def content(digest):
        info = rpc('content', 'Info', {'digest': digest})['info']
        size = int(info['size'])
        assert info['digest'] == digest and 0 < size <= 65536
        payload = bytearray()
        for chunk in rpc('content', 'Read', {'digest': digest, 'size': size}):
            assert int(chunk.get('offset', 0)) == len(payload)
            payload.extend(base64.b64decode(chunk['data'], validate=True))
        assert len(payload) == size
        assert 'sha256:' + hashlib.sha256(payload).hexdigest() == digest
        return json.loads(payload), info

    image_record = rpc('images', 'Get', {'name': 'docker.io/library/python:3.12-slim'})['image']
    index_digest = image.split('@', 1)[1]
    assert image_record['target']['digest'] == index_digest
    index, _ = content(index_digest)
    matches = [m for m in index['manifests'] if m.get('platform', {}).get('os') == 'linux'
               and m.get('platform', {}).get('architecture') == 'amd64'
               and not m.get('platform', {}).get('variant')]
    assert len(matches) == 1 and matches[0]['digest'] == _MANIFEST
    manifest, _ = content(_MANIFEST)
    assert manifest['config']['digest'] == _CONFIG
    config, config_info = content(_CONFIG)
    assert config['os'] == 'linux' and config['architecture'] == 'amd64'
    inspected, = docker('image', 'inspect', image)
    diff_ids = config['rootfs']['diff_ids']
    assert inspected['Id'] == index_digest and inspected['Descriptor']['digest'] == index_digest
    assert inspected['RootFS']['Layers'] == diff_ids
    assert len(diff_ids) == len(manifest['layers']) == 4
    parent = ''
    for diff_id, layer in zip(diff_ids, manifest['layers']):
        layer_info = rpc('content', 'Info', {'digest': layer['digest']})['info']
        assert layer_info['digest'] == layer['digest'] and int(layer_info['size']) == layer['size']
        assert layer_info.get('labels', {}).get('containerd.io/uncompressed') == diff_id
        chain = diff_id if not parent else 'sha256:' + hashlib.sha256((parent + ' ' + diff_id).encode()).hexdigest()
        snapshot = rpc('snapshots', 'Stat', {'snapshotter': 'overlayfs', 'key': chain})['info']
        assert snapshot['name'] == chain and snapshot['kind'] == 'COMMITTED'
        assert snapshot.get('parent', '') == parent
        parent = chain
    assert config_info['labels']['containerd.io/gc.ref.snapshot.overlayfs'] == parent
    assert rpc('images', 'Get', {'name': 'docker.io/library/python:3.12-slim'})['image']['target']['digest'] == index_digest
    assert _fresh_containerd_association(call, target, docker('info', '--format', '{{json .}}')) == before
    assert time.monotonic() < deadline
    return True


def fixed_probe_source():
    """Fixed, bounded host code; no package import or worker command override."""
    return '''import hashlib, json, pathlib, sys, time
request = json.loads(sys.stdin.buffer.readline(65538))
assert request['schema_version'] == 1 and request['operation'] == 'probe'
identity = {k: request[k] for k in ('schema_version', 'job_id', 'attempt_id', 'stage_id')}
def event(sequence, kind, payload):
    print(json.dumps(dict(identity, sequence=sequence, event=kind, payload=payload)), flush=True)
event(0, 'started', {})
event(1, 'progress', {'percent': 50})
inspected = pathlib.Path('/scratch') / ('.inspected-' + request['stage_id'])
deadline = time.monotonic() + 12
while not inspected.exists():
    assert time.monotonic() < deadline, 'Host inspection did not finish within the fixture bound'
    time.sleep(0.05)
data = b'fixture\\n'
name = request['assigned_slots']['probe_metadata']
assert '/' not in name and '\\\\' not in name and '..' not in name
with (pathlib.Path('/scratch') / name).open('xb') as handle:
    handle.write(data)
event(2, 'artifact', {'slot': 'probe_metadata', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
event(3, 'completed', {})
'''


def prepare_runtime_fixture(tmp_path):
    """Read-only asset preparation is separate from the worker's wall deadline."""
    image = os.environ.get('PODMUNCHER_TEST_RUNTIME_IMAGE')
    compressed = os.environ.get('PODMUNCHER_TEST_RUNTIME_COMPRESSED_BYTES')
    unpacked = os.environ.get('PODMUNCHER_TEST_RUNTIME_UNPACKED_BYTES')
    assert image == IMAGE, 'Required independently pinned Python 3.12 fixture image missing'
    assert compressed == unpacked == '0', 'This fixture requires the freshly materialized cache branch'
    provider = make_docker_storage_provider(timeout=60)
    session = provider.prepare(manual_backing='')
    assert session.initial_evidence.can_create_cached, session.initial_evidence.reason
    target = _live_target(session.initial_evidence.context.value)
    if sys.platform == 'win32':
        assert target.endpoint == TARGET.endpoint
    assert _cached_materialized_proof(target, image)
    paths = {role: tmp_path / role for role in ('scratch', 'cache', 'output', 'state')}
    for path in paths.values():
        path.mkdir()
    if os.name != 'nt':
        paths['scratch'].chmod(0o777)
    probe = tmp_path / 'fixed-probe.py'
    probe.write_text(fixed_probe_source(), encoding='utf-8')
    limits = Limits()
    growth = WorkloadGrowth(0, 0, 0, 0, (8,), 0, 0, 0, 0, (), 0, 0)
    storage = RuntimeStorage(session, provider.volumes, paths=paths, growth=growth, limits=limits,
                             sample_budget_seconds=1.45)
    policy = WorkerPolicy('asr_cpu', image, ('python', '-I', '/probe.py'), 0, 0,
                          lambda: _cached_materialized_proof(target, image))
    return SimpleNamespace(session=session, storage=storage, policy=policy, target=target,
                           mounts=WorkerMounts(paths['scratch'], probe_file=probe))


@pytest.mark.live
def test_native_admitted_runtime_smoke(tmp_path):
    fixture = prepare_runtime_fixture(tmp_path)
    async def execute():
        # Five real complete samples reproduce the required P2 composed gate.
        # The bound is the product's 2.5 s storage sample budget (SPEC_GAP-16):
        # an idle composed sample measured up to 1.83 s on Docker Desktop.
        for _ in range(5):
            began = time.monotonic()
            sampled = await fixture.storage.sample()
            assert time.monotonic() - began <= 2.5 and sampled.plan.admitted
            assert began <= sampled.sampled_at <= time.monotonic()
        lease = Lease()
        coordinator = CancellationCoordinator('fixture-' + uuid4().hex, lease,
            stop_grace_seconds=12, reap_timeout_seconds=12, callback_timeout_seconds=12)
        guard = WriteGuard(coordinator, fixture.storage.sample, byte_limit=65536, timeout_seconds=90,
                           sample_timeout_seconds=2.9)
        cli = NativeDockerCLI(fixture.target)
        runtime = DockerRuntime(cli, guard, instance='fixture-' + uuid4().hex,
                                policy=fixture.policy, mounts=fixture.mounts)
        req = request()
        inspection = []
        seen = []

        async def inspect_running():
            ref = runtime.reference
            assert ref is not None
            raw = await cli.run('container', 'inspect', ref.container_id, owner=coordinator.operation_id)
            data, = json.loads(raw.stdout)
            assert raw.returncode == 0 and data['Id'] == ref.container_id
            assert all(data['Config']['Labels'][k] == v for k, v in ref.expected_labels.items())
            host = data['HostConfig']
            assert host['ReadonlyRootfs'] and data['Config']['User'] == _runtime_module._worker_user()
            assert host['CapDrop'] == ['ALL'] and host['SecurityOpt'] == ['no-new-privileges']
            assert host['NetworkMode'] == 'none' and host['PidsLimit'] == fixture.policy.pids
            assert host['Memory'] == host['MemorySwap'] == fixture.policy.memory_bytes
            assert host['NanoCpus'] == 1_000_000_000 and not host.get('DeviceRequests')
            assert host['LogConfig'] == {'Type': 'json-file', 'Config': {'max-file': '2', 'max-size': '10m'}}
            assert {(m['Destination'], m['RW']) for m in data['Mounts']} == {('/scratch', True), ('/probe.py', False)}
            owned = OwnedContainer(cli, ref, coordinator.operation_id, data['Config']['Labels']['podmuncher.runtime'])
            logged = await owned.logs(owned.binding)
            assert logged.returncode == 0 and b'"started"' in logged.stdout
            memory = DockerCgroupMemory(ref)
            ancestry = await asyncio.to_thread(memory.read_memory_cgroup, deadline=time.monotonic() + 12)
            assert ancestry.identity == ref and ancestry.ancestry is not None
            assert 0 < ancestry.effective_limit <= fixture.policy.memory_bytes
            assert ancestry.available is not None and ancestry.available > 0
            assert ancestry.ancestry.physical_available > 0
            hardware = await HardwareSampler(worker=memory, timeout=12).sample_async()
            assert hardware.worker_ram_limit.status is FieldStatus.OK
            assert 0 < hardware.worker_ram_limit.value <= fixture.policy.memory_bytes
            assert hardware.worker_ram_available.value > 0
            assert hardware.worker_cpu_limit_cores.value == 1
            assert hardware.worker_gpu_visible.value is False
            assert hardware.runtime_compute_types.value is None
            marker = fixture.mounts.scratch / ('.inspected-' + req.stage_id)
            await guard.io(lambda: asyncio.to_thread(marker.touch, exist_ok=False))
            return True

        def consume(item):
            seen.append(item.event)
            if item.event == 'started':
                inspection.append(coordinator.track(inspect_running()))
        result = await runtime.run(req, on_event=consume)
        assert inspection and await inspection[0]
        assert result.completed and result.exit_code == 0 and seen == ['started', 'progress', 'artifact', 'completed']
        artifact = fixture.mounts.scratch / req.assigned_slots['probe_metadata']
        assert artifact.read_bytes() == b'fixture\n'
        assert result.artifact_claims[0].payload['sha256'] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert lease.released == 1 and not lease.held
        removed = await cli.run('ps', '-a', '--no-trunc', '--format', '{{.ID}}',
            '--filter', 'label=podmuncher.instance=' + runtime.instance, owner='fixture-verification-' + uuid4().hex)
        assert removed.returncode == 0 and not removed.stdout.strip()

        # A second admitted worker exercises a real owned stop, not CLI death.
        cancel_lease = Lease()
        cancel_coordinator = CancellationCoordinator('fixture-cancel-' + uuid4().hex, cancel_lease,
            stop_grace_seconds=12, reap_timeout_seconds=12, callback_timeout_seconds=12)
        cancelled = DockerRuntime(cli, WriteGuard(cancel_coordinator, fixture.storage.sample,
            byte_limit=65536, timeout_seconds=90, sample_timeout_seconds=2.9),
            instance='fixture-cancel-' + uuid4().hex,
            policy=fixture.policy, mounts=fixture.mounts)
        with pytest.raises(RuntimeFailure) as error:
            await cancelled.run(request(), on_event=lambda e: cancelled.obsolete() if e.event == 'started' else None)
        assert error.value.reason_code is ReasonCode.CANCELLED
        assert cancel_lease.released == 1 and not cancel_lease.held
        removed = await cli.run('ps', '-a', '--no-trunc', '--format', '{{.ID}}',
            '--filter', 'label=podmuncher.instance=' + cancelled.instance, owner='fixture-verification-' + uuid4().hex)
        assert removed.returncode == 0 and not removed.stdout.strip()
    try:
        asyncio.run(execute())
    finally:
        fixture.session.close()
    print('PODMUNCHER_SMOKE_OK')


@pytest.mark.parametrize('uid,gid,expected', [(1000, 1000, '1000:1000'), (1001, 100, '1001:100')])
def test_posix_host_worker_runs_as_invoking_uid_gid(tmp_path, monkeypatch, uid, gid, expected):
    monkeypatch.setattr(os, 'getuid', lambda: uid, raising=False)
    monkeypatch.setattr(os, 'getgid', lambda: gid, raising=False)
    runtime, factory, _lease, req = make_runtime(tmp_path)
    asyncio.run(runtime.run(req))
    create = next(c[3:] for c in factory.calls if c[3] == 'create')
    assert create[create.index('--user') + 1] == expected
    assert '--read-only' in create and create[create.index('--cap-drop') + 1] == 'ALL'


def test_posix_root_host_uid_is_refused_before_any_docker_command(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'getuid', lambda: 0, raising=False)
    monkeypatch.setattr(os, 'getgid', lambda: 0, raising=False)
    runtime, factory, _lease, req = make_runtime(tmp_path)
    with pytest.raises(RuntimeFailure) as error:
        asyncio.run(runtime.run(req))
    assert error.value.reason_code is ReasonCode.DOCKER_CONTEXT_UNSUPPORTED
    assert factory.calls == []


def test_windows_host_keeps_fixed_worker_user(tmp_path, monkeypatch):
    monkeypatch.delattr(os, 'getuid', raising=False)
    monkeypatch.delattr(os, 'getgid', raising=False)
    assert _runtime_module._worker_user() == '65532:65532'
    runtime, factory, _lease, req = make_runtime(tmp_path)
    asyncio.run(runtime.run(req))
    create = next(c[3:] for c in factory.calls if c[3] == 'create')
    assert create[create.index('--user') + 1] == '65532:65532'


def test_guard_timeout_above_one_hour_reaches_pull_and_attach_unchanged(tmp_path):
    policy = WorkerPolicy('asr_cpu', IMAGE, ('python', '-I', '/probe.py'), 1, 1, allow_pull=True)
    runtime, factory, _lease, req = make_runtime(tmp_path, policy=policy, timeout=7200)
    seen, original = {}, runtime.cli.run
    async def record(*args, **kwargs):
        if args[0] in ('pull', 'start'):
            seen[args[0]] = kwargs['timeout']
        return await original(*args, **kwargs)
    runtime.cli.run = record
    asyncio.run(runtime.run(req))
    assert seen == {'pull': 7200, 'start': 7200}


def test_raw_accepts_long_finite_timeout_and_rejects_bad_ones(tmp_path):
    runtime, factory, _lease, req = make_runtime(tmp_path)
    cli = runtime.cli
    async def run():
        await cli._raw(('--host', TARGET.endpoint, 'ps'), owner='long-timeout', timeout=7200)
        for bad in (0, -1, float('inf'), float('nan'), True, '5'):
            with pytest.raises(RuntimeFailure) as error:
                await cli._raw(('--host', TARGET.endpoint, 'ps'), owner='long-timeout', timeout=bad)
            assert error.value.reason_code is ReasonCode.INVALID_INPUT
    asyncio.run(run())


def test_native_owner_is_released_after_completion_and_stays_sealed(tmp_path):
    runtime, factory, _lease, req = make_runtime(tmp_path)
    operation = runtime.guard.coordinator.operation_id
    asyncio.run(runtime.run(req))
    assert operation not in runtime.cli._owners
    async def late():
        await runtime.cli.run('ps', owner=operation)
    before = len(factory.calls)
    with pytest.raises(RuntimeFailure) as error:
        asyncio.run(late())
    assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert len(factory.calls) == before
