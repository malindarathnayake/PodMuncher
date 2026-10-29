"""Identity-bound hardware probes; live case only samples the native host."""

import asyncio
import json
import threading
from collections import namedtuple
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from podmuncher.docker_storage import DiscoveryError, ProcessResult
from podmuncher.domains import Backend, FieldStatus, ReasonCode
from podmuncher.hardware import (
    _CGROUP_SCRIPT,
    _DAEMON_SCRIPT,
    _ENGINE_INFO_FORMAT,
    _GPU_ARGV,
    _NAMESPACED_SCRIPT,
    _SYSCTL_CPU_BRAND_ARGV,
    _SYSCTL_TRANSLATED_ARGV,
    AdmittedWorkerRef,
    CgroupAncestor,
    CgroupAncestry,
    CgroupMemory,
    DockerCgroupMemory,
    HardwareEvidence,
    HardwareSampler,
    RuntimeComputeEvidence,
    _cpu_percent,
    _DockerProbe,
    _mib,
)
from podmuncher.snapshots import Measurement

NOW = datetime(2026, 10, 9, tzinfo=UTC)
GPU_UUID = 'GPU-12345678-1234-1234-1234-123456789012'
REF = AdmittedWorkerRef('default', 'npipe:////./pipe/docker_engine', 'a' * 64,
                        'mine', 'job1', 'attempt1')
MAC_REF = AdmittedWorkerRef('desktop-linux', 'unix:///Users/me/.docker/run/docker.sock', 'a' * 64,
                            'mine', 'job1', 'attempt1')
CPU = namedtuple('CPU', 'user nice system idle iowait irq softirq steal guest guest_nice')
GIB = 1024**3
PIDS = (411, 412)
WORKER_PATH = '/' + REF.container_id
ENGINE_ARGV = ('docker', 'info', '--format', _ENGINE_INFO_FORMAT)
MAC_ENGINE_INFO = b'aarch64 8217059328 10\n'


def namespaced_transcript(*, limit=4 * GIB, current=GIB, current_after=None, total=16 * GIB,
                          available=12 * GIB, available_after=None, identity='23:4567:41ed',
                          identity_after=None, membership='0::/'):
    """The darwin in-namespace reader output: two leaf samples with VM physical memory."""
    rows = ['namespaced-v1']
    for number in (1, 2):
        usage = current_after if number == 2 and current_after is not None else current
        physical = available_after if number == 2 and available_after is not None else available
        ident = identity_after if number == 2 and identity_after is not None else identity
        rows.append(f'sample\t{number}\t{membership}\t{ident}\t{"max" if limit is None else limit}\t{usage}')
        rows.append(f'physical\t{total // 1024}\t{physical // 1024}')
    return ('\n'.join(rows) + '\n').encode()


def exec_argv(ref=MAC_REF):
    return ('docker', '--context', ref.context, 'exec', ref.container_id, 'sh', '-c',
            _NAMESPACED_SCRIPT, 'cgroup-reader')


def hierarchy_transcript(nodes=None, *, after=None, total=16 * GIB, available=12 * GIB,
                         available_after=None):
    nodes = nodes or (('/worker', 4 * GIB, GIB),)
    after = nodes if after is None else after
    view = ('view\tcgroup:[10]\tpid:[11]\tmnt:[12]\t100\t23:1:41ed\t'
            '295 289 0:23 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw')
    rows = ['hierarchy-v1']
    for number, chain in enumerate((nodes, after), 1):
        chain = tuple((path.replace('/worker', WORKER_PATH), limit, current)
                      for path, limit, current in chain)
        processes = [f'path\t{chain[-1][0]}', *(f'pid\t{pid}\t{1000 + pid}' for pid in PIDS)]
        rows.extend([f'sample\t{number}', view, *processes])
        for index, (path, limit, current) in enumerate(chain):
            rows.append(f'node\t{path}\t23:{100 + index}:41ed\t'
                        f'{"max" if limit is None else limit}\t{current}')
        physical = available_after if number == 2 and available_after is not None else available
        rows.extend([f'physical\t{total // 1024}\t{physical // 1024}', view, *processes])
    return ('\n'.join(rows) + '\n').encode()


def reader_argv(*, native=False, seconds='30.000'):
    prefix = () if native else ('wsl.exe', '-d', 'docker-desktop', '-u', 'root', '--exec')
    return (*prefix, 'timeout', '-s', 'KILL', seconds, 'sh', '-c', _DAEMON_SCRIPT, 'daemon-reader', _CGROUP_SCRIPT,
            str(PIDS[0]), REF.container_id, *(str(pid) for pid in PIDS))


def fake_clock():
    state = {'time': NOW, 'mono': 100.0}
    return state, lambda: state['time']


def docker(*args):
    return ('docker', '--context', REF.context, *args)


def make_probe_results(*, gpu=True, stats=b'12.5%', context=True, nano=4000000000,
                       quota=0, period=0, cpuset='', cores=8):
    endpoint = REF.endpoint if context else 'tcp://remote:2375'
    return {
        docker('context', 'inspect', REF.context): json.dumps([{
            'Name': REF.context, 'Endpoints': {'docker': {'Host': endpoint}}}]).encode(),
        docker('container', 'inspect', REF.container_id): json.dumps([{
            'Id': REF.container_id, 'State': {'Running': True, 'Pid': PIDS[0],
                                            'StartedAt': '2026-10-09T00:00:00.000000000Z'},
            'Config': {'Labels': REF.expected_labels}, 'HostConfig': {
                'NanoCpus': nano, 'CpuQuota': quota, 'CpuPeriod': period,
                'CpusetCpus': cpuset}}]).encode(),
        docker('info', '--format', '{{json .NCPU}}'): json.dumps(cores).encode(),
        docker('info', '--format', _ENGINE_INFO_FORMAT): b'x86_64 16755163136 8\n',
        ENGINE_ARGV: b'x86_64 16755163136 8\n',
        docker('stats', '--no-stream', '--format', '{{.CPUPerc}}', REF.container_id): stats,
        docker('top', REF.container_id, '-eo', 'pid'): b'PID\n411\n412\n',
        reader_argv(): hierarchy_transcript(),
        _GPU_ARGV: f'0, {GPU_UUID}, NVIDIA RTX 5090, 580.97, 32768, 16000, 16768, 20.5\n'.encode() if gpu else b'',
    }


def make_mac_results(*, translated=b'0\n', brand=b'Apple M4\n', engine=MAC_ENGINE_INFO):
    results = make_probe_results(gpu=False)
    results.update({_SYSCTL_TRANSLATED_ARGV: translated, _SYSCTL_CPU_BRAND_ARGV: brand, ENGINE_ARGV: engine,
                    docker('info', '--format', _ENGINE_INFO_FORMAT): engine})
    return results


class FakeRunner:
    def __init__(self, results, *, timeout=None):
        self.results, self.calls, self.timeout = results, [], timeout
        self.deadlines = []

    def __call__(self, argv, *, deadline, env):
        self.calls.append(argv)
        self.deadlines.append(deadline)
        assert deadline > 0
        if argv == self.timeout:
            raise DiscoveryError(ReasonCode.REQUEST_TIMEOUT)
        value = self.results.get(argv, b'')
        return ProcessResult(0 if value else 1, value)


def test_large_owned_inspect_is_accepted_with_bounded_transport():
    results = make_probe_results()
    inspect_argv = docker('container', 'inspect', REF.container_id)
    owned = json.loads(results[inspect_argv])
    owned[0]['Irrelevant'] = 'x' * 9000
    results[inspect_argv] = json.dumps(owned).encode()
    assert 8192 < len(results[inspect_argv]) <= 65536
    results[reader_argv(seconds='1.000')] = hierarchy_transcript()

    memory = DockerCgroupMemory(
        REF, runner=FakeRunner(results), environ={}, monotonic=lambda: 100,
        platform='win32', clock=lambda: NOW,
    ).read_memory_cgroup(deadline=101)

    assert tuple(node.path for node in memory.ancestry.nodes) == (WORKER_PATH,)
    assert memory.effective_limit == 4 * GIB
    assert memory.available == 3 * GIB
    assert DockerCgroupMemory(REF).runner.max_output == 65536
    assert HardwareSampler().runner.max_output == 65536


def test_docker_probe_rejects_oversized_owned_inspect():
    argv = docker('container', 'inspect', REF.container_id)
    probe = _DockerProbe(REF, FakeRunner({argv: b'x' * 65537}), {}, lambda: 100, 101)
    with pytest.raises(ValueError, match='Malformed process evidence'):
        probe.call('container', 'inspect', REF.container_id)


def test_docker_probe_keeps_non_inspect_evidence_at_8192():
    argv = docker('info')
    probe = _DockerProbe(REF, FakeRunner({argv: b'x' * 8193}), {}, lambda: 100, 101)
    with pytest.raises(ValueError, match='Malformed process evidence'):
        probe.call('info')

    probe = _DockerProbe(REF, FakeRunner({argv: b'x' * 8192}), {}, lambda: 100, 101)
    assert probe.call('info') == b'x' * 8192


class FakeWorker:
    def __init__(self, limit=4294967296, current=1073741824, identity=REF):
        self.identity = identity
        self.limit, self.current, self.calls = limit, current, 0

    def read_memory_cgroup(self, *, deadline):
        self.calls += 1
        ancestry = CgroupAncestry((CgroupAncestor('/worker', 23, 100, self.limit,
                                                 self.current, self.current),),
                                 16 * GIB, 12 * GIB, NOW)
        return CgroupMemory(self.identity, 2, self.limit, self.current, ancestry)


class Runtime:
    def __init__(self, identity=REF):
        self.identity, self.calls, self.backends = identity, 0, []

    def compute_types(self, backend, *, deadline):
        self.calls += 1
        self.backends.append(backend)
        return RuntimeComputeEvidence(self.identity, backend, ('int8', 'float32'),
                                      self.identity.gpu_uuid if backend is Backend.CUDA else None)


class FakeMemory:
    total = 64 * 1024**3
    available = 40 * 1024**3
    used = 24 * 1024**3


def make_sampler(*, runner=None, worker=None, clock=None, runtime=None, **kwargs):
    counter = {'n': 0}

    def cpu_times():
        counter['n'] += 1
        return CPU(counter['n'], 0, 0, counter['n'], 0, 0, 0, 0, 0, 0)

    defaults = {'runner': runner or FakeRunner(make_probe_results()), 'worker': worker,
                    'runtime': runtime, 'clock': clock or (lambda: NOW), 'sleep': lambda _: None,
                    'cpu_times': cpu_times, 'virtual_memory': lambda: FakeMemory(),
                    'cpu_count': lambda logical: 16, 'environ': {}, 'system': lambda: 'Windows',
                    'machine': lambda: 'AMD64', 'processor': lambda: 'Intel64 Family 6 Model 183, GenuineIntel'}
    defaults.update(kwargs)
    return HardwareSampler(**defaults)


def test_host_gpu_worker_lower_limit_and_cpu_runtime_despite_host_gpu():
    runner, worker, runtime = FakeRunner(make_probe_results()), FakeWorker(), Runtime()
    evidence = make_sampler(runner=runner, worker=worker, runtime=runtime).sample()
    assert evidence.host_cpu_percent.value == 50
    assert evidence.host_ram_total.value == 64 * 1024**3
    assert evidence.worker_ram_limit.value == 4 * 1024**3
    assert evidence.worker_ram_current.value == 1024**3
    assert evidence.worker_ram_available.value == 3 * 1024**3
    assert evidence.worker_ram_limit.source == 'worker_cgroup_v2_hierarchy'
    assert evidence.worker_gpu_visible.value is False
    assert evidence.gpu_name.value == 'NVIDIA RTX 5090'
    assert evidence.gpu_vram_free.value == 16000 * 1048576
    assert evidence.runtime_compute_types.value == ('int8', 'float32')
    assert runtime.backends == [Backend.CPU]
    assert worker.calls == 1
    assert all(argv[:3] == ('docker', '--context', 'default') for argv in runner.calls if argv[0] == 'docker')


def test_explicit_unlimited_hierarchy_is_bounded_by_vm_physical_memory():
    item = make_sampler(worker=FakeWorker(limit=None)).sample()
    assert item.worker_ram_limit.value == 16 * GIB
    assert item.worker_ram_limit.status is FieldStatus.OK
    assert item.worker_ram_available.value == 12 * GIB
    assert item.worker_ram_current.value == 1024**3


@pytest.mark.parametrize('limit', [None, 8 * GIB])
def test_leaf_numbers_never_prove_effective_limit_or_headroom(limit):
    class LeafOnly(FakeWorker):
        def read_memory_cgroup(self, *, deadline):
            return CgroupMemory(REF, 2, limit, GIB)

    item = make_sampler(worker=LeafOnly()).sample()
    assert item.worker_ram_limit.value is None
    assert item.worker_ram_available.value is None
    assert item.worker_ram_current.value == GIB


@pytest.mark.parametrize('kind', ['remote', 'labels', 'id', 'stopped', 'running_string'])
def test_wrong_context_or_ownership_never_reaches_cgroup(kind):
    results = make_probe_results(context=kind != 'remote')
    key = docker('container', 'inspect', REF.container_id)
    owned = json.loads(results[key])
    if kind == 'labels':
        owned[0]['Config']['Labels'] = {}
    if kind == 'id':
        owned[0]['Id'] = 'b' * 64
    if kind in ('stopped', 'running_string'):
        owned[0]['State']['Running'] = False if kind == 'stopped' else 'true'
    results[key] = json.dumps(owned).encode()
    worker = FakeWorker()
    item = make_sampler(runner=FakeRunner(results), worker=worker).sample()
    assert worker.calls == 0
    assert item.worker_ram_limit.value is None
    assert item.worker_ram_limit.reason in (ReasonCode.DOCKER_CONTEXT_UNSUPPORTED, ReasonCode.UNAUTHORIZED)


def test_context_active_swap_is_pinned_and_endpoint_change_discards_bundle():
    class SwappingRunner(FakeRunner):
        active = 'default'
        checks = 0

        def __call__(self, argv, *, deadline, env):
            self.active = 'other'
            assert argv[0] != 'docker' or argv[1:3] == ('--context', 'default')
            if argv == docker('context', 'inspect', REF.context):
                self.checks += 1
                if self.checks == 2:
                    self.results[argv] = json.dumps([{'Name': REF.context,
                        'Endpoints': {'docker': {'Host': 'npipe:////./pipe/dockerDesktopLinuxEngine'}}}]).encode()
            return super().__call__(argv, deadline=deadline, env=env)

    runner = SwappingRunner(make_probe_results())
    item = make_sampler(runner=runner, worker=FakeWorker()).sample()
    assert runner.active == 'other'
    assert item.worker_ram_limit.value is None
    assert item.worker_ram_limit.reason is ReasonCode.DOCKER_CONTEXT_UNSUPPORTED
    assert item.worker_cpu_percent.value is None


def test_environment_conflict_rechecked_before_each_docker_command():
    env = {}

    class ChangingRunner(FakeRunner):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            if argv[0] == 'docker':
                env['DOCKER_HOST'] = 'tcp://remote:2375'
            return result

    worker = FakeWorker()
    item = make_sampler(worker=worker, runner=ChangingRunner(make_probe_results()), environ=env).sample()
    assert worker.calls == 0
    assert item.worker_ram_limit.reason is ReasonCode.DOCKER_CONTEXT_UNSUPPORTED


@pytest.mark.parametrize(('params', 'expected'), [
    ({'stats': b'250%'}, 62.5),
    ({'stats': b'25%', 'nano': 0, 'quota': 50000, 'period': 100000}, 50),
    ({'stats': b'100%', 'nano': 0, 'cpuset': '0-1,3'}, 100 / 3),
    ({'stats': b'50%', 'nano': 500000000}, 100),
])
def test_cpu_percent_normalized_by_measured_effective_entitlement(params, expected):
    item = make_sampler(worker=FakeWorker(), runner=FakeRunner(make_probe_results(**params))).sample()
    assert item.worker_cpu_percent.value == pytest.approx(expected)
    assert item.worker_cpu_raw_percent.value == float(params['stats'].strip(b'%'))
    assert item.worker_cpu_limit_cores.status is FieldStatus.OK


@pytest.mark.parametrize('params', [{'cores': None}, {'cores': True}, {'nano': '4'},
                                   {'quota': 1, 'period': 0}, {'cpuset': '0-9999999'}])
def test_unknown_or_malformed_entitlement_is_not_inferred_from_host(params):
    item = make_sampler(worker=FakeWorker(), runner=FakeRunner(make_probe_results(**params))).sample()
    assert item.worker_cpu_limit_cores.value is None
    assert item.worker_cpu_percent.value is None
    assert item.worker_cpu_raw_percent.value == 12.5


def test_cpu_over_limit_race_is_unknown_and_memory_clip_has_provenance():
    item = make_sampler(worker=FakeWorker(limit=100, current=101),
                        runner=FakeRunner(make_probe_results(stats=b'450%'))).sample()
    assert item.worker_cpu_percent.value is None
    assert item.worker_cpu_percent.reason is ReasonCode.CONFLICT
    assert item.worker_cpu_raw_percent.value == 450
    assert item.worker_ram_available.value == 0
    assert 'clipped_current_above_limit' in item.worker_ram_available.source


def test_slow_cpu_stats_runs_before_selection_reads_and_cannot_age_them():
    state, clock = fake_clock()
    events = []
    stats_argv = docker('stats', '--no-stream', '--format', '{{.CPUPerc}}', REF.container_id)
    info_argv = docker('info', '--format', '{{json .NCPU}}')
    costs = {stats_argv: 1.5, info_argv: 0.3, docker('context', 'inspect', REF.context): 0.1,
             docker('container', 'inspect', REF.container_id): 0.1}

    class TimedRunner(FakeRunner):
        def __call__(self, argv, **kwargs):
            start = clock()
            state['time'] += timedelta(seconds=costs.get(argv, 0))
            name = argv[3] if argv[0] == 'docker' else argv[0]
            if argv[-1] == _ENGINE_INFO_FORMAT:
                name = 'engine_info'  # host-level engine facts; not a selection read
            events.append((name, start, clock()))
            return super().__call__(argv, **kwargs)

    class TimedWorker(FakeWorker):
        def read_memory_cgroup(self, *, deadline):
            stamp = clock()  # stamped at read start, as the real adapter does
            state['time'] += timedelta(seconds=1.0)
            events.append(('cgroup', stamp, clock()))
            memory = super().read_memory_cgroup(deadline=deadline)
            return replace(memory, ancestry=replace(memory.ancestry, sample_time=stamp))

    item = make_sampler(runner=TimedRunner(make_probe_results()), worker=TimedWorker(),
                        runtime=Runtime(), clock=clock, monotonic=lambda: 100.0).sample()
    end = clock()
    names = [name for name, _, _ in events]
    timing = {name: (begin, finish) for name, begin, finish in events}
    # Call order stats < info < cgroup, each with observable simulated duration.
    assert names.index('stats') < names.index('info') < names.index('cgroup')
    assert (timing['stats'][1] - timing['stats'][0]).total_seconds() == pytest.approx(1.5)
    assert (timing['info'][1] - timing['info'][0]).total_seconds() == pytest.approx(0.3)
    assert (timing['cgroup'][1] - timing['cgroup'][0]).total_seconds() == pytest.approx(1.0)
    assert timing['stats'][1] <= timing['info'][0] and timing['info'][1] <= timing['cgroup'][0]

    def age(field):
        assert field.status is FieldStatus.OK
        return (end - datetime.fromisoformat(field.sample_time)).total_seconds()

    selection = ('worker_ram_limit', 'worker_ram_current', 'worker_ram_available',
                 'worker_gpu_visible', 'runtime_compute_types')
    # cgroup read 1.0 s + final identity recheck 0.2 s; the 1.5 s stats call is excluded.
    assert all(age(getattr(item, key)) <= 1.2 + 1e-9 for key in selection)  # < 2.0 s
    # GPU visibility is stamped at the docker info read (end of info), not the recheck.
    info_age = (end - timing['info'][1]).total_seconds()
    assert age(item.worker_gpu_visible) == pytest.approx(info_age) == pytest.approx(1.2)
    assert age(item.worker_gpu_visible) == pytest.approx(age(item.worker_cpu_limit_cores))
    assert age(item.worker_gpu_visible) > 0.2  # older than the 0.2 s recheck window
    assert item.worker_cpu_raw_percent.value == 12.5
    assert age(item.worker_cpu_raw_percent) >= age(item.worker_ram_limit)


def test_unrelated_runtime_or_cgroup_identity_is_rejected():
    other = replace(REF, job='other')
    runtime = Runtime(other)
    item = make_sampler(worker=FakeWorker(), runtime=runtime).sample()
    assert item.runtime_compute_types.reason is ReasonCode.WORKER_PROTOCOL_MISMATCH
    assert runtime.calls == 0

    class WrongWorker(FakeWorker):
        def read_memory_cgroup(self, *, deadline):
            return CgroupMemory(other, 2, 100, 10)

    item = make_sampler(worker=WrongWorker()).sample()
    assert item.worker_ram_limit.reason is ReasonCode.WORKER_PROTOCOL_MISMATCH
    assert make_sampler(runtime=Runtime()).sample().runtime_compute_types.value is None


def test_cuda_requires_matching_protected_slot_zero_identity():
    identity = replace(REF, gpu_visible=True, gpu_uuid=GPU_UUID)
    runtime = Runtime(identity)
    item = make_sampler(worker=FakeWorker(identity=identity), runtime=runtime).sample()
    assert item.runtime_compute_types.value == ('int8', 'float32')
    assert runtime.backends == [Backend.CUDA]
    unknown = replace(identity, gpu_uuid='GPU-99999999-1234-1234-1234-123456789012')
    runtime = Runtime(unknown)
    item = make_sampler(worker=FakeWorker(identity=unknown), runtime=runtime).sample()
    assert item.runtime_compute_types.value is None
    assert runtime.calls == 0


def test_real_cgroup_adapter_exact_bounded_argv_and_protocol():
    runner = FakeRunner(make_probe_results())
    runner.results[reader_argv(seconds='1.000')] = hierarchy_transcript()
    adapter = DockerCgroupMemory(REF, runner=runner, environ={}, monotonic=lambda: 100,
                                 platform='win32', clock=lambda: NOW)
    result = adapter.read_memory_cgroup(deadline=101)
    assert result.limit == 4 * GIB and result.current == GIB
    assert result.effective_limit == 4 * GIB and result.available == 3 * GIB
    assert result.ancestry.sample_time == NOW
    assert runner.calls == [docker('context', 'inspect', REF.context),
                           docker('container', 'inspect', REF.container_id),
                           docker('top', REF.container_id, '-eo', 'pid'), reader_argv(seconds='1.000'),
                           docker('context', 'inspect', REF.context),
                           docker('container', 'inspect', REF.container_id),
                           docker('top', REF.container_id, '-eo', 'pid')]
    assert 'memory.max' in _CGROUP_SCRIPT and 'memory.current' in _CGROUP_SCRIPT
    assert '--cgroup' not in _DAEMON_SCRIPT
    assert not any(argv[0] == 'docker' and 'exec' in argv for argv in runner.calls)
    assert set(runner.deadlines) == {101}


def read_hierarchy(raw, **kwargs):
    results = make_probe_results()
    results[reader_argv()] = raw
    return DockerCgroupMemory(REF, runner=FakeRunner(results), environ={}, platform='win32',
                              clock=lambda: NOW, **kwargs).read_memory_cgroup(deadline=1e20)


def test_parent_sibling_usage_controls_available_not_leaf_subtraction():
    nodes = (('/restricted', 2 * GIB, 7 * GIB // 4),
             ('/restricted/worker', 8 * GIB, GIB))
    memory = read_hierarchy(hierarchy_transcript(nodes))
    assert memory.limit == 8 * GIB and memory.current == GIB
    assert memory.effective_limit == 2 * GIB
    assert memory.available == GIB // 4

    class Worker(FakeWorker):
        def read_memory_cgroup(self, *, deadline):
            return memory

    evidence = make_sampler(worker=Worker()).sample()
    assert evidence.worker_ram_limit.value == 2 * GIB
    assert evidence.worker_ram_current.value == GIB
    assert evidence.worker_ram_available.value == GIB // 4


def test_tighter_grandparent_and_normal_usage_change_are_conservative():
    nodes = (('/outer', 3 * GIB, 2 * GIB), ('/outer/parent', 5 * GIB, 2 * GIB),
             ('/outer/parent/worker', 8 * GIB, GIB))
    after = (('/outer', 3 * GIB, 11 * GIB // 4), ('/outer/parent', 5 * GIB, GIB),
             ('/outer/parent/worker', 8 * GIB, GIB // 2))
    memory = read_hierarchy(hierarchy_transcript(nodes, after=after))
    assert memory.effective_limit == 3 * GIB
    assert memory.current == GIB // 2
    assert memory.available == GIB // 4


def test_all_max_and_physical_pressure_bound_headroom():
    nodes = (('/restricted', None, 3 * GIB), ('/restricted/worker', None, GIB))
    memory = read_hierarchy(hierarchy_transcript(nodes, total=4 * GIB, available=2 * GIB,
                                               available_after=GIB // 4))
    assert memory.limit is None
    assert memory.effective_limit == 4 * GIB
    assert memory.available == GIB // 4


@pytest.mark.parametrize('change', [
    lambda raw: b'',
    lambda raw: b'v2\n0::/\n8589934592\n1073741824\n',
    lambda raw: raw.replace(b'cgroup2 cgroup2', b'cgroup cgroup'),
    lambda raw: raw.replace(b'23:1:41ed', b'23:999:41ed'),
    lambda raw: raw.replace(b'0:23 / /sys/fs/cgroup', b'0:23 /hidden /sys/fs/cgroup'),
    lambda raw: raw.replace(b'23:100:41ed', b'23:100:a1ff'),
    lambda raw: raw.replace(b'23:100:41ed', b'24:100:41ed'),
    lambda raw: raw.replace(WORKER_PATH.encode(), WORKER_PATH.encode() + b' (deleted)'),
    lambda raw: raw.replace(WORKER_PATH.encode(), b'/../..' + WORKER_PATH.encode()),
    lambda raw: raw.replace(REF.container_id.encode(), b'b' * 64),
    lambda raw: raw.replace(b'pid\t411\t1411', b'pid\t411\t1412', 1),
    lambda raw: raw.replace(b'cgroup:[10]', b'cgroup:[99]', 1),
    lambda raw: raw.replace(b'mnt:[12]', b'mnt:[99]', 1),
    lambda raw: raw.replace(b'23:1:41ed', b'24:1:41ed', 1),
    lambda raw: raw.replace(b'23:100:41ed', b'23:101:41ed', 1),
    lambda raw: raw.replace(b'4294967296', b'2147483648', 1),
    lambda raw: raw.replace(b'4294967296', b'0'),
    lambda raw: raw.replace(b'1073741824', b'-1'),
    lambda raw: raw.replace(b'1073741824', b'NaN'),
    lambda raw: raw.replace(b'path\t' + WORKER_PATH.encode(),
                            b'path\t' + WORKER_PATH.encode() + b'/subgroup', 1),
    lambda raw: raw + b'extra\n',
    lambda raw: b'x' * 32769,
])
def test_missing_hidden_changed_or_malformed_hierarchy_is_unknown(change):
    memory = read_hierarchy(change(hierarchy_transcript()))
    assert memory.ancestry is None
    assert memory.current is None and memory.effective_limit is None and memory.available is None


@pytest.mark.parametrize('change', ['pid', 'started', 'process_set', 'labels', 'stopped'])
def test_owned_identity_bracket_discards_restart_and_process_changes(change):
    class ChangingRunner(FakeRunner):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            if argv == reader_argv():
                key = docker('container', 'inspect', REF.container_id)
                info = json.loads(self.results[key])[0]
                if change == 'pid':
                    info['State']['Pid'] += 1
                elif change == 'started':
                    info['State']['StartedAt'] = '2026-10-09T00:00:01Z'
                elif change == 'labels':
                    info['Config']['Labels'] = {}
                elif change == 'stopped':
                    info['State']['Running'] = False
                else:
                    self.results[docker('top', REF.container_id, '-eo', 'pid')] = b'PID\n411\n413\n'
                self.results[key] = json.dumps([info]).encode()
            return result

    with pytest.raises(DiscoveryError):
        DockerCgroupMemory(REF, runner=ChangingRunner(make_probe_results()), environ={},
                           platform='win32').read_memory_cgroup(deadline=1e20)


def test_native_linux_uses_same_daemon_view_without_wsl():
    identity = replace(REF, endpoint='unix:///var/run/docker.sock')
    results = make_probe_results()
    context = json.loads(results[docker('context', 'inspect', REF.context)])
    context[0]['Endpoints']['docker']['Host'] = identity.endpoint
    results[docker('context', 'inspect', REF.context)] = json.dumps(context).encode()
    results[reader_argv(native=True)] = hierarchy_transcript()
    runner = FakeRunner(results)
    memory = DockerCgroupMemory(identity, runner=runner, environ={}, platform='linux',
                                 clock=lambda: NOW).read_memory_cgroup(deadline=1e20)
    assert memory.available == 3 * GIB
    assert reader_argv(native=True) in runner.calls


def test_reader_timeout_uses_shared_deadline_and_original_memory_timestamp():
    runner = FakeRunner(make_probe_results(), timeout=reader_argv(seconds='1.000'))
    with pytest.raises(DiscoveryError) as error:
        DockerCgroupMemory(REF, runner=runner, environ={}, platform='win32',
                           monotonic=lambda: 100).read_memory_cgroup(deadline=101)
    assert error.value.reason is ReasonCode.REQUEST_TIMEOUT
    memory = read_hierarchy(hierarchy_transcript())

    class Worker(FakeWorker):
        def read_memory_cgroup(self, *, deadline):
            return memory

    item = make_sampler(worker=Worker(), clock=lambda: NOW + timedelta(seconds=11)).sample()
    assert item.worker_ram_available.status is FieldStatus.STALE
    assert item.worker_ram_available.sample_time == NOW.isoformat().replace('+00:00', 'Z')


def test_concrete_adapter_remote_context_never_executes():
    runner = FakeRunner(make_probe_results(context=False))
    with pytest.raises(DiscoveryError):
        DockerCgroupMemory(REF, runner=runner, environ={}).read_memory_cgroup(deadline=1e20)
    assert not any('exec' in argv for argv in runner.calls)


@pytest.mark.parametrize('memory', ['100, 101, 0', '100, 60, 50', 'N/A, N/A, N/A', 'NaN, -1, 100'])
def test_incoherent_or_missing_gpu_memory_is_null(memory):
    results = make_probe_results()
    results[_GPU_ARGV] = f'0, {GPU_UUID}, NVIDIA RTX 5090, 580.97, {memory}, 20\n'.encode()
    item = make_sampler(runner=FakeRunner(results)).sample()
    assert item.gpu_name.value == 'NVIDIA RTX 5090'
    assert all(getattr(item, key).value is None for key in ('gpu_vram_total', 'gpu_vram_free', 'gpu_vram_used'))
    assert _mib('20 GiB') is None


@pytest.mark.parametrize(('total', 'free', 'used'), [
    (100, 40, 40),
    (12282, 6044, 5972),
])
def test_gpu_memory_with_reserved_gap_preserves_measured_values(total, free, used):
    results = make_probe_results()
    results[_GPU_ARGV] = (
        f'0, {GPU_UUID}, NVIDIA RTX 5090, 580.97, {total}, {free}, {used}, 20\n'
    ).encode()
    item = make_sampler(runner=FakeRunner(results)).sample()
    assert item.gpu_vram_total.value == total * 1048576
    assert item.gpu_vram_free.value == free * 1048576
    assert item.gpu_vram_used.value == used * 1048576


@pytest.mark.parametrize('raw', [b'', b'0, GPU-unknown, GPU, 1, 1, 1, 0, 0\n',
    f'1, {GPU_UUID}, GPU, 1, 1, 1, 0, 0\n'.encode(),
    (f'0, {GPU_UUID}, GPU, 1, 1, 1, 0, 0\n' * 2).encode()])
def test_gpu_absent_or_slot_ambiguous_remains_unknown(raw):
    results = make_probe_results()
    results[_GPU_ARGV] = raw
    item = make_sampler(runner=FakeRunner(results)).sample()
    assert item.gpu_name.value is None
    assert item.gpu_vram_free.status is FieldStatus.UNKNOWN


def test_cpu_guest_and_decreasing_counter_semantics():
    zero = CPU(*([0] * 10))
    # user 50 includes guest 40; nice 10 includes guest_nice 5.
    assert _cpu_percent(zero, CPU(50, 10, 0, 40, 0, 0, 0, 0, 40, 5)) == 60
    assert _cpu_percent(CPU(10, 0, 0, 10, 0, 0, 0, 0, 0, 0),
                        CPU(9, 0, 0, 11, 0, 0, 0, 0, 0, 0)) == 0
    assert _cpu_percent(zero, zero) is None


def test_timeout_stale_original_timestamps_and_no_callback_after_deadline():
    runner = FakeRunner(make_probe_results(), timeout=_GPU_ARGV)
    item = make_sampler(runner=runner).sample()
    assert item.gpu_vram_free.reason is ReasonCode.REQUEST_TIMEOUT
    state, clock = fake_clock()

    class AdvancingRunner(FakeRunner):
        def __call__(self, argv, **kwargs):
            state['time'] += timedelta(seconds=11)
            return super().__call__(argv, **kwargs)

    item = make_sampler(clock=clock, runner=AdvancingRunner(make_probe_results())).sample()
    assert item.host_ram_total.status is FieldStatus.STALE
    assert item.host_ram_total.value is None
    assert item.host_ram_total.sample_time == NOW.isoformat().replace('+00:00', 'Z')

    class SlowWorker(FakeWorker):
        def read_memory_cgroup(self, *, deadline):
            state['mono'] = deadline + 1
            return super().read_memory_cgroup(deadline=deadline)

    runtime = Runtime()
    item = make_sampler(worker=SlowWorker(), runtime=runtime, monotonic=lambda: state['mono']).sample()
    assert runtime.calls == 0
    assert item.worker_ram_limit.value is None
    assert item.worker_ram_limit.reason is ReasonCode.REQUEST_TIMEOUT


def test_cached_cpu_baseline_and_async_sampling_off_event_loop():
    caller, seen = threading.get_ident(), []
    sampler = make_sampler(sleep=lambda _: seen.append(threading.get_ident()))
    assert asyncio.run(sampler.sample_async()).host_cpu_percent.value == 50
    assert asyncio.run(sampler.sample_async()).host_cpu_percent.value == 50
    assert len(seen) == 1 and seen[0] != caller
    sampler._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='already in progress'):
            sampler.sample()
    finally:
        sampler._lock.release()


@pytest.mark.parametrize('value', [True, False, float('inf'), float('nan'), '4', 0, -1])
def test_constructor_bounds_reject_bool_and_nonfinite(value):
    with pytest.raises(ValueError):
        HardwareSampler(timeout=value)
    with pytest.raises(ValueError):
        HardwareSampler(max_age=value)


def test_typed_identity_and_transport_reject_coercions():
    with pytest.raises(FrozenInstanceError):
        REF.context = 'other'
    with pytest.raises(ValueError):
        replace(REF, gpu_visible='false')
    with pytest.raises(ValueError):
        CgroupMemory(REF, 2, True, 0)
    with pytest.raises(ValueError):
        HardwareSampler(worker=object())


def test_hardware_api_contract_registry_negatives():
    domains = json.loads((Path(__file__).parents[1] / 'Docs' / 'contract-domains.json').read_text())
    for invalid in domains['negative_examples']:
        with pytest.raises(ValueError):
            RuntimeComputeEvidence(REF, invalid, ('int8',))
    assert RuntimeComputeEvidence(REF, domains['backend'][0], ('int8',)).backend is Backend.CPU


# ------------------------------------------------------------- host identity and engine facts


def test_windows_host_identity_and_engine_facts_without_sysctl():
    runner = FakeRunner(make_probe_results())
    item = make_sampler(runner=runner).sample()
    assert (item.host_os.value, item.host_os.source) == ('Windows', 'host_platform')
    assert (item.host_arch.value, item.host_arch.source) == ('amd64', 'host_platform')
    assert (item.rosetta_translated.value, item.rosetta_translated.source) == (False, 'host_platform')
    assert item.host_cpu_model.value == 'Intel64 Family 6 Model 183, GenuineIntel'
    assert item.host_cpu_model.source == 'host_platform'
    assert (item.engine_arch.value, item.engine_arch.source) == ('amd64', 'docker_info')
    assert item.engine_mem_total_bytes.value == 16755163136
    assert item.engine_ncpu.value == 8
    assert ENGINE_ARGV in runner.calls
    assert not any(argv[0] == 'sysctl' for argv in runner.calls)


def test_darwin_apple_silicon_native_identity_and_engine_arch():
    runner = FakeRunner(make_mac_results())
    item = make_sampler(runner=runner, system=lambda: 'Darwin', machine=lambda: 'arm64',
                        processor=lambda: 'arm').sample()
    assert item.host_os.value == 'Darwin'
    assert (item.host_arch.value, item.host_arch.source) == ('arm64', 'host_platform')
    assert (item.rosetta_translated.value, item.rosetta_translated.source) == (False, 'host_sysctl')
    assert (item.host_cpu_model.value, item.host_cpu_model.source) == ('Apple M4', 'host_sysctl')
    assert (item.engine_arch.value, item.engine_arch.source) == ('arm64', 'docker_info')
    assert item.engine_mem_total_bytes.value == 8217059328
    assert item.engine_ncpu.value == 10
    assert all(getattr(item, key).status is FieldStatus.OK for key in
               ('host_os', 'host_arch', 'rosetta_translated', 'host_cpu_model',
                'engine_arch', 'engine_mem_total_bytes', 'engine_ncpu'))
    assert runner.calls.index(_SYSCTL_TRANSLATED_ARGV) < runner.calls.index(_SYSCTL_CPU_BRAND_ARGV)
    assert item.gpu_name.status is FieldStatus.UNKNOWN


def test_darwin_rosetta_translated_python_still_reports_arm64_host():
    item = make_sampler(runner=FakeRunner(make_mac_results(translated=b'1\n')),
                        system=lambda: 'Darwin', machine=lambda: 'x86_64').sample()
    assert item.rosetta_translated.value is True
    assert (item.host_arch.value, item.host_arch.source) == ('arm64', 'host_sysctl_proc_translated')
    assert item.engine_arch.value == 'arm64'


@pytest.mark.parametrize('translated', [b'', b'2\n', b'yes\n'])
def test_darwin_sysctl_failure_is_unknown_never_zero(translated):
    results = make_mac_results(translated=translated, brand=b'')
    item = make_sampler(runner=FakeRunner(results), system=lambda: 'Darwin', machine=lambda: 'arm64').sample()
    assert item.rosetta_translated.value is None
    assert item.rosetta_translated.status is FieldStatus.UNKNOWN
    assert item.rosetta_translated.reason is ReasonCode.STORAGE_UNKNOWN
    assert item.host_cpu_model.value is None
    assert item.host_cpu_model.source == 'host_sysctl'
    assert (item.host_arch.value, item.host_arch.source) == ('arm64', 'host_platform')
    assert item.host_os.value == 'Darwin'


def test_intel_mac_unknown_sysctl_key_keeps_machine_architecture():
    item = make_sampler(runner=FakeRunner(make_mac_results(translated=b'', brand=b'Intel(R) Core(TM) i9 @ 2.3GHz\n')),
                        system=lambda: 'Darwin', machine=lambda: 'x86_64').sample()
    assert item.rosetta_translated.status is FieldStatus.UNKNOWN
    assert item.host_arch.value == 'amd64'
    assert item.host_cpu_model.value == 'Intel(R) Core(TM) i9 @ 2.3GHz'


@pytest.mark.parametrize(('engine', 'arch', 'mem', 'ncpu'), [
    (b'', None, None, None),
    (b'aarch64 8217059328\n', None, None, None),
    (b'aarch64 0 10\n', 'arm64', None, 10),
    (b'aarch64 8217059328 0\n', 'arm64', 8217059328, None),
    (b'aarch64 -1 NaN\n', 'arm64', None, None),
    (b'riscv64 8217059328 10\n', 'riscv64', 8217059328, 10),
    (b'x86_64 ' + b'9' * 20 + b' 10\n', 'amd64', None, 10),
])
def test_engine_facts_unknown_when_docker_fails_or_reports_zero(engine, arch, mem, ncpu):
    item = make_sampler(runner=FakeRunner(make_mac_results(engine=engine)), system=lambda: 'Darwin',
                        machine=lambda: 'arm64').sample()
    assert item.engine_arch.value == arch
    assert item.engine_mem_total_bytes.value == mem
    assert item.engine_ncpu.value == ncpu
    for key in ('engine_arch', 'engine_mem_total_bytes', 'engine_ncpu'):
        reading = getattr(item, key)
        assert reading.source == 'docker_info'
        if reading.value is None:
            assert reading.status is FieldStatus.UNKNOWN
            assert reading.reason is (ReasonCode.DOCKER_UNAVAILABLE if engine == b'' else ReasonCode.STORAGE_UNKNOWN)
    assert item.host_cpu_model.value == 'Apple M4'


def test_engine_facts_use_bound_worker_context_and_refuse_environment_overrides():
    runner = FakeRunner(make_probe_results())
    make_sampler(runner=runner, worker=FakeWorker()).sample()
    assert docker('info', '--format', _ENGINE_INFO_FORMAT) in runner.calls
    assert ENGINE_ARGV not in runner.calls

    runner = FakeRunner(make_probe_results())
    item = make_sampler(runner=runner, environ={'DOCKER_CONTEXT': 'remote'}).sample()
    assert not any(argv[0] == 'docker' for argv in runner.calls)
    assert item.engine_arch.value is None
    assert item.engine_arch.reason is ReasonCode.DOCKER_CONTEXT_UNSUPPORTED
    assert item.host_arch.value == 'amd64'


def test_engine_probe_timeout_is_reported_not_swallowed():
    runner = FakeRunner(make_probe_results(), timeout=ENGINE_ARGV)
    item = make_sampler(runner=runner).sample()
    assert item.engine_arch.reason is ReasonCode.REQUEST_TIMEOUT
    assert item.host_cpu_cores.value == 16


def test_hardware_evidence_from_original_fields_defaults_new_fields_unmeasured():
    stamp = NOW.isoformat().replace('+00:00', 'Z')
    ok = Measurement(1, 'fixture', FieldStatus.OK, stamp, None)
    original = [name for name in HardwareEvidence.__dataclass_fields__
                if not name.startswith(('host_os', 'host_arch', 'rosetta', 'host_cpu_model', 'engine_'))]
    assert len(original) == 19
    evidence = HardwareEvidence(**{name: ok for name in original})
    for name in ('host_os', 'host_arch', 'rosetta_translated', 'host_cpu_model',
                 'engine_arch', 'engine_mem_total_bytes', 'engine_ncpu'):
        reading = getattr(evidence, name)
        assert reading.value is None and reading.status is FieldStatus.UNKNOWN
        assert reading.source == 'unmeasured'
    assert len(HardwareEvidence.__dataclass_fields__) == 26


# ------------------------------------------------------------- darwin worker cgroup route


def make_mac_probe_results(transcript=None):
    results = make_mac_results()
    for key in [k for k in results if k[:1] == ('docker',) and k[1:3] == ('--context', REF.context)]:
        results[('docker', '--context', MAC_REF.context, *key[3:])] = results.pop(key)
    context = [{'Name': MAC_REF.context, 'Endpoints': {'docker': {'Host': MAC_REF.endpoint}}}]
    results[('docker', '--context', MAC_REF.context, 'context', 'inspect', MAC_REF.context)] = json.dumps(context).encode()
    results[exec_argv()] = namespaced_transcript() if transcript is None else transcript
    return results


def test_darwin_reads_worker_leaf_through_private_cgroup_namespace_exec():
    runner = FakeRunner(make_mac_probe_results(namespaced_transcript(current_after=GIB + 1024,
                                                                     available_after=11 * GIB)))
    memory = DockerCgroupMemory(MAC_REF, runner=runner, environ={}, platform='darwin',
                                monotonic=lambda: 100, clock=lambda: NOW).read_memory_cgroup(deadline=101)
    assert memory.version == 2
    assert memory.limit == 4 * GIB
    assert memory.current == GIB + 1024
    assert memory.effective_limit == 4 * GIB
    assert memory.available == 3 * GIB - 1024
    assert memory.ancestry.namespaced is True
    assert memory.ancestry.physical_total == 16 * GIB
    assert memory.ancestry.physical_available == 11 * GIB
    assert memory.ancestry.sample_time == NOW
    leaf = memory.ancestry.nodes[-1]
    assert (leaf.path, leaf.device, leaf.inode) == ('/' + MAC_REF.container_id, 23, 4567)
    assert (leaf.current_before, leaf.current_after) == (GIB, GIB + 1024)
    mac = ('docker', '--context', MAC_REF.context)
    assert runner.calls == [(*mac, 'context', 'inspect', MAC_REF.context),
                           (*mac, 'container', 'inspect', MAC_REF.container_id),
                           (*mac, 'top', MAC_REF.container_id, '-eo', 'pid'), exec_argv(),
                           (*mac, 'context', 'inspect', MAC_REF.context),
                           (*mac, 'container', 'inspect', MAC_REF.container_id),
                           (*mac, 'top', MAC_REF.container_id, '-eo', 'pid')]
    assert not any('wsl.exe' in argv or 'nsenter' in argv or 'timeout' in argv for argv in runner.calls)
    assert set(runner.deadlines) == {101}
    assert '0::/' in _NAMESPACED_SCRIPT and 'memory.max' in _NAMESPACED_SCRIPT
    assert 'MemAvailable' in _NAMESPACED_SCRIPT


def test_darwin_unlimited_leaf_is_bounded_by_vm_physical_memory():
    results = make_mac_probe_results(namespaced_transcript(limit=None, total=8 * GIB, available=5 * GIB,
                                                           available_after=2 * GIB))
    memory = DockerCgroupMemory(MAC_REF, runner=FakeRunner(results), environ={}, platform='darwin',
                                clock=lambda: NOW).read_memory_cgroup(deadline=1e20)
    assert memory.limit is None
    assert memory.effective_limit == 8 * GIB
    assert memory.available == 2 * GIB


@pytest.mark.parametrize('transcript', [
    b'',
    b'hierarchy-v1\n',
    namespaced_transcript(membership='0::/docker/' + 'a' * 64),
    namespaced_transcript(membership='0::/'.ljust(5)),
    namespaced_transcript(identity='23:1:41ed'),
    namespaced_transcript(identity='23:4567:81a4'),
    namespaced_transcript(identity_after='23:4568:41ed'),
    namespaced_transcript(identity_after='24:4567:41ed'),
    namespaced_transcript().replace(b'4294967296', b'2147483648', 1),
    namespaced_transcript(limit=0),
    namespaced_transcript(current=-1),
    namespaced_transcript(total=0),
    namespaced_transcript(available=20 * GIB),
    namespaced_transcript().replace(b'physical\t16777216\t12582912\n', b'physical\t8388608\t4194304\n', 1),
    namespaced_transcript() + b'extra\n',
    namespaced_transcript()[:-20],
    b'x' * 32769,
], ids=lambda raw: f'{len(raw)}b')
def test_darwin_malformed_or_changed_namespaced_transcript_is_unknown(transcript):
    memory = DockerCgroupMemory(MAC_REF, runner=FakeRunner(make_mac_probe_results(transcript)), environ={},
                                platform='darwin', clock=lambda: NOW).read_memory_cgroup(deadline=1e20)
    assert memory.ancestry is None
    assert memory.current is None and memory.effective_limit is None and memory.available is None


def test_darwin_exec_route_requires_local_unix_endpoint_and_darwin_platform():
    remote = replace(MAC_REF, endpoint='npipe:////./pipe/docker_engine')
    results = make_mac_probe_results()
    results[('docker', '--context', MAC_REF.context, 'context', 'inspect', MAC_REF.context)] = json.dumps(
        [{'Name': MAC_REF.context, 'Endpoints': {'docker': {'Host': remote.endpoint}}}]).encode()
    runner = FakeRunner(results)
    memory = DockerCgroupMemory(remote, runner=runner, environ={}, platform='darwin',
                                clock=lambda: NOW).read_memory_cgroup(deadline=1e20)
    assert memory.ancestry is None
    assert not any('exec' in argv for argv in runner.calls)

    runner = FakeRunner(make_mac_probe_results())
    memory = DockerCgroupMemory(MAC_REF, runner=runner, environ={}, platform='linux',
                                clock=lambda: NOW).read_memory_cgroup(deadline=1e20)
    assert memory.ancestry is None
    assert not any('exec' in argv for argv in runner.calls)


def test_darwin_namespaced_leaf_provenance_reaches_sampler_fields():
    memory = DockerCgroupMemory(MAC_REF, runner=FakeRunner(make_mac_probe_results()), environ={},
                                platform='darwin', clock=lambda: NOW).read_memory_cgroup(deadline=1e20)

    class Worker(FakeWorker):
        def read_memory_cgroup(self, *, deadline):
            return memory

    runner = FakeRunner(make_mac_probe_results())
    item = make_sampler(runner=runner, worker=Worker(identity=MAC_REF), system=lambda: 'Darwin',
                        machine=lambda: 'arm64').sample()
    assert item.worker_ram_limit.value == 4 * GIB
    assert item.worker_ram_limit.source == 'worker_cgroup_v2_namespaced_leaf'
    assert item.worker_ram_available.value == 3 * GIB
    assert item.worker_ram_available.source == 'worker_cgroup_v2_namespaced_leaf'
    assert item.worker_ram_current.source == 'worker_cgroup_v2'
    assert item.worker_ram_limit.sample_time == NOW.isoformat().replace('+00:00', 'Z')
    assert item.engine_arch.value == 'arm64'
    assert ('docker', '--context', MAC_REF.context, 'info', '--format', _ENGINE_INFO_FORMAT) in runner.calls


def test_namespaced_ancestry_requires_single_node_and_bool_flag():
    node = CgroupAncestor('/worker', 23, 100, GIB, 0, 0)
    child = CgroupAncestor('/worker/child', 23, 101, GIB, 0, 0)
    with pytest.raises(ValueError):
        CgroupAncestry((node, child), 16 * GIB, 12 * GIB, NOW, namespaced=True)
    with pytest.raises(ValueError):
        CgroupAncestry((node,), 16 * GIB, 12 * GIB, NOW, namespaced=1)
    assert CgroupAncestry((node, child), 16 * GIB, 12 * GIB, NOW).namespaced is False


@pytest.mark.live
def test_real_host_sampling():
    import platform

    evidence = HardwareSampler().sample()
    assert evidence.host_cpu_cores.status is FieldStatus.OK
    assert evidence.host_cpu_cores.value > 0
    assert evidence.host_ram_total.status is FieldStatus.OK
    assert 0 < evidence.host_ram_available.value <= evidence.host_ram_total.value
    assert evidence.host_cpu_percent.status is FieldStatus.OK
    assert 0 <= evidence.host_cpu_percent.value <= 100
    assert evidence.gpu_vram_free.status in (FieldStatus.OK, FieldStatus.UNKNOWN)
    assert evidence.host_os.value == platform.system()
    assert evidence.host_arch.value in ('amd64', 'arm64')
    if platform.system() == 'Darwin':
        assert evidence.rosetta_translated.status is FieldStatus.OK
        assert evidence.host_cpu_model.status is FieldStatus.OK
        assert evidence.host_cpu_model.source == 'host_sysctl'
    else:
        assert evidence.rosetta_translated.value is False
    for key in ('engine_arch', 'engine_mem_total_bytes', 'engine_ncpu'):
        reading = getattr(evidence, key)
        assert reading.source == 'docker_info'
        assert reading.status in (FieldStatus.OK, FieldStatus.UNKNOWN)
    if evidence.engine_arch.status is FieldStatus.OK:
        assert evidence.engine_arch.value in ('amd64', 'arm64')
        assert evidence.engine_mem_total_bytes.value > 0 and evidence.engine_ncpu.value > 0
    print('host', evidence.host_os.value, evidence.host_arch.value, evidence.host_cpu_model.value,
          'rosetta', evidence.rosetta_translated.value, 'engine', evidence.engine_arch.value,
          evidence.engine_mem_total_bytes.value, evidence.engine_ncpu.value)
    print('PODMUNCHER_SMOKE_OK')
