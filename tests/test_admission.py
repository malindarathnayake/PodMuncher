"""Peak-growth admission arithmetic with injected, read-only evidence."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from podmuncher.admission import (
    AdmissionError, CONTROL_BOUND, GIB, LOG_FLOOR, WORKER_FLOOR,
    GrowthInterval, StoragePlacement, WorkloadGrowth, make_plan,
    make_workload_plan, may_acquire_lease, peak_remaining, reserve_bytes,
    standard_growth_intervals,
)
from podmuncher.config import Limits
from podmuncher.docker_storage import (
    BackingAssociation, Constraint, DockerStorageEvidence, DockerStorageProvider,
    LinuxLayout, ProcessResult,
)
from podmuncher.domains import DomainValidationError, FieldStatus, JobState, ReasonCode
from podmuncher.snapshots import Measurement
from podmuncher.volumes import FileIdentity, HandleInfo, RawVolume, VolumeEvidence, VolumeProvider


NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
STAMP = NOW.isoformat().replace('+00:00', 'Z')
ZERO_RESERVE = Limits(reserve_fraction=0, reserve_floor_bytes=0,
                      reserve_ceiling_bytes=0)


def m(value, *, status=FieldStatus.OK, reason=None, stamp=STAMP):
    return Measurement(value, 'test_sample', status, stamp, reason)


def fake_volume(identity, capacity, free, *, inodes=100, state=True):
    return VolumeEvidence(m(identity), m(capacity), m(free),
                          m(inodes) if inodes is not None else m(None, status=FieldStatus.NOT_APPLICABLE,
                                                                 reason=ReasonCode.STORAGE_UNKNOWN),
                          m(True), m(True), m(False),
                          m(state) if state is not None else m(None, status=FieldStatus.UNKNOWN,
                                                               reason=ReasonCode.STORAGE_UNKNOWN))


def fake_docker(vm='vm', host='host', *, status=FieldStatus.OK):
    admission = m(True) if status is FieldStatus.OK else m(None, status=status,
                                                           reason=ReasonCode.STORAGE_STALE)
    constraints = ((Constraint(('docker_data',), ('/docker',), fake_volume(vm, 3 * GIB, 2 * GIB, state=None)),)
                   if vm == host else
                   (Constraint(('docker_vm',), ('/docker',), fake_volume(vm, 3 * GIB, 2 * GIB, state=None)),
                    Constraint(('docker_backing',), ('/disk.vhdx',), fake_volume(host, 3 * GIB, 2 * GIB, state=None))))
    backing = () if vm == host else (BackingAssociation(m(vm), m(host), m(True)),)
    return DockerStorageEvidence(m('context'), m('/docker'), m('4.40.0'), constraints,
                                 backing, admission)


def discover_native(stores):
    """Exercise production discovery with local, read-only metadata fakes."""
    assert next(iter(stores)) == '/docker'

    class Volumes:
        def sample(self, path):
            return stores[path]

    class Platform:
        def linux_layout(self, info, call):
            assert info['DockerRootDir'] == '/docker'
            assert info['Driver'] == 'overlay2'
            assert info['_local_endpoint'] == 'unix:///var/run/docker.sock'
            return LinuxLayout(tuple(stores), verified=True)

    def runner(argv, *, deadline, env):
        if argv == ('docker', 'context', 'show'):
            return ProcessResult(0, b'default\n')
        if argv == ('docker', 'context', 'inspect', 'default'):
            result = [{'Name': 'default', 'Endpoints': {
                'docker': {'Host': 'unix:///var/run/docker.sock'}}}]
        else:
            assert argv == ('docker', '--context', 'default', 'info', '--format', '{{json .}}')
            result = {'DockerRootDir': '/docker', 'OSType': 'linux',
                      'Driver': 'overlay2', 'SecurityOptions': []}
        return ProcessResult(0, json.dumps(result).encode())

    evidence = DockerStorageProvider(runner=runner, volumes=Volumes(),
                                    platform_provider=Platform(), platform='linux',
                                    environ={}, clock=lambda: NOW).sample()
    assert evidence.can_pull and evidence.can_create_cached
    return evidence


def growth(**kwargs):
    base = dict(image_compressed=0, image_unpacked=0, model_partial=0,
                model_unpacked=0, source_streams=(0,), merge=0,
                copied_source=0, copied_video=0, publication_staging=0,
                transcript_outputs=(), sidecars=0, state=0)
    base.update(kwargs)
    return WorkloadGrowth(**base)


def test_exact_integer_reserve_and_one_byte_boundary():
    limits = Limits(reserve_fraction=.05, reserve_floor_bytes=0,
                    reserve_ceiling_bytes=10 * GIB)
    assert reserve_bytes(21, limits) == 2
    item = GrowthInterval('source', 'disk', 100, 0, 0, 1)
    okay = make_plan((fake_volume('disk', 1000, 150),), (item,), limits, now=NOW)
    short = make_plan((fake_volume('disk', 1000, 149),), (item,), limits, now=NOW)
    assert okay.admitted and okay.volumes[0].available_after_job.value == 50
    assert short.reason is ReasonCode.STORAGE_INSUFFICIENT


def test_peak_overlapping_lifetimes_and_reusable_asset_deduplication():
    items = (GrowthInterval('a', 'disk', 100, 20, 0, 3),
             GrowthInterval('a', 'disk', 100, 20, 1, 4),
             GrowthInterval('b', 'disk', 50, 0, 2, 3),
             GrowthInterval('c', 'disk', 70, 0, 3, 5))
    assert peak_remaining(items) == 150
    assert make_plan((fake_volume('disk', 1000, 150),), items,
                     ZERO_RESERVE, now=NOW).admitted
    assert make_plan((fake_volume('disk', 1000, 149),), items,
                     ZERO_RESERVE, now=NOW).reason is ReasonCode.STORAGE_INSUFFICIENT


def test_shared_cache_output_and_backing_have_one_physical_budget():
    items = (GrowthInterval('cache', 'same', 40, 0, 0, 3),
             GrowthInterval('output', 'same', 60, 0, 1, 4),
             GrowthInterval('backing_image', 'same', 30, 0, 1, 3))
    aliases = (fake_volume('same', 1000, 130), fake_volume('same', 2000, 150))
    result = make_plan(aliases, items, ZERO_RESERVE, now=NOW)
    assert result.admitted and len(result.volumes) == 1
    assert result.volumes[0].required.value == 130
    assert result.volumes[0].free.value == 130


def test_standard_builder_charges_worker_control_logs_and_vm_host_independently():
    placement = StoragePlacement('scratch', 'cache', 'out', 'state', 'vm', 'host')
    intervals = standard_growth_intervals(growth(image_compressed=100, image_unpacked=200),
                                          placement, ZERO_RESERVE)
    assert peak_remaining(tuple(x for x in intervals if x.volume_identity == 'vm')) >= WORKER_FLOOR + LOG_FLOOR + 200
    assert peak_remaining(tuple(x for x in intervals if x.volume_identity == 'state')) >= CONTROL_BOUND
    assert any(x.asset_key == 'host:image_unpacked' and x.total_bytes == 220 for x in intervals)
    volumes = (fake_volume('scratch', GIB, GIB), fake_volume('cache', GIB, GIB), fake_volume('out', GIB, GIB),
               fake_volume('state', GIB, GIB), fake_volume('vm', GIB, GIB),
               fake_volume('host', GIB, 1))
    result = make_plan(volumes, intervals, ZERO_RESERVE, now=NOW, docker=fake_docker())
    assert result.reason is ReasonCode.STORAGE_INSUFFICIENT
    assert next(x for x in result.volumes if x.volume_identity == 'vm').reason is ReasonCode.OK


def test_host_growth_uses_remaining_bytes_without_unproved_vhd_credit():
    placement = StoragePlacement('scratch', 'cache', 'out', 'state', 'vm', 'host')
    intervals = standard_growth_intervals(growth(image_unpacked=10), placement,
                                          ZERO_RESERVE, allocated={'image_unpacked': 9})
    vm = next(x for x in intervals if x.asset_key == 'image_unpacked')
    host = next(x for x in intervals if x.asset_key == 'host:image_unpacked')
    assert vm.remaining_bytes == 1
    assert host.remaining_bytes == 2


def test_huge_integer_reserve_does_not_round_down():
    limits = Limits(reserve_fraction=.05, reserve_floor_bytes=0,
                    reserve_ceiling_bytes=10 ** 110)
    capacity = 10 ** 80 + 1
    assert reserve_bytes(capacity, limits) == 5 * 10 ** 78 + 1


def test_high_level_cached_worker_and_native_linux_same_volume():
    placement = StoragePlacement('disk', 'disk', 'disk', 'disk', 'disk', 'disk')
    volume = fake_volume('disk', 3 * GIB, 2 * GIB)
    docker = fake_docker('disk', 'disk')
    plan = make_workload_plan((volume,), growth(), placement, ZERO_RESERVE,
                              now=NOW, docker=docker, state_evidence=volume)
    assert plan.admitted
    assert all(not x.asset_key.startswith('host:') for x in plan.intervals)
    assert plan.volumes[0].required.value == CONTROL_BOUND + WORKER_FLOOR + LOG_FLOOR


def test_shared_host_aliases_only_state_path_needs_state_proof():
    placement = StoragePlacement('host', 'host', 'host', 'host', 'vm', 'host')
    class Adapter:
        def split_absolute(self, path):
            return path, ()
        def open_anchor(self, anchor):
            return anchor
        def info(self, handle):
            return HandleInfo(FileIdentity('host', handle), True, False, 1)
        def probe(self, handle):
            return RawVolume(3 * GIB, 2 * GIB, 100, True, True, False, 'adapter')
        def close(self, handle):
            pass
    provider = VolumeProvider(Adapter(), clock=lambda: NOW)
    state = provider.sample('/state', sqlite_state=True)
    ordinary = provider.sample('/cache')
    assert ordinary.state_allowed.status is FieldStatus.UNKNOWN
    vm = fake_volume('vm', GIB, GIB, state=None)
    plan = make_workload_plan((ordinary, vm), growth(), placement, ZERO_RESERVE,
                              now=NOW, docker=fake_docker(), state_evidence=state)
    assert plan.admitted and len(plan.volumes) == 2
    disallowed = replace(state, state_allowed=m(False))
    rejected = make_workload_plan((ordinary, vm), growth(), placement, ZERO_RESERVE,
                                  now=NOW, docker=fake_docker(), state_evidence=disallowed)
    assert rejected.reason is ReasonCode.STORAGE_UNKNOWN
    unknown = replace(state, state_allowed=m(None, status=FieldStatus.UNKNOWN,
                                             reason=ReasonCode.STORAGE_UNKNOWN))
    assert not make_workload_plan((ordinary, vm), growth(), placement, ZERO_RESERVE,
                                  now=NOW, docker=fake_docker(), state_evidence=unknown).admitted
    network = replace(ordinary, network=m(True))
    assert not make_workload_plan((network, vm), growth(), placement, ZERO_RESERVE,
                                  now=NOW, docker=fake_docker(), state_evidence=state).admitted


def test_contradictory_reusable_asset_size_is_rejected():
    items = (GrowthInterval('same', 'disk', 10, 0, 0, 2),
             GrowthInterval('same', 'disk', 11, 0, 1, 3))
    with pytest.raises(AdmissionError):
        peak_remaining(items)


def test_unknown_size_uses_media_cap_known_oversize_rejected_and_partial_remaining():
    limits = Limits(media_bytes=100, reserve_fraction=0,
                    reserve_floor_bytes=0, reserve_ceiling_bytes=0)
    placement = StoragePlacement('same', 'same', 'same', 'same', 'same', 'same')
    unknown = standard_growth_intervals(growth(source_streams=(None,)), placement, limits)
    assert next(x for x in unknown if x.asset_key == 'stream:0').remaining_bytes == 100
    partial = standard_growth_intervals(growth(source_streams=(100,)), placement,
                                        limits, allocated={'stream:0': 70})
    assert next(x for x in partial if x.asset_key == 'stream:0').remaining_bytes == 30
    with pytest.raises(AdmissionError) as error:
        standard_growth_intervals(growth(source_streams=(101,)), placement, limits)
    assert error.value.reason_code is ReasonCode.MEDIA_LIMIT_REACHED


@pytest.mark.parametrize('cached', [False, True])
def test_unknown_or_stale_required_volume_blocks_first_pull_and_cached_creation(cached):
    placement = StoragePlacement('disk', 'disk', 'disk', 'disk', 'disk', 'disk')
    volume = fake_volume('disk', 3 * GIB, 2 * GIB)
    workload = growth(image_compressed=100, image_unpacked=200)
    allocated = {'image_compressed': 100, 'image_unpacked': 200} if cached else {}
    def plan(evidence):
        return make_workload_plan((evidence,), workload, placement, ZERO_RESERVE,
                                  now=NOW, docker=fake_docker('disk', 'disk'),
                                  state_evidence=volume, allocated=allocated)
    healthy = plan(volume)
    assert healthy.admitted
    assert next(x for x in healthy.intervals if x.asset_key == 'image_unpacked').remaining_bytes == (0 if cached else 200)
    assert healthy.volumes[0].required.value == CONTROL_BOUND + WORKER_FLOOR + LOG_FLOOR + (0 if cached else 300)
    stale = replace(volume, free=m(GIB, stamp=(NOW - timedelta(seconds=11)).isoformat()))
    result = plan(stale)
    assert result.reason is ReasonCode.STORAGE_STALE
    unknown = replace(volume, free=m(None, status=FieldStatus.UNKNOWN,
                                     reason=ReasonCode.STORAGE_UNKNOWN))
    result = plan(unknown)
    assert result.reason is ReasonCode.STORAGE_UNKNOWN
    assert not result.admitted


def test_docker_discovery_or_backing_failure_blocks_even_when_budget_passes():
    placement = StoragePlacement('scratch', 'cache', 'out', 'state', 'vm', 'host')
    intervals = standard_growth_intervals(growth(), placement, ZERO_RESERVE)
    volumes = tuple(fake_volume(x, GIB, GIB) for x in ('scratch', 'cache', 'out', 'state', 'vm', 'host'))
    assert make_plan(volumes, intervals, ZERO_RESERVE, now=NOW).reason is ReasonCode.BACKING_UNRESOLVED
    assert make_plan(volumes, intervals, ZERO_RESERVE, now=NOW,
                     docker=fake_docker(status=FieldStatus.STALE)).reason is ReasonCode.STORAGE_STALE
    wrong = fake_docker(host='other')
    assert make_plan(volumes, intervals, ZERO_RESERVE, now=NOW,
                     docker=wrong).reason is ReasonCode.BACKING_UNRESOLVED


def test_queued_and_blocked_never_acquire_lease_and_reaped_contract_is_explicit():
    item = GrowthInterval('worker', 'disk', 1, 0, 0, 1)
    plan = make_plan((fake_volume('disk', 100, 100),), (item,), ZERO_RESERVE, now=NOW)
    assert may_acquire_lease(JobState.CHECKING, plan)
    assert not may_acquire_lease(JobState.QUEUED, plan)
    assert not may_acquire_lease(JobState.BLOCKED, plan)
    from podmuncher.admission import ExecutionLease
    assert 'writers_reaped' in ExecutionLease.release.__annotations__


@pytest.mark.parametrize('value', [True, -1, 1.5, '1'])
def test_strict_byte_validation(value):
    with pytest.raises(AdmissionError):
        GrowthInterval('x', 'disk', value, 0, 0, 1)


def test_linux_inode_unknown_blocks_windows_optional_inode_can_pass():
    volume = fake_volume('disk', 100, 100, inodes=None)
    item = GrowthInterval('worker', 'disk', 1, 0, 0, 1)
    assert make_plan((volume,), (item,), ZERO_RESERVE, now=NOW).reason is ReasonCode.STORAGE_UNKNOWN
    assert make_plan((volume,), (item,), ZERO_RESERVE, now=NOW,
                     inode_optional_identities=frozenset({'disk'})).admitted


def test_invalid_plan_reasons_are_closed_domain_values():
    item = GrowthInterval('worker', 'disk', 100, 0, 0, 1)
    result = make_plan((fake_volume('disk', 100, 1),), (item,), ZERO_RESERVE, now=NOW)
    assert result.reason is ReasonCode.STORAGE_INSUFFICIENT
    assert result.reason.value not in ('__invalid__', '')


def test_vhd_growth_survives_vm_temp_deletion_and_overlaps_later_host_writes():
    placement = StoragePlacement('vm', 'host', 'host', 'host', 'vm', 'host')
    workload = growth(image_compressed=100, image_unpacked=200, source_streams=(100,),
                      model_unpacked=300, publication_staging=400)
    intervals = standard_growth_intervals(workload, placement, ZERO_RESERVE)
    compressed = next(x for x in intervals if x.asset_key == 'image_compressed')
    host_compressed = next(x for x in intervals if x.asset_key == 'host:image_compressed')
    assert compressed.end == 2 and host_compressed.end == 8
    # The VM can reuse 100 compressed bytes for the later 100-byte stream.
    assert peak_remaining(tuple(x for x in intervals if x.volume_identity == 'vm')) == WORKER_FLOOR + LOG_FLOOR + 300
    expected_host = CONTROL_BOUND + 300 + 400 + sum(
        x.remaining_bytes for x in intervals if x.asset_key.startswith('host:'))
    assert peak_remaining(tuple(x for x in intervals if x.volume_identity == 'host')) == expected_host
    state = fake_volume('host', 3 * GIB, expected_host - 1)
    plan = make_workload_plan((state, fake_volume('vm', GIB, GIB)), workload,
                              placement, ZERO_RESERVE, now=NOW,
                              docker=fake_docker(), state_evidence=state)
    assert plan.reason is ReasonCode.STORAGE_INSUFFICIENT


def test_control_state_and_diagnostics_share_one_bounded_pool():
    placement = StoragePlacement('disk', 'disk', 'disk', 'disk', 'disk', 'disk')
    workload = growth(state=100)
    def state_remaining(allocated):
        items = standard_growth_intervals(workload, placement, ZERO_RESERVE, allocated=allocated)
        assert not any(x.asset_key == 'state' for x in items)
        return next(x.remaining_bytes for x in items if x.asset_key == 'control')
    assert state_remaining({}) == CONTROL_BOUND
    assert state_remaining({'state': 100}) == CONTROL_BOUND - 100
    assert state_remaining({'control': 150, 'state': 100}) == CONTROL_BOUND - 150
    with pytest.raises(AdmissionError):
        growth(state=CONTROL_BOUND + 1)
    with pytest.raises(AdmissionError):
        state_remaining({'state': 101})
    with pytest.raises(AdmissionError):
        state_remaining({'control': CONTROL_BOUND + 1})


@pytest.mark.parametrize('role', ['docker_vm', 'docker_backing'])
def test_actual_docker_budgets_cannot_be_bypassed_by_richer_aliases(role):
    placement = StoragePlacement('host', 'host', 'host', 'host', 'vm', 'host')
    docker = fake_docker()
    constraints = tuple(replace(c, volume=replace(c.volume, free=m(1)))
                        if role in c.roles else c for c in docker.constraints)
    docker = replace(docker, constraints=constraints)
    state = fake_volume('host', 3 * GIB, 2 * GIB)
    plan = make_workload_plan((state, fake_volume('vm', 3 * GIB, 2 * GIB)),
                              growth(), placement, ZERO_RESERVE, now=NOW,
                              docker=docker, state_evidence=state)
    assert plan.reason is ReasonCode.STORAGE_INSUFFICIENT
    constrained = next(x for x in plan.volumes if x.volume_identity == ('vm' if role == 'docker_vm' else 'host'))
    assert constrained.free.value == 1


@pytest.mark.parametrize('case', ['empty', 'wrong_vm', 'wrong_host', 'wrong_roles'])
def test_workload_requires_actual_docker_roles_and_matching_placement(case):
    placement = StoragePlacement('host', 'host', 'host', 'host', 'vm', 'host')
    docker = fake_docker()
    if case == 'empty':
        docker = replace(docker, constraints=())
    elif case == 'wrong_vm':
        placement = replace(placement, docker_vm='other')
    elif case == 'wrong_host':
        placement = replace(placement, docker_backing='other')
    else:
        docker = replace(docker, constraints=tuple(replace(c, roles=('docker_data',)) for c in docker.constraints))
    state = fake_volume('host', 3 * GIB, 2 * GIB)
    plan = make_workload_plan((state, fake_volume('vm', GIB, GIB), fake_volume('other', GIB, GIB)),
                              growth(), placement, ZERO_RESERVE, now=NOW,
                              docker=docker, state_evidence=state)
    assert plan.reason is ReasonCode.BACKING_UNRESOLVED


def test_native_separate_containerd_image_store_has_its_own_required_budget():
    placement = StoragePlacement('disk', 'disk', 'disk', 'disk', 'disk', 'disk', 'images')
    state = fake_volume('disk', 3 * GIB, 2 * GIB)
    docker = fake_docker('disk', 'disk')
    required = WORKER_FLOOR + LOG_FLOOR + 300
    images = Constraint(('containerd',), ('/containerd',), fake_volume('images', GIB, required - 1, state=None))
    docker = replace(docker, constraints=(*docker.constraints, images))
    workload = growth(image_compressed=100, image_unpacked=200)
    plan = make_workload_plan((state, fake_volume('images', GIB, GIB)), workload,
                              placement, ZERO_RESERVE, now=NOW, docker=docker, state_evidence=state)
    assert plan.reason is ReasonCode.STORAGE_INSUFFICIENT
    requirement = next(x for x in plan.volumes if x.volume_identity == 'images')
    assert requirement.required.value == required and requirement.free.value == required - 1
    sufficient = replace(docker, constraints=(*docker.constraints[:-1], replace(images, volume=replace(images.volume, free=m(required)))))
    assert make_workload_plan((state,), workload, placement, ZERO_RESERVE, now=NOW,
                              docker=sufficient, state_evidence=state).admitted


@pytest.mark.parametrize('image_store', [None, 'docker', 'images'])
@pytest.mark.parametrize('deficient', [None, 'docker', 'images'])
def test_discovered_classic_docker_and_containerd_each_need_complete_growth(image_store, deficient):
    required = WORKER_FLOOR + LOG_FLOOR + 300
    docker = discover_native({path: fake_volume(identity, GIB, required - (identity == deficient), state=None)
                              for path, identity in (('/docker', 'docker'), ('/containerd', 'images'))})
    state = fake_volume('app', GIB, GIB)
    placement = StoragePlacement('app', 'app', 'app', 'app', 'docker', 'docker', image_store)
    plan = make_workload_plan((state,), growth(image_compressed=100, image_unpacked=200),
                              placement, ZERO_RESERVE, now=NOW, docker=docker, state_evidence=state)
    assert plan.admitted is (deficient is None)
    for identity in ('docker', 'images'):
        requirement = next(x for x in plan.volumes if x.volume_identity == identity)
        assert requirement.required.value == required
        assert requirement.reason is (ReasonCode.STORAGE_INSUFFICIENT if identity == deficient else ReasonCode.OK)


def test_discovered_cached_worker_requires_containerd_floors_plus_reserve():
    limits = replace(ZERO_RESERVE, reserve_floor_bytes=17, reserve_ceiling_bytes=17)
    required = WORKER_FLOOR + LOG_FLOOR
    docker = discover_native({'/docker': fake_volume('docker', GIB, GIB, state=None),
                              '/containerd': fake_volume('runtime', GIB, required + 16, state=None)})
    state = fake_volume('app', GIB, GIB)
    placement = StoragePlacement('app', 'app', 'app', 'app', 'docker', 'docker')
    plan = make_workload_plan((state,), growth(worker=0, container_metadata_and_logs=0),
                              placement, limits, now=NOW, docker=docker, state_evidence=state)
    assert plan.reason is ReasonCode.STORAGE_INSUFFICIENT
    runtime = next(x for x in plan.volumes if x.volume_identity == 'runtime')
    assert runtime.required.value == required and runtime.reserve.value == 17
    assert runtime.available_after_job.value == 16


@pytest.mark.parametrize('deficient', [None, 'docker', 'content', 'snapshot', 'runtime', 'runtime_state'])
def test_discovered_independent_native_stores_each_enforce_exact_boundary(deficient):
    limits = replace(ZERO_RESERVE, reserve_floor_bytes=17, reserve_ceiling_bytes=17)
    workload = growth(image_compressed=101, image_unpacked=203,
                      worker=WORKER_FLOOR + 7, container_metadata_and_logs=LOG_FLOOR + 11,
                      model_partial=37, model_unpacked=41, source_streams=(13,), merge=17,
                      copied_source=19, copied_video=23, publication_staging=29,
                      transcript_outputs=(31,), sidecars=43, state=47)
    required = WORKER_FLOOR + LOG_FLOOR + 322
    identities = ('docker', 'content', 'snapshot', 'runtime', 'runtime_state')
    docker = discover_native({'/' + identity: fake_volume(identity, GIB, required + 17 - (identity == deficient), state=None)
                              for identity in identities})
    state = fake_volume('app', GIB, GIB)
    placement = StoragePlacement('app', 'app', 'app', 'app', 'docker', 'docker')
    plan = make_workload_plan((state,), workload, placement, limits, now=NOW,
                              docker=docker, state_evidence=state)
    assert plan.admitted is (deficient is None)
    for identity in identities:
        requirement = next(x for x in plan.volumes if x.volume_identity == identity)
        assert requirement.required.value == required and requirement.reserve.value == 17
        assert requirement.reason is (ReasonCode.STORAGE_INSUFFICIENT if identity == deficient else ReasonCode.OK)
        lifetimes = {x.asset_key: (x.start, x.end) for x in plan.intervals if x.volume_identity == identity}
        assert lifetimes == {'image_compressed': (0, 2), 'image_unpacked': (1, 8),
                             'worker': (0, 8), 'container_metadata_and_logs': (0, 8)}
    standard = standard_growth_intervals(workload, placement, limits)
    assert tuple(x for x in plan.intervals if x.volume_identity == 'app') == tuple(
        x for x in standard if x.volume_identity == 'app')


@pytest.mark.parametrize('short', [False, True])
def test_discovered_native_aliases_deduplicate_assets_and_keep_tightest_capacity(short):
    required = WORKER_FLOOR + LOG_FLOOR + 300
    docker = discover_native({'/docker': fake_volume('disk', 3 * GIB, 2 * GIB, state=None),
                              '/content': fake_volume('disk', GIB, GIB, state=None),
                              '/snapshot': fake_volume('disk', 2 * GIB, required - short, state=None)})
    assert len(docker.constraints) == 1
    assert set(docker.constraints[0].roles) == {'docker_data', 'containerd'}
    state = fake_volume('app', GIB, GIB)
    placement = StoragePlacement('app', 'app', 'app', 'app', 'disk', 'disk', 'disk')
    richer_alias = fake_volume('disk', 4 * GIB, 3 * GIB, state=None)
    plan = make_workload_plan((state, richer_alias), growth(image_compressed=100, image_unpacked=200),
                              placement, ZERO_RESERVE, now=NOW, docker=docker, state_evidence=state)
    assert plan.admitted is (not short)
    disk = next(x for x in plan.volumes if x.volume_identity == 'disk')
    assert disk.required.value == required and disk.free.value == required - short
    assert disk.capacity.value == GIB
    assert len({(x.volume_identity, x.asset_key) for x in plan.intervals}) == len(plan.intervals)


@pytest.mark.parametrize('bad_join', ['image_store', 'docker_vm'])
def test_discovered_native_layout_rejects_unrelated_image_store_and_wrong_data_identity(bad_join):
    docker = discover_native({'/docker': fake_volume('docker', GIB, GIB, state=None),
                              '/containerd': fake_volume('images', GIB, GIB, state=None)})
    state = fake_volume('app', GIB, GIB)
    placement = StoragePlacement('app', 'app', 'app', 'app', 'docker', 'docker')
    placement = (replace(placement, image_store='app') if bad_join == 'image_store'
                 else replace(placement, docker_vm='images', docker_backing='images'))
    plan = make_workload_plan((state,), growth(), placement, ZERO_RESERVE,
                              now=NOW, docker=docker, state_evidence=state)
    assert plan.reason is ReasonCode.BACKING_UNRESOLVED


@pytest.mark.parametrize('image_store', [None, 'images'])
@pytest.mark.parametrize('short', [False, True])
def test_discovered_native_flat_credits_apply_only_on_each_original_primary_volume(image_store, short):
    required = WORKER_FLOOR + LOG_FLOOR + 300
    docker = discover_native({'/docker': fake_volume('docker', GIB, GIB, state=None),
                              '/content': fake_volume('images', GIB, GIB, state=None),
                              '/snapshot': fake_volume('snapshots', GIB, required - short, state=None)})
    state = fake_volume('app', GIB, GIB)
    placement = StoragePlacement('app', 'app', 'app', 'app', 'docker', 'docker', image_store)
    credits = {'image_compressed': 100, 'image_unpacked': 200,
               'worker': WORKER_FLOOR, 'container_metadata_and_logs': LOG_FLOOR}
    plan = make_workload_plan((state,), growth(image_compressed=100, image_unpacked=200),
                              placement, ZERO_RESERVE, now=NOW, docker=docker,
                              state_evidence=state, allocated=credits)
    assert plan.admitted is (not short)
    for key, amount in credits.items():
        primary = (image_store or 'docker') if key.startswith('image_') else 'docker'
        copies = tuple(x for x in plan.intervals if x.asset_key == key)
        assert len(copies) == 3
        assert sum(x.allocated_bytes for x in copies) == amount
        assert all(x.allocated_bytes == (amount if x.volume_identity == primary else 0) for x in copies)
    expected = {'docker': 300 if image_store else 0,
                'images': WORKER_FLOOR + LOG_FLOOR if image_store else required,
                'snapshots': required}
    for identity, amount in expected.items():
        requirement = next(x for x in plan.volumes if x.volume_identity == identity)
        assert requirement.required.value == amount
        assert requirement.reason is (ReasonCode.STORAGE_INSUFFICIENT
                                       if identity == 'snapshots' and short else ReasonCode.OK)


def test_huge_integer_fraction_is_exact_without_float_or_string_conversion():
    limits = Limits(reserve_fraction=10 ** 10000, reserve_floor_bytes=0,
                    reserve_ceiling_bytes=10 ** 110)
    assert reserve_bytes(1, limits) == limits.reserve_ceiling_bytes
    assert reserve_bytes(0, limits) == 0


@pytest.mark.parametrize('fraction', [True, -1, float('nan'), float('inf'), '0.05'])
def test_invalid_reserve_fractions_have_safe_closed_errors(fraction):
    with pytest.raises(AdmissionError) as error:
        reserve_bytes(100, replace(ZERO_RESERVE, reserve_fraction=fraction))
    assert error.value.reason_code is ReasonCode.INVALID_CONFIG
    assert str(error.value) == 'Invalid input.'


def test_admission_accepted_domains_reject_shared_contract_negative_examples():
    contract = json.loads((Path(__file__).parents[1] / 'Docs' / 'contract-domains.json').read_text())
    item = GrowthInterval('worker', 'disk', 1, 0, 0, 1)
    plan = make_plan((fake_volume('disk', 100, 100),), (item,), ZERO_RESERVE, now=NOW)
    for domain, enum in (('job_state', JobState), ('reason_code', ReasonCode), ('field_status', FieldStatus)):
        assert [x.value for x in enum] == contract[domain]
    for invalid in contract['negative_examples']:
        with pytest.raises(AdmissionError):
            may_acquire_lease(invalid, plan)
        with pytest.raises(DomainValidationError):
            AdmissionError(invalid)
        with pytest.raises(DomainValidationError):
            m(None, status=invalid, reason=ReasonCode.STORAGE_UNKNOWN)
        with pytest.raises(DomainValidationError):
            m(None, status=FieldStatus.UNKNOWN, reason=invalid)


def test_plan_preserves_sample_provenance_and_stale_derived_values():
    old_stamp = (NOW - timedelta(seconds=11)).isoformat()
    volume = replace(fake_volume('disk', 100, 90),
                     capacity=Measurement(100, 'capacity_probe', FieldStatus.OK, old_stamp, None),
                     free=Measurement(90, 'quota_probe', FieldStatus.OK, old_stamp, None))
    plan = make_plan((volume,), (GrowthInterval('work', 'disk', 10, 0, 0, 1),),
                     ZERO_RESERVE, now=NOW)
    assert plan.reason is ReasonCode.STORAGE_STALE
    requirement = plan.volumes[0]
    assert requirement.capacity is volume.capacity and requirement.free is volume.free
    for derived in (requirement.reserve, requirement.available_after_job):
        assert derived.status is not FieldStatus.OK
        assert derived.value is None and derived.reason is ReasonCode.STORAGE_STALE


def test_first_pull_with_compressed_vm_temp_and_retained_host_growth_is_admitted():
    state = fake_volume('host', 3 * GIB, 2 * GIB)
    placement = StoragePlacement('host', 'host', 'host', 'host', 'vm', 'host')
    plan = make_workload_plan((state,), growth(image_compressed=100, image_unpacked=200),
                              placement, ZERO_RESERVE, now=NOW, docker=fake_docker(),
                              state_evidence=state)
    assert plan.admitted
    assert next(x for x in plan.intervals if x.asset_key == 'host:image_compressed').end == 8
