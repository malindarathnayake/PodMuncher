"""Unit contracts plus explicitly marked native, owned-file transport smoke."""
import ctypes
import errno
import json
import os
import stat
import sys
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from podmuncher.admission import GrowthInterval, make_plan
from podmuncher.config import Limits
from podmuncher.domains import FieldStatus, ReasonCode
from podmuncher.volumes import (
    DarwinAdapter,
    FileIdentity,
    HandleInfo,
    PosixAdapter,
    RawVolume,
    StorageError,
    WindowsAdapter,
    _DarwinStatfsAPI,
    _LinuxQuotaAPI,
    make_volume_provider,
    native_adapter,
    nearest_existing_ancestor,
    open_contained,
    relative_components,
)

NOW = datetime(2026, 10, 9, tzinfo=UTC)


class FakeAdapter:
    """Deterministic filesystem adapter: each open gets an independently owned ID."""
    def __init__(self):
        self.nodes = {'/': self.node('root'), '/owned': self.node('owned'),
                      '/owned/file': self.node('file', directory=False),
                      '/alias': self.node('alias')}
        self.handles, self.closed = {}, []
        self.serial = 0
        self.raw = RawVolume(1000, 700, 25, True, True, False, 'injected_adapter')
        self.fail_probe = False
        self.mutate = None

    def node(self, file, *, directory=True, volume='actual-volume'):
        return HandleInfo(FileIdentity(volume, file), directory, not directory, 1)

    def split_absolute(self, path):
        return '/', relative_components(path[1:]) if path != '/' else ()

    def _open(self, path):
        if path not in self.nodes:
            raise FileNotFoundError()
        self.serial += 1
        self.handles[self.serial] = (path, self.nodes[path])
        return self.serial

    def open_anchor(self, anchor):
        return self._open(anchor)

    def open_child(self, parent, name, directory):
        path = self.handles[parent][0].rstrip('/') + '/' + name
        if self.mutate:
            callback, self.mutate = self.mutate, None
            callback(self)
        return self._open(path)

    def info(self, handle):
        return self.handles[handle][1]

    def edge_matches(self, parent, name, child):
        path = self.handles[parent][0].rstrip('/') + '/' + name
        return self.nodes.get(path) == self.info(child)

    def close(self, handle):
        assert handle not in self.closed
        self.closed.append(handle)
        del self.handles[handle]

    def read(self, handle, limit):
        return b'transport'[:limit]

    def probe(self, handle):
        if self.fail_probe:
            raise OSError('sensitive path and credential must not escape')
        return self.raw


@pytest.fixture
def make_provider():
    def factory(**kwargs):
        adapter = kwargs.pop('adapter', FakeAdapter())
        return make_volume_provider(adapter=adapter, clock=lambda: NOW, **kwargs), adapter
    return factory


def test_same_volume_aliases_retained_with_immutable_measurements(make_provider):
    provider, adapter = make_provider()
    result = provider.sample_many(('/owned', '/alias'))
    assert len(result) == 2
    assert result[0] is not result[1]
    for sample in result:
        assert sample.identity.value == 'actual-volume'
        assert sample.capacity.value == 1000
        assert sample.free.value == 700
        assert sample.free.source == 'injected_adapter'
        assert sample.free.sample_time == '2026-10-09T00:00:00Z'
        assert sample.free.reason is None
        with pytest.raises(FrozenInstanceError):
            sample.free.value = 2
    assert not adapter.handles


@pytest.mark.parametrize('paths', [('/owned', '/alias'), ('/alias', '/owned')])
def test_alias_path_tighter_budget_reaches_admission_once(make_provider, paths):
    provider, adapter = make_provider()
    ordinary = adapter.raw
    tighter = replace(ordinary, capacity=900, free=80, free_inodes=4)
    adapter.probe = lambda handle: (tighter if adapter.handles[handle][0] == '/alias'
                                    else ordinary)

    samples = provider.sample_many(paths)
    assert len(samples) == 2
    assert {sample.identity.value for sample in samples} == {'actual-volume'}
    assert {sample.free.value for sample in samples} == {700, 80}
    assert {sample.capacity.value for sample in samples} == {1000, 900}
    assert {sample.free_inodes.value for sample in samples} == {25, 4}
    assert not adapter.handles

    # Existing allocated bytes are subtracted once from this one asset's size.
    interval = GrowthInterval('cache', 'actual-volume', 120, 20, 0, 1)
    limits = Limits(reserve_fraction=0, reserve_floor_bytes=0, reserve_ceiling_bytes=0)
    plan = make_plan(samples, (interval,), limits, now=NOW)
    assert plan.reason is ReasonCode.STORAGE_INSUFFICIENT
    assert len(plan.volumes) == 1
    budget = plan.volumes[0]
    assert budget.capacity.value == 900
    assert budget.free.value == 80
    assert budget.required.value == 100


@pytest.mark.parametrize('paths', [('/owned', '/alias'), ('/alias', '/owned')])
@pytest.mark.parametrize('patch', [
    {'writable': False}, {'writable': None}, {'free': None}, {'free_inodes': None},
])
def test_alias_path_blocking_evidence_reaches_admission(make_provider, paths, patch):
    provider, adapter = make_provider()
    ordinary = adapter.raw
    values = {'capacity': 900, 'free': 80, 'free_inodes': 4}
    values.update(patch)
    restrictive = replace(ordinary, **values)
    adapter.probe = lambda handle: (restrictive if adapter.handles[handle][0] == '/alias'
                                    else ordinary)

    samples = provider.sample_many(paths)
    assert len(samples) == 2
    assert {sample.identity.value for sample in samples} == {'actual-volume'}
    alias = samples[paths.index('/alias')]
    for field, value in values.items():
        assert getattr(alias, field).value == value
    limits = Limits(reserve_fraction=0, reserve_floor_bytes=0, reserve_ceiling_bytes=0)
    interval = GrowthInterval('cache', 'actual-volume', 10, 0, 0, 1)
    plan = make_plan(samples, (interval,), limits, now=NOW)
    assert plan.reason is ReasonCode.STORAGE_UNKNOWN
    assert len(plan.volumes) == 1
    assert plan.volumes[0].capacity.value == 900
    assert plan.volumes[0].free.value == values['free']
    assert not adapter.handles


def test_nearest_ancestor_does_not_create(make_provider, tmp_path):
    provider, adapter = make_provider()
    assert provider.sample('/owned/missing/deep').free.value == 700
    assert '/owned/missing' not in adapter.nodes
    target = tmp_path / 'uncreated' / 'deeper'
    with nearest_existing_ancestor(target) as held:
        assert held.identity.volume
    assert not target.parent.exists()


def test_existing_regular_file_uses_containing_volume(make_provider, tmp_path):
    provider, adapter = make_provider()
    adapter.nodes['/owned'] = adapter.node('owned', volume='containing-volume')
    adapter.nodes['/owned/file'] = adapter.node('file', directory=False, volume='leaf-volume')
    probed = []
    original_probe = adapter.probe

    def probe(handle):
        probed.append(adapter.handles[handle][0])
        return original_probe(handle)

    adapter.probe = probe
    sample = provider.sample('/owned/file')
    assert sample.identity.value == 'containing-volume'
    assert sample.free.value == 700
    assert probed == ['/owned']
    assert not adapter.handles

    target = tmp_path / 'bounded.txt'
    target.write_bytes(b'bounded')
    native_provider = make_volume_provider(adapter=native_adapter(), clock=lambda: NOW)
    file_sample = native_provider.sample(target)
    parent_sample = native_provider.sample(tmp_path)
    assert file_sample.identity.status is FieldStatus.OK
    assert parent_sample.identity.status is FieldStatus.OK
    assert file_sample.identity.value == parent_sample.identity.value


def test_missing_mount_and_changed_identity_are_stale(make_provider):
    provider, adapter = make_provider()
    previous = provider.sample('/owned').identity.value
    adapter.fail_probe = True
    result = provider.sample('/owned', previous_identity=previous)
    assert result.free.value is None
    assert result.free.status is FieldStatus.STALE
    assert result.free.reason is ReasonCode.STORAGE_STALE
    adapter.fail_probe = False
    adapter.nodes['/owned'] = adapter.node('changed', volume='new-volume')
    assert provider.sample('/owned', previous_identity=previous).identity.value is None
    assert not adapter.handles


@pytest.mark.parametrize('field,value', [('capacity', None), ('free', None), ('free_inodes', None),
                                        ('mounted', None), ('writable', None), ('network', None)])
def test_unknown_is_null_never_zero(make_provider, field, value):
    provider, adapter = make_provider()
    adapter.raw = replace(adapter.raw, **{field: value})
    measurement = getattr(provider.sample('/owned'), field)
    assert measurement.value is None
    assert measurement.status is FieldStatus.UNKNOWN
    assert measurement.reason is ReasonCode.STORAGE_UNKNOWN


class Telemetry:
    def __init__(self):
        self.events = []

    @contextmanager
    def span(self, name, **labels):
        self.events.append((name, labels))
        yield

    def result(self, **labels):
        self.events.append(('result', labels))


@pytest.mark.parametrize('patch,reason', [({'writable': False}, 'permission'),
                                         ({'free': 0}, 'disk'), ({'free_inodes': 0}, 'disk'),
                                         ({'mounted': False}, 'disk'), ({'free': None}, 'unknown')])
def test_telemetry_closed_labels_permission_disk_unknown(make_provider, patch, reason):
    telemetry = Telemetry()
    provider, adapter = make_provider(telemetry=telemetry)
    adapter.raw = replace(adapter.raw, **patch)
    provider.sample('/owned')
    assert telemetry.events == [('preflight', {'component': 'storage'}),
                                 ('result', {'result': 'blocked', 'reason': reason, 'component': 'storage'})]
    assert 'actual-volume' not in str(telemetry.events)


def test_pass_and_redacted_failure_telemetry(make_provider):
    telemetry = Telemetry()
    provider, adapter = make_provider(telemetry=telemetry)
    provider.sample('/owned')
    assert telemetry.events[-1][1]['result'] == 'pass'
    adapter.fail_probe = True
    assert provider.sample('/owned').capacity.value is None
    assert 'sensitive' not in str(telemetry.events)


@pytest.mark.parametrize('network,shared,expected', [(True, False, False), (False, True, False),
                                                   (False, False, True), (None, False, None)])
def test_network_shared_sqlite_state_rejected(make_provider, network, shared, expected):
    provider, adapter = make_provider()
    adapter.raw = replace(adapter.raw, network=network)
    assert provider.sample('/owned', sqlite_state=True, shared=shared).state_allowed.value is expected


@pytest.mark.parametrize('path', ['', '..', '../secret', '/absolute', 'C:\\absolute', 'C:relative',
                                 '\\\\server\\share', 'file:stream', 'a/../b', 'a//b', 'a\\..\\b',
                                 '.', 'a/', 'NUL', 'COM1.txt', 'trailing.', 'x\x00y'])
def test_traversal_and_ambiguous_names_rejected(path):
    with pytest.raises(StorageError) as caught:
        relative_components(path)
    assert caught.value.reason_code is ReasonCode.ARTIFACT_REJECTED
    assert str(caught.value) == 'Invalid input.'


def test_owned_handles_identity_read_limit_and_closure():
    adapter = FakeAdapter()
    expected = adapter.nodes['/owned/file'].identity
    with open_contained('/owned', 'file', adapter=adapter, expected=expected) as held:
        assert len(adapter.handles) == 3
        assert held.identity == expected
        assert held.read(3) == b'tra'
        with pytest.raises(StorageError):
            held.read(1048577)
    assert held.closed and not adapter.handles
    assert adapter.closed == [3, 2, 1]
    with pytest.raises(StorageError):
        held.read(1)


@pytest.mark.parametrize('change', ['link', 'hardlink', 'special', 'volume', 'identity'])
def test_unsafe_final_identity_and_all_error_handles_closed(change):
    adapter = FakeAdapter()
    info = adapter.nodes['/owned/file']
    patch = {'link': {'reparse': True}, 'hardlink': {'links': 2}, 'special': {'regular': False},
             'volume': {'identity': FileIdentity('other-volume', 'file')}, 'identity': {}}[change]
    adapter.nodes['/owned/file'] = replace(info, **patch)
    expected = FileIdentity('actual-volume', 'different') if change == 'identity' else None
    with pytest.raises(StorageError), open_contained('/owned', 'file', expected=expected, adapter=adapter):
        pytest.fail('unsafe file admitted')
    assert not adapter.handles


def test_reparse_ancestor_and_swapped_ancestor_rejected():
    adapter = FakeAdapter()
    adapter.nodes['/owned'] = replace(adapter.nodes['/owned'], reparse=True)
    with pytest.raises(StorageError), open_contained('/owned', 'file', adapter=adapter):
        pytest.fail('reparse admitted')
    assert not adapter.handles
    adapter = FakeAdapter()
    with open_contained('/owned', 'file', adapter=adapter) as held:
        adapter.nodes['/owned'] = adapter.node('replacement')
        with pytest.raises(StorageError):
            held.read(4)
    assert not adapter.handles


def test_swap_during_walk_rejected_and_closes():
    adapter = FakeAdapter()
    def replace_owned(a):
        # Trigger while opening the final component, after the ancestor is held.
        a.nodes['/owned'] = a.node('replacement')
    original_open = adapter.open_child
    def open_child(parent, name, directory):
        if name == 'file':
            replace_owned(adapter)
        return original_open(parent, name, directory)
    adapter.open_child = open_child
    with pytest.raises(StorageError), open_contained('/owned', 'file', adapter=adapter):
        pytest.fail('swapped ancestor admitted')
    assert not adapter.handles


class PosixOS:
    O_RDONLY, O_DIRECTORY, O_NOFOLLOW, O_CLOEXEC, O_NONBLOCK, ST_RDONLY = 0, 1, 2, 4, 8, 1
    def __init__(self):
        self.opens = []
        self.closed = []
    def open(self, name, flags, **kwargs):
        self.opens.append((name, flags, kwargs))
        return len(self.opens)
    def fstat(self, fd):
        return SimpleNamespace(st_dev=42, st_ino=fd, st_mode=stat.S_IFDIR, st_nlink=1,
                               st_size=0, st_mtime_ns=0, st_ctime_ns=0)
    def fstatvfs(self, fd):
        return SimpleNamespace(f_blocks=1000, f_frsize=4096, f_bavail=500, f_favail=73, f_flag=0)
    def major(self, dev): return 8
    def minor(self, dev): return 1
    def close(self, fd): self.closed.append(fd)


def test_injected_posix_flags_mount_quota_inodes():
    api = PosixOS()
    mount = '1 0 8:1 / / rw - ext4 /dev/sda1 rw,usrquota\n'
    adapter = PosixAdapter(os_api=api, mount_reader=lambda: mount)
    fd = adapter.open_anchor('/')
    adapter.open_child(fd, 'owned', True)
    assert api.opens[1] == ('owned', 15, {'dir_fd': 1})
    raw = adapter.probe(fd)
    assert raw.capacity is None and raw.free is None  # bavail is not quota proof.
    assert raw.free_inodes is None and raw.network is False
    calls = []
    def quota(*args):
        calls.append(args)
        return (100000, 50000, 20)
    adapter.quota = quota
    raw = adapter.probe(fd)
    assert (raw.capacity, raw.free, raw.free_inodes) == (100000, 50000, 20)
    assert 'usrquota' in calls[0][2]
    adapter.mount_reader = lambda: ''
    assert adapter.probe(fd).mounted is False


def test_linux_readonly_network_and_unknown_mount():
    adapter = PosixAdapter(os_api=PosixOS(), mount_reader=lambda: '1 0 8:1 / / ro - nfs server:/ rw\n')
    raw = adapter.probe(1)
    assert raw.writable is False and raw.network is True
    adapter.mount_reader = lambda: '1 0 8:1 / / rw - mystery /dev/x rw\n'
    assert adapter.probe(1).network is None


@pytest.mark.parametrize('allowed', [True, False])
def test_posix_effective_identity_permission_probe_is_handle_anchored(allowed):
    api = PosixOS()
    api.W_OK, api.X_OK = 2, 1
    calls = []
    def access(*args, **kwargs):
        calls.append((args, kwargs))
        return allowed
    api.access = access
    adapter = PosixAdapter(os_api=api, mount_reader=lambda: '1 0 8:1 / / rw - ext4 /dev/x rw\n')
    assert adapter.probe(123).writable is allowed
    assert calls == [(('.', 3), {'dir_fd': 123, 'effective_ids': True, 'follow_symlinks': False})]


class BlockPosixOS(PosixOS):
    O_PATH = 16

    def fstat(self, fd):
        if fd == 1:  # Quota device opened by the adapter; target is fd 123.
            return SimpleNamespace(st_mode=stat.S_IFBLK, st_rdev=42)
        return super().fstat(fd)


class QuotaAPI:
    def __init__(self, states=(False, False, False)):
        self.states, self.calls = states, []

    def enabled(self, fd, kind):
        self.calls.append((fd, kind))
        state = self.states[kind]
        if isinstance(state, Exception):
            raise state
        return state


def linux_quota_adapter(states=(False, False, False), *, filesystem='ext4', options='rw'):
    os_api, quota_api = BlockPosixOS(), QuotaAPI(states)
    adapter = PosixAdapter(os_api=os_api, quota_api=quota_api,
                           mount_reader=lambda: f'1 0 8:1 / / rw - {filesystem} /dev/sda1 {options}\n')
    return adapter, os_api, quota_api


def test_linux_native_default_accepts_statvfs_only_after_three_disabled_quotas(monkeypatch):
    adapter, api, quota = linux_quota_adapter()
    # Exercise default native resolver construction as well as the injectable API.
    monkeypatch.setattr('podmuncher.volumes._LinuxQuotaAPI', lambda: quota)
    adapter.quota_api = None
    raw = adapter.probe(123)
    assert (raw.capacity, raw.free, raw.free_inodes) == (4096000, 2048000, 73)
    assert quota.calls == [(1, 0), (1, 1), (1, 2)]
    assert api.opens == [('/dev/sda1', api.O_PATH | api.O_CLOEXEC, {})]
    assert api.closed == [1]
    assert raw.source == 'linux_mountinfo_statvfs_quotactl'


@pytest.mark.parametrize('kind', [0, 1, 2])
def test_linux_each_enabled_quota_stays_unknown_without_applicable_limit_proof(kind):
    states = [False, False, False]
    states[kind] = True
    adapter, api, quota = linux_quota_adapter(states)
    raw = adapter.probe(123)
    # No quota mount option is needed for an enabled user/group/project quota.
    assert (raw.capacity, raw.free, raw.free_inodes) == (None, None, None)
    assert quota.calls == [(1, 0), (1, 1), (1, 2)]
    assert api.closed == [1]


@pytest.mark.parametrize('code', [errno.ENOSYS, errno.EINVAL, errno.EPERM, errno.EACCES,
                                 errno.EOPNOTSUPP, errno.EIO])
def test_linux_unresolved_quota_query_keeps_unknown_metadata_and_closes(code, make_provider):
    adapter, api, _ = linux_quota_adapter((False, OSError(code, 'private device path'), False))
    raw = adapter.probe(123)
    assert (raw.capacity, raw.free, raw.free_inodes) == (None, None, None)
    assert raw.mounted is True and raw.network is False
    assert api.closed == [1]
    telemetry = Telemetry()
    provider, fake = make_provider(telemetry=telemetry)
    fake.raw = raw
    sample = provider.sample('/owned')
    for field in (sample.capacity, sample.free, sample.free_inodes):
        assert field.value is None and field.status is FieldStatus.UNKNOWN
        assert field.reason is ReasonCode.STORAGE_UNKNOWN
        assert field.source == 'linux_mountinfo_statvfs_quotactl'
        assert field.sample_time == '2026-10-09T00:00:00Z'
        with pytest.raises(FrozenInstanceError):
            field.value = 0
    assert 'private' not in str(telemetry.events)


@pytest.mark.parametrize('filesystem', ['tmpfs', 'xfs', 'btrfs', 'overlay', 'nfs', 'mystery'])
def test_linux_unimplemented_filesystem_quota_semantics_stay_unknown(filesystem):
    adapter, api, quota = linux_quota_adapter(filesystem=filesystem)
    raw = adapter.probe(123)
    assert raw.capacity is None and raw.free is None
    assert not api.opens and not quota.calls


@pytest.mark.parametrize('failure', ['absent', 'denied', 'wrong_device', 'regular_file'])
def test_linux_quota_device_must_be_accessible_and_match_held_filesystem(failure):
    adapter, api, quota = linux_quota_adapter()
    if failure in ('absent', 'denied'):
        def fail_open(*args):
            raise OSError(errno.ENOENT if failure == 'absent' else errno.EACCES, 'private')
        api.open = fail_open
    else:
        original = api.fstat
        def fstat(fd):
            if fd == 1:
                return SimpleNamespace(st_mode=stat.S_IFREG if failure == 'regular_file' else stat.S_IFBLK,
                                       st_rdev=43 if failure == 'wrong_device' else 42)
            return original(fd)
        api.fstat = fstat
    raw = adapter.probe(123)
    assert raw.capacity is None and raw.free is None
    assert not quota.calls
    assert api.closed == ([] if failure in ('absent', 'denied') else [1])


def test_linux_mount_source_is_unescaped_and_ambiguous_devices_are_not_queried():
    adapter, api, quota = linux_quota_adapter()
    adapter.mount_reader = lambda: r'1 0 8:1 / / rw - ext4 /dev/a\040b rw'
    assert adapter.probe(123).capacity == 4096000
    assert api.opens[0][0] == '/dev/a b'
    api.opens.clear()
    quota.calls.clear()
    adapter.mount_reader = lambda: ('1 0 8:1 / / rw - ext4 /dev/a rw\n'
                                    '2 0 8:1 / /alias rw - ext4 /dev/b rw\n')
    assert adapter.probe(123).capacity is None
    assert not api.opens and not quota.calls


@pytest.mark.parametrize('code,expected', [(0, True), (errno.ESRCH, False),
                                         (errno.ENOSYS, None), (errno.EPERM, None),
                                         (errno.EINVAL, None)])
def test_typed_libc_quota_binding_uses_only_readonly_getfmt(code, expected):
    calls = []
    def quotactl(op, path, identity, output):
        calls.append((op, path, identity))
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint32))[0] = 2
        ctypes.set_errno(code)
        return -1 if code else 0
    binding = _LinuxQuotaAPI(library=SimpleNamespace(quotactl=quotactl))
    assert quotactl.argtypes == [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                ctypes.POINTER(ctypes.c_uint32)]
    assert quotactl.restype is ctypes.c_int
    if expected is None:
        with pytest.raises(OSError) as caught:
            binding.enabled(17, 2)
        assert caught.value.errno == code
    else:
        assert binding.enabled(17, 2) is expected
    assert calls == [(ctypes.c_int(0x80000402).value, b'/proc/self/fd/17', 0)]
    with pytest.raises(ValueError):
        binding.enabled(17, 3)


MNT_RDONLY, MNT_LOCAL = 0x1, 0x1000
MAC_DEV = 16777231


class DarwinOS(PosixOS):
    """Fake macOS os module: no O_PATH, no /proc, handle-anchored access."""
    W_OK, X_OK = 2, 1

    def __init__(self, *, writable=True, f_flag=0):
        super().__init__()
        self.fds, self.access_calls = {}, []
        self.writable, self.f_flag = writable, f_flag

    def open(self, name, flags, **kwargs):
        fd = super().open(name, flags, **kwargs)
        self.fds[name] = fd
        return fd

    def fstat(self, fd):
        return SimpleNamespace(st_dev=MAC_DEV, st_ino=fd, st_mode=stat.S_IFDIR, st_nlink=1,
                               st_size=0, st_mtime_ns=0, st_ctime_ns=0)

    def stat(self, name, *, dir_fd, follow_symlinks):
        assert follow_symlinks is False
        return SimpleNamespace(st_dev=MAC_DEV, st_ino=self.fds[name], st_mode=stat.S_IFDIR)

    def fstatvfs(self, fd):
        return SimpleNamespace(f_blocks=1000, f_frsize=4096, f_bavail=500, f_favail=73,
                               f_flag=self.f_flag)

    def access(self, *args, **kwargs):
        self.access_calls.append((args, kwargs))
        return self.writable

    def major(self, dev):
        raise AssertionError('darwin probe must not use Linux device numbers')


def darwin_provider(statfs, **os_kwargs):
    api = DarwinOS(**os_kwargs)
    adapter = DarwinAdapter(os_api=api, statfs=statfs)
    return make_volume_provider(adapter=adapter, clock=lambda: NOW), adapter, api


def test_darwin_local_apfs_identity_counts_and_sqlite_state_allowed():
    calls = []

    def statfs(fd):
        calls.append(fd)
        return 'apfs', MNT_LOCAL
    provider, _, api = darwin_provider(statfs)
    sample = provider.sample('/Users/owner/podmuncher/state', sqlite_state=True)
    # Mac failure was `assert None == 'posix:16777231'`: identity must survive probe.
    assert sample.identity.value == f'posix:{MAC_DEV}'
    assert (sample.capacity.value, sample.free.value, sample.free_inodes.value) == (4096000, 2048000, 73)
    assert sample.mounted.value is True and sample.writable.value is True
    assert sample.network.value is False and sample.state_allowed.value is True
    assert sample.free.source == 'darwin_statfs_statvfs'
    assert calls == [api.fds['state']]
    assert api.access_calls == [(('.', 3), {'dir_fd': api.fds['state'], 'effective_ids': True,
                                            'follow_symlinks': False})]
    # Only directory opens: O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC(|O_NONBLOCK), never O_PATH.
    assert {flags for _, flags, _ in api.opens} <= {7, 15}
    assert sorted(api.closed) == sorted(api.fds.values())


@pytest.mark.parametrize('flags,f_flag', [(MNT_LOCAL | MNT_RDONLY, 0), (MNT_LOCAL, 1)])
def test_darwin_read_only_from_statfs_or_statvfs(flags, f_flag):
    provider, _, api = darwin_provider(lambda fd: ('apfs', flags), f_flag=f_flag)
    sample = provider.sample('/Volumes/ro', sqlite_state=True)
    assert sample.writable.value is False and sample.network.value is False
    assert sample.identity.value == f'posix:{MAC_DEV}'
    assert not api.access_calls


@pytest.mark.parametrize('filesystem,flags', [('smbfs', 0), ('nfs', 0), ('afpfs', 0),
                                              ('webdav', 0), ('macfuse', MNT_LOCAL),
                                              ('apfs', 0)])
def test_darwin_network_or_remote_refused_for_sqlite_state(filesystem, flags):
    provider, _, _ = darwin_provider(lambda fd: (filesystem, flags))
    sample = provider.sample('/Volumes/share', sqlite_state=True)
    assert sample.network.value is True and sample.state_allowed.value is False
    assert sample.identity.value == f'posix:{MAC_DEV}'
    assert sample.free.value == 2048000


def test_darwin_unknown_local_filesystem_stays_unknown_for_state():
    provider, _, _ = darwin_provider(lambda fd: ('mysteryfs', MNT_LOCAL))
    sample = provider.sample('/Volumes/x', sqlite_state=True)
    assert sample.network.value is None and sample.network.status is FieldStatus.UNKNOWN
    assert sample.state_allowed.value is None and sample.mounted.value is True


@pytest.mark.parametrize('failure', [OSError(errno.EIO, 'private'), ('apfs', None), None])
def test_darwin_statfs_unavailable_keeps_counts_and_unknown_mount_facts(failure):
    def statfs(fd):
        if isinstance(failure, Exception):
            raise failure
        return failure
    provider, _, _ = darwin_provider(statfs)
    sample = provider.sample('/Users/owner', sqlite_state=True)
    assert sample.identity.value == f'posix:{MAC_DEV}'
    assert sample.capacity.value == 4096000
    assert sample.mounted.value is None and sample.network.value is None
    assert sample.state_allowed.value is None


def test_native_adapter_selects_by_injected_platform():
    assert type(native_adapter(platform='darwin')) is DarwinAdapter
    assert type(native_adapter(platform='linux')) is PosixAdapter
    if sys.platform == 'win32':
        assert type(native_adapter(platform='win32')) is WindowsAdapter
        assert type(native_adapter()) is WindowsAdapter


@pytest.mark.parametrize('machine,symbol', [('arm64', 'fstatfs'), ('x86_64', 'fstatfs$INODE64')])
def test_darwin_typed_statfs_binding_layout_and_symbol(machine, symbol):
    state = {'code': 0, 'name': b'apfs'}
    calls = []

    def fstatfs(fd, output):
        calls.append(fd)
        target = ctypes.cast(output, ctypes.POINTER(binding.Statfs))[0]
        target.f_fstypename = state['name']
        target.f_flags = MNT_LOCAL | MNT_RDONLY
        ctypes.set_errno(state['code'])
        return -1 if state['code'] else 0
    binding = _DarwinStatfsAPI(library=SimpleNamespace(**{symbol: fstatfs}), machine=machine)
    assert ctypes.sizeof(binding.Statfs) == 2168  # 64-bit-inode struct statfs
    assert binding.Statfs.f_flags.offset == 64 and binding.Statfs.f_fstypename.offset == 72
    assert fstatfs.restype is ctypes.c_int
    assert binding(7) == ('apfs', MNT_LOCAL | MNT_RDONLY)
    assert calls == [7]
    state['name'] = b'\xff\x01'
    with pytest.raises(OSError):
        binding(7)
    state.update(name=b'apfs', code=errno.EBADF)
    with pytest.raises(OSError) as caught:
        binding(7)
    assert caught.value.errno == errno.EBADF


@pytest.mark.skipif(sys.platform == 'darwin', reason='native statfs exists on macOS')
def test_darwin_statfs_binding_refuses_off_platform():
    with pytest.raises(OSError):
        _DarwinStatfsAPI()


def test_windows_adapter_injected_transport_uses_held_parent():
    class API:
        def __init__(self): self.opened = []
        def path(self, handle): return 'C:\\owned'
        def open(self, path, directory):
            self.opened.append((path, directory))
            return 999
    api = API()
    adapter = WindowsAdapter(api=api)
    assert adapter.open_child(123, 'file', False) == 999
    assert api.opened == [('C:\\owned\\file', False)]
    with pytest.raises(StorageError):
        adapter.split_absolute('\\\\server\\share\\state')


@pytest.mark.skipif(os.name != 'nt', reason='Windows share modes')
def test_windows_directory_open_waits_out_a_concurrent_holder(tmp_path):
    import threading
    import time

    from podmuncher.volumes import _WindowsAPI

    api = _WindowsAPI()
    first = api.open(str(tmp_path), True)
    assert api.permissions[first] is True
    timer = threading.Timer(0.02, api.close, args=(first,))
    timer.start()
    began = time.monotonic()
    second = api.open(str(tmp_path), True)  # collides until the first handle closes
    try:
        assert api.permissions[second] is True  # write permission known, not "unknown"
        assert time.monotonic() - began >= 0.01
    finally:
        timer.join()
        api.close(second)


def test_native_capacity_stable_identity_bounded_read(tmp_path):
    target = tmp_path / 'owned.txt'
    target.write_bytes(b'owned tiny transport')
    adapter = native_adapter()
    with open_contained(tmp_path, target.name, adapter=adapter) as held:
        identity = held.identity
        assert held.read(5) == b'owned'
        if os.name == 'nt':
            # All ancestors and file remain protected against rename until close.
            with pytest.raises(OSError):
                target.rename(tmp_path / 'moved.txt')
            with pytest.raises(OSError):
                tmp_path.rename(tmp_path.with_name(tmp_path.name + '-moved'))
    with open_contained(tmp_path, target.name, adapter=adapter, expected=identity) as held:
        assert held.read(1024) == b'owned tiny transport'
    provider = make_volume_provider(adapter=adapter, clock=lambda: NOW)
    sample = provider.sample(tmp_path)
    assert sample.identity.value == identity.volume
    assert sample.mounted.value is True
    if os.name == 'nt':
        assert sample.capacity.value > 0 and sample.free.value > 0
        assert sample.writable.value is True
        assert not adapter.api.permissions
    elif sys.platform == 'darwin':
        # The temp directory is on the local APFS data volume.
        assert sample.identity.value.startswith('posix:')
        assert sample.capacity.value > 0 and sample.free.value >= 0
        assert sample.network.value is False and sample.writable.value is True
        state = provider.sample(tmp_path, sqlite_state=True)
        assert state.state_allowed.value is True
    else:
        # Native support depends on filesystem, device visibility and quota state.
        # Deterministic tests above require real default resolution when supported.
        if sample.capacity.status is FieldStatus.OK:
            assert sample.capacity.value > 0 and sample.free.value >= 0
        else:
            assert sample.capacity.value is None
            assert sample.capacity.reason is ReasonCode.STORAGE_UNKNOWN
    target.unlink()


def test_native_hardlink_rejected(tmp_path):
    original = tmp_path / 'original'
    original.write_bytes(b'owned')
    alias = tmp_path / 'alias'
    os.link(original, alias)
    with pytest.raises(StorageError), open_contained(tmp_path, 'alias'):
        pytest.fail('hardlink admitted')
    alias.unlink()
    original.unlink()


def test_registry_negative_measurements_and_safe_errors(make_provider):
    registry = json.loads((Path(__file__).parents[1] / 'Docs/contract-domains.json').read_text())
    provider, adapter = make_provider()
    for negative in registry['negative_examples']:
        from podmuncher.snapshots import Measurement
        with pytest.raises(ValueError):
            Measurement(None, 'host_volume', negative, NOW.isoformat(), ReasonCode.STORAGE_UNKNOWN)
        with pytest.raises(ValueError):
            Measurement(None, 'host_volume', FieldStatus.UNKNOWN, NOW.isoformat(), negative)
    adapter.fail_probe = True
    result = provider.sample('/owned')
    for name in result.__dataclass_fields__:
        field = getattr(result, name)
        assert field.status.value in registry['field_status']
        assert field.reason is None or field.reason.value in registry['reason_code']


@pytest.mark.live
def test_own_host_transport_smoke(tmp_path):
    test_native_capacity_stable_identity_bounded_read(tmp_path)
    assert not tuple(tmp_path.iterdir())
    print('PODMUNCHER_SMOKE_OK')
