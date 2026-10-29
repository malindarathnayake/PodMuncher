"""Deterministic provider/transport contracts and an explicit real read-only smoke."""
# ruff: noqa: B023 - the live-smoke loop closures below run inside their own iteration.
import io
import json
import os
import posixpath
import stat
import subprocess
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import podmuncher.docker_storage as storage_module
from podmuncher.admission import (
    GrowthInterval,
    StoragePlacement,
    WorkloadGrowth,
    make_plan,
    make_workload_plan,
)
from podmuncher.config import Limits
from podmuncher.docker_storage import (
    BoundedRunner,
    DarwinIdentity,
    DarwinLayout,
    DiscoveryError,
    DockerStorageProvider,
    LinuxLayout,
    NativePlatformProvider,
    ProcessResult,
    WindowsVirtualDiskAPI,
    WSLLayout,
    make_docker_storage_provider,
    parse_msft_vpd83,
)
from podmuncher.domains import FieldStatus, ReasonCode
from podmuncher.snapshots import Measurement
from podmuncher.volumes import VolumeEvidence

NOW = datetime(2026, 10, 9, tzinfo=UTC)
STAMP = '2026-10-09T00:00:00Z'
GUID = '2280837d-08be-435f-8f95-e9e0637635a3'
# Measured on the owner's Mac (design-mac.md): sparse Docker.raw limit/allocation and host free.
DARWIN_HOME = '/Users/tester'
DOCKER_RAW = DARWIN_HOME + '/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw'
DARWIN_SOCKET = 'unix://' + DARWIN_HOME + '/.docker/run/docker.sock'
DARWIN_HOST = 'posix:16777231'
DARWIN_LIMIT, DARWIN_USED, DARWIN_HOST_FREE = 494384709632, 62797369344, 87543820288


def fake_process_result(payload=None, *, returncode=0, raw=None):
    return ProcessResult(returncode, raw if raw is not None else json.dumps(payload).encode())


def measure(value, *, stamp=STAMP):
    return Measurement(value, 'fixture', FieldStatus.OK if value is not None else FieldStatus.UNKNOWN,
                       stamp, None if value is not None else ReasonCode.STORAGE_UNKNOWN)


def volume(identity='host-volume', free=600, *, inodes=50, stamp=STAMP):
    return VolumeEvidence(measure(identity, stamp=stamp), measure(1000, stamp=stamp),
                          measure(free, stamp=stamp), measure(inodes, stamp=stamp),
                          measure(True, stamp=stamp), measure(True, stamp=stamp),
                          measure(False, stamp=stamp), measure(None, stamp=stamp))


class FakeVolumes:
    def __init__(self, values=None):
        self.paths = []
        self.values = values or {}

    def sample(self, path):
        self.paths.append(path)
        return self.values.get(path, volume())


class FakePlatform:
    def __init__(self):
        self.linux = LinuxLayout(('/var/lib/docker', '/var/lib/containerd'), True)
        self.wsl = WSLLayout('4.93.0.240920', '/var/lib/docker',
                             '/tmp/docker-desktop-root/var/lib', '/dev/sdd',
                             volume('wsl:uuid:' + GUID), r'D:\Docker\disk\docker_data.vhdx',
                             GUID, GUID, True, '/tmp/docker-desktop-root/var/lib/docker')
        self.darwin = DarwinLayout(None, '/var/lib/docker', DOCKER_RAW, (16777231, 4242),
                                   DARWIN_LIMIT, DARWIN_USED)
        self.darwin_calls = 0

    def linux_layout(self, info, call):
        return self.linux

    def wsl_layout(self, info, call, manual_backing):
        return self.wsl

    def darwin_layout(self, info, call, manual_backing):
        self.darwin_calls += 1
        return self.darwin


class FakeDocker:
    def __init__(self, platform='linux'):
        self.calls = []
        self.endpoint = {'win32': 'npipe:////./pipe/dockerDesktopLinuxEngine',
                         'darwin': DARWIN_SOCKET}.get(platform, 'unix:///var/run/docker.sock')
        self.info = {'OSType': 'linux', 'DockerRootDir': '/var/lib/docker', 'SecurityOptions': [],
                     'DriverStatus': [['driver-type', 'io.containerd.snapshotter.v1']]}
        if platform == 'darwin':
            self.info.update(OperatingSystem='Docker Desktop', Driver='overlayfs', Architecture='aarch64')
        self.platform_name = 'Docker Desktop 4.47.0 (206054)'
        self.selected = 'default'
        self.change_after_info = False

    def __call__(self, argv, *, deadline, env):
        self.calls.append(argv)
        assert 'DOCKER_HOST' not in env and 'DOCKER_CONTEXT' not in env
        assert deadline > time.monotonic()
        if argv[0] == storage_module._DARWIN_DOCKER:
            argv = ('docker', *argv[1:])
        if argv == ('docker', '--context', 'default', 'version', '--format', '{{json .Server.Platform.Name}}'):
            return fake_process_result(self.platform_name)
        if argv == ('docker', 'context', 'show'):
            return fake_process_result(raw=self.selected.encode())
        if argv[:3] == ('docker', 'context', 'inspect'):
            return fake_process_result([{'Name': argv[3] if len(argv) == 4 else self.selected,
                'Endpoints': {'docker': {'Host': self.endpoint}}}])
        if argv == ('docker', '--context', 'default', 'info', '--format', '{{json .}}'):
            if self.change_after_info:
                self.selected = 'changed'
            return fake_process_result(self.info)
        raise AssertionError('Unexpected command')


@pytest.fixture(autouse=True)
def hermetic_docker_lookup(request, monkeypatch):
    """The darwin CLI lookup resolves against this fake PATH, never the host's.

    The default plants docker on the child's PATH; tests that want the
    /usr/local/bin fallback clear ``on_path`` and mark that file executable.
    Live tests keep the real lookup.
    """
    fake = SimpleNamespace(on_path={'docker': '/fake/bin/docker'}, executable=set(), seen=[])
    if request.node.get_closest_marker('live'):
        return fake
    def which(name, mode=os.F_OK | os.X_OK, path=None):
        fake.seen.append((name, path))
        return fake.on_path.get(name)
    original_access = os.access
    def access(path, mode, *args, **kwargs):
        if str(path) == storage_module._DARWIN_DOCKER:
            return path in fake.executable
        return original_access(path, mode, *args, **kwargs)
    monkeypatch.setattr(storage_module.shutil, 'which', which)
    monkeypatch.setattr(storage_module.os, 'access', access)
    return fake


def provider(platform='linux', **kwargs):
    docker, disks, metadata = FakeDocker(platform), FakeVolumes(), FakePlatform()
    instance = DockerStorageProvider(runner=docker, volumes=disks, platform_provider=metadata,
                                      platform=platform, environ={}, clock=lambda: NOW, **kwargs)
    return instance, docker, disks, metadata


def test_linux_separate_containerd_constraints_and_no_images():
    instance, docker, disks, _ = provider()
    disks.values['/var/lib/containerd'] = volume('second-volume', 300)
    result = instance.sample()
    assert result.can_pull and result.can_create_cached
    assert len(result.constraints) == 2
    assert [x.volume.free.value for x in result.constraints] == [600, 300]
    assert disks.paths == ['/var/lib/docker', '/var/lib/containerd']
    assert all(not {'pull', 'run', 'create', 'images', 'image'} & set(cmd) for cmd in docker.calls)
    assert result.context.sample_time == STAMP
    with pytest.raises(FrozenInstanceError):
        result.constraints = ()


def test_aliases_deduplicate_without_summing_free():
    instance, _, disks, _ = provider()
    disks.values['/var/lib/containerd'] = volume('host-volume', 500)
    result = instance.sample()
    assert result.can_pull and len(result.constraints) == 1
    assert result.constraints[0].volume.free.value == 500
    assert result.constraints[0].roles == ('docker_data', 'containerd')


def test_wsl_vm_and_host_are_independent_constraints():
    instance, _, disks, _ = provider('win32')
    disks.values[r'D:\Docker\disk'] = volume('host-volume', 200, inodes=None)
    result = instance.sample()
    assert result.can_pull and result.can_create_cached
    assert result.desktop_version.value == '4.93.0.240920'
    assert [c.volume.free.value for c in result.constraints] == [600, 200]
    assert result.backing[0].verified.value is True
    assert result.backing[0].vm_identity.value != result.backing[0].host_identity.value


@pytest.mark.parametrize('change,reason', [
    ({'desktop_version': '29.8.1'}, ReasonCode.DOCKER_CONTEXT_UNSUPPORTED),
    ({'desktop_version': '4.29.0'}, ReasonCode.DOCKER_CONTEXT_UNSUPPORTED),
    ({'desktop_version': '5.0.0'}, ReasonCode.DOCKER_CONTEXT_UNSUPPORTED),
    ({'candidate_disk_id': 'unrelated'}, ReasonCode.BACKING_UNRESOLVED),
    ({'association_verified': False}, ReasonCode.BACKING_UNRESOLVED),
    ({'candidate_vhdx': None}, ReasonCode.BACKING_UNRESOLVED),
    ({'active_disk_id': None}, ReasonCode.BACKING_UNRESOLVED),
    ({'mount_target': '/'}, ReasonCode.STORAGE_UNKNOWN),
    ({'mount_target': '/unrelated'}, ReasonCode.STORAGE_UNKNOWN),
    ({'data_root': '/other'}, ReasonCode.STORAGE_UNKNOWN),
])
def test_wsl_unproved_or_unsupported_blocks_both(change, reason):
    instance, _, _, metadata = provider('win32')
    metadata.wsl = replace(metadata.wsl, **change)
    result = instance.sample()
    assert not result.can_pull and not result.can_create_cached and result.reason is reason


@pytest.mark.parametrize('endpoint', ['tcp://127.0.0.1:2375', 'ssh://server', 'unix://server/socket',
                                     '', 'npipe:////remote/pipe/docker_engine'])
def test_remote_or_ambiguous_endpoints_do_not_call_info(endpoint):
    instance, docker, _, _ = provider()
    docker.endpoint = endpoint
    result = instance.sample()
    assert result.reason is ReasonCode.DOCKER_CONTEXT_UNSUPPORTED
    assert not any('info' in cmd for cmd in docker.calls)


@pytest.mark.parametrize('environment', [{'DOCKER_HOST': 'ssh://private'},
                                        {'DOCKER_HOST': 'tcp://localhost:2375'},
                                        {'DOCKER_HOST': 'unix:///other.sock'},
                                        {'DOCKER_CONTEXT': '--remote'}])
def test_environment_cannot_override_validated_context(environment):
    instance, docker, _, _ = provider()
    instance.environ = environment
    result = instance.sample()
    assert not result.can_pull and result.reason is ReasonCode.DOCKER_CONTEXT_UNSUPPORTED
    assert not any('info' in cmd for cmd in docker.calls)


def test_changed_context_invalidates_all_storage_evidence():
    instance, docker, _, _ = provider()
    docker.change_after_info = True
    result = instance.sample()
    assert result.reason is ReasonCode.STORAGE_STALE
    assert result.admission.status is FieldStatus.STALE
    assert result.constraints == ()


@pytest.mark.parametrize('change', ['unknown', 'rootless', 'stale', 'full', 'inode_unknown'])
def test_unusable_layout_or_volume_blocks_both(change):
    instance, docker, disks, metadata = provider()
    if change == 'unknown':
        metadata.linux = replace(metadata.linux, verified=False)
    elif change == 'rootless':
        docker.info['SecurityOptions'] = ['name=rootless']
    elif change == 'stale':
        disks.values['/var/lib/docker'] = replace(volume(), free=measure(600, stamp='2026-10-08T00:00:00Z'))
    elif change == 'inode_unknown':
        disks.values['/var/lib/docker'] = volume(inodes=None)
    else:
        disks.values['/var/lib/docker'] = volume(free=0)
    result = instance.sample()
    assert not result.can_pull and not result.can_create_cached


def test_provider_exception_has_closed_diagnostic():
    instance, _docker, _, _ = provider()
    def failure(*args, **kwargs):
        raise OSError('private credentials C:/secret')
    instance.runner = failure
    result = instance.sample()
    assert result.reason is ReasonCode.STORAGE_UNKNOWN
    assert 'private' not in repr(result)


def test_negative_domain_examples():
    contract = json.loads(Path('Docs/contract-domains.json').read_text(encoding='utf-8'))
    for invalid in contract['negative_examples']:
        with pytest.raises(ValueError):
            ReasonCode.parse(invalid)
        with pytest.raises(ValueError):
            FieldStatus.parse(invalid)


class FakeChild:
    def __init__(self, stdout=b'', stderr=b'', running=False, fail_kill=False, fail_wait=False):
        self.stdout, self.stderr = io.BytesIO(stdout), io.BytesIO(stderr)
        self.returncode = None if running else 0
        self.killed = self.reaped = False
        self.fail_kill, self.fail_wait = fail_kill, fail_wait

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        if self.fail_kill:
            raise OSError('private process detail')
        self.returncode = -9

    def wait(self, timeout):
        assert timeout <= 1
        self.reaped = True
        if self.fail_wait:
            raise subprocess.TimeoutExpired('private', timeout)
        return self.returncode


def test_bounded_runner_success_and_no_shell():
    child = FakeChild(b'hello', b'ignored private diagnostic')
    def popen(argv, **kwargs):
        assert argv == ('docker', 'context', 'show') and kwargs['shell'] is False
        assert kwargs['stdin'] is subprocess.DEVNULL
        return child
    result = BoundedRunner(popen=popen)(('docker', 'context', 'show'), deadline=time.monotonic() + 2, env={})
    assert result.stdout == b'hello' and child.reaped


@pytest.mark.parametrize('prefix', [storage_module._WSL_PREFIX,
    ('wsl.exe', '--distribution', 'docker-desktop', '--exec')])
@pytest.mark.parametrize('payload', [b'', b'no final newline', b'\x00\x83\xff\n@@done:0\n@@reaped\n'])
def test_initial_wsl_transport_preserves_argv_and_binary_output(prefix, payload):
    arguments = ('cat', '/path with space/\'$(not-executed);value')
    child = FakeChild(payload + b'@@done:0\n@@reaped\n')
    child.stdin = io.BytesIO()
    def popen(argv, **kwargs):
        assert argv[:len(prefix)] == prefix
        offset = len(prefix)
        assert argv[offset:offset + 4] == ('/bin/sh', '-c', storage_module._WSL_SUPERVISOR,
                                            'storage-supervisor')
        assert argv[offset + 5:] == (storage_module._WSL_ONESHOT, '0', *arguments)
        assert kwargs['shell'] is False and kwargs['stdin'] is subprocess.PIPE
        return child
    result = BoundedRunner(popen=popen)((*prefix, *arguments),
        deadline=time.monotonic() + 1, env={})
    assert result == ProcessResult(0, payload) and child.reaped and child.stdin.closed


@pytest.mark.parametrize('footer,reason', [
    (b'@@done:17\n@@reaped\n', ReasonCode.DOCKER_UNAVAILABLE),
    (b'@@done:256\n@@reaped\n', ReasonCode.STORAGE_UNKNOWN),
    (b'@@done:bad\n@@reaped\n', ReasonCode.STORAGE_UNKNOWN),
    (b'@@reaped\n', ReasonCode.STORAGE_UNKNOWN),
    (b'@@done:0\n', ReasonCode.SERVICE_UNAVAILABLE),
])
def test_initial_wsl_transport_validates_exit_and_cleanup_footer(footer, reason):
    child = FakeChild(b'output' + footer)
    child.stdin = io.BytesIO()
    with pytest.raises(DiscoveryError) as error:
        BoundedRunner(popen=lambda *a, **k: child)((*storage_module._WSL_PREFIX, 'cat', '/proc/uptime'),
            deadline=time.monotonic() + 1, env={})
    assert error.value.reason is reason and child.reaped


def test_staged_wsl_observation_is_not_wrapped_twice():
    command = (*storage_module._WSL_PREFIX, '/bin/sh', '-c', storage_module._WSL_SUPERVISOR,
               'storage-supervisor', '60', storage_module._WSL_WORKER, '2')
    assert storage_module._wsl_transport_command(command, time.monotonic() + 1) == (command, False)


@pytest.mark.parametrize('footer', [b'', b'@@done:0', b'@@done:0\n',
    b'@@done:0\n@@reaped', b'@@done:1\n@@reaped\n',
    b'@@done:0\n@@done:0\n@@reaped\n', b'@@done:0\n@@reaped\npartial'])
def test_staged_completion_requires_one_complete_success_and_cleanup_footer(footer):
    with pytest.raises(DiscoveryError) as error:
        storage_module._witness_finished(b'@@identity_before\nrecord\n' + footer)
    assert error.value.reason is ReasonCode.STORAGE_UNKNOWN


def test_success_footer_cannot_substitute_for_owned_process_completion():
    # A real local child advertises success but waits on its owned input pipe.
    # The runner must still time out, close the pipe, and reap the actual child.
    children = []
    def spawn(argv, **kwargs):
        assert storage_module._WSL_SUPERVISOR in argv
        child = subprocess.Popen((sys.executable, '-c',
            ('import sys; sys.stdout.buffer.write(b"@@done:0\\n@@reaped\\n"); '
            'sys.stdout.flush(); sys.stdin.buffer.read()')), **kwargs)
        children.append(child)
        return child
    with pytest.raises(DiscoveryError) as error:
        BoundedRunner(popen=spawn)((*storage_module._WSL_PREFIX, 'true'),
            deadline=time.monotonic() + 0.5, env=os.environ)
    assert error.value.reason is ReasonCode.REQUEST_TIMEOUT
    assert len(children) == 1 and children[0].poll() == 0 and children[0].stdin.closed


@pytest.mark.parametrize('stream', ['stdout', 'stderr'])
def test_output_overflow_kills_and_reaps(stream):
    child = FakeChild(**{stream: b'x' * 20000}, running=True)
    with pytest.raises(DiscoveryError) as error:
        BoundedRunner(popen=lambda *a, **k: child, max_output=1000)(('docker',),
            deadline=time.monotonic() + 2, env={})
    assert error.value.reason is ReasonCode.REQUEST_TOO_LARGE
    assert child.killed and child.reaped


@pytest.mark.parametrize('fail_kill,fail_wait', [(False, False), (True, False), (False, True)])
def test_timeout_cleanup_is_bounded_and_sanitized(fail_kill, fail_wait):
    child = FakeChild(running=True, fail_kill=fail_kill, fail_wait=fail_wait)
    start = time.monotonic()
    with pytest.raises(DiscoveryError) as error:
        BoundedRunner(popen=lambda *a, **k: child)(('docker',), deadline=start + 0.02, env={})
    assert str(error.value) == ('service_unavailable' if fail_kill or fail_wait else 'request_timeout')
    assert child.killed and time.monotonic() - start < 2
    if not fail_kill:
        assert child.reaped


def test_real_transport_timeout_reaps_local_process():
    children = []
    def spawn(*args, **kwargs):
        child = subprocess.Popen(*args, **kwargs)
        children.append(child)
        return child
    with pytest.raises(DiscoveryError):
        BoundedRunner(popen=spawn)((sys.executable, '-c', 'import time; time.sleep(10)'),
                                  deadline=time.monotonic() + 0.1, env=dict(os.environ))
    assert children[0].poll() is not None


@pytest.mark.parametrize('stall', ['constructor', 'inspect'])
def test_initial_native_windows_api_stall_is_in_owned_reaped_helper(monkeypatch, stall):
    metadata, wsl, _ = native_wsl_fixture(monkeypatch)
    metadata.virtual_disk = None
    instance, docker, _, _ = provider('win32')
    instance.platform_provider = metadata
    instance.timeout = 0.3
    def forbidden(*args):
        raise AssertionError('Native Windows discovery must not run in the parent')
    monkeypatch.setattr(storage_module, '_desktop_version', forbidden)
    monkeypatch.setattr(storage_module, 'WindowsVirtualDiskAPI', forbidden)
    monkeypatch.setattr(metadata, '_candidates', forbidden)
    # Exercise the real same-module inventory body in an owned child. Its
    # injected API stalls indefinitely at each independently blocking seam.
    script = ('import time\nfrom podmuncher import docker_storage as s\n'
        's._desktop_version=lambda: "4.93.0.240920"\n'
        's.NativePlatformProvider._candidates=lambda self,p: (s.__file__,)\n'
        'class API:\n'
        ' def __init__(self):\n'
        + ('  time.sleep(30)\n' if stall == 'constructor' else '  pass\n')
        + ' def inspect(self,p): time.sleep(30)\n'
        's.WindowsVirtualDiskAPI=API\ns._native_windows_inventory("")\n')
    children = []
    def spawn(argv, **kwargs):
        assert argv[1:4] == ('-m', 'podmuncher.docker_storage', '--native-windows-inventory')
        assert kwargs['shell'] is False
        child = subprocess.Popen((sys.executable, '-c', script), **kwargs)
        children.append(child)
        return child
    bounded = BoundedRunner(popen=spawn)
    def runner(argv, **kwargs):
        if argv[0] == sys.executable:
            return bounded(argv, **kwargs)
        return wsl(argv) if argv[0] == 'wsl.exe' else docker(argv, **kwargs)
    instance.runner = runner
    started = time.monotonic()
    session = instance.prepare()
    assert time.monotonic() - started <= instance.timeout
    assert session.closed and session.initial_evidence.reason is ReasonCode.REQUEST_TIMEOUT
    assert len(children) == 1 and children[0].poll() is not None


@pytest.mark.parametrize('acknowledged', [False, True])
def test_supervised_cleanup_acknowledgement_is_required_even_after_timeout(acknowledged):
    script = ('import sys\nsys.stdout.buffer.write(b"ready\\n"); sys.stdout.flush()\n'
              'sys.stdin.read()\n' + ('sys.stdout.buffer.write(b"@@reaped\\n")\n' if acknowledged else ''))
    group = storage_module._OwnedProcesses(BoundedRunner(), time.monotonic() + 0.5, os.environ)
    child = group.spawn((sys.executable, '-c', script, storage_module._WSL_SUPERVISOR), interactive=True)
    try:
        group.wait(lambda: b'ready\n' in group.output(child))
    finally:
        assert group.close() is acknowledged
    assert group.children[child].poll() is not None


def vpd_page(ident=GUID):
    payload = b'MSFT    ' + uuid.UUID(ident).bytes_le
    descriptor = bytes((1, 1, 0, len(payload))) + payload
    return b'\x00\x83' + len(descriptor).to_bytes(2, 'big') + descriptor


def test_active_vpd_guid_uses_bytes_le():
    assert parse_msft_vpd83(vpd_page()) == GUID


@pytest.mark.parametrize('page', [b'', b'\0\x83\0\xff', vpd_page()[:-1],
                                 vpd_page().replace(b'MSFT', b'FAKE'),
                                 vpd_page(str(uuid.UUID(int=0))),
                                 b'\0\x83\0\x38' + vpd_page()[4:] * 2])
def test_vpd_rejects_truncated_unrelated_or_ambiguous(page):
    with pytest.raises(DiscoveryError):
        parse_msft_vpd83(page)


class APICall:
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, *args):
        return self.fn(*args)


@pytest.mark.parametrize('failure', [None, 13, 14, 9])
def test_virtual_disk_typed_v2_read_only_and_cleanup(failure):
    calls = []
    def opened(storage, path, access, flags, params, handle):
        assert storage._obj.DeviceId == 3 and access == flags == 0
        assert params._obj.Version == 2
        assert params._obj.data.V2.GetInfoOnly == params._obj.data.V2.ReadOnly == 1
        handle._obj.value = 123
        return 0
    def queried(handle, size, info, used):
        assert size._obj.value == 32
        version = info._obj.Version
        calls.append(version)
        if version == failure:
            return 5
        if version == 14:
            info._obj.data.VirtualDiskId.bytes[:] = uuid.UUID(GUID).bytes_le
        elif version == 13:
            info._obj.data.IsLoaded = 1
        else:
            info._obj.data.Physical.IsRemote = 0
        return 0
    closed = []
    api = WindowsVirtualDiskAPI(
        library=SimpleNamespace(OpenVirtualDisk=APICall(opened), GetVirtualDiskInformation=APICall(queried)),
        kernel=SimpleNamespace(CloseHandle=APICall(lambda h: closed.append(h.value) or 1)))
    assert api.c.sizeof(api.OpenParameters) == 44
    if failure:
        with pytest.raises(DiscoveryError):
            api.inspect('private.vhdx')
    else:
        assert api.inspect('private.vhdx') == (GUID, True, False)
        assert calls == [14, 13, 9]
    assert closed == [123]


def native_wsl_fixture(monkeypatch, *, version='4.93.0.240920', disk_result=(GUID, True, False)):
    registry = SimpleNamespace(HKEY_LOCAL_MACHINE=1, HKEY_CURRENT_USER=2,
                               OpenKey=lambda *a: nullcontext(object()),
                               QueryValueEx=lambda *a: (version, 1))
    monkeypatch.setitem(sys.modules, 'winreg', registry)
    metadata = NativePlatformProvider(virtual_disk=SimpleNamespace(inspect=lambda p: disk_result))
    monkeypatch.setattr(metadata, '_candidates', lambda manual: (r'D:\Docker\docker_data.vhdx',))
    state = {'mount_calls': 0, 'vpd_calls': 0, 'proof_calls': 0, 'change': None, 'calls': []}
    mount = {'target': '/tmp/docker-desktop-root/var/lib', 'source': '/dev/sdd[/data]',
             'fstype': 'ext4', 'options': 'rw,relatime', 'uuid': GUID, 'maj:min': '8:48'}
    credentials = 'Uid: 0 0 0 0\nGid: 0 0 0 0\nGroups: 0 1 2\nCapPrm: 000001ffffffffff\nCapEff: 000001ffffffffff'
    def proof(permission, identity=False):
        record = {
            'credentials': credentials, 'label': '6b 65 72 6e 65 6c 00',
            'mnt': 'mnt:[123]', 'user': 'user:[456]', 'docker_stat': '2096:21',
            'container_stat': '2096:22', 'docker_mount': json.dumps({'filesystems': [mount]}),
            'container_mount': json.dumps({'filesystems': [mount]})}
        if permission:
            for key in ('docker_mount', 'container_mount'):
                record[key] = json.dumps({'filesystems': [{**mount, 'target': '/var/lib'}]})
            record['capacity'] = state.get('capacity', '4096 100000 90000 1000')
            names = 'credentials label mnt user docker_stat container_stat docker_mount container_mount capacity'
        else:
            record.update(pid='155', comm='dockerd', exe='/usr/bin/dockerd',
                          process='155 (dockerd) S ' + '0 ' * 18 + state.get('epoch', '585'),
                          kernel='6.6.87.2-microsoft-standard-WSL2', canonical='/dev/sdd',
                          block='8:30 block special file', sysfs='/sys/devices/example/block/sdd', dev='8:48',
                          options='rw\nnoquota\nresuid=0',
                          superblock='Filesystem UUID: ' + GUID + '\nFilesystem magic number: 0xEF53\nFilesystem features: has_journal extent metadata_csum\nUser quota inode: 0\nGroup quota inode: 0\nProject quota inode: 0')
            names = ('pid comm exe process credentials label mnt user kernel canonical block sysfs dev '
                     'options superblock docker_stat container_stat docker_mount container_mount')
        change = state['change']
        if permission and change in ('uid', 'group', 'capability'):
            old, new = {'uid': ('Uid: 0 0 0 0', 'Uid: 0 1 0 0'),
                        'group': ('Groups: 0 1 2', 'Groups: 0 1'),
                        'capability': ('CapEff: 000001ffffffffff', 'CapEff: 0000000000000000')}[change]
            record['credentials'] = record['credentials'].replace(old, new)
        if permission and change in ('namespace', 'user_namespace', 'label', 'inode'):
            record[{'namespace': 'mnt', 'user_namespace': 'user', 'label': 'label', 'inode': 'docker_stat'}[change]] += '9'
        if not permission:
            changes = {
                'quota': ('options', 'rw\nusrquota'), 'unknown_quota': ('options', 'rw'),
                'unknown_option': ('options', 'rw\nnoquota\nunknown'),
                'kernel': ('kernel', '6.6.88-unknown'), 'block': ('block', '8:31 block special file'),
                'sysfs': ('sysfs', '/sys/devices/example/block/sde'), 'sysdev': ('dev', '8:49'),
                'exe': ('exe', '/tmp/dockerd'), 'pid': ('pid', '1 2'),
                'malformed_status': ('credentials', 'Uid: 0'), 'bad_label': ('label', ''),
            }
            if change in changes:
                key, value = changes[change]
                record[key] = value
            if change in ('hidden_quota', 'project', 'unknown_feature'):
                record['superblock'] = record['superblock'].replace('has_journal', {
                    'hidden_quota': 'quota', 'project': 'project', 'unknown_feature': 'FEATURE_R99'}[change])
            if change == 'quota_inode':
                record['superblock'] = record['superblock'].replace('User quota inode: 0', 'User quota inode: 3')
            if change == 'magic':
                record['superblock'] = record['superblock'].replace('0xEF53', '0xEF54')
            if change == 'uuid':
                record['superblock'] = record['superblock'].replace(GUID, '11111111-1111-1111-1111-111111111111')
            if state['proof_calls'] > 1:
                if change == 'starttime':
                    record['process'] = record['process'].replace('585', '586')
                if change == 'changed_device':
                    record['dev'] = '8:49'
                if change == 'changed_uuid':
                    record['superblock'] = record['superblock'].replace(GUID, '11111111-1111-1111-1111-111111111111')
        if identity:
            names = storage_module._IDENTITY_NAMES
        return fake_process_result(raw=''.join('@' + key + '\n' + record[key] + '\n' for key in names.split()).encode())
    def call(argv):
        state['calls'].append(argv)
        if argv[:6] == ('wsl.exe', '--distribution', 'docker-desktop', '--user', 'root', '--exec'):
            if argv[6:9] == ('/bin/sh', '-c', storage_module._WSL_IDENTITY):
                return proof(False, identity=True)
            if argv[6:9] == ('/bin/sh', '-c', storage_module._WSL_BATCH):
                def block():
                    state['vpd_calls'] += 1
                    ident = ('11111111-1111-1111-1111-111111111111'
                             if state['change'] == 'stale_vpd' else GUID)
                    leaf = '/sys/devices/example/block/sdd'
                    if state['change'] == 'parent':
                        leaf = '/sys/devices/example/block/sde'
                    return ('@device\nsdd\n@leaf\n' + leaf + '\n@whole\n' + leaf +
                            '\n@vpd\n' + vpd_page(ident).hex() + '\n').encode()
                first = block()
                state['proof_calls'] += 1
                before = proof(False).stdout
                if state['change'] in ('tools_absent', 'permission'):
                    raise DiscoveryError(ReasonCode.DOCKER_UNAVAILABLE)
                permission = proof(True).stdout
                state['proof_calls'] += 1
                after = proof(False).stdout
                return fake_process_result(raw=b'@@block_before\n' + first +
                    b'@@before\n' + before + b'@@permission\n' + permission +
                    b'@@after\n' + after + b'@@block_after\n' + block())
            permission = argv[6] == 'nsenter'
            if state['change'] == 'tools_absent' or permission and state['change'] == 'permission':
                raise DiscoveryError(ReasonCode.DOCKER_UNAVAILABLE)
            if permission:
                assert '--wdns=/' in argv and '--wd=/' not in argv
                assert argv[-2:] == ('/var/lib/docker', '/var/lib/containerd')
            else:
                state['proof_calls'] += 1
                assert argv[8] == storage_module._WSL_METADATA
            return proof(permission)
        assert argv[:4] == ('wsl.exe', '--distribution', 'docker-desktop', '--exec')
        if argv[4] == 'findmnt':
            state['mount_calls'] += 1
            current = dict(mount)
            if state['change'] == 'root':
                current['target'] = '/'
            if state['change'] == 'separate' and 'containerd' in argv[7]:
                current['source'] = '/dev/sde'
            if state['change'] == 'stale_mount' and state['mount_calls'] >= 3:
                current['uuid'] = 'different'
            return fake_process_result({'filesystems': [current]})
        if argv[4] == 'stat':
            assert argv[-1] == '/tmp/docker-desktop-root/var/lib/docker'
            return fake_process_result(raw=b'4096 100000 90000 1000\n')
        if argv[4] == 'lsblk':
            return fake_process_result(raw=b'\n')
        if argv[4] == 'cat':
            state['vpd_calls'] += 1
            changed = state['change'] == 'stale_vpd' and state['vpd_calls'] > 1
            return fake_process_result(raw=vpd_page('11111111-1111-1111-1111-111111111111' if changed else GUID))
        raise AssertionError('Unexpected command')
    return metadata, call, state


def test_native_wsl_proves_data_and_containerd_mounts_and_backing(monkeypatch):
    metadata, call, state = native_wsl_fixture(monkeypatch)
    layout = metadata.wsl_layout({'DockerRootDir': '/var/lib/docker'}, call, '')
    assert layout.association_verified
    assert layout.active_disk_id == layout.candidate_disk_id == GUID
    assert layout.mount_target == '/tmp/docker-desktop-root/var/lib'
    assert layout.vm_volume.identity.value == 'wsl:' + GUID + ':' + GUID
    assert layout.vm_volume.writable.value is True and state['proof_calls'] == 2
    assert state['mount_calls'] == 4 and state['vpd_calls'] == 3
    assert all(not {'pull', 'run', 'create'} & set(cmd) for cmd in state['calls'])


@pytest.mark.parametrize('change,reason', [('root', ReasonCode.STORAGE_UNKNOWN),
                                         ('separate', ReasonCode.BACKING_UNRESOLVED),
                                         ('stale_mount', ReasonCode.STORAGE_STALE),
                                         ('stale_vpd', ReasonCode.STORAGE_STALE)])
def test_native_wsl_wrong_or_changed_mount_metadata_blocks(monkeypatch, change, reason):
    metadata, call, state = native_wsl_fixture(monkeypatch)
    state['change'] = change
    with pytest.raises(DiscoveryError) as error:
        metadata.wsl_layout({'DockerRootDir': '/var/lib/docker'}, call, '')
    assert error.value.reason is reason


@pytest.mark.parametrize('disk_result', [('unrelated', True, False), (GUID, False, False),
                                        (GUID, True, True)])
def test_native_wsl_rejects_unrelated_unloaded_or_remote_vhdx(monkeypatch, disk_result):
    metadata, call, _ = native_wsl_fixture(monkeypatch, disk_result=disk_result)
    layout = metadata.wsl_layout({'DockerRootDir': '/var/lib/docker'}, call, '')
    assert not layout.association_verified
    assert layout.candidate_vhdx is None


def test_native_wsl_checks_installed_desktop_version_before_commands(monkeypatch):
    metadata, call, state = native_wsl_fixture(monkeypatch, version='29.8.1')
    with pytest.raises(DiscoveryError) as error:
        metadata.wsl_layout({'DockerRootDir': '/var/lib/docker'}, call, '')
    assert error.value.reason is ReasonCode.DOCKER_CONTEXT_UNSUPPORTED
    assert not state['calls']


@pytest.mark.parametrize('change', [
    'quota', 'unknown_quota', 'unknown_option', 'hidden_quota', 'project', 'unknown_feature',
    'quota_inode', 'magic', 'uuid', 'kernel', 'block', 'sysfs', 'sysdev', 'exe', 'pid',
    'malformed_status', 'bad_label', 'permission', 'uid', 'group', 'capability',
    'namespace', 'user_namespace', 'label', 'inode', 'starttime', 'changed_device',
    'changed_uuid', 'tools_absent',
])
def test_native_wsl_requires_quota_and_actual_daemon_permission(monkeypatch, change):
    metadata, call, state = native_wsl_fixture(monkeypatch)
    state['change'] = change
    instance, docker, _, _ = provider('win32')
    instance.platform_provider = metadata
    def runner(argv, **kwargs):
        return call(argv) if argv[0] == 'wsl.exe' else docker(argv, **kwargs)
    instance.runner = runner
    result = instance.sample()
    assert not result.can_pull and not result.can_create_cached
    assert not result.constraints


@pytest.mark.parametrize('endpoint,environment', [
    ('npipe:////./pipe/docker_engine', {}),
    ('npipe:////./pipe/docker_engine', {'DOCKER_HOST': 'npipe:////./pipe/docker_engine'}),
    ('ssh://server', {}),
    ('npipe:////./pipe/dockerDesktopLinuxEngine', {'DOCKER_HOST': 'tcp://localhost:2375'}),
])
def test_desktop_endpoint_requires_specific_pipe_before_layout(endpoint, environment):
    instance, docker, _, metadata = provider('win32')
    docker.endpoint, instance.environ = endpoint, environment
    metadata.wsl_layout = lambda *args: pytest.fail('Unbound daemon reached Desktop layout')
    result = instance.sample()
    assert not result.can_pull and not result.can_create_cached
    assert result.reason is ReasonCode.DOCKER_CONTEXT_UNSUPPORTED
    assert not any('info' in cmd for cmd in docker.calls)
    # Shared generic-local helper remains usable by other hardware discovery.
    assert storage_module._local_endpoint('npipe:////./pipe/docker_engine', 'win32')


def host_volume(free=DARWIN_HOST_FREE, *, capacity=1000000000000, stamp=STAMP):
    return replace(volume(DARWIN_HOST, free, inodes=10 ** 9, stamp=stamp),
                   capacity=measure(capacity, stamp=stamp))


def darwin_provider(**kwargs):
    instance, docker, disks, metadata = provider('darwin', **kwargs)
    disks.values[posixpath.dirname(DOCKER_RAW)] = host_volume()
    return instance, docker, disks, metadata


def test_darwin_docker_desktop_disk_bounds_one_host_volume_constraint():
    instance, docker, disks, metadata = darwin_provider()
    result = instance.sample()
    assert result.can_pull and result.can_create_cached
    assert result.desktop_version.value == '4.47.0' and result.data_root.value == '/var/lib/docker'
    assert len(result.constraints) == 1 and result.backing == ()
    [constraint] = result.constraints
    assert constraint.roles == ('docker_data',) and constraint.paths == ('/var/lib/docker',)
    # The sparse file lives on the host volume: same identity as other host samples.
    assert constraint.volume.identity.value == DARWIN_HOST
    assert constraint.volume.free.value == min(DARWIN_LIMIT - DARWIN_USED, DARWIN_HOST_FREE) == DARWIN_HOST_FREE
    assert constraint.volume.free.source == 'fixture'
    assert constraint.volume.free_inodes.value == 10 ** 9
    assert disks.paths == [posixpath.dirname(DOCKER_RAW)]
    assert metadata.darwin_calls == 2  # discovery, then a replacement recheck after host sampling
    assert all(not {'pull', 'run', 'create', 'images', 'image'} & set(cmd) for cmd in docker.calls)
    assert ('docker', '--context', 'default', 'version', '--format', '{{json .Server.Platform.Name}}') in docker.calls


def test_darwin_small_disk_limit_tightens_free_without_summing():
    instance, _, _, metadata = darwin_provider()
    limit = 64 * 1024 ** 3
    metadata.darwin = replace(metadata.darwin, limit=limit)
    result = instance.sample()
    assert result.can_pull
    [constraint] = result.constraints
    assert constraint.volume.free.value == limit - DARWIN_USED < DARWIN_HOST_FREE
    assert constraint.volume.free.source == 'docker_desktop_disk'
    assert constraint.volume.capacity.value == 1000000000000


@pytest.mark.parametrize('change,reason', [
    ('orbstack', ReasonCode.DOCKER_CONTEXT_UNSUPPORTED),
    ('rootless', ReasonCode.DOCKER_CONTEXT_UNSUPPORTED),
    ('root', ReasonCode.STORAGE_UNKNOWN),
    ('relative', ReasonCode.STORAGE_UNKNOWN),
    ('not_raw', ReasonCode.STORAGE_UNKNOWN),
    ('zero_limit', ReasonCode.STORAGE_UNKNOWN),
    ('full', ReasonCode.STORAGE_UNKNOWN),
    ('host_unknown', ReasonCode.STORAGE_UNKNOWN),
    ('host_network', ReasonCode.STORAGE_UNKNOWN),
    ('missing', ReasonCode.BACKING_UNRESOLVED),
    ('replaced', ReasonCode.STORAGE_STALE),
])
def test_darwin_unsupported_engine_or_unproved_disk_blocks_both(change, reason):
    instance, docker, disks, metadata = darwin_provider()
    parent = posixpath.dirname(DOCKER_RAW)
    if change == 'orbstack':
        docker.info['OperatingSystem'] = 'OrbStack'
    elif change == 'rootless':
        docker.info['SecurityOptions'] = ['name=rootless']
    elif change == 'root':
        metadata.darwin = replace(metadata.darwin, data_root='/other')
    elif change == 'relative':
        metadata.darwin = replace(metadata.darwin, backing='Library/Docker.raw')
    elif change == 'not_raw':
        metadata.darwin = replace(metadata.darwin, backing=parent + '/Docker.qcow2')
    elif change == 'zero_limit':
        metadata.darwin = replace(metadata.darwin, limit=0)
    elif change == 'full':
        metadata.darwin = replace(metadata.darwin, used=DARWIN_LIMIT + 4096)
    elif change == 'host_unknown':
        disks.values[parent] = replace(host_volume(), free=measure(None))
    elif change == 'host_network':
        disks.values[parent] = replace(host_volume(), network=measure(True))
    elif change == 'missing':
        def missing(info, call, manual_backing):
            raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
        metadata.darwin_layout = missing
    else:
        layouts = iter((metadata.darwin, replace(metadata.darwin, file=(16777231, 9999))))
        metadata.darwin_layout = lambda info, call, manual_backing: next(layouts)
    result = instance.sample()
    assert not result.can_pull and not result.can_create_cached
    assert result.reason is reason and result.constraints == ()
    if change in ('orbstack', 'rootless'):
        assert not any('version' in cmd for cmd in docker.calls)


def test_darwin_unparsed_desktop_version_is_unknown_not_blocking():
    instance, docker, _, _ = darwin_provider()
    docker.platform_name = 'Docker Engine - Community'
    result = instance.sample()
    assert result.can_pull
    assert result.desktop_version.value is None and result.desktop_version.status is FieldStatus.UNKNOWN


def test_darwin_manual_backing_reaches_layout_and_baseline():
    instance, _, _, metadata = darwin_provider()
    seen = []
    layout = metadata.darwin
    def record(info, call, manual_backing):
        seen.append(manual_backing)
        return layout
    metadata.darwin_layout = record
    with instance.prepare(manual_backing='/Volumes/Big/Docker.raw') as session:
        assert session.initial_evidence.can_pull
        assert session.refresh(deadline=time.monotonic() + 1).can_pull
    assert set(seen) == {'/Volumes/Big/Docker.raw'} and len(seen) == 4


def test_darwin_prepared_refresh_remeasures_counters_and_closes_on_replacement():
    instance, docker, disks, metadata = darwin_provider()
    instance.clock = lambda: datetime.now(UTC)
    parent = posixpath.dirname(DOCKER_RAW)
    def fresh_host(free):
        disks.values[parent] = host_volume(free, stamp=datetime.now(UTC).isoformat().replace('+00:00', 'Z'))
    fresh_host(DARWIN_HOST_FREE)
    with instance.prepare() as session:
        old = session.initial_evidence
        assert old.can_pull and old.constraints[0].volume.free.value == DARWIN_HOST_FREE
        version_calls = sum('version' in cmd for cmd in docker.calls)
        metadata.darwin = replace(metadata.darwin, used=DARWIN_LIMIT - 1024)
        fresh_host(DARWIN_HOST_FREE - 1)
        fresh = session.refresh(deadline=time.monotonic() + 1)
        assert fresh.can_pull and fresh.constraints[0].volume.free.value == 1024
        assert fresh.desktop_version.value == '4.47.0'
        assert sum('version' in cmd for cmd in docker.calls) == version_calls  # reused, not re-asked
        assert old.constraints[0].volume.free.value == DARWIN_HOST_FREE
        metadata.darwin = replace(metadata.darwin, file=(16777231, 9999))
        fresh_host(DARWIN_HOST_FREE)
        stale = session.refresh(deadline=time.monotonic() + 1)
        assert not stale.can_pull and stale.reason is ReasonCode.STORAGE_STALE and session.closed


def native_darwin_fixture(monkeypatch, *, files=None, settings=None):
    """Fake darwin: lstat of Docker.raw candidates, stat of the engine socket, settings JSON."""
    def node(size=DARWIN_LIMIT, blocks=DARWIN_USED // 512, *, dev=16777231, ino=4242, mode=stat.S_IFREG):
        return SimpleNamespace(st_mode=mode, st_size=size, st_blocks=blocks, st_dev=dev, st_ino=ino)
    files = {DOCKER_RAW: node()} if files is None else files
    state = {'files': files, 'socket_ino': 777, 'lstat_calls': []}
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: Path(DARWIN_HOME)))
    original_lstat, original_stat = os.lstat, os.stat
    def lstat(path, *args, **kwargs):
        key = str(path).replace('\\', '/')
        if key.startswith('/'):
            state['lstat_calls'].append(key)
            if key in state['files']:
                return state['files'][key]
            raise FileNotFoundError(key)
        return original_lstat(path, *args, **kwargs)
    def stat_(path, *args, **kwargs):
        key = str(path).replace('\\', '/')
        if key == DARWIN_SOCKET[7:]:
            return SimpleNamespace(st_mode=state.get('socket_mode', stat.S_IFSOCK), st_dev=1, st_ino=state['socket_ino'])
        if key.startswith('/'):
            raise FileNotFoundError(key)
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(storage_module.os, 'lstat', lstat)
    monkeypatch.setattr(storage_module.os, 'stat', stat_)
    payloads = {} if settings is None else {
        DARWIN_HOME + '/Library/Group Containers/group.com.docker/settings-store.json': json.dumps(settings).encode()}
    def read(path, *args):
        key = path.as_posix()
        if key in payloads:
            return payloads[key]
        raise FileNotFoundError(key)
    monkeypatch.setattr(storage_module, '_bounded_file', read)
    state['node'] = node
    return state


def native_darwin_provider(monkeypatch, **fixture):
    state = native_darwin_fixture(monkeypatch, **fixture)
    instance, docker, disks, _ = darwin_provider()
    instance.platform_provider = NativePlatformProvider()
    for parent in ('/Volumes/Big', '/Volumes/Big/vms/0/data'):
        disks.values[parent] = host_volume()
    return instance, docker, disks, state


def test_native_darwin_default_docker_raw_measures_limit_and_allocation(monkeypatch):
    instance, _, disks, state = native_darwin_provider(monkeypatch)
    result = instance.sample()
    assert result.can_pull, result.reason
    assert result.constraints[0].volume.free.value == DARWIN_HOST_FREE
    assert disks.paths == [posixpath.dirname(DOCKER_RAW)] * 1
    layout = NativePlatformProvider().darwin_layout({'DockerRootDir': '/var/lib/docker'}, None, '')
    assert layout == DarwinLayout(None, '/var/lib/docker', DOCKER_RAW, (16777231, 4242), DARWIN_LIMIT, DARWIN_USED)
    # Only the settings file and the default location are consulted without an override.
    assert state['lstat_calls'][:1] == [DOCKER_RAW]


@pytest.mark.parametrize('change', ['missing', 'symlink', 'directory', 'ambiguous', 'zero', 'no_blocks'])
def test_native_darwin_unproved_docker_raw_is_backing_unresolved(monkeypatch, change):
    settings = {'DataFolder': '/Volumes/Big/vms/0/data'} if change == 'ambiguous' else None
    instance, _, _, state = native_darwin_provider(monkeypatch, settings=settings)
    node = state['node']
    if change == 'missing':
        state['files'].clear()
    elif change == 'symlink':
        state['files'][DOCKER_RAW] = node(mode=stat.S_IFLNK)
    elif change == 'directory':
        state['files'][DOCKER_RAW] = node(mode=stat.S_IFDIR)
    elif change == 'ambiguous':
        state['files']['/Volumes/Big/vms/0/data/Docker.raw'] = node(ino=5)
    elif change == 'zero':
        state['files'][DOCKER_RAW] = node(size=0)
    else:
        state['files'][DOCKER_RAW] = SimpleNamespace(st_mode=stat.S_IFREG, st_size=DARWIN_LIMIT, st_dev=1, st_ino=2)
    result = instance.sample()
    assert not result.can_pull and result.constraints == ()
    expected = ReasonCode.STORAGE_UNKNOWN if change in ('zero', 'no_blocks') else ReasonCode.BACKING_UNRESOLVED
    assert result.reason is expected


@pytest.mark.parametrize('manual,planted,expected', [
    ('/Volumes/Big/Docker.raw', '/Volumes/Big/Docker.raw', '/Volumes/Big/Docker.raw'),
    ('/Volumes/Big', '/Volumes/Big/Docker.raw', '/Volumes/Big/Docker.raw'),
    ('/Volumes/Big/', '/Volumes/Big/Docker.raw', '/Volumes/Big/Docker.raw'),
    ('/Volumes/Big', '/Volumes/Big/vms/0/data/Docker.raw', '/Volumes/Big/vms/0/data/Docker.raw'),
    ('/Volumes/Big/vms/0/data', '/Volumes/Big/vms/0/data/Docker.raw', '/Volumes/Big/vms/0/data/Docker.raw'),
    ('/Volumes/Empty', '/Volumes/Big/Docker.raw', None),
    ('relative/Docker.raw', '/Volumes/Big/Docker.raw', None),
    ('/Volumes/../Docker.raw', '/Volumes/Big/Docker.raw', None),
])
def test_native_darwin_manual_backing_overrides_discovery(monkeypatch, manual, planted, expected):
    instance, _, disks, state = native_darwin_provider(monkeypatch)
    state['files'][planted] = state['node'](dev=99, ino=7)
    result = instance.sample(manual_backing=manual)
    if expected is None:
        assert not result.can_pull and result.reason is ReasonCode.BACKING_UNRESOLVED
        assert DOCKER_RAW not in state['lstat_calls']  # the override never falls back to discovery
        return
    assert result.can_pull, result.reason
    assert disks.paths == [posixpath.dirname(expected)]
    with instance.prepare(manual_backing=manual) as session:
        assert session.refresh(deadline=time.monotonic() + 1).can_pull


def test_native_darwin_override_directory_with_two_disks_is_ambiguous(monkeypatch):
    instance, _, _, state = native_darwin_provider(monkeypatch)
    state['files']['/Volumes/Big/Docker.raw'] = state['node'](dev=99, ino=7)
    state['files']['/Volumes/Big/vms/0/data/Docker.raw'] = state['node'](dev=99, ino=8)
    assert instance.sample(manual_backing='/Volumes/Big').reason is ReasonCode.BACKING_UNRESOLVED
    assert instance.sample(manual_backing='/Volumes/Big/Docker.raw').can_pull


def test_native_darwin_settings_data_folder_is_preferred_candidate(monkeypatch):
    instance, _, disks, state = native_darwin_provider(monkeypatch, files={},
                                                        settings={'DataFolder': '/Volumes/Big/vms/0/data'})
    state['files']['/Volumes/Big/vms/0/data/Docker.raw'] = state['node'](dev=99, ino=8)
    result = instance.sample()
    assert result.can_pull, result.reason
    assert disks.paths == ['/Volumes/Big/vms/0/data']


@pytest.mark.parametrize('change,reason', [
    ('not_socket', ReasonCode.STORAGE_UNKNOWN),
    ('restarted', ReasonCode.STORAGE_STALE),
])
def test_native_darwin_engine_socket_identity_brackets_the_sample(monkeypatch, change, reason):
    instance, docker, _, state = native_darwin_provider(monkeypatch)
    if change == 'not_socket':
        state['socket_mode'] = stat.S_IFREG
    else:
        inner = docker.__call__
        def restart(argv, **kwargs):
            if 'info' in argv:
                state['socket_ino'] += 1  # Docker Desktop restarted after we identified the engine
            return inner(argv, **kwargs)
        instance.runner = restart
    result = instance.sample()
    assert not result.can_pull and result.reason is reason and result.constraints == ()
    if change == 'not_socket':
        assert not any('info' in cmd for cmd in docker.calls)


def test_native_darwin_identity_requires_local_socket_endpoint(monkeypatch):
    native_darwin_fixture(monkeypatch)
    identity = storage_module._darwin_identity(DARWIN_SOCKET)
    assert identity == DarwinIdentity(DARWIN_SOCKET[7:], (1, 777))
    with pytest.raises(DiscoveryError) as error:
        storage_module._darwin_identity(DARWIN_SOCKET, DarwinIdentity(DARWIN_SOCKET[7:], (1, 1)))
    assert error.value.reason is ReasonCode.STORAGE_STALE
    with pytest.raises(ValueError):
        storage_module._darwin_identity('unix://relative.sock')


@pytest.mark.parametrize('on_path', [True, False])
def test_darwin_docker_cli_falls_back_to_usr_local_bin_when_absent_from_path(hermetic_docker_lookup, on_path):
    instance, docker, _, _ = darwin_provider()
    hermetic_docker_lookup.on_path = {'docker': '/opt/docker'} if on_path else {}
    hermetic_docker_lookup.executable = {storage_module._DARWIN_DOCKER}
    assert instance.sample().can_pull
    executables = {cmd[0] for cmd in docker.calls}
    assert executables == ({'docker'} if on_path else {storage_module._DARWIN_DOCKER})
    assert storage_module._docker_executable('linux', {}) == 'docker'


def test_darwin_executable_lookup_follows_child_path(hermetic_docker_lookup):
    hermetic_docker_lookup.on_path = {}
    assert storage_module._docker_executable('darwin', {'PATH': '/usr/bin:/bin'}) == 'docker'
    assert hermetic_docker_lookup.seen == [('docker', '/usr/bin:/bin')]


def test_darwin_fallback_requires_executable_desktop_cli(hermetic_docker_lookup):
    hermetic_docker_lookup.on_path = {}
    assert storage_module._docker_executable('darwin', {}) == 'docker'
    hermetic_docker_lookup.executable = {storage_module._DARWIN_DOCKER}
    assert storage_module._docker_executable('darwin', {}) == storage_module._DARWIN_DOCKER
    assert hermetic_docker_lookup.seen == [('docker', os.defpath)] * 2


ACTIVE_CONTENT = '/runtime/store-a/io.containerd.content.v1.content'
ACTIVE_METADATA = '/runtime/store-a/io.containerd.metadata.v1.bolt'
ACTIVE_TASK = '/runtime/store-a/io.containerd.runtime.v2.task'
ACTIVE_SANDBOX = '/runtime/store-a/io.containerd.sandbox.controller.v1.shim'
ACTIVE_TASK_STATE = '/run/active-containerd/io.containerd.runtime.v2.task'
ACTIVE_SANDBOX_STATE = '/run/active-containerd/io.containerd.sandbox.controller.v1.shim'
NORMAL_LINUX_ROOTS = ('/var/lib/docker', ACTIVE_CONTENT, ACTIVE_METADATA, '/snapshots/live',
                      '/native/live', '/opt/containerd/live', ACTIVE_TASK, ACTIVE_SANDBOX,
                      ACTIVE_TASK_STATE, ACTIVE_SANDBOX_STATE)


def normal_linux_plugins():
    # Complete representative v2.0.0 Linux builtins with CRI disabled (a Docker
    # profile), overlayfs/native active, optional filesystem plugins skipped.
    # Identities/exports follow cmd/containerd/builtins and their registrations,
    # not the production parser's allowlist. v1.7 compatibility is tested below.
    records = [
        f'Type:    io.containerd.content.v1\nID:    content\nExports:\n    root    {ACTIVE_CONTENT}\n',
        ('Type:    io.containerd.metadata.v1\nID:    bolt\nRequires:\n    io.containerd.content.v1\n'
         '    io.containerd.event.v1\n    io.containerd.snapshotter.v1\nExports:\n    policy    shared\n'
         f'    path    {ACTIVE_METADATA}/meta.db\n'),
        ('Type:    io.containerd.snapshotter.v1\nID:    overlayfs\nPlatforms:    linux/amd64\n'
         'Exports:\n    root    /snapshots/live\nCapabilities:    remap-ids,only-remap-ids\n'),
        ('Type:    io.containerd.snapshotter.v1\nID:    native\nPlatforms:    linux/amd64\n'
         'Exports:\n    root    /native/live\n'),
        'Type:    io.containerd.internal.v1\nID:    opt\nExports:\n    path    /opt/containerd/live\n',
    ]
    services = {
        'io.containerd.event.v1': ('exchange',),
        'io.containerd.gc.v1': ('scheduler',),
        'io.containerd.differ.v1': ('walking',),
        'io.containerd.lease.v1': ('manager',),
        'io.containerd.service.v1': ('containers-service', 'content-service', 'diff-service',
            'images-service', 'introspection-service', 'namespaces-service', 'snapshots-service', 'tasks-service'),
        'io.containerd.shim.v1': ('manager',),
        'io.containerd.monitor.task.v1': ('cgroups',),
        'io.containerd.monitor.container.v1': ('restart',),
        'io.containerd.sandbox.controller.v1': ('shim',),
        'io.containerd.sandbox.store.v1': ('local',),
        'io.containerd.streaming.v1': ('manager',),
        'io.containerd.transfer.v1': ('local',),
        'io.containerd.image-verifier.v1': ('bindir',),
        'io.containerd.nri.v1': ('nri',),
        'io.containerd.warning.v1': ('deprecations',),
        'io.containerd.grpc.v1': ('containers', 'content', 'diff', 'events', 'healthcheck',
            'images', 'introspection', 'leases', 'namespaces', 'sandbox-controllers',
            'sandboxes', 'snapshots', 'streaming', 'tasks', 'transfer', 'version'),
    }
    for kind, names in services.items():
        records.extend(f'Type:    {kind}\nID:    {name}\n' for name in names)
    records.append('Type:    io.containerd.runtime.v2\nID:    task\nRequires:\n'
                   '    io.containerd.shim.v1\nPlatforms:    linux/amd64\n')
    for name in ('btrfs', 'blockfile', 'devmapper', 'zfs'):
        records.append(f'Type:    io.containerd.snapshotter.v1\nID:    {name}\n'
                       'Platforms:    linux/amd64\nError:\n    Code:    Unknown\n'
                       '    Message:    unavailable on this filesystem: skip plugin\n')
    for kind, name in (('io.containerd.tracing.processor.v1', 'otlp'), ('io.containerd.internal.v1', 'tracing')):
        records.append(f'Type:    {kind}\nID:    {name}\nError:\n    Code:    Unknown\n'
                       '    Message:    tracing endpoint not configured: skip plugin\n')
    return records


def native_linux_fixture(monkeypatch, *, imports='', override=False):
    # Runtime exports are deliberately independent of mutable config and CLI
    # root hints. Editing config to store B must never conceal active store A.
    original_iterdir = Path.iterdir
    def entries(path):
        if path.as_posix() == '/proc':
            return iter((Path('/proc/100'), Path('/proc/200')))
        if path.as_posix() == '/proc/100/fd':
            return iter((Path('/proc/100/fd/3'),))
        if path.as_posix() == '/proc/200/fd':
            return iter((Path('/proc/200/fd/4'),))
        return original_iterdir(path)
    monkeypatch.setattr(Path, 'iterdir', entries)
    payloads = {
        '/proc/100/cmdline': b'/usr/bin/dockerd\0--containerd=/run/containerd.sock\0',
        '/proc/200/cmdline': b'/usr/bin/containerd\0--state=/run/active-containerd\0--config=/etc/containerd/config.toml\0' +
            (b'--root=/cli/store\0' if override else b''),
        '/proc/100/stat': b'100 (dockerd) S ' + b'0 ' * 18 + b'585',
        '/proc/200/stat': b'200 (containerd) S ' + b'0 ' * 18 + b'586',
        '/proc/net/unix': (b'header\n0: 2 0 10000 1 1 123 /var/run/docker.sock\n'
                           b'0: 2 0 10000 1 1 456 /run/containerd.sock\n'),
        '/etc/docker/daemon.json': b'{}',
        '/etc/containerd/config.toml': (imports + 'root = "/configured/store-b"\n[grpc]\naddress = "/run/containerd.sock"').encode(),
    }
    def read(path, *args):
        return payloads[path.as_posix()]
    monkeypatch.setattr(storage_module, '_bounded_file', read)
    links = {'/proc/100/fd/3': 'socket:[123]', '/proc/200/fd/4': 'socket:[456]',
             '/proc/100/exe': '/usr/bin/dockerd', '/proc/200/exe': '/usr/bin/containerd'}
    for pid in ('self', '100', '200'):
        for key in ('mnt', 'net', 'user'):
            links[f'/proc/{pid}/ns/{key}'] = key + ':[1]'
    monkeypatch.setattr(storage_module.os, 'readlink', lambda path: links[str(path).replace('\\', '/')])
    stats = {path: SimpleNamespace(st_dev=10, st_ino=inode, st_mode=stat.S_IFSOCK)
             for path, inode in (('/', 1), ('/proc/100/root', 1), ('/proc/200/root', 1),
                                ('/proc/100/exe', 100), ('/proc/200/exe', 200),
                                ('/var/run/docker.sock', 300), ('/run/containerd.sock', 400))}
    original_stat = os.stat
    def node(path, *args, **kwargs):
        key = str(path).replace('\\', '/')
        return stats[key] if key in stats else original_stat(path, *args, **kwargs)
    monkeypatch.setattr(storage_module.os, 'stat', node)
    records = normal_linux_plugins()
    state = {'records': records, 'payloads': payloads, 'links': links, 'stats': stats,
             'calls': [], 'after': None, 'missing_ctr': False}
    def call(argv):
        state['calls'].append(argv)
        assert argv == ('ctr', '--address', '/run/containerd.sock', '--timeout', '5s',
                        'plugins', 'list', '--detailed')
        if state['missing_ctr']:
            raise DiscoveryError(ReasonCode.DOCKER_UNAVAILABLE)
        result = fake_process_result(raw='\n'.join(state['records']).encode())
        if state['after']:
            state['after']()
        return result
    info = {
        'DockerRootDir': '/var/lib/docker', '_local_endpoint': 'unix:///var/run/docker.sock',
        'Driver': 'overlayfs', 'SecurityOptions': [],
        'DriverStatus': [['driver-type', 'io.containerd.snapshotter.v1']],
    }
    return NativePlatformProvider(), info, call, state


@pytest.mark.parametrize('imports,override,verified', [
    ('', False, True), ('imports = []\n', False, True), ('imports = []\n', True, True),
    ('imports = ["root-override.toml"]\n', False, False),
    ('imports = ["address-override.toml"]\n', True, False),
    ('imports = "other.toml"\n', True, False), ('imports = 1\n', False, False),
    ('imports = {}\n', False, False), ('imports = [1]\n', False, False),
])
def test_native_linux_containerd_imports_fail_closed(monkeypatch, imports, override, verified):
    metadata, info, call, state = native_linux_fixture(monkeypatch, imports=imports, override=override)
    layout = metadata.linux_layout(info, call)
    assert layout.verified is verified
    assert layout.roots == (NORMAL_LINUX_ROOTS if verified else ())
    assert len(state['calls']) == int(verified)


@pytest.mark.parametrize('driver', ['overlayfs', 'overlay2'])
def test_native_linux_budgets_runtime_stores_after_config_edit(monkeypatch, driver):
    metadata, info, call, state = native_linux_fixture(monkeypatch)
    if driver == 'overlay2':
        info.update(Driver=driver, DriverStatus=[['Backing Filesystem', 'extfs'], ['Supports d_type', 'true']])
    instance, docker, disks, _ = provider()
    instance.platform_provider = metadata
    docker.info.update(info)
    disks.values.update({
        ACTIVE_CONTENT: volume('active-a', 40),
        ACTIVE_METADATA: volume('active-a', 35),
        '/snapshots/live': volume('snapshot-override', 80),
        '/native/live': volume('native-snapshots', 70),
        '/opt/containerd/live': volume('opt-volume', 90),
        ACTIVE_TASK: volume('task-work', 60),
        ACTIVE_SANDBOX: volume('sandbox-work', 50),
        ACTIVE_TASK_STATE: volume('task-state', 45),
        ACTIVE_SANDBOX_STATE: volume('sandbox-state', 30),
        '/configured/store-b': volume('irrelevant-b', 950),
    })
    instance.runner = lambda argv, **kwargs: call(argv) if argv[0] == 'ctr' else docker(argv, **kwargs)
    evidence = instance.sample()
    assert evidence.can_pull and evidence.can_create_cached
    assert {c.volume.identity.value: c.volume.free.value for c in evidence.constraints} == {
        'host-volume': 600, 'active-a': 35, 'snapshot-override': 80, 'native-snapshots': 70,
        'opt-volume': 90, 'task-work': 60, 'sandbox-work': 50, 'task-state': 45, 'sandbox-state': 30}
    assert tuple(disks.paths) == NORMAL_LINUX_ROOTS
    assert '/configured/store-b' not in disks.paths and '/cli/store' not in disks.paths
    assert ACTIVE_METADATA + '/meta.db' not in disks.paths
    assert len(state['calls']) == 1
    state['payloads']['/etc/containerd/config.toml'] = (
        b'root = "/edited-again"\nstate = "/inactive-state"\n[grpc]\naddress = "/wrong-hint.sock"')
    assert instance.sample().can_pull
    assert not any(p.startswith(('/edited-again', '/inactive-state')) for p in disks.paths)


def test_native_linux_normal_task_record_adds_proved_work_and_state_roots(monkeypatch):
    metadata, info, call, state = native_linux_fixture(monkeypatch)
    state['records'] = state['records'][:3]
    before = metadata.linux_layout(info, call)
    assert before.verified
    state['records'].append('Type:    io.containerd.runtime.v2\nID:    task\nPlatforms:    linux/amd64\n')
    after = metadata.linux_layout(info, call)
    assert after.verified and after.roots == (*before.roots, ACTIVE_TASK, ACTIVE_TASK_STATE)


def test_native_linux_classic_overlay2_does_not_require_image_snapshotters(monkeypatch):
    metadata, info, call, state = native_linux_fixture(monkeypatch)
    info.update(Driver='overlay2', DriverStatus=[['Backing Filesystem', 'extfs']])
    state['records'] = [r for r in state['records'] if not r.startswith('Type:    io.containerd.snapshotter.v1\n')]
    layout = metadata.linux_layout(info, call)
    assert layout.verified
    assert layout.roots == tuple(p for p in NORMAL_LINUX_ROOTS if p not in ('/snapshots/live', '/native/live'))
    assert len(state['calls']) == 1


def test_native_linux_reviewed_legacy_builtins_budget_runtime_v1(monkeypatch):
    metadata, info, call, state = native_linux_fixture(monkeypatch)
    state['records'] += [
        'Type:    io.containerd.runtime.v1\nID:    linux\nPlatforms:    linux/amd64\n',
        'Type:    io.containerd.runtime.v2\nID:    shim\n',
        'Type:    io.containerd.monitor.v1\nID:    cgroups\n',
        'Type:    io.containerd.internal.v1\nID:    restart\n',
        'Type:    io.containerd.sandbox.controller.v1\nID:    local\n',
    ]
    layout = metadata.linux_layout(info, call)
    assert layout.verified
    assert '/runtime/store-a/io.containerd.runtime.v1.linux' in layout.roots
    assert '/run/active-containerd/io.containerd.runtime.v1.linux' in layout.roots


@pytest.mark.parametrize('path', NORMAL_LINUX_ROOTS[1:])
def test_native_linux_every_live_store_constrains_admission(monkeypatch, path):
    metadata, info, call, _ = native_linux_fixture(monkeypatch)
    instance, docker, disks, _ = provider()
    docker.info.update(info)
    instance.platform_provider = metadata
    disks.values[path] = volume('full-independent-volume', 0)
    instance.runner = lambda argv, **kwargs: call(argv) if argv[0] == 'ctr' else docker(argv, **kwargs)
    evidence = instance.sample()
    assert path in disks.paths
    assert not evidence.can_pull and not evidence.can_create_cached


@pytest.mark.parametrize('change', [
    'missing_state', 'config_only_state', 'relative_state', 'duplicate_state', 'empty_state',
    'wrong_content_suffix', 'wrong_metadata_suffix', 'different_runtime_roots',
    'task_export', 'opt_extra_export', 'opt_missing_path', 'opt_relative_path',
    'custom_service', 'custom_service_export', 'failed_custom_plugin', 'healthy_unreviewed_snapshotter',
])
def test_native_linux_unproved_builtin_or_custom_store_blocks(monkeypatch, change):
    metadata, info, call, state = native_linux_fixture(monkeypatch)
    records = state['records']
    if change in ('missing_state', 'config_only_state'):
        state['payloads']['/proc/200/cmdline'] = state['payloads']['/proc/200/cmdline'].replace(
            b'--state=/run/active-containerd\0', b'')
        if change == 'config_only_state':
            state['payloads']['/etc/containerd/config.toml'] = b'state = "/run/active-containerd"'
    elif change in ('relative_state', 'empty_state'):
        state['payloads']['/proc/200/cmdline'] = state['payloads']['/proc/200/cmdline'].replace(
            b'--state=/run/active-containerd', b'--state=' + (b'relative' if change == 'relative_state' else b''))
    elif change == 'duplicate_state':
        state['payloads']['/proc/200/cmdline'] += b'--state=/run/another\0'
    elif change == 'wrong_content_suffix':
        records[0] = records[0].replace(ACTIVE_CONTENT, '/runtime/store-a/content')
    elif change == 'wrong_metadata_suffix':
        records[1] = records[1].replace(ACTIVE_METADATA, '/runtime/store-a/metadata')
    elif change == 'different_runtime_roots':
        records[1] = records[1].replace('/runtime/store-a/', '/other-active-root/')
    elif change == 'task_export':
        index = next(i for i, r in enumerate(records) if r.startswith('Type:    io.containerd.runtime.v2\n'))
        records[index] += 'Exports:\n    root    /unreviewed/task/store\n'
    elif change.startswith('opt_'):
        records[4] = {
            'opt_extra_export': records[4] + '    root    /unreviewed/opt/store\n',
            'opt_missing_path': records[4].split('Exports:')[0],
            'opt_relative_path': records[4].replace('/opt/containerd/live', 'relative'),
        }[change]
    elif change == 'healthy_unreviewed_snapshotter':
        state['records'] = [r.split('Error:')[0] if '\nID:    devmapper\n' in r else r for r in records]
    else:
        record = 'Type:    io.containerd.service.v1\nID:    custom-service\n'
        if change == 'custom_service_export':
            record += 'Exports:\n    root    /custom/persistent\n'
        elif change == 'failed_custom_plugin':
            record += 'Error:\n    Code:    Unknown\n    Message:    unknown\n'
        records.append(record)
    assert metadata.linux_layout(info, call) == LinuxLayout((), False)


@pytest.mark.parametrize('change', [
    'missing_content', 'missing_metadata', 'missing_snapshot', 'missing_root',
    'duplicate_plugin', 'duplicate_export', 'errored_content', 'errored_snapshot',
    'custom_store', 'extra_snapshot', 'proxy_export', 'unknown_export', 'unknown_field',
    'relative', 'dotdot', 'dot', 'double_slash', 'trailing_slash', 'metadata_path',
    'malformed', 'table_format', 'oversized', 'unknown_policy',
])
def test_native_linux_runtime_exports_fail_closed(monkeypatch, change):
    metadata, info, call, state = native_linux_fixture(monkeypatch)
    records = state['records']
    if change.startswith('missing_') and change != 'missing_root':
        records.pop({'missing_content': 0, 'missing_metadata': 1, 'missing_snapshot': 2}[change])
    elif change == 'missing_root':
        records[0] = records[0].split('Exports:')[0]
    elif change == 'duplicate_plugin':
        records.append(records[0])
    elif change == 'duplicate_export':
        records[0] += '    root    /other\n'
    elif change.startswith('errored_'):
        records[0 if change == 'errored_content' else 2] += 'Error:\n    Code:    Unknown\n    Message:    failed\n'
    elif change in ('custom_store', 'extra_snapshot'):
        kind = 'io.custom.storage.v1' if change == 'custom_store' else 'io.containerd.snapshotter.v1'
        records.append('Type:    ' + kind + '\nID:    custom\nExports:\n    root    /custom\n')
    elif change == 'proxy_export':
        records[2] = records[2].replace('root    /snapshots/live', 'address    /run/proxy.sock')
    elif change == 'unknown_export':
        records[0] += '    extra    /hidden/store\n'
    elif change == 'unknown_field':
        records[0] += 'Future:    extra\n'
    elif change in ('relative', 'dotdot', 'dot', 'double_slash', 'trailing_slash'):
        bad = {'relative': 'relative', 'dotdot': '/a/../b', 'dot': '/a/./b',
               'double_slash': '/a//b', 'trailing_slash': '/a/b/'}[change]
        records[0] = records[0].replace(ACTIVE_CONTENT, bad)
    elif change == 'metadata_path':
        records[1] = records[1].replace(ACTIVE_METADATA + '/meta.db', ACTIVE_METADATA + '/')
    elif change == 'unknown_policy':
        records[1] = records[1].replace('shared', 'unknown')
    else:
        state['records'] = [{'malformed': '{"plugins": []}', 'table_format': 'TYPE ID STATUS',
                             'oversized': 'x' * 1048577}[change]]
    assert metadata.linux_layout(info, call) == LinuxLayout((), False)


@pytest.mark.parametrize('change', [
    'docker_restart', 'container_restart', 'docker_argv', 'container_argv', 'exe',
    'docker_namespace', 'container_namespace', 'network_namespace', 'user_namespace',
    'chroot', 'docker_socket', 'container_socket', 'socket_kernel_inode', 'fd_ownership',
])
def test_native_linux_runtime_identity_changes_close_admission(monkeypatch, change):
    metadata, info, call, state = native_linux_fixture(monkeypatch)
    def mutate():
        if change in ('docker_restart', 'container_restart'):
            key = '/proc/' + ('100' if change == 'docker_restart' else '200') + '/stat'
            state['payloads'][key] = state['payloads'][key][:-3] + b'999'
        elif change in ('docker_argv', 'container_argv'):
            key = '/proc/' + ('100' if change == 'docker_argv' else '200') + '/cmdline'
            state['payloads'][key] += b'--changed\0'
        elif change in ('docker_namespace', 'container_namespace', 'network_namespace', 'user_namespace'):
            key, value = {
                'docker_namespace': ('/proc/100/ns/mnt', 'mnt:[2]'),
                'container_namespace': ('/proc/200/ns/mnt', 'mnt:[2]'),
                'network_namespace': ('/proc/200/ns/net', 'net:[2]'),
                'user_namespace': ('/proc/200/ns/user', 'user:[2]'),
            }[change]
            state['links'][key] = value
        elif change == 'socket_kernel_inode':
            state['payloads']['/proc/net/unix'] = state['payloads']['/proc/net/unix'].replace(b'456', b'789')
        elif change == 'fd_ownership':
            state['links']['/proc/200/fd/4'] = 'socket:[999]'
        else:
            key = {'exe': '/proc/200/exe', 'chroot': '/proc/200/root',
                   'docker_socket': '/var/run/docker.sock', 'container_socket': '/run/containerd.sock'}[change]
            state['stats'][key].st_ino += 1
    state['after'] = mutate
    instance, docker, _, _ = provider()
    docker.info.update(info)
    instance.platform_provider = metadata
    instance.runner = lambda argv, **kwargs: call(argv) if argv[0] == 'ctr' else docker(argv, **kwargs)
    result = instance.sample()
    assert not result.can_pull and not result.can_create_cached and not result.constraints


@pytest.mark.parametrize('driver', ['overlayfs', 'overlay2'])
@pytest.mark.parametrize('change', ['config_only', 'missing_ctr', 'wrong_owner', 'proxy',
                                   'duplicate_socket', 'non_socket', 'rootless', 'unknown_driver', 'driver_status'])
def test_native_linux_requires_live_endpoint_and_supported_driver(monkeypatch, change, driver):
    metadata, info, call, state = native_linux_fixture(monkeypatch)
    if driver == 'overlay2':
        info.update(Driver=driver, DriverStatus=[['Backing Filesystem', 'extfs']])
    if change == 'config_only':
        state['payloads']['/proc/100/cmdline'] = b'/usr/bin/dockerd\0'
        state['payloads']['/etc/docker/daemon.json'] = b'{"containerd":"/run/containerd.sock"}'
    elif change == 'missing_ctr':
        state['missing_ctr'] = True
    elif change == 'wrong_owner':
        state['links']['/proc/200/fd/4'] = 'socket:[999]'
    elif change == 'proxy':
        state['payloads']['/etc/containerd/config.toml'] += b'\n[proxy_plugins.overlayfs]\naddress="/run/remote.sock"'
    elif change == 'duplicate_socket':
        state['payloads']['/proc/net/unix'] += b'0: 2 0 10000 1 1 789 /run/containerd.sock\n'
    elif change == 'non_socket':
        state['stats']['/run/containerd.sock'].st_mode = stat.S_IFREG
    elif change == 'rootless':
        info['SecurityOptions'] = ['name=rootless']
    elif change == 'driver_status':
        info['DriverStatus'] = [['driver-type', 'custom']]
    else:
        info['Driver'] = 'custom'
    instance, docker, _, _ = provider()
    docker.info.update(info)
    instance.platform_provider = metadata
    instance.runner = lambda argv, **kwargs: call(argv) if argv[0] == 'ctr' else docker(argv, **kwargs)
    evidence = instance.sample()
    assert not evidence.can_pull and not evidence.can_create_cached and not evidence.constraints


def test_native_linux_known_extra_snapshot_store_is_also_budgeted(monkeypatch):
    metadata, info, call, state = native_linux_fixture(monkeypatch)
    state['records'] = [r for r in state['records'] if '\nID:    btrfs\n' not in r]
    state['records'].append('Type:    io.containerd.snapshotter.v1\nID:    btrfs\nExports:\n    root    /btrfs/live\n')
    assert '/btrfs/live' in metadata.linux_layout(info, call).roots


def test_native_linux_deadline_covers_proc_discovery(monkeypatch):
    metadata, info, call, state = native_linux_fixture(monkeypatch)
    call.deadline = time.monotonic() - 1
    with pytest.raises(DiscoveryError) as error:
        metadata.linux_layout(info, call)
    assert error.value.reason is ReasonCode.REQUEST_TIMEOUT and not state['calls']


def test_decision_stamp_does_not_freshen_volume_samples():
    instance, _, _, _ = provider()
    later = NOW.replace(second=1)
    times = iter((NOW, NOW, later))
    instance.clock = lambda: next(times)
    result = instance.sample()
    assert result.can_pull and result.admission.sample_time == later.isoformat().replace('+00:00', 'Z')
    assert result.constraints[0].volume.free.sample_time == STAMP


def test_slow_final_context_check_cannot_admit_stale_capacity():
    instance, _, _, _ = provider()
    later = NOW.replace(second=11)
    times = iter((NOW, later, later))
    instance.clock = lambda: next(times)
    result = instance.sample()
    assert result.reason is ReasonCode.STORAGE_STALE and not result.can_pull
    assert result.constraints == ()


def timed_wsl_discovery():
    instance, docker, disks, metadata = provider('win32')
    current = [NOW]
    instance.clock = lambda: current[0]
    original_layout = metadata.wsl

    def slow_layout(info, call, manual_backing):
        current[0] = NOW + timedelta(seconds=3, milliseconds=200)
        stamp = current[0].isoformat().replace('+00:00', 'Z')
        return replace(original_layout,
                       vm_volume=volume('wsl:uuid:' + GUID, stamp=stamp))

    metadata.wsl_layout = slow_layout
    return instance, docker, disks, current


def test_verified_association_uses_layout_completion_time_for_real_plan():
    instance, _, disks, current = timed_wsl_discovery()

    def fresh_host(path):
        disks.paths.append(path)
        return volume(stamp=current[0].isoformat().replace('+00:00', 'Z'))

    disks.sample = fresh_host
    evidence = instance.sample()
    association = evidence.backing[0]
    proof_stamp = (NOW + timedelta(seconds=3, milliseconds=200)).isoformat().replace('+00:00', 'Z')
    assert evidence.can_pull and association.verified.sample_time == proof_stamp
    assert association.vm_identity.sample_time == proof_stamp
    assert association.host_identity.sample_time == proof_stamp
    assert evidence.context.sample_time == STAMP

    plan_now = NOW + timedelta(seconds=3, milliseconds=500)
    plan = make_plan((evidence.constraints[1].volume,),
                     (GrowthInterval('worker', 'host-volume', 1, 0, 0, 1),),
                     Limits(reserve_fraction=0, reserve_floor_bytes=0, reserve_ceiling_bytes=0),
                     now=plan_now, docker=evidence)
    assert plan.admitted and plan.reason is ReasonCode.OK


@pytest.mark.parametrize('slow_stage', ['host', 'context'])
def test_late_host_or_context_keeps_actual_storage_staleness(slow_stage):
    instance, docker, disks, current = timed_wsl_discovery()
    late = NOW + timedelta(seconds=14)

    def host(path):
        disks.paths.append(path)
        sampled = volume(stamp=current[0].isoformat().replace('+00:00', 'Z'))
        if slow_stage == 'host':
            current[0] = late
        return sampled

    disks.sample = host
    if slow_stage == 'context':
        def slow_runner(argv, **kwargs):
            result = docker(argv, **kwargs)
            if argv == ('docker', 'context', 'show') and docker.calls.count(argv) == 2:
                current[0] = late
            return result
        instance.runner = slow_runner
    evidence = instance.sample()
    assert evidence.reason is ReasonCode.STORAGE_STALE
    assert not evidence.can_pull and not evidence.can_create_cached
    assert evidence.constraints == () and evidence.backing == ()


def test_prepared_refresh_measures_changed_counts_and_keeps_original_evidence():
    instance, _, disks, _ = provider()
    session = instance.prepare()
    assert not session.closed and session.initial_evidence.can_pull
    old = session.initial_evidence
    disks.values['/var/lib/docker'] = replace(volume(free=200), capacity=measure(900))
    disks.values['/var/lib/containerd'] = replace(volume(free=180), capacity=measure(900))
    before = time.monotonic()
    fresh = session.refresh(deadline=before + 1)
    assert fresh.can_pull and fresh.constraints[0].volume.free.value == 180
    assert fresh.constraints[0].volume.capacity.value == 900
    assert before <= fresh.oldest_observed_monotonic <= time.monotonic()
    assert old is session.initial_evidence and old.constraints[0].volume.free.value == 600
    assert old.constraints[0].volume.capacity.value == 1000
    with pytest.raises(FrozenInstanceError):
        session._baseline.root = '/replacement'


@pytest.mark.parametrize('change', ['context', 'environment', 'root', 'layout', 'volume', 'stale', 'permission', 'timeout'])
def test_prepared_failure_invalidates_without_retargeting(change):
    instance, docker, disks, metadata = provider()
    session = instance.prepare()
    if change == 'context':
        docker.change_after_info = True
    elif change == 'environment':
        instance.environ['DOCKER_CONTEXT'] = 'elsewhere'
    elif change == 'root':
        docker.info['DockerRootDir'] = '/different'
    elif change == 'layout':
        metadata.linux = replace(metadata.linux, roots=('/var/lib/docker', '/changed'))
    elif change == 'volume':
        disks.values['/var/lib/docker'] = volume('replacement')
    elif change == 'stale':
        disks.values['/var/lib/docker'] = volume(stamp='2026-10-08T00:00:00Z')
    elif change == 'permission':
        disks.values['/var/lib/docker'] = replace(volume(), writable=measure(False))
    deadline = time.monotonic() + (-1 if change == 'timeout' else 1)
    result = session.refresh(deadline=deadline)
    assert not result.can_pull and session.closed
    assert not result.constraints and not result.backing and result.oldest_observed_monotonic is None
    if change == 'stale':
        assert result.reason is ReasonCode.STORAGE_STALE
        assert disks.values['/var/lib/docker'].identity.sample_time == '2026-10-08T00:00:00Z'
    calls = len(docker.calls)
    assert not session.refresh(deadline=time.monotonic() + 1).can_pull
    assert len(docker.calls) == calls


def test_session_close_and_failed_preparation_do_not_refresh():
    instance, docker, _, metadata = provider()
    with instance.prepare() as session:
        assert not session.closed
    calls = len(docker.calls)
    assert session.closed and not session.refresh(deadline=time.monotonic() + 1).can_pull
    assert len(docker.calls) == calls
    metadata.linux = LinuxLayout((), False)
    failed = instance.prepare()
    assert failed.closed and not failed.initial_evidence.can_pull


@pytest.mark.parametrize('deadline', [float('nan'), float('inf'), True, '1'])
def test_prepared_deadline_requires_finite_absolute_monotonic(deadline):
    with pytest.raises(ValueError):
        provider()[0].prepare().refresh(deadline=deadline)


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('window', ['docker_info', 'last_volume'])
def test_complete_linux_observation_rejects_restart_before_info_or_during_volumes(monkeypatch, prepared, window):
    metadata, info, query, state = native_linux_fixture(monkeypatch)
    instance, docker, disks, _ = provider()
    instance.platform_provider = metadata
    docker.info.update(info)
    armed = [False]
    def mutate(pid):
        key = f'/proc/{pid}/stat'
        state['payloads'][key] = state['payloads'][key][:-3] + b'999'
    def runner(argv, **kwargs):
        result = query(argv) if argv[0] == 'ctr' else docker(argv, **kwargs)
        if armed[0] and window == 'docker_info' and 'info' in argv:
            mutate('100')
        return result
    instance.runner = runner
    original_sample = disks.sample
    def sample(path):
        result = original_sample(path)
        if armed[0] and window == 'last_volume' and path == NORMAL_LINUX_ROOTS[-1]:
            mutate('200')
        return result
    disks.sample = sample
    session = instance.prepare() if prepared else None
    if session:
        assert session.initial_evidence.can_pull
    armed[0] = True
    evidence = session.refresh(deadline=time.monotonic() + 1) if session else instance.sample()
    assert not evidence.can_pull and not evidence.constraints and not evidence.backing
    assert evidence.oldest_observed_monotonic is None


def test_prepared_native_linux_keeps_all_supported_runtime_stores(monkeypatch):
    metadata, info, query, _ = native_linux_fixture(monkeypatch)
    instance, docker, disks, _ = provider()
    instance.platform_provider = metadata
    docker.info.update(info)
    instance.runner = lambda argv, **kwargs: query(argv) if argv[0] == 'ctr' else docker(argv, **kwargs)
    session = instance.prepare()
    disks.values[ACTIVE_CONTENT] = volume('live-container-volume', 250)
    # Volume identity is pinned: discover the intended physical layout first.
    session = instance.prepare()
    disks.values[ACTIVE_CONTENT] = volume('live-container-volume', 100)
    evidence = session.refresh(deadline=time.monotonic() + 1)
    assert evidence.can_pull
    assert tuple(dict.fromkeys(disks.paths)) == NORMAL_LINUX_ROOTS
    assert {c.volume.identity.value: c.volume.free.value for c in evidence.constraints}['live-container-volume'] == 100


def prepared_native_windows(monkeypatch):
    metadata, wsl_call, state = native_wsl_fixture(monkeypatch)
    instance, docker, _, _ = provider('win32')
    instance.clock = lambda: datetime.now(UTC)
    instance.platform_provider = metadata
    instance._native_volume_provider = True
    state['host_free'] = 600
    state['host_file'] = [10, 20]
    state['host_disk'] = [GUID, True, False]
    state['host_version'] = '4.93.0.240920'
    def runner(argv, **kwargs):
        if argv[0] == 'wsl.exe':
            return wsl_call(argv)
        if argv[0] == sys.executable:
            assert argv[1:4] == ('-m', 'podmuncher.docker_storage', '--native-host-probe')
            stamp = state.get('host_stamp', datetime.now(UTC).isoformat().replace('+00:00', 'Z'))
            host = volume(free=state['host_free'], stamp=stamp)
            if state.get('host_permission') is False:
                host = replace(host, writable=measure(False, stamp=stamp))
            return fake_process_result({'version': state['host_version'], 'disk': state['host_disk'],
                'file': state['host_file'], 'volume': asdict(host)})
        return docker(argv, **kwargs)
    instance.runner = runner
    session = instance.prepare()
    assert session.initial_evidence.can_pull
    return instance, docker, state, session


def test_complete_initial_native_discovery_supervises_every_wsl_call(monkeypatch):
    instance, _, state, _ = prepared_native_windows(monkeypatch)
    fixture, native_calls = instance.runner, []
    state['calls'].clear()
    def popen(argv, **kwargs):
        native_calls.append(argv)
        if argv[0] == 'wsl.exe':
            assert argv.count(storage_module._WSL_SUPERVISOR) == 1
            assert kwargs['stdin'] is subprocess.PIPE and kwargs['shell'] is False
            offset = argv.index(storage_module._WSL_ONESHOT)
            assert argv[offset + 1] == '0'
            prefix = argv[:argv.index('/bin/sh')]
            raw = (*prefix, *argv[offset + 2:])
            result = fixture(raw, deadline=time.monotonic() + 1, env={})
            child = FakeChild(result.stdout + b'@@done:0\n@@reaped\n')
            child.stdin = io.BytesIO()
            return child
        return FakeChild(fixture(argv, deadline=time.monotonic() + 1, env={}).stdout)
    instance.runner = BoundedRunner(popen=popen)
    session = instance.prepare()
    assert session.initial_evidence.can_pull
    raw_calls = state['calls']
    assert len(raw_calls) == sum(cmd[0] == 'wsl.exe' for cmd in native_calls)
    assert raw_calls[0][8] == raw_calls[-1][8] == storage_module._WSL_IDENTITY
    assert any(storage_module._WSL_METADATA in cmd for cmd in raw_calls)
    assert any('nsenter' in cmd for cmd in raw_calls)
    assert all(any(tool in cmd for cmd in raw_calls) for tool in ('findmnt', 'lsblk', 'cat'))
    first_identity = next(i for i, cmd in enumerate(native_calls) if storage_module._WSL_IDENTITY in cmd)
    assert first_identity < next(i for i, cmd in enumerate(native_calls) if 'info' in cmd)
    assert native_calls[-1][native_calls[-1].index(storage_module._WSL_ONESHOT) + 4] == storage_module._WSL_IDENTITY


def test_native_prepared_windows_rechecks_counters_and_reuses_only_choices(monkeypatch):
    _instance, docker, state, session = prepared_native_windows(monkeypatch)
    old = session.initial_evidence
    mount_calls = state['mount_calls']
    state['capacity'] = '4096 80000 60000 500'
    state['host_free'] = 300
    evidence = session.refresh(deadline=time.monotonic() + 1)
    assert evidence.can_pull
    assert evidence.constraints[0].volume.capacity.value == 4096 * 80000
    assert evidence.constraints[0].volume.free.value == 4096 * 60000
    assert evidence.constraints[1].volume.free.value == 300
    assert old.constraints[0].volume.capacity.value == 4096 * 100000
    assert old.constraints[1].volume.free.value == 600
    assert state['mount_calls'] == mount_calls
    assert sum(storage_module._WSL_BATCH in cmd for cmd in state['calls']) == 1
    assert ('docker', 'context', 'inspect') in docker.calls


@pytest.mark.parametrize('change', [
    'quota', 'hidden_quota', 'quota_inode', 'kernel', 'sysfs', 'changed_device', 'changed_uuid',
    'uid', 'group', 'capability', 'namespace', 'user_namespace', 'label', 'inode',
    'starttime', 'exe', 'permission', 'parent', 'stale_vpd', 'host_file', 'host_id',
    'detached', 'remote', 'host_permission', 'host_stale', 'version', 'context', 'environment',
])
def test_native_prepared_windows_changes_invalidate_all_evidence(monkeypatch, change):
    instance, docker, state, session = prepared_native_windows(monkeypatch)
    if change == 'host_file':
        state['host_file'] = [10, 21]
    elif change == 'host_id':
        state['host_disk'][0] = '11111111-1111-1111-1111-111111111111'
    elif change == 'detached':
        state['host_disk'][1] = False
    elif change == 'remote':
        state['host_disk'][2] = True
    elif change == 'host_permission':
        state['host_permission'] = False
    elif change == 'host_stale':
        state['host_stamp'] = '2026-10-08T00:00:00Z'
    elif change == 'version':
        state['host_version'] = '4.94.0'
    elif change == 'context':
        docker.selected = 'changed'
    elif change == 'environment':
        instance.environ['DOCKER_HOST'] = 'tcp://remote'
    else:
        state['change'] = change
    result = session.refresh(deadline=time.monotonic() + 1)
    assert not result.can_pull and session.closed and not result.constraints and not result.backing
    assert result.oldest_observed_monotonic is None


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('window', ['info', 'host_volume'])
def test_complete_windows_observation_brackets_daemon_and_host(monkeypatch, prepared, window):
    instance, _, state, session = prepared_native_windows(monkeypatch)
    runner = instance.runner
    def changed(argv, **kwargs):
        result = runner(argv, **kwargs)
        if (window == 'info' and 'info' in argv
                or window == 'host_volume' and '--native-host-probe' in argv):
            state['epoch'] = '999'
        return result
    instance.runner = changed
    evidence = session.refresh(deadline=time.monotonic() + 1) if prepared else instance.sample()
    assert not evidence.can_pull and not evidence.constraints


def test_prepared_explicit_context_stays_named_and_env_is_pinned(monkeypatch):
    instance, docker, _state, _old = prepared_native_windows(monkeypatch)
    instance.environ['DOCKER_CONTEXT'] = 'default'
    instance.environ['DOCKER_HOST'] = docker.endpoint
    session = instance.prepare()
    docker.calls.clear()
    assert session.refresh(deadline=time.monotonic() + 1).can_pull
    assert ('docker', 'context', 'inspect', 'default') in docker.calls
    assert ('docker', 'context', 'inspect') not in docker.calls
    instance.environ['DOCKER_CONTEXT'] = 'changed'
    assert not session.refresh(deadline=time.monotonic() + 1).can_pull


def test_parallel_runner_timeout_terminates_and_reaps_all_children_inside_budget():
    children = []
    def spawn(*args, **kwargs):
        child = subprocess.Popen(*args, **kwargs)
        children.append(child)
        return child
    runner = BoundedRunner(popen=spawn)
    started = time.monotonic()
    command = (sys.executable, '-c', 'import time; time.sleep(10)')
    with pytest.raises(DiscoveryError) as error:
        runner.parallel((command, command), deadline=started + 0.5, env=dict(os.environ))
    assert error.value.reason is ReasonCode.REQUEST_TIMEOUT
    assert time.monotonic() - started <= 0.5
    assert len(children) == 2 and all(child.poll() is not None for child in children)


@pytest.mark.parametrize('frame', [b'', b'@@ready\n', b'@@identity_before\n@@ready\n@@ready\n'])
def test_staged_witness_rejects_missing_or_duplicate_frames(frame):
    with pytest.raises((DiscoveryError, ValueError)):
        storage_module._witness_identity(frame, b'before')


def test_prepared_batch_rejects_output_overflow_without_reissuing_evidence(monkeypatch):
    instance, _, _, session = prepared_native_windows(monkeypatch)
    runner = instance.runner
    def oversized(argv, **kwargs):
        if storage_module._WSL_BATCH in argv:
            return ProcessResult(0, b'x' * 524289)
        return runner(argv, **kwargs)
    instance.runner = oversized
    result = session.refresh(deadline=time.monotonic() + 1)
    assert result.reason is ReasonCode.REQUEST_TOO_LARGE and session.closed
    assert not result.constraints and not result.backing


@pytest.mark.parametrize('failure', [None, 'before', 'sample', 'after', 'overflow', 'early', 'eof'])
def test_staged_transport_owns_all_processes_and_brackets_observation(monkeypatch, failure):
    _, call, _ = native_wsl_fixture(monkeypatch)
    identity = call((*storage_module._WSL_PREFIX, '/bin/sh', '-c',
                     storage_module._WSL_IDENTITY, 'storage-identity')).stdout
    expected = storage_module._identity_record(storage_module._proof_sections(
        ProcessResult(0, identity), storage_module._IDENTITY_NAMES))
    script = ('import sys,time\n'
        + ('time.sleep(10)\n' if failure == 'before' else '')
        + ('sys.stdout.write("x" * 2000000); sys.stdout.flush()\n' if failure == 'overflow' else '')
        + 'sys.stdout.buffer.write(' + repr(b'@@identity_before\n' + identity + b'@@ready\n'
          + (b'unexpected\n' if failure == 'early' else b'')) + '); sys.stdout.flush()\n'
        + ('sys.exit(0)\n' if failure == 'eof' else '')
        + 'assert sys.stdin.readline() == "continue\\n"\n'
        + ('time.sleep(10)\n' if failure == 'sample' else '')
        + 'sys.stdout.buffer.write(b"@@before\\nmetadata\\n@@sampled\\n"); sys.stdout.flush()\n'
        + 'assert sys.stdin.readline() == "continue\\n"\n'
        + ('time.sleep(10)\n' if failure == 'after' else '')
        + 'sys.stdout.buffer.write(' + repr(b'@@identity_after\n' + identity) + '); sys.stdout.flush()\n')
    children, observations = [], []
    def spawn(*args, **kwargs):
        child = subprocess.Popen(*args, **kwargs)
        children.append(child)
        return child
    def validate(context, proof):
        assert context.stdout == b'context' and proof == expected
        observations.append(time.monotonic())
    runner = BoundedRunner(popen=spawn)
    context = (sys.executable, '-c', 'import sys; sys.stdout.write("context")')
    info = (sys.executable, '-c', 'import sys,time; time.sleep(0.02); sys.stdout.write("info")')
    host = (sys.executable, '-c', 'import sys,time; time.sleep(0.03); sys.stdout.write("host")')
    witness = (sys.executable, '-c', script)
    started = time.monotonic()
    if failure:
        with pytest.raises(DiscoveryError) as error:
            runner.observation(context, info, host, witness, deadline=started + 0.5,
                               env=dict(os.environ), validate=validate)
        assert error.value.reason in {ReasonCode.REQUEST_TIMEOUT, ReasonCode.REQUEST_TOO_LARGE,
                                      ReasonCode.STORAGE_UNKNOWN}
    else:
        result = runner.observation(context, info, host, witness, deadline=started + 0.5,
                                    env=dict(os.environ), validate=validate)
        assert tuple(item.stdout for item in result) == (b'info', b'@@before\nmetadata\n', b'host')
        assert len(observations) == 2 and observations[1] > observations[0]
    assert time.monotonic() - started <= 0.5
    assert all(child.poll() is not None for child in children)


_BENIGN_WSL_TREE = r'''
trap '' HUP INT TERM
record_task() {
    IFS= read -r row < "/proc/$1/stat"
    pid=$1; row=${row##*) }; set -- $row; session=$4; shift 19
    printf '@task %s %s %s\n' "$pid" "$1" "$session"
}
record_task "$$"
stage=waiting
trap 'stage=continue' USR1
advance() { while [ "$stage" = waiting ]; do sleep 0.002; done; stage=waiting; }
if [ "$1" != before ]; then printf '@@stage1\n'; advance; fi
if [ "$1" = after ]; then printf '@@stage2\n'; advance; fi
/bin/sh -c '
trap "" HUP INT TERM
sleep 30 &
leaf=$!
for pid in "$$" "$leaf"; do
    IFS= read -r row < "/proc/$pid/stat"
    row=${row##*) }; set -- $row; session=$4; shift 19
    printf "@task %s %s %s\n" "$pid" "$1" "$session"
done
printf "@@tree\n"
wait
' &
tree=$!
if [ "$1" = overflow ]; then advance; dd if=/dev/zero bs=65536 count=32; fi
wait "$tree"
'''

_CHECK_OWNED_WSL_TASKS = r'''
mode=$1
shift
while [ "$#" -gt 0 ]; do
    pid=$1; expected=$2; expected_session=$3; shift 3
    if IFS= read -r row < "/proc/$pid/stat" 2>/dev/null; then
        row=${row##*) }
        saved=$*
        set -- $row
        session=$4; shift 19
        start=$1
        set -- $saved
        if [ "$start" = "$expected" ]; then
            if [ "$mode" = cleanup ] && [ "$pid" = "$expected_session" ] &&
               [ "$session" = "$expected_session" ]; then
                kill -KILL "-$session"
            else
                exit 1
            fi
        fi
    fi
done
if [ "$mode" = verify ]; then
    for node in /proc/[0-9]*; do
        IFS= read -r row < "$node/stat" 2>/dev/null || continue
        row=${row##*) }; set -- $row
        [ "$4" != "$expected_session" ] || exit 1
    done
fi
'''


_BENIGN_WSL_COMPLETION = r'''
released=0; stage=waiting
trap 'released=1' USR2
trap 'stage=continue' USR1
record_task() {
    IFS= read -r row < "/proc/$1/stat"
    pid=$1; row=${row##*) }; set -- $row; session=$4; shift 19
    printf '@task %s %s %s\n' "$pid" "$1" "$session"
}
record_task "$$"
sleep 30 &
record_task "$!"
if [ "$1" = 2 ]; then
    for phase in 1 2; do
        printf '@@stage%s\n' "$phase"
        while [ "$stage" = waiting ]; do sleep 0.002; done
        stage=waiting
    done
fi
kill -USR2 "$controller"
while [ "$released" = 0 ]; do sleep 0.002; done
printf '\000\203\377native-completion\n'
'''


def _completion_window_supervisor(target_phase, partial=False):
    # An in-memory instrumentation seam, never a replacement supervisor. The
    # worker cannot finish until released here, after the pre-read flag check.
    # Its real USR1 must then arrive before the unchanged production read.
    setup = 'owned=; anchor=; finished=0\n'
    read = '    if IFS= read -r -n 1 -d \'\' -t 0.01 byte; then\n'
    rendezvous = r'''
    if [ "$phase" = TARGET_PHASE ] && [ "$stage" = 'PENDING_STAGE' ] && [ "$window_forced" = 0 ]; then
        while [ "$window_ready" = 0 ]; do
            tick && [ "$now" -lt "$end" ] || exit 1
            sleep 0.002
        done
        [ "$finished" = 0 ] || exit 1
        printf '@@window-before:%s\n' "$finished" >&2
        kill -USR2 "$owned" || exit 1
        while [ "$finished" = 0 ]; do
            tick && [ "$now" -lt "$end" ] || exit 1
            sleep 0.002
        done
        printf '@@window-finished:%s\n' "$finished" >&2
        window_forced=1
    fi
'''.replace('TARGET_PHASE', str(target_phase)).replace('PENDING_STAGE', 'continue' if partial else '')
    script = storage_module._WSL_SUPERVISOR
    assert script.count(setup) == script.count(read) == 1
    script = script.replace(setup, setup +
        "window_ready=0; window_forced=0\ntrap 'window_ready=1' USR2\n")
    return script.replace(read, rendezvous + read), rendezvous


@pytest.mark.parametrize('phase', [0, 2])
def test_native_completion_regression_seam_preserves_production_supervisor(phase):
    # This checks seam integrity only; the registered live smoke executes the
    # signal handshake and proves prompt success with native ash and processes.
    script, rendezvous = _completion_window_supervisor(phase)
    assert script.count(rendezvous) == 1
    assert script.replace(rendezvous, '').replace(
        "window_ready=0; window_forced=0\ntrap 'window_ready=1' USR2\n", '') == storage_module._WSL_SUPERVISOR
    assert rendezvous.endswith('    fi\n')
    assert script.split(rendezvous)[1].startswith('    if IFS= read -r -n 1 ')


def _interrupted_completion_supervisor(target_phase, eof=False):
    # Model only the native-probed interrupted-read return (1, unchanged
    # sentinel). This does NOT claim to interrupt a real read. Completion is
    # still a real worker signal; the following poll uses unchanged native ash.
    script, rendezvous = _completion_window_supervisor(target_phase)
    eof_ready = r'''
        # Establish closed-pipe ordering before injecting the return. EOF is
        # persistent; the next production read must independently reject it.
        while :; do
            tick && [ "$now" -lt "$end" ] || exit 1
            eof_byte=waiting
            if IFS= read -r -n 1 -d '' -t 0.01 eof_byte; then exit 1; else eof_status=$?; fi
            [ "$eof_status" = 1 ] || exit 1
            [ -n "$eof_byte" ] || break
            [ "$eof_byte" = waiting ] || exit 1
        done
        printf '@@interrupted-eof-ready\n' >&2
'''
    injection = rendezvous.replace(
        '        [ "$finished" = 0 ] || exit 1\n',
        (eof_ready if eof else '') + '        [ "$finished" = 0 ] || exit 1\n')
    injection = injection.replace('        window_forced=1\n', r'''
        window_forced=1
        [ "$byte" = waiting ] || exit 1
        printf '@@interrupted-return:%s:%s:1:%s\n' "$completion_seen" "$finished" "$byte" >&2
        return 1
''') + r'''
    if [ "$window_forced" = 1 ]; then
        printf '@@interrupted-native-poll:%s:%s\n' "$completion_seen" "$finished" >&2
        window_forced=2
    fi
'''
    return script.replace(rendezvous, injection), injection


@pytest.mark.parametrize('phase,eof', [(0, False), (2, False), (2, True)])
def test_interrupted_completion_regression_seam_preserves_production_supervisor(phase, eof):
    # Structural seam integrity only. The registered native smoke supplies
    # behavioral proof for the injected outcome and subsequent native read.
    script, injection = _interrupted_completion_supervisor(phase, eof)
    assert script.count(injection) == 1
    assert script.replace(injection, '').replace(
        "window_ready=0; window_forced=0\ntrap 'window_ready=1' USR2\n", '') == storage_module._WSL_SUPERVISOR
    assert injection.endswith('    fi\n')
    assert script.split(injection)[1].startswith('    if IFS= read -r -n 1 ')


def _native_wsl_completion_wait_smoke(*, interrupted=False):
    cases = ((0, None), (2, None), (2, 'early'), (2, 'stage-partial'),
             (2, 'partial'), (2, 'nul'), (2, 'eof'))
    if interrupted:
        cases = ((0, None), (2, None), (2, 'partial'), (2, 'nul'), (2, 'eof'))
    for stages, failure in cases:
        trace = bytearray()
        class CapturedErrors:
            def __init__(self, stream):
                self.stream = stream
            def read(self, size):
                chunk = self.stream.read1(size)
                trace.extend(chunk)
                return chunk
            read1 = read
            def close(self):
                self.stream.close()
        early = failure in ('early', 'stage-partial')
        if interrupted:
            supervisor, _ = _interrupted_completion_supervisor(stages, failure == 'eof')
        else:
            supervisor, _ = _completion_window_supervisor(0 if early else stages, failure == 'stage-partial')
        def spawn(argv, **kwargs):
            assert argv.count(storage_module._WSL_SUPERVISOR) == 1
            argv = tuple(supervisor if value == storage_module._WSL_SUPERVISOR else value for value in argv)
            child = subprocess.Popen(argv, **kwargs)
            child.stderr = CapturedErrors(child.stderr)
            return child
        started = time.monotonic()
        group = storage_module._OwnedProcesses(BoundedRunner(popen=spawn), started + 1, os.environ)
        key = group.spawn((*storage_module._WSL_PREFIX, '/bin/sh', '-c',
            storage_module._WSL_SUPERVISOR, 'storage-supervisor', '70',
            storage_module._WSL_LEADER + _BENIGN_WSL_COMPLETION, str(stages),
            '0' if early else str(stages)), interactive=True)
        tasks, cleaned = [], False
        try:
            if failure == 'stage-partial':
                group.children[key].stdin.write(b'continue')
                group.children[key].stdin.flush()
            if stages and not early:
                for phase in (1, 2):
                    group.wait(lambda: f'@@stage{phase}\n'.encode() in group.output(key))
                    extra = {'partial': b'partial', 'nul': b'\0'}.get(failure, b'') if phase == 2 else b''
                    group.children[key].stdin.write(b'continue\n' + extra)
                    group.children[key].stdin.flush()
                    if phase == 2 and failure == 'eof':
                        group.children[key].stdin.close()
            if failure:
                with pytest.raises(DiscoveryError) as error:
                    group.wait(lambda: group.complete(key))
                assert error.value.reason is ReasonCode.DOCKER_UNAVAILABLE
            else:
                group.wait(lambda: group.complete(key))
                result = storage_module._wsl_command_result(group.result(key))
                assert result.returncode == 0 and result.stdout.endswith(b'\0\x83\xffnative-completion\n')
                assert storage_module._witness_finished(group.output(key)) == result.stdout
        finally:
            tasks = [tuple(line.split()[1:]) for line in group.output(key).decode(errors='replace').splitlines()
                     if line.startswith('@task ')]
            cleaned = group.close()
            elapsed = time.monotonic() - started
            if not cleaned and tasks:
                BoundedRunner()((*storage_module._WSL_PREFIX, '/bin/sh', '-c',
                    _CHECK_OWNED_WSL_TASKS, 'storage-check', 'cleanup', *tasks[0]),
                    deadline=time.monotonic() + 1, env=os.environ)
        assert cleaned and group.children[key].poll() == (1 if failure else 0)
        assert elapsed < 0.5, (stages, failure, elapsed)
        assert [line for line in bytes(trace).splitlines() if line.startswith(b'@@window-')] == [
            b'@@window-before:0', b'@@window-finished:1']
        if interrupted:
            assert [line for line in bytes(trace).splitlines() if line.startswith(b'@@interrupted-')] == (
                ([b'@@interrupted-eof-ready'] if failure == 'eof' else []) + [
                    b'@@interrupted-return:0:1:1:waiting', b'@@interrupted-native-poll:1:1'])
        assert len(tasks) == 2 and all(len(t) == 3 and all(v.isdecimal() for v in t) for t in tasks)
        assert tasks[0][0] == tasks[0][2] and tasks[0][2] == tasks[1][2]
        assert group.output(key).endswith(b'@@done:0\n@@reaped\n')
        result = BoundedRunner()((*storage_module._WSL_PREFIX, '/bin/sh', '-c',
            _CHECK_OWNED_WSL_TASKS, 'storage-check', 'verify', *(v for task in tasks for v in task)),
            deadline=time.monotonic() + 1, env=os.environ)
        assert result.returncode == 0
        mode = 'Injected interrupted-read outcome; next poll native' if interrupted else 'Native completion window'
        print(f'{mode}: stages={stages}; failure={failure}; '
              f'finished=1 before native read; elapsed={elapsed:.6f}s; reaped=True')


_BENIGN_INITIAL_WSL_TREE = r'''
failure=$1
trap '' HUP INT TERM
record_task() {
    IFS= read -r row < "/proc/$1/stat"
    pid=$1; row=${row##*) }; set -- $row; session=$4; shift 19
    printf '@task %s %s %s\n' "$pid" "$1" "$session"
}
IFS= read -r row < "/proc/$$/stat"
row=${row##*) }; set -- $row; leader=$2
record_task "$leader"
record_task "$$"
ready=0
trap 'ready=1' USR1
/bin/sh -c '
trap "" HUP INT TERM
sleep 30 &
leaf=$!
for pid in "$$" "$leaf"; do
    IFS= read -r row < "/proc/$pid/stat"
    row=${row##*) }; set -- $row; session=$4; shift 19
    printf "@task %s %s %s\n" "$pid" "$1" "$session"
done
kill -USR1 "$PPID"
wait
' &
while [ "$ready" = 0 ]; do sleep 0.002; done
if [ "$failure" = overflow ]; then dd if=/dev/zero bs=65536 count=32; fi
if [ "$failure" = failure ]; then exit 17; fi
wait
'''


def _native_initial_wsl_cleanup_smoke():
    # Exercise sample -> initial identity -> BoundedRunner.parallel. Only the
    # fixed identity command is substituted; supervisor, leader, argv routing,
    # deadline, readers, footer checks and cleanup are the production path.
    for failure in ('timeout', 'overflow', 'failure'):
        trace, children = {'prefix': b'', 'tail': b''}, []
        class CapturedOutput:
            def __init__(self, stream):
                self.stream = stream
            def read(self, size):
                chunk = self.stream.read1(size)
                trace['prefix'] = (trace['prefix'] + chunk)[:4096]
                trace['tail'] = (trace['tail'] + chunk)[-128:]
                return chunk
            read1 = read
            def close(self):
                self.stream.close()
        def spawn(argv, **kwargs):
            if argv[0] == 'wsl.exe':
                offset = argv.index(storage_module._WSL_ONESHOT) + 2
                assert argv[offset:] == ('/bin/sh', '-c', storage_module._WSL_IDENTITY, 'storage-identity')
                assert argv.count(storage_module._WSL_SUPERVISOR) == 1
                argv = (*argv[:offset], '/bin/sh', '-c', _BENIGN_INITIAL_WSL_TREE,
                        'storage-initial-test', failure)
            child = subprocess.Popen(argv, **kwargs)
            children.append(child)
            if argv[0] == 'wsl.exe':
                child.stdout = CapturedOutput(child.stdout)
            return child
        tasks, acknowledged = [], False
        started = time.monotonic()
        try:
            evidence = make_docker_storage_provider(
                runner=BoundedRunner(popen=spawn, max_output=65536), timeout=1).sample()
        finally:
            elapsed = time.monotonic() - started
            tasks = [tuple(line.split()[1:]) for line in trace['prefix'].decode(errors='replace').splitlines()
                     if line.startswith('@task ')]
            acknowledged = trace['tail'].endswith(b'@@reaped\n')
            if not acknowledged and tasks:
                BoundedRunner()((*storage_module._WSL_PREFIX, '/bin/sh', '-c',
                    _CHECK_OWNED_WSL_TASKS, 'storage-check', 'cleanup', *tasks[0]),
                    deadline=time.monotonic() + 1, env=os.environ)
        assert acknowledged and elapsed <= 1
        assert len(tasks) == 4 and all(len(t) == 3 and all(v.isdecimal() for v in t) for t in tasks)
        assert len({t[0] for t in tasks}) == 4 and len({t[2] for t in tasks}) == 1
        assert tasks[0][0] == tasks[0][2]
        assert all(child.poll() is not None for child in children)
        assert not evidence.can_pull and not evidence.constraints and not evidence.backing
        assert evidence.reason in ({ReasonCode.REQUEST_TOO_LARGE} if failure == 'overflow'
            else {ReasonCode.DOCKER_UNAVAILABLE, ReasonCode.REQUEST_TIMEOUT})
        result = BoundedRunner()((*storage_module._WSL_PREFIX, '/bin/sh', '-c',
            _CHECK_OWNED_WSL_TASKS, 'storage-check', 'verify', *(v for task in tasks for v in task)),
            deadline=time.monotonic() + 1, env=os.environ)
        assert result.returncode == 0
        print(f'Initial WSL cleanup: failure={failure}; elapsed={elapsed:.6f}s; tasks={tasks}; reaped=True')


def _native_wsl_descendant_cleanup_smoke():
    # Use the production supervisor and real WSL processes, with no Docker
    # writes. Every descendant is confined to its freshly verified session.
    for phase, failure in (('before', 'eof'), ('sample', 'cancel'),
                           ('after', 'malformed'), ('sample', 'timeout'),
                           ('overflow', 'overflow')):
        started = time.monotonic()
        deadline = started + 1
        worker = storage_module._WSL_WORKER.replace(storage_module._WSL_OBSERVATION, _BENIGN_WSL_TREE)
        group = storage_module._OwnedProcesses(BoundedRunner(max_output=65536), deadline, os.environ)
        key = group.spawn((*storage_module._WSL_PREFIX, '/bin/sh', '-c',
            storage_module._WSL_SUPERVISOR, 'storage-supervisor', '60', worker, '2', phase), interactive=True)
        tasks, cleaned = [], False
        try:
            if phase != 'before':
                group.wait(lambda: b'@@stage1\n' in group.output(key))
                group.signal(key)
            if phase == 'after':
                group.wait(lambda: b'@@stage2\n' in group.output(key))
                group.signal(key)
            group.wait(lambda: b'@@tree\n' in group.output(key))
            tasks = [tuple(line.split()[1:]) for line in group.output(key).decode().splitlines()
                     if line.startswith('@task ')]
            assert len(tasks) == 3 and all(len(t) == 3 and all(v.isdecimal() for v in t) for t in tasks)
            assert len({t[2] for t in tasks}) == 1 and tasks[0][0] == tasks[0][2]
            if failure == 'malformed':
                group.children[key].stdin.write(b'malformed\n')
                group.children[key].stdin.flush()
            if failure == 'overflow':
                group.signal(key)
            if failure in ('timeout', 'malformed', 'overflow'):
                with pytest.raises(DiscoveryError) as error:
                    group.wait(lambda: False)
                assert error.value.reason is (ReasonCode.REQUEST_TOO_LARGE if failure == 'overflow'
                                              else ReasonCode.DOCKER_UNAVAILABLE)
        finally:
            cleaned = group.close()
            if not cleaned and tasks:
                # Test failure must still leave no injected stall. This fallback
                # checks the exact leader start/session before one owned kill.
                BoundedRunner()((*storage_module._WSL_PREFIX, '/bin/sh', '-c',
                    _CHECK_OWNED_WSL_TASKS, 'storage-check', 'cleanup', *tasks[0]),
                    deadline=time.monotonic() + 1, env=os.environ)
        assert cleaned and group.children[key].poll() is not None
        assert time.monotonic() <= deadline
        result = BoundedRunner()((*storage_module._WSL_PREFIX, '/bin/sh', '-c',
            _CHECK_OWNED_WSL_TASKS, 'storage-check', 'verify', *(v for task in tasks for v in task)),
            deadline=time.monotonic() + 1, env=os.environ)
        assert result.returncode == 0
        print(f'Owned WSL cleanup: phase={phase}; failure={failure}; tasks={tasks}; reaped=True')


@pytest.mark.live
def test_native_read_only_docker_storage_smoke():
    # No fake transport, skips, image commands or marker emitted before checks.
    evidence = make_docker_storage_provider(timeout=60).sample()
    if evidence.can_pull:
        if sys.platform == 'win32':
            _native_wsl_completion_wait_smoke()
            _native_wsl_completion_wait_smoke(interrupted=True)
            _native_initial_wsl_cleanup_smoke()
            _native_wsl_descendant_cleanup_smoke()
        assert evidence.can_create_cached and evidence.constraints
        for item in evidence.constraints:
            assert item.volume.capacity.value > 0 and item.volume.free.value > 0
        if sys.platform == 'win32':
            assert evidence.desktop_version.value and evidence.backing
            assert evidence.backing[0].verified.value is True
            assert {'docker_vm', 'docker_backing'} <= {r for c in evidence.constraints for r in c.roles}
        elif sys.platform == 'darwin':
            # Docker Desktop for Mac: one docker_data constraint on the Docker.raw host volume.
            assert evidence.data_root.value == '/var/lib/docker' and not evidence.backing
            assert [c.roles for c in evidence.constraints] == [('docker_data',)]
            assert evidence.constraints[0].volume.identity.value.startswith('posix:')
            print('Docker Desktop version: ' + str(evidence.desktop_version.value))
        provider = make_docker_storage_provider(timeout=60)
        with provider.prepare() as session:
            assert session.initial_evidence.can_pull
            for _ in range(5):
                started = time.monotonic()
                fresh = session.refresh(deadline=started + 1)
                assert fresh.can_pull, fresh.reason
                state = provider.volumes.sample(str(Path.cwd()), sqlite_state=True)
                roles = {role: c.volume.identity.value for c in fresh.constraints for role in c.roles}
                vm = roles.get('docker_vm', roles.get('docker_data'))
                backing = roles.get('docker_backing', vm)
                host = state.identity.value
                placement = StoragePlacement(host, host, host, host, vm, backing)
                growth = WorkloadGrowth(0, 0, 0, 0, (0,), 0, 0, 0, 0, (), 0, 0)
                now = datetime.now(UTC)
                plan = make_workload_plan((state,), growth, placement, Limits(), now=now,
                    docker=fresh, state_evidence=state,
                    inode_optional_identities=frozenset((host, backing)) if sys.platform == 'win32' else frozenset())
                elapsed = time.monotonic() - started
                print(f'Prepared complete admission: {elapsed:.6f}s; admitted={plan.admitted}')
                assert elapsed <= 1 and plan.admitted
                assert started <= fresh.oldest_observed_monotonic <= time.monotonic()
                fields = [fresh.context, fresh.data_root, fresh.admission,
                          *(value for c in fresh.constraints for value in asdict(c.volume).values())]
                # Inspect the real measurements (asdict preserves their values),
                # never manufacture fresh wrappers around discovery evidence.
                for field in fields:
                    status = field['status'] if isinstance(field, dict) else field.status
                    stamp = field['sample_time'] if isinstance(field, dict) else field.sample_time
                    if status == FieldStatus.OK:
                        assert 0 <= (now - datetime.fromisoformat(stamp)).total_seconds() <= 3
                assert all(0 <= (now - datetime.fromisoformat(
                    association.verified.sample_time)).total_seconds() <= 3
                    for association in fresh.backing)
    else:
        assert not evidence.can_create_cached
        assert evidence.reason in {ReasonCode.DOCKER_UNAVAILABLE, ReasonCode.DOCKER_CONTEXT_UNSUPPORTED,
                                   ReasonCode.STORAGE_UNKNOWN, ReasonCode.STORAGE_STALE,
                                   ReasonCode.BACKING_UNRESOLVED, ReasonCode.REQUEST_TIMEOUT,
                                   ReasonCode.REQUEST_TOO_LARGE}
        print('Docker storage diagnostic: ' + evidence.reason.value)
    print('PODMUNCHER_SMOKE_OK')
