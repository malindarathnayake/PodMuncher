"""Bounded host and identity-bound, already-admitted worker hardware probes.

Runtime callbacks are trusted adapters: they must honor the supplied monotonic
absolute deadline and output bound. No callback is started in a detached thread.
"""

import asyncio
import json
import math
import os
import platform as host_platform
import re
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

import psutil

from .docker_storage import (
    BoundedRunner,
    DiscoveryError,
    ProcessResult,
    _local_endpoint,
)
from .domains import Backend, FieldStatus, ReasonCode
from .snapshots import Measurement

_LABEL = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z')
_UUID = re.compile(r'GPU-[a-fA-F0-9-]{16,64}\Z')
_MAX = 2**63 - 1
_DOCKER_EVIDENCE_MAX_BYTES = 65536
_DOCKER_SMALL_EVIDENCE_MAX_BYTES = 8192
_GPU_ARGV = ('nvidia-smi', '--id=0',
             '--query-gpu=index,uuid,name,driver_version,memory.total,memory.free,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits')
# Darwin host identity. sysctl.proc_translated is 1 when this Python runs under
# Rosetta 2, in which case platform.machine() reports x86_64 on an arm64 host.
_SYSCTL_TRANSLATED_ARGV = ('sysctl', '-n', 'sysctl.proc_translated')
_SYSCTL_CPU_BRAND_ARGV = ('sysctl', '-n', 'machdep.cpu.brand_string')
# Engine facts come from the daemon the product uses, never from the host.
_ENGINE_INFO_FORMAT = '{{.Architecture}} {{.MemTotal}} {{.NCPU}}'
_ARCH_ALIASES = {'x86_64': 'amd64', 'amd64': 'amd64', 'arm64': 'arm64', 'aarch64': 'arm64'}
_CPU_LABEL = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9 ._+()/,@-]*')
_UNMEASURED_STAMP = '1970-01-01T00:00:00Z'
# Docker Desktop for Mac keeps the engine in a VM with no host-side nsenter
# route. The darwin reader runs inside the admitted worker's private cgroup
# namespace: the worker's own cgroup is that view's root, so its memory.max and
# memory.current are the leaf values, and /proc/meminfo is the VM's physical
# memory. Ancestors above the namespace root are not observable there; Docker
# Desktop applies no memory limit to them, and the VM's MemAvailable still
# bounds headroom.
_NAMESPACED_SCRIPT = r'''
set -eu
root=/sys/fs/cgroup
membership() { head -c 4096 /proc/self/cgroup; }
[ "$(membership)" = 0::/ ]
[ ! -L "$root" ]
[ -d "$root" ]
case " $(head -c 4096 "$root/cgroup.controllers") " in *' memory '*) :;; *) exit 1;; esac
physical() {
    awk '/^MemTotal:/ {t=$2; tc++; if ($3 != "kB") bad=1}
         /^MemAvailable:/ {a=$2; ac++; if ($3 != "kB") bad=1}
         END {if (bad || tc != 1 || ac != 1) exit 1; printf "physical\t%s\t%s\n", t, a}' /proc/meminfo
}
printf 'namespaced-v1\n'
for pass in 1 2; do
    printf 'sample\t%s\t%s\t%s\t%s\t%s\n' "$pass" "$(membership)" "$(stat -c '%d:%i:%f' "$root")" \
        "$(head -c 64 "$root/memory.max")" "$(head -c 64 "$root/memory.current")"
    physical
done
'''
# These fixed readers run in the daemon's mount/PID view, retaining the
# observer's cgroup namespace. Only validated PIDs and the full owned ID enter argv.
_DAEMON_SCRIPT = r'''
set -eu
reader=$1; shift
daemon=
for d in /proc/[0-9]*; do
    name=$(cat "$d/comm" 2>/dev/null) || { [ ! -d "$d" ] && continue; exit 1; }
    [ "$name" = dockerd ] || continue
    [ -z "$daemon" ] || exit 1
    case "$(readlink "$d/exe")" in */dockerd) :;; *) exit 1;; esac
    daemon=${d##*/}
done
[ -n "$daemon" ]
ns=$(readlink /proc/self/ns/cgroup)
[ "$ns" = "$(readlink /proc/1/ns/cgroup)" ]
[ "$ns" = "$(readlink /proc/$daemon/ns/cgroup)" ]
inner=$(awk '/^NSpid:/ {print $NF}' /proc/$daemon/status)
case "$inner" in ''|*[!0-9]*) exit 1;; esac
line=$(cat /proc/$daemon/stat); fields=${line##*) }
started=$(printf '%s\n' "$fields" | awk '{print $20}')
exec nsenter -t "$daemon" --mount --pid --root --wdns=/ \
    sh -c "$reader" cgroup-reader "$ns" "$inner" "$started" "$@"
'''

_CGROUP_SCRIPT = r'''
set -efu
expected_ns=$1; daemon=$2; daemon_start=$3; init=$4; container=$5; shift 5
pids="$*"
root=/sys/fs/cgroup
# cgroup2 cannot contain user-created links. Verify the filesystem, reject
# covering mounts, then pin each directory and file with descriptor/lstat
# identity checks before and after reads; never traverse an unchecked parent.
ident() { stat -c '%d:%i:%f' "$1"; }
fdident() { stat -L -c '%d:%i:%f' "/proc/self/fd/$1"; }
read_file() {
    file=$1
    [ ! -L "$file" ] && [ -f "$file" ]
    old=$(ident "$file")
    exec 5< "$file"
    [ "$old" = "$(fdident 5)" ]
    value=$(head -c 4097 <&5)
    [ "${#value}" -le 4096 ]
    [ "$old" = "$(ident "$file")" ] && [ ! -L "$file" ]
    exec 5<&-
    printf '%s' "$value"
}
starttime() {
    line=$(cat "/proc/$1/stat")
    fields=${line##*) }
    printf '%s\n' "$fields" | awk 'NF >= 20 && $1 != "Z" {print $20}'
}
view() {
    [ "$(readlink /proc/self/ns/cgroup)" = "$expected_ns" ]
    [ "$(readlink /proc/1/ns/cgroup)" = "$expected_ns" ]
    [ "$(readlink /proc/$daemon/ns/cgroup)" = "$expected_ns" ]
    [ "$(readlink /proc/self/ns/mnt)" = "$(readlink /proc/$daemon/ns/mnt)" ]
    [ "$(readlink /proc/self/ns/pid)" = "$(readlink /proc/$daemon/ns/pid)" ]
    [ "$(cat /proc/$daemon/comm)" = dockerd ]
    [ "$(starttime "$daemon")" = "$daemon_start" ]
    case "$(readlink /proc/$daemon/exe)" in */dockerd) :;; *) exit 1;; esac
    mount=$(awk '$5 == "/sys/fs/cgroup" {n++; line=$0; if ($4 != "/" || $0 !~ / - cgroup2 /) bad=1}
        index($5, "/sys/fs/cgroup/") == 1 {bad=1}
        END {if (n != 1 || bad) exit 1; print line}' /proc/self/mountinfo)
    for path in /sys /sys/fs "$root"; do [ ! -L "$path" ] && [ -d "$path" ]; done
    [ "$(stat -f -c %t "$root")" = 63677270 ]
    rootid=$(ident "$root")
    [ "$(stat -c %i "$root")" = 1 ]
    exec 9< "$root"
    [ "$rootid" = "$(fdident 9)" ]
    for absent in cgroup.type memory.max memory.current; do
        [ ! -e "/proc/self/fd/9/$absent" ] && [ ! -L "/proc/self/fd/9/$absent" ]
    done
    controllers=$(read_file "/proc/self/fd/9/cgroup.controllers")
    case " $controllers " in *' memory '*) :;; *) exit 1;; esac
    printf 'view\t%s\t%s\t%s\t%s\t%s\t%s\n' "$expected_ns" \
        "$(readlink /proc/self/ns/pid)" "$(readlink /proc/self/ns/mnt)" \
        "$(starttime "$daemon")" "$rootid" "$mount"
}
processes() {
    membership=$(read_file "/proc/$init/cgroup")
    case "$membership" in 0::/*) :;; *) exit 1;; esac
    path=${membership#0::}
    case "$path" in /|*'//'*|*'..'*|*[!a-zA-Z0-9_./:-]*) exit 1;; esac
    case "$path" in */"$container"|*/docker-"$container".scope) :;; *) exit 1;; esac
    [ "${#path}" -le 2048 ]
    printf 'path\t%s\n' "$path"
    for pid in $pids; do
        [ "$(read_file "/proc/$pid/cgroup")" = "$membership" ]
        printf 'pid\t%s\t%s\n' "$pid" "$(starttime "$pid")"
    done
}
chain() {
    exec 3<&9
    rest=${path#/}; relative=; depth=0
    while [ -n "$rest" ]; do
        part=${rest%%/*}
        case "$part" in ''|.|..) exit 1;; esac
        if [ "$rest" = "$part" ]; then rest=; else rest=${rest#*/}; fi
        depth=$((depth+1)); [ "$depth" -le 32 ]
        child=/proc/self/fd/3/$part
        [ ! -L "$child" ] && [ -d "$child" ]
        old=$(ident "$child")
        exec 4< "$child"
        [ "$old" = "$(fdident 4)" ] && [ "$old" = "$(ident "$child")" ]
        [ ! -L "$child" ]
        [ "${old%%:*}" = "${rootid%%:*}" ]
        exec 3<&4; exec 4<&-
        base=/proc/self/fd/3
        [ "$(read_file "$base/cgroup.type")" = domain ]
        limit=$(read_file "$base/memory.max")
        current=$(read_file "$base/memory.current")
        relative=$relative/$part
        printf 'node\t%s\t%s\t%s\t%s\n' "$relative" "$old" "$limit" "$current"
    done
    exec 3<&-
    awk '/^MemTotal:/ {t=$2; tc++; if ($3 != "kB") bad=1}
         /^MemAvailable:/ {a=$2; ac++; if ($3 != "kB") bad=1}
         END {if (bad || tc != 1 || ac != 1) exit 1; printf "physical\t%s\t%s\n", t, a}' /proc/meminfo
}
printf 'hierarchy-v1\n'
for pass in 1 2; do
    printf 'sample\t%s\n' "$pass"
    view
    processes
    chain
    # Check identity again after every observed chain; the parser compares all
    # records, retaining the lower headroom across the two current readings.
    view
    processes
done
'''


def _bounded_number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


@dataclass(frozen=True, slots=True)
class AdmittedWorkerRef:
    """Protected admission identity; creation belongs to the worker lifecycle.

    UUID is protected join evidence, never a metric label. This reference alone
    does not prove ownership: each probe also checks Docker inspect evidence.
    """
    context: str
    endpoint: str
    container_id: str
    instance: str
    job: str
    attempt: str
    gpu_visible: bool = False
    gpu_uuid: str | None = None

    def __post_init__(self):
        if (any(type(v) is not str or not _LABEL.fullmatch(v)
                for v in (self.context, self.instance, self.job, self.attempt))
            or type(self.container_id) is not str
            or not re.fullmatch(r'[a-f0-9]{64}', self.container_id)
            or type(self.endpoint) is not str or len(self.endpoint) > 1024
            or not (_local_endpoint(self.endpoint, 'win32') or _local_endpoint(self.endpoint, 'linux'))
            or type(self.gpu_visible) is not bool
            or (self.gpu_uuid is not None and (type(self.gpu_uuid) is not str
                                              or not _UUID.fullmatch(self.gpu_uuid)))
            or (not self.gpu_visible and self.gpu_uuid is not None)):
            raise ValueError('Invalid admitted worker identity')

    @property
    def expected_labels(self):
        return {'podmuncher.instance': self.instance, 'podmuncher.job': self.job,
                'podmuncher.attempt': self.attempt}


@dataclass(frozen=True, slots=True)
class CgroupAncestor:
    """One pinned domain, with two non-atomic observations of its usage."""
    path: str
    device: int
    inode: int
    limit: int | None
    current_before: int
    current_after: int

    def __post_init__(self):
        if (type(self.path) is not str or len(self.path) > 2048
            or not re.fullmatch(r'(?:/[a-zA-Z0-9_][a-zA-Z0-9_.:-]*)+', self.path)
            or any(type(v) is not int or not 0 <= v <= _MAX
                   for v in (self.device, self.inode, self.current_before, self.current_after))
            or self.inode <= 1
            or self.limit is not None and (type(self.limit) is not int or not 0 < self.limit <= _MAX)):
            raise ValueError('Invalid cgroup ancestor')


@dataclass(frozen=True, slots=True)
class CgroupAncestry:
    """Explicit proof supplied by the trusted transport, never leaf inference.

    Nodes include every non-root ancestor in root-to-leaf order. The adapter
    proves a genuine cgroup2 root and stable process/namespace/mount identities.
    Physical readings belong to that daemon's VM/native host, not Windows RAM.

    ``namespaced`` marks the darwin route: the single node is the worker's own
    cgroup seen as the root of its private cgroup namespace, named by the owned
    container id. Ancestors above that root were not observable.
    """
    nodes: tuple[CgroupAncestor, ...]
    physical_total: int
    physical_available: int
    sample_time: datetime
    namespaced: bool = False

    def __post_init__(self):
        if (type(self.nodes) is not tuple or not 1 <= len(self.nodes) <= 32
            or any(type(node) is not CgroupAncestor for node in self.nodes)
            or any(type(v) is not int or not 0 <= v <= _MAX
                   for v in (self.physical_total, self.physical_available))
            or not 0 < self.physical_total or self.physical_available > self.physical_total
            or type(self.sample_time) is not datetime or self.sample_time.tzinfo is None
            or self.sample_time.utcoffset().total_seconds() != 0
            or type(self.namespaced) is not bool
            or (self.namespaced and len(self.nodes) != 1)):
            raise ValueError('Invalid hierarchy evidence')
        previous = ''
        for node in self.nodes:
            if (node.path.rpartition('/')[0] != previous
                or node.device != self.nodes[0].device):
                raise ValueError('Incomplete cgroup ancestry')
            previous = node.path


@dataclass(frozen=True, slots=True)
class CgroupMemory:
    identity: AdmittedWorkerRef
    version: int | None
    limit: int | None
    current: int | None
    ancestry: CgroupAncestry | None = None

    def __post_init__(self):
        if (type(self.identity) is not AdmittedWorkerRef
            or (self.version is not None and (type(self.version) is not int or self.version != 2))
            or any(v is not None and (type(v) is not int or not 0 <= v <= _MAX)
                   for v in (self.limit, self.current))
            or self.limit == 0
            or (self.version is None and (self.limit is not None or self.current is not None))
            or self.ancestry is not None and (type(self.ancestry) is not CgroupAncestry
                or self.version != 2 or self.limit != self.ancestry.nodes[-1].limit
                or self.current != self.ancestry.nodes[-1].current_after)):
            raise ValueError('Invalid cgroup evidence')

    @property
    def effective_limit(self):
        if self.ancestry is None:
            return None
        return min([self.ancestry.physical_total] +
                   [node.limit for node in self.ancestry.nodes if node.limit is not None])

    @property
    def available(self):
        if self.ancestry is None:
            return None
        leaf = self.ancestry.nodes[-1]
        return min([self.ancestry.physical_available,
                    max(0, self.effective_limit - max(leaf.current_before, leaf.current_after))] +
                   [max(0, node.limit - max(node.current_before, node.current_after))
                    for node in self.ancestry.nodes if node.limit is not None])


@dataclass(frozen=True, slots=True)
class RuntimeComputeEvidence:
    identity: AdmittedWorkerRef
    backend: Backend
    compute_types: tuple[str, ...]
    gpu_uuid: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'backend', Backend.parse(self.backend))
        if (type(self.identity) is not AdmittedWorkerRef or type(self.compute_types) is not tuple
            or not 1 <= len(self.compute_types) <= 16
            or any(type(v) is not str or not _LABEL.fullmatch(v) for v in self.compute_types)
            or (self.backend is Backend.CUDA and (not self.identity.gpu_visible
                 or self.gpu_uuid != self.identity.gpu_uuid or self.gpu_uuid is None))
            or (self.backend is Backend.CPU and self.gpu_uuid is not None)):
            raise ValueError('Invalid runtime evidence')


class WorkerMemoryTransport(Protocol):
    identity: AdmittedWorkerRef
    def read_memory_cgroup(self, *, deadline: float) -> CgroupMemory: ...


class RuntimeCapabilities(Protocol):
    identity: AdmittedWorkerRef
    def compute_types(self, backend: Backend, *, deadline: float) -> RuntimeComputeEvidence: ...


def _unmeasured():
    return Measurement(None, 'unmeasured', FieldStatus.UNKNOWN, _UNMEASURED_STAMP, ReasonCode.STORAGE_UNKNOWN)


@dataclass(frozen=True, slots=True)
class HardwareEvidence:
    """Host, engine and admitted-worker readings; every field is a Measurement.

    The host identity and engine fields default to an unmeasured Unknown so
    that evidence built elsewhere from the original readings stays valid; the
    sampler always fills them.
    """
    host_cpu_percent: Measurement[float]
    host_cpu_cores: Measurement[int]
    host_ram_total: Measurement[int]
    host_ram_available: Measurement[int]
    host_ram_used: Measurement[int]
    worker_cpu_percent: Measurement[float]
    worker_cpu_raw_percent: Measurement[float]
    worker_cpu_limit_cores: Measurement[float]
    worker_ram_limit: Measurement[int]
    worker_ram_current: Measurement[int]
    worker_ram_available: Measurement[int]
    worker_gpu_visible: Measurement[bool]
    gpu_name: Measurement[str]
    gpu_driver: Measurement[str]
    gpu_vram_total: Measurement[int]
    gpu_vram_free: Measurement[int]
    gpu_vram_used: Measurement[int]
    gpu_utilization_percent: Measurement[float]
    runtime_compute_types: Measurement[tuple[str, ...]]
    # platform.system(): Windows, Linux, Darwin.
    host_os: Measurement[str] = field(default_factory=_unmeasured)
    # Normalised amd64/arm64 (other machine names pass through as labels). On
    # darwin a Rosetta-translated Python still reports the arm64 host.
    host_arch: Measurement[str] = field(default_factory=_unmeasured)
    rosetta_translated: Measurement[bool] = field(default_factory=_unmeasured)
    host_cpu_model: Measurement[str] = field(default_factory=_unmeasured)
    # docker info: the engine's architecture, VM/native memory and CPU count.
    engine_arch: Measurement[str] = field(default_factory=_unmeasured)
    engine_mem_total_bytes: Measurement[int] = field(default_factory=_unmeasured)
    engine_ncpu: Measurement[int] = field(default_factory=_unmeasured)


_HOST_IDENTITY_SOURCES = {'host_os': 'host_platform', 'host_arch': 'host_platform',
                          'rosetta_translated': 'host_sysctl', 'host_cpu_model': 'host_sysctl'}
_ENGINE_FIELDS = ('engine_arch', 'engine_mem_total_bytes', 'engine_ncpu')


def _normalise_arch(raw):
    if type(raw) is not str:
        return None
    value = raw.strip()
    return _ARCH_ALIASES.get(value.lower()) or _safe_label(value)


def _sysctl_flag(raw):
    return {'0': False, '1': True}.get(raw)


def _cpu_label(raw):
    return raw if 0 < len(raw) <= 120 and _CPU_LABEL.fullmatch(raw) else None


def _cpu_percent(before, after):
    # psutil semantics: guest counters are included in user/nice. As psutil
    # cpu_percent does, clip individual decreasing counters before summing.
    fields = before._fields if hasattr(before, '_fields') else tuple(vars(before))
    delta = {}
    for name in fields:
        left, right = getattr(before, name), getattr(after, name)
        if not _bounded_number(left, 0, 1e30) or not _bounded_number(right, 0, 1e30):
            return None
        delta[name] = max(0.0, right - left)
    total = sum(delta.values()) - delta.get('guest', 0) - delta.get('guest_nice', 0)
    idle = delta.get('idle', 0) + delta.get('iowait', 0)
    if total <= 0 or not 0 <= idle <= total:
        return None
    return 100 * (total - idle) / total


def _mib(raw):
    if type(raw) is not str or not re.fullmatch(r'\d+(?:\.\d+)?', raw.strip()):
        return None
    number = float(raw)
    return int(number * 1048576) if _bounded_number(number, 0, 2**53 / 1048576) else None


def _percent(raw, *, maximum=100):
    if type(raw) is not str or not re.fullmatch(r'\d+(?:\.\d+)?', raw.strip()):
        return None
    value = float(raw)
    return value if _bounded_number(value, 0, maximum) else None


def _safe_label(raw):
    return raw if 0 < len(raw) <= 120 and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9 ._+()/-]*', raw) else None


class _DockerProbe:
    def __init__(self, identity, runner, environ, monotonic, deadline):
        self.identity, self.runner, self.environ = identity, runner, environ
        self.monotonic, self.deadline = monotonic, deadline

    def check_deadline(self):
        if self.monotonic() >= self.deadline:
            raise DiscoveryError(ReasonCode.REQUEST_TIMEOUT)

    def call(self, *args):
        self.check_deadline()
        if self.environ.get('DOCKER_HOST') or self.environ.get('DOCKER_CONTEXT'):
            raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
        result = self.runner(('docker', '--context', self.identity.context, *args),
                             deadline=self.deadline, env=dict(self.environ))
        self.check_deadline()
        max_output = (_DOCKER_EVIDENCE_MAX_BYTES
                      if args[:2] == ('container', 'inspect')
                      else _DOCKER_SMALL_EVIDENCE_MAX_BYTES)
        if (type(result) is not ProcessResult or type(result.returncode) is not int
            or type(result.stdout) is not bytes or len(result.stdout) > max_output):
            raise ValueError('Malformed process evidence')
        if result.returncode != 0:
            raise DiscoveryError(ReasonCode.DOCKER_UNAVAILABLE)
        return result.stdout

    def verify(self):
        data = json.loads(self.call('context', 'inspect', self.identity.context))
        if (type(data) is not list or len(data) != 1
            or data[0]['Name'] != self.identity.context
            or data[0]['Endpoints']['docker']['Host'] != self.identity.endpoint):
            raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
        owned = json.loads(self.call('container', 'inspect', self.identity.container_id))
        if type(owned) is not list or len(owned) != 1:
            raise DiscoveryError(ReasonCode.UNAUTHORIZED)
        info = owned[0]
        if (info['Id'] != self.identity.container_id or info['State']['Running'] is not True
            or type(info['Config']['Labels']) is not dict
            or not all(info['Config']['Labels'].get(k) == v
                       for k, v in self.identity.expected_labels.items())):
            raise DiscoveryError(ReasonCode.UNAUTHORIZED)
        return info


class DockerCgroupMemory:
    """Bounded concrete v2 adapter for an existing admitted worker only.

    All owned PIDs must share the init cgroup. Workers in deeper subgroups need
    an actual worker PID join and remain unknown here. On win32 and linux the
    reader runs in the daemon's view and no container exec occurs; on darwin
    (Docker Desktop VM) the reader runs via ``docker exec`` inside the owned
    worker's private cgroup namespace (see ``_NAMESPACED_SCRIPT``).
    """
    def __init__(self, identity: AdmittedWorkerRef, *, runner=None, environ=None,
                 monotonic=time.monotonic, platform=sys.platform, clock=None):
        if type(identity) is not AdmittedWorkerRef:
            raise ValueError('Admission identity required')
        self.identity = identity
        self.runner = runner if runner is not None else BoundedRunner(max_output=_DOCKER_EVIDENCE_MAX_BYTES)
        self.environ = os.environ if environ is None else environ
        self.monotonic = monotonic
        self.platform = platform
        self.clock = clock if clock is not None else lambda: datetime.now(UTC)

    def read_memory_cgroup(self, *, deadline):
        if not _bounded_number(deadline, 0, 1e30):
            raise ValueError('Invalid deadline')
        probe = _DockerProbe(self.identity, self.runner, self.environ, self.monotonic, deadline)
        unknown = CgroupMemory(self.identity, None, None, None)
        before = probe.verify()
        try:
            signature = _container_process_identity(before)
            pids = _owned_pids(probe, signature[0])
            namespaced = False
            if self.platform == 'win32' and _local_endpoint(self.identity.endpoint, 'win32'):
                prefix = ('wsl.exe', '-d', 'docker-desktop', '-u', 'root', '--exec')
            elif self.platform == 'linux' and _local_endpoint(self.identity.endpoint, 'linux'):
                prefix = ()
            elif self.platform == 'darwin' and _local_endpoint(self.identity.endpoint, 'linux'):
                namespaced = True
            else:
                return unknown
            probe.check_deadline()
            if namespaced:
                # The outer runner bounds the request and reaps the docker CLI;
                # the reader itself only performs instant reads.
                argv = ('docker', '--context', self.identity.context, 'exec', self.identity.container_id,
                        'sh', '-c', _NAMESPACED_SCRIPT, 'cgroup-reader')
            else:
                seconds = f'{max(0.001, min(30, deadline - self.monotonic())):.3f}'
                # The outer runner bounds the whole request. This second bound
                # also terminates the Linux reader if a WSL client is interrupted.
                argv = (*prefix, 'timeout', '-s', 'KILL', seconds, 'sh', '-c', _DAEMON_SCRIPT, 'daemon-reader',
                        _CGROUP_SCRIPT, str(signature[0]), self.identity.container_id,
                        *(str(pid) for pid in pids))
            probe.check_deadline()
            if self.environ.get('DOCKER_HOST') or self.environ.get('DOCKER_CONTEXT'):
                raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
            stamp = self.clock()
            result = self.runner(argv, deadline=deadline, env=dict(self.environ))
            probe.check_deadline()
            after = probe.verify()
            if (_container_process_identity(after) != signature
                or _owned_pids(probe, signature[0]) != pids):
                raise DiscoveryError(ReasonCode.CONFLICT)
            if (type(result) is not ProcessResult or type(result.returncode) is not int
                or result.returncode != 0 or type(result.stdout) is not bytes
                or len(result.stdout) > 32768):
                return unknown
            if namespaced:
                ancestry = _parse_namespaced(result.stdout, stamp, self.identity.container_id)
            else:
                ancestry = _parse_hierarchy(result.stdout, pids, stamp, self.identity.container_id)
            leaf = ancestry.nodes[-1]
            return CgroupMemory(self.identity, 2, leaf.limit, leaf.current_after, ancestry)
        except (ValueError, OSError, KeyError, UnicodeError, TypeError):
            return unknown


def _container_process_identity(info):
    state = info['State']
    pid, started = state['Pid'], state['StartedAt']
    if (type(pid) is not int or not 0 < pid <= 2**31 - 1 or type(started) is not str
        or not re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,9})?Z', started)
        or datetime.fromisoformat(started).year < 1970):
        raise ValueError('Missing owned process identity')
    return pid, started


def _owned_pids(probe, init):
    rows = probe.call('top', probe.identity.container_id, '-eo', 'pid').decode('ascii').splitlines()
    if not 2 <= len(rows) <= 65 or rows[0].strip() != 'PID':
        raise ValueError('Unresolved worker process set')
    pids = tuple(sorted(_positive_integer(row.strip(), maximum=2**31 - 1) for row in rows[1:]))
    if init not in pids or len(set(pids)) != len(pids):
        raise ValueError('Unresolved container init')
    return pids


def _positive_integer(raw, *, maximum=_MAX, zero=False):
    if type(raw) is not str or not re.fullmatch(r'\d{1,19}', raw):
        raise ValueError('Invalid integer evidence')
    value = int(raw)
    if not (0 if zero else 1) <= value <= maximum:
        raise ValueError('Out of range evidence')
    return value


def _optional_positive(raw, maximum):
    try:
        return _positive_integer(raw, maximum=maximum)
    except ValueError:
        return None


def _parse_hierarchy(raw, pids, stamp, container_id):
    """Validate the bounded fixed-reader transcript before trusting any number."""
    lines = iter(raw.decode('ascii').splitlines())

    def take(kind, size):
        parts = next(lines, '').split('\t')
        if len(parts) != size or parts[0] != kind:
            raise ValueError('Incomplete hierarchy protocol')
        return parts[1:]

    def view():
        values = take('view', 7)
        for name, value in zip(('cgroup', 'pid', 'mnt'), values[:3]):
            if not re.fullmatch(name + r':\[\d{1,19}\]', value):
                raise ValueError('Invalid namespace identity')
        _positive_integer(values[3])
        root_device, root_inode, root_mode = values[4].split(':')
        _positive_integer(root_device, zero=True)
        if root_inode != '1' or int(root_mode, 16) & 0o170000 != 0o040000:
            raise ValueError('Hidden hierarchy root')
        mount = values[5].split()
        if (len(mount) < 10 or mount[3:5] != ['/', '/sys/fs/cgroup']
            or mount[mount.index('-') + 1] != 'cgroup2'):
            raise ValueError('Unsupported root mount')
        return tuple(values)

    def processes():
        path, = take('path', 2)
        identities = []
        for expected in pids:
            pid, started = take('pid', 3)
            if _positive_integer(pid) != expected:
                raise ValueError('Worker process identity changed')
            identities.append((expected, _positive_integer(started)))
        return path, tuple(identities)

    if next(lines, '') != 'hierarchy-v1':
        raise ValueError('Unknown hierarchy protocol')
    snapshots = []
    for number in ('1', '2'):
        if take('sample', 2) != [number]:
            raise ValueError('Missing hierarchy recheck')
        before_view = view()
        before_processes = processes()
        path = before_processes[0]
        if path.rsplit('/', 1)[-1] not in (container_id, f'docker-{container_id}.scope'):
            raise ValueError('Process cgroup does not identify the owned container')
        components = path.split('/')[1:]
        if not 1 <= len(components) <= 32:
            raise ValueError('Unbounded ancestry')
        nodes = []
        for depth in range(1, len(components) + 1):
            relative, identity, limit, current = take('node', 5)
            device, inode, mode = identity.split(':')
            if (relative != '/' + '/'.join(components[:depth])
                or int(mode, 16) & 0o170000 != 0o040000
                or device != before_view[4].split(':')[0]):
                raise ValueError('Link, covering mount or incomplete ancestry')
            usage = _positive_integer(current, zero=True)
            nodes.append(CgroupAncestor(relative, _positive_integer(device, zero=True),
                _positive_integer(inode), None if limit == 'max' else _positive_integer(limit), usage, usage))
        total, available = (_positive_integer(v, maximum=_MAX // 1024, zero=True) * 1024
                            for v in take('physical', 3))
        if total <= 0 or available > total:
            raise ValueError('Invalid physical memory')
        if view() != before_view or processes() != before_processes:
            raise ValueError('Hierarchy identity changed')
        snapshots.append((before_view, before_processes, tuple(nodes), total, available))
    if next(lines, None) is not None:
        raise ValueError('Trailing hierarchy data')
    first, last = snapshots
    if first[:2] != last[:2] or first[3] != last[3]:
        raise ValueError('Process or physical identity changed')
    joined = []
    for before, after in zip(first[2], last[2]):
        if (before.path, before.device, before.inode, before.limit) != (
                after.path, after.device, after.inode, after.limit):
            raise ValueError('Ancestor identity or limit changed')
        joined.append(CgroupAncestor(before.path, before.device, before.inode, before.limit,
                                     before.current_before, after.current_after))
    return CgroupAncestry(tuple(joined), first[3], min(first[4], last[4]), stamp)


def _parse_namespaced(raw, stamp, container_id):
    """Validate the darwin in-namespace reader transcript (two leaf samples)."""
    lines = iter(raw.decode('ascii').splitlines())

    def take(kind, size):
        parts = next(lines, '').split('\t')
        if len(parts) != size or parts[0] != kind:
            raise ValueError('Incomplete namespaced protocol')
        return parts[1:]

    if next(lines, '') != 'namespaced-v1':
        raise ValueError('Unknown namespaced protocol')
    samples = []
    for number in ('1', '2'):
        index, membership, identity, limit, current = take('sample', 6)
        if index != number or membership != '0::/':
            raise ValueError('Reader is not at its private cgroup namespace root')
        device, inode, mode = identity.split(':')
        if int(mode, 16) & 0o170000 != 0o040000:
            raise ValueError('Namespace root is not a directory')
        node = CgroupAncestor('/' + container_id, _positive_integer(device, zero=True), _positive_integer(inode),
                              None if limit == 'max' else _positive_integer(limit),
                              _positive_integer(current, zero=True), _positive_integer(current, zero=True))
        total, available = (_positive_integer(v, maximum=_MAX // 1024, zero=True) * 1024
                            for v in take('physical', 3))
        if total <= 0 or available > total:
            raise ValueError('Invalid physical memory')
        samples.append((node, total, available))
    if next(lines, None) is not None:
        raise ValueError('Trailing namespaced data')
    (before, total, available), (after, total_after, available_after) = samples
    if ((before.device, before.inode, before.limit) != (after.device, after.inode, after.limit)
        or total != total_after):
        raise ValueError('Leaf identity, limit or physical memory changed')
    leaf = CgroupAncestor(before.path, before.device, before.inode, before.limit,
                          before.current_before, after.current_after)
    return CgroupAncestry((leaf,), total, min(available, available_after), stamp, namespaced=True)


def _cpu_entitlement(config, available):
    if type(available) is not int or not 1 <= available <= 1048576 or type(config) is not dict:
        return None
    nano, quota, period, cpuset = (config.get(k) for k in ('NanoCpus', 'CpuQuota', 'CpuPeriod', 'CpusetCpus'))
    if (type(nano) is not int or not 0 <= nano <= _MAX or type(quota) is not int
        or not -1 <= quota <= _MAX or type(period) is not int or not 0 <= period <= _MAX
        or type(cpuset) is not str or len(cpuset) > 4096):
        return None
    limits = [float(available)]
    if nano:
        limits.append(nano / 1e9)
    if quota > 0:
        if period <= 0:
            return None
        limits.append(quota / period)
    if cpuset:
        cores = set()
        for part in cpuset.split(','):
            if not re.fullmatch(r'\d{1,7}(?:-\d{1,7})?', part):
                return None
            bounds = [int(v) for v in part.split('-')]
            start, end = bounds[0], bounds[-1]
            if not 0 <= start <= end < available:
                return None
            cores.update(range(start, end + 1))
        limits.append(float(len(cores)))
    return min(limits)


class HardwareSampler:
    def __init__(self, *, runner=None, worker: WorkerMemoryTransport | None = None,
                 runtime: RuntimeCapabilities | None = None,
                 clock: Callable[[], datetime] | None = None,
                 monotonic=time.monotonic, sleep=time.sleep, cpu_times=psutil.cpu_times,
                 virtual_memory=psutil.virtual_memory, cpu_count=psutil.cpu_count,
                 environ: Mapping[str, str] | None = None, platform=os.name,
                 system=host_platform.system, machine=host_platform.machine,
                 processor=host_platform.processor, timeout=4.0, max_age=3.0):
        if not _bounded_number(timeout, 1.001, 30) or not _bounded_number(max_age, 0.001, 3):
            raise ValueError('Invalid sampler bounds')
        if worker is not None and type(getattr(worker, 'identity', None)) is not AdmittedWorkerRef:
            raise ValueError('Admission identity required')
        if runtime is not None and type(getattr(runtime, 'identity', None)) is not AdmittedWorkerRef:
            raise ValueError('Runtime identity required')
        self.runner = runner if runner is not None else BoundedRunner(max_output=_DOCKER_EVIDENCE_MAX_BYTES)
        self.worker, self.runtime = worker, runtime
        self.clock = clock if clock is not None else lambda: datetime.now(UTC)
        self.monotonic, self.sleep = monotonic, sleep
        self.cpu_times, self.virtual_memory, self.cpu_count = cpu_times, virtual_memory, cpu_count
        self.environ = os.environ if environ is None else environ
        self.platform, self.timeout, self.max_age = platform, timeout, max_age
        self.system, self.machine, self.processor = system, machine, processor
        self._baseline = None
        self._lock = threading.Lock()

    def sample(self):
        """Blocking bounded sample; concurrent samplers cannot accumulate threads.

        First CPU window overlaps subprocess work. Subsequent calls use the
        previous CPU baseline; the caller schedules its one-second cadence.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Sampling already in progress')
        try:
            return self._sample()
        finally:
            self._lock.release()

    def _sample(self):
        started = self.monotonic()
        deadline = started + self.timeout
        readings = {}

        def check():
            if self.monotonic() >= deadline:
                raise DiscoveryError(ReasonCode.REQUEST_TIMEOUT)

        def put(name, value, source, reason=ReasonCode.STORAGE_UNKNOWN, status=FieldStatus.UNKNOWN, stamp=None):
            stamp = self.clock() if stamp is None else stamp
            if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
                raise ValueError('UTC clock required')
            if self.monotonic() >= deadline:
                value, reason = None, ReasonCode.REQUEST_TIMEOUT
            readings[name] = Measurement(value, source, FieldStatus.OK if value is not None else status,
                stamp.isoformat().replace('+00:00', 'Z'), None if value is not None else reason)

        before = self._baseline
        try:
            if before is None:
                check()
                before = self.cpu_times()
            check()
            cores = self.cpu_count(logical=True)
            put('host_cpu_cores', cores if type(cores) is int and 0 < cores <= 1048576 else None, 'host_psutil')
            check()
            mem = self.virtual_memory()
            values = (mem.total, mem.available, mem.used)
            if (any(type(v) is not int or not 0 <= v <= _MAX for v in values)
                or not 0 < mem.total or mem.available > mem.total or mem.used > mem.total):
                raise ValueError('Invalid host memory')
            for key, value in zip(('host_ram_total', 'host_ram_available', 'host_ram_used'), values):
                put(key, value, 'host_psutil')
        except (OSError, ValueError, RuntimeError, AttributeError, DiscoveryError):
            pass

        def text(argv):
            """One bounded host command; None on any non-timeout failure."""
            check()
            try:
                result = self.runner(argv, deadline=deadline, env=dict(self.environ))
            except DiscoveryError as exc:
                if exc.reason is ReasonCode.REQUEST_TIMEOUT:
                    raise
                return None
            check()
            if (type(result) is not ProcessResult or type(result.returncode) is not int
                or type(result.stdout) is not bytes
                or len(result.stdout) > _DOCKER_SMALL_EVIDENCE_MAX_BYTES):
                raise ValueError('Malformed process evidence')
            if result.returncode != 0:
                return None
            try:
                return result.stdout.decode('ascii').strip()
            except UnicodeError:
                return None

        # Host identity: never 0 or a guess; Unknown when a probe fails.
        try:
            check()
            system = self.system()
            darwin = system == 'Darwin'
            put('host_os', _safe_label(system) if type(system) is str else None, 'host_platform')
            arch = _normalise_arch(self.machine())
            if darwin:
                translated = _sysctl_flag(text(_SYSCTL_TRANSLATED_ARGV))
                if translated:
                    arch = 'arm64'
                put('rosetta_translated', translated, 'host_sysctl')
                put('host_arch', arch, 'host_sysctl_proc_translated' if translated else 'host_platform')
                brand = text(_SYSCTL_CPU_BRAND_ARGV)
                put('host_cpu_model', _cpu_label(brand) if brand is not None else None, 'host_sysctl')
            else:
                put('rosetta_translated', False, 'host_platform')
                put('host_arch', arch, 'host_platform')
                processor = self.processor()
                put('host_cpu_model', _cpu_label(processor) if type(processor) is str else None, 'host_platform')
        except (OSError, ValueError, UnicodeError, TypeError, AttributeError, RuntimeError, DiscoveryError):
            pass

        # Engine facts from the daemon the product uses (the admitted worker's
        # context when one is bound, else the CLI's current context).
        engine_reason = ReasonCode.DOCKER_UNAVAILABLE
        try:
            if self.environ.get('DOCKER_HOST') or self.environ.get('DOCKER_CONTEXT'):
                raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
            bound = self.worker if self.worker is not None else self.runtime
            context = () if bound is None else ('--context', bound.identity.context)
            raw = text(('docker', *context, 'info', '--format', _ENGINE_INFO_FORMAT))
            if raw is not None:
                parts = raw.split()
                if len(parts) != 3:
                    raise ValueError('Malformed engine evidence')
                arch, memory_total, ncpu = parts
                put('engine_arch', _normalise_arch(arch), 'docker_info')
                put('engine_mem_total_bytes', _optional_positive(memory_total, _MAX), 'docker_info')
                put('engine_ncpu', _optional_positive(ncpu, 1048576), 'docker_info')
        except DiscoveryError as exc:
            engine_reason = exc.reason
        except (OSError, ValueError, UnicodeError, TypeError, AttributeError, RuntimeError):
            engine_reason = ReasonCode.STORAGE_UNKNOWN

        gpu_uuid = None
        gpu_reason = ReasonCode.GPU_UNAVAILABLE
        try:
            check()
            result = self.runner(_GPU_ARGV, deadline=deadline, env=dict(self.environ))
            check()
            if (type(result) is not ProcessResult or type(result.returncode) is not int
                or type(result.stdout) is not bytes
                or len(result.stdout) > _DOCKER_SMALL_EVIDENCE_MAX_BYTES):
                raise ValueError('Malformed GPU evidence')
            lines = result.stdout.decode('utf-8').strip().splitlines()
            if result.returncode == 0 and len(lines) == 1:
                parts = [p.strip() for p in lines[0].split(',')]
                if len(parts) == 8 and parts[0] == '0' and _UUID.fullmatch(parts[1]):
                    gpu_uuid = parts[1]
                    total, free, used = map(_mib, parts[4:7])
                    if (total is None or free is None or used is None or total <= 0
                        or free > total or used > total or free + used > total + 1048576):
                        total = free = used = None
                    for key, value in zip(('gpu_name', 'gpu_driver', 'gpu_vram_total', 'gpu_vram_free',
                                           'gpu_vram_used', 'gpu_utilization_percent'),
                                          (_safe_label(parts[2]), _safe_label(parts[3]), total, free, used,
                                           _percent(parts[7]))):
                        put(key, value, 'host_nvidia_smi', gpu_reason)
        except DiscoveryError as exc:
            gpu_reason = exc.reason
        except (OSError, ValueError, UnicodeError, TypeError):
            pass

        worker_reason = ReasonCode.STORAGE_UNKNOWN
        worker_readings = {}
        if self.worker is not None:
            try:
                identity = self.worker.identity
                probe = _DockerProbe(identity, self.runner, self.environ, self.monotonic, deadline)
                owned = probe.verify()
                config = owned['HostConfig']
                process_identity = _container_process_identity(owned)
                # The slow non-selection CPU stats call (1-2 s live) runs before
                # every selection-field read so it cannot age their stamps.
                raw = probe.call('stats', '--no-stream', '--format', '{{.CPUPerc}}', identity.container_id)
                stats_stamp = self.clock()
                cpu_count = json.loads(probe.call('info', '--format', '{{json .NCPU}}'))
                entitlement = _cpu_entitlement(config, cpu_count)
                limit_stamp = self.clock()
                check()
                memory = self.worker.read_memory_cgroup(deadline=deadline)
                check()
                if type(memory) is not CgroupMemory or memory.identity != identity:
                    raise DiscoveryError(ReasonCode.WORKER_PROTOCOL_MISMATCH)
                memory_stamp = memory.ancestry.sample_time if memory.ancestry is not None else self.clock()
                hierarchy = ('worker_cgroup_v2_namespaced_leaf'
                             if memory.ancestry is not None and memory.ancestry.namespaced
                             else 'worker_cgroup_v2_hierarchy')
                raw_percent = _percent(raw.decode('ascii').strip().removesuffix('%'), maximum=104857600)
                normalized = raw_percent / entitlement if raw_percent is not None and entitlement else None
                # A limit/stats race is not evidence for a clipped percentage.
                if normalized is not None and normalized > 100:
                    normalized = None
                worker_readings = {
                    'worker_cpu_percent': (normalized, 'docker_stats_effective_entitlement'),
                    'worker_cpu_raw_percent': (raw_percent, 'docker_stats'),
                    'worker_cpu_limit_cores': (entitlement, 'docker_inspect_and_info'),
                    'worker_ram_limit': (memory.effective_limit, hierarchy),
                    'worker_ram_current': (memory.current, 'worker_cgroup_v2'),
                    'worker_ram_available': (memory.available,
                         'worker_cgroup_v2_clipped_current_above_limit' if memory.effective_limit is not None
                         and memory.current is not None and memory.current > memory.effective_limit
                         else hierarchy),
                    'worker_gpu_visible': (identity.gpu_visible, 'owned_worker'),
                }
                if raw_percent is not None and entitlement and raw_percent / entitlement > 100:
                    worker_readings['worker_cpu_percent'] = (None, 'docker_stats_limit_race')
                compute = None
                if self.runtime is not None:
                    if self.runtime.identity != identity:
                        raise DiscoveryError(ReasonCode.WORKER_PROTOCOL_MISMATCH)
                    backend = Backend.CUDA if identity.gpu_visible else Backend.CPU
                    if backend is Backend.CPU or gpu_uuid is not None and gpu_uuid == identity.gpu_uuid:
                        check()
                        result = self.runtime.compute_types(backend, deadline=deadline)
                        check()
                        if (type(result) is not RuntimeComputeEvidence or result.identity != identity
                            or result.backend is not backend or result.gpu_uuid != (gpu_uuid if backend is Backend.CUDA else None)):
                            raise DiscoveryError(ReasonCode.WORKER_PROTOCOL_MISMATCH)
                        compute = list(result.compute_types)
                runtime_stamp = self.clock()
                rechecked = probe.verify()
                if (rechecked['HostConfig'] != config
                    or _container_process_identity(rechecked) != process_identity):
                    raise DiscoveryError(ReasonCode.CONFLICT)
                # GPU visibility is stamped at the docker inspect/info read that
                # yields it (after the slow stats call), not at the recheck,
                # which does not re-verify GPU assignment.
                stamps = {'worker_cpu_limit_cores': limit_stamp, 'worker_gpu_visible': limit_stamp}
                for key, (value, source) in worker_readings.items():
                    stamp = memory_stamp if key.startswith('worker_ram') else stamps.get(key, stats_stamp)
                    put(key, value, source, ReasonCode.CONFLICT if source.endswith('race') else ReasonCode.STORAGE_UNKNOWN, stamp=stamp)
                put('runtime_compute_types', compute, 'admitted_runtime', stamp=runtime_stamp)
            except DiscoveryError as exc:
                worker_reason = exc.reason
            except (OSError, ValueError, UnicodeError, KeyError, TypeError, AttributeError, RuntimeError):
                worker_reason = ReasonCode.SERVICE_UNAVAILABLE

        try:
            check()
            if self._baseline is None:
                self.sleep(max(0, 1.0 - (self.monotonic() - started)))
            check()
            after = self.cpu_times()
            check()
            self._baseline = after
            put('host_cpu_percent', _cpu_percent(before, after), 'host_psutil')
        except (OSError, ValueError, RuntimeError, AttributeError, DiscoveryError):
            self._baseline = None

        for key in HardwareEvidence.__dataclass_fields__:
            if key in readings:
                continue
            if key.startswith('gpu_'):
                reason, source = gpu_reason, 'host_nvidia_smi'
            elif key.startswith(('worker_', 'runtime_')):
                reason, source = worker_reason, 'admitted_worker'
            elif key in _ENGINE_FIELDS:
                reason, source = engine_reason, 'docker_info'
            else:
                reason, source = ReasonCode.STORAGE_UNKNOWN, _HOST_IDENTITY_SOURCES.get(key, 'host_psutil')
            put(key, None, source, reason)
        end = self.clock()
        for key, item in tuple(readings.items()):
            age = (end - datetime.fromisoformat(item.sample_time)).total_seconds()
            if age < 0 or age > self.max_age:
                readings[key] = Measurement(None, item.source, FieldStatus.STALE, item.sample_time, ReasonCode.STORAGE_STALE)
        return HardwareEvidence(**readings)

    async def sample_async(self):
        return await asyncio.to_thread(self.sample)
