"""Read-only host storage evidence and owned, no-follow handle containment.

No probing occurs on import. Adapters are injectable; native Windows APIs are
loaded only on construction. Paths and identities are protected evidence, never
telemetry. A HeldPath owns every ancestor handle until context exit. Its handle
is borrowed, must not be closed by callers, and is never a publication/delete API.
POSIX cannot prevent renames: validation detects edge changes, while reads always
use the admitted file descriptor. Future mutators must operate through held
parents and provide their own atomic mutation contract.
"""
from __future__ import annotations

import errno
import ntpath
import os
import re
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from .domains import (
    Component,
    DomainValidationError,
    FieldStatus,
    PreflightReason,
    PreflightResult,
    ReasonCode,
)
from .snapshots import Measurement


class StorageError(DomainValidationError):
    def __init__(self, reason_code: ReasonCode = ReasonCode.ARTIFACT_REJECTED):
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class FileIdentity:
    volume: str
    file: str


@dataclass(frozen=True, slots=True)
class HandleInfo:
    identity: FileIdentity
    directory: bool
    regular: bool
    links: int
    reparse: bool = False


@dataclass(frozen=True, slots=True)
class RawVolume:
    """None means unproved. Capacity/free must already account for quotas."""
    capacity: int | None
    free: int | None
    free_inodes: int | None
    mounted: bool | None
    writable: bool | None
    network: bool | None
    source: str


@dataclass(frozen=True, slots=True)
class VolumeEvidence:
    identity: Measurement[str]
    capacity: Measurement[int]
    free: Measurement[int]
    free_inodes: Measurement[int]
    mounted: Measurement[bool]
    writable: Measurement[bool]
    network: Measurement[bool]
    state_allowed: Measurement[bool]


class VolumeAdapter(Protocol):
    def split_absolute(self, path: str) -> tuple[str, tuple[str, ...]]: ...
    def open_anchor(self, anchor: str) -> object: ...
    def open_child(self, parent: object, name: str, directory: bool) -> object: ...
    def info(self, handle: object) -> HandleInfo: ...
    def edge_matches(self, parent: object, name: str, child: object) -> bool: ...
    def close(self, handle: object) -> None: ...
    def read(self, handle: object, limit: int) -> bytes: ...
    def probe(self, handle: object) -> RawVolume: ...


class PreflightTelemetry(Protocol):
    def span(self, name: str, *, component: str) -> AbstractContextManager[object]: ...
    def result(self, *, result: str, reason: str, component: str) -> None: ...


def relative_components(path: str) -> tuple[str, ...]:
    """Reject both platform syntaxes, streams, ambiguous Windows names/devices."""
    if type(path) is not str or not path or '\x00' in path:
        raise StorageError()
    if ntpath.splitdrive(path)[0] or path.startswith(('/', '\\')):
        raise StorageError()
    parts = tuple(re.split(r'[/\\]', path))
    for part in parts:
        if (not part or part in ('.', '..') or part.endswith((' ', '.'))
                or any(c in part for c in ':<>"|?*')
                or any(ord(c) < 32 for c in part)
                or re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', part)):
            raise StorageError()
    if len(parts) > 256:
        raise StorageError()
    return parts


def _safe(info: HandleInfo, directory: bool) -> None:
    if info.reparse or (directory and not info.directory) or (
            not directory and (not info.regular or info.links != 1)):
        raise StorageError()


class HeldPath:
    """Owned handle chain; use only inside its context manager. Not thread safe."""
    def __init__(self, adapter: VolumeAdapter):
        self._adapter = adapter
        self._chain: list[tuple[object, HandleInfo, str | None, bool]] = []
        self._closed = False

    @property
    def handle(self) -> object:
        if self._closed or not self._chain:
            raise StorageError()
        return self._chain[-1][0]

    @property
    def identity(self) -> FileIdentity:
        _ = self.handle  # Raises when closed or empty.
        return self._chain[-1][1].identity

    @property
    def closed(self) -> bool:
        return self._closed

    def _append(self, handle: object, name: str | None, directory: bool) -> None:
        try:
            info = self._adapter.info(handle)
            _safe(info, directory)
        except BaseException:
            self._adapter.close(handle)
            raise
        self._chain.append((handle, info, name, directory))

    def verify(self) -> None:
        _ = self.handle  # Raises when closed or empty.
        for index, (handle, original, name, directory) in enumerate(self._chain):
            current = self._adapter.info(handle)
            _safe(current, directory)
            if current.identity != original.identity:
                raise StorageError()
            if index and not self._adapter.edge_matches(self._chain[index - 1][0], name, handle):
                raise StorageError()

    def read(self, limit: int = 65536) -> bytes:
        """Read at most limit bytes from the current offset; never an unbounded read."""
        if type(limit) is not int or not 0 <= limit <= 1048576:
            raise StorageError()
        try:
            self.verify()
            _safe(self._adapter.info(self.handle), False)
            data = self._adapter.read(self.handle, limit)
            if type(data) is not bytes or len(data) > limit:
                raise StorageError()
            self.verify()
            return data
        except OSError:
            raise StorageError() from None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            first_error = None
            for handle, _, _, _ in reversed(self._chain):
                try:
                    self._adapter.close(handle)
                except OSError as error:
                    first_error = error
            self._chain.clear()
            if first_error:
                raise StorageError() from None


def _walk(held: HeldPath, path: str, *, nearest: bool) -> None:
    adapter = held._adapter
    anchor, parts = adapter.split_absolute(path)
    held._append(adapter.open_anchor(anchor), None, True)
    for index, name in enumerate(parts):
        held.verify()
        try:
            child = adapter.open_child(held.handle, name, True)
        except FileNotFoundError:
            if nearest:
                break
            raise
        except NotADirectoryError:
            if not nearest or index != len(parts) - 1:
                raise
            leaf = adapter.open_child(held.handle, name, False)
            try:
                _safe(adapter.info(leaf), False)
                if not adapter.edge_matches(held.handle, name, leaf):
                    raise StorageError()
                held.verify()
            finally:
                adapter.close(leaf)
            break
        if nearest and index == len(parts) - 1:
            try:
                info = adapter.info(child)
            except BaseException:
                adapter.close(child)
                raise
            if not info.directory:
                try:
                    _safe(info, False)
                    if not adapter.edge_matches(held.handle, name, child):
                        raise StorageError()
                    held.verify()
                finally:
                    adapter.close(child)
                break
        held._append(child, name, True)
    held.verify()


@contextmanager
def nearest_existing_ancestor(path: str | os.PathLike, *, adapter: VolumeAdapter | None = None) -> Iterator[HeldPath]:
    """Hold the nearest existing directory; never mkdir or follow links."""
    held = HeldPath(adapter if adapter is not None else native_adapter())
    try:
        _walk(held, os.fspath(path), nearest=True)
        yield held
    except OSError:
        raise StorageError(ReasonCode.STORAGE_UNKNOWN) from None
    finally:
        held.close()


@contextmanager
def open_contained(root: str | os.PathLike, relative: str, *, directory: bool = False,
                   expected: FileIdentity | None = None,
                   adapter: VolumeAdapter | None = None) -> Iterator[HeldPath]:
    """Open existing regular single-link file (or directory) below a held root."""
    parts = relative_components(relative)
    held = HeldPath(adapter if adapter is not None else native_adapter())
    try:
        _walk(held, os.fspath(root), nearest=False)
        volume = held.identity.volume
        for index, name in enumerate(parts):
            held.verify()
            is_dir = directory or index < len(parts) - 1
            held._append(held._adapter.open_child(held.handle, name, is_dir), name, is_dir)
            if held.identity.volume != volume:
                raise StorageError()
        held.verify()
        if expected is not None and held.identity != expected:
            raise StorageError()
        yield held
    except OSError:
        raise StorageError() from None
    finally:
        held.close()


class VolumeProvider:
    def __init__(self, adapter: VolumeAdapter, *, clock: Callable[[], datetime],
                 telemetry: PreflightTelemetry | None = None):
        self.adapter, self.clock, self.telemetry = adapter, clock, telemetry

    def sample(self, path: str | os.PathLike, *, sqlite_state: bool = False,
               shared: bool = False, previous_identity: str | None = None) -> VolumeEvidence:
        now = self.clock()
        if now.tzinfo is None or now.utcoffset().total_seconds() != 0:
            raise StorageError(ReasonCode.INVALID_INPUT)
        stamp = now.isoformat().replace('+00:00', 'Z')
        scope = self.telemetry.span('preflight', component=Component.STORAGE.value) if self.telemetry else nullcontext()
        with scope:
            source = 'host_volume'
            reason = ReasonCode.STORAGE_UNKNOWN
            status = FieldStatus.UNKNOWN
            identity = None
            raw = RawVolume(None, None, None, None, None, None, source)
            try:
                with nearest_existing_ancestor(path, adapter=self.adapter) as held:
                    identity = held.identity.volume
                    if previous_identity is not None and previous_identity != identity:
                        raise StorageError(ReasonCode.STORAGE_STALE)
                    raw = self.adapter.probe(held.handle)
                    held.verify()
                    source = raw.source
                    for count in (raw.capacity, raw.free, raw.free_inodes):
                        if count is not None and (type(count) is not int or count < 0):
                            raise StorageError(ReasonCode.STORAGE_UNKNOWN)
                    for flag in (raw.mounted, raw.writable, raw.network):
                        if flag is not None and type(flag) is not bool:
                            raise StorageError(ReasonCode.STORAGE_UNKNOWN)
            except (OSError, StorageError) as error:
                reason = getattr(error, 'reason_code', ReasonCode.STORAGE_UNKNOWN)
                if reason is ReasonCode.ARTIFACT_REJECTED:
                    reason = ReasonCode.STORAGE_UNKNOWN
                if reason is ReasonCode.STORAGE_STALE or previous_identity is not None:
                    status, reason = FieldStatus.STALE, ReasonCode.STORAGE_STALE
                identity = None
                raw = RawVolume(None, None, None, None, None, None, source)

            def measured(value):
                return Measurement(value, source, FieldStatus.OK if value is not None else status,
                                   stamp, None if value is not None else reason)

            allowed = None
            if sqlite_state:
                if shared or raw.network is True:
                    allowed = False
                elif raw.network is False and raw.mounted is True:
                    allowed = True
            values = VolumeEvidence(measured(identity), measured(raw.capacity), measured(raw.free),
                                    measured(raw.free_inodes), measured(raw.mounted), measured(raw.writable),
                                    measured(raw.network), measured(allowed))
            if self.telemetry:
                blocked = (raw.writable is not True or raw.mounted is not True or raw.free is None
                           or raw.capacity is None or raw.free == 0 or raw.free_inodes == 0
                           or (sqlite_state and allowed is not True))
                why = (PreflightReason.PERMISSION if raw.writable is False or allowed is False
                       else PreflightReason.DISK if raw.free == 0 or raw.free_inodes == 0 or raw.mounted is False
                       else PreflightReason.UNKNOWN if blocked else PreflightReason.DISK)
                self.telemetry.result(result=(PreflightResult.BLOCKED if blocked else PreflightResult.PASS).value,
                                      reason=why.value, component=Component.STORAGE.value)
            return values

    def sample_many(self, paths: tuple[str, ...]) -> tuple[VolumeEvidence, ...]:
        """Retain each path's evidence, including aliases of one actual volume.

        Admission groups known identities into one physical budget while using
        every alias's constraints. Unknown identities also remain separate.
        """
        return tuple(self.sample(path) for path in paths)


def make_volume_provider(*, adapter: VolumeAdapter | None = None,
                         clock: Callable[[], datetime] | None = None,
                         telemetry: PreflightTelemetry | None = None) -> VolumeProvider:
    return VolumeProvider(adapter if adapter is not None else native_adapter(),
                          clock=clock if clock is not None else lambda: datetime.now(UTC),
                          telemetry=telemetry)


class _LinuxQuotaAPI:
    """Read-only, typed libc equivalent of quotactl_fd for block filesystems.

    quotactl_fd has no standard libc wrapper. Instead of guessing architecture
    syscall numbers, use quotactl(2) on an already pinned block device through
    /proc/self/fd. Q_GETFMT uses a uint32 buffer and QCMD from Linux's stable
    uapi/linux/quota.h. Only ESRCH proves this quota type is disabled; ENOSYS,
    EINVAL, EPERM, etc. are unresolved. No quota-changing operation is exposed.
    https://www.man7.org/linux/man-pages/man2/quotactl.2.html
    https://www.man7.org/linux/man-pages/man2/Q_GETFMT.2const.html
    """
    def __init__(self, *, library=None):
        import ctypes as c
        if library is None:
            if not sys.platform.startswith('linux'):
                raise OSError(errno.ENOSYS, 'Quota API unavailable')
            library = c.CDLL(None, use_errno=True)
        self.c = c
        self.call = library.quotactl
        self.call.argtypes = [c.c_int, c.c_char_p, c.c_int, c.POINTER(c.c_uint32)]
        self.call.restype = c.c_int

    def enabled(self, device_fd: int, quota_type: int) -> bool:
        if quota_type not in (0, 1, 2):
            raise ValueError('Invalid quota type')
        fmt = self.c.c_uint32()
        op = self.c.c_int((0x800004 << 8) | quota_type).value  # QCMD(Q_GETFMT, type)
        self.c.set_errno(0)
        result = self.call(op, f'/proc/self/fd/{device_fd}'.encode('ascii'), 0,
                           self.c.byref(fmt))
        if result == 0:
            return True
        code = self.c.get_errno()
        if result == -1 and code == errno.ESRCH:
            return False
        raise OSError(code or errno.EIO, 'Quota state unavailable')


class PosixAdapter:
    """dir_fd traversal and bounded, read-only Linux quota evidence.

    quota receives (fd, filesystem, mount options, physical capacity, bavail,
    favail) and returns caller-safe (capacity, free, inodes), or None. The native
    default checks ext4 user/group/project enablement using Q_GETFMT. Enabled
    limits are not yet resolved (including inherited group/project ownership),
    so any enabled or unresolved quota leaves all allocation counts Unknown.
    Missing mount options never prove quotas disabled. tmpfs also supports
    quotas on recent kernels and is deliberately not a statvfs-only exception.
    Counts are point-in-time filesystem allocation evidence, not a guarantee
    against later quota changes, concurrent allocations, or memory exhaustion.
    """
    def __init__(self, *, os_api=os, mount_reader: Callable[[], str] | None = None,
                 quota: Callable | None = None, quota_api=None):
        self.os, self.mount_reader, self.quota = os_api, mount_reader, quota
        self.quota_api = quota_api

    def split_absolute(self, path):
        if type(path) is not str or not path.startswith('/') or path.startswith('//'):
            raise StorageError()
        return '/', relative_components(path[1:]) if path != '/' else ()

    def open_anchor(self, anchor):
        return self.os.open(anchor, self.os.O_RDONLY | self.os.O_DIRECTORY | self.os.O_NOFOLLOW | self.os.O_CLOEXEC)

    def open_child(self, parent, name, directory):
        flags = self.os.O_RDONLY | self.os.O_NOFOLLOW | self.os.O_CLOEXEC | self.os.O_NONBLOCK
        if directory:
            flags |= self.os.O_DIRECTORY
        return self.os.open(name, flags, dir_fd=parent)

    def info(self, handle):
        s = self.os.fstat(handle)
        return HandleInfo(FileIdentity(f'posix:{s.st_dev}', str(s.st_ino)),
                          stat.S_ISDIR(s.st_mode), stat.S_ISREG(s.st_mode), s.st_nlink,
                          stat.S_ISLNK(s.st_mode))

    def edge_matches(self, parent, name, child):
        s = self.os.stat(name, dir_fd=parent, follow_symlinks=False)
        return (not stat.S_ISLNK(s.st_mode)
                and self.info(child).identity == FileIdentity(f'posix:{s.st_dev}', str(s.st_ino)))

    def close(self, handle):
        self.os.close(handle)

    def read(self, handle, limit):
        return self.os.read(handle, limit)

    def _access(self, handle):
        # Effective-identity ACL evidence at sample time, not a guarantee that a
        # later write succeeds (quota, races and allocation can still fail).
        try:
            return self.os.access('.', self.os.W_OK | self.os.X_OK,
                                  dir_fd=handle, effective_ids=True,
                                  follow_symlinks=False)
        except (AttributeError, NotImplementedError, OSError):
            return None

    def _mounts(self):
        if self.mount_reader:
            text = self.mount_reader()
        else:
            with open('/proc/self/mountinfo', encoding='utf-8') as stream:
                text = stream.read(1048577)
        if len(text) > 1048576:
            raise StorageError(ReasonCode.STORAGE_UNKNOWN)
        return text.splitlines()

    def _native_quota(self, handle, filesystem, sources, physical):
        # The documented quotactl equivalent requires a backing block device.
        # ext4 implements all three generic quota types. XFS, btrfs, overlay,
        # remote filesystems and tmpfs need different evidence and stay Unknown.
        if filesystem != 'ext4' or len(sources) != 1:
            return None
        source = next(iter(sources))
        source = re.sub(r'\\(040|011|012|134)',
                        lambda match: chr(int(match[1], 8)), source)
        if not source.startswith('/') or '\x00' in source:
            return None
        try:
            api = self.quota_api if self.quota_api is not None else _LinuxQuotaAPI()
            # O_PATH does not read the device, require block read permission,
            # or trigger device I/O. Device aliases may resolve to a block inode;
            # only its verified st_rdev binds it to the held target filesystem.
            device_fd = self.os.open(source, self.os.O_PATH | self.os.O_CLOEXEC)
            try:
                device = self.os.fstat(device_fd)
                if (not stat.S_ISBLK(device.st_mode)
                        or device.st_rdev != self.os.fstat(handle).st_dev):
                    return None
                states = tuple(api.enabled(device_fd, kind) for kind in (0, 1, 2))
                if all(state is False for state in states):
                    return physical
            finally:
                self.os.close(device_fd)
        except (AttributeError, NotImplementedError, OSError):
            # Includes absent libc symbols, unavailable /dev paths in containers,
            # seccomp/LSM denial and kernels without a usable quota API.
            pass
        return None

    def probe(self, handle):
        s = self.os.fstat(handle)
        v = self.os.fstatvfs(handle)
        device = f'{self.os.major(s.st_dev)}:{self.os.minor(s.st_dev)}'
        matches = []
        for line in self._mounts():
            before, separator, after = line.partition(' - ')
            fields, details = before.split(), after.split()
            if separator and len(fields) >= 6 and len(details) >= 3 and fields[2] == device:
                matches.append((details[0], frozenset((fields[5] + ',' + details[2]).split(',')), details[1]))
        if not matches:
            return RawVolume(None, None, None, False, None, None, 'linux_mountinfo_statvfs')
        filesystem = matches[0][0]
        options = frozenset().union(*(entry[1] for entry in matches))
        network = filesystem in {'nfs', 'nfs4', 'cifs', 'smb3', '9p', 'ceph', 'afs', 'glusterfs'} or filesystem.startswith('fuse')
        known_local = filesystem in {'ext2', 'ext3', 'ext4', 'xfs', 'btrfs', 'tmpfs', 'ramfs', 'overlay', 'zfs', 'vfat', 'exfat', 'ntfs3'}
        writable = False if 'ro' in options or v.f_flag & self.os.ST_RDONLY else self._access(handle)
        capacity, free, inodes = None, None, None
        source = 'linux_mountinfo_statvfs'
        if self.quota is not None:
            proved = self.quota(handle, filesystem, options, v.f_blocks * v.f_frsize,
                                v.f_bavail * v.f_frsize, v.f_favail)
        else:
            source = 'linux_mountinfo_statvfs_quotactl'
            proved = self._native_quota(handle, filesystem, {entry[2] for entry in matches},
                                       (v.f_blocks * v.f_frsize, v.f_bavail * v.f_frsize, v.f_favail))
        if proved is not None:
            capacity, free, inodes = proved
        return RawVolume(capacity, free, inodes, True, writable,
                         network if network or known_local else None, source)


class _DarwinStatfsAPI:
    """Typed libc fstatfs(2) for macOS, bound to the held directory descriptor.

    Layout is the 64-bit-inode struct statfs from <sys/mount.h> (2168 bytes).
    arm64 exports it as fstatfs; x86_64 (including Rosetta) as fstatfs$INODE64.
    Only f_fstypename and f_flags are returned; counts come from fstatvfs.
    https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fstatfs.2.html
    """
    def __init__(self, *, library=None, machine: str | None = None):
        import ctypes as c
        if library is None:
            if sys.platform != 'darwin':
                raise OSError(errno.ENOSYS, 'statfs API unavailable')
            library = c.CDLL(None, use_errno=True)
        if machine is None:
            import platform
            machine = platform.machine()

        class Statfs(c.Structure):
            _fields_ = [('f_bsize', c.c_uint32), ('f_iosize', c.c_int32),
                        ('f_blocks', c.c_uint64), ('f_bfree', c.c_uint64),
                        ('f_bavail', c.c_uint64), ('f_files', c.c_uint64),
                        ('f_ffree', c.c_uint64), ('f_fsid', c.c_int32 * 2),
                        ('f_owner', c.c_uint32), ('f_type', c.c_uint32),
                        ('f_flags', c.c_uint32), ('f_fssubtype', c.c_uint32),
                        ('f_fstypename', c.c_char * 16), ('f_mntonname', c.c_char * 1024),
                        ('f_mntfromname', c.c_char * 1024), ('f_flags_ext', c.c_uint32),
                        ('f_reserved', c.c_uint32 * 7)]

        self.c, self.Statfs = c, Statfs
        name = 'fstatfs$INODE64' if machine == 'x86_64' else 'fstatfs'
        self.call = getattr(library, name)
        self.call.argtypes = [c.c_int, c.POINTER(Statfs)]
        self.call.restype = c.c_int

    def __call__(self, fd: int) -> tuple[str, int]:
        result = self.Statfs()
        self.c.set_errno(0)
        if self.call(fd, self.c.byref(result)) != 0:
            raise OSError(self.c.get_errno() or errno.EIO, 'statfs unavailable')
        name = result.f_fstypename.decode('ascii', 'replace')
        if not re.fullmatch(r'[a-z0-9_-]{1,15}', name):
            raise OSError(errno.EIO, 'statfs layout unrecognised')
        return name, int(result.f_flags)


class DarwinAdapter(PosixAdapter):
    """macOS: PosixAdapter dir_fd traversal, statfs/statvfs volume evidence.

    No /proc and no O_PATH. Identity stays posix:<st_dev> from fstat. statfs
    gives the filesystem type and MNT_RDONLY/MNT_LOCAL flags of the held
    descriptor's mount; fstatvfs gives the counts as the kernel reports them
    (macOS user quotas are not resolved: proportionate for a personal Mac).
    Only MNT_LOCAL APFS/HFS+/msdos/exfat/ufs mounts are network=False; remote,
    FUSE or non-MNT_LOCAL mounts are network=True and unknown types stay None,
    so sqlite state is allowed only on known local volumes.
    statfs is injectable: (fd) -> (fstypename, f_flags).
    """
    MNT_RDONLY, MNT_LOCAL = 0x1, 0x1000
    REMOTE = frozenset({'smbfs', 'nfs', 'afpfs', 'webdav', 'ftp', 'cifs'})
    LOCAL = frozenset({'apfs', 'hfs', 'msdos', 'exfat', 'ufs'})

    def __init__(self, *, os_api=os, statfs: Callable[[int], tuple[str, int]] | None = None):
        super().__init__(os_api=os_api)
        self.statfs = statfs

    def probe(self, handle):
        v = self.os.fstatvfs(handle)
        filesystem = flags = None
        try:
            if self.statfs is None:
                self.statfs = _DarwinStatfsAPI()  # Bound once; ctypes load only on darwin.
            filesystem, flags = self.statfs(handle)
        except (AttributeError, NotImplementedError, OSError, TypeError, ValueError):
            pass
        if type(filesystem) is not str or type(flags) is not int:
            filesystem = flags = None
        if filesystem is None:
            network = None
        elif (filesystem in self.REMOTE or filesystem.startswith(('fuse', 'osxfuse', 'macfuse'))
                or not flags & self.MNT_LOCAL):
            network = True
        else:
            network = False if filesystem in self.LOCAL else None
        read_only = v.f_flag & self.os.ST_RDONLY or (flags is not None and flags & self.MNT_RDONLY)
        writable = False if read_only else self._access(handle)
        return RawVolume(v.f_blocks * v.f_frsize, v.f_bavail * v.f_frsize, v.f_favail,
                         True if filesystem is not None else None, writable, network,
                         'darwin_statfs_statvfs')


class WindowsAdapter:
    """Typed Win32 bindings, 128-bit file IDs and retained no-delete ancestors.

    FILE_SHARE_READ only also prevents acquiring write handles while admitted.
    GetDiskFreeSpaceExW reports caller quota capacity/free, in 64-bit values.
    """
    def __init__(self, *, api=None):
        if api is not None:
            self.api = api
            return
        self.api = _WindowsAPI()

    def split_absolute(self, path):
        if type(path) is not str or path.startswith(('\\\\', '//')):
            raise StorageError()
        drive, tail = ntpath.splitdrive(path)
        if not re.fullmatch(r'[A-Za-z]:', drive) or not tail.startswith(('\\', '/')):
            raise StorageError()
        return drive + '\\', relative_components(tail[1:]) if len(tail) > 1 else ()

    def open_anchor(self, anchor):
        return self.api.open(anchor, True)

    def open_child(self, parent, name, directory):
        # Parent and all its ancestors stay held without FILE_SHARE_DELETE.
        path = self.api.path(parent)
        return self.api.open(ntpath.join(path, name), directory)

    def info(self, handle):
        return self.api.info(handle)

    def edge_matches(self, parent, name, child):
        return ntpath.normcase(self.api.path(child)) == ntpath.normcase(ntpath.join(self.api.path(parent), name))

    def close(self, handle):
        self.api.close(handle)

    def read(self, handle, limit):
        return self.api.read(handle, limit)

    def probe(self, handle):
        return self.api.probe(handle)


class _WindowsAPI:
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w
        self.c, self.w = c, w
        self.k = c.WinDLL('kernel32', use_last_error=True)
        self.permissions = {}

        class ByHandle(c.Structure):
            _fields_ = [('attributes', w.DWORD), ('created', w.FILETIME), ('accessed', w.FILETIME),
                        ('written', w.FILETIME), ('serial', w.DWORD), ('size_high', w.DWORD),
                        ('size_low', w.DWORD), ('links', w.DWORD), ('index_high', w.DWORD), ('index_low', w.DWORD)]

        class FileId(c.Structure):
            _fields_ = [('serial', c.c_ulonglong), ('file_id', c.c_ubyte * 16)]

        self.ByHandle, self.FileId = ByHandle, FileId
        signatures = {
            'CreateFileW': ([w.LPCWSTR, w.DWORD, w.DWORD, c.c_void_p, w.DWORD, w.DWORD, w.HANDLE], w.HANDLE),
            'GetFileInformationByHandle': ([w.HANDLE, c.POINTER(ByHandle)], w.BOOL),
            'GetFileInformationByHandleEx': ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            'GetFinalPathNameByHandleW': ([w.HANDLE, w.LPWSTR, w.DWORD, w.DWORD], w.DWORD),
            'GetDiskFreeSpaceExW': ([w.LPCWSTR, c.POINTER(c.c_ulonglong), c.POINTER(c.c_ulonglong), c.POINTER(c.c_ulonglong)], w.BOOL),
            'GetDriveTypeW': ([w.LPCWSTR], w.UINT),
            'GetVolumeInformationW': ([w.LPCWSTR, w.LPWSTR, w.DWORD, c.POINTER(w.DWORD), c.POINTER(w.DWORD), c.POINTER(w.DWORD), w.LPWSTR, w.DWORD], w.BOOL),
            'GetFileType': ([w.HANDLE], w.DWORD),
            'ReadFile': ([w.HANDLE, c.c_void_p, w.DWORD, c.POINTER(w.DWORD), c.c_void_p], w.BOOL),
            'CloseHandle': ([w.HANDLE], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.k, name)
            function.argtypes, function.restype = args, result

    def _check(self, result):
        if not result:
            code = self.c.get_last_error()
            if code in (2, 3):
                raise FileNotFoundError()
            if code == 5:
                raise PermissionError()
            raise OSError()
        return result

    def open(self, path, directory):
        # FILE_ADD_FILE checks the ACL on this held directory, without creation.
        # Keep permission evidence on this same handle: a second write-access
        # handle would conflict with our deliberate absence of FILE_SHARE_WRITE.
        permission = None
        handle = self.k.CreateFileW(path, 0x82 if directory else 0x80000000,
                                    1, None, 3, 0x00200000 | 0x02000000, None)
        if directory:
            permission = True
            if handle == self.c.c_void_p(-1).value:
                permission = False if self.c.get_last_error() == 5 else None
                handle = self.k.CreateFileW(path, 0x80, 1, None, 3,
                                           0x00200000 | 0x02000000, None)
        if handle == self.c.c_void_p(-1).value:
            self._check(False)
        self.permissions[handle] = permission
        return handle

    def path(self, handle):
        buffer = self.c.create_unicode_buffer(32768)
        length = self._check(self.k.GetFinalPathNameByHandleW(handle, buffer, len(buffer), 1))
        if length >= len(buffer):
            raise StorageError()
        return buffer.value

    def info(self, handle):
        info, identity = self.ByHandle(), self.FileId()
        self._check(self.k.GetFileInformationByHandle(handle, self.c.byref(info)))
        self._check(self.k.GetFileInformationByHandleEx(handle, 18, self.c.byref(identity), self.c.sizeof(identity)))
        path = self.path(handle)
        match = re.match(r'(?i)^(\\\\\?\\Volume\{[0-9a-f-]+\}\\)', path)
        if not match:
            raise StorageError()
        directory = bool(info.attributes & 0x10)
        return HandleInfo(FileIdentity(match[1].lower(), bytes(identity.file_id).hex()), directory,
                          not directory and self.k.GetFileType(handle) == 1, info.links,
                          bool(info.attributes & 0x400))

    def close(self, handle):
        self._check(self.k.CloseHandle(handle))
        self.permissions.pop(handle, None)

    def read(self, handle, limit):
        buffer, count = self.c.create_string_buffer(limit), self.w.DWORD()
        self._check(self.k.ReadFile(handle, buffer, limit, self.c.byref(count), None))
        return buffer.raw[:count.value]

    def probe(self, handle):
        path = self.path(handle)
        identity = self.info(handle).identity
        available, capacity, physical_free = (self.c.c_ulonglong() for _ in range(3))
        self._check(self.k.GetDiskFreeSpaceExW(path, self.c.byref(available), self.c.byref(capacity), self.c.byref(physical_free)))
        kind = self.k.GetDriveTypeW(identity.volume)
        flags = self.w.DWORD()
        fs = self.c.create_unicode_buffer(64)
        self._check(self.k.GetVolumeInformationW(identity.volume, None, 0, None, None, self.c.byref(flags), fs, len(fs)))
        network = kind == 4 or fs.value.lower() == 'csvfs'
        known_local = kind in (2, 3, 5, 6)
        # Permission evidence only; subsequent allocation can still fail.
        writable = False if flags.value & 0x80000 else self.permissions.get(handle)
        return RawVolume(capacity.value, available.value, None, kind not in (0, 1),
                         writable, network if network or known_local else None, 'win32_caller_quota')


def native_adapter(*, platform: str | None = None) -> VolumeAdapter:
    platform = sys.platform if platform is None else platform
    if platform == 'win32':
        return WindowsAdapter()
    return DarwinAdapter() if platform == 'darwin' else PosixAdapter()
