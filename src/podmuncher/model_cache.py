"""Strict HTTPS acquisition into a bounded, verified managed model cache."""

from __future__ import annotations

import hashlib
import hmac
import http.client
import ipaddress
import json
import ntpath
import os
import re
import socket
import ssl
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from typing import NoReturn, Protocol, Self
from urllib.parse import urljoin, urlsplit, urlunsplit

from .domains import DomainValidationError, ReasonCode
from .manifests import ManifestFile, ManifestModel

MAX_METADATA_BYTES = 1024 * 1024
MAX_CACHE_ENTRIES = 256
MAX_AUDIT_LOSS_EVENTS = 32
MAX_SPANS_DROPPED = 1_000_000
_ID = re.compile(r"snapshot-[0-9a-f]{32}\Z")
_MANIFEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REPARSE_POINT = 0x400
_MAX_CA_BYTES = 1024 * 1024
_METADATA_BACKUP = ".metadata-transaction.backup"
_METADATA_MARKER = ".metadata-transaction.json"
_INTERNAL_PATH_PREFIXES = (
    ".metadata.",
    ".metadata-control.",
    ".metadata-old.",
    ".metadata-transaction.",
    ".remove.",
    ".retained-directory.",
    ".retained-file.",
)
_RESOLVER_CLEANUP_SECONDS = 0.25
_HTTPS_CONNECTION_TYPE = http.client.HTTPSConnection
_REDIRECT_STATUSES = frozenset((301, 302, 303, 307, 308))
_MAX_REDIRECTS = 5
_NAT64_PREFIX = ipaddress.IPv6Network("64:ff9b::/96")
_RESOLVER_SCRIPT = (
    "import json,socket,sys;"
    "host,port=json.load(sys.stdin);"
    "json.dump(socket.getaddrinfo(host,port,type=socket.SOCK_STREAM),sys.stdout)"
)


def _run_cleanups(*cleanups: Callable[[], object]) -> None:
    """Attempt every cleanup without replacing an exception already in flight."""
    active = sys.exception()
    first_error: BaseException | None = None
    for cleanup in cleanups:
        try:
            cleanup()
        except BaseException as error:  # noqa: BLE001 - preserves active exception.
            if first_error is None:
                first_error = error
    if active is None and first_error is not None:
        raise first_error


def _raise_after_cleanups(
    error: BaseException, *cleanups: Callable[[], object]
) -> NoReturn:
    try:
        raise error
    finally:
        _run_cleanups(*cleanups)


class _DroppableSpan(AbstractContextManager[None]):
    """Wrap a sink span so its ``__enter__``/``__exit__`` failures drop the span.

    The wrapped scope never suppresses the caller's exception and a sink that
    raises is treated as dropped telemetry rather than a cache failure. Every
    dropped span is reported through ``on_drop`` so the owner can count it.
    """

    def __init__(
        self, scope: AbstractContextManager[object], on_drop: Callable[[], None]
    ) -> None:
        self._scope = scope
        self._on_drop = on_drop
        self._entered = False

    def __enter__(self) -> None:
        try:
            self._scope.__enter__()
        except Exception:  # noqa: BLE001 - telemetry is optional; drop the span.
            self._on_drop()
            return
        self._entered = True

    def __exit__(self, *exc_info: object) -> bool:
        if self._entered:
            try:
                self._scope.__exit__(*exc_info)
            except Exception:  # noqa: BLE001 - telemetry is optional; drop the span.
                self._on_drop()
        return False


class CacheError(DomainValidationError):
    """A closed cache failure that never forwards transport or path text."""

    def __init__(self, reason_code: ReasonCode):
        super().__init__(ReasonCode.parse(reason_code))


class _MetadataCommittedError(BaseException):
    """Carries a post-commit failure without allowing caller rollback."""

    def __init__(self, error: BaseException):
        self.error = error


def _raise_initialization_error(error: BaseException) -> NoReturn:
    if isinstance(error, (CacheError, KeyboardInterrupt, SystemExit)):
        raise error
    if isinstance(error, (OSError, ValueError, TypeError)):
        raise CacheError(ReasonCode.CACHE_UNKNOWN) from None
    raise error


class AuditSink(Protocol):
    def record(
        self,
        *,
        action: str,
        reason: str,
        asset_id: str,
        failure_code: str,
    ) -> None: ...


class SpanSink(Protocol):
    def span(self, name: str, *, asset_id: str) -> AbstractContextManager[object]: ...


Admission = Callable[..., bool]


@dataclass(frozen=True, slots=True)
class CacheSnapshot:
    snapshot_id: str
    path: Path
    total_bytes: int
    manifest_digest: str
    files: tuple[ManifestFile, ...]


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _is_reparse(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)


def _safe_directory(path: Path) -> tuple[int, int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
        raise CacheError(ReasonCode.CACHE_UNKNOWN)
    return _identity(info)


def _safe_regular(path: Path, *, reason: ReasonCode) -> os.stat_result:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or _is_reparse(info):
        raise CacheError(reason)
    return info


@dataclass(frozen=True, slots=True)
class _Directory:
    path: Path
    handle: int
    identity: tuple[int, int]


class _HTTPResponseOwner:
    """Own a response, connection, and absolute-deadline timer as one resource."""

    def __init__(
        self,
        response: http.client.HTTPResponse,
        connection: http.client.HTTPSConnection,
        timer: threading.Timer,
        url: str,
    ) -> None:
        self._response = response
        self._connection = connection
        self._timer = timer
        self._url = url
        self._closed = False
        self.status = response.status
        self.headers = response.headers

    def geturl(self) -> str:
        return self._url

    def read1(self, size: int) -> bytes:
        return self._response.read1(size)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cleanups = [self._timer.cancel, self._response.close, self._connection.close]
        if self._timer is not threading.current_thread():
            cleanups.append(self._timer.join)
        _run_cleanups(*cleanups)


class _OwnedTLSContext:
    """Expose the TLS socket before its blocking handshake begins."""

    def __init__(
        self,
        context: ssl.SSLContext,
        transports: list[object],
        remaining: Callable[[], float],
    ) -> None:
        self._context = context
        self._transports = transports
        self._remaining = remaining

    def __getattr__(self, name: str) -> object:
        return getattr(self._context, name)

    def wrap_socket(self, sock: socket.socket, *args: object, **kwargs: object):
        kwargs["do_handshake_on_connect"] = False
        tls_socket = self._context.wrap_socket(sock, *args, **kwargs)
        self._transports.append(tls_socket)
        tls_socket.settimeout(self._remaining())
        tls_socket.do_handshake()
        return tls_socket


class _Containment:
    """Small platform boundary for cache namespace operations."""

    def __init__(self, root: Path) -> None:
        if os.name not in ("posix", "nt"):
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        self.root = root
        self._closed = False
        self._directories: dict[tuple[str, ...], _Directory] = {}
        self._active_directories: dict[tuple[str, ...], _Directory] = {}
        self._posix_ancestor_handles: list[int] = []
        if os.name == "posix":
            root_directory = self._ensure_posix_root()
        else:
            self._ensure_root()
            root_directory = self._open_directory_path(root)
        self._directories[()] = root_directory
        try:
            for name in ("snapshots", "staging", "quarantine"):
                self.mkdir((), name, exist_ok=True)
                self._directories[(name,)] = self._open_directory((name,))
        except BaseException:
            _run_cleanups(self.close)
            raise

    def _ensure_root(self) -> None:
        missing: list[str] = []
        current = self.root
        while True:
            try:
                _safe_directory(current)
                break
            except FileNotFoundError:
                if current.parent == current:
                    raise CacheError(ReasonCode.CACHE_UNKNOWN) from None
                missing.append(current.name)
                current = current.parent
        parent = self._open_directory_path(current)
        try:
            for name in reversed(missing):
                self._mkdir(parent, name, exist_ok=False)
                child = self._open_child_directory(parent, name)
                self._close_directory(parent)
                parent = child
        finally:
            self._close_directory(parent)

    def _ensure_posix_root(self) -> _Directory:
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
        if not directory_flag or not nofollow_flag or not self.root.is_absolute():
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        opened: list[int] = []
        try:
            current = os.open(self.root.anchor, os.O_RDONLY | directory_flag | nofollow_flag)
            opened.append(current)
            for name in self.root.parts[1:]:
                try:
                    child = os.open(
                        name,
                        os.O_RDONLY | directory_flag | nofollow_flag,
                        dir_fd=current,
                    )
                except FileNotFoundError:
                    os.mkdir(name, 0o700, dir_fd=current)
                    child = os.open(
                        name,
                        os.O_RDONLY | directory_flag | nofollow_flag,
                        dir_fd=current,
                    )
                opened.append(child)
                current = child
            info = os.fstat(current)
            if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            self._posix_ancestor_handles = opened[:-1]
            opened = []
            return _Directory(self.root, current, _identity(info))
        finally:
            _run_cleanups(*(lambda value=value: os.close(value) for value in reversed(opened)))

    @staticmethod
    def _close_windows_handle(handle: int) -> None:
        import ctypes
        from ctypes import wintypes

        close = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        if not close(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    @staticmethod
    def _windows_path(handle: int) -> str:
        import ctypes
        from ctypes import wintypes

        function = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).GetFinalPathNameByHandleW
        function.argtypes = (
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        function.restype = wintypes.DWORD
        needed = function(handle, None, 0, 0)
        if needed == 0:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_unicode_buffer(needed + 1)
        written = function(handle, buffer, len(buffer), 0)
        if written == 0 or written >= len(buffer):
            raise OSError("invalid final path")
        value = buffer.value.replace("/", "\\")
        lowered = value.lower()
        if lowered.startswith("\\\\?\\unc\\"):
            value = "\\\\" + value[8:]
        elif lowered.startswith("\\\\?\\"):
            value = value[4:]
        elif lowered.startswith("\\??\\unc\\"):
            value = "\\\\" + value[8:]
        elif lowered.startswith("\\??\\"):
            value = value[4:]
        elif lowered.startswith("\\\\.\\"):
            raise OSError("device namespace rejected")
        value = ntpath.normcase(ntpath.normpath(value))
        if not ntpath.isabs(value):
            raise OSError("non-absolute final path")
        return value

    @staticmethod
    def _windows_open(path: Path, access: int, *, directory: bool) -> int:
        import ctypes
        from ctypes import wintypes

        create = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
        create.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create.restype = wintypes.HANDLE
        flags = 0x00200000  # FILE_FLAG_OPEN_REPARSE_POINT
        if directory:
            flags |= 0x02000000  # FILE_FLAG_BACKUP_SEMANTICS
        handle = create(
            os.fspath(path),
            access,
            0x1 | 0x2,
            None,
            3,
            flags,
            None,
        )
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    @staticmethod
    def _windows_identity(handle: int) -> tuple[int, int]:
        import ctypes
        from ctypes import wintypes

        class Information(ctypes.Structure):
            _fields_ = [
                ("attributes", wintypes.DWORD),
                ("creation", wintypes.FILETIME),
                ("access", wintypes.FILETIME),
                ("write", wintypes.FILETIME),
                ("volume", wintypes.DWORD),
                ("size_high", wintypes.DWORD),
                ("size_low", wintypes.DWORD),
                ("links", wintypes.DWORD),
                ("index_high", wintypes.DWORD),
                ("index_low", wintypes.DWORD),
            ]

        function = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).GetFileInformationByHandle
        function.argtypes = (wintypes.HANDLE, ctypes.POINTER(Information))
        function.restype = wintypes.BOOL
        value = Information()
        if not function(handle, ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.volume, (value.index_high << 32) | value.index_low

    def _expected_windows_path(self, parts: tuple[str, ...]) -> str:
        root_path = ntpath.normcase(ntpath.normpath(os.path.abspath(self.root)))
        value = ntpath.normcase(ntpath.normpath(ntpath.join(root_path, *parts)))
        if parts and not value.startswith(root_path.rstrip("\\") + "\\"):
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        return value

    def _verify_windows_handle(self, handle: int, parts: tuple[str, ...]) -> None:
        root = self._directories.get(())
        if root is not None and self._windows_path(root.handle) != self._expected_windows_path(()) :
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        if self._windows_path(handle) != self._expected_windows_path(parts):
            raise CacheError(ReasonCode.CACHE_UNKNOWN)

    def _open_directory_path(self, path: Path) -> _Directory:
        if os.name == "posix":
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            if not nofollow or not getattr(os, "O_DIRECTORY", 0):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            handle = os.open(path, flags | nofollow)
        else:
            handle = self._windows_open(
                path, 0x10000 | 0x80 | 0x1 | 0x2 | 0x4 | 0x100000, directory=True
            )
        try:
            info = path.stat(follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            if os.name == "nt":
                expected = ntpath.normcase(ntpath.normpath(os.path.abspath(path)))
                if self._windows_path(handle) != expected:
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
            return _Directory(path, handle, _identity(info))
        except BaseException:
            _run_cleanups(
                lambda: (
                    os.close(handle)
                    if os.name == "posix"
                    else self._close_windows_handle(handle)
                )
            )
            raise

    def _close_directory(self, directory: _Directory) -> None:
        if os.name == "posix":
            os.close(directory.handle)
        else:
            self._close_windows_handle(directory.handle)

    def _open_child_directory(self, parent: _Directory, name: str) -> _Directory:
        path = parent.path / name
        if os.name == "posix":
            handle = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent.handle,
            )
            info = os.fstat(handle)
        else:
            root = self._directories.get(())
            if root is None:
                expected = ntpath.normcase(
                    ntpath.normpath(os.path.abspath(parent.path))
                )
                if self._windows_path(parent.handle) != expected:
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
            else:
                self._verify_windows_handle(parent.handle, self._parts(parent.path))
            handle = self._windows_open(
                path, 0x10000 | 0x80 | 0x1 | 0x2 | 0x4 | 0x100000, directory=True
            )
        try:
            if os.name == "nt":
                info = path.stat(follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            result = _Directory(path, handle, _identity(info))
            if os.name == "nt":
                if self._directories.get(()) is None:
                    expected = ntpath.normcase(ntpath.normpath(os.path.abspath(path)))
                    if self._windows_path(handle) != expected:
                        raise CacheError(ReasonCode.CACHE_UNKNOWN)
                else:
                    self._verify_windows_handle(handle, self._parts(path))
            return result
        except BaseException:
            _run_cleanups(
                lambda: (
                    os.close(handle)
                    if os.name == "posix"
                    else self._close_windows_handle(handle)
                )
            )
            raise

    def _parts(self, path: Path) -> tuple[str, ...]:
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            raise CacheError(ReasonCode.CACHE_UNKNOWN) from None
        return () if relative == Path(".") else relative.parts

    @contextmanager
    def directory(self, parts: tuple[str, ...]):
        retained = self._directories.get(parts) or self._active_directories.get(parts)
        if retained is not None:
            self.verify(retained)
            yield retained
            return
        available = self._directories | self._active_directories
        prefix = max(
            (key for key in available if parts[: len(key)] == key),
            key=len,
        )
        current = available[prefix]
        opened: list[_Directory] = []
        try:
            for index, name in enumerate(parts[len(prefix) :], start=len(prefix) + 1):
                child = self._open_child_directory(current, name)
                opened.append(child)
                self._active_directories[parts[:index]] = child
                current = child
            yield current
        finally:
            for index, item in reversed(tuple(enumerate(opened, start=len(prefix) + 1))):
                self._active_directories.pop(parts[:index], None)
            _run_cleanups(
                *(lambda value=value: self._close_directory(value) for value in reversed(opened))
            )

    def _open_directory(self, parts: tuple[str, ...]) -> _Directory:
        current = self._directories[()]
        opened: list[_Directory] = []
        try:
            for name in parts:
                child = self._open_child_directory(current, name)
                opened.append(child)
                current = child
            for item in opened[:-1]:
                self._close_directory(item)
            return current
        except BaseException:
            _run_cleanups(
                *(lambda value=value: self._close_directory(value) for value in reversed(opened))
            )
            raise

    def verify(self, directory: _Directory) -> None:
        if os.name == "posix":
            info = os.fstat(directory.handle)
            current = directory.path.stat(follow_symlinks=False)
            if _identity(current) != directory.identity:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
        else:
            self._verify_windows_handle(directory.handle, self._parts(directory.path))
            info = directory.path.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(info.st_mode)
            or _is_reparse(info)
            or _identity(info) != directory.identity
        ):
            raise CacheError(ReasonCode.CACHE_UNKNOWN)

    def _mkdir(self, parent: _Directory, name: str, *, exist_ok: bool) -> None:
        try:
            if os.name == "posix":
                os.mkdir(name, 0o700, dir_fd=parent.handle)
            else:
                root = self._directories.get(())
                if root is None:
                    expected = ntpath.normcase(
                        ntpath.normpath(os.path.abspath(parent.path))
                    )
                    if self._windows_path(parent.handle) != expected:
                        raise CacheError(ReasonCode.CACHE_UNKNOWN)
                else:
                    self._verify_windows_handle(parent.handle, self._parts(parent.path))
                os.mkdir(parent.path / name, 0o700)
        except FileExistsError:
            if not exist_ok:
                raise
        child = self._open_child_directory(parent, name)
        self._close_directory(child)

    def mkdir(
        self, parent_parts: tuple[str, ...], name: str, *, exist_ok: bool = False
    ) -> None:
        with self.directory(parent_parts) as parent:
            self._mkdir(parent, name, exist_ok=exist_ok)

    def stat(
        self, parent_parts: tuple[str, ...], name: str
    ) -> os.stat_result:
        with self.directory(parent_parts) as parent:
            if os.name == "posix":
                return os.stat(name, dir_fd=parent.handle, follow_symlinks=False)
            handle = self._windows_open(parent.path / name, 0x80, directory=True)
            try:
                self._verify_windows_handle(handle, parent_parts + (name,))
                return (parent.path / name).stat(follow_symlinks=False)
            finally:
                _run_cleanups(lambda: self._close_windows_handle(handle))

    def exists(self, parent_parts: tuple[str, ...], name: str) -> bool:
        try:
            self.stat(parent_parts, name)
            return True
        except FileNotFoundError:
            return False

    def open_file(
        self,
        parent_parts: tuple[str, ...],
        name: str,
        flags: int,
        mode: int = 0o600,
    ) -> int:
        with self.directory(parent_parts) as parent:
            safe_flags = flags | getattr(os, "O_NOFOLLOW", 0)
            descriptor = (
                os.open(name, safe_flags, mode, dir_fd=parent.handle)
                if os.name == "posix"
                else os.open(parent.path / name, safe_flags, mode)
            )
            try:
                opened = os.fstat(descriptor)
                if _is_reparse(opened):
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
                if os.name == "nt":
                    import msvcrt

                    self._verify_windows_handle(
                        msvcrt.get_osfhandle(descriptor), parent_parts + (name,)
                    )
                return descriptor
            except BaseException:
                _run_cleanups(lambda: os.close(descriptor))
                raise

    def list(self, parts: tuple[str, ...]) -> tuple[str, ...]:
        with self.directory(parts) as directory:
            return self.list_retained(directory)

    def list_retained(self, directory: _Directory) -> tuple[str, ...]:
        self.verify(directory)
        names = tuple(
            os.listdir(directory.handle)
            if os.name == "posix"
            else os.listdir(directory.path)
        )
        self.verify(directory)
        return names

    def stat_retained(self, directory: _Directory, name: str) -> os.stat_result:
        self.verify(directory)
        if os.name == "posix":
            return os.stat(name, dir_fd=directory.handle, follow_symlinks=False)
        handle = self._windows_open(directory.path / name, 0x80, directory=True)
        try:
            self._verify_windows_handle(
                handle, self._parts(directory.path) + (name,)
            )
            return (directory.path / name).stat(follow_symlinks=False)
        finally:
            _run_cleanups(lambda: self._close_windows_handle(handle))

    @contextmanager
    def child_directory(self, parent: _Directory, name: str):
        self.verify(parent)
        child = self._open_child_directory(parent, name)
        try:
            yield child
        finally:
            _run_cleanups(lambda: self._close_directory(child))

    def delete_retained_directory(
        self, parent: _Directory, name: str, child: _Directory
    ) -> None:
        """Delete the verified, already-emptied directory ``name`` under ``parent``.

        POSIX has no delete-by-handle, so the public name is first moved to a
        private random ``.retained-directory.<hex>`` name inside the retained
        parent (relative to its handle, never following links). The moved
        entry is re-opened and pinned, its identity compared with the verified
        child, and only then removed with ``rmdir`` by that private name. A
        swap between the rename and the pinned open fails the identity check
        and leaves the replacement in place under the private name; recovery
        removes such stale tombstones later. Removing the directory frees its
        inode, so a removed snapshot does not linger on disk (the files inside
        were already truncated and unlinked by ``delete_retained_file``).
        """
        self.verify(parent)
        self.verify(child)
        if os.name == "posix":
            current = os.stat(name, dir_fd=parent.handle, follow_symlinks=False)
            if _identity(current) != child.identity:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            private_name = f".retained-directory.{uuid.uuid4().hex}"
            os.rename(
                name,
                private_name,
                src_dir_fd=parent.handle,
                dst_dir_fd=parent.handle,
            )
            moved = None
            try:
                moved = os.open(
                    private_name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent.handle,
                )
                moved_info = os.fstat(moved)
                if (
                    not stat.S_ISDIR(moved_info.st_mode)
                    or _is_reparse(moved_info)
                    or _identity(moved_info) != child.identity
                ):
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
                os.rmdir(private_name, dir_fd=parent.handle)
            finally:
                if moved is not None:
                    _run_cleanups(lambda: os.close(moved))
            return
        self._windows_delete_handle(child.handle)

    def delete_retained_file(
        self, parent: _Directory, name: str, expected_identity: tuple[int, int]
    ) -> None:
        self.verify(parent)
        if os.name == "posix":
            current = os.stat(name, dir_fd=parent.handle, follow_symlinks=False)
            if _identity(current) != expected_identity:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            private_name = f".retained-file.{uuid.uuid4().hex}"
            os.rename(
                name,
                private_name,
                src_dir_fd=parent.handle,
                dst_dir_fd=parent.handle,
            )
            descriptor = None
            try:
                descriptor = os.open(
                    private_name,
                    os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent.handle,
                )
                moved = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(moved.st_mode)
                    or moved.st_nlink != 1
                    or _is_reparse(moved)
                    or _identity(moved) != expected_identity
                ):
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
                # Truncate through the pinned descriptor so the payload bytes
                # are released even if the unlink below is interrupted, then
                # unlink the private name so the entry does not linger.
                os.ftruncate(descriptor, 0)
                os.unlink(private_name, dir_fd=parent.handle)
            finally:
                if descriptor is not None:
                    _run_cleanups(lambda: os.close(descriptor))
            return
        self._windows_delete(
            parent.path / name,
            self._parts(parent.path) + (name,),
            directory=False,
            expected_identity=expected_identity,
        )

    def rename(
        self,
        source_parent: tuple[str, ...],
        source_name: str,
        target_parent: tuple[str, ...],
        target_name: str,
        *,
        replace: bool = False,
    ) -> None:
        with self.directory(source_parent) as source, self.directory(
            target_parent
        ) as target:
            if os.name == "posix":
                os.rename(
                    source_name,
                    target_name,
                    src_dir_fd=source.handle,
                    dst_dir_fd=target.handle,
                )
                return
            self._windows_rename(
                source.path / source_name,
                source_parent + (source_name,),
                target,
                target_name,
                replace=replace,
            )

    def _windows_rename(
        self,
        path: Path,
        source_parts: tuple[str, ...],
        target: _Directory,
        target_name: str,
        *,
        replace: bool = False,
    ) -> None:
        import ctypes
        from ctypes import wintypes

        handle = self._windows_open(path, 0x10000 | 0x80, directory=True)
        try:
            self._verify_windows_handle(handle, source_parts)
            self._verify_windows_handle(target.handle, self._parts(target.path))
            target_parts = self._parts(target.path)
            destination = os.fspath(target.path / target_name)
            function = ctypes.WinDLL(
                "kernel32", use_last_error=True
            ).SetFileInformationByHandle
            function.argtypes = (
                wintypes.HANDLE,
                ctypes.c_int,
                wintypes.LPVOID,
                wintypes.DWORD,
            )
            function.restype = wintypes.BOOL

            def rename_to(name: str, root_handle: int | None) -> int:
                encoded_name = name.encode("utf-16-le")
                code_unit_count = len(encoded_name) // ctypes.sizeof(wintypes.WCHAR)

                class RenameInfo(ctypes.Structure):
                    _fields_ = [
                        ("replace_if_exists", wintypes.BOOLEAN),
                        ("root_directory", wintypes.HANDLE),
                        ("file_name_length", wintypes.DWORD),
                        ("file_name", wintypes.WCHAR * (code_unit_count + 1)),
                    ]

                value = RenameInfo()
                value.replace_if_exists = replace
                value.root_directory = root_handle
                value.file_name_length = len(encoded_name)
                value.file_name = name
                value_size = ctypes.sizeof(value)
                if function(handle, 3, ctypes.byref(value), value_size):
                    return 0
                return ctypes.get_last_error()

            error = rename_to(target_name, target.handle)
            if error == 87:  # Win32 rejects RootDirectory on supported host builds.
                error = rename_to(destination, None)
            if error:
                raise ctypes.WinError(error)
            target_handle = self._windows_open(
                target.path / target_name, 0x80, directory=True
            )
            try:
                self._verify_windows_handle(
                    target_handle, target_parts + (target_name,)
                )
                if self._windows_identity(target_handle) != self._windows_identity(handle):
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
            finally:
                _run_cleanups(lambda: self._close_windows_handle(target_handle))
        finally:
            _run_cleanups(lambda: self._close_windows_handle(handle))

    def unlink(self, parent_parts: tuple[str, ...], name: str) -> None:
        self._delete(parent_parts, name, directory=False)

    def rmdir(self, parent_parts: tuple[str, ...], name: str) -> None:
        self._delete(parent_parts, name, directory=True)

    def _delete(
        self, parent_parts: tuple[str, ...], name: str, *, directory: bool
    ) -> None:
        with self.directory(parent_parts) as parent:
            if os.name == "posix":
                if directory:
                    os.rmdir(name, dir_fd=parent.handle)
                else:
                    os.unlink(name, dir_fd=parent.handle)
                return
            self._windows_delete(
                parent.path / name, parent_parts + (name,), directory=directory
            )

    def _windows_delete(
        self,
        path: Path,
        parts: tuple[str, ...],
        *,
        directory: bool,
        expected_identity: tuple[int, int] | None = None,
    ) -> None:
        handle = self._windows_open(path, 0x10000 | 0x80, directory=directory)
        try:
            self._verify_windows_handle(handle, parts)
            if (
                expected_identity is not None
                # Windows stat().st_dev is not the volume serial returned by
                # GetFileInformationByHandle, while st_ino is the same stable
                # file index. The retained parent pins the volume.
                and self._windows_identity(handle)[1] != expected_identity[1]
            ):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            self._windows_delete_handle(handle)
        finally:
            _run_cleanups(lambda: self._close_windows_handle(handle))

    @staticmethod
    def _windows_delete_handle(handle: int) -> None:
        import ctypes
        from ctypes import wintypes

        class DispositionInfo(ctypes.Structure):
            _fields_ = [("flags", wintypes.DWORD)]

        value = DispositionInfo(0x1 | 0x2 | 0x10)
        function = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).SetFileInformationByHandle
        function.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        function.restype = wintypes.BOOL
        if not function(handle, 21, ctypes.byref(value), ctypes.sizeof(value)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        directories = tuple(reversed(tuple(self._directories.values())))
        ancestor_handles = tuple(reversed(self._posix_ancestor_handles))
        self._directories.clear()
        self._posix_ancestor_handles.clear()
        _run_cleanups(
            *(
                lambda value=value: self._close_directory(value)
                for value in directories
            ),
            *(lambda value=value: os.close(value) for value in ancestor_handles),
        )


def _relative(value: object) -> tuple[str, ...]:
    if type(value) is not str:
        raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
    pure = PurePosixPath(value)
    if (
        not value
        or value.startswith(("/", "\\"))
        or "\\" in value
        or ":" in value
        or pure.is_absolute()
        or str(pure) != value
        or len(pure.parts) > 16
        or any(part in ("", ".", "..") for part in pure.parts)
        or any(
            part.startswith(_INTERNAL_PATH_PREFIXES)
            for part in pure.parts
        )
    ):
        raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
    return pure.parts


def _verified_url(value: object) -> str:
    if type(value) is not str or len(value) > 4096:
        raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED) from None
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or "\\" in value
        or port is not None
        and not 1 <= port <= 65535
    ):
        raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED)
    return value


def _public_flags(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return address.is_global and not any((
        address.is_private,
        address.is_loopback,
        address.is_link_local,
        address.is_multicast,
        address.is_unspecified,
        address.is_reserved,
    ))


def _public_peer(address: object) -> bool:
    """Return whether one resolved or connected peer address is publicly routable."""
    if type(address) is not str or not address or "%" in address:
        return False
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address):
        if (
            parsed.ipv4_mapped is not None
            or int(parsed) >> 32 == 0
            or parsed.is_site_local
        ):
            return False
        embedded: list[ipaddress.IPv4Address] = []
        if parsed.sixtofour is not None:
            embedded.append(parsed.sixtofour)
        if parsed.teredo is not None:
            embedded.extend(parsed.teredo)
        if parsed in _NAT64_PREFIX:
            embedded.append(ipaddress.IPv4Address(int(parsed) & 0xFFFFFFFF))
        if not all(_public_flags(item) for item in embedded):
            return False
    return _public_flags(parsed)


def _redirect_target(current: str, headers: object) -> str:
    get_all = getattr(headers, "get_all", None)
    if not callable(get_all):
        raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED)
    values = get_all("Location") or []
    if len(values) != 1:
        raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED)
    location = values[0]
    if (
        type(location) is not str
        or not location
        or len(location) > 4096
        or any(not 0x21 <= ord(character) <= 0x7E for character in location)
    ):
        raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED)
    try:
        target = urljoin(current, location)
    except ValueError:
        raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED) from None
    return _verified_url(target)


def _snapshot_identity(model: ManifestModel) -> tuple[str, str]:
    declarations = [
        {
            "path": item.relative_path,
            "bytes": item.bytes,
            "sha256": item.sha256,
            "url": url,
        }
        for item, url in zip(model.files, model.source_urls, strict=True)
    ]
    raw = json.dumps(
        {
            "model": model.artifact.model_id,
            "revision": model.artifact.revision,
            "digest": model.artifact.sha256,
            "files": declarations,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return f"snapshot-{digest[:32]}", digest


class ModelCache:
    """Managed immutable snapshots acquired only from typed manifest entries."""

    def __init__(
        self,
        root: str | os.PathLike,
        *,
        audit: AuditSink,
        admission: Admission,
        spans: SpanSink | None = None,
        ca_bundle: str | os.PathLike | None = None,
        max_entries: int = 64,
        chunk_size: int = 1024 * 1024,
        timeout_seconds: float = 30,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            not callable(getattr(audit, "record", None))
            or not callable(admission)
            or type(max_entries) is not int
            or not 1 <= max_entries <= MAX_CACHE_ENTRIES
            or type(chunk_size) is not int
            or not 1 <= chunk_size <= 16 * 1024 * 1024
            or type(timeout_seconds) not in (int, float)
            or not 0 < timeout_seconds <= 3600
            or not callable(monotonic)
        ):
            raise CacheError(ReasonCode.INVALID_INPUT)
        try:
            self.root = Path(os.path.abspath(os.fspath(root)))
        except (TypeError, ValueError, OSError):
            raise CacheError(ReasonCode.INVALID_INPUT) from None
        self.audit = audit
        self.admission = admission
        self.spans = spans
        self.max_entries = max_entries
        self.chunk_size = chunk_size
        self.timeout_seconds = float(timeout_seconds)
        self.monotonic = monotonic
        self._lock = threading.RLock()
        self._closed = False
        self._containment: _Containment | None = None
        self._layout_identities: dict[str, tuple[int, int]] = {}
        self.ca_bundle = None if ca_bundle in (None, "") else Path(ca_bundle)
        self._ca_data = self._read_ca_bundle()
        self._in_use: dict[str, int] = {}
        self._audit_loss: list[tuple[ReasonCode, str]] = []
        self._audit_blocked = False
        self._spans_dropped = 0
        self._snapshots: dict[str, CacheSnapshot] = {}
        try:
            self._validate_root_ancestors()
            self._admission_call("cache", 0, 0, verify=False)
            self._containment = _Containment(self.root)
            self._root_identity = self._containment._directories[()].identity
            for name in ("snapshots", "staging", "quarantine"):
                self._layout_identities[name] = self._containment._directories[
                    (name,)
                ].identity
            self.recover()
            self._audit("initialize", "cache_initialize", "cache", ReasonCode.OK)
        except BaseException as error:
            try:
                self.close()
            except BaseException:  # noqa: BLE001,S110 - preserve original failure.
                pass
            if isinstance(error, _MetadataCommittedError):
                _raise_initialization_error(error.error)
            if isinstance(error, CacheError):
                raise
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(error, (OSError, ValueError, TypeError)):
                raise CacheError(ReasonCode.CACHE_UNKNOWN) from None
            raise

    def _validate_root_ancestors(self) -> None:
        current = self.root
        while True:
            try:
                _safe_directory(current)
            except FileNotFoundError:
                pass
            if current.parent == current:
                return
            current = current.parent

    @property
    def audit_loss_events(self) -> tuple[tuple[ReasonCode, str], ...]:
        with self._lock:
            return tuple(self._audit_loss)

    @property
    def spans_dropped(self) -> int:
        """Bounded count of telemetry spans dropped because the sink failed."""
        with self._lock:
            return self._spans_dropped

    def _count_dropped_span(self) -> None:
        with self._lock:
            if self._spans_dropped < MAX_SPANS_DROPPED:
                self._spans_dropped += 1

    def snapshots(self) -> tuple[CacheSnapshot, ...]:
        with self._lock:
            self._verify_root()
            return tuple(self._snapshots[key] for key in sorted(self._snapshots))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            containment, self._containment = self._containment, None
            if containment is not None:
                try:
                    containment.close()
                except OSError:
                    raise CacheError(ReasonCode.CACHE_UNKNOWN) from None

    def __enter__(self) -> Self:
        if self._closed:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _initialize_root(self) -> None:
        raise AssertionError("initialization is owned by _Containment")

    def _read_ca_bundle(self) -> str | None:
        if self.ca_bundle is None:
            return None
        descriptor = None
        try:
            before = _safe_regular(
                self.ca_bundle, reason=ReasonCode.ASSET_SOURCE_REJECTED
            )
            descriptor = os.open(
                self.ca_bundle,
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                opened = os.fstat(stream.fileno())
                if _identity(opened) != _identity(before):
                    raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED)
                raw = stream.read(_MAX_CA_BYTES + 1)
            after = self.ca_bundle.stat(follow_symlinks=False)
            if _identity(after) != _identity(before) or not raw or len(raw) > _MAX_CA_BYTES:
                raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED)
            return raw.decode("ascii", errors="strict")
        except CacheError:
            raise
        except (OSError, UnicodeError, ValueError):
            raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED) from None
        finally:
            if descriptor is not None:
                _run_cleanups(lambda: os.close(descriptor))

    @staticmethod
    def _windows_directory_handle(path: Path) -> int:
        import ctypes
        from ctypes import wintypes

        create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            os.fspath(path), 0x80, 0x1 | 0x2, None, 3,
            0x02000000 | 0x00200000, None,
        )
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    @staticmethod
    def _close_windows_handle(handle: int) -> None:
        import ctypes
        from ctypes import wintypes

        close = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        if not close(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def _pin_layout(self) -> None:
        raise AssertionError("layout pinning is owned by _Containment")

    def _audit(
        self,
        action: str,
        reason: str,
        asset_id: str,
        failure_code: ReasonCode,
    ) -> None:
        if self._audit_blocked:
            raise CacheError(ReasonCode.AUDIT_LOSS)
        try:
            self.audit.record(
                action=action,
                reason=reason,
                asset_id=asset_id,
                failure_code=failure_code.value,
            )
        except Exception:  # noqa: BLE001 - injected audit failures fail closed.
            self._audit_blocked = True
            raise CacheError(ReasonCode.AUDIT_LOSS) from None

    def _span(self, name: str, asset_id: str) -> AbstractContextManager[object]:
        """Open a telemetry span; a failing sink drops the span, never the job."""
        if self.spans is None:
            return nullcontext()
        try:
            scope = self.spans.span(name, asset_id=asset_id)
        except Exception:  # noqa: BLE001 - telemetry is optional; drop the span.
            self._count_dropped_span()
            return nullcontext()
        if not hasattr(scope, "__enter__") or not hasattr(scope, "__exit__"):
            self._count_dropped_span()
            return nullcontext()
        return _DroppableSpan(scope, self._count_dropped_span)

    def _verify_root(self) -> None:
        try:
            containment = self._containment
            if self._closed or containment is None:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            root = containment._directories[()]
            containment.verify(root)
            if root.identity != self._root_identity:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            for name, identity in self._layout_identities.items():
                directory = containment._directories[(name,)]
                containment.verify(directory)
                if directory.identity != identity:
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
        except CacheError:
            raise
        except OSError:
            raise CacheError(ReasonCode.CACHE_UNKNOWN) from None

    def _verify_layout(self) -> None:
        self._verify_root()

    def _admit(self, snapshot_id: str, declared: int, allocated: int) -> None:
        self._verify_root()
        self._admission_call(snapshot_id, declared, allocated)

    def _admission_call(
        self, snapshot_id: str, declared: int, allocated: int, *, verify: bool = True
    ) -> None:
        if verify and hasattr(self, "_root_identity"):
            self._verify_root()
        try:
            admitted = self.admission(
                asset_id=snapshot_id,
                declared_bytes=declared,
                allocated_bytes=allocated,
            )
        except Exception:  # noqa: BLE001 - injected admission failures fail closed.
            raise CacheError(ReasonCode.STORAGE_UNKNOWN) from None
        if admitted is not True:
            raise CacheError(ReasonCode.STORAGE_GUARD_TRIPPED)

    def _ssl_context(self) -> ssl.SSLContext:
        if self._ca_data is None:
            return ssl.create_default_context()
        try:
            return ssl.create_default_context(cadata=self._ca_data)
        except (OSError, ssl.SSLError, ValueError):
            raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED) from None

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self.monotonic()
        if remaining <= 0:
            raise CacheError(ReasonCode.UPSTREAM_UNAVAILABLE)
        return remaining

    def _resolve_addresses(
        self, host: str, port: int, deadline: float
    ) -> list[tuple[int, int, int, str, tuple[object, ...]]]:
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", _RESOLVER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            try:
                output, _unused = process.communicate(
                    json.dumps((host, port)), timeout=self._remaining(deadline)
                )
            except subprocess.TimeoutExpired:
                raise CacheError(ReasonCode.UPSTREAM_UNAVAILABLE)
            if process.returncode != 0:
                raise CacheError(ReasonCode.UPSTREAM_UNAVAILABLE)
            value = json.loads(output)
            if type(value) is not list or not value:
                raise CacheError(ReasonCode.UPSTREAM_UNAVAILABLE)
            addresses = []
            for item in value:
                if (
                    type(item) is not list
                    or len(item) != 5
                    or any(type(item[index]) is not int for index in range(3))
                    or type(item[3]) is not str
                    or type(item[4]) is not list
                ):
                    raise CacheError(ReasonCode.UPSTREAM_UNAVAILABLE)
                if not item[4] or type(item[4][0]) is not str:
                    raise CacheError(ReasonCode.UPSTREAM_UNAVAILABLE)
                if _public_peer(item[4][0]):
                    addresses.append((*item[:4], tuple(item[4])))
            if not addresses:
                raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED)
            return addresses
        except CacheError:
            raise
        except (OSError, ValueError, json.JSONDecodeError):
            raise CacheError(ReasonCode.UPSTREAM_UNAVAILABLE) from None
        finally:
            self._cleanup_resolver(process)

    @staticmethod
    def _cleanup_resolver(process: subprocess.Popen[str]) -> None:
        def wait_bounded() -> None:
            process.wait(timeout=_RESOLVER_CLEANUP_SECONDS)

        if process.poll() is None:
            try:
                process.terminate()
            except BaseException:  # noqa: BLE001,S110 - kill is still mandatory.
                pass
            try:
                wait_bounded()
            except BaseException:  # noqa: BLE001,S110 - escalate to kill.
                pass
        if process.poll() is None:
            try:
                process.kill()
            except BaseException:  # noqa: BLE001,S110 - still close owned pipes.
                pass
            try:
                wait_bounded()
            except BaseException:  # noqa: BLE001,S110 - preserve active exception.
                pass
        cleanups: list[Callable[[], object]] = []
        if process.stdin is not None and not process.stdin.closed:
            cleanups.append(process.stdin.close)
        if process.stdout is not None and not process.stdout.closed:
            cleanups.append(process.stdout.close)
        if process.poll() is None and sys.exception() is None:
            # The resolver survived terminate and kill; report it rather than
            # silently leaking the child. An exception already in flight stays
            # primary, so this is only raised on an otherwise clean path.
            _raise_after_cleanups(CacheError(ReasonCode.CACHE_UNKNOWN), *cleanups)
        _run_cleanups(*cleanups)

    def _resolved_connection(
        self,
        addresses: list[tuple[int, int, int, str, tuple[object, ...]]],
        deadline: float,
        source_address: tuple[str, int] | None,
        owned_transports: list[object] | None = None,
    ) -> socket.socket:
        last_error: OSError | None = None
        for family, kind, protocol, _canonical, address in addresses:
            stream = socket.socket(family, kind, protocol)
            if owned_transports is not None:
                owned_transports.append(stream)
            try:
                stream.settimeout(self._remaining(deadline))
                if source_address is not None:
                    stream.bind(source_address)
                stream.connect(address)
                peer = stream.getpeername()
                if (
                    type(peer) not in (tuple, list)
                    or not peer
                    or not _public_peer(peer[0])
                    or not address
                    or not _public_peer(address[0])
                    or ipaddress.ip_address(peer[0])
                    != ipaddress.ip_address(address[0])
                ):
                    raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED)
                return stream
            except OSError as error:
                last_error = error
                _run_cleanups(stream.close)
            except BaseException:
                _run_cleanups(stream.close)
                raise
        if last_error is not None:
            raise last_error
        raise OSError("resolver returned no usable address")

    def _open(self, url: str, deadline: float):
        """Open the manifest URL, following at most five gated HTTPS redirects."""
        current = url
        redirects = 0
        while True:
            owner = self._request(current, deadline)
            try:
                if owner.status in _REDIRECT_STATUSES:
                    if redirects >= _MAX_REDIRECTS:
                        raise CacheError(ReasonCode.ASSET_SOURCE_REJECTED)
                    following = _redirect_target(current, owner.headers)
                    self._remaining(deadline)
                elif owner.status != 200:
                    reason = (
                        ReasonCode.ASSET_SOURCE_REJECTED
                        if 300 <= owner.status < 400
                        else ReasonCode.UPSTREAM_UNAVAILABLE
                    )
                    raise CacheError(reason)
                else:
                    if owner.headers.get("Content-Encoding") not in (
                        None,
                        "",
                        "identity",
                    ):
                        raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
                    if not callable(getattr(owner, "read1", None)):
                        raise CacheError(ReasonCode.UPSTREAM_UNAVAILABLE)
                    self._remaining(deadline)
                    return owner
            except BaseException:
                owner.close()
                raise
            owner.close()
            redirects += 1
            current = following

    def _request(self, url: str, deadline: float) -> _HTTPResponseOwner:
        parsed = urlsplit(url)
        target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        factory = http.client.HTTPSConnection
        transports: list[object] = []
        addresses = (
            self._resolve_addresses(parsed.hostname, parsed.port or 443, deadline)
            if factory is _HTTPS_CONNECTION_TYPE
            else None
        )
        context = self._ssl_context()
        if addresses is not None:
            context = _OwnedTLSContext(
                context, transports, lambda: self._remaining(deadline)
            )
        connection = factory(
            parsed.hostname,
            parsed.port or 443,
            context=context,
            timeout=self._remaining(deadline),
        )
        transports.insert(0, connection)
        if addresses is not None:
            connection._create_connection = (  # type: ignore[attr-defined]
                lambda _address, _timeout, source_address=None: self._resolved_connection(
                    addresses, deadline, source_address, transports
                )
            )
        timer = threading.Timer(
            self._remaining(deadline),
            self._close_expired_connection,
            (connection, transports),
        )
        timer.daemon = True
        try:
            timer.start()
            connection.request(
                "GET",
                target,
                headers={
                    "Accept-Encoding": "identity",
                    "User-Agent": "podmuncher-model-cache/1",
                },
            )
            response = connection.getresponse()
            transports.append(response)
            owner = _HTTPResponseOwner(response, connection, timer, url)
        except BaseException as error:
            active = error
            cleanups = [
                timer.cancel,
                lambda: self._close_expired_connection(connection, transports),
            ]
            if timer is not threading.current_thread() and timer.is_alive():
                cleanups.append(timer.join)
            _run_cleanups(*cleanups)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise active
            if isinstance(
                error, (OSError, TimeoutError, ssl.SSLError, HTTPException)
            ):
                raise CacheError(ReasonCode.UPSTREAM_UNAVAILABLE) from None
            raise
        return owner

    @staticmethod
    def _close_expired_connection(
        connection: http.client.HTTPSConnection, transports: list[object]
    ) -> None:
        sockets: list[object] = []
        seen: set[int] = set()
        pending = list(transports)
        while pending and len(seen) < 32:
            candidate = pending.pop()
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            if callable(getattr(candidate, "shutdown", None)):
                sockets.append(candidate)
            for attribute in ("sock", "_sock", "raw", "fp", "_fp", "_response"):
                child = getattr(candidate, attribute, None)
                if child is not None:
                    pending.append(child)
        for owned_socket in sockets:
            try:
                owned_socket.shutdown(socket.SHUT_RDWR)
            except BaseException:  # noqa: BLE001,S110 - continue owned cleanup.
                pass
        for transport in reversed(transports[1:]):
            try:
                transport.close()
            except BaseException:  # noqa: BLE001,S110 - close every transport.
                pass
        try:
            connection.close()
        except BaseException:  # noqa: BLE001,S110 - timer cleanup cannot escape.
            pass

    @staticmethod
    def _response_socket(response: object) -> object | None:
        pending = [response]
        seen: set[int] = set()
        while pending and len(seen) < 16:
            candidate = pending.pop()
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            socket_value = getattr(candidate, "_sock", None)
            if socket_value is None:
                socket_value = getattr(candidate, "sock", None)
            if socket_value is not None and hasattr(socket_value, "settimeout"):
                return socket_value
            for attribute in ("raw", "fp", "_fp", "_response"):
                next_candidate = getattr(candidate, attribute, None)
                if next_candidate is not None:
                    pending.append(next_candidate)
        return None

    def acquire(self, model: ManifestModel) -> CacheSnapshot:
        with self._lock:
            try:
                return self._acquire_locked(model)
            except CacheError:
                raise
            except OSError:
                raise CacheError(ReasonCode.STATE_WRITE_FAILED) from None

    def _acquire_locked(self, model: ManifestModel) -> CacheSnapshot:
        self._recover_before_mutation()
        if type(model) is not ManifestModel:
            raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
        if (
            not model.files
            or len(model.files) != len(model.source_urls)
            or len(model.files) > 256
            or type(model.synthetic) is not bool
            or model.synthetic
            or type(model.artifact.model_id) is not str
            or _MANIFEST_ID.fullmatch(model.artifact.model_id) is None
        ):
            raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
        try:
            for declared, url in zip(model.files, model.source_urls, strict=True):
                _relative(declared.relative_path)
                _verified_url(url)
                if (
                    type(declared.bytes) is not int
                    or declared.bytes <= 0
                    or type(declared.sha256) is not str
                    or _SHA256.fullmatch(declared.sha256) is None
                ):
                    raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
            if len({item.relative_path for item in model.files}) != len(model.files):
                raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
            snapshot_id, manifest_digest = _snapshot_identity(model)
        except (TypeError, ValueError):
            raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED) from None
        if _ID.fullmatch(snapshot_id) is None:
            raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
        existing = self._snapshots.get(snapshot_id)
        if existing is not None:
            self._verify_snapshot(existing, model.files)
            return existing
        if len(self._snapshots) >= self.max_entries:
            raise CacheError(ReasonCode.STORAGE_INSUFFICIENT)
        total = sum(item.bytes for item in model.files)
        self._admit(snapshot_id, total, 0)
        self._audit("stage", "verified_manifest_acquisition", snapshot_id, ReasonCode.OK)
        staging_name = f"{snapshot_id}.{uuid.uuid4().hex}"
        staging_parts = ("staging", staging_name)
        staging = self.root.joinpath(*staging_parts)
        final = self.root / "snapshots" / snapshot_id
        allocated = 0
        deadline = self.monotonic() + self.timeout_seconds
        try:
            self._verify_root()
            containment = self._require_containment()
            containment.mkdir(("staging",), staging_name)
            with containment.directory(staging_parts) as opened_staging:
                staging_identity = opened_staging.identity
            with self._span("acquire_asset", snapshot_id):
                for declared, url in zip(model.files, model.source_urls, strict=True):
                    allocated = self._download_file(
                        staging, declared, url, snapshot_id, total, allocated, deadline
                    )
                self._verify_tree(staging, model.files)
                with containment.directory(staging_parts) as opened_staging:
                    if opened_staging.identity != staging_identity:
                        raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
                self._verify_hashes(staging, model.files)
                self._admit(snapshot_id, total, allocated)
                self._verify_tree(staging, model.files)
                self._verify_hashes(staging, model.files)
                self._verify_root()
                if containment.exists(("snapshots",), snapshot_id):
                    raise CacheError(ReasonCode.CONFLICT)
                rename_error = self._rename_reconciled(
                    ("staging",),
                    staging_name,
                    ("snapshots",),
                    snapshot_id,
                    staging_identity,
                )
                with containment.directory(("snapshots", snapshot_id)) as published:
                    if published.identity != staging_identity:
                        raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
                snapshot = CacheSnapshot(
                    snapshot_id, final, total, manifest_digest, model.files
                )
                self._snapshots[snapshot_id] = snapshot
                try:
                    self._commit_metadata(
                        "publish", "verified_snapshot", snapshot_id, ReasonCode.OK
                    )
                except BaseException as error:  # noqa: BLE001 - reconcile durable commit.
                    propagated = (
                        error.error
                        if isinstance(error, _MetadataCommittedError)
                        else error
                    )
                    if self._metadata_mutation_is_durable(
                        "publish", snapshot_id, propagated
                    ):
                        raise propagated
                    self._snapshots.pop(snapshot_id, None)
                    if containment.exists(("snapshots",), snapshot_id):
                        self._quarantine(
                            final, snapshot_id, ReasonCode.STATE_WRITE_FAILED
                        )
                    raise propagated
                if rename_error is not None:
                    raise rename_error
                return snapshot
        except BaseException as error:
            containment = self._containment
            if containment is not None and containment.exists(("staging",), staging_name):
                reason = (
                    error.reason_code
                    if isinstance(error, CacheError)
                    else ReasonCode.STATE_WRITE_FAILED
                )
                self._quarantine(staging, snapshot_id, reason)
            if isinstance(error, CacheError):
                raise
            if isinstance(error, (OSError, ValueError)):
                raise CacheError(ReasonCode.STATE_WRITE_FAILED) from None
            raise

    def _require_containment(self) -> _Containment:
        containment = self._containment
        if self._closed or containment is None:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        return containment

    def _rename_reconciled(
        self,
        source_parent: tuple[str, ...],
        source_name: str,
        target_parent: tuple[str, ...],
        target_name: str,
        expected_identity: tuple[int, int],
    ) -> BaseException | None:
        containment = self._require_containment()
        try:
            containment.rename(
                source_parent, source_name, target_parent, target_name
            )
            return None
        except BaseException as error:
            def identity(parent: tuple[str, ...], name: str) -> tuple[int, int] | None:
                try:
                    info = containment.stat(parent, name)
                except FileNotFoundError:
                    return None
                if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
                return _identity(info)

            source_identity = identity(source_parent, source_name)
            target_identity = identity(target_parent, target_name)
            if source_identity is None and target_identity == expected_identity:
                return error
            if source_identity == expected_identity and target_identity is None:
                raise
            raise CacheError(ReasonCode.CACHE_UNKNOWN) from None

    def _download_file(
        self,
        staging: Path,
        declared: ManifestFile,
        url: str,
        snapshot_id: str,
        total: int,
        allocated: int,
        deadline: float,
    ) -> int:
        parts = _relative(declared.relative_path)
        containment = self._require_containment()
        parent_parts = containment._parts(staging)
        for part in parts[:-1]:
            containment.mkdir(parent_parts, part, exist_ok=True)
            parent_parts += (part,)
        digest = hashlib.sha256()
        count = 0
        self._admit(snapshot_id, total, allocated)
        response = self._open(url, deadline)
        socket_owner = self._response_socket(response)
        if socket_owner is None:
            _raise_after_cleanups(
                CacheError(ReasonCode.UPSTREAM_UNAVAILABLE), response.close
            )
        header = response.headers.get("Content-Length")
        if header is not None:
            try:
                header_size = int(header)
            except ValueError:
                _raise_after_cleanups(
                    CacheError(ReasonCode.ASSET_VERIFICATION_FAILED), response.close
                )
            if header_size != declared.bytes:
                _raise_after_cleanups(
                    CacheError(ReasonCode.ASSET_VERIFICATION_FAILED), response.close
                )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = None
        try:
            descriptor = containment.open_file(parent_parts, parts[-1], flags, 0o600)
            with os.fdopen(descriptor, "wb", buffering=0) as stream:
                descriptor = None
                while True:
                    if header is not None and count == declared.bytes:
                        break
                    try:
                        remaining = self._remaining(deadline)
                        if socket_owner.fileno() >= 0:
                            socket_owner.settimeout(remaining)
                        chunk = response.read1(
                            min(self.chunk_size, declared.bytes - count + 1)
                        )
                        self._remaining(deadline)
                    except CacheError:
                        raise
                    except (OSError, TimeoutError, HTTPException):
                        raise CacheError(ReasonCode.UPSTREAM_UNAVAILABLE) from None
                    if not chunk:
                        break
                    if len(chunk) > declared.bytes - count:
                        raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
                    self._admit(snapshot_id, total, allocated + count + len(chunk))
                    written = stream.write(chunk)
                    if written != len(chunk):
                        raise CacheError(ReasonCode.STATE_WRITE_FAILED)
                    count += written
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
                opened = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or _is_reparse(opened)
                ):
                    raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
        finally:
            cleanups = [response.close]
            if descriptor is not None:
                cleanups.append(lambda: os.close(descriptor))
            _run_cleanups(*cleanups)
        if count != declared.bytes:
            reason = (
                ReasonCode.UPSTREAM_UNAVAILABLE
                if header is not None and count < declared.bytes
                else ReasonCode.ASSET_VERIFICATION_FAILED
            )
            raise CacheError(reason)
        if not hmac.compare_digest(digest.hexdigest(), declared.sha256):
            raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
        return allocated + count

    def _verify_hashes(
        self, root: Path, files: tuple[ManifestFile, ...]
    ) -> None:
        containment = self._require_containment()
        root_parts = containment._parts(root)
        for declared in files:
            parts = _relative(declared.relative_path)
            parent_parts = root_parts + parts[:-1]
            before = containment.stat(parent_parts, parts[-1])
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or _is_reparse(before)
            ):
                raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
            digest = hashlib.sha256()
            count = 0
            descriptor = containment.open_file(
                parent_parts,
                parts[-1],
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            with os.fdopen(descriptor, "rb", buffering=0) as stream:
                opened = os.fstat(stream.fileno())
                if _identity(opened) != _identity(before):
                    raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
                while chunk := stream.read(
                    min(self.chunk_size, declared.bytes - count + 1)
                ):
                    count += len(chunk)
                    if count > declared.bytes:
                        raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
                    digest.update(chunk)
                after = os.fstat(stream.fileno())
                if _identity(after) != _identity(opened):
                    raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
            current = containment.stat(parent_parts, parts[-1])
            if (
                _identity(current) != _identity(before)
                or count != declared.bytes
                or not hmac.compare_digest(digest.hexdigest(), declared.sha256)
            ):
                raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)

    def _verify_tree(self, root: Path, declared: tuple[ManifestFile, ...]) -> None:
        expected = {Path(*_relative(item.relative_path)) for item in declared}
        if len(expected) != len(declared):
            raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
        containment = self._require_containment()
        root_parts = containment._parts(root)
        actual: set[Path] = set()

        def visit(parts: tuple[str, ...], relative: Path) -> None:
            for name in containment.list(parts):
                info = containment.stat(parts, name)
                child_relative = relative / name
                if stat.S_ISDIR(info.st_mode) and not _is_reparse(info):
                    visit(parts + (name,), child_relative)
                elif (
                    stat.S_ISREG(info.st_mode)
                    and info.st_nlink == 1
                    and not _is_reparse(info)
                ):
                    actual.add(child_relative)
                else:
                    raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)

        visit(root_parts, Path())
        if actual != expected:
            raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)

    def _verify_snapshot(
        self, snapshot: CacheSnapshot, files: tuple[ManifestFile, ...]
    ) -> None:
        self._verify_root()
        if files != snapshot.files:
            raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
        self._verify_tree(snapshot.path, files)
        self._verify_hashes(snapshot.path, files)

    def mark_in_use(self, snapshot_id: str) -> None:
        with self._lock:
            self._known_id(snapshot_id)
            count = self._in_use.get(snapshot_id, 0)
            if count >= 2**31 - 1:
                raise CacheError(ReasonCode.CONFLICT)
            self._in_use[snapshot_id] = count + 1

    def release(self, snapshot_id: str) -> None:
        with self._lock:
            self._known_id(snapshot_id)
            count = self._in_use.get(snapshot_id, 0)
            if count <= 0:
                raise CacheError(ReasonCode.CONFLICT)
            if count == 1:
                self._in_use.pop(snapshot_id)
            else:
                self._in_use[snapshot_id] = count - 1

    def _known_id(self, snapshot_id: object) -> CacheSnapshot:
        if type(snapshot_id) is not str or _ID.fullmatch(snapshot_id) is None:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        return snapshot

    def remove(self, snapshot_id: str, *, audit_loss_recovery: bool = False) -> None:
        with self._lock:
            self._remove_locked(snapshot_id, audit_loss_recovery=audit_loss_recovery)

    def _remove_locked(
        self, snapshot_id: str, *, audit_loss_recovery: bool
    ) -> None:
        if type(audit_loss_recovery) is not bool:
            raise CacheError(ReasonCode.INVALID_INPUT)
        if type(snapshot_id) is not str or _ID.fullmatch(snapshot_id) is None:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        if self._in_use.get(snapshot_id, 0):
            raise CacheError(ReasonCode.CACHE_IN_USE)
        authorization = (
            self._audit_loss_recovery_authorization(snapshot_id)
            if audit_loss_recovery
            else None
        )
        # An exempt known-unused removal (spec 1194) must not be blocked by the
        # fail-closed quarantine of unrelated unknown partials or orphans while
        # the audit sink is lost; those entries wait for a healthy recovery.
        self._recover_before_mutation(
            audit_loss_authorization=authorization,
            defer_unrelated_quarantines=audit_loss_recovery,
        )
        if authorization == ("remove", snapshot_id):
            return
        snapshot = self._known_id(snapshot_id)
        if self._in_use.get(snapshot_id, 0):
            raise CacheError(ReasonCode.CACHE_IN_USE)
        self._verify_root()
        self._verify_snapshot(snapshot, snapshot.files)
        containment = self._require_containment()
        with containment.directory(("snapshots", snapshot_id)) as opened:
            expected_identity = opened.identity
        transaction_name = f".remove.{snapshot_id}.{uuid.uuid4().hex}"
        transaction = self.root / "quarantine" / transaction_name
        rename_error = self._rename_reconciled(
            ("snapshots",),
            snapshot_id,
            ("quarantine",),
            transaction_name,
            expected_identity,
        )
        with containment.directory(("quarantine", transaction_name)) as opened:
            if opened.identity != expected_identity:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
        self._snapshots.pop(snapshot_id)
        try:
            self._commit_metadata(
                "remove",
                "unused_snapshot_tombstoned",
                snapshot_id,
                ReasonCode.OK,
                allow_audit_loss=audit_loss_recovery,
            )
        except BaseException as error:  # noqa: BLE001 - reconcile durable tombstone.
            propagated = (
                error.error
                if isinstance(error, _MetadataCommittedError)
                else error
            )
            if self._metadata_mutation_is_durable(
                "remove", snapshot_id, propagated
            ):
                if not self._metadata_mutation_audit_pending(
                    "remove", snapshot_id
                ):
                    try:
                        self._delete_tree(
                            transaction, expected_identity, parent="quarantine"
                        )
                    except (CacheError, OSError):
                        pass
                raise propagated
            self._snapshots[snapshot_id] = snapshot
            if containment.exists(
                ("quarantine",), transaction_name
            ) and not containment.exists(("snapshots",), snapshot_id):
                containment.rename(
                    ("quarantine",), transaction_name, ("snapshots",), snapshot_id
                )
            raise propagated
        try:
            self._delete_tree(transaction, expected_identity, parent="quarantine")
        except (CacheError, OSError):
            # The committed tombstone is a truthful logical removal. Recovery
            # retries physical cleanup without reporting another remove event.
            pass
        if rename_error is not None:
            raise rename_error

    def _delete_tree(
        self, path: Path, expected_identity: tuple[int, int], *, parent: str = "snapshots"
    ) -> None:
        self._verify_root()
        containment = self._require_containment()
        parts = containment._parts(path)
        if len(parts) != 2 or parts[0] != parent:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        def remove_children(retained: _Directory) -> None:
            # Stale ``.retained-*`` tombstones left by an interrupted earlier
            # deletion are ordinary children here: they must go too, or the
            # directory cannot be removed and would linger on disk.
            expected = retained.identity
            for name in containment.list_retained(retained):
                info = containment.stat_retained(retained, name)
                child_identity = _identity(info)
                if stat.S_ISDIR(info.st_mode) and not _is_reparse(info):
                    with containment.child_directory(retained, name) as child:
                        if child.identity != child_identity:
                            raise CacheError(ReasonCode.CACHE_UNKNOWN)
                        remove_children(child)
                        containment.delete_retained_directory(retained, name, child)
                elif (
                    stat.S_ISREG(info.st_mode)
                    and info.st_nlink == 1
                    and not _is_reparse(info)
                ):
                    containment.delete_retained_file(
                        retained, name, child_identity
                    )
                else:
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
            containment.verify(retained)
            if retained.identity != expected:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)

        with containment.directory(
            (parent,)
        ) as retained_parent, containment.child_directory(
            retained_parent, parts[-1]
        ) as opened:
            if opened.identity != expected_identity:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            remove_children(opened)
            containment.delete_retained_directory(
                retained_parent, parts[-1], opened
            )

    def _audit_quarantine(
        self,
        snapshot_id: str,
        failure_code: ReasonCode,
        *,
        fail_closed: bool,
    ) -> None:
        """Record a quarantine with its failure code.

        On the acquisition failure path the caller's original reason must stay
        primary, so an ``AUDIT_LOSS`` is remembered through the bounded
        audit-loss events instead of replacing it. Recovery quarantines of
        unknown or orphan entries are not exempt from the audit-failure latch
        (spec 1194): with ``fail_closed`` the loss is bookkept and re-raised so
        the entry is left where it is. An exempt known-unused removal defers
        those quarantines instead (see ``_recover_before_mutation``).
        """
        try:
            self._audit(
                "quarantine", "failed_or_unrecognized_partial", snapshot_id, failure_code
            )
        except CacheError as error:
            if error.reason_code is not ReasonCode.AUDIT_LOSS:
                raise
            self._remember_audit_loss(snapshot_id)
            if fail_closed:
                raise

    def _quarantine(
        self,
        path: Path,
        snapshot_id: str,
        failure_code: ReasonCode,
        *,
        fail_closed: bool = False,
    ) -> None:
        try:
            self._audit_quarantine(snapshot_id, failure_code, fail_closed=fail_closed)
            self._verify_root()
            if path.parent not in (self.root / "staging", self.root / "snapshots"):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            containment = self._require_containment()
            source_parent = path.parent.name
            target_name = f"{snapshot_id}.{uuid.uuid4().hex}"
            containment.rename(
                (source_parent,), path.name, ("quarantine",), target_name
            )
        except CacheError:
            raise
        except OSError:
            raise CacheError(ReasonCode.CACHE_UNKNOWN) from None

    def recover(self) -> None:
        with self._lock:
            self._recover_locked(retry_blocked_audit=True)

    def _reconcile_namespaces(
        self, *, defer_unrelated_quarantines: bool = False
    ) -> None:
        self._verify_root()
        containment = self._require_containment()
        for name in containment.list(("staging",)):
            item = self.root / "staging" / name
            info = containment.stat(("staging",), name)
            if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            if defer_unrelated_quarantines:
                # Exempt audit-loss removal: leave the unknown partial in
                # place for a later recovery with a healthy audit sink.
                continue
            self._quarantine(
                item,
                "unknown-partial",
                ReasonCode.ASSET_VERIFICATION_FAILED,
                fail_closed=True,
            )
        for name in containment.list(("quarantine",)):
            item = self.root / "quarantine" / name
            if name.startswith((".retained-file.", ".retained-directory.")):
                # A private-name tombstone is a verified, logically removed
                # entry whose physical deletion was interrupted. Finish the
                # cleanup best-effort so it does not linger; nothing to audit.
                self._delete_stale_tombstone(name)
                continue
            if not name.startswith(".remove."):
                continue
            info = containment.stat(("quarantine",), name)
            if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            parts = item.name.split(".")
            snapshot_id = parts[2] if len(parts) >= 4 else ""
            if snapshot_id in self._snapshots:
                if containment.exists(("snapshots",), snapshot_id):
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
                containment.rename(
                    ("quarantine",), name, ("snapshots",), snapshot_id
                )
            else:
                self._delete_tree(
                    item, _identity(info), parent="quarantine"
                )

    def _delete_stale_tombstone(self, name: str) -> None:
        containment = self._require_containment()
        try:
            info = containment.stat(("quarantine",), name)
            if _is_reparse(info):
                return
            if stat.S_ISDIR(info.st_mode):
                self._delete_tree(
                    self.root / "quarantine" / name,
                    _identity(info),
                    parent="quarantine",
                )
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                with containment.directory(("quarantine",)) as quarantine:
                    containment.delete_retained_file(
                        quarantine, name, _identity(info)
                    )
        except (CacheError, OSError):
            # Leaving a tombstone behind is safe; the next recovery retries.
            pass

    def _metadata_payload(self) -> bytes:
        values = [
            {
                "id": item.snapshot_id,
                "bytes": item.total_bytes,
                "manifest_sha256": item.manifest_digest,
                "files": [
                    {
                        "path": declared.relative_path,
                        "bytes": declared.bytes,
                        "sha256": declared.sha256,
                    }
                    for declared in item.files
                ],
            }
            for item in self.snapshots()
        ]
        raw = json.dumps(
            {"schema_version": 2, "snapshots": values},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(raw) > MAX_METADATA_BYTES:
            raise CacheError(ReasonCode.STATE_WRITE_FAILED)
        return raw

    def _replace_metadata(self, raw: bytes) -> None:
        containment = self._require_containment()
        temporary_name = f".metadata.{uuid.uuid4().hex}.tmp"
        descriptor = None
        try:
            descriptor = containment.open_file(
                (),
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb", buffering=0) as stream:
                descriptor = None
                if stream.write(raw) != len(raw):
                    raise CacheError(ReasonCode.STATE_WRITE_FAILED)
                os.fsync(stream.fileno())
            info = containment.stat((), temporary_name)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or _is_reparse(info):
                raise CacheError(ReasonCode.STATE_WRITE_FAILED)
            self._verify_root()
            if containment.exists((), "metadata.json"):
                old_name = f".metadata-old.{uuid.uuid4().hex}.tmp"
                containment.rename((), "metadata.json", (), old_name)
                try:
                    containment.rename((), temporary_name, (), "metadata.json")
                except BaseException:
                    containment.rename((), old_name, (), "metadata.json")
                    raise
                containment.unlink((), old_name)
            else:
                containment.rename((), temporary_name, (), "metadata.json")
            if os.name == "posix":
                os.fsync(containment._directories[()].handle)
        finally:
            if descriptor is not None:
                def descriptor_cleanup() -> None:
                    os.close(descriptor)
            else:
                def descriptor_cleanup() -> None:
                    return None
            _run_cleanups(
                descriptor_cleanup,
                lambda: (
                    containment.unlink((), temporary_name)
                    if containment.exists((), temporary_name)
                    else None
                ),
            )

    def _write_metadata_control(self, name: str, raw: bytes) -> None:
        containment = self._require_containment()
        descriptor = containment.open_file(
            (),
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", buffering=0) as stream:
                descriptor = -1
                if stream.write(raw) != len(raw):
                    raise CacheError(ReasonCode.STATE_WRITE_FAILED)
                os.fsync(stream.fileno())
            if os.name == "posix":
                os.fsync(containment._directories[()].handle)
        finally:
            if descriptor != -1:
                _run_cleanups(lambda: os.close(descriptor))

    def _replace_metadata_control(self, name: str, raw: bytes) -> None:
        containment = self._require_containment()
        temporary_name = f".metadata-control.{uuid.uuid4().hex}.tmp"
        try:
            self._write_metadata_control(temporary_name, raw)
            containment.rename((), temporary_name, (), name, replace=True)
            if os.name == "posix":
                os.fsync(containment._directories[()].handle)
        finally:
            _run_cleanups(
                lambda: (
                    containment.unlink((), temporary_name)
                    if containment.exists((), temporary_name)
                    else None
                )
            )

    def _read_metadata_control(self, name: str) -> bytes:
        containment = self._require_containment()
        descriptor = containment.open_file(
            (),
            name,
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or _is_reparse(opened)
                or opened.st_size > MAX_METADATA_BYTES
            ):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            with os.fdopen(descriptor, "rb", buffering=0) as stream:
                descriptor = -1
                raw = stream.read(MAX_METADATA_BYTES + 1)
        finally:
            if descriptor != -1:
                _run_cleanups(lambda: os.close(descriptor))
        if len(raw) > MAX_METADATA_BYTES:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        return raw

    def _metadata_transaction_state(self, transaction_id: str) -> str | None:
        containment = self._require_containment()
        if not containment.exists((), _METADATA_MARKER):
            return None
        try:
            marker = json.loads(
                self._read_metadata_control(_METADATA_MARKER).decode("ascii")
            )
        except (
            CacheError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            return None
        if (
            type(marker) is dict
            and marker.get("transaction_id") == transaction_id
            and marker.get("state") in ("prepared", "audit_pending", "committed")
        ):
            return marker["state"]
        return None

    def _metadata_mutation_is_durable(
        self, action: str, asset_id: str, original_error: BaseException
    ) -> bool:
        """Inspect durable truth before a caller rolls back a cache mutation."""
        try:
            raw = self._read_metadata_raw(required=True)
            assert raw is not None
            metadata = json.loads(raw.decode("utf-8"))
            snapshots = metadata.get("snapshots") if type(metadata) is dict else None
            if type(snapshots) is not list:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            identifiers = {
                item.get("id")
                for item in snapshots
                if type(item) is dict and type(item.get("id")) is str
            }
            metadata_matches = (
                asset_id in identifiers
                if action == "publish"
                else asset_id not in identifiers
            )
            containment = self._require_containment()
            if not containment.exists((), _METADATA_MARKER):
                return metadata_matches
            marker = json.loads(
                self._read_metadata_control(_METADATA_MARKER).decode("ascii")
            )
            return (
                metadata_matches
                and type(marker) is dict
                and marker.get("action") == action
                and marker.get("asset_id") == asset_id
                and marker.get("state") in ("audit_pending", "committed")
            )
        except BaseException:  # noqa: BLE001 - preserve the contract failure.
            # If durable truth cannot be inspected, destructive rollback is
            # unsafe. Preserve the caller's cancellation/failure unchanged.
            raise original_error

    def _metadata_mutation_audit_pending(
        self, action: str, asset_id: str
    ) -> bool:
        containment = self._require_containment()
        if not containment.exists((), _METADATA_MARKER):
            return False
        try:
            marker = json.loads(
                self._read_metadata_control(_METADATA_MARKER).decode("ascii")
            )
            return (
                type(marker) is dict
                and marker.get("state") == "audit_pending"
                and marker.get("action") == action
                and marker.get("asset_id") == asset_id
            )
        except (
            CacheError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            return True

    def _clear_metadata_transaction(self) -> None:
        containment = self._require_containment()

        def remove(name: str) -> None:
            if containment.exists((), name):
                containment.unlink((), name)

        remove(_METADATA_MARKER)
        if os.name == "posix":
            os.fsync(containment._directories[()].handle)
        remove(_METADATA_BACKUP)
        if os.name == "posix":
            os.fsync(containment._directories[()].handle)

    def _audit_loss_recovery_authorization(
        self, snapshot_id: str
    ) -> tuple[str, str] | None:
        containment = self._require_containment()
        if not containment.exists((), _METADATA_MARKER):
            return None
        try:
            marker = json.loads(
                self._read_metadata_control(_METADATA_MARKER).decode("ascii")
            )
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise CacheError(ReasonCode.CACHE_UNKNOWN) from None
        if type(marker) is not dict or marker.get("state") != "audit_pending":
            return None
        action = marker.get("action")
        asset_id = marker.get("asset_id")
        if action not in ("publish", "remove") or asset_id != snapshot_id:
            return None
        if set(marker) != {
            "action",
            "asset_id",
            "failure_code",
            "had_previous",
            "reason",
            "state",
            "transaction_id",
        }:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        self._validate_audit_pending_transaction(marker)
        return action, asset_id

    def _validate_audit_pending_transaction(self, marker: dict[str, object]) -> None:
        action = marker.get("action")
        asset_id = marker.get("asset_id")
        try:
            current_raw = self._read_metadata_raw(required=True)
            assert current_raw is not None
            current = self._parse_metadata(current_raw)
            containment = self._require_containment()
            backup_exists = containment.exists((), _METADATA_BACKUP)
            if action == "metadata":
                if (
                    marker.get("reason") != "managed_snapshot_index"
                    or asset_id != "cache"
                    or marker.get("failure_code") != ReasonCode.OK.value
                    or marker.get("had_previous") is not False
                    or current
                    or backup_exists
                ):
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
                return
            if (
                action not in ("publish", "remove")
                or type(asset_id) is not str
                or _ID.fullmatch(asset_id) is None
                or marker.get("failure_code") != ReasonCode.OK.value
                or marker.get("reason") != (
                    "verified_snapshot"
                    if action == "publish"
                    else "unused_snapshot_tombstoned"
                )
            ):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            had_previous = marker.get("had_previous")
            if type(had_previous) is not bool:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            if had_previous:
                if not backup_exists:
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
                previous = self._parse_metadata(
                    self._read_metadata_control(_METADATA_BACKUP)
                )
            else:
                if backup_exists:
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
                previous = {}

            def semantic_record(snapshot: CacheSnapshot) -> tuple[object, ...]:
                return (
                    snapshot.total_bytes,
                    snapshot.manifest_digest,
                    tuple(
                        (item.relative_path, item.bytes, item.sha256)
                        for item in snapshot.files
                    ),
                )

            if action == "publish":
                snapshot = current.get(asset_id)
                if (
                    snapshot is None
                    or asset_id in previous
                    or set(current) != set(previous) | {asset_id}
                    or any(
                        semantic_record(current[snapshot_id])
                        != semantic_record(previous_snapshot)
                        for snapshot_id, previous_snapshot in previous.items()
                    )
                ):
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
                self._verify_snapshot(snapshot, snapshot.files)
                return
            if (
                had_previous is not True
                or asset_id not in previous
                or asset_id in current
                or set(current) != set(previous) - {asset_id}
                or any(
                    semantic_record(current[snapshot_id])
                    != semantic_record(previous_snapshot)
                    for snapshot_id, previous_snapshot in previous.items()
                    if snapshot_id != asset_id
                )
            ):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            snapshot = previous.get(asset_id)
            if snapshot is None:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            candidates = tuple(
                name
                for name in containment.list(("quarantine",))
                if name.startswith(f".remove.{asset_id}.")
            )
            if len(candidates) != 1:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            transaction_name = candidates[0]
            info = containment.stat(("quarantine",), transaction_name)
            if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            with containment.directory(
                ("quarantine", transaction_name)
            ) as opened:
                if opened.identity != _identity(info):
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
            tombstone = CacheSnapshot(
                snapshot.snapshot_id,
                self.root / "quarantine" / transaction_name,
                snapshot.total_bytes,
                snapshot.manifest_digest,
                snapshot.files,
            )
            self._verify_snapshot(tombstone, tombstone.files)
        except CacheError as error:
            if error.reason_code is ReasonCode.CACHE_UNKNOWN:
                raise
            raise CacheError(ReasonCode.CACHE_UNKNOWN) from None
        except OSError:
            raise CacheError(ReasonCode.CACHE_UNKNOWN) from None

    def _recover_before_mutation(
        self,
        *,
        audit_loss_authorization: tuple[str, str] | None = None,
        defer_unrelated_quarantines: bool = False,
    ) -> None:
        self._recover_locked(
            audit_loss_authorization=audit_loss_authorization,
            defer_unrelated_quarantines=defer_unrelated_quarantines,
        )

    def _recover_locked(
        self,
        *,
        audit_loss_authorization: tuple[str, str] | None = None,
        retry_blocked_audit: bool = False,
        defer_unrelated_quarantines: bool = False,
    ) -> None:
        try:
            self._recover_metadata_transaction(
                audit_loss_authorization=audit_loss_authorization,
                retry_blocked_audit=retry_blocked_audit,
            )
            self._load_metadata(
                defer_unrelated_quarantines=defer_unrelated_quarantines
            )
            self._reconcile_namespaces(
                defer_unrelated_quarantines=defer_unrelated_quarantines
            )
        except CacheError:
            raise
        except OSError:
            raise CacheError(ReasonCode.CACHE_UNKNOWN) from None

    def _recover_metadata_transaction(
        self,
        *,
        audit_loss_authorization: tuple[str, str] | None = None,
        retry_blocked_audit: bool = False,
    ) -> bool:
        containment = self._require_containment()
        if not containment.exists((), _METADATA_MARKER):
            if containment.exists((), _METADATA_BACKUP):
                containment.unlink((), _METADATA_BACKUP)
            return False
        try:
            marker = json.loads(
                self._read_metadata_control(_METADATA_MARKER).decode("ascii")
            )
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise CacheError(ReasonCode.CACHE_UNKNOWN) from None
        if type(marker) is not dict:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        legacy = set(marker) == {"had_previous"}
        simple = set(marker) == {"had_previous", "state"}
        transactional = set(marker) == {
            "action",
            "asset_id",
            "failure_code",
            "had_previous",
            "reason",
            "state",
            "transaction_id",
        }
        if not legacy and not simple and not transactional:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        had_previous = marker["had_previous"]
        if type(had_previous) is not bool:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        state = "prepared" if legacy else marker["state"]
        if state not in ("prepared", "audit_pending", "committed"):
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        if state == "audit_pending":
            if not transactional:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            action = marker["action"]
            reason = marker["reason"]
            asset_id = marker["asset_id"]
            failure_code = marker["failure_code"]
            transaction_id = marker["transaction_id"]
            if (
                any(type(value) is not str for value in (
                    action,
                    reason,
                    asset_id,
                    failure_code,
                    transaction_id,
                ))
                or not transaction_id
            ):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            try:
                parsed_failure = ReasonCode.parse(failure_code)
            except (DomainValidationError, TypeError, ValueError):
                raise CacheError(ReasonCode.CACHE_UNKNOWN) from None
            self._validate_audit_pending_transaction(marker)
            if retry_blocked_audit:
                self._audit_blocked = False
            try:
                self._audit(action, reason, asset_id, parsed_failure)
            except CacheError as error:
                if (
                    audit_loss_authorization != (action, asset_id)
                    or error.reason_code is not ReasonCode.AUDIT_LOSS
                ):
                    raise
                self._remember_audit_loss(asset_id)
            committed = dict(marker)
            committed["state"] = "committed"
            self._replace_metadata_control(
                _METADATA_MARKER,
                json.dumps(
                    committed, sort_keys=True, separators=(",", ":")
                ).encode("ascii"),
            )
            state = "committed"
        if state == "committed":
            try:
                self._clear_metadata_transaction()
            except BaseException:  # noqa: BLE001,S110 - durable commit remains valid.
                pass
            return True
        if had_previous:
            if not containment.exists((), _METADATA_BACKUP):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            previous = self._read_metadata_control(_METADATA_BACKUP)
            if not previous:
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            self._replace_metadata(previous)
        elif containment.exists((), "metadata.json"):
            containment.unlink((), "metadata.json")
        self._clear_metadata_transaction()
        return False

    def _read_metadata_raw(self, *, required: bool) -> bytes | None:
        containment = self._require_containment()
        try:
            descriptor = containment.open_file(
                (),
                "metadata.json",
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except FileNotFoundError:
            if required:
                raise CacheError(ReasonCode.CACHE_UNKNOWN) from None
            return None
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or _is_reparse(opened)
                or opened.st_size > MAX_METADATA_BYTES
            ):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            with os.fdopen(descriptor, "rb", buffering=0) as stream:
                descriptor = -1
                raw = stream.read(MAX_METADATA_BYTES + 1)
        finally:
            if descriptor not in (-1,):
                _run_cleanups(lambda: os.close(descriptor))
        if not raw or len(raw) > MAX_METADATA_BYTES:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        return raw

    def _commit_metadata(
        self,
        action: str,
        reason: str,
        asset_id: str,
        failure_code: ReasonCode,
        *,
        allow_audit_loss: bool = False,
    ) -> None:
        self._recover_metadata_transaction()
        raw = self._metadata_payload()
        previous = self._read_metadata_raw(required=False)
        transaction = {
            "action": action,
            "asset_id": asset_id,
            "failure_code": failure_code.value,
            "had_previous": previous is not None,
            "reason": reason,
            "state": "prepared",
            "transaction_id": uuid.uuid4().hex,
        }

        def marker_raw(state: str) -> bytes:
            value = dict(transaction)
            value["state"] = state
            return json.dumps(
                value, sort_keys=True, separators=(",", ":")
            ).encode("ascii")

        try:
            if previous is not None:
                self._write_metadata_control(_METADATA_BACKUP, previous)
            self._replace_metadata_control(_METADATA_MARKER, marker_raw("prepared"))
            self._replace_metadata(raw)
            self._replace_metadata_control(
                _METADATA_MARKER, marker_raw("audit_pending")
            )
        except BaseException as error:
            state = self._metadata_transaction_state(transaction["transaction_id"])
            if state in ("audit_pending", "committed"):
                try:
                    self._recover_metadata_transaction()
                except BaseException:  # noqa: BLE001,S110
                    pass
                raise _MetadataCommittedError(error) from None
            try:
                self._recover_metadata_transaction()
            except (CacheError, OSError):
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise error
                raise CacheError(ReasonCode.STATE_WRITE_FAILED) from None
            raise
        try:
            try:
                self._audit(action, reason, asset_id, failure_code)
            except BaseException as error:
                if (
                    allow_audit_loss
                    and isinstance(error, CacheError)
                    and error.reason_code is ReasonCode.AUDIT_LOSS
                ):
                    self._remember_audit_loss(asset_id)
                else:
                    raise
            self._replace_metadata_control(
                _METADATA_MARKER, marker_raw("committed")
            )
        except BaseException as error:  # noqa: BLE001
            raise _MetadataCommittedError(error) from None
        try:
            self._clear_metadata_transaction()
        except (CacheError, OSError):
            # Marker-first cleanup is restartable; ordinary cleanup failure
            # does not turn a durable, audited commit into operation failure.
            pass
        except BaseException as error:  # noqa: BLE001
            raise _MetadataCommittedError(error) from None

    def _parse_metadata(self, raw: bytes) -> dict[str, CacheSnapshot]:
        try:
            value = json.loads(raw.decode("utf-8"))
            if (
                type(value) is not dict
                or set(value) != {"schema_version", "snapshots"}
                or value["schema_version"] != 2
                or type(value["snapshots"]) is not list
                or len(value["snapshots"]) > self.max_entries
            ):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            loaded: dict[str, CacheSnapshot] = {}
            for item in value["snapshots"]:
                if type(item) is not dict or set(item) != {
                    "id", "bytes", "manifest_sha256", "files"
                }:
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
                snapshot_id = item["id"]
                digest = item["manifest_sha256"]
                size = item["bytes"]
                if (
                    type(snapshot_id) is not str
                    or _ID.fullmatch(snapshot_id) is None
                    or type(size) is not int
                    or size <= 0
                    or type(digest) is not str
                    or _SHA256.fullmatch(digest) is None
                    or snapshot_id != f"snapshot-{digest[:32]}"
                    or snapshot_id in loaded
                    or type(item["files"]) is not list
                    or not item["files"]
                    or len(item["files"]) > 256
                ):
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
                files = []
                seen_paths = set()
                for declaration in item["files"]:
                    if type(declaration) is not dict or set(declaration) != {
                        "path", "bytes", "sha256"
                    }:
                        raise CacheError(ReasonCode.CACHE_UNKNOWN)
                    relative_path = declaration["path"]
                    file_bytes = declaration["bytes"]
                    file_hash = declaration["sha256"]
                    _relative(relative_path)
                    if (
                        relative_path in seen_paths
                        or type(file_bytes) is not int
                        or file_bytes <= 0
                        or type(file_hash) is not str
                        or _SHA256.fullmatch(file_hash) is None
                    ):
                        raise CacheError(ReasonCode.CACHE_UNKNOWN)
                    seen_paths.add(relative_path)
                    files.append(ManifestFile(relative_path, file_bytes, file_hash))
                declarations = tuple(files)
                if sum(entry.bytes for entry in declarations) != size:
                    raise CacheError(ReasonCode.CACHE_UNKNOWN)
                loaded[snapshot_id] = CacheSnapshot(
                    snapshot_id,
                    self.root / "snapshots" / snapshot_id,
                    size,
                    digest,
                    declarations,
                )
            return loaded
        except CacheError:
            raise
        except (
            UnicodeError,
            json.JSONDecodeError,
            RecursionError,
            MemoryError,
            TypeError,
            ValueError,
        ):
            raise CacheError(ReasonCode.CACHE_UNKNOWN) from None

    def _load_metadata(self, *, defer_unrelated_quarantines: bool = False) -> None:
        containment = self._require_containment()
        if not containment.exists((), "metadata.json"):
            self._commit_metadata(
                "metadata", "managed_snapshot_index", "cache", ReasonCode.OK
            )
            return
        try:
            raw = self._read_metadata_raw(required=True)
            assert raw is not None
            loaded = self._parse_metadata(raw)
            for snapshot_id, snapshot in loaded.items():
                path = snapshot.path
                if not containment.exists(("snapshots",), snapshot_id):
                    candidates = tuple(
                        name
                        for name in containment.list(("quarantine",))
                        if name.startswith(f".remove.{snapshot_id}.")
                    )
                    if len(candidates) != 1:
                        raise CacheError(ReasonCode.CACHE_UNKNOWN)
                    containment.rename(
                        ("quarantine",), candidates[0], ("snapshots",), snapshot_id
                    )
                with containment.directory(("snapshots", snapshot_id)):
                    pass
                self._verify_tree(path, snapshot.files)
                self._verify_hashes(path, snapshot.files)
            actual = set()
            for child_name in containment.list(("snapshots",)):
                with containment.directory(("snapshots", child_name)):
                    pass
                actual.add(child_name)
            for orphan_name in actual - set(loaded):
                if defer_unrelated_quarantines:
                    # Exempt audit-loss removal: leave the orphan in place for
                    # a later recovery with a healthy audit sink.
                    continue
                orphan = self.root / "snapshots" / orphan_name
                # Only a well-formed snapshot id reaches the audit record; any
                # other on-disk name is reported under a fixed label.
                label = (
                    orphan_name
                    if _ID.fullmatch(orphan_name) is not None
                    else "unrecognized"
                )
                self._quarantine(
                    orphan, label, ReasonCode.STATE_WRITE_FAILED, fail_closed=True
                )
            actual &= set(loaded)
            if actual != set(loaded):
                raise CacheError(ReasonCode.CACHE_UNKNOWN)
            self._snapshots = loaded
            self._in_use = {
                snapshot_id: count
                for snapshot_id, count in self._in_use.items()
                if snapshot_id in loaded
            }
        except CacheError:
            raise
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            RecursionError,
            MemoryError,
            TypeError,
            ValueError,
        ):
            raise CacheError(ReasonCode.CACHE_UNKNOWN) from None

    def _remember_audit_loss(self, asset_id: str) -> None:
        if len(self._audit_loss) == MAX_AUDIT_LOSS_EVENTS:
            self._audit_loss.pop(0)
        self._audit_loss.append((ReasonCode.AUDIT_LOSS, asset_id))


__all__ = (
    "AuditSink",
    "CacheError",
    "CacheSnapshot",
    "ModelCache",
    "SpanSink",
)
