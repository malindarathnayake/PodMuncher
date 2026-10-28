# ruff: noqa: BLE001, S110
"""Controlled, direct-only asynchronous HTTPS media acquisition.

The adapter deliberately implements the small HTTP/1.1 subset needed for
streaming downloads.  It never consults proxy environment variables, stores
cookies, follows non-HTTPS redirects, or connects to an address that was not
validated immediately before that hop.
"""

from __future__ import annotations

import asyncio
import io
import ipaddress
import math
import os
import re
import socket
import ssl
import stat
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, urljoin, urlsplit

from .domains import DomainValidationError, ReasonCode

MAX_HEADER_BYTES = 16 * 1024
MAX_CA_BYTES = 1024 * 1024
MAX_RESOLVED_ADDRESSES = 32
MAX_CONTENT_LENGTH_DIGITS = 20
DEFAULT_CHUNK_BYTES = 64 * 1024
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_TOKEN = re.compile(rb"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_TOKEN_PART = rb"[!#$%&'*+\-.^_`|~0-9A-Za-z]+"
_QUOTED_PART = rb'"(?:[\t !#-\[\]-~]|\\[\t -~])*"'
_MEDIA_TYPE = re.compile(
    rb"^(" + _TOKEN_PART + rb")/(" + _TOKEN_PART + rb")"
    rb"(?:[ \t]*;[ \t]*" + _TOKEN_PART + rb"[ \t]*=[ \t]*"
    rb"(?:" + _TOKEN_PART + rb"|" + _QUOTED_PART + rb"))*[ \t]*$"
)
_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")
_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class DownloadError(DomainValidationError):
    """A closed download failure that never contains URL or adapter text."""

    def __init__(self, reason_code: ReasonCode):
        super().__init__(ReasonCode.parse(reason_code))


@dataclass(frozen=True, slots=True)
class DownloadPolicy:
    byte_limit: int
    admitted_bytes: int
    overall_timeout_seconds: float
    read_timeout_seconds: float
    attempts: int = 3
    max_redirects: int = 5
    max_retry_after_seconds: float = 60
    chunk_bytes: int = DEFAULT_CHUNK_BYTES
    ca_file: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        integers = (
            (self.byte_limit, 1, None),
            (self.admitted_bytes, 1, None),
            (self.attempts, 1, 3),
            (self.max_redirects, 0, 5),
            (self.chunk_bytes, 1, 1024 * 1024),
        )
        if any(type(value) is not int or value < low or (high is not None and value > high)
               for value, low, high in integers):
            raise DownloadError(ReasonCode.INVALID_INPUT)
        for value in (
            self.overall_timeout_seconds,
            self.read_timeout_seconds,
            self.max_retry_after_seconds,
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise DownloadError(ReasonCode.INVALID_INPUT)
        if self.admitted_bytes > self.byte_limit:
            raise DownloadError(ReasonCode.INVALID_INPUT)


@dataclass(frozen=True, slots=True)
class DownloadResult:
    bytes_written: int
    content_type: str
    redirects: int
    attempts: int


class Resolver(Protocol):
    async def resolve(self, host: str, port: int) -> tuple[str, ...] | list[str]: ...


class Connection(Protocol):
    @property
    def peer_address(self) -> str: ...

    async def read(self, size: int) -> bytes: ...
    async def write(self, data: bytes) -> None: ...
    def abort(self) -> None: ...


class Connector(Protocol):
    async def connect(
        self,
        address: str,
        port: int,
        *,
        server_hostname: str,
        ssl_context: ssl.SSLContext,
    ) -> Connection: ...


class SystemResolver:
    def __init__(self, getaddrinfo=socket.getaddrinfo):
        self._getaddrinfo = getaddrinfo

    async def resolve(self, host: str, port: int) -> tuple[str, ...]:
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()

            def deliver(records=None, error=None) -> None:
                if future.cancelled():
                    return
                if error is not None:
                    future.set_exception(error)
                else:
                    future.set_result(records)

            def lookup() -> None:
                try:
                    found = self._getaddrinfo(
                        host,
                        port,
                        family=socket.AF_UNSPEC,
                        type=socket.SOCK_STREAM,
                        proto=socket.IPPROTO_TCP,
                    )
                except BaseException as error:
                    outcome = (None, error)
                else:
                    outcome = (found, None)
                try:
                    loop.call_soon_threadsafe(deliver, *outcome)
                except RuntimeError:
                    pass  # loop already closed; the abandoned lookup is discarded

            # Daemon thread: an abandoned lookup must never block loop shutdown.
            threading.Thread(target=lookup, daemon=True).start()
            records = await future
            return tuple(dict.fromkeys(record[4][0] for record in records))
        return (str(literal),)


class DirectConnection:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._reader = reader
        self._writer = writer
        self._aborted = False

    @property
    def peer_address(self) -> str:
        peer = self._writer.get_extra_info("peername")
        if type(peer) not in (tuple, list) or not peer or type(peer[0]) is not str:
            raise DownloadError(ReasonCode.CONTENT_DENIED)
        return peer[0]

    async def read(self, size: int) -> bytes:
        return await self._reader.read(size)

    async def write(self, data: bytes) -> None:
        self._writer.write(data)
        await self._writer.drain()

    def abort(self) -> None:
        if self._aborted:
            return
        self._aborted = True
        try:
            self._writer.close()
        except BaseException:
            try:
                self._writer.transport.abort()
            except BaseException as abort_error:
                if not isinstance(abort_error, Exception):
                    raise
            raise
        self._writer.transport.abort()


class DirectConnector:
    async def connect(
        self,
        address: str,
        port: int,
        *,
        server_hostname: str,
        ssl_context: ssl.SSLContext,
    ) -> Connection:
        parsed = ipaddress.ip_address(address)
        family = socket.AF_INET6 if parsed.version == 6 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        handed_off = False
        try:
            sock.setblocking(False)
            loop = asyncio.get_running_loop()
            endpoint = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
            await loop.sock_connect(sock, endpoint)
            peer = sock.getpeername()
            if type(peer) not in (tuple, list) or not peer or _public_address(peer[0]) != address:
                raise DownloadError(ReasonCode.CONTENT_DENIED)
            # Only now may TLS start; the connected peer has been matched to
            # the immediately preceding validated resolution result.
            reader, writer = await asyncio.open_connection(
                sock=sock,
                ssl=ssl_context,
                server_hostname=server_hostname,
            )
            handed_off = True
            return DirectConnection(reader, writer)
        finally:
            if not handed_off:
                active_error = sys.exception()
                try:
                    sock.close()
                except BaseException as close_error:
                    if not isinstance(close_error, Exception):
                        raise
                    if active_error is None:
                        raise


@dataclass(frozen=True, slots=True)
class _Target:
    host: str
    port: int
    request_target: str
    canonical_url: str

    @property
    def host_header(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return host if self.port == 443 else f"{host}:{self.port}"


@dataclass(frozen=True, slots=True)
class _Response:
    status: int
    headers: dict[bytes, tuple[bytes, ...]]
    body: _BufferedBody


class _BufferedBody:
    def __init__(
        self,
        connection: Connection,
        initial: bytes,
        read: Callable[[int], Awaitable[bytes]],
    ):
        self._connection = connection
        self._buffer = bytearray(initial)
        self._read = read

    async def read(self, size: int) -> bytes:
        if self._buffer:
            data = bytes(self._buffer[:size])
            del self._buffer[:size]
            return data
        return await self._read(size)

    def take_buffered(self, size: int) -> bytes:
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    async def read_exactly(self, size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = await self.read(size - len(result))
            if not chunk:
                raise _TransientDownloadError(ReasonCode.UPSTREAM_UNAVAILABLE)
            result.extend(chunk)
        return bytes(result)

    async def read_line(self, limit: int = 128) -> bytes:
        while True:
            marker = self._buffer.find(b"\r\n")
            if marker >= 0:
                if marker > limit:
                    raise DownloadError(ReasonCode.CONTENT_DENIED)
                result = bytes(self._buffer[:marker])
                del self._buffer[:marker + 2]
                return result
            if len(self._buffer) > limit:
                raise DownloadError(ReasonCode.CONTENT_DENIED)
            chunk = await self._read(min(4096, limit + 2 - len(self._buffer)))
            if not chunk:
                raise _TransientDownloadError(ReasonCode.UPSTREAM_UNAVAILABLE)
            self._buffer.extend(chunk)


class _Retryable(Exception):
    def __init__(self, delay: float = 0):
        self.delay = delay


class _TransientDownloadError(DownloadError):
    """An upstream transport failure eligible for a fresh attempt."""


def _public_address(value: object) -> str:
    if type(value) is not str or not value or "%" in value:
        raise DownloadError(ReasonCode.CONTENT_DENIED)
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise DownloadError(ReasonCode.CONTENT_DENIED) from None
    mapped = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) else None
    if mapped is not None or not address.is_global or any((
        address.is_private,
        address.is_loopback,
        address.is_link_local,
        isinstance(address, ipaddress.IPv6Address) and address.is_site_local,
        address.is_multicast,
        address.is_unspecified,
        address.is_reserved,
    )):
        raise DownloadError(ReasonCode.CONTENT_DENIED)
    return address.compressed


def _canonical_target(url: object) -> _Target:
    if type(url) is not str or not url:
        raise DownloadError(ReasonCode.INPUT_REJECTED)
    try:
        encoded_url = url.encode("utf-8", "strict")
    except UnicodeEncodeError:
        raise DownloadError(ReasonCode.INPUT_REJECTED) from None
    if len(encoded_url) > 8192:
        raise DownloadError(ReasonCode.INPUT_REJECTED)
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in url):
        raise DownloadError(ReasonCode.INPUT_REJECTED)
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (ValueError, UnicodeError):
        raise DownloadError(ReasonCode.INPUT_REJECTED) from None
    if (parsed.scheme != "https" or not parsed.netloc or parsed.username is not None
            or parsed.password is not None or parsed.fragment or "\\" in parsed.netloc):
        raise DownloadError(ReasonCode.INPUT_REJECTED)
    raw_host = parsed.hostname
    if raw_host is None or raw_host.endswith(".") or len(raw_host) > 253:
        raise DownloadError(ReasonCode.INPUT_REJECTED)
    try:
        host = ipaddress.ip_address(raw_host).compressed
    except ValueError:
        try:
            host = raw_host.encode("idna").decode("ascii").lower()
        except UnicodeError:
            raise DownloadError(ReasonCode.INPUT_REJECTED) from None
        labels = host.split(".")
        if not labels or any(_HOST_LABEL.fullmatch(label) is None for label in labels):
            raise DownloadError(ReasonCode.INPUT_REJECTED)
    if port is None:
        port = 443
    if type(port) is not int or not 1 <= port <= 65535:
        raise DownloadError(ReasonCode.INPUT_REJECTED)
    if _PERCENT.search(parsed.path) or _PERCENT.search(parsed.query):
        raise DownloadError(ReasonCode.INPUT_REJECTED)
    # Preserve escaped delimiters (notably %2F) while normalizing escape case.
    try:
        path = quote(parsed.path or "/", safe="/:@!$&'()*+,;=-._~%")
        query = quote(parsed.query, safe="!$&'()*+,;=:@/?-._~%")
    except UnicodeEncodeError:
        raise DownloadError(ReasonCode.INPUT_REJECTED) from None
    path = _ESCAPE.sub(lambda match: match.group(0).upper(), path)
    query = _ESCAPE.sub(lambda match: match.group(0).upper(), query)
    request_target = path + (f"?{query}" if parsed.query else "")
    host_text = f"[{host}]" if ":" in host else host
    authority = host_text if port == 443 else f"{host_text}:{port}"
    return _Target(host, port, request_target, f"https://{authority}{request_target}")


def _ssl_context(ca_file: str | os.PathLike[str] | None) -> ssl.SSLContext:
    if ca_file is None:
        return ssl.create_default_context()
    try:
        path = Path(ca_file)
        stat = path.stat()
        if not path.is_file() or stat.st_size <= 0 or stat.st_size > MAX_CA_BYTES:
            raise DownloadError(ReasonCode.INVALID_CONFIG)
        return ssl.create_default_context(cafile=os.fspath(path))
    except DownloadError:
        raise
    except (OSError, ssl.SSLError, TypeError, ValueError):
        raise DownloadError(ReasonCode.INVALID_CONFIG) from None


def _content_length(headers: dict[bytes, tuple[bytes, ...]]) -> int | None:
    values = headers.get(b"content-length", ())
    if not values:
        return None
    if (len(values) != 1 or not values[0].isdigit()
            or len(values[0]) > MAX_CONTENT_LENGTH_DIGITS):
        raise DownloadError(ReasonCode.CONTENT_DENIED)
    try:
        length = int(values[0])
    except (ValueError, OverflowError):
        raise DownloadError(ReasonCode.CONTENT_DENIED) from None
    if length < 0:
        raise DownloadError(ReasonCode.CONTENT_DENIED)
    return length


def _response_framing(
    headers: dict[bytes, tuple[bytes, ...]],
) -> tuple[int | None, bool]:
    declared = _content_length(headers)
    transfer = headers.get(b"transfer-encoding", ())
    if transfer:
        if len(transfer) != 1 or transfer[0].lower() != b"chunked":
            raise DownloadError(ReasonCode.CONTENT_DENIED)
        if declared is not None:
            raise DownloadError(ReasonCode.CONTENT_DENIED)
    return declared, bool(transfer)


def _media_type(headers: dict[bytes, tuple[bytes, ...]]) -> str:
    values = headers.get(b"content-type", ())
    if len(values) != 1:
        raise DownloadError(ReasonCode.CONTENT_DENIED)
    match = _MEDIA_TYPE.fullmatch(values[0])
    if match is None:
        raise DownloadError(ReasonCode.CONTENT_DENIED) from None
    major = match.group(1).decode("ascii").lower()
    subtype = match.group(2).decode("ascii").lower()
    if major not in {"audio", "video"} and (
        major != "application" or subtype not in {"octet-stream", "ogg"}
    ):
        raise DownloadError(ReasonCode.CONTENT_DENIED)
    encoding = headers.get(b"content-encoding", ())
    if encoding and encoding != (b"identity",):
        raise DownloadError(ReasonCode.CONTENT_DENIED)
    return f"{major}/{subtype}"


def _retry_after(headers: dict[bytes, tuple[bytes, ...]], maximum: float) -> float:
    values = headers.get(b"retry-after", ())
    if not values:
        return 0
    if len(values) != 1 or not values[0].isdigit() or len(values[0]) > 10:
        raise DownloadError(ReasonCode.UPSTREAM_UNAVAILABLE)
    return min(float(int(values[0])), maximum)


def _output_identity(output: object) -> tuple[int, tuple[int, int]]:
    if type(output) is not io.FileIO or output.closed or not output.writable():
        raise DownloadError(ReasonCode.INVALID_INPUT)
    try:
        descriptor = output.fileno()
        metadata = os.fstat(descriptor)
    except (OSError, ValueError):
        raise DownloadError(ReasonCode.INVALID_INPUT) from None
    if (
        type(descriptor) is not int
        or descriptor < 0
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise DownloadError(ReasonCode.INVALID_INPUT)
    return descriptor, (metadata.st_dev, metadata.st_ino)


def _check_output(
    descriptor: int,
    identity: tuple[int, int],
    check_deadline: Callable[[], None],
) -> None:
    check_deadline()
    try:
        metadata = os.fstat(descriptor)
    except OSError:
        raise DownloadError(ReasonCode.STATE_WRITE_FAILED) from None
    check_deadline()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (metadata.st_dev, metadata.st_ino) != identity
    ):
        raise DownloadError(ReasonCode.STATE_WRITE_FAILED)


def _reset_output(
    descriptor: int,
    identity: tuple[int, int],
    check_deadline: Callable[[], None],
) -> None:
    _check_output(descriptor, identity, check_deadline)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        check_deadline()
        os.ftruncate(descriptor, 0)
    except OSError:
        raise DownloadError(ReasonCode.STATE_WRITE_FAILED) from None
    _check_output(descriptor, identity, check_deadline)


def _write_output(
    descriptor: int,
    identity: tuple[int, int],
    data: bytes,
    check_deadline: Callable[[], None],
) -> None:
    _check_output(descriptor, identity, check_deadline)
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        check_deadline()
        try:
            count = os.write(descriptor, view[offset:])
        except OSError:
            raise DownloadError(ReasonCode.STORAGE_GUARD_TRIPPED) from None
        check_deadline()
        if type(count) is not int or count <= 0 or count > len(view) - offset:
            raise DownloadError(ReasonCode.STATE_WRITE_FAILED)
        offset += count
    _check_output(descriptor, identity, check_deadline)


def _sync_output(
    descriptor: int,
    identity: tuple[int, int],
    check_deadline: Callable[[], None],
) -> None:
    _check_output(descriptor, identity, check_deadline)
    try:
        os.fsync(descriptor)
    except OSError:
        raise DownloadError(ReasonCode.STATE_WRITE_FAILED) from None
    _check_output(descriptor, identity, check_deadline)


async def _dispose_awaitable(awaitable: Awaitable) -> None:
    if isinstance(awaitable, asyncio.Future):
        awaitable.cancel()
        try:
            await awaitable
        except BaseException:
            pass
        return
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()


def _abort_connection(connection: Connection) -> None:
    active_error = sys.exception()
    try:
        connection.abort()
    except ssl.SSLCertVerificationError:
        if active_error is not None and not isinstance(active_error, Exception):
            return
        raise DownloadError(ReasonCode.CONTENT_DENIED) from None
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        if isinstance(active_error, ssl.SSLCertVerificationError):
            return
        if active_error is not None and not isinstance(active_error, Exception):
            return
        raise DownloadError(ReasonCode.UPSTREAM_UNAVAILABLE) from None


async def download(
    url: str,
    output: io.FileIO,
    policy: DownloadPolicy,
    *,
    free_space: Callable[[], int],
    resolver: Resolver | None = None,
    connector: Connector | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> DownloadResult:
    """Stream one public HTTPS media response into a caller-owned temp output."""
    if type(policy) is not DownloadPolicy or not callable(free_space):
        raise DownloadError(ReasonCode.INVALID_INPUT)
    output_descriptor, output_identity = _output_identity(output)
    resolver = SystemResolver() if resolver is None else resolver
    connector = DirectConnector() if connector is None else connector
    if not callable(getattr(resolver, "resolve", None)) or not callable(getattr(connector, "connect", None)):
        raise DownloadError(ReasonCode.INVALID_INPUT)
    target = _canonical_target(url)
    context = _ssl_context(policy.ca_file)
    try:
        started = clock()
    except Exception:
        raise DownloadError(ReasonCode.INVALID_INPUT) from None
    if type(started) not in (int, float) or not math.isfinite(started):
        raise DownloadError(ReasonCode.INVALID_INPUT)
    deadline = float(started) + float(policy.overall_timeout_seconds)

    def remaining_time() -> float:
        try:
            current = clock()
        except Exception:
            raise DownloadError(ReasonCode.DOWNLOAD_TIMEOUT) from None
        if type(current) not in (int, float) or not math.isfinite(current):
            raise DownloadError(ReasonCode.DOWNLOAD_TIMEOUT)
        return deadline - float(current)

    def check_deadline() -> None:
        if remaining_time() <= 0:
            raise DownloadError(ReasonCode.DOWNLOAD_TIMEOUT)

    async def bounded(awaitable: Awaitable, *, read: bool = False):
        try:
            remaining = remaining_time()
        except BaseException:
            try:
                await _dispose_awaitable(awaitable)
            except BaseException:
                pass
            raise
        read_timeout = float(policy.read_timeout_seconds)
        read_timeout_applies = read and read_timeout < remaining
        timeout = min(remaining, read_timeout) if read else remaining
        if timeout <= 0:
            await _dispose_awaitable(awaitable)
            raise DownloadError(ReasonCode.DOWNLOAD_TIMEOUT)
        timeout_context = asyncio.timeout(timeout)
        try:
            async with timeout_context:
                result = await awaitable
        except TimeoutError:
            if not timeout_context.expired():
                raise
            if read_timeout_applies and remaining_time() > 0:
                raise _TransientDownloadError(ReasonCode.UPSTREAM_UNAVAILABLE) from None
            raise DownloadError(ReasonCode.DOWNLOAD_TIMEOUT) from None
        check_deadline()
        return result

    async def one_attempt(attempt: int) -> DownloadResult:
        current = target
        redirects = 0
        while True:
            connection: Connection | None = None
            successful = False

            async def connect_and_own(address: str, connect_target: _Target) -> Connection:
                nonlocal connection
                connection = await connector.connect(
                    address,
                    connect_target.port,
                    server_hostname=connect_target.host,
                    ssl_context=context,
                )
                check_deadline()
                return connection

            try:
                raw_addresses = await bounded(resolver.resolve(current.host, current.port))
                if type(raw_addresses) not in (tuple, list):
                    raise DownloadError(ReasonCode.CONTENT_DENIED)
                check_deadline()
                if len(raw_addresses) > MAX_RESOLVED_ADDRESSES:
                    raise DownloadError(ReasonCode.CONTENT_DENIED)
                addresses = tuple(dict.fromkeys(_public_address(item) for item in raw_addresses))
                if not addresses:
                    raise DownloadError(ReasonCode.CONTENT_DENIED)
                last_error: BaseException | None = None
                for address in addresses:
                    try:
                        await bounded(connect_and_own(address, current))
                        break
                    except (DownloadError, asyncio.CancelledError):
                        raise
                    except ssl.SSLCertVerificationError:
                        raise DownloadError(ReasonCode.CONTENT_DENIED) from None
                    except (OSError, ssl.SSLError) as error:
                        last_error = error
                if connection is None:
                    raise _Retryable() from last_error
                peer = _public_address(connection.peer_address)
                if peer not in addresses:
                    raise DownloadError(ReasonCode.CONTENT_DENIED)
                request = (
                    f"GET {current.request_target} HTTP/1.1\r\n"
                    f"Host: {current.host_header}\r\n"
                    "User-Agent: podmuncher/0.1\r\n"
                    "Accept: audio/*, video/*, application/octet-stream, application/ogg\r\n"
                    "Accept-Encoding: identity\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
                await bounded(connection.write(request))
                response = await _read_response(connection, bounded, check_deadline)
                declared, chunked = _response_framing(response.headers)
                if response.status in _REDIRECT_STATUSES:
                    locations = response.headers.get(b"location", ())
                    if len(locations) != 1 or redirects >= policy.max_redirects:
                        raise DownloadError(ReasonCode.CONTENT_DENIED)
                    try:
                        location = locations[0].decode("ascii")
                    except UnicodeDecodeError:
                        raise DownloadError(ReasonCode.CONTENT_DENIED) from None
                    try:
                        redirect_url = urljoin(current.canonical_url, location)
                        current = _canonical_target(redirect_url)
                    except DownloadError:
                        raise
                    except (UnicodeError, ValueError):
                        raise DownloadError(ReasonCode.CONTENT_DENIED) from None
                    redirects += 1
                    continue
                if response.status in _RETRY_STATUSES:
                    raise _Retryable(_retry_after(
                        response.headers,
                        float(policy.max_retry_after_seconds),
                    ))
                if response.status != 200:
                    raise DownloadError(ReasonCode.CONTENT_DENIED)
                check_deadline()
                content_type = _media_type(response.headers)
                if declared is None and not chunked:
                    # Close-delimited bodies cannot prove completeness (TLS EOF
                    # without close_notify looks clean), so they are refused.
                    raise DownloadError(ReasonCode.CONTENT_DENIED)
                check_deadline()
                cap = min(policy.byte_limit, policy.admitted_bytes)
                if declared is not None and declared > cap:
                    raise DownloadError(ReasonCode.MEDIA_LIMIT_REACHED)
                written = await _stream_body(
                    response,
                    output_descriptor,
                    output_identity,
                    cap=cap,
                    declared=declared,
                    chunked=chunked,
                    chunk_bytes=policy.chunk_bytes,
                    free_space=free_space,
                    bounded=bounded,
                    check_deadline=check_deadline,
                )
                _sync_output(output_descriptor, output_identity, check_deadline)
                check_deadline()
                result = DownloadResult(written, content_type, redirects, attempt)
                successful = True
                return result
            finally:
                if connection is not None:
                    _abort_connection(connection)
                if successful:
                    check_deadline()

    async def backoff(attempt: int, retry_after: float) -> None:
        if attempt == policy.attempts:
            raise DownloadError(ReasonCode.UPSTREAM_UNAVAILABLE) from None
        delay = max(retry_after, min(0.5 * 2 ** (attempt - 1), 8.0))
        if delay >= remaining_time():
            raise DownloadError(ReasonCode.DOWNLOAD_TIMEOUT)
        await bounded(sleep(delay))

    _reset_output(output_descriptor, output_identity, check_deadline)
    try:
        for attempt in range(1, policy.attempts + 1):
            try:
                return await one_attempt(attempt)
            except _Retryable as error:
                _reset_output(output_descriptor, output_identity, check_deadline)
                await backoff(attempt, error.delay)
            except _TransientDownloadError:
                _reset_output(output_descriptor, output_identity, check_deadline)
                await backoff(attempt, 0.0)
            except ssl.SSLCertVerificationError:
                _reset_output(output_descriptor, output_identity, check_deadline)
                raise DownloadError(ReasonCode.CONTENT_DENIED) from None
            except (OSError, ssl.SSLError, ConnectionError):
                _reset_output(output_descriptor, output_identity, check_deadline)
                await backoff(attempt, 0.0)
    except BaseException:
        try:
            _reset_output(output_descriptor, output_identity, lambda: None)
        except Exception:
            pass
        raise
    raise AssertionError("unreachable")


async def _read_response(
    connection: Connection,
    bounded: Callable[..., Awaitable],
    check_deadline: Callable[[], None],
) -> _Response:
    data = bytearray()
    marker = -1
    while marker < 0:
        if len(data) >= MAX_HEADER_BYTES:
            raise DownloadError(ReasonCode.CONTENT_DENIED)
        chunk = await bounded(connection.read(min(4096, MAX_HEADER_BYTES - len(data))), read=True)
        if not chunk:
            raise _TransientDownloadError(ReasonCode.UPSTREAM_UNAVAILABLE)
        if type(chunk) is not bytes:
            raise DownloadError(ReasonCode.UPSTREAM_UNAVAILABLE)
        check_deadline()
        data.extend(chunk)
        marker = data.find(b"\r\n\r\n")
        check_deadline()
    if marker + 4 > MAX_HEADER_BYTES:
        raise DownloadError(ReasonCode.CONTENT_DENIED)
    check_deadline()
    header_block = bytes(data[:marker])
    lines = header_block.split(b"\r\n")
    if not lines or not re.fullmatch(rb"HTTP/1\.[01] [0-9]{3}(?: [\x20-\x7e]*)?", lines[0]):
        raise DownloadError(ReasonCode.CONTENT_DENIED)
    status = int(lines[0][9:12])
    headers: dict[bytes, list[bytes]] = {}
    for line in lines[1:]:
        if line[:1] in b" \t" or b":" not in line:
            raise DownloadError(ReasonCode.CONTENT_DENIED)
        name, value = line.split(b":", 1)
        name = name.lower()
        value = value.strip(b" \t")
        if not _TOKEN.fullmatch(name) or any(byte < 0x20 or byte == 0x7F for byte in value):
            raise DownloadError(ReasonCode.CONTENT_DENIED)
        headers.setdefault(name, []).append(value)
        check_deadline()
    frozen = {name: tuple(values) for name, values in headers.items()}
    check_deadline()

    async def read(size: int) -> bytes:
        return await bounded(connection.read(size), read=True)

    return _Response(status, frozen, _BufferedBody(connection, bytes(data[marker + 4:]), read))


async def _stream_body(
    response: _Response,
    output_descriptor: int,
    output_identity: tuple[int, int],
    *,
    cap: int,
    declared: int | None,
    chunked: bool,
    chunk_bytes: int,
    free_space: Callable[[], int],
    bounded: Callable[..., Awaitable],
    check_deadline: Callable[[], None],
) -> int:
    written = 0

    def write(chunk: bytes) -> None:
        nonlocal written
        if not chunk:
            return
        check_deadline()
        try:
            available = free_space()
        except Exception:
            raise DownloadError(ReasonCode.STORAGE_UNKNOWN) from None
        check_deadline()
        if type(available) is not int or available < 0:
            raise DownloadError(ReasonCode.STORAGE_UNKNOWN)
        if len(chunk) > cap - written:
            raise DownloadError(ReasonCode.MEDIA_LIMIT_REACHED)
        if len(chunk) > available:
            raise DownloadError(ReasonCode.STORAGE_GUARD_TRIPPED)
        _write_output(output_descriptor, output_identity, chunk, check_deadline)
        written += len(chunk)

    async def require_eof(extra_byte_reason: ReasonCode) -> None:
        check_deadline()
        if response.body.take_buffered(1):
            raise DownloadError(extra_byte_reason)
        if await response.body.read(1) != b"":
            raise DownloadError(extra_byte_reason)
        check_deadline()

    if chunked:
        while True:
            check_deadline()
            line = await response.body.read_line()
            if not line or b";" in line or not re.fullmatch(rb"[0-9A-Fa-f]+", line):
                raise DownloadError(ReasonCode.CONTENT_DENIED)
            size = int(line, 16)
            check_deadline()
            if size == 0:
                if await response.body.read_line() != b"":
                    raise DownloadError(ReasonCode.CONTENT_DENIED)
                await require_eof(ReasonCode.CONTENT_DENIED)
                break
            if size > cap - written:
                raise DownloadError(ReasonCode.MEDIA_LIMIT_REACHED)
            remaining = size
            while remaining:
                chunk = await response.body.read(min(chunk_bytes, remaining))
                if not chunk:
                    raise _TransientDownloadError(ReasonCode.UPSTREAM_UNAVAILABLE)
                if len(chunk) > remaining:
                    raise DownloadError(ReasonCode.CONTENT_DENIED)
                write(chunk)
                remaining -= len(chunk)
                check_deadline()
            if await response.body.read_exactly(2) != b"\r\n":
                raise DownloadError(ReasonCode.CONTENT_DENIED)
            check_deadline()
    elif declared is not None:
        remaining = declared
        while remaining:
            check_deadline()
            chunk = await response.body.read(min(chunk_bytes, remaining))
            if not chunk:
                raise _TransientDownloadError(ReasonCode.UPSTREAM_UNAVAILABLE)
            if len(chunk) > remaining:
                raise DownloadError(ReasonCode.CONTENT_DENIED)
            write(chunk)
            remaining -= len(chunk)
            check_deadline()
        await require_eof(ReasonCode.MEDIA_LIMIT_REACHED)
    if written == 0:
        raise DownloadError(ReasonCode.CONTENT_DENIED)
    check_deadline()
    return written


__all__ = (
    "Connection",
    "Connector",
    "DirectConnection",
    "DirectConnector",
    "DownloadError",
    "DownloadPolicy",
    "DownloadResult",
    "Resolver",
    "SystemResolver",
    "download",
)
