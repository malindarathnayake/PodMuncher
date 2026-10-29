# ruff: noqa: ASYNC230, SIM117
from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import socket
import ssl
import sys
import tempfile
from pathlib import Path

import pytest

from podmuncher.direct_download import (
    DirectConnection,
    DirectConnector,
    DownloadError,
    DownloadPolicy,
    SystemResolver,
    download,
)
from podmuncher.domains import DomainValidationError, ReasonCode

PUBLIC_V4 = "93.184.216.34"
PUBLIC_V6 = "2606:2800:220:1:248:1893:25c8:1946"


class FakeResolver:
    def __init__(self, answers=None):
        self.answers = answers or {"media.example": (PUBLIC_V4,)}
        self.calls = []

    async def resolve(self, host, port):
        self.calls.append((host, port))
        answer = self.answers[host]
        if isinstance(answer, list):
            return answer.pop(0)
        return answer


class FakeConnection:
    def __init__(self, response, *, peer=PUBLIC_V4, delay=0):
        self.response = bytearray(response)
        self._peer = peer
        self.delay = delay
        self.request = bytearray()
        self.closed = False
        self.aborted = False
        self.abort_calls = 0

    @property
    def peer_address(self):
        return self._peer

    async def read(self, size):
        if self.delay:
            await asyncio.sleep(self.delay)
        data = bytes(self.response[:size])
        del self.response[:size]
        return data

    async def write(self, data):
        self.request.extend(data)

    def abort(self):
        self.abort_calls += 1
        self.aborted = True

    async def close(self):
        self.closed = True


class StagedConnection(FakeConnection):
    def __init__(self, response, *next_reads, peer=PUBLIC_V4):
        super().__init__(response, peer=peer)
        self.next_reads = list(next_reads)
        self.read_calls = 0

    async def read(self, size):
        self.read_calls += 1
        if self.response:
            return await super().read(size)
        if self.next_reads:
            return self.next_reads.pop(0)
        return b""


class FakeConnector:
    def __init__(self, *connections):
        self.connections = list(connections)
        self.calls = []

    async def connect(self, address, port, *, server_hostname, ssl_context):
        self.calls.append((address, port, server_hostname, ssl_context))
        return self.connections.pop(0)


def response(
    body=b"media",
    *,
    status=200,
    content_type=b"audio/mpeg",
    content_length=True,
    extra=(),
):
    headers = [
        f"HTTP/1.1 {status} Status".encode(),
        b"Content-Type: " + content_type,
        *extra,
    ]
    if content_length is True:
        headers.append(f"Content-Length: {len(body)}".encode())
    elif content_length is not False:
        headers.append(b"Content-Length: " + str(content_length).encode())
    return b"\r\n".join(headers) + b"\r\n\r\n" + body


def policy(**changes):
    values = {
        "byte_limit": 1024,
        "admitted_bytes": 1024,
        "overall_timeout_seconds": 2,
        "read_timeout_seconds": 0.2,
    }
    values.update(changes)
    return DownloadPolicy(**values)


async def run_download(tmp_path, connection, *, resolver=None, connector=None, **policy_changes):
    target = tmp_path / "owned.partial"
    with target.open("w+b", buffering=0) as output:
        result = await download(
            "https://media.example/file.mp3",
            output,
            policy(**policy_changes),
            free_space=lambda: 4096,
            resolver=resolver or FakeResolver(),
            connector=connector or FakeConnector(connection),
        )
        output.seek(0)
        content = output.read()
    return result, content


@pytest.mark.asyncio
async def test_streams_chunked_missing_length_and_fsyncs_caller_output(tmp_path):
    result, content = await run_download(
        tmp_path,
        FakeConnection(response(
            b"3\r\nabc\r\n3\r\ndef\r\n0\r\n\r\n",
            content_length=False,
            extra=(b"Transfer-Encoding: chunked",),
        )),
    )
    assert content == b"abcdef"
    assert result.bytes_written == 6
    assert result.content_type == "audio/mpeg"


@pytest.mark.asyncio
async def test_close_delimited_body_is_denied_without_output_or_retry(tmp_path):
    connection = FakeConnection(response(b"abcdef", content_length=False))
    connector = FakeConnector(connection)
    target = tmp_path / "out"
    with target.open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=connector,
            )
        output.seek(0)
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED
    assert len(connector.calls) == 1
    assert connection.abort_calls == 1


class _ErrorThenOk:
    """Connector whose first connect raises a transient OSError."""

    def __init__(self, ok):
        self.ok = ok
        self.calls = 0

    async def connect(self, address, port, *, server_hostname, ssl_context):
        self.calls += 1
        if self.calls <= 2:
            raise ConnectionResetError()
        return self.ok


@pytest.mark.asyncio
async def test_transient_exception_backs_off_per_attempt(tmp_path):
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    with (tmp_path / "out").open("w+b", buffering=0) as output:
        result = await download(
            "https://media.example/file",
            output,
            policy(attempts=3),
            free_space=lambda: 4096,
            resolver=FakeResolver(),
            connector=_ErrorThenOk(FakeConnection(response(b"ok"))),
            sleep=fake_sleep,
        )
    assert result.attempts == 3
    assert delays == [0.5, 1.0]


@pytest.mark.asyncio
async def test_retryable_status_without_retry_after_backs_off_per_attempt(tmp_path):
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    connector = FakeConnector(
        FakeConnection(response(b"", status=503)),
        FakeConnection(response(b"", status=503)),
        FakeConnection(response(b"ok")),
    )
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        result = await download(
            "https://media.example/file",
            output,
            policy(attempts=3),
            free_space=lambda: 4096,
            resolver=FakeResolver(),
            connector=connector,
            sleep=fake_sleep,
        )
    assert result.attempts == 3
    assert delays == [0.5, 1.0]


@pytest.mark.asyncio
async def test_backoff_not_shorter_than_remaining_time_is_timeout(tmp_path):
    async def fake_sleep(delay):
        raise AssertionError("must not sleep")

    connector = FakeConnector(FakeConnection(response(b"", status=503)))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(overall_timeout_seconds=0.3),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=connector,
                sleep=fake_sleep,
            )
    assert caught.value.reason_code is ReasonCode.DOWNLOAD_TIMEOUT


def test_abandoned_lookup_does_not_block_event_loop_shutdown(tmp_path):
    import threading
    import time

    release = threading.Event()

    def blocking_getaddrinfo(*args, **kwargs):
        release.wait(10)
        return []

    async def attempt():
        with (tmp_path / "out").open("w+b", buffering=0) as output:
            with pytest.raises(DownloadError) as caught:
                await download(
                    "https://media.example/file",
                    output,
                    policy(overall_timeout_seconds=0.2),
                    free_space=lambda: 4096,
                    resolver=SystemResolver(blocking_getaddrinfo),
                    connector=FakeConnector(),
                )
        assert caught.value.reason_code is ReasonCode.DOWNLOAD_TIMEOUT

    started = time.monotonic()
    try:
        asyncio.run(attempt())
        assert time.monotonic() - started < 5
    finally:
        release.set()


@pytest.mark.asyncio
async def test_fd_write_retries_exact_partial_writes(tmp_path, monkeypatch):
    real_write = os.write
    write_sizes = []

    def partial_write(descriptor, data):
        limited = data[:2]
        count = real_write(descriptor, limited)
        write_sizes.append(count)
        return count

    monkeypatch.setattr(os, "write", partial_write)
    result, content = await run_download(
        tmp_path,
        FakeConnection(response(b"abcdef")),
    )
    assert content == b"abcdef"
    assert result.bytes_written == 6
    assert write_sizes == [2, 2, 2]


@pytest.mark.asyncio
async def test_connects_only_to_validated_address_and_sends_canonical_request(tmp_path):
    connection = FakeConnection(response())
    resolver = FakeResolver({"media.example": (PUBLIC_V4, PUBLIC_V6)})
    connector = FakeConnector(connection)
    await run_download(tmp_path, connection, resolver=resolver, connector=connector)
    assert connector.calls[0][:3] == (PUBLIC_V4, 443, "media.example")
    assert connection.request.startswith(b"GET /file.mp3 HTTP/1.1\r\n")
    assert b"\r\nHost: media.example\r\n" in connection.request
    assert b"Cookie:" not in connection.request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.1.1",
        "224.0.0.1",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "ff02::1",
        "::",
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.1",
        "fec0::1",
    ],
)
async def test_rejects_every_non_public_or_mapped_resolution(tmp_path, address):
    connector = FakeConnector(FakeConnection(response()))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver({"media.example": (address,)}),
                connector=connector,
            )
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED
    assert connector.calls == []


@pytest.mark.asyncio
async def test_rejects_resolver_iterable_over_explicit_cardinality_limit(tmp_path):
    yielded = 0

    def infinite_addresses():
        nonlocal yielded
        while True:
            yielded += 1
            yield PUBLIC_V4

    connector = FakeConnector(FakeConnection(response()))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver({"media.example": infinite_addresses()}),
                connector=connector,
            )
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED
    assert yielded == 0
    assert connector.calls == []


@pytest.mark.asyncio
async def test_rejects_concrete_resolution_over_cardinality_limit(tmp_path):
    class LongListResolver:
        async def resolve(self, host, port):
            return [PUBLIC_V4] * 33

    connector = FakeConnector(FakeConnection(response()))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=LongListResolver(),
                connector=connector,
            )
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED
    assert connector.calls == []


@pytest.mark.asyncio
async def test_rejects_blocking_resolver_iterable_without_iterating(tmp_path):
    iterated = False

    class BlockingIterable:
        def __iter__(self):
            nonlocal iterated
            iterated = True
            raise AssertionError("must not iterate")

    connector = FakeConnector(FakeConnection(response()))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver({"media.example": BlockingIterable()}),
                connector=connector,
            )
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED
    assert not iterated
    assert connector.calls == []


@pytest.mark.asyncio
async def test_rejects_arbitrary_output_without_invoking_methods():
    invoked = False

    class BlockingOutput(io.BytesIO):
        def seek(self, *args, **kwargs):
            nonlocal invoked
            invoked = True
            raise AssertionError("must not invoke")

        def write(self, *args, **kwargs):
            nonlocal invoked
            invoked = True
            raise AssertionError("must not invoke")

    with pytest.raises(DownloadError) as caught:
        await download(
            "https://media.example/file",
            BlockingOutput(),
            policy(),
            free_space=lambda: 4096,
            resolver=FakeResolver(),
            connector=FakeConnector(FakeConnection(response())),
        )
    assert caught.value.reason_code is ReasonCode.INVALID_INPUT
    assert not invoked


@pytest.mark.asyncio
async def test_rejects_multiply_linked_output_before_network(tmp_path):
    target = tmp_path / "out"
    alias = tmp_path / "alias"
    with target.open("w+b", buffering=0) as output:
        os.link(target, alias)
        connector = FakeConnector(FakeConnection(response()))
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=connector,
            )
    assert caught.value.reason_code is ReasonCode.INVALID_INPUT
    assert connector.calls == []


@pytest.mark.asyncio
async def test_rejects_output_link_race_before_fd_write(tmp_path):
    target = tmp_path / "out"
    alias = tmp_path / "alias"
    connection = FakeConnection(response())

    def linking_free_space():
        os.link(target, alias)
        return 4096

    with target.open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=linking_free_space,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.STATE_WRITE_FAILED
    assert connection.aborted


@pytest.mark.asyncio
async def test_rejects_rebinding_peer_before_request(tmp_path):
    connection = FakeConnection(response(), peer="127.0.0.1")
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.tell() == 0
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED
    assert connection.request == b""
    assert connection.abort_calls == 1
    assert not connection.closed


@pytest.mark.asyncio
async def test_validates_redirect_host_before_second_connect(tmp_path):
    first = FakeConnection(response(
        b"",
        status=302,
        extra=(b"Location: https://private.example/secret",),
    ))
    connector = FakeConnector(first, FakeConnection(response()))
    resolver = FakeResolver({
        "media.example": (PUBLIC_V4,),
        "private.example": ("10.1.2.3",),
    })
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=resolver,
                connector=connector,
            )
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED
    assert resolver.calls == [("media.example", 443), ("private.example", 443)]
    assert len(connector.calls) == 1


@pytest.mark.asyncio
async def test_malformed_redirect_join_is_closed_and_not_followed(tmp_path):
    location = "https://[malformed.example/secret"
    first = FakeConnection(response(
        b"",
        status=302,
        extra=(f"Location: {location}".encode(),),
    ))
    connector = FakeConnector(first, FakeConnection(response()))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=connector,
            )
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED
    assert location not in str(caught.value)
    assert len(connector.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 503, 404])
@pytest.mark.parametrize(
    ("content_length", "framing_headers"),
    [
        (False, (b"Content-Length: invalid",)),
        (True, (b"Content-Length: 0",)),
        (False, (b"Transfer-Encoding: gzip",)),
        (
            False,
            (b"Transfer-Encoding: chunked", b"Transfer-Encoding: chunked"),
        ),
        (True, (b"Transfer-Encoding: chunked",)),
    ],
)
async def test_invalid_framing_is_terminal_before_status_dispatch(
    tmp_path,
    status,
    content_length,
    framing_headers,
):
    status_headers = framing_headers
    if status == 302:
        status_headers += (b"Location: https://other.example/file",)
    first = FakeConnection(response(
        b"",
        status=status,
        content_length=content_length,
        extra=status_headers,
    ))
    connector = FakeConnector(first, FakeConnection(response()))
    resolver = FakeResolver({
        "media.example": (PUBLIC_V4,),
        "other.example": (PUBLIC_V4,),
    })
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=resolver,
                connector=connector,
            )
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED
    assert resolver.calls == [("media.example", 443)]
    assert len(connector.calls) == 1


@pytest.mark.asyncio
async def test_rejects_declared_length_over_cap_without_writing(tmp_path):
    connection = FakeConnection(response(b"x", content_length=2048))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(byte_limit=10, admitted_bytes=10),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.MEDIA_LIMIT_REACHED


@pytest.mark.asyncio
async def test_rejects_body_longer_than_declared_and_cleans_partial(tmp_path):
    connection = FakeConnection(response(b"abcdef", content_length=3))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.tell() == 0
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.MEDIA_LIMIT_REACHED


@pytest.mark.asyncio
async def test_rejects_body_shorter_than_declared_and_cleans_partial(tmp_path):
    connection = FakeConnection(response(b"abc", content_length=9))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(attempts=1),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.UPSTREAM_UNAVAILABLE


@pytest.mark.asyncio
async def test_actual_bytes_enforce_cap_when_length_missing(tmp_path):
    connection = FakeConnection(response(
        b"6\r\nabcdef\r\n0\r\n\r\n",
        content_length=False,
        extra=(b"Transfer-Encoding: chunked",),
    ))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(byte_limit=5, admitted_bytes=5),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.MEDIA_LIMIT_REACHED


@pytest.mark.asyncio
async def test_free_space_is_checked_at_write_boundary(tmp_path):
    connection = FakeConnection(response(b"abcdef"))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 5,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.STORAGE_GUARD_TRIPPED


@pytest.mark.asyncio
async def test_slow_body_hits_finite_read_deadline_and_cleans_output(tmp_path):
    class BlockingConnection(FakeConnection):
        async def read(self, size):
            await asyncio.Event().wait()

    connection = BlockingConnection(response(b"abc"))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(
                    overall_timeout_seconds=0.1,
                    read_timeout_seconds=0.2,
                    attempts=1,
                ),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.DOWNLOAD_TIMEOUT
    assert connection.abort_calls == 1
    assert not connection.closed


@pytest.mark.asyncio
async def test_rejects_non_media_response(tmp_path):
    connection = FakeConnection(response(b"<html>", content_type=b"text/html"))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED


@pytest.mark.asyncio
async def test_retry_after_is_bounded_and_retry_uses_fresh_resolution(tmp_path):
    first = FakeConnection(response(
        b"",
        status=503,
        extra=(b"Retry-After: 999",),
    ))
    second = FakeConnection(response(b"ok"))
    resolver = FakeResolver()
    connector = FakeConnector(first, second)
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    target = tmp_path / "out"
    with target.open("w+b", buffering=0) as output:
        result = await download(
            "https://media.example/file",
            output,
            policy(max_retry_after_seconds=2, overall_timeout_seconds=10),
            free_space=lambda: 4096,
            resolver=resolver,
            connector=connector,
            sleep=fake_sleep,
        )
        output.seek(0)
        assert output.read() == b"ok"
    assert result.attempts == 2
    assert delays == [2.0]
    assert resolver.calls == [("media.example", 443), ("media.example", 443)]


@pytest.mark.asyncio
async def test_baseexception_aborts_connection_and_cleans_partial(tmp_path):
    class Abort(BaseException):
        pass

    connection = FakeConnection(response(b"abcdef"))
    calls = 0

    def aborting_space():
        nonlocal calls
        calls += 1
        raise Abort()

    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(Abort):
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=aborting_space,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.tell() == 0
        assert output.read() == b""
    assert calls == 1
    assert connection.abort_calls == 1
    assert not connection.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_type", [SystemExit, KeyboardInterrupt])
async def test_caller_control_flow_baseexceptions_propagate_after_cleanup(tmp_path, abort_type):
    connection = FakeConnection(response(b"abcdef"))

    def aborting_space():
        raise abort_type()

    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(abort_type):
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=aborting_space,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.tell() == 0
        assert output.read() == b""
    assert connection.abort_calls == 1
    assert not connection.closed


@pytest.mark.asyncio
async def test_reset_failure_does_not_mask_active_baseexception(tmp_path, monkeypatch):
    class Abort(BaseException):
        pass

    real_write = os.write
    real_ftruncate = os.ftruncate
    write_called = False
    truncate_calls = 0

    def aborting_write(descriptor, data):
        nonlocal write_called
        write_called = True
        real_write(descriptor, data)
        raise Abort()

    def failing_cleanup(descriptor, length):
        nonlocal truncate_calls
        truncate_calls += 1
        if truncate_calls > 1:
            raise OSError("reset failed")
        return real_ftruncate(descriptor, length)

    monkeypatch.setattr(os, "write", aborting_write)
    monkeypatch.setattr(os, "ftruncate", failing_cleanup)
    connection = FakeConnection(response(b"abcdef"))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(Abort):
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
    assert write_called
    assert truncate_calls == 2
    assert connection.abort_calls == 1
    assert not connection.closed


@pytest.mark.asyncio
async def test_cancellation_during_abort_propagates_and_resets_output(tmp_path):
    class CancellingAbortConnection(FakeConnection):
        def abort(self):
            super().abort()
            raise asyncio.CancelledError()

    connection = CancellingAbortConnection(response())
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(asyncio.CancelledError):
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.tell() == 0
        assert output.read() == b""
    assert connection.abort_calls == 1
    assert not connection.closed


@pytest.mark.asyncio
async def test_503_abort_failure_is_closed_and_never_retried(tmp_path):
    class FailingAbortConnection(FakeConnection):
        def abort(self):
            super().abort()
            raise OSError("abort failed")

    first = FailingAbortConnection(response(b"", status=503))
    second = FakeConnection(response(b"unexpected"))
    connector = FakeConnector(first, second)
    target = tmp_path / "out"
    target.write_bytes(b"stale")

    with target.open("r+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=connector,
            )
        assert caught.value.reason_code is ReasonCode.UPSTREAM_UNAVAILABLE
        assert len(connector.calls) == 1
        assert output.tell() == 0
        assert output.read() == b""
    assert first.abort_calls == 1
    assert not first.closed


@pytest.mark.asyncio
async def test_active_cancellation_survives_ordinary_abort_failure_without_retry(tmp_path):
    class FailingAbortConnection(FakeConnection):
        def abort(self):
            super().abort()
            raise OSError("abort failed")

    def cancel_during_body():
        raise asyncio.CancelledError()

    first = FailingAbortConnection(response(b"partial"))
    second = FakeConnection(response(b"unexpected"))
    connector = FakeConnector(first, second)
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(asyncio.CancelledError):
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=cancel_during_body,
                resolver=FakeResolver(),
                connector=connector,
            )
        assert output.read() == b""
    assert len(connector.calls) == 1
    assert first.abort_calls == 1


@pytest.mark.asyncio
async def test_arbitrary_active_baseexception_survives_abort_and_reset_failures(
    tmp_path,
    monkeypatch,
):
    class Stop(BaseException):
        pass

    class FailingAbortConnection(FakeConnection):
        def abort(self):
            super().abort()
            raise OSError("abort failed")

    real_ftruncate = os.ftruncate
    truncate_calls = 0

    def failing_reset(descriptor, length):
        nonlocal truncate_calls
        truncate_calls += 1
        if truncate_calls > 1:
            raise OSError("reset failed")
        return real_ftruncate(descriptor, length)

    def stop_during_body():
        raise Stop()

    monkeypatch.setattr(os, "ftruncate", failing_reset)
    connection = FailingAbortConnection(response(b"partial"))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(Stop):
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=stop_during_body,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
    assert truncate_calls == 2
    assert connection.abort_calls == 1


@pytest.mark.asyncio
async def test_truncated_length_retries_with_reset_and_fresh_resolution(tmp_path):
    first = FakeConnection(response(b"partial", content_length=12))
    second = FakeConnection(response(b"valid"))
    resolver = FakeResolver({
        "media.example": [(PUBLIC_V4,), (PUBLIC_V4,)],
    })
    connector = FakeConnector(first, second)

    result, content = await run_download(
        tmp_path,
        first,
        resolver=resolver,
        connector=connector,
    )

    assert content == b"valid"
    assert result.attempts == 2
    assert resolver.calls == [("media.example", 443), ("media.example", 443)]
    assert len(connector.calls) == 2
    assert first.abort_calls == second.abort_calls == 1


@pytest.mark.asyncio
async def test_truncated_length_exhaustion_resets_and_resolves_every_attempt(tmp_path):
    connections = [
        FakeConnection(response(b"partial", content_length=12))
        for _ in range(3)
    ]
    resolver = FakeResolver({
        "media.example": [(PUBLIC_V4,), (PUBLIC_V4,), (PUBLIC_V4,)],
    })
    connector = FakeConnector(*connections)
    target = tmp_path / "out"
    target.write_bytes(b"stale")

    with target.open("r+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=resolver,
                connector=connector,
            )
        assert output.tell() == 0
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.UPSTREAM_UNAVAILABLE
    assert resolver.calls == [("media.example", 443)] * 3
    assert len(connector.calls) == 3
    assert [connection.abort_calls for connection in connections] == [1, 1, 1]


@pytest.mark.asyncio
async def test_read_timeout_retries_only_while_overall_budget_remains(tmp_path):
    class BlockingConnection(FakeConnection):
        async def read(self, size):
            await asyncio.Event().wait()

    first = BlockingConnection(response(b"slow"))
    second = FakeConnection(response(b"valid"))
    resolver = FakeResolver({
        "media.example": [(PUBLIC_V4,), (PUBLIC_V4,)],
    })
    connector = FakeConnector(first, second)

    result, content = await run_download(
        tmp_path,
        first,
        resolver=resolver,
        connector=connector,
        read_timeout_seconds=0.1,
    )

    assert content == b"valid"
    assert result.attempts == 2
    assert resolver.calls == [("media.example", 443)] * 2
    assert len(connector.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control_flow",
    [asyncio.CancelledError, KeyboardInterrupt, SystemExit],
)
async def test_control_flow_during_retryable_abort_propagates(tmp_path, control_flow):
    class InterruptingAbortConnection(FakeConnection):
        def abort(self):
            super().abort()
            raise control_flow()

    first = InterruptingAbortConnection(response(b"", status=503))
    second = FakeConnection(response(b"unexpected"))
    connector = FakeConnector(first, second)
    target = tmp_path / "out"
    target.write_bytes(b"stale")

    with target.open("r+b", buffering=0) as output:
        with pytest.raises(control_flow):
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=connector,
            )
        assert len(connector.calls) == 1
        assert output.tell() == 0
        assert output.read() == b""
    assert first.abort_calls == 1
    assert not first.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control_flow",
    [asyncio.CancelledError, KeyboardInterrupt, SystemExit],
)
async def test_control_flow_during_active_exception_reset_propagates(
    tmp_path,
    monkeypatch,
    control_flow,
):
    real_ftruncate = os.ftruncate
    truncate_calls = 0

    def interrupting_reset(descriptor, length):
        nonlocal truncate_calls
        truncate_calls += 1
        if truncate_calls > 1:
            raise control_flow()
        return real_ftruncate(descriptor, length)

    def failing_space():
        raise RuntimeError("active failure")

    monkeypatch.setattr(os, "ftruncate", interrupting_reset)
    connection = FakeConnection(response())
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(control_flow):
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=failing_space,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
    assert truncate_calls == 2
    assert connection.abort_calls == 1


def test_direct_connection_abort_closes_and_aborts_transport_once():
    class Transport:
        abort_calls = 0

        def abort(self):
            self.abort_calls += 1

    class Writer:
        def __init__(self):
            self.transport = Transport()
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    writer = Writer()
    connection = DirectConnection(None, writer)
    connection.abort()
    connection.abort()
    assert writer.close_calls == 1
    assert writer.transport.abort_calls == 1


@pytest.mark.asyncio
async def test_cleanup_creates_no_task_or_deadline_overrun(tmp_path):
    class BlockingGracefulCloseConnection(FakeConnection):
        close_called = False

        async def close(self):
            self.close_called = True
            await asyncio.Event().wait()

    connection = BlockingGracefulCloseConnection(response())
    tasks_before = asyncio.all_tasks()
    started = asyncio.get_running_loop().time()
    result, content = await run_download(
        tmp_path,
        connection,
        overall_timeout_seconds=2.0,  # generous for slow CI disks; a blocked close would hang past it
        read_timeout_seconds=0.1,
    )
    elapsed = asyncio.get_running_loop().time() - started
    assert result.bytes_written == 5
    assert content == b"media"
    assert elapsed < 2.0
    assert asyncio.all_tasks() == tasks_before
    assert connection.abort_calls == 1
    assert not connection.close_called


@pytest.mark.asyncio
async def test_post_connect_deadline_expiry_aborts_owned_connection(tmp_path):
    class Clock:
        current = 0.0

        def __call__(self):
            return self.current

    clock = Clock()
    connection = FakeConnection(response())

    class ExpiringConnector(FakeConnector):
        async def connect(self, address, port, *, server_hostname, ssl_context):
            result = await super().connect(
                address,
                port,
                server_hostname=server_hostname,
                ssl_context=ssl_context,
            )
            clock.current = 3
            return result

    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=ExpiringConnector(connection),
                clock=clock,
            )
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.DOWNLOAD_TIMEOUT
    assert connection.abort_calls == 1
    assert not connection.closed


@pytest.mark.asyncio
async def test_prompt_native_connect_timeout_falls_back_to_next_validated_address(
    tmp_path,
):
    connection = FakeConnection(response(), peer=PUBLIC_V6)

    class PromptTimeoutConnector(FakeConnector):
        async def connect(self, address, port, *, server_hostname, ssl_context):
            self.calls.append((address, port, server_hostname, ssl_context))
            if len(self.calls) == 1:
                raise TimeoutError("prompt native timeout")
            return self.connections.pop(0)

    connector = PromptTimeoutConnector(connection)
    result, content = await run_download(
        tmp_path,
        connection,
        resolver=FakeResolver({"media.example": (PUBLIC_V4, PUBLIC_V6)}),
        connector=connector,
    )
    assert result.attempts == 1
    assert content == b"media"
    assert [call[0] for call in connector.calls] == [PUBLIC_V4, PUBLIC_V6]


@pytest.mark.asyncio
async def test_connect_wait_expiry_is_terminal_without_address_fallback(tmp_path):
    class SlowConnector:
        def __init__(self):
            self.calls = []

        async def connect(self, address, port, *, server_hostname, ssl_context):
            self.calls.append(address)
            await asyncio.sleep(1)
            raise AssertionError("unreachable")

    connector = SlowConnector()
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(overall_timeout_seconds=0.1),
                free_space=lambda: 4096,
                resolver=FakeResolver({"media.example": (PUBLIC_V4, PUBLIC_V6)}),
                connector=connector,
            )
    assert caught.value.reason_code is ReasonCode.DOWNLOAD_TIMEOUT
    assert connector.calls == [PUBLIC_V4]


@pytest.mark.asyncio
async def test_certificate_verification_failure_is_terminal_without_retry(tmp_path):
    class CertificateFailureConnector:
        def __init__(self):
            self.calls = []

        async def connect(self, address, port, *, server_hostname, ssl_context):
            self.calls.append(address)
            raise ssl.SSLCertVerificationError(1, "hostname mismatch")

    connector = CertificateFailureConnector()
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver({"media.example": (PUBLIC_V4, PUBLIC_V6)}),
                connector=connector,
            )
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED
    assert connector.calls == [PUBLIC_V4]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    ["connector", "write", "header", "body", "eof", "abort"],
)
@pytest.mark.parametrize(
    "certificate_error_type",
    [
        ssl.SSLCertVerificationError,
        type(
            "HostnameCertificateError",
            (ssl.SSLCertVerificationError,),
            {},
        ),
    ],
)
async def test_certificate_failure_is_terminal_at_every_transport_phase(
    tmp_path,
    phase,
    certificate_error_type,
):
    certificate_error = certificate_error_type(1, "hostname mismatch")

    class CertificatePhaseConnection(FakeConnection):
        async def write(self, data):
            if phase == "write":
                raise certificate_error
            await super().write(data)

        async def read(self, size):
            if phase == "header":
                raise certificate_error
            if phase in {"body", "eof"} and not self.response:
                raise certificate_error
            return await super().read(size)

        def abort(self):
            super().abort()
            if phase == "abort":
                raise certificate_error

    if phase == "body":
        payload = response(b"partial", content_length=12)
    else:
        payload = response(b"media")
    first = CertificatePhaseConnection(payload)
    valid_second = FakeConnection(response(b"valid"))

    class CertificatePhaseConnector(FakeConnector):
        async def connect(self, address, port, *, server_hostname, ssl_context):
            self.calls.append((address, port, server_hostname, ssl_context))
            if phase == "connector":
                raise certificate_error
            return self.connections.pop(0)

    connector = CertificatePhaseConnector(first, valid_second)
    resolver = FakeResolver({
        "media.example": [(PUBLIC_V4,), (PUBLIC_V4,)],
    })
    target = tmp_path / "out"
    target.write_bytes(b"stale")

    with target.open("r+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=resolver,
                connector=connector,
            )
        assert output.tell() == 0
        assert output.read() == b""

    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED
    assert resolver.calls == [("media.example", 443)]
    assert len(connector.calls) == 1
    assert valid_second.request == b""
    assert first.abort_calls == (0 if phase == "connector" else 1)


@pytest.mark.asyncio
async def test_direct_connector_setblocking_oserror_closes_before_connect_or_tls(
    monkeypatch,
):
    failure = OSError("setblocking failed")

    class Socket:
        close_calls = 0

        def setblocking(self, value):
            assert value is False
            raise failure

        def close(self):
            self.close_calls += 1

    connect_calls = 0
    tls_calls = 0

    async def connect(sock, endpoint):
        nonlocal connect_calls
        connect_calls += 1

    async def open_connection(**kwargs):
        nonlocal tls_calls
        tls_calls += 1

    sock = Socket()
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(socket, "socket", lambda *args: sock)
    monkeypatch.setattr(loop, "sock_connect", connect)
    monkeypatch.setattr(asyncio, "open_connection", open_connection)

    with pytest.raises(OSError) as caught:
        await DirectConnector().connect(
            PUBLIC_V4,
            443,
            server_hostname="media.example",
            ssl_context=ssl.create_default_context(),
        )

    assert caught.value is failure
    assert sock.close_calls == 1
    assert connect_calls == 0
    assert tls_calls == 0


@pytest.mark.asyncio
async def test_direct_connector_setblocking_baseexception_survives_close_failure(
    monkeypatch,
):
    class Stop(BaseException):
        pass

    failure = Stop()

    class Socket:
        close_calls = 0

        def setblocking(self, value):
            assert value is False
            raise failure

        def close(self):
            self.close_calls += 1
            raise OSError("close failed")

    connect_calls = 0
    tls_calls = 0

    async def connect(sock, endpoint):
        nonlocal connect_calls
        connect_calls += 1

    async def open_connection(**kwargs):
        nonlocal tls_calls
        tls_calls += 1

    sock = Socket()
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(socket, "socket", lambda *args: sock)
    monkeypatch.setattr(loop, "sock_connect", connect)
    monkeypatch.setattr(asyncio, "open_connection", open_connection)

    with pytest.raises(Stop) as caught:
        await DirectConnector().connect(
            PUBLIC_V4,
            443,
            server_hostname="media.example",
            ssl_context=ssl.create_default_context(),
        )

    assert caught.value is failure
    assert sock.close_calls == 1
    assert connect_calls == 0
    assert tls_calls == 0


@pytest.mark.asyncio
async def test_direct_connector_close_cannot_mask_active_control_flow(monkeypatch):
    class Stop(BaseException):
        pass

    class Socket:
        def setblocking(self, value):
            assert value is False

        def close(self):
            raise OSError("close failed")

    async def fail_connect(sock, endpoint):
        raise Stop()

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(socket, "socket", lambda *args: Socket())
    monkeypatch.setattr(loop, "sock_connect", fail_connect)
    with pytest.raises(Stop):
        await DirectConnector().connect(
            PUBLIC_V4,
            443,
            server_hostname="media.example",
            ssl_context=ssl.create_default_context(),
        )


@pytest.mark.asyncio
async def test_direct_connector_new_close_control_flow_propagates(monkeypatch):
    class Stop(BaseException):
        pass

    class Socket:
        def setblocking(self, value):
            assert value is False

        def close(self):
            raise Stop()

    async def fail_connect(sock, endpoint):
        raise OSError("connect failed")

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(socket, "socket", lambda *args: Socket())
    monkeypatch.setattr(loop, "sock_connect", fail_connect)
    with pytest.raises(Stop):
        await DirectConnector().connect(
            PUBLIC_V4,
            443,
            server_hostname="media.example",
            ssl_context=ssl.create_default_context(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    [
        b"audio/",
        b"audio/mpeg; charset",
        b"audio/mpeg; charset=",
        b"audio/mpeg; bad@name=value",
        b"audio/mpeg; name=\"unterminated",
        b"audio/mpeg trailing",
        b"application/json",
    ],
)
async def test_rejects_syntactically_invalid_or_unapproved_media_types(
    tmp_path,
    content_type,
):
    connection = FakeConnection(response(content_type=content_type))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED


@pytest.mark.asyncio
async def test_accepts_valid_media_type_parameters(tmp_path):
    result, content = await run_download(
        tmp_path,
        FakeConnection(response(content_type=b'Audio/MPEG; name="a;b.mp3"; x=y')),
    )
    assert result.content_type == "audio/mpeg"
    assert content == b"media"


@pytest.mark.asyncio
async def test_accepts_case_insensitive_chunked_transfer_coding(tmp_path):
    result, content = await run_download(
        tmp_path,
        FakeConnection(response(
            b"5\r\nmedia\r\n0\r\n\r\n",
            content_length=False,
            extra=(b"Transfer-Encoding: ChUnKeD",),
        )),
    )
    assert result.bytes_written == 5
    assert content == b"media"


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", ["content-length", "chunked"])
async def test_exact_framed_body_requires_connection_close_eof(tmp_path, framing):
    if framing == "chunked":
        payload = response(
            b"5\r\nmedia\r\n0\r\n\r\n",
            content_length=False,
            extra=(b"Transfer-Encoding: chunked",),
        )
    else:
        payload = response(b"media")
    connection = StagedConnection(payload, b"")

    result, content = await run_download(tmp_path, connection)

    assert result.bytes_written == 5
    assert content == b"media"
    assert connection.read_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("framing", "expected_reason"),
    [
        ("content-length", ReasonCode.MEDIA_LIMIT_REACHED),
        ("chunked", ReasonCode.CONTENT_DENIED),
    ],
)
async def test_rejects_late_next_read_byte_after_exact_framed_body(
    tmp_path,
    framing,
    expected_reason,
):
    if framing == "chunked":
        payload = response(
            b"5\r\nmedia\r\n0\r\n\r\n",
            content_length=False,
            extra=(b"Transfer-Encoding: chunked",),
        )
    else:
        payload = response(b"media")
    connection = StagedConnection(payload, b"x")

    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.read() == b""
    assert caught.value.reason_code is expected_reason
    assert connection.read_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("trailing", [b"x", b"\r", b"\r\nHTTP/1.1 200 OK\r\n"])
async def test_rejects_buffered_bytes_after_chunked_terminator(tmp_path, trailing):
    connection = FakeConnection(response(
        b"5\r\nmedia\r\n0\r\n\r\n" + trailing,
        content_length=False,
        extra=(b"Transfer-Encoding: chunked",),
    ))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transfer_headers",
    [
        (b"Transfer-Encoding: chunked; x=y",),
        (b"Transfer-Encoding: gzip, chunked",),
        (b"Transfer-Encoding: gzip",),
        (b"Transfer-Encoding: chunked", b"Transfer-Encoding: chunked"),
    ],
)
async def test_rejects_parameterized_multiple_or_unsupported_transfer_codings(
    tmp_path,
    transfer_headers,
):
    connection = FakeConnection(response(
        b"5\r\nmedia\r\n0\r\n\r\n",
        content_length=False,
        extra=transfer_headers,
    ))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED


@pytest.mark.asyncio
async def test_rejects_oversized_content_length_digits_before_conversion(tmp_path):
    connection = FakeConnection(response(content_length=int(b"9" * 21)))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
            )
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.CONTENT_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize("clock_failure", ["error", "expired"])
async def test_pre_wait_clock_failure_closes_unstarted_resolver_coroutine(
    tmp_path,
    clock_failure,
):
    class Clock:
        fail = False

        def __call__(self):
            if not self.fail:
                return 0
            if clock_failure == "error":
                raise RuntimeError("clock failed")
            return 2

    clock = Clock()

    class TrackingResolver:
        def __init__(self):
            self.awaitable = None

        def resolve(self, host, port):
            async def resolution():
                return (PUBLIC_V4,)

            self.awaitable = resolution()
            clock.fail = True
            return self.awaitable

    resolver = TrackingResolver()
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=resolver,
                connector=FakeConnector(FakeConnection(response())),
                clock=clock,
            )
    assert caught.value.reason_code is ReasonCode.DOWNLOAD_TIMEOUT
    assert resolver.awaitable.cr_frame is None


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["free_space", "write", "fsync"])
async def test_absolute_deadline_is_rechecked_after_local_io(tmp_path, monkeypatch, boundary):
    class Clock:
        current = 0.0

        def __call__(self):
            return self.current

    clock = Clock()
    real_write = os.write
    real_fsync = os.fsync

    def free_space():
        if boundary == "free_space":
            clock.current = 3
        return 4096

    def write(descriptor, data):
        count = real_write(descriptor, data)
        if boundary == "write":
            clock.current = 3
        return count

    def fsync(descriptor):
        real_fsync(descriptor)
        if boundary == "fsync":
            clock.current = 3

    monkeypatch.setattr(os, "write", write)
    monkeypatch.setattr(os, "fsync", fsync)
    connection = FakeConnection(response())
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/file",
                output,
                policy(),
                free_space=free_space,
                resolver=FakeResolver(),
                connector=FakeConnector(connection),
                clock=clock,
            )
        assert output.read() == b""
    assert caught.value.reason_code is ReasonCode.DOWNLOAD_TIMEOUT
    assert connection.aborted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://media.example/file",
        "ftp://media.example/file",
        "https://user:secret@media.example/file",
        "https://media.example/file#fragment",
        "https://127.0.0.1/file",
    ],
)
async def test_rejects_unsafe_url_forms(tmp_path, url):
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError):
            await download(
                url,
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver({"127.0.0.1": ("127.0.0.1",)}),
                connector=FakeConnector(FakeConnection(response())),
            )


@pytest.mark.asyncio
async def test_rejects_initial_url_utf8_lone_surrogate_before_network(tmp_path):
    connector = FakeConnector(FakeConnection(response()))
    with (tmp_path / "out").open("w+b", buffering=0) as output:
        with pytest.raises(DownloadError) as caught:
            await download(
                "https://media.example/\ud800",
                output,
                policy(),
                free_space=lambda: 4096,
                resolver=FakeResolver(),
                connector=connector,
            )
    assert caught.value.reason_code is ReasonCode.INPUT_REJECTED
    assert connector.calls == []


def test_policy_rejects_unadmitted_or_unbounded_values():
    with pytest.raises(DownloadError):
        policy(admitted_bytes=1025)
    with pytest.raises(DownloadError):
        policy(attempts=4)
    with pytest.raises(DownloadError):
        policy(max_redirects=6)
    with pytest.raises(DownloadError):
        policy(read_timeout_seconds=float("inf"))


def test_reason_domain_rejects_registry_negative_examples():
    registry = json.loads(
        (Path(__file__).parents[1] / "Docs" / "contract-domains.json").read_text(
            encoding="utf-8"
        )
    )
    assert ReasonCode.parse("content_denied") is ReasonCode.CONTENT_DENIED
    for value in registry["negative_examples"]:
        with pytest.raises(DomainValidationError):
            ReasonCode.parse(value)


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_own_transport_smoke():
    url = os.environ.get("PODMUNCHER_TEST_DIRECT_URL")
    assert url, "PODMUNCHER_TEST_DIRECT_URL is required for the live smoke"
    descriptor, raw_path = tempfile.mkstemp(prefix="podmuncher-direct-", suffix=".partial")
    path = Path(raw_path)
    descriptor_open = True
    try:
        os.close(descriptor)
        descriptor_open = False
        with path.open("w+b", buffering=0) as output:
            result = await download(
                url,
                output,
                DownloadPolicy(
                    byte_limit=64 * 1024 * 1024,
                    admitted_bytes=64 * 1024 * 1024,
                    overall_timeout_seconds=300,
                    read_timeout_seconds=30,
                ),
                free_space=lambda: shutil.disk_usage(path.parent).free,
                resolver=SystemResolver(),
                connector=DirectConnector(),
            )
            assert result.bytes_written > 0
            assert path.stat().st_size == result.bytes_written
    finally:
        active_error = sys.exception()
        cleanup_error = None
        if descriptor_open:
            try:
                os.close(descriptor)
            except BaseException as error:  # noqa: BLE001 - preserve active error
                cleanup_error = error
        try:
            path.unlink(missing_ok=True)
        except BaseException as error:  # noqa: BLE001 - cleanup must not mask active error
            if cleanup_error is None:
                cleanup_error = error
        if active_error is None and cleanup_error is not None:
            raise cleanup_error
    assert not path.exists()
    print("PODMUNCHER_SMOKE_OK")
