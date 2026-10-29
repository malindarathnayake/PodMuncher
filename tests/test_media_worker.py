# ruff: noqa: C408
"""Media worker (P3-U5): deterministic contract tests and an admitted live smoke.

Deterministic cases use injected fakes only: no network, no Docker, no wall clock.
The live case builds the owned Dockerfile through an admitted helper and drives the
real DockerRuntime; it requires PODMUNCHER_TEST_DIRECT_URL and
PODMUNCHER_TEST_YOUTUBE_URL and never prints their values.
"""
import asyncio
import errno
import functools
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from podmuncher import direct_download, media_worker
from podmuncher.admission import WorkloadGrowth
from podmuncher.config import Limits
from podmuncher.direct_download import DownloadError, DownloadPolicy, DownloadResult
from podmuncher.docker_storage import make_docker_storage_provider
from podmuncher.domains import (
    Capability,
    DomainValidationError,
    ReasonCode,
    Source,
    VideoQuality,
    WorkerSlot,
)
from podmuncher.domains import WorkerEvent as WorkerEventKind
from podmuncher.guards import CancellationCoordinator, WriteGuard
from podmuncher.protocol import WorkerEventRecord, WorkerEventStream, WorkerRequest
from podmuncher.requests import RequestOutputs, classify_source
from podmuncher.runtime import (
    DockerRuntime,
    DockerTarget,
    NativeDockerCLI,
    RuntimeStorage,
    WorkerMounts,
    WorkerPolicy,
)

JOB, ATTEMPT, STAGE = ('11111111-1111-4111-8111-111111111111', '22222222-2222-4222-8222-222222222222',
                       '33333333-3333-4333-8333-333333333333')
VIDEO_ID = 'abcDEF_-123'
YOUTUBE = {'kind': 'youtube', 'input': 'https://www.youtube.com/watch?v=' + VIDEO_ID}
DIRECT = {'kind': 'direct', 'input': 'https://media.example.com/episode.mp3'}
LIMITS = Limits()
DENO = '/opt/deno'
META = 'meta.json'
SECRET_TITLE = 'Secret Title Do Not Leak'
PROBE_KEYS = ['schema_version', 'operation', 'source_kind', 'duration_seconds', 'has_audio',
              'has_video', 'video_height', 'audio_streams', 'video_streams', 'estimated_bytes']


def make_worker_request(**changes):
    """Mirror tests/test_runtime.py request(), defaulting to the media capability."""
    data = dict(schema_version=1, job_id=JOB, attempt_id=ATTEMPT, stage_id=STAGE,
        capability='media', operation='probe', assigned_slots={'probe_metadata': META}, source=None,
        outputs=dict(transcript=True, video=False, video_quality='source_best'),
        input_basename='input.bin', model_snapshot_id=None, limits=asdict(LIMITS),
        selection=dict(device='cpu', profile='balanced', model='auto', language='auto'))
    data.update(changes)
    return WorkerRequest.from_dict(data)


def remote_request(operation, *, source=YOUTUBE, slots=None, video=False, quality='1080p', limits=None):
    if slots is None:
        if operation == 'metadata':
            slots = {'probe_metadata': META}
        elif source['kind'] == 'direct':
            slots = {'downloaded_media': 'media.bin'}
        elif video:
            slots = {'downloaded_video': 'video.bin', 'downloaded_audio': 'audio.bin',
                     'merged_video': 'merged.bin'}
        else:
            slots = {'downloaded_audio': 'audio.bin'}
    return make_worker_request(operation=operation, source=dict(source), input_basename=None,
        assigned_slots=slots, outputs=dict(transcript=True, video=video, video_quality=quality),
        limits=asdict(limits or LIMITS))


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class Collector:
    """Event sink that remembers the injected clock at each write."""

    def __init__(self, clock=None):
        self.chunks, self.times, self.clock = [], [], clock

    def __call__(self, data):
        assert type(data) is bytes and data.endswith(b'\n') and data.count(b'\n') == 1
        self.chunks.append(data)
        self.times.append(None if self.clock is None else self.clock.now)

    @property
    def records(self):
        return [WorkerEventRecord.from_json(c[:-1]) for c in self.chunks]

    @property
    def events(self):
        return [r.event for r in self.records]

    def of(self, kind):
        return [r for r in self.records if r.event == kind]


def fake_ffprobe(payload=None, *, returncode=0, raw=None, merge_bytes=b'MERGED', merge_returncode=0):
    """Scripted run_process: ffprobe JSON / return code, and an ffmpeg that writes its output."""
    calls = []

    def run_process(arguments, timeout):
        calls.append((tuple(arguments), timeout))
        assert all(type(a) is str for a in arguments)
        if arguments[0] == 'ffprobe':
            stdout = raw if raw is not None else json.dumps(payload).encode()
            return subprocess.CompletedProcess(arguments, returncode, stdout, None)
        assert arguments[0] == 'ffmpeg'
        if merge_returncode == 0:
            Path(arguments[-1]).write_bytes(merge_bytes)
        return subprocess.CompletedProcess(arguments, merge_returncode, b'', None)

    run_process.calls = calls
    return run_process


class FakeYdl:
    """Fake YoutubeDL factory: records options, scripts extract_info and dl, drives progress hooks.

    ``downloads`` maps a format_id to ``(bytes | None, [reported downloaded_bytes])``.
    """

    def __init__(self, info=None, downloads=None, *, clock=None, step=0.0, error=None):
        self.info, self.downloads, self.clock, self.step, self.error = info, downloads or {}, clock, step, error
        self.instances, self.calls, self.dl_calls = [], [], []

    def __call__(self, options):
        instance = _FakeInstance(self, options)
        self.instances.append(instance)
        return instance


class _FakeInstance:
    def __init__(self, owner, options):
        self.owner, self.options = owner, options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        owner, options = self.owner, self.options
        owner.calls.append((url, download, options))
        assert download is False, 'the worker must never let extract_info download'
        return owner.info(options) if callable(owner.info) else owner.info

    def dl(self, name, info, subtitle=False, test=False):
        owner, options = self.owner, self.options
        owner.dl_calls.append((name, info, options))
        assert len(owner.calls) == 1 and len(owner.instances) == 1, 'download must reuse the one extraction'
        assert subtitle is False and test is False
        if owner.error is not None:
            raise owner.error
        data, counts = owner.downloads[info['format_id']]
        for count in counts:
            if owner.clock is not None:
                owner.clock.now += owner.step
            for hook in options['progress_hooks']:
                hook({'status': 'downloading', 'filename': name, 'downloaded_bytes': count})
        if data is None:  # scripts yt-dlp declining a stream above max_filesize
            return False, False
        Path(name).write_bytes(data)
        return True, True


def fake_ydl(info=None, downloads=None, **kwargs):
    return FakeYdl(info, downloads, **kwargs)


def forbidden(*_args, **_kwargs):
    raise AssertionError('unexpected I/O')


async def forbidden_direct(*_args, **_kwargs):
    raise AssertionError('unexpected I/O')


def merged_info(**changes):
    info = {'id': VIDEO_ID, 'title': SECRET_TITLE, 'uploader': 'Someone Private',
            'description': 'private words', 'webpage_url': YOUTUBE['input'], 'duration': 120.5,
            'requested_formats': [
                {'format_id': '137', 'vcodec': 'avc1.640028', 'acodec': 'none', 'height': 1080},
                {'format_id': '140', 'vcodec': 'none', 'acodec': 'mp4a.40.2'}]}
    info.update(changes)
    return info


def sized(info, video=None, audio=None):
    info = dict(info, requested_formats=[dict(f) for f in info['requested_formats']])
    if video is not None:
        info['requested_formats'][0]['filesize'] = video
    if audio is not None:
        info['requested_formats'][1]['filesize_approx'] = audio
    return info


class Outcome:
    def __init__(self, code, collector, scratch, input_root):
        self.code, self.collector, self.scratch, self.input_root = code, collector, scratch, input_root

    @property
    def events(self):
        return self.collector.events

    @property
    def error(self):
        errors = self.collector.of('error')
        return errors[0].payload['reason_code'] if errors else None

    @property
    def files(self):
        return sorted(p.name for p in self.scratch.iterdir())


def execute(tmp_path, request, *, ydl=None, process=None, direct=forbidden_direct, clock=None,
            free_space=lambda: 10**12, **kwargs):
    """Run the worker once and prove the emitted bytes are a host-valid event stream."""
    scratch, input_root = tmp_path / 'scratch', tmp_path / 'input'
    scratch.mkdir(parents=True, exist_ok=True)
    input_root.mkdir(parents=True, exist_ok=True)
    clock = clock or Clock()
    collector = Collector(clock)
    code = media_worker.run(request, collector, scratch=scratch, input_root=input_root,
        ydl_factory=ydl or forbidden, run_process=process or forbidden, direct=direct,
        free_space=free_space, clock=clock, js_runtime_path=DENO, **kwargs)
    stream = WorkerEventStream(request)
    for chunk, written_at in zip(collector.chunks, collector.times):
        stream.clock = lambda value=written_at: value  # the host sees each event when it was written
        stream.feed(chunk)
    stream.finish()
    events = []
    while (item := stream.pop()) is not None:
        events.append(item)
    assert events[0].event == 'started' and [e.sequence for e in events] == list(range(len(events)))
    assert events[-1].event == ('completed' if code == 0 else 'error')
    assert sum(e.event in ('completed', 'error') for e in events) == 1
    assert (code == 0) == (events[-1].event == 'completed')
    return Outcome(code, collector, scratch, input_root)


def probe_outcome(tmp_path, payload=None, *, returncode=0, raw=None, limits=None, create=True):
    request = make_worker_request(limits=asdict(limits or LIMITS))
    (tmp_path / 'input').mkdir(parents=True, exist_ok=True)
    if create:
        (tmp_path / 'input' / 'input.bin').write_bytes(b'media')
    process = fake_ffprobe(payload, returncode=returncode, raw=raw)
    outcome = execute(tmp_path, request, process=process)
    return outcome, process


def read_meta(outcome, name=META):
    return json.loads((outcome.scratch / name).read_bytes())


def record(**changes):
    base = dict(schema_version=1, operation='probe', source_kind=None, duration_seconds=None,
                has_audio=None, has_video=None, video_height=None, audio_streams=None,
                video_streams=None, estimated_bytes=None)
    base.update(changes)
    return base


AUDIO = {'codec_type': 'audio', 'codec_name': 'aac'}
VIDEO = {'codec_type': 'video', 'codec_name': 'h264', 'height': 720, 'disposition': {'attached_pic': 0}}
COVER = {'codec_type': 'video', 'codec_name': 'mjpeg', 'height': 600, 'disposition': {'attached_pic': 1}}


# ---------------------------------------------------------------------------
# Argument arrays
# ---------------------------------------------------------------------------

def test_ffprobe_arguments_are_a_literal_array():
    assert media_worker.ffprobe_arguments('/input/a.bin') == (
        'ffprobe', '-v', 'error', '-hide_banner', '-protocol_whitelist', 'file',
        '-print_format', 'json', '-show_format', '-show_streams', '-i', '/input/a.bin')
    for bad in ('', '-evil', 'a\nb', 5, None):
        with pytest.raises(ValueError):
            media_worker.ffprobe_arguments(bad)


def test_ffmpeg_merge_arguments_are_literal_stream_copy():
    assert media_worker.ffmpeg_merge_arguments('/scratch/v', '/scratch/a', '/scratch/m', 4096) == (
        'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
        '-protocol_whitelist', 'file', '-i', '/scratch/v', '-protocol_whitelist', 'file', '-i', '/scratch/a',
        '-map', '0:v:0', '-map', '1:a:0', '-c', 'copy', '-fs', '4096', '-f', 'matroska', '/scratch/m')
    assert media_worker.ffmpeg_merge_arguments('/scratch/v', None, '/scratch/m', 7) == (
        'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
        '-protocol_whitelist', 'file', '-i', '/scratch/v',
        '-map', '0:v:0', '-map', '0:a:0?', '-c', 'copy', '-fs', '7', '-f', 'matroska', '/scratch/m')
    arguments = media_worker.ffmpeg_merge_arguments('/s/v', '/s/a', '/s/m', 1)
    assert '-y' not in arguments and arguments[-1] == '/s/m' and '-c' in arguments
    assert arguments[arguments.index('-c') + 1] == 'copy'
    for bad in (0, -1, True, '5', None):
        with pytest.raises(ValueError):
            media_worker.ffmpeg_merge_arguments('/s/v', None, '/s/m', bad)


class FakePopen:
    """Popen stand-in: stdout is a bytes stream; records kill/wait."""

    def __init__(self, arguments, data=b'{}', returncode=0, **kwargs):
        self.arguments, self.kwargs, self.returncode = arguments, kwargs, returncode
        self.stdout, self.killed, self.reaped = io.BytesIO(data), False, False
        self.read_sizes = []
        original = self.stdout.read
        self.stdout.read = lambda size=-1: (self.read_sizes.append(size), original(size))[1]

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.reaped = True
        return self.returncode


def test_default_process_runner_uses_argument_array_without_shell(monkeypatch):
    made = []

    def fake_popen(arguments, **kwargs):
        made.append(FakePopen(arguments, **kwargs))
        return made[-1]

    monkeypatch.setattr(media_worker.subprocess, 'Popen', fake_popen)
    result = media_worker._default_run_process(('ffprobe', '-i', 'x'), 12.5)
    process = made[0]
    assert process.arguments == ['ffprobe', '-i', 'x'] and result.stdout == b'{}' and result.returncode == 0
    assert process.kwargs['shell'] is False
    assert process.kwargs['stderr'] is subprocess.DEVNULL and process.kwargs['stdin'] is subprocess.DEVNULL
    assert process.kwargs['stdout'] is subprocess.PIPE and not process.killed


def test_default_process_runner_caps_stdout_then_kills_and_reaps(monkeypatch):
    cap = media_worker._MAX_PROBE_STDOUT
    made = []

    def fake_popen(arguments, **kwargs):
        made.append(FakePopen(arguments, b'x' * (cap * 2), **kwargs))
        return made[-1]

    monkeypatch.setattr(media_worker.subprocess, 'Popen', fake_popen)
    result = media_worker._default_run_process(('ffprobe', '-i', 'x'), 12.5)
    process = made[0]
    assert process.read_sizes == [cap + 1] and len(result.stdout) == cap + 1
    assert process.killed and process.reaped and result.returncode != 0


def test_oversized_probe_output_is_media_probe_failed(tmp_path):
    cap = media_worker._MAX_PROBE_STDOUT
    outcome, _ = probe_outcome(tmp_path, raw=b' ' * (cap + 1))
    assert outcome.error == 'media_probe_failed' and outcome.files == []


@pytest.mark.parametrize(('video', 'quality', 'expected'), [
    (False, '1080p', 'ba/b'), (False, 'source_best', 'ba/b'),
    (True, '1080p', 'bv*[height<=1080]+ba/b[height<=1080]'), (True, 'source_best', 'bv*+ba/b')])
def test_format_selectors(video, quality, expected):
    assert media_worker.format_selector(video, quality) == expected


def test_module_imports_without_yt_dlp_and_run_defaults_reuse_direct_download():
    assert 'yt_dlp' not in sys.modules
    assert media_worker.run.__kwdefaults__['direct'] is direct_download.download


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def test_probe_audio_only(tmp_path):
    outcome, process = probe_outcome(tmp_path, {'streams': [AUDIO], 'format': {'duration': '60.5', 'size': '1000'}})
    assert outcome.code == 0 and outcome.events == ['started', 'artifact', 'completed']
    assert read_meta(outcome) == record(duration_seconds=60.5, has_audio=True, has_video=False,
        audio_streams=1, video_streams=0, estimated_bytes=1000)
    assert list(read_meta(outcome)) == PROBE_KEYS
    assert process.calls == [(media_worker.ffprobe_arguments(str(tmp_path / 'input' / 'input.bin')),
                              float(LIMITS.decode_timeout_seconds))]
    data = (outcome.scratch / META).read_bytes()
    artifact = outcome.collector.of('artifact')[0].payload
    assert dict(artifact) == {'slot': 'probe_metadata', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    assert b' ' not in data and len(data) <= 65536


def test_probe_video_with_audio(tmp_path):
    outcome, _ = probe_outcome(tmp_path, {'streams': [VIDEO, AUDIO, dict(AUDIO)],
                                          'format': {'duration': '10.25', 'size': '12'}})
    assert read_meta(outcome) == record(duration_seconds=10.25, has_audio=True, has_video=True,
        video_height=720, audio_streams=2, video_streams=1, estimated_bytes=12)


def test_probe_silent_video_has_no_audio(tmp_path):
    outcome, _ = probe_outcome(tmp_path, {'streams': [VIDEO], 'format': {'duration': '3'}})
    assert read_meta(outcome) == record(duration_seconds=3.0, has_audio=False, has_video=True,
        video_height=720, audio_streams=0, video_streams=1)


@pytest.mark.parametrize(('streams', 'audio'), [([AUDIO, COVER], 1), ([COVER], 0)])
def test_attached_picture_is_not_video(tmp_path, streams, audio):
    outcome, _ = probe_outcome(tmp_path, {'streams': streams, 'format': {'duration': '5'}})
    assert read_meta(outcome) == record(duration_seconds=5.0, has_audio=audio == 1, has_video=False,
        audio_streams=audio, video_streams=0)


@pytest.mark.parametrize('case', [
    dict(returncode=1, payload={'streams': [AUDIO]}),
    dict(raw=b'not json at all'),
    dict(raw=b''),
    dict(raw=b'[1,2]'),
    dict(payload={'streams': []}),
    dict(payload={'format': {'duration': '1'}}),
    dict(payload={'streams': ['x']}),
])
def test_corrupt_probe_is_media_probe_failed(tmp_path, case):
    outcome, _ = probe_outcome(tmp_path, case.get('payload'), returncode=case.get('returncode', 0),
                               raw=case.get('raw'))
    assert outcome.code == 1 and outcome.events == ['started', 'error']
    assert outcome.error == 'media_probe_failed' and outcome.files == []


def test_missing_input_file_is_media_probe_failed(tmp_path):
    outcome, process = probe_outcome(tmp_path, {'streams': [AUDIO]}, create=False)
    assert outcome.error == 'media_probe_failed' and process.calls == []


def test_probe_process_failures_are_media_probe_failed(tmp_path):
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input' / 'input.bin').write_bytes(b'x')
    for failure in (subprocess.TimeoutExpired('ffprobe', 1), FileNotFoundError(), OSError(errno.EIO, 'io')):
        def broken(arguments, timeout, failure=failure):
            raise failure
        outcome = execute(tmp_path, make_worker_request(), process=broken)
        assert outcome.error == 'media_probe_failed'


def test_probe_disk_full_is_storage_guard_tripped(tmp_path):
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input' / 'input.bin').write_bytes(b'x')

    def full(arguments, timeout):
        raise OSError(errno.ENOSPC, 'full')
    assert execute(tmp_path, make_worker_request(), process=full).error == 'storage_guard_tripped'


def test_duration_boundary_is_inclusive_at_the_limit(tmp_path):
    outcome, _ = probe_outcome(tmp_path, {'streams': [AUDIO], 'format': {'duration': '21600.000000'}})
    assert outcome.code == 0 and read_meta(outcome)['duration_seconds'] == 21600.0
    outcome, _ = probe_outcome(tmp_path / 'b', {'streams': [AUDIO], 'format': {'duration': 21600}})
    assert outcome.code == 0 and read_meta(outcome)['duration_seconds'] == 21600.0


@pytest.mark.parametrize('duration', ['21600.001', 21600.001, '99999'])
def test_duration_over_limit_is_exceeded(tmp_path, duration):
    outcome, _ = probe_outcome(tmp_path, {'streams': [AUDIO], 'format': {'duration': duration}})
    assert outcome.code == 1 and outcome.error == 'duration_exceeded' and outcome.files == []


def test_duration_limit_comes_from_request_limits(tmp_path):
    limits = replace(LIMITS, duration_seconds=100)
    ok, _ = probe_outcome(tmp_path / 'a', {'streams': [AUDIO], 'format': {'duration': '100'}}, limits=limits)
    bad, _ = probe_outcome(tmp_path / 'b', {'streams': [AUDIO], 'format': {'duration': '100.5'}}, limits=limits)
    assert ok.code == 0 and bad.error == 'duration_exceeded'


@pytest.mark.parametrize('format_', [{}, {'duration': 'N/A'}, {'duration': None}])
def test_unknown_duration_is_null_not_failure(tmp_path, format_):
    outcome, _ = probe_outcome(tmp_path, {'streams': [AUDIO], 'format': format_})
    assert outcome.code == 0
    assert read_meta(outcome) == record(duration_seconds=None, has_audio=True, has_video=False,
        audio_streams=1, video_streams=0)


@pytest.mark.parametrize('raw', [
    b'{"streams":[{"codec_type":"audio"}],"format":{"duration":NaN}}',
    b'{"streams":[{"codec_type":"audio"}],"format":{"duration":Infinity}}',
    b'{"streams":[{"codec_type":"audio"}],"format":{"duration":"nan"}}',
    b'{"streams":[{"codec_type":"audio"}],"format":{"duration":"inf"}}',
    b'{"streams":[{"codec_type":"audio"}],"format":{"duration":"-1"}}',
    b'{"streams":[{"codec_type":"audio"}],"format":{"duration":-0.5}}',
    b'{"streams":[{"codec_type":"audio"}],"format":{"duration":"abc"}}',
    b'{"streams":[{"codec_type":"audio"}],"format":{"size":"-4"}}',
])
def test_non_finite_or_negative_probe_numbers_fail(tmp_path, raw):
    outcome, _ = probe_outcome(tmp_path, raw=raw)
    assert outcome.code == 1 and outcome.error == 'media_probe_failed'


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_before_asr_downloads_nothing_and_leaks_nothing(tmp_path):
    ydl = fake_ydl(merged_info(requested_formats=[
        {'format_id': '137', 'vcodec': 'avc1', 'acodec': 'none', 'height': 1080, 'filesize': 700},
        {'format_id': '140', 'vcodec': 'none', 'acodec': 'mp4a', 'filesize_approx': 300}]))
    outcome = execute(tmp_path, remote_request('metadata', video=True), ydl=ydl)
    assert outcome.code == 0 and outcome.events == ['started', 'artifact', 'completed']
    assert [(url, download) for url, download, _ in ydl.calls] == [(VIDEO_ID, False)]
    assert outcome.files == [META]  # only probe_metadata; temp dir removed; no model or media files
    assert read_meta(outcome) == record(operation='metadata', source_kind='youtube', duration_seconds=120.5,
        has_audio=True, has_video=True, video_height=1080, audio_streams=1, video_streams=1,
        estimated_bytes=1000)
    assert list(read_meta(outcome)) == PROBE_KEYS
    raw = (outcome.scratch / META).read_bytes()
    for leaked in (SECRET_TITLE, 'Someone Private', 'private words', 'youtube.com', VIDEO_ID, 'http'):
        assert leaked.encode() not in raw
    for forbidden_key in ('title', 'url', 'uploader', 'description', 'webpage_url'):
        assert forbidden_key not in read_meta(outcome)
    assert dict(outcome.collector.of('artifact')[0].payload)['slot'] == 'probe_metadata'
    assert not (tmp_path / 'scratch' / '.media-tmp').exists()


def test_metadata_for_transcript_uses_audio_format_and_records_no_video(tmp_path):
    info = {'id': VIDEO_ID, 'duration': 30, 'format_id': '140', 'vcodec': 'none', 'acodec': 'mp4a',
            'filesize': 55}
    ydl = fake_ydl(info)
    outcome = execute(tmp_path, remote_request('metadata'), ydl=ydl)
    assert ydl.calls[0][2]['format'] == 'ba/b'
    assert read_meta(outcome) == record(operation='metadata', source_kind='youtube', duration_seconds=30.0,
        has_audio=True, has_video=False, audio_streams=1, video_streams=0, estimated_bytes=55)


def test_metadata_direct_performs_no_io_and_everything_is_null(tmp_path):
    request = remote_request('metadata', source=DIRECT)
    outcome = execute(tmp_path, request, ydl=forbidden, process=forbidden, direct=forbidden_direct)
    assert outcome.code == 0 and outcome.events == ['started', 'artifact', 'completed']
    assert read_meta(outcome) == record(operation='metadata', source_kind='direct')
    assert list(read_meta(outcome)) == PROBE_KEYS


def test_metadata_duration_limit_and_non_finite_duration(tmp_path):
    for duration, expected in ((21600, 21600.0), (21600.0, 21600.0)):
        outcome = execute(tmp_path / str(duration), remote_request('metadata', video=True),
                          ydl=fake_ydl(merged_info(duration=duration)))
        assert read_meta(outcome)['duration_seconds'] == expected
    over = execute(tmp_path / 'over', remote_request('metadata', video=True),
                   ydl=fake_ydl(merged_info(duration=21600.001)))
    assert over.error == 'duration_exceeded' and over.files == []
    for index, bad in enumerate((float('nan'), float('inf'), -5, None, 'x', True)):
        outcome = execute(tmp_path / f'bad{index}', remote_request('metadata', video=True),
                          ydl=fake_ydl(merged_info(duration=bad)))
        assert outcome.code == 0 and read_meta(outcome)['duration_seconds'] is None


@pytest.mark.parametrize('changes', [
    {'_type': 'playlist', 'entries': [{}]}, {'entries': []}, {'_type': 'multi_video'},
    {'is_live': True}, {'was_live': True}, {'live_status': 'is_upcoming'}, {'live_status': 'post_live'},
])
@pytest.mark.parametrize('operation', ['metadata', 'download'])
def test_playlist_and_live_results_are_rejected(tmp_path, operation, changes):
    ydl = fake_ydl(merged_info(**changes))
    outcome = execute(tmp_path, remote_request(operation, video=True), ydl=ydl, process=forbidden)
    assert outcome.code == 1 and outcome.error == 'input_rejected'
    assert [download for _, download, _ in ydl.calls] == [False] and outcome.files == []


def test_id_mismatch_in_result_is_source_changed(tmp_path):
    outcome = execute(tmp_path, remote_request('metadata'), ydl=fake_ydl(merged_info(id='ZZZZZZZZZZZ')))
    assert outcome.error == 'source_changed'


@pytest.mark.parametrize('bad_input', [
    'https://www.youtube.com/watch?v=short',
    'https://www.youtube.com/watch?v=abcDEF_-123&list=PL12345',
    'https://www.youtube.com/watch?v=abcDEF_-123&t=5',
    'https://www.youtube.com/watch?v=abcDEF_-1234',
    'https://evil.example.com/watch?v=abcDEF_-123',
    'https://www.youtube.com.evil.example/watch?v=abcDEF_-123',
    'http://www.youtube.com/watch?v=abcDEF_-123',
    'https://youtu.be/abcDEF_-123',
    'https://www.youtube.com/watch?v=abcDEF_-123\n',
    'https://www.youtube.com/watch?v=abcDEF_-12!',
    'abcDEF_-123', '', None,
])
@pytest.mark.parametrize('operation', ['metadata', 'download'])
def test_non_canonical_youtube_source_is_rejected_before_any_io(tmp_path, operation, bad_input):
    request = remote_request(operation)
    object.__setattr__(request, 'source', {'kind': 'youtube', 'input': bad_input})  # defence in depth
    outcome = execute(tmp_path, request, ydl=forbidden, process=forbidden, direct=forbidden_direct)
    assert outcome.code == 1 and outcome.events == ['started', 'error'] and outcome.error == 'input_rejected'
    assert outcome.files == []


def test_unknown_source_kind_is_rejected_before_any_io(tmp_path):
    request = remote_request('metadata')
    object.__setattr__(request, 'source', {'kind': 'vimeo', 'input': 'https://vimeo.com/1'})
    outcome = execute(tmp_path, request, ydl=forbidden, process=forbidden, direct=forbidden_direct)
    assert outcome.error == 'input_rejected'


def test_host_request_layer_already_refuses_playlist_urls():
    with pytest.raises(DomainValidationError):
        remote_request('metadata', source={'kind': 'youtube',
            'input': YOUTUBE['input'] + '&list=PL1'})


# ---------------------------------------------------------------------------
# Envelope rejections (before any ydl / ffprobe / network call)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(('operation', 'video', 'slots'), [
    ('download', False, {'downloaded_audio': 'a.bin', 'merged_video': 'm.bin'}),
    ('download', False, {'merged_video': 'm.bin'}),
    ('download', True, {'merged_video': 'm.bin'}),
    ('download', True, {'downloaded_video': 'v.bin', 'downloaded_audio': 'a.bin'}),
    ('download', True, {'downloaded_media': 'x.bin'}),
    ('download', False, {'downloaded_media': 'x.bin'}),
    ('metadata', False, {'downloaded_audio': 'a.bin'}),
    ('metadata', False, {'probe_metadata': 'm.json', 'downloaded_audio': 'a.bin'}),
    ('metadata', True, {'transcript_segments': 't.bin'}),
])
def test_wrong_slot_set_is_rejected_before_any_io(tmp_path, operation, video, slots):
    request = remote_request(operation, video=video, slots=slots)
    outcome = execute(tmp_path, request, ydl=forbidden, process=forbidden, direct=forbidden_direct)
    assert outcome.code == 1 and outcome.events == ['started', 'error']
    assert outcome.error == 'worker_protocol_mismatch' and outcome.files == []


def test_direct_download_slot_set_is_exact(tmp_path):
    request = remote_request('download', source=DIRECT,
                             slots={'downloaded_media': 'x.bin', 'probe_metadata': 'a.json'})
    assert execute(tmp_path, request, direct=forbidden_direct).error == 'worker_protocol_mismatch'


@pytest.mark.parametrize('slots', [{'downloaded_media': 'x.bin'},
                                   {'probe_metadata': 'm.json', 'downloaded_media': 'x.bin'}])
def test_probe_slot_set_is_exact(tmp_path, slots):
    outcome = execute(tmp_path, make_worker_request(assigned_slots=slots), process=forbidden)
    assert outcome.error == 'worker_protocol_mismatch' and outcome.events == ['started', 'error']


def test_wrong_capability_and_operation_are_rejected(tmp_path):
    for capability in ('asr_cpu', 'asr_cuda'):
        request = make_worker_request(capability=capability)
        outcome = execute(tmp_path / capability, request, ydl=forbidden, process=forbidden)
        assert outcome.error == 'worker_protocol_mismatch' and outcome.events == ['started', 'error']
    transcribe = make_worker_request(operation='transcribe')
    assert execute(tmp_path / 't', transcribe, process=forbidden).error == 'worker_protocol_mismatch'


# ---------------------------------------------------------------------------
# Download output modes
# ---------------------------------------------------------------------------

def test_transcript_only_downloads_best_audio_to_the_audio_slot(tmp_path):
    ydl = fake_ydl(merged_info(requested_formats=None, format_id='140', vcodec='none', acodec='mp4a'),
                   {'140': (b'AUDIO-BYTES', [4, 11])})
    outcome = execute(tmp_path, remote_request('download'), ydl=ydl, process=forbidden)
    assert outcome.code == 0 and outcome.events == ['started', 'artifact', 'completed']
    assert [(c[0], c[1], c[2]['format']) for c in ydl.calls] == [(VIDEO_ID, False, 'ba/b')]  # one extraction
    assert [(name, info['format_id'], 'requested_formats' in info) for name, info, _ in ydl.dl_calls] == [
        (str(outcome.scratch / 'audio.bin'), '140', False)]
    assert (outcome.scratch / 'audio.bin').read_bytes() == b'AUDIO-BYTES'
    assert dict(outcome.collector.of('artifact')[0].payload) == {
        'slot': 'downloaded_audio', 'bytes': 11, 'sha256': hashlib.sha256(b'AUDIO-BYTES').hexdigest()}
    assert outcome.files == ['audio.bin']


@pytest.mark.parametrize(('transcript', 'quality', 'selector'), [
    (True, '1080p', 'bv*[height<=1080]+ba/b[height<=1080]'),
    (False, '1080p', 'bv*[height<=1080]+ba/b[height<=1080]'),
    (True, 'source_best', 'bv*+ba/b'),
    (False, 'source_best', 'bv*+ba/b'),
])
def test_video_modes_merge_into_a_single_artifact(tmp_path, transcript, quality, selector):
    request = make_worker_request(operation='download', source=dict(YOUTUBE), input_basename=None,
        assigned_slots={'downloaded_video': 'video.bin', 'downloaded_audio': 'audio.bin', 'merged_video': 'merged.bin'},
        outputs=dict(transcript=transcript, video=True, video_quality=quality))
    assert request.outputs.mode.value == ('video_transcript' if transcript else 'video')
    ydl = fake_ydl(merged_info(), {'137': (b'VIDEO', [5]), '140': (b'AUDIO', [5])})
    process = fake_ffprobe(merge_bytes=b'MERGED-OUT')
    outcome = execute(tmp_path, request, ydl=ydl, process=process)
    assert outcome.code == 0 and outcome.events == ['started', 'artifact', 'completed']
    assert [(c[0], c[1], c[2]['format']) for c in ydl.calls] == [(VIDEO_ID, False, selector)]  # one extraction
    assert [(name, info['format_id'], info['vcodec'], info['acodec'], 'requested_formats' in info)
            for name, info, _ in ydl.dl_calls] == [
        (str(outcome.scratch / 'video.bin'), '137', 'avc1.640028', 'none', False),
        (str(outcome.scratch / 'audio.bin'), '140', 'none', 'mp4a.40.2', False)]
    assert all(info['id'] == VIDEO_ID and info['duration'] == 120.5 for _, info, _ in ydl.dl_calls)
    assert len(ydl.instances) == 1  # extraction and both downloads share one YoutubeDL instance
    assert process.calls == [(('ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
        '-protocol_whitelist', 'file', '-i', str(outcome.scratch / 'video.bin'),
        '-protocol_whitelist', 'file', '-i', str(outcome.scratch / 'audio.bin'),
        '-map', '0:v:0', '-map', '1:a:0', '-c', 'copy', '-fs', str(LIMITS.media_bytes + 1), '-f', 'matroska',
        str(outcome.scratch / 'merged.bin')), float(LIMITS.decode_timeout_seconds))]
    assert outcome.files == ['merged.bin']  # the two source streams are deleted after the merge
    assert dict(outcome.collector.of('artifact')[0].payload) == {
        'slot': 'merged_video', 'bytes': 10, 'sha256': hashlib.sha256(b'MERGED-OUT').hexdigest()}


def test_progressive_format_is_written_to_the_video_slot_and_remuxed(tmp_path):
    info = {'id': VIDEO_ID, 'duration': 9, 'format_id': '18', 'vcodec': 'avc1', 'acodec': 'mp4a',
            'height': 360, 'filesize': 40}
    selector = 'bv*[height<=1080]+ba/b[height<=1080]'
    ydl = fake_ydl(info, {'18': (b'PROGRESSIVE', [11])})
    process = fake_ffprobe(merge_bytes=b'REMUXED')
    outcome = execute(tmp_path, remote_request('download', video=True), ydl=ydl, process=process)
    assert outcome.code == 0 and outcome.files == ['merged.bin']
    assert [(c[1], c[2]['format']) for c in ydl.calls] == [(False, selector)]
    assert [(name, info['format_id']) for name, info, _ in ydl.dl_calls] == [
        (str(outcome.scratch / 'video.bin'), '18')]
    assert process.calls == [(('ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
        '-protocol_whitelist', 'file', '-i', str(outcome.scratch / 'video.bin'),
        '-map', '0:v:0', '-map', '0:a:0?', '-c', 'copy', '-fs', str(LIMITS.media_bytes + 1), '-f', 'matroska',
        str(outcome.scratch / 'merged.bin')), float(LIMITS.decode_timeout_seconds))]
    assert [e.payload['slot'] for e in outcome.collector.of('artifact')] == ['merged_video']


def test_unsafe_format_ids_are_rejected(tmp_path):
    info = merged_info()
    info['requested_formats'][0]['format_id'] = '137]+ba/b['
    ydl = fake_ydl(info)
    outcome = execute(tmp_path, remote_request('download', video=True), ydl=ydl, process=forbidden)
    assert outcome.error == 'input_rejected' and len(ydl.calls) == 1 and ydl.dl_calls == []


def test_no_format_satisfying_the_cap_is_input_rejected(tmp_path):
    class NoFormat(Exception):
        __module__ = 'yt_dlp.utils'

    def raising(options):
        raise NoFormat('Requested format is not available. Use --list-formats')

    outcome = execute(tmp_path, remote_request('metadata', video=True), ydl=raising)
    assert outcome.code == 1 and outcome.error == 'input_rejected'
    assert 'Requested format' not in b''.join(outcome.collector.chunks).decode()


@pytest.mark.parametrize(('message', 'reason'), [
    ('Sign in to confirm your age', 'content_denied'),
    ('Private video', 'content_denied'),
    ('[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed', 'upstream_unavailable'),
    ('Unable to download webpage: timed out', 'upstream_unavailable'),
    ('something else entirely', 'upstream_unavailable'),
])
def test_yt_dlp_failures_map_to_closed_reasons_without_text(tmp_path, message, reason):
    class DownloadFailure(Exception):
        __module__ = 'yt_dlp.utils'

    def raising(options):
        raise DownloadFailure(message + ' https://secret.example/x?token=1')

    outcome = execute(tmp_path, remote_request('metadata'), ydl=raising)
    assert outcome.error == reason
    wire = b''.join(outcome.collector.chunks).decode()
    assert 'secret.example' not in wire and message not in wire


# ---------------------------------------------------------------------------
# Byte caps
# ---------------------------------------------------------------------------

def capped(media_bytes):
    return replace(LIMITS, media_bytes=media_bytes)


def test_bytes_are_summed_across_streams_and_overlap_is_merged(tmp_path):
    # Stream one finishes at 600 bytes; stream two climbs to 500: 600 + 500 > 1000.
    ydl = fake_ydl(merged_info(), {'137': (b'v' * 600, [300, 300, 600]),
                                    '140': (b'a' * 500, [300, 500])})
    request = remote_request('download', video=True, limits=capped(1000))
    outcome = execute(tmp_path, request, ydl=ydl, process=forbidden)
    assert outcome.code == 1 and outcome.error == 'media_limit_reached' and outcome.files == []
    assert 'progress' not in outcome.events  # no estimate: progress is omitted


def test_repeated_reports_for_one_file_do_not_double_count(tmp_path):
    ydl = fake_ydl(merged_info(), {'137': (b'v' * 400, [100, 400, 400, 400]),
                                    '140': (b'a' * 500, [250, 500])})
    request = remote_request('download', video=True, limits=capped(1000))
    process = fake_ffprobe(merge_bytes=b'M' * 10)
    outcome = execute(tmp_path, request, ydl=ydl, process=process)
    assert outcome.code == 0 and outcome.files == ['merged.bin']


def test_estimate_over_cap_is_rejected_before_downloading(tmp_path):
    ydl = fake_ydl(sized(merged_info(), video=900, audio=200), {})
    request = remote_request('download', video=True, limits=capped(1000))
    outcome = execute(tmp_path, request, ydl=ydl, process=forbidden)
    assert outcome.error == 'media_limit_reached' and outcome.events == ['started', 'error']
    assert [download for _, download, _ in ydl.calls] == [False] and outcome.files == []


def test_estimate_exactly_at_cap_is_allowed(tmp_path):
    ydl = fake_ydl(sized(merged_info(), video=800, audio=200),
                   {'137': (b'v' * 800, [800]), '140': (b'a' * 200, [200])})
    request = remote_request('download', video=True, limits=capped(1000))
    outcome = execute(tmp_path, request, ydl=ydl, process=fake_ffprobe(merge_bytes=b'M' * 900))
    assert outcome.code == 0


def test_merge_output_over_cap_is_deleted(tmp_path):
    ydl = fake_ydl(merged_info(), {'137': (b'v' * 300, [300]), '140': (b'a' * 300, [300])})
    request = remote_request('download', video=True, limits=capped(1000))
    process = fake_ffprobe(merge_bytes=b'M' * 1001)
    outcome = execute(tmp_path, request, ydl=ydl, process=process)
    assert outcome.code == 1 and outcome.error == 'media_limit_reached'
    assert not (outcome.scratch / 'merged.bin').exists() and outcome.files == []
    assert process.calls[0][0][process.calls[0][0].index('-fs') + 1] == '1001'


def test_merge_cap_argument_is_limit_plus_one(tmp_path):
    ydl = fake_ydl(merged_info(), {'137': (b'v' * 300, [300]), '140': (b'a' * 300, [300])})
    process = fake_ffprobe(merge_bytes=b'M' * 1000)
    execute(tmp_path, remote_request('download', video=True, limits=capped(1000)), ydl=ydl, process=process)
    arguments = process.calls[0][0]
    assert arguments[arguments.index('-fs') + 1] == '1001'


def test_merge_output_exactly_at_cap_is_kept(tmp_path):
    ydl = fake_ydl(merged_info(), {'137': (b'v' * 300, [300]), '140': (b'a' * 300, [300])})
    request = remote_request('download', video=True, limits=capped(1000))
    outcome = execute(tmp_path, request, ydl=ydl, process=fake_ffprobe(merge_bytes=b'M' * 1000))
    assert outcome.code == 0 and outcome.files == ['merged.bin']


def test_download_duration_over_limit_is_exceeded_before_any_dl(tmp_path):
    ydl = fake_ydl(merged_info(duration=21600.001), {'137': (b'v', [1]), '140': (b'a', [1])})
    outcome = execute(tmp_path, remote_request('download', video=True), ydl=ydl, process=forbidden)
    assert outcome.error == 'duration_exceeded' and outcome.files == []
    assert ydl.dl_calls == [] and len(ydl.calls) == 1


def test_download_negative_duration_is_unknown_not_failure(tmp_path):
    ydl = fake_ydl(merged_info(duration=-5), {'137': (b'v', [1]), '140': (b'a', [1])})
    assert execute(tmp_path, remote_request('download', video=True), ydl=ydl,
                   process=fake_ffprobe()).code == 0


@pytest.mark.parametrize('operation', ['download', 'metadata'])
def test_extraction_past_the_operation_deadline_is_download_timeout(tmp_path, operation):
    clock = Clock()
    limits = replace(LIMITS, download_timeout_seconds=10)

    def slow_extract(options):
        clock.now += 11.0
        return merged_info()
    ydl = fake_ydl(slow_extract, {'137': (b'v', [1]), '140': (b'a', [1])})
    outcome = execute(tmp_path, remote_request(operation, video=True, limits=limits), ydl=ydl,
                      process=forbidden, clock=clock)
    assert outcome.error == 'download_timeout' and outcome.files == []
    assert ydl.dl_calls == []


def test_extraction_within_the_deadline_still_downloads(tmp_path):
    clock = Clock()
    limits = replace(LIMITS, download_timeout_seconds=10)

    def quick_extract(options):
        clock.now += 10.0
        return merged_info()
    ydl = fake_ydl(quick_extract, {'137': (b'v', [1]), '140': (b'a', [1])})
    outcome = execute(tmp_path, remote_request('download', video=True, limits=limits), ydl=ydl,
                      process=fake_ffprobe(), clock=clock)
    assert outcome.code == 0


def test_failed_merge_cleans_everything_up(tmp_path):
    ydl = fake_ydl(merged_info(), {'137': (b'v', [1]), '140': (b'a', [1])})
    outcome = execute(tmp_path, remote_request('download', video=True), ydl=ydl,
                      process=fake_ffprobe(merge_returncode=1))
    assert outcome.code == 1 and outcome.error == 'media_probe_failed' and outcome.files == []


def test_stream_skipped_by_max_filesize_is_a_limit_failure(tmp_path):
    info = merged_info(requested_formats=None, format_id='140', vcodec='none', acodec='mp4a')
    ydl = fake_ydl(info, {'140': (None, [])})  # yt-dlp returns without writing the stream
    outcome = execute(tmp_path, remote_request('download'), ydl=ydl, process=forbidden)
    assert outcome.code == 1 and outcome.error == 'media_limit_reached' and outcome.files == []


def test_download_overall_timeout_is_enforced_from_progress(tmp_path):
    clock = Clock()
    limits = replace(LIMITS, download_timeout_seconds=10)
    ydl = fake_ydl(merged_info(), {'137': (b'v', [1, 2, 3])}, clock=clock, step=6.0)
    outcome = execute(tmp_path, remote_request('download', video=True, limits=limits), ydl=ydl,
                      process=forbidden, clock=clock)
    assert outcome.error == 'download_timeout' and outcome.files == []


def test_disk_full_in_yt_dlp_is_storage_guard_tripped(tmp_path):
    class Wrapped(Exception):
        __module__ = 'yt_dlp.utils'

    try:
        try:
            raise OSError(errno.ENOSPC, 'No space left on device')
        except OSError as inner:
            raise Wrapped('unable to write') from inner
    except Wrapped as wrapped:
        failure = wrapped
    ydl = fake_ydl(merged_info(requested_formats=None, format_id='140', vcodec='none', acodec='mp4a'),
                   error=failure)
    outcome = execute(tmp_path, remote_request('download'), ydl=ydl, process=forbidden)
    assert outcome.error == 'storage_guard_tripped'


def test_progress_is_estimate_based_and_at_most_once_per_second(tmp_path):
    clock = Clock(0.0)
    info = sized(merged_info(), video=1000, audio=0)
    counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ydl = fake_ydl(info, {'137': (b'v' * 1000, counts), '140': (b'a', [1])},
                   clock=clock, step=0.25)
    process = fake_ffprobe(merge_bytes=b'M')
    outcome = execute(tmp_path, remote_request('download', video=True), ydl=ydl, process=process, clock=clock)
    assert outcome.code == 0
    progress = [(t, r.payload['percent']) for t, r in zip(outcome.collector.times, outcome.collector.records)
                if r.event == 'progress']
    assert progress[0][1] == 10.0
    assert all(b[0] - a[0] >= 1.0 for a, b in itertools.pairwise(progress))
    assert [p for _, p in progress] == [10.0, 50.0, 90.0]
    assert all(0 <= p <= 100 for _, p in progress)


# ---------------------------------------------------------------------------
# yt-dlp hardening
# ---------------------------------------------------------------------------

EXPECTED_OPTIONS = {
    'quiet': True, 'no_warnings': True, 'noprogress': True, 'noplaylist': True, 'cookiefile': None,
    'cachedir': False, 'allowed_extractors': ['^youtube$'], 'socket_timeout': 30,
    'retries': LIMITS.download_attempts, 'fragment_retries': LIMITS.download_attempts,
    'extractor_retries': LIMITS.download_attempts, 'max_filesize': LIMITS.media_bytes,
    'js_runtimes': {'deno': {'path': DENO}}, 'remote_components': [], 'nocheckcertificate': False,
    'proxy': '', 'format': 'bv*[height<=1080]+ba/b[height<=1080]', 'fixup': 'never', 'updatetime': False,
    'ignoreerrors': False, 'overwrites': False, 'external_downloader': None,
    'external_downloader_args': None,
}


def test_exact_yt_dlp_options_and_only_the_id_reaches_extract_info(tmp_path):
    ydl = fake_ydl(merged_info(), {'137': (b'v', [1]), '140': (b'a', [1])})
    outcome = execute(tmp_path, remote_request('download', video=True), ydl=ydl, process=fake_ffprobe())
    assert outcome.code == 0
    assert [url for url, _, _ in ydl.calls] == [VIDEO_ID]  # exactly one extraction, id only
    assert len(ydl.dl_calls) == 2
    options = ydl.instances[0].options
    logger = options['logger']
    assert logger.debug('x') is logger.info('x') is logger.warning('x') is logger.error('x') is None
    assert {k: v for k, v in options.items() if k not in ('logger', 'progress_hooks')} == EXPECTED_OPTIONS
    for absent in ('cookiesfrombrowser', 'exec_cmd', 'postprocessors', 'postprocessor_args', 'outtmpl',
                   'plugin_dirs', 'username', 'password', 'netrc', 'proxy_url', 'ffmpeg_location',
                   'playlist_items', 'extract_flat', 'live_from_start', 'enable_file_urls', 'download_archive'):
        assert absent not in options
    assert options['cookiefile'] is None and options['cachedir'] is False
    assert options['noplaylist'] is True and options['quiet'] is True and options['noprogress'] is True
    assert options['external_downloader'] is None and options['external_downloader_args'] is None
    assert options['nocheckcertificate'] is False and options['remote_components'] == []
    assert options['allowed_extractors'] == ['^youtube$'] and options['proxy'] == ''
    assert options['max_filesize'] == LIMITS.media_bytes and len(options['progress_hooks']) == 1
    assert [name for name, _, opts in ydl.dl_calls if opts is options] == [
        str(outcome.scratch / 'video.bin'), str(outcome.scratch / 'audio.bin')]


def test_download_never_re_extracts_even_when_a_second_extraction_would_differ(tmp_path):
    first = merged_info()
    later = merged_info(requested_formats=[{'format_id': '999', 'vcodec': 'avc1', 'acodec': 'none'}])
    answers = iter([first, later])
    ydl = fake_ydl(lambda _options: next(answers), {'137': (b'v', [1]), '140': (b'a', [1])})
    outcome = execute(tmp_path, remote_request('download', video=True), ydl=ydl, process=fake_ffprobe())
    assert outcome.code == 0 and len(ydl.calls) == 1 and len(ydl.instances) == 1
    assert [info['format_id'] for _, info, _ in ydl.dl_calls] == ['137', '140']


def test_video_download_without_a_video_stream_is_input_rejected(tmp_path):
    info = merged_info(requested_formats=None, format_id='140', vcodec='none', acodec='mp4a')
    ydl = fake_ydl(info, {'140': (b'a', [1])})
    outcome = execute(tmp_path, remote_request('download', video=True), ydl=ydl, process=forbidden)
    assert outcome.error == 'input_rejected' and ydl.dl_calls == [] and outcome.files == []


def test_video_pair_without_an_audio_stream_is_input_rejected(tmp_path):
    info = merged_info()
    info['requested_formats'][1]['acodec'] = 'none'
    ydl = fake_ydl(info, {})
    outcome = execute(tmp_path, remote_request('download', video=True), ydl=ydl, process=forbidden)
    assert outcome.error == 'input_rejected' and ydl.dl_calls == []


def test_yt_dlp_retries_follow_request_limits(tmp_path):
    limits = replace(LIMITS, download_attempts=1)
    ydl = fake_ydl(merged_info())
    execute(tmp_path, remote_request('metadata', video=True, limits=limits), ydl=ydl)
    options = ydl.calls[0][2]
    assert options['retries'] == options['fragment_retries'] == options['extractor_retries'] == 1


def test_temp_dir_is_removed_even_on_failure(tmp_path):
    ydl = fake_ydl(merged_info(is_live=True))
    outcome = execute(tmp_path, remote_request('metadata'), ydl=ydl)
    assert outcome.error == 'input_rejected' and outcome.files == []


def test_production_environment_stays_inside_the_scratch_temp_dir(tmp_path, monkeypatch):
    for name in ('HOME', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'DENO_DIR', 'TMPDIR', 'DENO_NO_UPDATE_CHECK',
                 'HTTP_PROXY', 'https_proxy'):
        monkeypatch.setenv(name, 'sentinel')
    monkeypatch.setattr(media_worker.tempfile, 'tempdir', None)
    tmp = tmp_path / '.media-tmp'
    tmp.mkdir()
    media_worker._prepare_environment(tmp)
    for name in ('HOME', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'DENO_DIR', 'TMPDIR'):
        assert Path(os.environ[name]).is_relative_to(tmp) or Path(os.environ[name]) == tmp
    assert 'HTTP_PROXY' not in os.environ and 'https_proxy' not in os.environ
    assert os.environ['DENO_NO_UPDATE_CHECK'] == '1'


# ---------------------------------------------------------------------------
# Direct download
# ---------------------------------------------------------------------------

class FakeDirect:
    def __init__(self, data=b'DIRECT-BYTES', error=None):
        self.data, self.error, self.calls = data, error, []

    async def __call__(self, url, output, policy, *, free_space, **kwargs):
        self.calls.append((url, policy, free_space(), kwargs, output))
        assert type(output) is io.FileIO and output.writable() and output.mode == 'rb+'
        output.write(self.data[:3])
        if self.error is not None:
            raise self.error
        output.write(self.data[3:])
        return DownloadResult(len(self.data), 'audio/mpeg', 0, 1)


def test_direct_download_delegates_with_policy_derived_from_limits(tmp_path):
    limits = replace(LIMITS, media_bytes=4096, download_timeout_seconds=100, download_attempts=2)
    direct = FakeDirect()
    outcome = execute(tmp_path, remote_request('download', source=DIRECT, limits=limits),
                      direct=direct, free_space=lambda: 777)
    assert outcome.code == 0 and outcome.events == ['started', 'artifact', 'completed']
    (url, policy, free, kwargs, output), = direct.calls
    assert url == DIRECT['input'] and free == 777 and kwargs == {} and output.closed
    assert policy == DownloadPolicy(byte_limit=4096, admitted_bytes=4096, overall_timeout_seconds=100,
                                    read_timeout_seconds=60, attempts=2)
    assert (outcome.scratch / 'media.bin').read_bytes() == b'DIRECT-BYTES'
    assert dict(outcome.collector.of('artifact')[0].payload) == {
        'slot': 'downloaded_media', 'bytes': 12, 'sha256': hashlib.sha256(b'DIRECT-BYTES').hexdigest()}
    assert outcome.files == ['media.bin']


def test_direct_read_timeout_never_exceeds_the_overall_timeout(tmp_path):
    limits = replace(LIMITS, download_timeout_seconds=30)
    direct = FakeDirect()
    execute(tmp_path, remote_request('download', source=DIRECT, limits=limits), direct=direct)
    assert direct.calls[0][1].read_timeout_seconds == 30


@pytest.mark.parametrize('reason', [ReasonCode.CONTENT_DENIED, ReasonCode.MEDIA_LIMIT_REACHED,
                                    ReasonCode.DOWNLOAD_TIMEOUT, ReasonCode.UPSTREAM_UNAVAILABLE,
                                    ReasonCode.STORAGE_GUARD_TRIPPED])
def test_direct_download_errors_map_straight_to_the_error_event(tmp_path, reason):
    direct = FakeDirect(error=DownloadError(reason))
    outcome = execute(tmp_path, remote_request('download', source=DIRECT), direct=direct)
    assert outcome.code == 1 and outcome.events == ['started', 'error']
    error = outcome.collector.of('error')[0].payload
    assert error['reason_code'] == reason.value and error['summary'] == reason.name.replace('_', ' ').capitalize()
    assert outcome.files == []  # the partial file is removed


def test_direct_disk_full_oserror_is_storage_guard_tripped(tmp_path):
    outcome = execute(tmp_path, remote_request('download', source=DIRECT),
                      direct=FakeDirect(error=OSError(errno.ENOSPC, 'full')))
    assert outcome.error == 'storage_guard_tripped'


def test_direct_refuses_to_overwrite_an_existing_file(tmp_path):
    (tmp_path / 'scratch').mkdir()
    (tmp_path / 'scratch' / 'media.bin').write_bytes(b'host-owned')
    direct = FakeDirect()
    outcome = execute(tmp_path, remote_request('download', source=DIRECT), direct=direct)
    assert outcome.error == 'destination_conflict' and direct.calls == []


def test_direct_size_mismatch_is_artifact_rejected(tmp_path):
    class Liar(FakeDirect):
        async def __call__(self, url, output, policy, *, free_space, **kwargs):
            output.write(b'abc')
            return DownloadResult(99, 'audio/mpeg', 0, 1)
    outcome = execute(tmp_path, remote_request('download', source=DIRECT), direct=Liar())
    assert outcome.error == 'artifact_rejected'


def test_unexpected_exceptions_become_a_fixed_generic_error(tmp_path):
    class Boom(FakeDirect):
        async def __call__(self, *args, **kwargs):
            raise RuntimeError('C:\\secret\\path https://leak.example/?x=1')
    outcome = execute(tmp_path, remote_request('download', source=DIRECT), direct=Boom())
    assert outcome.error == 'worker_exit_unexpected' and outcome.code == 1
    assert b'secret' not in b''.join(outcome.collector.chunks)


def test_existing_destination_for_metadata_is_a_conflict(tmp_path):
    (tmp_path / 'scratch').mkdir()
    (tmp_path / 'scratch' / META).write_bytes(b'old')
    outcome = execute(tmp_path, remote_request('metadata', source=DIRECT))
    assert outcome.error == 'destination_conflict'


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def test_worker_writes_only_events_to_the_sink_and_nothing_to_stdout(tmp_path, capsys):
    outcome, _ = probe_outcome(tmp_path, {'streams': [AUDIO], 'format': {'duration': '1'}})
    captured = capsys.readouterr()
    assert captured.out == '' and captured.err == '' and outcome.code == 0


def test_main_rejects_garbage_without_writing_events():
    for line in (b'', b'not json\n', b'{}\n', b'x' * 70000):
        out = io.BytesIO()
        assert media_worker.main([], stdin=io.BytesIO(line), stdout=out) == 2
        assert out.getvalue() == b''


def test_main_delegates_a_valid_request_and_returns_the_exit_code(monkeypatch):
    request = make_worker_request()
    seen = {}

    def fake_run(req, sink):
        seen['request'] = req
        sink(b'line\n')
        return 7

    monkeypatch.setattr(media_worker, 'run', fake_run)
    out = io.BytesIO()
    assert media_worker.main(stdin=io.BytesIO(request.to_json() + b'\n'), stdout=out) == 7
    assert seen['request'] == request and out.getvalue() == b'line\n'


def test_fixed_error_summaries_are_host_acceptable_for_every_reason():
    request = make_worker_request()
    for reason in (ReasonCode.INPUT_REJECTED, ReasonCode.MEDIA_LIMIT_REACHED, ReasonCode.DURATION_EXCEEDED,
                   ReasonCode.WORKER_PROTOCOL_MISMATCH, ReasonCode.STORAGE_GUARD_TRIPPED,
                   ReasonCode.MEDIA_PROBE_FAILED, ReasonCode.CONTENT_DENIED, ReasonCode.WORKER_EXIT_UNEXPECTED):
        chunks = []
        emitter = media_worker._Emitter(request, chunks.append, Clock())
        emitter.emit('started')
        emitter.error(reason)
        stream = WorkerEventStream(request)
        for chunk in chunks:
            stream.feed(chunk)
        stream.finish()


# ---------------------------------------------------------------------------
# Closed domain negative examples (Docs/contract-domains.json)
# ---------------------------------------------------------------------------

_NEGATIVE = json.loads((Path(__file__).parents[1] / 'Docs/contract-domains.json').read_text())['negative_examples']


_REJECTORS = {
    'worker_slot': (WorkerSlot.parse, lambda bad: make_worker_request(assigned_slots={bad: 'x.json'})),
    'video_quality': (VideoQuality.parse, lambda bad: RequestOutputs(video_quality=bad)),
    'capability': (Capability.parse, lambda bad: make_worker_request(capability=bad)),
    'worker_event': (WorkerEventKind.parse, lambda bad: WorkerEventRecord(1, JOB, ATTEMPT, STAGE, 0, bad, {})),
    'reason_code': (ReasonCode.parse, lambda bad: WorkerEventRecord(
        1, JOB, ATTEMPT, STAGE, 0, 'error', {'reason_code': bad, 'summary': 'x'})),
    'source': (Source.parse, lambda bad: remote_request('metadata', source={'kind': bad, 'input': DIRECT['input']})),
}


@pytest.mark.parametrize('bad', _NEGATIVE)
@pytest.mark.parametrize('domain', sorted(_REJECTORS))
def test_negative_examples_are_rejected_through_real_constructors(bad, domain):
    for rejector in _REJECTORS[domain]:
        with pytest.raises(DomainValidationError):
            rejector(bad)


# ---------------------------------------------------------------------------
# Build inputs: CA secret and build context
# ---------------------------------------------------------------------------

FIXED_TARGET = DockerTarget('desktop-linux', 'unix:///Users/someone/.docker/run/docker.sock')


def test_build_arguments_include_secret_only_when_ca_file_is_configured(tmp_path):
    ca = tmp_path / 'ca.pem'
    ca.write_text('placeholder', encoding='utf-8')
    base = (_docker_executable(), '--host', FIXED_TARGET.endpoint, 'build', '-f', 'containers/media.Dockerfile',
            '-t', IMAGE_TAG)
    assert build_arguments(None, target=FIXED_TARGET) == (*base, '.')
    assert build_arguments('', target=FIXED_TARGET) == (*base, '.')
    assert build_arguments(str(ca), target=FIXED_TARGET) == (*base, '--secret', 'id=build_ca,src=' + str(ca), '.')
    with pytest.raises(AssertionError, match='PODMUNCHER_BUILD_CA_FILE'):
        build_arguments(str(tmp_path / 'missing.pem'), target=FIXED_TARGET)


@pytest.mark.parametrize('platform', ['linux/amd64', 'linux/arm64'])
def test_build_arguments_pass_an_optional_target_platform(tmp_path, platform):
    ca = tmp_path / 'ca.pem'
    ca.write_text('placeholder', encoding='utf-8')
    base = (_docker_executable(), '--host', FIXED_TARGET.endpoint, 'build', '-f', 'containers/media.Dockerfile',
            '-t', IMAGE_TAG)
    assert build_arguments(None, platform, target=FIXED_TARGET) == (*base, '--platform', platform, '.')
    assert build_arguments(str(ca), platform, target=FIXED_TARGET) == (
        *base, '--secret', 'id=build_ca,src=' + str(ca), '--platform', platform, '.')


@pytest.mark.parametrize('platform', ['linux/386', 'windows/amd64', 'linux/arm64 --push', 'amd64'])
def test_build_arguments_reject_an_unsupported_platform(platform):
    with pytest.raises(AssertionError, match='platform'):
        build_arguments(None, platform, target=FIXED_TARGET)


def test_live_target_comes_from_the_current_docker_context(monkeypatch):
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        payload = [{'Name': 'desktop-linux',
                    'Endpoints': {'docker': {'Host': 'unix:///Users/someone/.docker/run/docker.sock'}}}]
        return subprocess.CompletedProcess(arguments, 0, json.dumps(payload), '')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    assert _inspect_current_context() == FIXED_TARGET
    (arguments, kwargs), = calls
    assert arguments == (_docker_executable(), 'context', 'inspect')
    assert kwargs['shell'] is False and kwargs['timeout'] > 0


_HASHED_PIN = re.compile(r'(?P<name>[A-Za-z0-9._-]+)==(?P<version>[A-Za-z0-9.]+)'
                         r'(?P<hashes>(?: --hash=sha256:[0-9a-f]{64})+)\Z')


def _media_pins():
    text, _ = _dockerfile_lines()
    body = text.split("COPY <<'REQUIREMENTS' /opt/media-requirements.txt\n", 1)[1].split('\nREQUIREMENTS\n', 1)[0]
    pins = {}
    for line in body.splitlines():
        match = _HASHED_PIN.fullmatch(line)
        assert match, 'Unhashed or malformed media pin: ' + line
        hashes = re.findall(r'[0-9a-f]{64}', match['hashes'])
        assert len(hashes) == len(set(hashes)) >= 1
        pins[match['name']] = (match['version'], hashes)
    return pins


def test_media_pins_carry_one_or_more_hashes_including_both_deno_architectures():
    pins = _media_pins()
    assert set(pins) == {'yt-dlp', 'yt-dlp-ejs', 'deno', 'platformdirs'}
    # deno ships native wheels: manylinux_2_27_x86_64 and manylinux_2_27_aarch64 (PyPI 2.9.5 digests).
    assert pins['deno'] == ('2.9.5', ['e5aec0419739fd0359663a78765d4feda19d1d0b60a682d25add9b78e19f95b4',
                                      'bccec0965917b6daa323f9ce6ad8e884e0854506b0cdaec9de41bdac7620109a'])
    for name in ('yt-dlp', 'yt-dlp-ejs', 'platformdirs'):
        assert len(pins[name][1]) == 1  # pure-Python py3-none-any wheel


def _dockerfile_lines():
    text = (Path(__file__).parents[1] / 'containers' / 'media.Dockerfile').read_text(encoding='utf-8')
    return text, re.sub(r'\\\n', ' ', text).splitlines()


def test_dockerfile_references_the_ca_only_through_the_secret_mount():
    text, lines = _dockerfile_lines()
    uses = [line for line in lines if 'build_ca' in line and not line.lstrip().startswith('#')]
    assert len(uses) == 1
    assert uses[0].startswith('RUN --mount=type=secret,id=build_ca,required=false ')
    assert uses[0].count('PIP_CERT=/run/secrets/build_ca') == 1 and 'pip install' in uses[0]
    for line in lines:
        words = line.split()
        if words and words[0].upper() in ('COPY', 'ADD', 'ENV', 'ARG'):
            assert 'build_ca' not in line and 'PIP_CERT' not in line and '.pem' not in line.lower()
    assert '.pem' not in text.lower()


def test_dockerignore_limits_the_context_to_the_copied_files():
    root = Path(__file__).parents[1]
    _, lines = _dockerfile_lines()
    copied = set()
    for line in lines:
        words = line.split()
        if words and words[0].upper() == 'COPY' and '<<' not in line and not words[1].startswith('--from='):
            copied.update(words[1:-1])
    assert len(copied) == 7 and all(item.startswith('src/podmuncher/') for item in copied)
    rules = [line.strip() for line in (root / '.dockerignore').read_text(encoding='utf-8').splitlines()
             if line.strip() and not line.startswith('#')]
    assert rules[0] in ('*', '**')
    assert all(rule.startswith('!') for rule in rules[1:])
    included = {rule[1:] for rule in rules[1:]}
    assert included == copied | {'containers/media.Dockerfile'}
    for excluded in ('HN844', 'research'):
        assert not any(item == excluded or item.startswith(excluded + '/') for item in included)


# ---------------------------------------------------------------------------
# Live smoke: admitted image build + real DockerRuntime
# ---------------------------------------------------------------------------

REPO = Path(__file__).parents[1]
IMAGE_TAG = 'podmuncher-media:dev-local'
# Declared build budget: the base image, apt FFmpeg and the Python wheels grow Docker
# storage by at most this much (1 GiB compressed + 2 GiB unpacked = 3 GiB).
BUILD_COMPRESSED_BYTES = 1024**3
BUILD_UNPACKED_BYTES = 2 * 1024**3
BUILD_TIMEOUT_SECONDS = 420
SMOKE_TIMEOUT_SECONDS = 600
SMOKE_MEDIA_BYTES = 256 * 1024**2
_IMAGE_ID = re.compile(r'sha256:[a-f0-9]{64}\Z')
_BUILD_PLATFORM = re.compile(r'linux/(?:amd64|arm64(?:/v8)?)\Z')


def _docker_executable():
    """The docker CLI; Docker Desktop for Mac installs it at /usr/local/bin, off non-login PATHs."""
    found = shutil.which('docker')
    if found:
        return found
    return '/usr/local/bin/docker' if Path('/usr/local/bin/docker').is_file() else 'docker'


def _inspect_current_context():
    """Target the current Docker context's endpoint (desktop-linux npipe on Windows, unix socket on a Mac)."""
    result = subprocess.run((_docker_executable(), 'context', 'inspect'), shell=False, check=False,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, 'docker context inspect failed: ' + result.stderr.strip()[-500:]
    context, = json.loads(result.stdout)
    return DockerTarget(context['Name'], context['Endpoints']['docker']['Host'])


@functools.cache
def live_target():
    return _inspect_current_context()


class _Lease:
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


def _docker(*arguments, timeout=60):
    """Bounded argument-array Docker call pinned to the local engine endpoint."""
    return subprocess.run((_docker_executable(), '--host', live_target().endpoint, *arguments), shell=False,
                          check=False,
                          capture_output=True, text=True, timeout=timeout)


def _storage(session, provider, paths, growth, limits):
    return RuntimeStorage(session, provider.volumes, paths=paths, growth=growth, limits=limits,
                          sample_budget_seconds=1.45)


def build_arguments(ca_file, platform=None, *, target=None):
    """Docker build argument array; the build-time CA is a BuildKit secret, only when configured.

    ``platform`` (linux/amd64 or linux/arm64) cross-builds; by default the engine's native
    architecture is built. ``target`` defaults to the current Docker context.
    """
    target = target or live_target()
    arguments = [_docker_executable(), '--host', target.endpoint, 'build', '-f', 'containers/media.Dockerfile',
                 '-t', IMAGE_TAG]
    if ca_file:
        path = Path(ca_file)
        assert path.is_file(), ('PODMUNCHER_BUILD_CA_FILE is set but does not name an existing file; '
                                'fix the path or unset it')
        arguments += ['--secret', 'id=build_ca,src=' + str(path)]
    if platform is not None:
        assert _BUILD_PLATFORM.fullmatch(platform), 'Unsupported build platform: use linux/amd64 or linux/arm64'
        arguments += ['--platform', platform]
    return (*arguments, '.')


def build_media_image(tmp_path):
    """Admit the declared build budget, then build and return the bare local image ID.

    The first (cold) build is run once as an explicit admitted prerequisite before the
    timed smoke; later runs rely on the Docker layer cache. Nothing is pushed or tagged
    as a registry digest.
    """
    provider = make_docker_storage_provider(timeout=60)
    session = provider.prepare(manual_backing='')
    try:
        assert session.initial_evidence.can_create_cached, session.initial_evidence.reason
        paths = {role: tmp_path / ('build-' + role) for role in ('scratch', 'cache', 'output', 'state')}
        for path in paths.values():
            path.mkdir()
        growth = WorkloadGrowth(BUILD_COMPRESSED_BYTES, BUILD_UNPACKED_BYTES, 0, 0, (8,), 0, 0, 0, 0, (), 0, 0)
        sampled = asyncio.run(_storage(session, provider, paths, growth, Limits()).sample())
        assert sampled.plan.admitted, ('The declared 3 GiB image build budget was not admitted: '
                                       + sampled.plan.reason.value)
    finally:
        session.close()
    built = subprocess.run(build_arguments(os.environ.get('PODMUNCHER_BUILD_CA_FILE')), cwd=REPO,
                           shell=False, check=False,
                           capture_output=True, text=True, timeout=BUILD_TIMEOUT_SECONDS)
    assert built.returncode == 0, 'docker build failed:\n' + '\n'.join(
        (built.stdout + built.stderr).splitlines()[-30:])
    inspected = _docker('image', 'inspect', '--format', '{{.Id}}', IMAGE_TAG)
    image_id = inspected.stdout.strip()
    assert inspected.returncode == 0 and _IMAGE_ID.fullmatch(image_id), 'No bare local image ID'
    return image_id


def _image_proof(image_id):
    def proof():
        result = _docker('image', 'inspect', '--format', '{{.Id}}', image_id, timeout=30)
        return result.returncode == 0 and result.stdout.strip() == image_id
    return proof


@pytest.mark.live
def test_media_worker_live_smoke(tmp_path):
    direct_url = os.environ.get('PODMUNCHER_TEST_DIRECT_URL')
    youtube_url = os.environ.get('PODMUNCHER_TEST_YOUTUBE_URL')
    assert direct_url and youtube_url, (
        'Both PODMUNCHER_TEST_DIRECT_URL and PODMUNCHER_TEST_YOUTUBE_URL must be set '
        '(load only those lines, plus PODMUNCHER_BUILD_CA_FILE when present, from ~/.foreman-mcp/.env; '
        'the live smoke never skips)')
    direct = classify_source(direct_url)
    youtube = classify_source(youtube_url)
    assert direct.kind == 'direct' and youtube.kind == 'youtube', 'Fixture URL kinds are wrong'
    video_id = youtube.input[-11:]

    image_id = build_media_image(tmp_path)  # explicit admitted prerequisite, outside the timer
    began = time.monotonic()
    work = tmp_path / 'live'
    work.mkdir()
    limits = replace(Limits(), media_bytes=SMOKE_MEDIA_BYTES)
    provider = make_docker_storage_provider(timeout=60)
    session = provider.prepare(manual_backing='')
    cli = NativeDockerCLI(live_target())
    instances, durations = [], {}

    policy = WorkerPolicy('media', image_id, ('python', '-I', '-m', 'podmuncher.media_worker'), 0, 0,
                          _image_proof(image_id), development_image=True, allow_pull=False,
                          memory_bytes=1024**3, pids=256)

    async def run_operation(label, req, scratch, input_file=None, inspect_network=False):
        paths = {role: work / (label + '-' + role) for role in ('cache', 'output', 'state')}
        for path in paths.values():
            path.mkdir()
        scratch.mkdir(exist_ok=True)
        growth = WorkloadGrowth(0, 0, 0, 0, (SMOKE_MEDIA_BYTES,), SMOKE_MEDIA_BYTES, 0, 0, 0, (), 0, 0)
        storage = RuntimeStorage(session, provider.volumes, paths={**paths, 'scratch': scratch},
                                 growth=growth, limits=limits, sample_budget_seconds=1.45)
        lease = _Lease()
        coordinator = CancellationCoordinator('media-' + uuid4().hex, lease, stop_grace_seconds=12,
            reap_timeout_seconds=12, callback_timeout_seconds=12)
        guard = WriteGuard(coordinator, storage.sample, byte_limit=SMOKE_MEDIA_BYTES, timeout_seconds=180,
                           sample_timeout_seconds=2.9)
        runtime = DockerRuntime(cli, guard, instance='media-' + uuid4().hex, policy=policy,
                                mounts=WorkerMounts(scratch, input_file=input_file))
        instances.append(runtime.instance)
        inspections = []

        async def inspect_running():
            reference = runtime.reference
            raw = await cli.run('container', 'inspect', reference.container_id, owner=coordinator.operation_id)
            data, = json.loads(raw.stdout)
            host = data['HostConfig']
            assert host['NetworkMode'] == 'none' and host['ReadonlyRootfs']
            assert data['Config']['User'] == '65532:65532' and host['CapDrop'] == ['ALL']
            return True

        def consume(item):
            if item.event == 'started' and inspect_network:
                inspections.append(coordinator.track(inspect_running()))

        if inspect_network:
            args, _ = runtime.create_arguments(req, 'a' * 32)
            assert args[args.index('--network') + 1] == 'none'
        result = await runtime.run(req, on_event=consume)
        for inspection in inspections:
            assert await inspection
        assert result.completed and result.exit_code == 0 and lease.released == 1 and not lease.held
        return result

    def request_for(source, operation, slots, *, video=False, input_basename=None):
        return make_worker_request(operation=operation, source=None if source is None else asdict(source),
            input_basename=input_basename, assigned_slots=slots, limits=asdict(limits),
            outputs=dict(transcript=True, video=video, video_quality='1080p'),
            selection=dict(device='auto', profile='balanced', model='auto', language='auto'))

    async def exercise(label, source, video):
        meta_scratch = work / (label + '-meta-scratch')
        result = await run_operation(label + '-meta', request_for(source, 'metadata', {'probe_metadata': 'meta.json'}),
                                     meta_scratch)
        metadata = json.loads((meta_scratch / 'meta.json').read_bytes())
        assert list(metadata) == PROBE_KEYS and metadata['operation'] == 'metadata'
        assert metadata['source_kind'] == source.kind and result.artifact_claims

        slots = ({'downloaded_video': 'video.bin', 'downloaded_audio': 'audio.bin', 'merged_video': 'merged.bin'}
                 if video else {'downloaded_media': 'media.bin'})
        artifact_name = 'merged.bin' if video else 'media.bin'
        download_scratch = work / (label + '-download-scratch')
        result = await run_operation(label + '-download', request_for(source, 'download', slots, video=video),
                                     download_scratch)
        claims = [c.payload for c in result.artifact_claims]
        assert [c['slot'] for c in claims] == ['merged_video' if video else 'downloaded_media']
        data = (download_scratch / artifact_name).read_bytes()
        assert claims[0]['bytes'] == len(data) > 0 and claims[0]['sha256'] == hashlib.sha256(data).hexdigest()
        assert len(data) <= SMOKE_MEDIA_BYTES

        probe_scratch = work / (label + '-probe-scratch')
        await run_operation(label + '-probe', request_for(None, 'probe', {'probe_metadata': 'probe.json'},
                            input_basename=artifact_name), probe_scratch,
                            input_file=download_scratch / artifact_name, inspect_network=True)
        probed = json.loads((probe_scratch / 'probe.json').read_bytes())
        assert probed['operation'] == 'probe' and list(probed) == PROBE_KEYS
        assert probed['duration_seconds'] is not None and 0 < probed['duration_seconds'] <= limits.duration_seconds
        assert probed['has_audio'] is True or probed['has_video'] is True
        if video:
            assert probed['has_video'] is True and probed['video_streams'] >= 1
        durations[label] = probed['duration_seconds']

    try:
        asyncio.run(exercise('direct', direct, False))
        asyncio.run(exercise('youtube', youtube, True))
        removed = asyncio.run(_leftover_containers(cli, instances))
        assert removed == [], 'Labelled worker containers remain'
        version = subprocess.run((_docker_executable(), '--host', live_target().endpoint, 'run', '--rm', '--pull', 'never',
            '--network', 'none', '--entrypoint', 'python', image_id, '-I', '-c',
            'import yt_dlp.version as v; print(v.__version__)'), shell=False, check=False,
            capture_output=True, text=True, timeout=60)
        assert version.returncode == 0 and version.stdout.strip()
        elapsed = time.monotonic() - began
        assert elapsed < SMOKE_TIMEOUT_SECONDS, 'Live smoke exceeded its declared 600 s bound'
    finally:
        session.close()
        shutil.rmtree(work, ignore_errors=True)
    assert not work.exists()
    print('evidence ts={} yt_dlp={} video_id={} durations=direct:{:.3f},youtube:{:.3f}'.format(
        datetime.now(UTC).isoformat(), version.stdout.strip(), video_id,
        durations['direct'], durations['youtube']))
    print('PODMUNCHER_SMOKE_OK')


async def _leftover_containers(cli, instances):
    left = []
    for instance in instances:
        listed = await cli.run('ps', '-a', '--no-trunc', '--format', '{{.ID}}',
                               '--filter', 'label=podmuncher.instance=' + instance,
                               owner='media-verify-' + uuid4().hex)
        assert listed.returncode == 0
        left.extend(listed.stdout.split())
    return left
