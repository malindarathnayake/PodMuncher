# ruff: noqa: C408
"""ASR worker (P3-U6): deterministic contract tests and an admitted live smoke.

Deterministic cases use injected fakes only: no network, no Docker, no wall clock,
and none of faster_whisper, av or ctranslate2 (the host venv has none of them).
The live case builds the owned Dockerfile's cpu target through an admitted helper,
acquires a pinned fixture model through ModelCache and drives the real DockerRuntime;
it requires PODMUNCHER_TEST_ASR_IMAGE and PODMUNCHER_TEST_MODEL_ROOT and never skips.
"""
import ast
import asyncio
import functools
import hashlib
import io
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from podmuncher import asr_worker
from podmuncher.admission import WorkloadGrowth
from podmuncher.config import Limits
from podmuncher.docker_storage import make_docker_storage_provider
from podmuncher.domains import (
    Backend,
    Capability,
    Device,
    DomainValidationError,
    ModelTier,
    ReasonCode,
    WorkerSlot,
)
from podmuncher.domains import WorkerEvent as WorkerEventKind
from podmuncher.guards import CancellationCoordinator, WriteGuard
from podmuncher.manifests import ManifestFile, ManifestModel
from podmuncher.model_cache import ModelCache
from podmuncher.model_policy import DECODE_BYTES_PER_SECOND, ModelArtifact
from podmuncher.protocol import WorkerEventRecord, WorkerEventStream, WorkerRequest
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
SNAPSHOT = 'snapshot-' + 'ab' * 16
SEGMENTS_NAME = 'segments.json'
LIMITS = Limits()
SECRET_TEXT = 'Secret Words Do Not Leak'
REPO = Path(__file__).parents[1]
RESULT_KEYS = ['schema_version', 'language', 'language_probability', 'duration_seconds',
               'no_speech_detected', 'segments', 'provenance', 'measurements']
PROVENANCE_KEYS = ['model_id', 'model_snapshot_id', 'device', 'compute_type', 'source_sha256',
                   'attempt_id', 'image_digest']
MEASUREMENT_KEYS = ['load_seconds', 'transcribe_wall_seconds', 'audio_seconds', 'peak_rss_bytes',
                    'peak_vram_bytes']
INPUT_BYTES = b'fake container bytes'
MODEL_FILES = ('model.bin', 'config.json', 'tokenizer.json', 'vocabulary.txt')


def make_asr_request(**changes):
    """Mirror tests/test_runtime.py request(), defaulting to a CPU transcribe request."""
    data = dict(schema_version=1, job_id=JOB, attempt_id=ATTEMPT, stage_id=STAGE,
        capability='asr_cpu', operation='transcribe', assigned_slots={'transcript_segments': SEGMENTS_NAME},
        source=None, outputs=dict(transcript=True, video=False, video_quality='source_best'),
        input_basename='input.bin', model_snapshot_id=SNAPSHOT, limits=asdict(LIMITS),
        selection=dict(device='cpu', profile='balanced', model='tiny', language='auto'))
    data.update(changes)
    return WorkerRequest.from_dict(data)


def with_limits(**changes):
    return asdict(replace(LIMITS, **changes))


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


class Seg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class FakeAudio:
    """Sized stand-in for a decoded float32 array."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return self.samples


class FakeWhisperModel:
    """Records transcribe calls; yields a scripted lazy segment generator.

    ``fail_after`` raises ``error`` after that many segments were yielded; ``call_error`` raises
    from ``transcribe`` itself; ``step`` advances the injected clock before each segment.
    """

    def __init__(self, segments=(), *, language='en', probability=0.97, clock=None, step=0.0,
                 fail_after=None, error=None, call_error=None):
        self.segments, self.language, self.probability = list(segments), language, probability
        self.clock, self.step, self.fail_after, self.error = clock, step, fail_after, error
        self.call_error = call_error
        self.calls, self.yielded = [], 0

    def transcribe(self, audio, **options):
        self.calls.append((audio, options))
        if self.call_error is not None:
            raise self.call_error
        info = SimpleNamespace(language=self.language, language_probability=self.probability,
                               duration=None)
        return self._generate(), info

    def _generate(self):
        for index, (start, end, text) in enumerate(self.segments):
            if self.fail_after is not None and index >= self.fail_after:
                raise self.error
            if self.clock is not None:
                self.clock.now += self.step
            self.yielded += 1
            yield Seg(start, end, text)
        if self.fail_after is not None and self.fail_after >= len(self.segments):
            raise self.error


class Factory:
    """Model factory: records constructor arguments and the offline environment at that moment."""

    def __init__(self, model=None, *, error=None, clock=None, load_seconds=0.0):
        self.model, self.error, self.clock, self.load_seconds = model, error, clock, load_seconds
        self.calls, self.environment = [], []

    def __call__(self, path, **kwargs):
        self.calls.append((path, kwargs))
        self.environment.append({name: os.environ.get(name) for name in (
            'HF_HUB_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'TRANSFORMERS_OFFLINE', 'HOME',
            'XDG_CACHE_HOME', 'TMPDIR')} | {'tempdir': tempfile.tempdir})
        if self.clock is not None:
            self.clock.now += self.load_seconds
        if self.error is not None:
            raise self.error
        return self.model


class FakeDecode:
    """``chunk`` samples per step; ``samples=None`` never ends (an unbounded container)."""

    def __init__(self, samples=16000 * 10, *, clock=None, advance=0.0, error=None, chunk=None):
        self.samples, self.clock, self.advance, self.error = samples, clock, advance, error
        self.chunk, self.chunks = chunk, 0
        self.calls = []
        self.audio = FakeAudio(samples or 0)

    def __call__(self, path, check):
        self.calls.append(path)
        if self.error is not None:
            raise self.error
        done = 0
        while self.samples is None or done < self.samples:
            step = self.chunk or self.samples
            done += step if self.samples is None else min(step, self.samples - done)
            self.chunks += 1
            if self.clock is not None:
                self.clock.now += self.advance
            check(done)
        return self.audio


class FakeProbe:
    def __init__(self, duration=10.0, *, has_audio=True, error=None):
        self.result = asr_worker.MediaProbe(has_audio, duration)
        self.error, self.calls = error, []

    def __call__(self, path):
        self.calls.append(path)
        if self.error is not None:
            raise self.error
        return self.result


def forbidden(*_args, **_kwargs):
    raise AssertionError('unexpected I/O')


class Outcome:
    def __init__(self, code, collector, scratch, input_root, model_root, factory, decode, probe):
        self.code, self.collector, self.scratch = code, collector, scratch
        self.input_root, self.model_root = input_root, model_root
        self.factory, self.decode, self.probe = factory, decode, probe

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

    @property
    def result(self):
        return json.loads((self.scratch / SEGMENTS_NAME).read_bytes())


def execute(tmp_path, request=None, *, model=None, factory=None, decode=None, probe=None, clock=None,
            memory_limit=lambda: None, peak_rss=lambda: 4096, model_files=MODEL_FILES, make_input=True,
            **kwargs):
    """Run the worker once and prove the emitted bytes are a host-valid event stream."""
    request = request or make_asr_request()
    scratch, input_root, model_root = tmp_path / 'scratch', tmp_path / 'input', tmp_path / 'model'
    for path in (scratch, input_root, model_root):
        path.mkdir(parents=True, exist_ok=True)
    for name in model_files:
        (model_root / name).write_bytes(b'x')
    if make_input:
        (input_root / 'input.bin').write_bytes(INPUT_BYTES)
    clock = clock or Clock()
    collector = Collector(clock)
    factory = factory or Factory(model or FakeWhisperModel(), clock=clock)
    decode = decode or FakeDecode()
    probe = probe or FakeProbe()
    code = asr_worker.run(request, collector, scratch=scratch, input_root=input_root,
        model_root=model_root, model_factory=factory, decode=decode, probe=probe,
        memory_limit=memory_limit, peak_rss=peak_rss, clock=clock, **kwargs)
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
    return Outcome(code, collector, scratch, input_root, model_root, factory, decode, probe)


@pytest.fixture(autouse=True)
def restore_process_state(monkeypatch):
    """The worker forces offline env and tempfile.tempdir in-process; undo it after each test."""
    for name in ('HF_HUB_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'TRANSFORMERS_OFFLINE', 'HOME',
                 'XDG_CACHE_HOME', 'TMPDIR'):
        monkeypatch.setenv(name, 'original-' + name)
    monkeypatch.setattr(tempfile, 'tempdir', tempfile.tempdir)


GOOD = [(0.0, 2.5, ' Hello world '), (2.5, 5.0, 'second part')]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_transcribe_options_auto_and_explicit_language():
    assert asr_worker.transcribe_options('auto') == dict(
        language=None, task='transcribe', beam_size=5, vad_filter=True, word_timestamps=False,
        condition_on_previous_text=False)
    assert asr_worker.transcribe_options('en')['language'] == 'en'
    assert asr_worker.transcribe_options('en') | {'language': None} == asr_worker.transcribe_options('auto')


def test_compute_type_is_fixed_per_device():
    assert asr_worker.compute_type_for('cpu') == 'int8'
    assert asr_worker.compute_type_for('cuda') == 'int8_float16'
    for bad in ('auto', '', 'gpu', None):
        with pytest.raises(ValueError):
            asr_worker.compute_type_for(bad)


def test_decode_working_set_constants_match_the_host_policy():
    assert asr_worker.DECODE_BYTES_PER_SECOND == DECODE_BYTES_PER_SECOND
    assert asr_worker.PCM_BYTES_PER_SECOND == 64000 and asr_worker.SAMPLE_RATE == 16000


def test_module_imports_without_heavy_dependencies():
    for name in ('faster_whisper', 'av', 'ctranslate2', 'yt_dlp'):
        assert name not in sys.modules


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_exact_document_events_and_artifact(tmp_path):
    clock = Clock()
    model = FakeWhisperModel(GOOD, clock=clock, step=0.5)
    factory = Factory(model, clock=clock, load_seconds=2.0)
    outcome = execute(tmp_path, model=model, factory=factory, clock=clock, peak_rss=lambda: 123456789)
    assert outcome.code == 0 and outcome.events[0] == 'started' and outcome.events[-2:] == ['artifact', 'completed']
    assert set(outcome.events) <= {'started', 'progress', 'artifact', 'completed'}
    data = (outcome.scratch / SEGMENTS_NAME).read_bytes()
    result = json.loads(data)
    assert list(result) == RESULT_KEYS
    assert result['schema_version'] == 1 and result['language'] == 'en'
    assert result['language_probability'] == 0.97 and result['duration_seconds'] == 10.0
    assert result['no_speech_detected'] is False
    assert result['segments'] == [{'start_seconds': 0.0, 'end_seconds': 2.5, 'text': 'Hello world'},
                                  {'start_seconds': 2.5, 'end_seconds': 5.0, 'text': 'second part'}]
    assert list(result['provenance']) == PROVENANCE_KEYS
    assert result['provenance'] == dict(
        model_id='tiny', model_snapshot_id=SNAPSHOT, device='cpu', compute_type='int8',
        source_sha256=hashlib.sha256(INPUT_BYTES).hexdigest(), attempt_id=ATTEMPT, image_digest=None)
    assert list(result['measurements']) == MEASUREMENT_KEYS
    assert result['measurements'] == dict(load_seconds=2.0, transcribe_wall_seconds=1.0, audio_seconds=10.0,
                                          peak_rss_bytes=123456789, peak_vram_bytes=None)
    assert b' ' not in data.replace(b'Hello world', b'').replace(b'second part', b'')
    for forbidden_word in (b'speaker', b'diariz', b'words'):
        assert forbidden_word not in data
    claim = outcome.collector.of('artifact')[0].payload
    assert dict(claim) == {'slot': 'transcript_segments', 'bytes': len(data),
                           'sha256': hashlib.sha256(data).hexdigest()}
    assert outcome.files == [SEGMENTS_NAME]  # the temp dir is gone


def test_model_is_constructed_offline_with_exact_arguments(tmp_path):
    outcome = execute(tmp_path)
    path, kwargs = outcome.factory.calls[0]
    assert path == str(outcome.model_root)
    assert kwargs.pop('cpu_threads') >= 1
    assert kwargs == dict(device='cpu', compute_type='int8', local_files_only=True, num_workers=1)
    audio, options = outcome.factory.model.calls[0]
    assert audio is outcome.decode.audio and options == asr_worker.transcribe_options('auto')
    assert outcome.decode.calls == [str(outcome.input_root / 'input.bin')]
    assert outcome.probe.calls == [outcome.input_root / 'input.bin']


def test_cuda_request_uses_cuda_device_and_int8_float16(tmp_path):
    request = make_asr_request(capability='asr_cuda', selection=dict(
        device='cuda', profile='balanced', model='tiny', language='en'))
    outcome = execute(tmp_path, request, model=FakeWhisperModel(GOOD, language='en'))
    _, kwargs = outcome.factory.calls[0]
    assert kwargs['device'] == 'cuda' and kwargs['compute_type'] == 'int8_float16'
    assert outcome.factory.model.calls[0][1]['language'] == 'en'
    result = outcome.result
    assert result['provenance']['device'] == 'cuda' and result['provenance']['compute_type'] == 'int8_float16'
    assert result['measurements']['peak_vram_bytes'] is None


def test_progress_is_covered_audio_time_throttled_and_never_100(tmp_path):
    clock = Clock()
    segments = [(i * 1.25, (i + 1) * 1.25, f'word{i}') for i in range(8)]  # ends exactly at 10.0
    model = FakeWhisperModel(segments, clock=clock, step=0.25)
    outcome = execute(tmp_path, model=model, clock=clock)
    progress = outcome.collector.of('progress')
    times = [t for r, t in zip(outcome.collector.records, outcome.collector.times) if r.event == 'progress']
    percents = [r.payload['percent'] for r in progress]
    assert 1 < len(progress) < len(segments)
    assert all(b - a >= 1.0 for a, b in itertools.pairwise(times))
    assert percents == sorted(percents) and all(0 <= p <= 99.0 for p in percents)
    assert percents[0] == 12.5 and 100 not in percents
    # a segment covering all the audio is capped below 100 until the flush
    done = execute(tmp_path / 'full', model=FakeWhisperModel([(0.0, 10.0, 'all')]))
    assert [r.payload['percent'] for r in done.collector.of('progress')] == [99.0]
    assert done.events.index('progress') < done.events.index('artifact') < done.events.index('completed')


def test_events_never_carry_transcript_text_or_paths(tmp_path):
    outcome = execute(tmp_path, model=FakeWhisperModel([(0.0, 1.0, SECRET_TEXT)]))
    wire = b''.join(outcome.collector.chunks)
    assert SECRET_TEXT.encode() not in wire and str(tmp_path).encode() not in wire
    failed = execute(tmp_path / 'bad', model=FakeWhisperModel(
        [(0.0, 1.0, SECRET_TEXT)], fail_after=1, error=RuntimeError('CUDA out of memory ' + SECRET_TEXT)))
    assert failed.error == 'model_oom'
    assert SECRET_TEXT.encode() not in b''.join(failed.collector.chunks)


def test_explicit_language_passes_through_and_probability_must_be_finite(tmp_path):
    request = make_asr_request(selection=dict(device='cpu', profile='balanced', model='tiny', language='de'))
    model = FakeWhisperModel(GOOD, language='de', probability=float('nan'))
    outcome = execute(tmp_path, request, model=model)
    assert model.calls[0][1]['language'] == 'de'
    assert outcome.result['language'] == 'de' and outcome.result['language_probability'] is None


def test_missing_detected_language_is_artifact_rejected_when_auto(tmp_path):
    outcome = execute(tmp_path, model=FakeWhisperModel(GOOD, language=None))
    assert outcome.code == 1 and outcome.error == 'artifact_rejected' and outcome.files == []


# ---------------------------------------------------------------------------
# Envelope rejections: before any I/O
# ---------------------------------------------------------------------------

def _device_mismatch_cpu():
    return make_asr_request(selection=dict(device='cuda', profile='balanced', model='tiny', language='auto'))


def _device_mismatch_cuda():
    return make_asr_request(capability='asr_cuda')


def _auto_device():
    return make_asr_request(selection=dict(device='auto', profile='balanced', model='tiny', language='auto'))


def _no_input():
    request = make_asr_request()
    object.__setattr__(request, 'input_basename', None)  # the protocol forbids building this
    return request


@pytest.mark.parametrize('make', [
    lambda: make_asr_request(operation='probe'),
    lambda: make_asr_request(capability='media'),
    _device_mismatch_cpu,
    _device_mismatch_cuda,
    _auto_device,
    lambda: make_asr_request(assigned_slots={'probe_metadata': 'meta.json'}),
    lambda: make_asr_request(assigned_slots={'transcript_segments': SEGMENTS_NAME, 'probe_metadata': 'meta.json'}),
    lambda: make_asr_request(model_snapshot_id=None),
    lambda: make_asr_request(selection=dict(device='cpu', profile='balanced', model='auto', language='auto')),
    _no_input,
])
def test_envelope_violations_are_rejected_before_any_io(tmp_path, make):
    request = make()
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    collector = Collector()
    code = asr_worker.run(request, collector, scratch=scratch, input_root=tmp_path / 'no-input',
        model_root=tmp_path / 'no-model', model_factory=forbidden, decode=forbidden, probe=forbidden,
        memory_limit=forbidden, peak_rss=forbidden)
    assert code == 1 and collector.events == ['started', 'error']
    assert collector.of('error')[0].payload['reason_code'] == 'worker_protocol_mismatch'
    assert list(scratch.iterdir()) == [] and not (tmp_path / 'no-input').exists()


def test_transcribe_without_input_is_refused_by_the_request_layer_too():
    with pytest.raises(DomainValidationError):
        make_asr_request(input_basename=None)
    with pytest.raises(DomainValidationError):
        make_asr_request(source={'kind': 'direct', 'input': 'https://media.example.com/a.mp3'})


def test_fake_model_is_never_constructed_on_envelope_rejection(tmp_path):
    factory = Factory(FakeWhisperModel())
    outcome = execute(tmp_path, make_asr_request(model_snapshot_id=None), factory=factory)
    assert outcome.error == 'worker_protocol_mismatch' and factory.calls == []


# ---------------------------------------------------------------------------
# Model verification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('missing', ['model.bin', 'config.json', 'tokenizer.json', 'vocabulary.txt'])
def test_missing_model_file_is_asset_verification_failed_and_never_loaded(tmp_path, missing):
    files = tuple(name for name in MODEL_FILES if name != missing)
    outcome = execute(tmp_path, model_files=files)
    assert outcome.code == 1 and outcome.error == 'asset_verification_failed'
    assert outcome.factory.calls == [] and outcome.files == []


def test_vocabulary_may_use_any_extension(tmp_path):
    files = ('model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json')
    assert execute(tmp_path, model_files=files).code == 0


def test_symlinked_model_file_is_asset_verification_failed(tmp_path, monkeypatch):
    real = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda self: self.name == 'model.bin' or real(self))
    outcome = execute(tmp_path)
    assert outcome.error == 'asset_verification_failed' and outcome.factory.calls == []


def test_symlinked_vocabulary_is_asset_verification_failed(tmp_path, monkeypatch):
    real = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda self: self.name.startswith('vocabulary') or real(self))
    assert execute(tmp_path).error == 'asset_verification_failed'


def test_model_directory_entries_must_be_regular_files(tmp_path):
    scratch, model_root = tmp_path / 'scratch', tmp_path / 'model'
    scratch.mkdir()
    model_root.mkdir()
    for name in MODEL_FILES:
        (model_root / name).mkdir()  # directories are not files
    collector = Collector()
    factory = Factory(FakeWhisperModel())
    code = asr_worker.run(make_asr_request(), collector, scratch=scratch, input_root=tmp_path,
        model_root=model_root, model_factory=factory, decode=forbidden, probe=forbidden)
    assert code == 1 and collector.of('error')[0].payload['reason_code'] == 'asset_verification_failed'
    assert factory.calls == []


def test_missing_model_directory_is_asset_verification_failed(tmp_path):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    collector = Collector()
    code = asr_worker.run(make_asr_request(), collector, scratch=scratch, input_root=tmp_path,
        model_root=tmp_path / 'absent', model_factory=forbidden, decode=forbidden, probe=forbidden)
    assert code == 1 and collector.of('error')[0].payload['reason_code'] == 'asset_verification_failed'


# ---------------------------------------------------------------------------
# Load and transcribe failures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(('error', 'reason'), [
    (MemoryError(), 'model_oom'),
    (RuntimeError('CUDA failed with error out of memory'), 'model_oom'),
    (RuntimeError('wrapper'), 'model_load_failed'),
    (ValueError('bad config'), 'model_load_failed'),
    (OSError('disk gone'), 'model_load_failed'),
])
def test_load_failures_map_to_closed_reasons(tmp_path, error, reason):
    if str(error) == 'wrapper':
        error.__cause__ = None
    outcome = execute(tmp_path, factory=Factory(error=error))
    assert outcome.code == 1 and outcome.error == reason
    assert outcome.decode.calls == [] and outcome.files == []


def test_oom_text_in_a_chained_cause_is_model_oom(tmp_path):
    try:
        try:
            raise RuntimeError('CUDA out of memory')
        except RuntimeError as inner:
            raise RuntimeError('load failed') from inner
    except RuntimeError as outer:
        error = outer
    assert execute(tmp_path, factory=Factory(error=error)).error == 'model_oom'


def test_oom_mid_iteration_discards_everything(tmp_path):
    model = FakeWhisperModel([(0.0, 1.0, 'one'), (1.0, 2.0, 'two'), (2.0, 3.0, 'three')],
                             fail_after=2, error=MemoryError())
    outcome = execute(tmp_path, model=model)
    assert outcome.code == 1 and outcome.error == 'model_oom'
    assert model.yielded == 2 and outcome.files == []
    assert 'artifact' not in outcome.events and 'completed' not in outcome.events


def test_cuda_oom_text_mid_iteration_is_model_oom_and_leaves_no_artifact(tmp_path):
    model = FakeWhisperModel(GOOD, fail_after=1, error=RuntimeError('CUDA failed: out of memory'))
    outcome = execute(tmp_path, model=model)
    assert outcome.error == 'model_oom' and outcome.files == []


def test_oom_inside_the_transcribe_call_is_model_oom(tmp_path):
    outcome = execute(tmp_path, model=FakeWhisperModel(call_error=MemoryError()))
    assert outcome.error == 'model_oom' and outcome.files == []


def test_other_transcribe_failures_are_a_generic_exit_reason(tmp_path):
    model = FakeWhisperModel(GOOD, fail_after=1, error=RuntimeError('private ' + SECRET_TEXT))
    outcome = execute(tmp_path, model=model)
    assert outcome.error == 'worker_exit_unexpected' and outcome.files == []
    assert SECRET_TEXT.encode() not in b''.join(outcome.collector.chunks)


def test_slow_decode_is_request_timeout(tmp_path):
    clock = Clock()
    decode = FakeDecode(clock=clock, advance=LIMITS.decode_timeout_seconds + 1)
    model = FakeWhisperModel(GOOD)
    outcome = execute(tmp_path, clock=clock, decode=decode, model=model, factory=Factory(model, clock=clock))
    assert outcome.code == 1 and outcome.error == 'request_timeout'
    assert model.calls == [] and outcome.files == []


def test_deadline_is_checked_between_segments_and_excludes_load_time(tmp_path):
    clock = Clock()
    model = FakeWhisperModel([(i, i + 1.0, f't{i}') for i in range(5)], clock=clock, step=700.0)
    outcome = execute(tmp_path, clock=clock, model=model, factory=Factory(model, clock=clock, load_seconds=5000.0))
    assert outcome.error == 'request_timeout' and model.yielded == 3 and outcome.files == []
    fast_clock = Clock()
    fast = FakeWhisperModel(GOOD, clock=fast_clock, step=100.0)
    ok = execute(tmp_path / 'ok', clock=fast_clock, model=fast, factory=Factory(fast, clock=fast_clock, load_seconds=5000.0))
    assert ok.code == 0  # a long load does not consume the decode+transcribe budget


# ---------------------------------------------------------------------------
# Media probe, duration and memory
# ---------------------------------------------------------------------------

def test_declared_duration_over_limit_is_rejected_before_decode(tmp_path):
    request = make_asr_request(limits=with_limits(duration_seconds=100))
    outcome = execute(tmp_path, request, probe=FakeProbe(100.5), decode=FakeDecode(samples=1))
    assert outcome.error == 'duration_exceeded' and outcome.decode.calls == [] and outcome.files == []


def test_declared_duration_equal_to_the_limit_is_accepted(tmp_path):
    request = make_asr_request(limits=with_limits(duration_seconds=100))
    outcome = execute(tmp_path, request, probe=FakeProbe(100.0), decode=FakeDecode(samples=16000 * 100))
    assert outcome.code == 0 and outcome.result['duration_seconds'] == 100.0


def test_inaccurate_metadata_is_caught_after_decoding(tmp_path):
    request = make_asr_request(limits=with_limits(duration_seconds=100))
    decode = FakeDecode(samples=16000 * 100 + 1)  # declared 10 s, decoded just over the limit
    outcome = execute(tmp_path, request, probe=FakeProbe(10.0), decode=decode)
    assert outcome.error == 'duration_exceeded' and len(decode.calls) == 1 and outcome.files == []
    assert outcome.factory.model.calls == []


@pytest.mark.parametrize('duration', [None, 0.0, -1.0, float('nan'), float('inf')])
def test_unknown_or_unusable_declared_duration_falls_back_to_the_decoded_one(tmp_path, duration):
    request = make_asr_request(limits=with_limits(duration_seconds=100))
    outcome = execute(tmp_path, request, probe=FakeProbe(duration), decode=FakeDecode(samples=16000 * 20))
    assert outcome.code == 0 and outcome.result['duration_seconds'] == 20.0


def test_no_audio_stream_is_no_audio_without_decoding(tmp_path):
    outcome = execute(tmp_path, probe=FakeProbe(has_audio=False))
    assert outcome.error == 'no_audio' and outcome.decode.calls == [] and outcome.files == []


@pytest.mark.parametrize('error', [asr_worker.MediaProbeError(), RuntimeError('probe boom'), OSError('io')])
def test_unopenable_or_corrupt_container_is_media_probe_failed(tmp_path, error):
    outcome = execute(tmp_path, probe=FakeProbe(error=error))
    assert outcome.error == 'media_probe_failed' and outcome.decode.calls == []


def test_missing_input_file_is_media_probe_failed(tmp_path):
    probe = FakeProbe()
    outcome = execute(tmp_path, probe=probe, make_input=False)
    assert outcome.error == 'media_probe_failed' and probe.calls == []


def test_corrupt_input_never_constructs_the_model(tmp_path):
    outcome = execute(tmp_path, probe=FakeProbe(error=asr_worker.MediaProbeError()))
    assert outcome.error == 'media_probe_failed' and outcome.factory.model.calls == []
    assert outcome.factory.calls == [] and not (outcome.scratch / '.asr-tmp').exists()


def test_zero_decoded_samples_is_no_audio(tmp_path):
    outcome = execute(tmp_path, decode=FakeDecode(samples=0))
    assert outcome.error == 'no_audio' and outcome.factory.model.calls == [] and outcome.files == []


def test_decode_failure_is_media_probe_failed_and_decode_oom_is_model_oom(tmp_path):
    assert execute(tmp_path, decode=FakeDecode(error=RuntimeError('invalid data'))).error == 'media_probe_failed'
    oom = execute(tmp_path / 'oom', decode=FakeDecode(error=MemoryError()))
    assert oom.error == 'model_oom' and oom.files == []


def test_memory_precheck_against_the_cgroup_limit(tmp_path):
    needed = 10 * (64000 + 256000)
    refused = execute(tmp_path, memory_limit=lambda: needed - 1, probe=FakeProbe(10.0))
    assert refused.error == 'ram_insufficient' and refused.decode.calls == [] and refused.files == []
    exact = execute(tmp_path / 'exact', memory_limit=lambda: needed, probe=FakeProbe(10.0))
    assert exact.code == 0


def test_memory_check_is_skipped_when_the_limit_is_unknown(tmp_path):
    assert execute(tmp_path, memory_limit=lambda: None, probe=FakeProbe(10.0)).code == 0


def test_memory_is_rechecked_with_the_true_duration_when_metadata_is_missing(tmp_path):
    outcome = execute(tmp_path, memory_limit=lambda: 1000, probe=FakeProbe(None))
    assert outcome.error == 'ram_insufficient' and len(outcome.decode.calls) == 1 and outcome.files == []


def test_unavailable_peak_rss_is_null_in_the_result(tmp_path):
    outcome = execute(tmp_path, peak_rss=lambda: None)
    assert outcome.code == 0 and outcome.result['measurements']['peak_rss_bytes'] is None


def test_unbounded_decode_is_stopped_at_the_duration_limit(tmp_path):
    decode = FakeDecode(samples=None, chunk=16000)
    outcome = execute(tmp_path, probe=FakeProbe(None), decode=decode)
    assert outcome.error == 'duration_exceeded' and outcome.files == []
    assert decode.chunks == int(LIMITS.duration_seconds) + 1 and outcome.factory.model.calls == []


def test_clock_crossing_the_deadline_mid_decode_is_request_timeout(tmp_path):
    clock = Clock()
    decode = FakeDecode(samples=16000 * 10, chunk=16000, clock=clock,
                        advance=LIMITS.decode_timeout_seconds / 2 + 1)
    outcome = execute(tmp_path, clock=clock, decode=decode, probe=FakeProbe(10.0))
    assert outcome.error == 'request_timeout' and outcome.files == []
    assert decode.chunks == 2 and outcome.factory.model.calls == []


def test_memory_cap_reached_mid_decode_is_ram_insufficient(tmp_path):
    decode = FakeDecode(samples=16000 * 10, chunk=16000)
    outcome = execute(tmp_path, probe=FakeProbe(None), decode=decode, memory_limit=lambda: 1_000_000)
    assert outcome.error == 'ram_insufficient' and outcome.files == []
    assert decode.chunks == 4 and outcome.factory.model.calls == []  # 4 s * 320000 B/s > 1 MB


def test_default_cgroup_readers(tmp_path, monkeypatch):
    limit, peak = tmp_path / 'memory.max', tmp_path / 'memory.peak'
    monkeypatch.setattr(asr_worker, '_CGROUP_MEMORY_MAX', str(limit))
    monkeypatch.setattr(asr_worker, '_CGROUP_MEMORY_PEAK', str(peak))
    assert asr_worker._default_memory_limit() is None  # unreadable
    limit.write_text('max\n')
    assert asr_worker._default_memory_limit() is None
    limit.write_text('2147483648\n')
    assert asr_worker._default_memory_limit() == 2147483648
    peak.write_text('777\n')
    monkeypatch.setitem(sys.modules, 'resource', SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _who: SimpleNamespace(ru_maxrss=5)))
    assert asr_worker._default_peak_rss() == 5 * 1024  # max of getrusage and cgroup
    peak.write_text(str(10**9))
    assert asr_worker._default_peak_rss() == 10**9
    monkeypatch.setitem(sys.modules, 'resource', None)  # import fails
    peak.unlink()
    assert asr_worker._default_peak_rss() is None  # nothing readable: null, never an invented zero


# ---------------------------------------------------------------------------
# Result validation and output
# ---------------------------------------------------------------------------

def test_no_speech_is_a_successful_empty_result(tmp_path):
    outcome = execute(tmp_path, model=FakeWhisperModel([], language='en'))
    assert outcome.code == 0 and outcome.events[-2:] == ['artifact', 'completed']
    result = outcome.result
    assert result['no_speech_detected'] is True and result['segments'] == []
    assert result['duration_seconds'] == 10.0 and list(result) == RESULT_KEYS


def test_empty_text_segments_are_dropped_and_all_empty_means_no_speech(tmp_path):
    mixed = execute(tmp_path, model=FakeWhisperModel([(0.0, 1.0, '  '), (1.0, 2.0, ' kept '), (2.0, 3.0, '')]))
    assert mixed.result['segments'] == [{'start_seconds': 1.0, 'end_seconds': 2.0, 'text': 'kept'}]
    assert mixed.result['no_speech_detected'] is False
    silent = execute(tmp_path / 's', model=FakeWhisperModel([(0.0, 1.0, ' '), (1.0, 2.0, '\n')]))
    assert silent.result['no_speech_detected'] is True and silent.result['segments'] == []


@pytest.mark.parametrize('segments', [
    [(float('nan'), 1.0, 'x')],
    [(0.0, float('nan'), 'x')],
    [(0.0, float('inf'), 'x')],
    [(-0.1, 1.0, 'x')],
    [(2.0, 1.0, 'x')],
    [(12.0, 11.0, 'x')],  # reversed past the end: rejected before any clamping
    [(-1.0, 12.0, 'x')],
    [(2.0, 3.0, 'a'), (1.0, 4.0, 'b')],
    [(11.0, 12.0, 'a'), (10.5, 13.0, 'b')],  # non-monotonic in raw time, even past the end
    [(0.0, 1.0, None)],
    [(0.0, 1.0, b'bytes')],
    [(True, 1.0, 'x')],
    [(0.0, 1.0, ''), (float('nan'), 2.0, 'x')],
])
def test_invalid_segments_are_artifact_rejected(tmp_path, segments):
    outcome = execute(tmp_path, model=FakeWhisperModel(segments))
    assert outcome.code == 1 and outcome.error == 'artifact_rejected'
    assert outcome.files == [] and 'artifact' not in outcome.events


@pytest.mark.parametrize('overshoot', [0.96, 5.0])
def test_final_segment_past_the_duration_is_clamped_and_the_transcript_kept(tmp_path, overshoot):
    segments = [(0.0, 4.0, 'first'), (4.0, 10.0 + overshoot, 'last')]
    outcome = execute(tmp_path, model=FakeWhisperModel(segments))
    assert outcome.code == 0 and outcome.events[-2:] == ['artifact', 'completed']
    assert outcome.result['segments'] == [
        {'start_seconds': 0.0, 'end_seconds': 4.0, 'text': 'first'},
        {'start_seconds': 4.0, 'end_seconds': 10.0, 'text': 'last'}]


def test_segment_wholly_past_the_duration_is_kept_at_the_boundary(tmp_path):
    segments = [(9.0, 10.0, 'edge'), (10.7, 10.8, 'late'), (11.0, 12.0, 'later')]
    outcome = execute(tmp_path, model=FakeWhisperModel(segments))
    assert outcome.code == 0
    assert outcome.result['segments'] == [
        {'start_seconds': 9.0, 'end_seconds': 10.0, 'text': 'edge'},
        {'start_seconds': 10.0, 'end_seconds': 10.0, 'text': 'late'},
        {'start_seconds': 10.0, 'end_seconds': 10.0, 'text': 'later'}]


def test_validate_keeps_every_clamped_time_inside_zero_to_duration():
    clamped = asr_worker._Job._validate([(0.0, 1.0, 'a'), (9.5, 10.96, 'b'), (12.0, 13.0, 'c')], 10.0)
    assert [(s['start_seconds'], s['end_seconds']) for s in clamped] == [(0.0, 1.0), (9.5, 10.0), (10.0, 10.0)]


def test_text_is_unicode_and_the_document_is_compact(tmp_path):
    outcome = execute(tmp_path, model=FakeWhisperModel([(0.0, 1.0, 'café 日本')]))
    data = (outcome.scratch / SEGMENTS_NAME).read_bytes()
    assert 'café 日本' in data.decode('utf-8') and b', ' not in data and b': ' not in data


def test_export_size_cap_is_artifact_rejected(tmp_path):
    request = make_asr_request(limits=with_limits(export_bytes=300))
    big = FakeWhisperModel([(0.0, 1.0, 'x' * 500)])
    outcome = execute(tmp_path, request, model=big)
    assert outcome.error == 'artifact_rejected' and outcome.files == []
    request = make_asr_request(limits=with_limits(export_bytes=10_000))
    assert execute(tmp_path / 'ok', request, model=FakeWhisperModel([(0.0, 1.0, 'x' * 500)])).code == 0


def test_existing_destination_is_a_conflict_and_is_never_deleted_or_overwritten(tmp_path):
    (tmp_path / 'scratch').mkdir()
    (tmp_path / 'scratch' / SEGMENTS_NAME).write_bytes(b'host-owned')
    outcome = execute(tmp_path)
    assert outcome.code == 1 and outcome.error == 'destination_conflict'
    assert (outcome.scratch / SEGMENTS_NAME).read_bytes() == b'host-owned'
    assert 'artifact' not in outcome.events


def test_an_early_failure_never_touches_a_pre_existing_destination(tmp_path):
    (tmp_path / 'scratch').mkdir()
    (tmp_path / 'scratch' / SEGMENTS_NAME).write_bytes(b'host-owned')
    outcome = execute(tmp_path, probe=FakeProbe(has_audio=False))
    assert outcome.error == 'no_audio' and (outcome.scratch / SEGMENTS_NAME).read_bytes() == b'host-owned'


def test_disk_full_is_storage_guard_tripped_and_the_partial_file_is_removed(tmp_path, monkeypatch):
    import errno
    real_fsync = os.fsync

    def failing_fsync(descriptor):
        real_fsync(descriptor)
        raise OSError(errno.ENOSPC, 'No space left on device')

    monkeypatch.setattr(os, 'fsync', failing_fsync)
    outcome = execute(tmp_path)
    assert outcome.error == 'storage_guard_tripped' and outcome.files == []
    assert 'artifact' not in outcome.events and 'completed' not in outcome.events


def test_completion_is_emitted_only_after_the_flush(tmp_path, monkeypatch):
    order = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, 'fsync', lambda descriptor: (order.append('fsync'), real_fsync(descriptor))[1])

    class Recording(Collector):
        def __call__(self, data):
            super().__call__(data)
            order.append(WorkerEventRecord.from_json(data[:-1]).event)

    clock = Clock()
    collector = Recording(clock)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    (tmp_path / 'model').mkdir()
    for name in MODEL_FILES:
        (tmp_path / 'model' / name).write_bytes(b'x')
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input' / 'input.bin').write_bytes(INPUT_BYTES)
    code = asr_worker.run(make_asr_request(), collector, scratch=scratch, input_root=tmp_path / 'input',
        model_root=tmp_path / 'model', model_factory=Factory(FakeWhisperModel(GOOD), clock=clock),
        decode=FakeDecode(), probe=FakeProbe(), memory_limit=lambda: None, peak_rss=lambda: 1, clock=clock)
    assert code == 0
    assert order.index('fsync') < order.index('artifact') < order.index('completed')
    assert order[-1] == 'completed'


# ---------------------------------------------------------------------------
# Offline environment and temp directory
# ---------------------------------------------------------------------------

def test_offline_environment_is_forced_before_the_model_loads_and_tmp_is_removed(tmp_path):
    outcome = execute(tmp_path)
    seen, = outcome.factory.environment
    tmp = str(outcome.scratch / '.asr-tmp')
    assert seen['HF_HUB_OFFLINE'] == '1' and seen['HF_HUB_DISABLE_TELEMETRY'] == '1'
    assert seen['TRANSFORMERS_OFFLINE'] == '1'
    assert seen['HOME'] == seen['XDG_CACHE_HOME'] == seen['TMPDIR'] == seen['tempdir'] == tmp
    assert outcome.files == [SEGMENTS_NAME]  # worker-owned dir deleted in finally


def test_tmp_dir_is_removed_even_on_failure(tmp_path):
    for index, factory in enumerate((Factory(error=RuntimeError('x')), Factory(error=MemoryError()))):
        outcome = execute(tmp_path / str(index), factory=factory)
        assert outcome.code == 1 and outcome.files == []
    mid = execute(tmp_path / 'mid', model=FakeWhisperModel(GOOD, fail_after=1, error=MemoryError()))
    assert mid.files == []


def test_no_tmp_dir_is_created_when_the_model_files_are_missing(tmp_path):
    outcome = execute(tmp_path, model_files=())
    assert outcome.error == 'asset_verification_failed' and outcome.files == []
    assert outcome.factory.environment == []


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def test_worker_writes_only_events_to_the_sink_and_nothing_to_stdout(tmp_path, capsys):
    execute(tmp_path)
    captured = capsys.readouterr()
    assert captured.out == '' and captured.err == ''


def test_main_rejects_garbage_without_writing_events():
    for garbage in (b'not json\n', b'{}\n', b'', b'x' * 70000):
        out = io.BytesIO()
        assert asr_worker.main(stdin=io.BytesIO(garbage), stdout=out) == 2
        assert out.getvalue() == b''


def test_main_delegates_a_valid_request_and_returns_the_exit_code(monkeypatch):
    request = make_asr_request()
    seen = {}

    def fake_run(req, sink, **kwargs):
        seen['request'], seen['kwargs'] = req, kwargs
        sink(b'line\n')
        return 7

    monkeypatch.setattr(asr_worker, 'run', fake_run)
    out = io.BytesIO()
    assert asr_worker.main(stdin=io.BytesIO(request.to_json() + b'\n'), stdout=out) == 7
    assert seen['request'] == request and out.getvalue() == b'line\n' and seen['kwargs'] == {}


def test_run_defaults_use_the_container_mount_points():
    defaults = asr_worker.run.__kwdefaults__
    assert defaults['scratch'] == Path('/scratch') and defaults['input_root'] == Path('/input')
    assert defaults['model_root'] == Path('/model')
    assert defaults['clock'] is time.monotonic


def test_fixed_error_summaries_are_host_acceptable_for_every_reason():
    request = make_asr_request()
    for reason in (ReasonCode.WORKER_PROTOCOL_MISMATCH, ReasonCode.ASSET_VERIFICATION_FAILED,
                   ReasonCode.MODEL_OOM, ReasonCode.MODEL_LOAD_FAILED, ReasonCode.NO_AUDIO,
                   ReasonCode.MEDIA_PROBE_FAILED, ReasonCode.DURATION_EXCEEDED, ReasonCode.RAM_INSUFFICIENT,
                   ReasonCode.REQUEST_TIMEOUT, ReasonCode.ARTIFACT_REJECTED, ReasonCode.DESTINATION_CONFLICT,
                   ReasonCode.STORAGE_GUARD_TRIPPED, ReasonCode.WORKER_EXIT_UNEXPECTED):
        chunks = []
        emitter = asr_worker._Emitter(request, chunks.append, Clock())
        emitter.emit('started')
        emitter.error(reason)
        stream = WorkerEventStream(request)
        for chunk in chunks:
            stream.feed(chunk)
        stream.finish()


def test_asr_worker_does_not_import_media_modules():
    tree = ast.parse((REPO / 'src' / 'podmuncher' / 'asr_worker.py').read_text(encoding='utf-8'))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(('.' * node.level) + (node.module or ''))
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not imported & {'media_worker', 'direct_download', 'yt_dlp', '.media_worker', '.direct_download'}
    lazy = {'faster_whisper', 'av', 'ctranslate2'}
    top_level = {(getattr(node, 'module', None) or alias.name).split('.')[0] for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not top_level & lazy


# ---------------------------------------------------------------------------
# Probe operation: in-worker runtime facts plus a bounded hold (SPEC_GAP-12)
# ---------------------------------------------------------------------------

PROBE_NAME = 'probe.json'
PROBE_KEYS = ['schema_version', 'runtime', 'compute_types', 'device', 'cuda_device_uuid',
              'faster_whisper_version']
CPU_TYPES = ('int8_float32', 'float32', 'int8', 'int16', 'int8')
GPU_UUID = 'GPU-0123abcd-4567-89ab-cdef-0123456789ab'


def make_probe_request(**changes):
    data = dict(capability='asr_cpu', operation='probe', assigned_slots={'probe_metadata': PROBE_NAME},
                input_basename=None, model_snapshot_id=None,
                selection=dict(device='auto', profile='balanced', model='auto', language='auto'))
    data.update(changes)
    return make_asr_request(**data)


class FakeCTranslate2:
    """Stands in for the lazily imported ctranslate2 module; records every device query."""

    def __init__(self, types=CPU_TYPES, *, version='4.6.0', cuda_count=0, error=None, count_error=None):
        self.__version__ = version
        self.types, self.cuda_count, self.error, self.count_error = types, cuda_count, error, count_error
        self.calls = []

    def get_supported_compute_types(self, device):
        self.calls.append(('types', device))
        if self.error is not None:
            raise self.error
        return set(self.types)

    def get_cuda_device_count(self):
        self.calls.append(('count',))
        if self.count_error is not None:
            raise self.count_error
        return self.cuda_count


class Sleeper:
    """Injected sleep: advances the injected clock and records each requested step."""

    def __init__(self, clock, *, advance=True):
        self.clock, self.advance, self.calls = clock, advance, []

    def __call__(self, seconds):
        self.calls.append(seconds)
        if self.advance:
            self.clock.now += seconds


def run_probe(tmp_path, request=None, *, runtime=None, versions=None, cuda_uuid=None, sleep=None,
              clock=None, **kwargs):
    clock = clock or Clock()
    runtime = runtime or FakeCTranslate2()
    sleep = sleep or Sleeper(clock)
    versions = versions or (lambda name: {'faster-whisper': '1.2.1'}.get(name))
    outcome = execute(tmp_path, request or make_probe_request(), factory=forbidden, decode=forbidden,
                      probe=forbidden, clock=clock, memory_limit=forbidden, peak_rss=forbidden,
                      model_files=(), make_input=False, runtime_module=lambda: runtime,
                      package_version=versions, cuda_uuid=cuda_uuid or (lambda: GPU_UUID),
                      sleep=sleep, **kwargs)
    outcome.runtime, outcome.sleep = runtime, sleep
    return outcome


def test_probe_emits_started_artifact_completed_with_a_bounded_cpu_document(tmp_path):
    outcome = run_probe(tmp_path)
    assert outcome.code == 0 and outcome.events == ['started', 'artifact', 'completed']
    assert all(dict(r.payload) == {} for r in outcome.collector.records if r.event != 'artifact')
    assert outcome.files == [PROBE_NAME]  # worker temp dir removed, nothing else written
    data = (outcome.scratch / PROBE_NAME).read_bytes()
    claim = outcome.collector.of('artifact')[0].payload
    assert dict(claim) == {'slot': 'probe_metadata', 'bytes': len(data),
                           'sha256': hashlib.sha256(data).hexdigest()}
    assert len(data) <= 4096
    document = json.loads(data)
    assert list(document) == PROBE_KEYS
    assert document == {'schema_version': 1, 'runtime': {'name': 'ctranslate2', 'version': '4.6.0'},
                        'compute_types': ['float32', 'int16', 'int8', 'int8_float32'], 'device': 'cpu',
                        'cuda_device_uuid': None, 'faster_whisper_version': '1.2.1'}
    assert outcome.runtime.calls == [('types', 'cpu')]  # CPU never asks about CUDA


def test_probe_cuda_reports_the_cuda_compute_types_and_the_device_uuid(tmp_path):
    runtime = FakeCTranslate2(('float16', 'int8_float16', 'int8'), cuda_count=1)
    request = make_probe_request(capability='asr_cuda',
                                 selection=dict(device='cuda', profile='balanced', model='auto', language='auto'))
    outcome = run_probe(tmp_path, request, runtime=runtime)
    assert outcome.code == 0 and outcome.events == ['started', 'artifact', 'completed']
    document = json.loads((outcome.scratch / PROBE_NAME).read_bytes())
    assert document['device'] == 'cuda' and document['cuda_device_uuid'] == GPU_UUID
    assert document['compute_types'] == ['float16', 'int8', 'int8_float16']
    assert runtime.calls == [('count',), ('types', 'cuda')]


@pytest.mark.parametrize('value', [None, 'not-a-uuid', 'GPU-xyz', 7])
def test_probe_unknown_or_malformed_cuda_uuid_is_null(tmp_path, value):
    runtime = FakeCTranslate2(('float16',), cuda_count=2)
    outcome = run_probe(tmp_path, make_probe_request(capability='asr_cuda'), runtime=runtime,
                        cuda_uuid=lambda: value)
    assert outcome.code == 0
    assert json.loads((outcome.scratch / PROBE_NAME).read_bytes())['cuda_device_uuid'] is None


def test_probe_unknown_faster_whisper_version_is_null(tmp_path):
    for versions in (lambda name: None, lambda name: 'bad version!', forbidden):
        outcome = run_probe(tmp_path / str(len(list(tmp_path.iterdir()))), versions=versions)
        assert outcome.code == 0
        assert json.loads((outcome.scratch / PROBE_NAME).read_bytes())['faster_whisper_version'] is None


def test_probe_holds_for_the_default_twelve_seconds_on_the_injected_clock(tmp_path):
    clock = Clock()
    outcome = run_probe(tmp_path, clock=clock)
    assert outcome.sleep.calls == [0.25] * 48 and all(step <= 0.25 for step in outcome.sleep.calls)
    times = dict(zip(outcome.events, outcome.collector.times))
    assert times['completed'] - times['artifact'] == 12.0
    assert not outcome.collector.of('progress')  # the hold never emits progress


@pytest.mark.parametrize(('hold', 'expected'), [
    (0, 0.0), (-3, 0.0), (1.1, 1.1), (15, 15.0), (100, 15.0), (float('nan'), 12.0), (float('inf'), 12.0)])
def test_probe_hold_is_clamped(tmp_path, hold, expected):
    assert asr_worker.probe_hold_seconds_for(hold) == expected
    clock = Clock()
    outcome = run_probe(tmp_path, clock=clock, probe_hold_seconds=hold)
    times = dict(zip(outcome.events, outcome.collector.times))
    assert outcome.code == 0 and math.isclose(times['completed'] - times['artifact'], expected)


def test_probe_hold_is_bounded_even_when_sleep_does_not_advance_the_clock(tmp_path):
    clock = Clock()
    outcome = run_probe(tmp_path, clock=clock, sleep=Sleeper(clock, advance=False))
    assert outcome.code == 0 and outcome.events[-1] == 'completed'
    assert len(outcome.sleep.calls) == 61


def test_probe_defaults_are_a_twelve_second_hold_with_real_sleep():
    defaults = asr_worker.run.__kwdefaults__
    assert defaults['probe_hold_seconds'] == 12.0 and defaults['sleep'] is time.sleep


def test_probe_ignores_the_model_snapshot_and_never_loads_a_model(tmp_path):
    outcome = run_probe(tmp_path, make_probe_request(model_snapshot_id=SNAPSHOT,
        selection=dict(device='cpu', profile='balanced', model='tiny', language='en')))
    assert outcome.code == 0 and outcome.events == ['started', 'artifact', 'completed']
    assert outcome.files == [PROBE_NAME]


def _probe_wrong_slot():
    return make_probe_request(assigned_slots={'transcript_segments': PROBE_NAME})


def _probe_extra_slot():
    return make_probe_request(assigned_slots={'probe_metadata': PROBE_NAME, 'transcript_segments': 'x.json'})


def _probe_with_input():
    return make_probe_request(input_basename='input.bin')


def _probe_media_capability():
    return make_probe_request(capability='media', input_basename='input.bin')


def _probe_device_mismatch():
    return make_probe_request(selection=dict(device='cuda', profile='balanced', model='auto', language='auto'))


@pytest.mark.parametrize('make', [_probe_wrong_slot, _probe_extra_slot, _probe_with_input,
                                  _probe_media_capability, _probe_device_mismatch])
def test_probe_envelope_violations_are_rejected_without_touching_the_runtime(tmp_path, make):
    runtime = FakeCTranslate2()
    outcome = run_probe(tmp_path, make(), runtime=runtime)
    assert outcome.code == 1 and outcome.events == ['started', 'error']
    assert outcome.error == 'worker_protocol_mismatch'
    assert runtime.calls == [] and outcome.sleep.calls == [] and outcome.files == []


@pytest.mark.parametrize('runtime', [
    FakeCTranslate2(()),
    FakeCTranslate2(error=RuntimeError('no backend')),
    FakeCTranslate2(version=None),
    FakeCTranslate2(version='bad version!'),
    FakeCTranslate2(('int8', 'Not A Type')),
    FakeCTranslate2(tuple(f't{i}' for i in range(33))),
], ids=['empty', 'raises', 'no-version', 'bad-version', 'bad-type', 'too-many'])
def test_probe_runtime_doubt_is_envelope_unvalidated_with_no_artifact(tmp_path, runtime):
    outcome = run_probe(tmp_path, runtime=runtime)
    assert outcome.code == 1 and outcome.events == ['started', 'error']
    assert outcome.error == 'envelope_unvalidated'
    assert outcome.files == [] and outcome.sleep.calls == []


def test_probe_unimportable_runtime_is_envelope_unvalidated(tmp_path):
    def missing():
        raise ImportError('no ctranslate2')
    outcome = execute(tmp_path, make_probe_request(), factory=forbidden, decode=forbidden, probe=forbidden,
                      memory_limit=forbidden, peak_rss=forbidden, model_files=(), make_input=False,
                      runtime_module=missing, sleep=forbidden)
    assert outcome.code == 1 and outcome.error == 'envelope_unvalidated' and outcome.files == []


@pytest.mark.parametrize('runtime', [
    FakeCTranslate2(('float16',), cuda_count=0),
    FakeCTranslate2(('float16',), cuda_count=True),
    FakeCTranslate2(('float16',), count_error=RuntimeError('driver')),
    FakeCTranslate2(cuda_count=1, error=RuntimeError('CUDA failed with error unknown')),
], ids=['no-device', 'bool-count', 'count-raises', 'types-raise'])
def test_probe_never_claims_cuda_when_the_device_query_fails(tmp_path, runtime):
    outcome = run_probe(tmp_path, make_probe_request(capability='asr_cuda'), runtime=runtime)
    assert outcome.code == 1 and outcome.error == 'envelope_unvalidated'
    assert outcome.files == [] and not outcome.collector.of('artifact')


def test_probe_document_over_the_size_cap_is_artifact_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(asr_worker, '_PROBE_MAX_BYTES', 64)
    outcome = run_probe(tmp_path)
    assert outcome.code == 1 and outcome.error == 'artifact_rejected' and outcome.files == []


def test_probe_largest_possible_document_fits_the_size_cap(tmp_path):
    runtime = FakeCTranslate2(tuple(f't{i:031d}' for i in range(32)), version='9' * 64, cuda_count=1)
    outcome = run_probe(tmp_path, make_probe_request(capability='asr_cuda'), runtime=runtime,
                        versions=lambda name: '8' * 64)
    assert outcome.code == 0 and len((outcome.scratch / PROBE_NAME).read_bytes()) <= 4096


def test_probe_existing_destination_is_a_conflict_and_is_kept(tmp_path):
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    (scratch / PROBE_NAME).write_bytes(b'host')
    outcome = run_probe(tmp_path)
    assert outcome.code == 1 and outcome.error == 'destination_conflict'
    assert (scratch / PROBE_NAME).read_bytes() == b'host'


def test_default_cuda_uuid_reads_exactly_one_gpu_from_procfs(tmp_path, monkeypatch):
    gpus = tmp_path / 'gpus'
    monkeypatch.setattr(asr_worker, '_NVIDIA_GPUS', str(gpus))
    assert asr_worker._default_cuda_uuid() is None  # no driver procfs
    one = gpus / '0000-01-00.0'
    one.mkdir(parents=True)
    (one / 'information').write_text('Model: \t X\nGPU UUID: \t ' + GPU_UUID + '\nBus Type: PCIe\n',
                                     encoding='ascii')
    assert asr_worker._default_cuda_uuid() == GPU_UUID
    (one / 'information').write_text('GPU UUID: GPU-bogus\n', encoding='ascii')
    assert asr_worker._default_cuda_uuid() is None
    (gpus / '0000-02-00.0').mkdir()
    assert asr_worker._default_cuda_uuid() is None  # ambiguous: more than one GPU visible


def test_default_package_version_never_raises():
    assert asr_worker._default_package_version('podmuncher-no-such-distribution') is None


# ---------------------------------------------------------------------------
# Closed domain negative examples (Docs/contract-domains.json)
# ---------------------------------------------------------------------------

_NEGATIVE = json.loads((REPO / 'Docs/contract-domains.json').read_text())['negative_examples']

_REJECTORS = {
    'capability': (Capability.parse, lambda bad: make_asr_request(capability=bad)),
    'device': (Device.parse, lambda bad: make_asr_request(
        selection=dict(device=bad, profile='balanced', model='tiny', language='auto'))),
    'backend': (Backend.parse,),
    'model_tier': (ModelTier.parse,),
    'worker_slot': (WorkerSlot.parse, lambda bad: make_asr_request(assigned_slots={bad: 'x.json'})),
    'worker_event': (WorkerEventKind.parse, lambda bad: WorkerEventRecord(1, JOB, ATTEMPT, STAGE, 0, bad, {})),
    'reason_code': (ReasonCode.parse, lambda bad: WorkerEventRecord(
        1, JOB, ATTEMPT, STAGE, 0, 'error', {'reason_code': bad, 'summary': 'x'})),
}


@pytest.mark.parametrize('bad', _NEGATIVE)
@pytest.mark.parametrize('domain', sorted(_REJECTORS))
def test_negative_examples_are_rejected_through_real_constructors(bad, domain):
    for rejector in _REJECTORS[domain]:
        with pytest.raises(DomainValidationError):
            rejector(bad)


def test_exercised_domain_values_are_accepted():
    assert Capability.parse('asr_cpu') is Capability.ASR_CPU and Capability.parse('asr_cuda') is Capability.ASR_CUDA
    assert Device.parse('cpu') is Device.CPU and Device.parse('cuda') is Device.CUDA
    assert Backend.parse('cpu') is Backend.CPU and Backend.parse('cuda') is Backend.CUDA
    assert WorkerSlot.parse('transcript_segments') is WorkerSlot.TRANSCRIPT_SEGMENTS
    for reason in ('model_oom', 'model_load_failed', 'no_audio', 'ram_insufficient', 'request_timeout',
                   'asset_verification_failed', 'artifact_rejected', 'duration_exceeded'):
        assert ReasonCode.parse(reason).value == reason


# ---------------------------------------------------------------------------
# Build inputs: Dockerfile structure, CA secret and build context
# ---------------------------------------------------------------------------

DOCKERFILE = REPO / 'containers' / 'asr.Dockerfile'
_HASHED = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*==[0-9][A-Za-z0-9_.]*(?: --hash=sha256:[0-9a-f]{64})+\Z')
_NVIDIA = {'nvidia-cublas-cu12', 'nvidia-cudnn-cu12', 'nvidia-cuda-nvrtc-cu12'}
# Compiled wheels: (manylinux x86_64, manylinux aarch64) sha256 for the cp312/abi3 wheel of the
# exact pinned version, both verified against https://pypi.org/pypi/<name>/<version>/json.
_MULTI_ARCH = {
    'ctranslate2==4.8.2': ('554a0b2abf098a11c66ef7930c0a05b3de683a455e8ea3f2e6ab6966aaabc136',
                           '781f835da7fdc4adcc38f9d674dbe9b37eaf3c9aee3b2bc17d684342ff59d549'),
    'onnxruntime==1.31.0': ('5e129d6c56abd53e659cb70f00a108d6824086470ff99c2e47a82e5786563db3',
                            '35758d7606d578ec5b9d65f6e8a1f488013194c3f6097038a3223cb26d35ef9a'),
    'PyYAML==6.0.3': ('ba1cc08a7ccde2d2ec775841541641e4548226580ab850948cbfda66a1befcdc',
                      '9149cad251584d5fb4981be1ecde53a1ca46c891a79788c0df828d2f166bda28'),
    'tokenizers==0.23.3': ('376851d22bcf9d650a5c3090bb83e6cf9e895fbf0595369fa4cd43c1f69b5f87',
                           'ec82e80e65a862275b97c3d90b7a523df8d9519ee48aeb4e9625b2cc909274e0'),
    'hf-xet==1.7.0': ('2814a6e999d13464c4d679b788cc5d784eb5a4edfc638a31f10e9a11ab531ef8',
                      'fcfd6c22418e57dd5b3aea649e813b2e2cfb2aebf317b210d90f1fe4b3018b52'),
    'av==19.0.1': ('1bea5b6134209305199bce7627ac3d33964de2cf2b09c77d08e7f67cf8bd4170',
                   '437d4c0d5a7d771f2c3af84cd28e6aac6e173851116c60b53e81dbf1eebe4eab'),
    'numpy==2.5.4': ('fbde6962867ee75b48b0ee29b2b9372ec5d617799dbaf38e82dc0596f2f7738a',
                     'fe4d21ab149f15e4e6043dfb0de87e6e5f34ac176cde83060e9802981fca2ac2'),
    'protobuf==7.36.2': ('89f23aa53c24553a2416fd4fd1ec06f74fa42b14b546d8883128813f775bbfd2',
                         'e11e1f0180583a2af89db6a2ecd9e8dc40aa6d2988ca175bfd0e6d12ea72d74e'),
}


def _dockerfile_lines():
    text = DOCKERFILE.read_text(encoding='utf-8')
    return text, re.sub(r'\\\n', ' ', text).splitlines()


def _requirement_blocks():
    """Heredoc contents of each COPY <<'REQUIREMENTS' block, keyed by destination file."""
    _, lines = _dockerfile_lines()
    blocks, current, name = {}, None, None
    for line in lines:
        if current is None:
            match = re.match(r"COPY <<'REQUIREMENTS' (\S+)\Z", line)
            if match:
                name, current = match.group(1), []
        elif line == 'REQUIREMENTS':
            blocks[name], current = current, None
        else:
            current.append(line)
    assert current is None
    return blocks


def _stage_lines(stage):
    """Instructions that belong to a build stage (the common ``base`` plus its own for targets)."""
    _, lines = _dockerfile_lines()
    stages, current = {}, None
    for line in lines:
        match = re.match(r'FROM (\S+)(?: AS (\w+))?\Z', line)
        if match:
            current = match.group(2)
            stages[current] = {'from': match.group(1), 'lines': []}
        elif current is not None and line.strip():
            stages[current]['lines'].append(line)
    return stages[stage]


def test_both_targets_exist_and_share_the_pinned_base():
    _, lines = _dockerfile_lines()
    stages = {m.group(2): m.group(1) for line in lines
              if (m := re.match(r'FROM (\S+) AS (\w+)\Z', line))}
    assert set(stages) == {'base', 'cpu', 'cuda'}
    assert stages['base'] == 'python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f'
    assert stages['cpu'] == stages['cuda'] == 'base'
    assert lines[0] == '# syntax=docker/dockerfile:1'
    assert not any(re.match(r'(?i)(ADD|ARG)\s', line) for line in lines)
    assert 'apt-get' not in DOCKERFILE.read_text(encoding='utf-8').replace('# ', '')


def test_every_pip_requirement_is_exact_and_hashed():
    blocks = _requirement_blocks()
    assert set(blocks) == {'/opt/asr-requirements.txt', '/opt/asr-cuda-requirements.txt'}
    for entries in blocks.values():
        assert entries and all(_HASHED.fullmatch(entry) for entry in entries)
    base = {entry.split('==')[0] for entry in blocks['/opt/asr-requirements.txt']}
    assert len(blocks['/opt/asr-requirements.txt']) == 25 and len(base) == 25
    assert {'faster-whisper', 'ctranslate2', 'av', 'onnxruntime', 'numpy', 'tokenizers'} <= base
    assert not base & _NVIDIA
    assert {e.split('==')[0] for e in blocks['/opt/asr-cuda-requirements.txt']} == _NVIDIA
    assert 'faster-whisper==1.2.1 --hash=sha256:79a66ad50688c0b794dd501dc340a736992a6342f7f95e5811be60b5224a26a7' \
        in blocks['/opt/asr-requirements.txt']
    for _, lines in [_dockerfile_lines()]:
        pips = [line for line in lines if 'pip install' in line and not line.lstrip().startswith('#')]
        assert len(pips) == 2
        for line in pips:
            assert '--require-hashes' in line and '--no-deps' in line and '--no-cache-dir' in line
            assert '-r /opt/asr-' in line and ' http' not in line and '--index-url' not in line


def test_compiled_pins_carry_x86_64_and_aarch64_hashes():
    """The cpu target is multi-arch: compiled wheels carry both hashes, pure wheels one."""
    blocks = _requirement_blocks()
    hashes = {}
    for entry in blocks['/opt/asr-requirements.txt']:
        pin, *rest = entry.split(' ')
        hashes[pin] = tuple(item.removeprefix('--hash=sha256:') for item in rest)
    for pin, expected in _MULTI_ARCH.items():
        assert hashes[pin] == expected, pin
    assert all(len(found) == 1 for pin, found in hashes.items() if pin not in _MULTI_ARCH)
    # The cuda target is linux/amd64 only: its NVIDIA pins stay single (x86_64) hashes.
    assert all(entry.count('--hash=') == 1 for entry in blocks['/opt/asr-cuda-requirements.txt'])
    text = DOCKERFILE.read_text(encoding='utf-8')
    assert 'linux/arm64' in text and 'linux/amd64 only' in text


def test_dockerfile_references_the_ca_only_through_the_secret_mount():
    text, lines = _dockerfile_lines()
    uses = [line for line in lines if 'build_ca' in line and not line.lstrip().startswith('#')]
    assert len(uses) == 2  # the common pip step and the cuda pip step
    for line in uses:
        assert line.startswith('RUN --mount=type=secret,id=build_ca,required=false ')
        assert line.count('PIP_CERT=/run/secrets/build_ca') == 1 and 'pip install' in line
    for line in lines:
        words = line.split()
        if words and words[0].upper() in ('COPY', 'ADD', 'ENV', 'ARG'):
            assert 'build_ca' not in line and 'PIP_CERT' not in line and '.pem' not in line.lower()
    assert '.pem' not in text.lower() and 'PODMUNCHER_BUILD_CA_FILE' not in text


def test_cpu_and_cuda_targets_run_nonroot_with_the_fixed_entrypoint():
    entry = 'ENTRYPOINT ["python", "-I", "-m", "podmuncher.asr_worker"]'
    for stage, title in (('cpu', 'podmuncher-asr-cpu'), ('cuda', 'podmuncher-asr-cuda')):
        lines = _stage_lines(stage)['lines']
        assert 'USER 65532:65532' in lines and entry in lines
        assert lines[-1] == entry and lines.index('USER 65532:65532') < len(lines) - 1
        joined = ' '.join(lines)
        assert f'org.opencontainers.image.title="{title}"' in joined and 'podmuncher.dev="1"' in joined


def test_cuda_target_installs_the_nvidia_wheels_and_sets_the_library_path():
    cuda = ' '.join(_stage_lines('cuda')['lines'])
    cpu = ' '.join(_stage_lines('cpu')['lines'])
    for name in ('cublas', 'cudnn', 'cuda_nvrtc'):
        assert f'/usr/local/lib/python3.12/site-packages/nvidia/{name}/lib' in cuda
    match = re.search(r'ENV LD_LIBRARY_PATH=(\S+)', cuda)
    assert match and match.group(1).split(':') == [
        f'/usr/local/lib/python3.12/site-packages/nvidia/{n}/lib' for n in ('cublas', 'cudnn', 'cuda_nvrtc')]
    assert 'asr-cuda-requirements.txt' in cuda and 'LD_LIBRARY_PATH' not in cpu and 'nvidia' not in cpu
    assert 'LD_LIBRARY_PATH' not in ' '.join(_stage_lines('base')['lines'])


def test_base_verifies_imports_and_the_vad_asset_at_build_time():
    base = ' '.join(_stage_lines('base')['lines'])
    assert 'import podmuncher.asr_worker, faster_whisper, ctranslate2, av' in base
    assert 'silero_vad' in base and 'assert found' in base and 'python -I -c' in base


def _copy_sources():
    _, lines = _dockerfile_lines()
    copied = set()
    for line in lines:
        words = line.split()
        if words and words[0].upper() == 'COPY' and '<<' not in line:
            copied.update(words[1:-1])
    return copied


def test_dockerignore_limits_the_context_to_the_copied_files():
    copied = _copy_sources()
    assert copied == {'src/podmuncher/' + name for name in (
        '__init__.py', 'domains.py', 'config.py', 'requests.py', 'protocol.py', 'asr_worker.py')}
    path = REPO / 'containers' / 'asr.Dockerfile.dockerignore'
    rules = [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
             if line.strip() and not line.startswith('#')]
    assert rules[0] in ('*', '**') and all(rule.startswith('!') for rule in rules[1:])
    included = {rule[1:] for rule in rules[1:]}
    assert len(included) == len(rules) - 1 and included == copied | {'containers/asr.Dockerfile'}
    assert not any('media_worker' in item or 'direct_download' in item for item in included)
    for excluded in ('HN844', 'research', 'Docs', 'tests'):
        assert not any(item == excluded or item.startswith(excluded + '/') for item in included)


def test_copied_modules_cover_the_asr_worker_import_closure():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import podmuncher.asr_worker; "
            "print(' '.join(sorted(m for m in sys.modules if m.startswith('podmuncher'))))")
    result = subprocess.run((sys.executable, '-c', code, str(REPO / 'src')), shell=False, check=False,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    loaded = set(result.stdout.split())
    copied = {'podmuncher'} | {'podmuncher.' + Path(item).stem for item in _copy_sources()
                               if Path(item).stem != '__init__'}
    assert loaded <= copied, sorted(loaded - copied)
    assert not loaded & {'podmuncher.media_worker', 'podmuncher.direct_download'}


# ---------------------------------------------------------------------------
# Live smoke: admitted cpu image build + ModelCache fixture model + real DockerRuntime
# ---------------------------------------------------------------------------

# The docker CLI: on PATH, else the Docker Desktop for Mac location (absent from PATH in
# non-login ssh shells).
DOCKER = shutil.which('docker') or ('/usr/local/bin/docker' if os.path.isfile('/usr/local/bin/docker')
                                    else 'docker')


@functools.cache
def _target():
    """The current Docker context and its endpoint (Windows npipe or Mac/Linux unix socket).

    Resolved once from ``docker context inspect``, only when the live smoke runs.
    """
    try:
        result = subprocess.run((DOCKER, 'context', 'inspect'), shell=False, check=False,
                                capture_output=True, text=True, encoding='utf-8', errors='replace',
                                timeout=30)
        data, = json.loads(result.stdout)
        return DockerTarget(data['Name'], data['Endpoints']['docker']['Host'])
    except (OSError, subprocess.SubprocessError, ValueError, LookupError, TypeError) as error:
        raise AssertionError('Could not resolve the current Docker context with "docker context inspect"; '
                             'start Docker and select a local context') from error


# Declared build budget: the base image and the Python wheels grow Docker storage by at
# most this much (1 GiB compressed + 2 GiB unpacked = 3 GiB).
BUILD_COMPRESSED_BYTES = 1024**3
BUILD_UNPACKED_BYTES = 2 * 1024**3
BUILD_TIMEOUT_SECONDS = 480
SMOKE_TIMEOUT_SECONDS = 600
SMOKE_OUTPUT_BYTES = 64 * 1024**2
PROBE_OUTPUT_BYTES = 1024**2
MODEL_ADMISSION_BYTES = 160 * 1024**2
_IMAGE_ID = re.compile(r'sha256:[a-f0-9]{64}\Z')
_IMAGE_TAG = re.compile(r'[a-z0-9][a-z0-9._/-]{0,100}:[A-Za-z0-9_][A-Za-z0-9_.-]{0,100}\Z')
FIXTURE_REVISION = 'd90ca5fe260221311c53c58e660288d3deb8d356'
FIXTURE_FILES = (
    ('config.json', 2249, 'a73a28cdfe1c43ccc7202fa333d1f89c202477271407ae9a7f19afa52039cac8'),
    ('model.bin', 75538270, 'dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1'),
    ('tokenizer.json', 2203239, 'fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab'),
    ('vocabulary.txt', 459861, '34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913'),
)


IN_IMAGE_CHECK = (
    'import pathlib, faster_whisper, ctranslate2, av, podmuncher.asr_worker; '
    'a = pathlib.Path(faster_whisper.__file__).parent / "assets"; '
    'v = sorted(p.name for p in a.glob("silero_vad*.onnx")); '
    'assert v, "VAD asset missing"; '
    'print(faster_whisper.__version__, ctranslate2.__version__, av.__version__, v[0])')


IN_IMAGE_DECODE = (
    'from podmuncher.asr_worker import _default_decode; '
    'a = _default_decode("/input/fixture.wav", lambda n: None); '
    'assert a.ndim == 1 and a.dtype.name == "float32" and float(abs(a).max()) <= 1.0; '
    'print("decode samples=%d dtype=%s ndim=%d" % (len(a), a.dtype.name, a.ndim))')


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


class _Audit:
    def __init__(self):
        self.events = []

    def record(self, *, action, reason, asset_id, failure_code):
        self.events.append((action, reason, asset_id, failure_code))


def _docker(*arguments, timeout=60):
    """Bounded argument-array Docker call pinned to the local engine endpoint."""
    return subprocess.run((DOCKER, '--host', _target().endpoint, *arguments), shell=False, check=False,
                          capture_output=True, text=True, encoding='utf-8', errors='replace',
                          timeout=timeout)


def build_arguments(tag, ca_file, *, endpoint, platform=None):
    """Docker build argument array for the cpu target; the build-time CA is a BuildKit secret.

    platform (e.g. 'linux/arm64') is passed as --platform; None builds for the engine's own
    architecture.
    """
    arguments = [DOCKER, '--host', endpoint, 'build', '-f', 'containers/asr.Dockerfile',
                 '--target', 'cpu', '-t', tag]
    if platform is not None:
        assert platform in ('linux/amd64', 'linux/arm64'), 'The cpu target builds for linux/amd64 or linux/arm64'
        arguments += ['--platform', platform]
    if ca_file:
        path = Path(ca_file)
        assert path.is_file(), ('PODMUNCHER_BUILD_CA_FILE is set but does not name an existing file; '
                                'fix the path or unset it')
        arguments += ['--secret', 'id=build_ca,src=' + str(path)]
    return (*arguments, '.')


def test_build_arguments_include_secret_only_when_ca_file_is_configured(tmp_path):
    ca = tmp_path / 'ca.pem'
    ca.write_text('placeholder', encoding='utf-8')
    endpoint = 'unix:///var/run/docker.sock'
    base = (DOCKER, '--host', endpoint, 'build', '-f', 'containers/asr.Dockerfile', '--target', 'cpu',
            '-t', 'podmuncher-asr:test')
    assert build_arguments('podmuncher-asr:test', None, endpoint=endpoint) == (*base, '.')
    assert build_arguments('podmuncher-asr:test', '', endpoint=endpoint) == (*base, '.')
    assert build_arguments('podmuncher-asr:test', str(ca), endpoint=endpoint) == (
        *base, '--secret', 'id=build_ca,src=' + str(ca), '.')
    with pytest.raises(AssertionError, match='PODMUNCHER_BUILD_CA_FILE'):
        build_arguments('podmuncher-asr:test', str(tmp_path / 'missing.pem'), endpoint=endpoint)


def test_build_arguments_pass_an_optional_platform(tmp_path):
    endpoint = 'npipe:////./pipe/dockerDesktopLinuxEngine'
    base = (DOCKER, '--host', endpoint, 'build', '-f', 'containers/asr.Dockerfile', '--target', 'cpu',
            '-t', 'podmuncher-asr:test')
    for platform in ('linux/amd64', 'linux/arm64'):
        assert build_arguments('podmuncher-asr:test', None, endpoint=endpoint, platform=platform) == (
            *base, '--platform', platform, '.')
    with pytest.raises(AssertionError, match='linux/arm64'):
        build_arguments('podmuncher-asr:test', None, endpoint=endpoint, platform='linux/arm/v7')


def test_docker_target_is_resolved_from_the_current_context(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        payload = [{'Name': 'desktop-linux',
                    'Endpoints': {'docker': {'Host': 'unix:///Users/u/.docker/run/docker.sock'}}}]
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr='')

    _target.cache_clear()
    monkeypatch.setattr(subprocess, 'run', fake_run)
    try:
        assert _target() == DockerTarget('desktop-linux', 'unix:///Users/u/.docker/run/docker.sock')
        assert calls == [(DOCKER, 'context', 'inspect')]
        _target.cache_clear()
        monkeypatch.setattr(subprocess, 'run', lambda argv, **kwargs: SimpleNamespace(stdout='[]'))
        with pytest.raises(AssertionError, match='docker context inspect'):
            _target()
    finally:
        _target.cache_clear()


def build_asr_image(tmp_path, tag, platform=None):
    """Admit the declared build budget, then build the cpu target and return the bare image ID.

    The first (cold) build is an explicit admitted prerequisite run once before the timed
    smoke; later runs use the Docker layer cache. Nothing is pushed or given a registry digest.
    platform (None = the engine's native architecture) is passed through to docker build.
    """
    provider = make_docker_storage_provider(timeout=60)
    session = provider.prepare(manual_backing='')
    try:
        assert session.initial_evidence.can_create_cached, session.initial_evidence.reason
        paths = {role: tmp_path / ('build-' + role) for role in ('scratch', 'cache', 'output', 'state')}
        for path in paths.values():
            path.mkdir()
        growth = WorkloadGrowth(BUILD_COMPRESSED_BYTES, BUILD_UNPACKED_BYTES, 0, 0, (8,), 0, 0, 0, 0, (), 0, 0)
        storage = RuntimeStorage(session, provider.volumes, paths=paths, growth=growth, limits=Limits(),
                                 sample_budget_seconds=1.45)
        sampled = asyncio.run(storage.sample())
        assert sampled.plan.admitted, (
            'The declared 3 GiB ASR image build budget was not admitted: ' + sampled.plan.reason.value
            + ' volumes=' + repr(sampled.plan.volumes) + ' intervals=' + repr(sampled.plan.intervals))
    finally:
        session.close()
    began = time.monotonic()
    built = subprocess.run(build_arguments(tag, os.environ.get('PODMUNCHER_BUILD_CA_FILE'),
                                           endpoint=_target().endpoint, platform=platform), cwd=REPO,
                           shell=False, check=False, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=BUILD_TIMEOUT_SECONDS)
    assert built.returncode == 0, 'docker build failed:\n' + '\n'.join(
        (built.stdout + built.stderr).splitlines()[-30:])
    inspected = _docker('image', 'inspect', '--format', '{{.Id}}', tag)
    image_id = inspected.stdout.strip()
    assert inspected.returncode == 0 and _IMAGE_ID.fullmatch(image_id), 'No bare local image ID'
    print(f'build target=cpu seconds={time.monotonic() - began:.1f}')
    return image_id


def _image_proof(image_id):
    def proof():
        result = _docker('image', 'inspect', '--format', '{{.Id}}', image_id, timeout=30)
        return result.returncode == 0 and result.stdout.strip() == image_id
    return proof


def fixture_model():
    """The pinned fixture/smoke model. It is a labelled fixture, never a production envelope."""
    base = 'https://huggingface.co/Systran/faster-whisper-tiny/resolve/' + FIXTURE_REVISION + '/'
    return ManifestModel(
        ModelArtifact('fixture-faster-whisper-tiny', ModelTier.TINY, FIXTURE_REVISION, FIXTURE_FILES[1][2]),
        tuple(base + name for name, _, _ in FIXTURE_FILES),
        tuple(ManifestFile(name, size, digest) for name, size, digest in FIXTURE_FILES),
        'MIT', 'fixture-smoke-asset-not-a-production-envelope', ('int8',),
        1, 0, 1, DECODE_BYTES_PER_SECOND, 1, ('wav',), '2' * 64, False, False)


def acquire_fixture_snapshot(model_root):
    """Admitted ModelCache acquisition; a cached snapshot is re-verified and not downloaded again."""
    cache = ModelCache(
        model_root, audit=_Audit(),
        admission=lambda **values: values['declared_bytes'] <= MODEL_ADMISSION_BYTES
        and values['allocated_bytes'] <= MODEL_ADMISSION_BYTES)  # default trust store: the build CA is build-time only
    try:
        return cache.acquire(fixture_model())
    finally:
        cache.close()


def write_fixture_wav(path, seconds=4):
    """Generated 16 kHz mono PCM: a quiet tone burst between silences (no recordings used)."""
    rate = 16000
    frames = bytearray()
    for index in range(seconds * rate):
        active = rate <= index < 3 * rate
        value = int(3000 * math.sin(2 * math.pi * 440 * index / rate)) if active else 0
        frames += value.to_bytes(2, 'little', signed=True)
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))


def check_result_document(result, *, snapshot_id, source_sha256, attempt_id):
    assert list(result) == RESULT_KEYS and result['schema_version'] == 1
    assert isinstance(result['language'], str) and result['language']
    probability = result['language_probability']
    assert probability is None or (type(probability) is float and math.isfinite(probability))
    duration = result['duration_seconds']
    assert type(duration) is float and math.isfinite(duration) and duration > 0
    assert type(result['no_speech_detected']) is bool
    assert result['no_speech_detected'] == (not result['segments'])
    previous = 0.0
    for segment in result['segments']:
        assert list(segment) == ['start_seconds', 'end_seconds', 'text']
        start, end = segment['start_seconds'], segment['end_seconds']
        assert math.isfinite(start) and math.isfinite(end) and 0 <= start <= end <= duration + 0.5
        assert start >= previous and segment['text'] == segment['text'].strip() and segment['text']
        previous = start
    assert list(result['provenance']) == PROVENANCE_KEYS
    provenance = result['provenance']
    assert provenance == dict(model_id='tiny', model_snapshot_id=snapshot_id, device='cpu',
                              compute_type='int8', source_sha256=source_sha256, attempt_id=attempt_id,
                              image_digest=None)
    measurements = result['measurements']
    assert list(measurements) == MEASUREMENT_KEYS and measurements['peak_vram_bytes'] is None
    for name in ('load_seconds', 'transcribe_wall_seconds', 'audio_seconds'):
        assert math.isfinite(measurements[name]) and measurements[name] >= 0
    assert measurements['audio_seconds'] == duration
    assert type(measurements['peak_rss_bytes']) is int and measurements['peak_rss_bytes'] > 0


async def _leftover_containers(cli, instances):
    left = []
    for instance in instances:
        listed = await cli.run('ps', '-a', '--no-trunc', '--format', '{{.ID}}',
                               '--filter', 'label=podmuncher.instance=' + instance,
                               owner='asr-verify-' + uuid4().hex)
        assert listed.returncode == 0
        left.extend(listed.stdout.split())
    return left


@pytest.mark.live
def test_asr_worker_live_smoke(tmp_path):
    tag = os.environ.get('PODMUNCHER_TEST_ASR_IMAGE')
    model_root_value = os.environ.get('PODMUNCHER_TEST_MODEL_ROOT')
    assert tag and model_root_value, (
        'Both PODMUNCHER_TEST_ASR_IMAGE (a local build tag) and PODMUNCHER_TEST_MODEL_ROOT (a ModelCache '
        'root outside the repository) must be set (load only those lines, plus PODMUNCHER_BUILD_CA_FILE '
        'for the image build when present, from ~/.foreman-mcp/.env; the live smoke never skips)')
    assert _IMAGE_TAG.fullmatch(tag), 'PODMUNCHER_TEST_ASR_IMAGE must be a local name:tag'
    model_root = Path(model_root_value).resolve()
    assert REPO.resolve() not in (model_root, *model_root.parents), 'The model root must be outside the repo'
    model_root.mkdir(parents=True, exist_ok=True)

    image_id = build_asr_image(tmp_path, tag)  # explicit admitted prerequisite, outside the timer
    snapshot = acquire_fixture_snapshot(model_root)  # admitted prerequisite; cached after the first run
    assert snapshot.path.is_dir() and snapshot.snapshot_id.startswith('snapshot-')
    assert sorted(p.name for p in snapshot.path.iterdir()) == sorted(name for name, _, _ in FIXTURE_FILES)

    began = time.monotonic()
    work = tmp_path / 'live'
    work.mkdir()
    scratch = work / 'scratch'
    scratch.mkdir()
    input_file = work / 'fixture.wav'
    write_fixture_wav(input_file)
    source_sha256 = hashlib.sha256(input_file.read_bytes()).hexdigest()
    limits = Limits()
    provider = make_docker_storage_provider(timeout=60)
    session = provider.prepare(manual_backing='')
    cli = NativeDockerCLI(_target())
    instances = []
    policy = WorkerPolicy('asr_cpu', image_id, ('python', '-I', '-m', 'podmuncher.asr_worker'), 0, 0,
                          _image_proof(image_id), development_image=True, allow_pull=False,
                          memory_bytes=2 * 1024**3, pids=256)
    request = make_asr_request(
        model_snapshot_id=snapshot.snapshot_id, input_basename='input.wav', limits=asdict(limits),
        selection=dict(device='cpu', profile='balanced', model='tiny', language='auto'))

    async def run_operation():
        paths = {role: work / ('live-' + role) for role in ('cache', 'output', 'state')}
        for path in paths.values():
            path.mkdir()
        growth = WorkloadGrowth(0, 0, 0, 0, (input_file.stat().st_size,), 0, 0, 0, 0,
                                (SMOKE_OUTPUT_BYTES,), 0, 0)
        storage = RuntimeStorage(session, provider.volumes, paths={**paths, 'scratch': scratch},
                                 growth=growth, limits=limits, sample_budget_seconds=1.45)
        lease = _Lease()
        coordinator = CancellationCoordinator('asr-' + uuid4().hex, lease, stop_grace_seconds=12,
            reap_timeout_seconds=12, callback_timeout_seconds=12)
        guard = WriteGuard(coordinator, storage.sample, byte_limit=SMOKE_OUTPUT_BYTES, timeout_seconds=300,
                           sample_timeout_seconds=2.9)
        runtime = DockerRuntime(cli, guard, instance='asr-' + uuid4().hex, policy=policy,
                                mounts=WorkerMounts(scratch, input_file=input_file, model=snapshot.path))
        instances.append(runtime.instance)
        inspections = []

        async def inspect_running():
            raw = await cli.run('container', 'inspect', runtime.reference.container_id,
                                owner=coordinator.operation_id)
            data, = json.loads(raw.stdout)
            host = data['HostConfig']
            assert host['NetworkMode'] == 'none' and host['ReadonlyRootfs']
            assert data['Config']['User'] == '65532:65532' and host['CapDrop'] == ['ALL']
            mounts = {mount['Destination']: mount for mount in data['Mounts']}
            assert mounts['/model']['RW'] is False and mounts['/input/input.wav']['RW'] is False
            return True

        def consume(item):
            if item.event == 'started':
                inspections.append(coordinator.track(inspect_running()))

        args, _ = runtime.create_arguments(request, 'a' * 32)
        assert args[args.index('--network') + 1] == 'none'
        result = await runtime.run(request, on_event=consume)
        for inspection in inspections:
            assert await inspection
        assert result.completed and result.exit_code == 0 and lease.released == 1 and not lease.held
        return result

    probe_scratch = work / 'probe-scratch'
    probe_scratch.mkdir()
    probe_request = make_probe_request(limits=asdict(limits), job_id=str(uuid4()),
                                       attempt_id=str(uuid4()), stage_id=str(uuid4()))

    async def run_probe_operation():
        """The probe from the same image: network none, no input, no model, alive through the hold."""
        paths = {role: work / ('probe-' + role) for role in ('cache', 'output', 'state')}
        for path in paths.values():
            path.mkdir()
        growth = WorkloadGrowth(0, 0, 0, 0, (0,), 0, 0, 0, 0, (PROBE_OUTPUT_BYTES,), 0, 0)
        storage = RuntimeStorage(session, provider.volumes, paths={**paths, 'scratch': probe_scratch},
                                 growth=growth, limits=limits, sample_budget_seconds=1.45)
        lease = _Lease()
        coordinator = CancellationCoordinator('asr-probe-' + uuid4().hex, lease, stop_grace_seconds=12,
            reap_timeout_seconds=12, callback_timeout_seconds=12)
        guard = WriteGuard(coordinator, storage.sample, byte_limit=PROBE_OUTPUT_BYTES, timeout_seconds=120,
                           sample_timeout_seconds=2.9)
        runtime = DockerRuntime(cli, guard, instance='asr-probe-' + uuid4().hex, policy=policy,
                                mounts=WorkerMounts(probe_scratch))
        instances.append(runtime.instance)
        held, seen = [], {}

        async def inspect_held():
            raw = await cli.run('container', 'inspect', runtime.reference.container_id,
                                owner=coordinator.operation_id)
            data, = json.loads(raw.stdout)
            host = data['HostConfig']
            assert host['NetworkMode'] == 'none' and host['ReadonlyRootfs']
            assert sorted(mount['Destination'] for mount in data['Mounts']) == ['/scratch']
            return data['State']['Running'] is True  # still alive after the artifact: the hold

        def consume(item):
            seen.setdefault(item.event, time.monotonic())
            if item.event == 'artifact':
                held.append(coordinator.track(inspect_held()))

        args, _ = runtime.create_arguments(probe_request, 'b' * 32)
        assert args[args.index('--network') + 1] == 'none'
        result = await runtime.run(probe_request, on_event=consume)
        assert len(held) == 1 and await held[0], 'The probe container was not running during its hold'
        assert result.completed and result.exit_code == 0 and lease.released == 1 and not lease.held
        assert list(seen) == ['started', 'artifact', 'completed'], list(seen)
        return result, seen['completed'] - seen['artifact']

    try:
        result = asyncio.run(run_operation())
        claims = [c.payload for c in result.artifact_claims]
        assert [c['slot'] for c in claims] == ['transcript_segments']
        data = (scratch / SEGMENTS_NAME).read_bytes()
        assert claims[0]['bytes'] == len(data) > 0 and claims[0]['sha256'] == hashlib.sha256(data).hexdigest()
        document = json.loads(data)
        check_result_document(document, snapshot_id=snapshot.snapshot_id, source_sha256=source_sha256,
                              attempt_id=ATTEMPT)
        assert abs(document['duration_seconds'] - 4.0) < 0.05
        assert sorted(p.name for p in scratch.iterdir()) == [SEGMENTS_NAME]  # worker temp dir removed

        probe_result, held_seconds = asyncio.run(run_probe_operation())
        probe_claims = [c.payload for c in probe_result.artifact_claims]
        assert [c['slot'] for c in probe_claims] == ['probe_metadata']
        probe_data = (probe_scratch / PROBE_NAME).read_bytes()
        assert probe_claims[0]['bytes'] == len(probe_data) and 0 < len(probe_data) <= 4096
        assert probe_claims[0]['sha256'] == hashlib.sha256(probe_data).hexdigest()
        assert sorted(p.name for p in probe_scratch.iterdir()) == [PROBE_NAME]
        probe_document = json.loads(probe_data)
        assert list(probe_document) == PROBE_KEYS and probe_document['schema_version'] == 1
        assert probe_document['device'] == 'cpu' and probe_document['cuda_device_uuid'] is None
        assert probe_document['runtime']['name'] == 'ctranslate2'
        compute_types = probe_document['compute_types']
        assert compute_types and compute_types == sorted(set(compute_types))
        assert {'int8', 'float32'} & set(compute_types), compute_types
        assert held_seconds >= 10.0, held_seconds  # default 12 s hold, observed host-side

        removed = asyncio.run(_leftover_containers(cli, instances))
        assert removed == [], 'Labelled worker containers remain'
        checks = subprocess.run((DOCKER, '--host', _target().endpoint, 'run', '--rm', '--pull', 'never',
            '--network', 'none', '--entrypoint', 'python', image_id, '-I', '-c',
            IN_IMAGE_CHECK),
            shell=False, check=False, capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=120)
        assert checks.returncode == 0 and checks.stdout.strip(), checks.stderr[-500:]
        image_versions = checks.stdout.split()
        assert probe_document['runtime']['version'] == image_versions[1], (probe_document, image_versions)
        assert probe_document['faster_whisper_version'] == image_versions[0], (probe_document, image_versions)
        decoded = subprocess.run((DOCKER, '--host', _target().endpoint, 'run', '--rm', '--pull', 'never',
            '--network', 'none', '--read-only', '--user', '65532:65532', '--entrypoint', 'python',
            '-v', str(input_file.resolve()) + ':/input/fixture.wav:ro', image_id, '-I', '-c',
            IN_IMAGE_DECODE),
            shell=False, check=False, capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=120)
        assert decoded.returncode == 0, decoded.stderr[-500:]
        decode_line = decoded.stdout.strip().splitlines()[-1]
        decode_samples = int(decode_line.split('=')[1].split()[0])
        assert abs(decode_samples - 64000) <= 1600 and 'dtype=float32' in decode_line, decode_line
        elapsed = time.monotonic() - began
        assert elapsed < SMOKE_TIMEOUT_SECONDS, 'Live smoke exceeded its declared 600 s bound'
    finally:
        session.close()
        shutil.rmtree(work, ignore_errors=True)
    assert not work.exists()
    measurements = document['measurements']
    print('in-image ' + checks.stdout.strip())
    print('in-image decode ' + decode_line)
    print('evidence fixture-smoke (not a production envelope) ts={} image={} model_revision={} '
          'device=cpu compute=int8 audio_seconds={:.3f} load_seconds={:.3f} '
          'transcribe_wall_seconds={:.3f} peak_rss_bytes={} no_speech={}'.format(
              datetime.now(UTC).isoformat(), image_id[7:19], FIXTURE_REVISION,
              measurements['audio_seconds'], measurements['load_seconds'],
              measurements['transcribe_wall_seconds'], measurements['peak_rss_bytes'],
              document['no_speech_detected']))
    print('probe ' + probe_data.decode('utf-8') + f' held_seconds={held_seconds:.2f}')
    print('cuda: untested (not built in P3-U6)')
    print('PODMUNCHER_SMOKE_OK')
