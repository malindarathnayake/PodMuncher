"""Artifact verification, no-clobber publication and sidecar manifest contracts."""
import asyncio
import errno
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from podmuncher import artifacts
from podmuncher.admission import AdmissionPlan
from podmuncher.artifacts import (
    ArtifactError,
    assign_slots,
    publish,
    verify_artifact,
    write_manifest,
)
from podmuncher.domains import (
    ArtifactSlot,
    DomainValidationError,
    FieldStatus,
    ReasonCode,
    WorkerSlot,
)
from podmuncher.guards import (
    AsyncFileWriter,
    CancellationCoordinator,
    StorageSample,
    WriteGuard,
)
from podmuncher.snapshots import Measurement, Provenance
from podmuncher.volumes import native_adapter

CHUNK_PLUS = artifacts.CHUNK + 17  # forces a multi-chunk streamed hash
JOB = '11111111-1111-4111-8111-111111111111'
OTHER_JOB = '22222222-2222-4222-8222-222222222222'
ATTEMPT = '33333333-3333-4333-8333-333333333333'
CONTRACT = json.loads((Path(__file__).parents[1] / 'Docs' / 'contract-domains.json')
                      .read_text(encoding='utf-8'))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Audit:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, action, artifact_id, reason_code):
        if self.fail:
            raise RuntimeError('sink down')
        self.calls.append((action, artifact_id, reason_code))


def make_attempt_files(tmp_path, files):
    """files: {WorkerSlot: bytes}. Returns (attempt_dir, assignment)."""
    attempt = tmp_path / 'attempt'
    attempt.mkdir(exist_ok=True)
    assignment = assign_slots(str(attempt), files)
    for slot, data in files.items():
        (attempt / assignment[slot]).write_bytes(data)
    return str(attempt), assignment


def verify(attempt, assignment, slot, data, **kwargs):
    kwargs.setdefault('writers_reaped', True)
    kwargs.setdefault('max_bytes', 1 << 20)
    return verify_artifact(attempt, assignment, slot, len(data), sha(data), **kwargs)


def swap_link_hook(attempt, assignment, slot, kind):
    """Swap the verified source for a hard link or symlink (returns False if impossible)."""
    path = Path(attempt) / assignment[slot]
    outside = Path(attempt).parent / 'outside.bin'
    outside.write_bytes(b'attacker')
    if kind == 'hardlink':
        os.link(path, Path(attempt).parent / 'extra-link.bin')
        return True
    path.unlink()
    try:
        os.symlink(outside, path)
    except (OSError, NotImplementedError):
        return False
    return True


def output_root(tmp_path):
    root = tmp_path / 'out'
    root.mkdir(exist_ok=True)
    return str(root)


def job_files(root, job=JOB):
    d = Path(root) / f'podmuncher-{job}'
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def never_same(a, b):
    return False


# --- guard helpers -------------------------------------------------------------------

class Lease:
    def try_acquire(self, operation_id, admission):
        return True

    def release(self, operation_id, *, writers_reaped):
        pass


def ok_plan(reason=ReasonCode.OK):
    okay = reason is ReasonCode.OK
    return AdmissionPlan(Measurement(True if okay else None, 'test',
                                    FieldStatus.OK if okay else FieldStatus.ERROR,
                                    '2026-10-09T12:00:00Z', None if okay else reason), (), ())


async def started_guard(limit=1 << 20, *, trip_after=None):
    calls = []

    async def sample():
        calls.append(1)
        bad = trip_after is not None and len(calls) > trip_after
        return StorageSample(ok_plan(ReasonCode.STORAGE_INSUFFICIENT if bad else ReasonCode.OK), 0.0)

    gate = asyncio.Event()

    async def sleep(seconds):
        await gate.wait()

    coordinator = CancellationCoordinator('op', Lease(), stop_grace_seconds=.05,
                                           reap_timeout_seconds=.05, callback_timeout_seconds=.05)
    guard = WriteGuard(coordinator, sample, byte_limit=limit, timeout_seconds=60,
                       clock=lambda: 0.0, sleep=sleep,
                       cap_reason=ReasonCode.EXPORT_LIMIT_REACHED)
    await guard.start()
    return guard


def reason_of(excinfo):
    return excinfo.value.reason_code


# --- hostile events -------------------------------------------------------------------

@pytest.mark.parametrize('slot', ['/etc/passwd', '../x', 'C:\\x', 'original_video',
                                  'probe_metadata', '', None, 7])
def test_hostile_or_unassigned_slots_rejected_without_reading(tmp_path, monkeypatch, slot):
    data = b'hello'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.DOWNLOADED_MEDIA: data})
    opened = []
    real = artifacts.open_contained
    monkeypatch.setattr(artifacts, 'open_contained',
                        lambda *a, **k: (opened.append(1), real(*a, **k))[1])
    audit = Audit()
    with pytest.raises(ArtifactError) as error:
        verify(attempt, assignment, slot, data, audit=audit)
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED
    assert opened == []
    assert audit.calls == [('reject', None, ReasonCode.ARTIFACT_REJECTED)]


def test_assignment_names_are_private_and_unique(tmp_path):
    a = assign_slots(str(tmp_path), [WorkerSlot.MERGED_VIDEO, WorkerSlot.PROBE_METADATA])
    b = assign_slots(str(tmp_path), [WorkerSlot.MERGED_VIDEO])
    assert set(a) == {WorkerSlot.MERGED_VIDEO, WorkerSlot.PROBE_METADATA}
    assert a[WorkerSlot.MERGED_VIDEO] != b[WorkerSlot.MERGED_VIDEO]
    assert all(n.endswith('.bin') and '/' not in n and '\\' not in n for n in a.values())


def test_tampered_assignment_name_is_rejected(tmp_path):
    attempt, _ = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: b'x'})
    with pytest.raises(ArtifactError) as error:
        verify(attempt, {WorkerSlot.MERGED_VIDEO: '../escape.bin'}, 'merged_video', b'x')
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED


@pytest.mark.parametrize('mutate', ['hash', 'size_short', 'size_long'])
def test_claim_mismatch_rejected(tmp_path, mutate):
    data = b'video-bytes'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    claimed_bytes, claimed_sha = len(data), sha(data)
    if mutate == 'hash':
        claimed_sha = sha(b'other')
    elif mutate == 'size_short':
        claimed_bytes -= 1
    else:
        claimed_bytes += 1
    with pytest.raises(ArtifactError) as error:
        verify_artifact(attempt, assignment, 'merged_video', claimed_bytes, claimed_sha,
                        writers_reaped=True, max_bytes=1 << 20)
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED


def test_writers_not_reaped_is_writer_unresolved(tmp_path):
    data = b'v'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    for flag in (False, None, 1):
        with pytest.raises(ArtifactError) as error:
            verify(attempt, assignment, 'merged_video', data, writers_reaped=flag)
        assert reason_of(error) is ReasonCode.WRITER_UNRESOLVED


def test_verified_file_hides_private_location(tmp_path):
    data = b'v'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, WorkerSlot.MERGED_VIDEO, data)
    assert attempt not in repr(verified) and assignment[WorkerSlot.MERGED_VIDEO] not in repr(verified)
    assert verified.identity.file and verified.bytes == 1


# --- content / caps -------------------------------------------------------------------

def test_transcript_content_checks(tmp_path):
    cases = [(WorkerSlot.TRANSCRIPT_SEGMENTS, b'\xff\xfe bad', ArtifactSlot.TRANSCRIPT_TXT),
             (WorkerSlot.TRANSCRIPT_SEGMENTS, b'{not json', ArtifactSlot.TRANSCRIPT_JSON),
             (WorkerSlot.TRANSCRIPT_SEGMENTS, b'[1, 2]', ArtifactSlot.TRANSCRIPT_JSON),
             (WorkerSlot.MERGED_VIDEO, b'', ArtifactSlot.ORIGINAL_VIDEO)]
    for slot, data, content in cases:
        sub = tmp_path / f'{content.value}-{len(data)}'
        sub.mkdir()
        attempt, assignment = make_attempt_files(sub, {slot: data})
        with pytest.raises(ArtifactError) as error:
            verify(attempt, assignment, slot, data, content=content)
        assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED
    good = b'{"segments": []}'
    (tmp_path / 'good').mkdir()
    attempt, assignment = make_attempt_files(tmp_path / 'good', {WorkerSlot.TRANSCRIPT_SEGMENTS: good})
    assert verify(attempt, assignment, 'transcript_segments', good).content is ArtifactSlot.TRANSCRIPT_JSON


def test_export_and_media_caps(tmp_path):
    data = b'x' * 64
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.TRANSCRIPT_SEGMENTS: b'{"a":"' + data + b'"}',
                                                        WorkerSlot.MERGED_VIDEO: data})
    text = b'{"a":"' + data + b'"}'
    with pytest.raises(ArtifactError) as error:
        verify(attempt, assignment, 'transcript_segments', text, max_bytes=16)
    assert reason_of(error) is ReasonCode.EXPORT_LIMIT_REACHED
    with pytest.raises(ArtifactError) as error:
        verify(attempt, assignment, 'merged_video', data, max_bytes=16)
    assert reason_of(error) is ReasonCode.MEDIA_LIMIT_REACHED


def test_file_larger_than_claim_stops_reading_and_rejects(tmp_path):
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: b'abcdef'})
    with pytest.raises(ArtifactError) as error:
        verify_artifact(attempt, assignment, 'merged_video', 3, sha(b'abc'),
                        writers_reaped=True, max_bytes=1 << 20)
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED


# --- link / swap cases ---------------------------------------------------------------

def test_symlink_in_place_of_file_rejected_at_verify(tmp_path):
    data = b'real'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    if not swap_link_hook(attempt, assignment, WorkerSlot.MERGED_VIDEO, 'symlink'):
        pytest.skip('symlink creation not permitted on this host')
    with pytest.raises(ArtifactError) as error:
        verify(attempt, assignment, 'merged_video', b'attacker')
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED


def test_hard_linked_source_rejected_at_verify(tmp_path):
    data = b'real'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    swap_link_hook(attempt, assignment, WorkerSlot.MERGED_VIDEO, 'hardlink')
    with pytest.raises(ArtifactError) as error:
        verify(attempt, assignment, 'merged_video', data)
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED


def test_junction_attempt_dir_rejected(tmp_path):
    if os.name != 'nt':
        pytest.skip('junctions are Windows-only')
    data = b'real'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    junction = tmp_path / 'junction'
    made = subprocess.run(['cmd', '/c', 'mklink', '/J', str(junction), attempt],
                          capture_output=True, check=False)
    if made.returncode != 0:
        pytest.skip('junction creation not permitted on this host')
    with pytest.raises(ArtifactError) as error:
        verify(str(junction), assignment, 'merged_video', data)
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['hardlink', 'symlink'])
async def test_swap_between_verify_and_publish_rejected(tmp_path, kind):
    data = b'real'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    if not swap_link_hook(attempt, assignment, WorkerSlot.MERGED_VIDEO, kind):
        pytest.skip('symlink creation not permitted on this host')
    root, audit = output_root(tmp_path), Audit()
    with pytest.raises(ArtifactError) as error:
        await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB, output_root=root,
                      guard=None, audit=audit)
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED
    assert job_files(root) == []
    assert audit.calls == [('reject', None, ReasonCode.ARTIFACT_REJECTED)]


@pytest.mark.asyncio
async def test_replacement_by_other_regular_file_rejected(tmp_path):
    data = b'real'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    path = Path(attempt) / assignment[WorkerSlot.MERGED_VIDEO]
    path.unlink()
    path.write_bytes(b'swapped')
    with pytest.raises(ArtifactError) as error:
        await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB,
                      output_root=output_root(tmp_path), guard=None, audit=Audit())
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED


@pytest.mark.asyncio
async def test_in_place_rewrite_after_verification_rejected(tmp_path):
    data = b'real'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    path = Path(attempt) / assignment[WorkerSlot.MERGED_VIDEO]
    before = path.stat().st_mtime_ns
    with open(path, 'r+b') as handle:
        handle.write(b'fake')
    # Coarse timestamp granularity could hide the rewrite; bump mtime explicitly.
    os.utime(path, ns=(before + 1_000_000_000, before + 1_000_000_000))
    root = output_root(tmp_path)
    with pytest.raises(ArtifactError) as error:
        await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB,
                      output_root=root, guard=None, audit=Audit())
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED
    assert job_files(root) == []


@pytest.mark.asyncio
async def test_job_dir_symlink_rejected(tmp_path):
    data = b'real'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    root = output_root(tmp_path)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    try:
        os.symlink(elsewhere, Path(root) / f'podmuncher-{JOB}', target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('symlink creation not permitted on this host')
    with pytest.raises(ArtifactError) as error:
        await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB, output_root=root,
                      guard=None, audit=Audit())
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED
    assert list(elsewhere.iterdir()) == []


# --- publish -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_same_volume_publish_moves_atomically_and_keeps_identity(tmp_path):
    data = b'video-bytes'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    root, audit = output_root(tmp_path), Audit()
    published = await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB,
                              output_root=root, guard=None, audit=audit, video_ext='mkv')
    final = Path(root) / f'podmuncher-{JOB}' / 'video.mkv'
    assert published.file_name == 'video.mkv' and final.read_bytes() == data
    assert published.identity == verified.identity and published.sha256 == sha(data)
    assert not (Path(attempt) / assignment[WorkerSlot.MERGED_VIDEO]).exists()
    assert job_files(root) == ['video.mkv']
    assert audit.calls == [('publish', published.artifact_id, ReasonCode.OK)]
    for call in audit.calls:
        assert str(tmp_path) not in repr(call) and published.sha256 not in repr(call)


@pytest.mark.asyncio
async def test_bad_video_extension_is_invalid_input(tmp_path):
    data = b'v'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    with pytest.raises(ArtifactError) as error:
        await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB, video_ext='../x',
                      output_root=output_root(tmp_path), guard=None, audit=Audit())
    assert reason_of(error) is ReasonCode.INVALID_INPUT


@pytest.mark.asyncio
async def test_slot_content_mismatch_rejected(tmp_path):
    data = b'v'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    with pytest.raises(ArtifactError) as error:
        await publish(verified, ArtifactSlot.TRANSCRIPT_TXT, job_id=JOB,
                      output_root=output_root(tmp_path), guard=None, audit=Audit())
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED


@pytest.mark.asyncio
@pytest.mark.parametrize('cross', [False, True])
async def test_existing_destination_is_conflict_and_untouched(tmp_path, cross):
    data = b'new-bytes'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    root = output_root(tmp_path)
    job_dir = Path(root) / f'podmuncher-{JOB}'
    job_dir.mkdir()
    (job_dir / 'video.mp4').write_bytes(b'precious')
    guard = await started_guard() if cross else None
    try:
        audit = Audit()
        with pytest.raises(ArtifactError) as error:
            await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB, output_root=root,
                          guard=guard, audit=audit,
                          same_volume=never_same if cross else None)
    finally:
        if guard:
            await guard.finish()
    assert reason_of(error) is ReasonCode.DESTINATION_CONFLICT
    assert (job_dir / 'video.mp4').read_bytes() == b'precious'
    assert job_files(root) == ['video.mp4']
    assert (Path(attempt) / assignment[WorkerSlot.MERGED_VIDEO]).read_bytes() == data
    assert audit.calls == [('reject', None, ReasonCode.DESTINATION_CONFLICT)]


@pytest.mark.asyncio
async def test_cross_volume_copy_validate_rename(tmp_path):
    data = os.urandom(3000)
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    root = output_root(tmp_path)
    guard = await started_guard()
    try:
        published = await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB,
                                  output_root=root, guard=guard, audit=Audit(),
                                  same_volume=never_same)
    finally:
        await guard.finish()
    assert (Path(root) / f'podmuncher-{JOB}' / 'video.mp4').read_bytes() == data
    assert guard.bytes_written == len(data)
    assert job_files(root) == ['video.mp4']
    # Scratch is kept until the caller deletes the scratch dir.
    assert (Path(attempt) / assignment[WorkerSlot.MERGED_VIDEO]).read_bytes() == data
    assert published.bytes == len(data)


def _stat_key(path):
    st = os.stat(path, follow_symlinks=False)
    return st.st_ino, st.st_mtime_ns, st.st_size


async def _publish_maybe_cross(verified, root, cross, audit, job=JOB):
    guard = await started_guard(limit=1 << 22) if cross else None
    try:
        return await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=job, output_root=root,
                             guard=guard, audit=audit, same_volume=never_same if cross else None)
    finally:
        if guard:
            await guard.finish()


@pytest.mark.asyncio
@pytest.mark.parametrize('cross', [False, True])
async def test_identical_orphan_destination_is_adopted(tmp_path, cross):
    data = os.urandom(CHUNK_PLUS)
    root = output_root(tmp_path)
    # Reference: a fresh publication of the same bytes into another job.
    (tmp_path / 'ref').mkdir()
    ref_attempt, ref_assignment = make_attempt_files(tmp_path / 'ref', {WorkerSlot.MERGED_VIDEO: data})
    fresh = await _publish_maybe_cross(verify(ref_attempt, ref_assignment, 'merged_video', data,
                                              max_bytes=1 << 22),
                                       root, cross, Audit(), job=OTHER_JOB)
    # Orphan: an earlier publication whose durable claim was lost.
    job_dir = Path(root) / f'podmuncher-{JOB}'
    job_dir.mkdir()
    orphan = job_dir / 'video.mp4'
    orphan.write_bytes(data)
    os.utime(orphan, ns=(1_000_000_000, 1_000_000_000))
    before = _stat_key(orphan)
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data, max_bytes=1 << 22)
    audit = Audit()
    adopted = await _publish_maybe_cross(verified, root, cross, audit)
    assert (adopted.slot, adopted.format, adopted.bytes, adopted.sha256, adopted.file_name) == (
        fresh.slot, fresh.format, fresh.bytes, fresh.sha256, fresh.file_name)
    assert adopted.artifact_id != fresh.artifact_id
    assert adopted.to_artifact(ATTEMPT, None) == adopted.to_artifact(ATTEMPT, None)
    assert type(adopted) is type(fresh)
    assert _stat_key(orphan) == before and orphan.read_bytes() == data
    assert job_files(root) == ['video.mp4']  # no partial temp left behind
    scratch = Path(attempt) / assignment[WorkerSlot.MERGED_VIDEO]
    # Same volume discards the scratch like a move; cross volume keeps it for the caller.
    assert scratch.exists() is cross
    assert audit.calls == [('publish', adopted.artifact_id, ReasonCode.OK)]


def _hostile_destination(kind, job_dir, data, tmp_path):
    """Plant a non-adoptable destination; returns a checker for 'untouched'."""
    target = job_dir / 'video.mp4'
    if kind == 'different_size':
        target.write_bytes(data + b'!')
    elif kind == 'same_size':
        target.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    elif kind == 'hardlink':
        outside = tmp_path / 'outside-hard.bin'
        outside.write_bytes(data)
        os.link(outside, target)
    elif kind == 'symlink':
        outside = tmp_path / 'outside-sym.bin'
        outside.write_bytes(data)
        try:
            os.symlink(outside, target)
        except (OSError, NotImplementedError):
            pytest.skip('symlink creation not permitted on this host')
    elif kind == 'directory':
        target.mkdir()
        (target / 'inner').write_bytes(data)
    elif kind == 'junction':
        if os.name != 'nt':
            pytest.skip('junctions are Windows-only')
        elsewhere = tmp_path / 'junction-target'
        elsewhere.mkdir()
        made = subprocess.run(['cmd', '/c', 'mklink', '/J', str(target), str(elsewhere)],
                              capture_output=True, check=False)
        if made.returncode != 0:
            pytest.skip('junction creation not permitted on this host')
    if target.is_dir():
        snapshot = sorted(p.name for p in target.iterdir())
        return lambda: sorted(p.name for p in target.iterdir()) == snapshot
    key, content = _stat_key(target), target.read_bytes()
    return lambda: _stat_key(target) == key and target.read_bytes() == content


@pytest.mark.asyncio
@pytest.mark.parametrize('cross', [False, True])
@pytest.mark.parametrize('kind', ['different_size', 'same_size', 'hardlink', 'symlink',
                                  'directory', 'junction'])
async def test_non_identical_destination_is_never_adopted(tmp_path, cross, kind):
    data = b'retry-bytes-' * 100
    root = output_root(tmp_path)
    job_dir = Path(root) / f'podmuncher-{JOB}'
    job_dir.mkdir()
    untouched = _hostile_destination(kind, job_dir, data, tmp_path)
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    audit = Audit()
    with pytest.raises(ArtifactError) as error:
        await _publish_maybe_cross(verified, root, cross, audit)
    assert reason_of(error) is ReasonCode.DESTINATION_CONFLICT
    assert untouched()
    assert job_files(root) == ['video.mp4']
    assert (Path(attempt) / assignment[WorkerSlot.MERGED_VIDEO]).read_bytes() == data
    assert audit.calls == [('reject', None, ReasonCode.DESTINATION_CONFLICT)]


@pytest.mark.asyncio
@pytest.mark.parametrize('cross', [False, True])
async def test_identical_destination_on_other_volume_is_conflict(tmp_path, cross, monkeypatch):
    data = b'same-bytes'
    root = output_root(tmp_path)
    job_dir = Path(root) / f'podmuncher-{JOB}'
    job_dir.mkdir()
    (job_dir / 'video.mp4').write_bytes(data)
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    real = artifacts._adopt_existing

    def foreign_volume(job_path, name, verified, job_volume, adapter):
        return real(job_path, name, verified, job_volume + '-elsewhere', adapter)

    monkeypatch.setattr(artifacts, '_adopt_existing', foreign_volume)
    with pytest.raises(ArtifactError) as error:
        await _publish_maybe_cross(verified, root, cross, Audit())
    assert reason_of(error) is ReasonCode.DESTINATION_CONFLICT
    assert (job_dir / 'video.mp4').read_bytes() == data
    assert (Path(attempt) / assignment[WorkerSlot.MERGED_VIDEO]).read_bytes() == data
    assert job_files(root) == ['video.mp4']


@pytest.mark.asyncio
@pytest.mark.parametrize('cross', [False, True])
async def test_adoption_hashes_off_the_event_loop(tmp_path, cross, monkeypatch):
    data = os.urandom(CHUNK_PLUS)
    root = output_root(tmp_path)
    job_dir = Path(root) / f'podmuncher-{JOB}'
    job_dir.mkdir()
    (job_dir / 'video.mp4').write_bytes(data)
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data, max_bytes=1 << 22)
    loop_thread, seen = threading.get_ident(), []
    real = artifacts._adopt_existing

    def recording(*args):
        seen.append(threading.get_ident())
        return real(*args)

    monkeypatch.setattr(artifacts, '_adopt_existing', recording)
    published = await _publish_maybe_cross(verified, root, cross, Audit())
    assert published.sha256 == hashlib.sha256(data).hexdigest()
    assert len(seen) == 1 and seen[0] != loop_thread


COPY_FINALIZERS = ('_flush_close', '_hash_held', '_no_clobber_move', '_prove_source',
                   '_prove_destination', '_prove_job_volume', '_sync_directory')


@pytest.mark.asyncio
async def test_copy_path_finalization_runs_off_the_event_loop(tmp_path, monkeypatch):
    data = os.urandom(CHUNK_PLUS)
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data, max_bytes=1 << 22)
    root = output_root(tmp_path)
    loop_thread, seen = threading.get_ident(), {name: [] for name in COPY_FINALIZERS}
    for name in COPY_FINALIZERS:
        real = getattr(artifacts, name)

        def recording(*args, _name=name, _real=real):
            seen[_name].append(threading.get_ident())
            return _real(*args)
        monkeypatch.setattr(artifacts, name, recording)
    published = await _publish_maybe_cross(verified, root, True, Audit())
    assert published.sha256 == sha(data)
    assert (Path(root) / f'podmuncher-{JOB}' / 'video.mp4').read_bytes() == data
    assert {name: len(idents) for name, idents in seen.items()} == dict.fromkeys(COPY_FINALIZERS, 1)
    assert all(idents[0] != loop_thread for idents in seen.values())


@pytest.mark.asyncio
async def test_guard_trip_after_last_chunk_never_publishes(tmp_path, monkeypatch):
    data = b'late-trip-bytes' * 500
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    root = output_root(tmp_path)
    guard = await started_guard()
    loop, real_hash, moves = asyncio.get_running_loop(), artifacts._hash_held, []

    def trip_then_hash(*args):
        # Every chunk has been accepted by guard.write; the storage watchdog
        # stops the operation while the finished copy is being re-hashed.
        loop.call_soon_threadsafe(guard.coordinator.request_stop,
                                  ReasonCode.STORAGE_GUARD_TRIPPED)
        return real_hash(*args)

    monkeypatch.setattr(artifacts, '_hash_held', trip_then_hash)
    monkeypatch.setattr(artifacts, '_no_clobber_move', lambda *args: moves.append(args))
    audit = Audit()
    try:
        with pytest.raises(ArtifactError) as error:
            await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB, output_root=root,
                          guard=guard, audit=audit, same_volume=never_same)
    finally:
        result = await guard.finish()
    assert reason_of(error) is ReasonCode.STORAGE_GUARD_TRIPPED
    assert result.reason is ReasonCode.STORAGE_GUARD_TRIPPED and result.writers_reaped
    assert guard.bytes_written == len(data)  # the whole copy was written before the trip
    assert moves == []  # the rename was never attempted
    assert job_files(root) == []  # no final file and no .partial temp
    assert (Path(attempt) / assignment[WorkerSlot.MERGED_VIDEO]).read_bytes() == data
    assert audit.calls == [('reject', None, ReasonCode.STORAGE_GUARD_TRIPPED)]


@pytest.mark.asyncio
async def test_guard_trip_before_same_volume_move_never_publishes(tmp_path, monkeypatch):
    data = b'move-trip'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    root = output_root(tmp_path)
    guard = await started_guard(trip_after=1)  # the pre-move sample trips
    moves = []
    monkeypatch.setattr(artifacts, '_no_clobber_move', lambda *args: moves.append(args))
    try:
        with pytest.raises(ArtifactError) as error:
            await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB, output_root=root,
                          guard=guard, audit=Audit())
    finally:
        await guard.finish()
    assert reason_of(error) is ReasonCode.STORAGE_GUARD_TRIPPED
    assert moves == [] and job_files(root) == []
    assert (Path(attempt) / assignment[WorkerSlot.MERGED_VIDEO]).read_bytes() == data


@pytest.mark.asyncio
@pytest.mark.parametrize('cross', [False, True])
async def test_rejection_after_adoption_keeps_scratch_source(tmp_path, cross, monkeypatch):
    data = b'adopt-then-reject' * 50
    root = output_root(tmp_path)
    job_dir = Path(root) / f'podmuncher-{JOB}'
    job_dir.mkdir()
    orphan = job_dir / 'video.mp4'
    orphan.write_bytes(data)
    before = _stat_key(orphan)
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    adoptions = []
    real = artifacts._adopt_existing

    def counting(*args):
        identity = real(*args)
        adoptions.append(identity)
        return identity

    def destination_changed(*args):
        raise ArtifactError(ReasonCode.DESTINATION_CONFLICT)

    monkeypatch.setattr(artifacts, '_adopt_existing', counting)
    monkeypatch.setattr(artifacts, '_prove_destination', destination_changed)
    audit = Audit()
    with pytest.raises(ArtifactError) as error:
        await _publish_maybe_cross(verified, root, cross, audit)
    assert len(adoptions) == 1  # adoption itself succeeded; the final check rejected
    assert reason_of(error) is ReasonCode.DESTINATION_CONFLICT
    assert (Path(attempt) / assignment[WorkerSlot.MERGED_VIDEO]).read_bytes() == data
    assert _stat_key(orphan) == before and orphan.read_bytes() == data
    assert job_files(root) == ['video.mp4']
    assert audit.calls == [('reject', None, ReasonCode.DESTINATION_CONFLICT)]


class FaultyWriter(AsyncFileWriter):
    after, code = 10, errno.EIO

    async def write(self, data):
        if self._handle.tell() + len(data) > self.after:
            part = max(self.after - self._handle.tell(), 0)
            if part:
                self._handle.write(data[:part])
            raise OSError(self.code, 'simulated')
        return await super().write(data)


async def two_publishes(tmp_path, monkeypatch, code=None, trip_after=None):
    first, second = b'transcript one', b'x' * 5000
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.TRANSCRIPT_SEGMENTS: first,
                                                        WorkerSlot.MERGED_VIDEO: second})
    root = output_root(tmp_path)
    ok = verify(attempt, assignment, 'transcript_segments', first, content=ArtifactSlot.TRANSCRIPT_TXT)
    guard = await started_guard()
    published = await publish(ok, ArtifactSlot.TRANSCRIPT_TXT, job_id=JOB, output_root=root,
                              guard=guard, audit=Audit(), same_volume=never_same)
    await guard.finish()
    video = verify(attempt, assignment, 'merged_video', second)
    guard = await started_guard(trip_after=trip_after)
    if code is not None:
        FaultyWriter.code = code
        monkeypatch.setattr(artifacts, 'AsyncFileWriter', FaultyWriter)
    try:
        with pytest.raises(ArtifactError) as error:
            await publish(video, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB, output_root=root,
                          guard=guard, audit=Audit(), same_volume=never_same)
    finally:
        await guard.finish()
    prior = Path(root) / f'podmuncher-{JOB}' / 'transcript.txt'
    assert prior.read_bytes() == first and published.file_name == 'transcript.txt'
    assert job_files(root) == ['transcript.txt']
    assert (Path(attempt) / assignment[WorkerSlot.MERGED_VIDEO]).read_bytes() == second
    return reason_of(error)


@pytest.mark.asyncio
async def test_partial_copy_fault_leaves_no_temp(tmp_path, monkeypatch):
    assert await two_publishes(tmp_path, monkeypatch, code=errno.EIO) is ReasonCode.PUBLICATION_FAILED


@pytest.mark.asyncio
async def test_enospc_through_guard_trips_storage_guard(tmp_path, monkeypatch):
    assert await two_publishes(tmp_path, monkeypatch, code=errno.ENOSPC) is ReasonCode.STORAGE_GUARD_TRIPPED


@pytest.mark.asyncio
async def test_quota_maps_to_quota_exceeded(tmp_path, monkeypatch):
    assert await two_publishes(tmp_path, monkeypatch, code=getattr(errno, 'EDQUOT', 122)) is ReasonCode.QUOTA_EXCEEDED


@pytest.mark.asyncio
async def test_guard_storage_sample_failure_before_copy(tmp_path, monkeypatch):
    assert await two_publishes(tmp_path, monkeypatch, trip_after=1) is ReasonCode.STORAGE_GUARD_TRIPPED


@pytest.mark.asyncio
async def test_extra_budget_cap_is_export_limit(tmp_path):
    data = b'y' * 400
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    root = output_root(tmp_path)
    guard = await started_guard(limit=100)
    try:
        with pytest.raises(ArtifactError) as error:
            await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB, output_root=root,
                          guard=guard, audit=Audit(), same_volume=never_same)
    finally:
        await guard.finish()
    assert reason_of(error) is ReasonCode.EXPORT_LIMIT_REACHED
    assert job_files(root) == []


@pytest.mark.asyncio
async def test_cross_volume_without_started_guard_fails_closed(tmp_path):
    data = b'y'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    root = output_root(tmp_path)
    guard = await started_guard()
    await guard.finish()
    with pytest.raises(ArtifactError) as error:
        await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB, output_root=root,
                      guard=guard, audit=Audit(), same_volume=never_same)
    assert reason_of(error) is ReasonCode.PUBLICATION_FAILED
    assert job_files(root) == []


@pytest.mark.asyncio
async def test_audit_failure_is_audit_loss(tmp_path):
    data = b'v'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    root = output_root(tmp_path)
    with pytest.raises(ArtifactError) as error:
        await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB,
                      output_root=root, guard=None, audit=Audit(fail=True))
    assert reason_of(error) is ReasonCode.AUDIT_LOSS
    published = [p for p in Path(root).rglob('*') if p.is_file()]
    assert [p.read_bytes() for p in published] == [data]
    (tmp_path / 'again').mkdir()
    attempt, assignment = make_attempt_files(tmp_path / 'again', {WorkerSlot.MERGED_VIDEO: data})
    with pytest.raises(ArtifactError) as error:
        verify(attempt, assignment, 'merged_video', data, audit=Audit(fail=True))
    assert reason_of(error) is ReasonCode.AUDIT_LOSS
    assert len([p for p in Path(root).rglob('*') if p.is_file()]) == 1


@pytest.mark.asyncio
async def test_audit_failure_on_rejection_keeps_original_reason(tmp_path):
    data = b'v'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.MERGED_VIDEO: data})
    verified = verify(attempt, assignment, 'merged_video', data)
    root = output_root(tmp_path)
    await publish(verified, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB,
                  output_root=root, guard=None, audit=Audit())
    attempt2 = tmp_path / 'second'
    attempt2.mkdir()
    other = b'w'  # identical bytes would be adopted, so the retry must differ
    attempt2, assignment2 = make_attempt_files(attempt2, {WorkerSlot.MERGED_VIDEO: other})
    verified2 = verify(attempt2, assignment2, 'merged_video', other)
    with pytest.raises(ArtifactError) as error:
        await publish(verified2, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB,
                      output_root=root, guard=None, audit=Audit(fail=True))
    assert reason_of(error) is ReasonCode.DESTINATION_CONFLICT


def test_error_message_has_no_path():
    error = ArtifactError(ReasonCode.DESTINATION_CONFLICT)
    assert str(error) == 'Artifact operation failed.'


# --- manifest -------------------------------------------------------------------------

async def published_pair(tmp_path):
    texts = b'hello'
    attempt, assignment = make_attempt_files(tmp_path, {WorkerSlot.TRANSCRIPT_SEGMENTS: texts})
    verified = verify(attempt, assignment, 'transcript_segments', texts,
                      content=ArtifactSlot.TRANSCRIPT_TXT)
    root = output_root(tmp_path)
    item = await publish(verified, ArtifactSlot.TRANSCRIPT_TXT, job_id=JOB, output_root=root,
                         guard=None, audit=Audit())
    return str(Path(root) / f'podmuncher-{JOB}'), item


@pytest.mark.asyncio
async def test_manifest_written_and_replaced_only_for_own_job(tmp_path):
    job_dir, item = await published_pair(tmp_path)
    provenance = Provenance(model_id='m', attempt_id=ATTEMPT)
    assert write_manifest(job_dir, JOB, ATTEMPT, [item], provenance) == 'podmuncher-manifest.json'
    doc = json.loads((Path(job_dir) / 'podmuncher-manifest.json').read_text('utf-8'))
    assert doc['schema_version'] == 1 and doc['job_id'] == JOB
    assert doc['artifacts'] == [{'artifact_id': item.artifact_id, 'slot': 'transcript_txt',
                                 'format': 'txt', 'bytes': 5, 'sha256': sha(b'hello'),
                                 'file_name': 'transcript.txt'}]
    assert str(tmp_path) not in json.dumps(doc)
    write_manifest(job_dir, JOB, ATTEMPT, [], None)
    assert json.loads((Path(job_dir) / 'podmuncher-manifest.json').read_text('utf-8'))['artifacts'] == []
    assert sorted(p.name for p in Path(job_dir).iterdir()) == ['podmuncher-manifest.json', 'transcript.txt']


@pytest.mark.asyncio
@pytest.mark.parametrize('existing', [
    b'not json', b'[1]', b'{"schema_version": 2, "job_id": "%s"}' % JOB.encode(),
    b'{"schema_version": 1, "job_id": "%s"}' % OTHER_JOB.encode()])
async def test_foreign_or_invalid_manifest_is_conflict(tmp_path, existing):
    job_dir, item = await published_pair(tmp_path)
    target = Path(job_dir) / 'podmuncher-manifest.json'
    target.write_bytes(existing)
    with pytest.raises(ArtifactError) as error:
        write_manifest(job_dir, JOB, ATTEMPT, [item], None)
    assert reason_of(error) is ReasonCode.DESTINATION_CONFLICT
    assert target.read_bytes() == existing
    assert sorted(p.name for p in Path(job_dir).iterdir()) == ['podmuncher-manifest.json', 'transcript.txt']


@pytest.mark.asyncio
async def test_manifest_symlink_is_conflict(tmp_path):
    job_dir, item = await published_pair(tmp_path)
    outside = tmp_path / 'outside.json'
    outside.write_text(json.dumps({'schema_version': 1, 'job_id': JOB}))
    try:
        os.symlink(outside, Path(job_dir) / 'podmuncher-manifest.json')
    except (OSError, NotImplementedError):
        pytest.skip('symlink creation not permitted on this host')
    with pytest.raises(ArtifactError) as error:
        write_manifest(job_dir, JOB, ATTEMPT, [item], None)
    assert reason_of(error) is ReasonCode.DESTINATION_CONFLICT


# --- domains --------------------------------------------------------------------------

@pytest.mark.parametrize('enum_name, enum', [('artifact_slot', ArtifactSlot),
                                             ('worker_slot', WorkerSlot)])
def test_domains_reject_contract_negatives_through_real_parse(enum_name, enum):
    for value in CONTRACT['negative_examples']:
        with pytest.raises(DomainValidationError) as error:
            enum.parse(value)
        assert error.value.reason_code is ReasonCode.INVALID_INPUT
    assert [m.value for m in enum] == CONTRACT[enum_name]


# --- live smoke -----------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.asyncio
async def test_own_transport_smoke(tmp_path, capsys):
    adapter = native_adapter()
    root = output_root(tmp_path)
    audit = Audit()
    video = os.urandom(70000)
    texts = {ArtifactSlot.TRANSCRIPT_TXT: b'plain text',
             ArtifactSlot.TRANSCRIPT_SRT: b'1\n00:00:00,000 --> 00:00:01,000\nhi\n',
             ArtifactSlot.TRANSCRIPT_JSON: b'{"segments": []}'}
    files = {WorkerSlot.MERGED_VIDEO: video, WorkerSlot.TRANSCRIPT_SEGMENTS: b'{"s": 1}',
             WorkerSlot.PROBE_METADATA: b'{"p": 1}', WorkerSlot.DOWNLOADED_AUDIO: b'audio!'}
    attempt, assignment = make_attempt_files(tmp_path, files)
    published = []
    v = verify(attempt, assignment, 'merged_video', video, adapter=adapter, audit=audit)
    published.append(await publish(v, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB, output_root=root,
                                   guard=None, audit=audit, adapter=adapter))
    # transcript set, same volume (txt + srt + json) via distinct scratch files
    extra = {}
    for index, (slot, data) in enumerate(texts.items()):
        scratch = Path(attempt) / f'{WorkerSlot.TRANSCRIPT_SEGMENTS.value}-{index:032x}.bin'
        scratch.write_bytes(data)
        extra[slot] = (scratch.name, data)
    for slot, (name, data) in extra.items():
        v = verify(attempt, {WorkerSlot.TRANSCRIPT_SEGMENTS: name}, 'transcript_segments', data,
                   content=slot, adapter=adapter, audit=audit)
        if slot is ArtifactSlot.TRANSCRIPT_SRT:
            guard = await started_guard()
            try:  # forced copy exercises the cross-volume path on the real filesystem
                published.append(await publish(v, slot, job_id=JOB, output_root=root, guard=guard,
                                               audit=audit, adapter=adapter, same_volume=never_same))
            finally:
                await guard.finish()
        else:
            published.append(await publish(v, slot, job_id=JOB, output_root=root, guard=None,
                                           audit=audit, adapter=adapter))
    job_dir = Path(root) / f'podmuncher-{JOB}'
    assert sorted(p.name for p in job_dir.iterdir()) == [
        'transcript.json', 'transcript.srt', 'transcript.txt', 'video.mp4']
    # conflicting publish
    again = verify(attempt, {WorkerSlot.DOWNLOADED_AUDIO: assignment[WorkerSlot.DOWNLOADED_AUDIO]},
                   'downloaded_audio', b'audio!', adapter=adapter)
    with pytest.raises(ArtifactError) as error:
        await publish(again, ArtifactSlot.ORIGINAL_VIDEO, job_id=JOB, output_root=root,
                      guard=None, audit=audit, adapter=adapter)
    assert reason_of(error) is ReasonCode.DESTINATION_CONFLICT
    assert (job_dir / 'video.mp4').read_bytes() == video
    # hard-link swap between verify and publish
    probe = verify(attempt, assignment, 'probe_metadata', b'{"p": 1}', adapter=adapter)
    swap_link_hook(attempt, assignment, WorkerSlot.PROBE_METADATA, 'hardlink')
    with pytest.raises(ArtifactError) as error:
        await publish(probe, ArtifactSlot.TRANSCRIPT_READABLE_TXT, job_id=JOB, output_root=root,
                      guard=None, audit=audit, adapter=adapter)
    assert reason_of(error) is ReasonCode.ARTIFACT_REJECTED
    assert not (job_dir / 'transcript.readable.txt').exists()
    # manifest
    write_manifest(str(job_dir), JOB, ATTEMPT, published, Provenance(attempt_id=ATTEMPT), adapter=adapter)
    doc = json.loads((job_dir / 'podmuncher-manifest.json').read_text('utf-8'))
    assert len(doc['artifacts']) == 4 and doc['job_id'] == JOB
    assert {a['file_name'] for a in doc['artifacts']} == {
        'video.mp4', 'transcript.txt', 'transcript.srt', 'transcript.json'}
    assert not any(str(tmp_path) in repr(call) for call in audit.calls)
    # clean everything
    for path in sorted(tmp_path.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        path.unlink() if path.is_file() or path.is_symlink() else path.rmdir()
    assert list(tmp_path.iterdir()) == []
    with capsys.disabled():
        print('PODMUNCHER_SMOKE_OK')
