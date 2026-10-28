"""Host artifact verification and no-clobber publication.

Worker events are never trusted: only host-assigned logical slots select a
private scratch basename, and every byte is re-verified through a held,
no-follow handle after the writer has been reaped. Publication never overwrites
an existing destination; a conflict is reported and the validated scratch file
is kept for an explicit retry. A retry that finds its own byte-identical earlier
publication (proved through a held, no-follow handle on the job volume) adopts it
instead, so an orphan left by an interrupted claim cannot block recovery. Paths, names and hashes are protected evidence:
errors carry only a closed reason code and audit arguments never contain them.

POSIX cannot atomically bind a rename to an inode, so the source identity is
re-proved immediately before the move and the destination identity immediately
after it. A swap in that window is reported as rejected and the destination is
left in place for the operator (it is never silently deleted).
"""
from __future__ import annotations

import asyncio
import codecs
import errno
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from .domains import (
    ArtifactFormat,
    ArtifactSlot,
    DomainValidationError,
    ReasonCode,
    WorkerSlot,
    validate_uuid_string,
)
from .guards import AsyncFileWriter, GuardError, WriteGuard
from .snapshots import Artifact, Provenance
from .volumes import (
    FileIdentity,
    StorageError,
    VolumeAdapter,
    native_adapter,
    open_contained,
)

CHUNK = 1048576
MANIFEST_NAME = 'podmuncher-manifest.json'
MANIFEST_SCHEMA = 1
VIDEO_EXTENSIONS = frozenset({'mp4', 'mkv', 'webm'})
_SCRATCH_NAME = re.compile(r'[a-z_]+-[0-9a-f]{32}\.bin')
_SHA256 = re.compile(r'[0-9a-f]{64}')
_BINARY = getattr(os, 'O_BINARY', 0)
_FINAL_NAMES = {
    ArtifactSlot.TRANSCRIPT_TXT: 'transcript.txt',
    ArtifactSlot.TRANSCRIPT_TIMESTAMPED_TXT: 'transcript.timestamped.txt',
    ArtifactSlot.TRANSCRIPT_READABLE_TXT: 'transcript.readable.txt',
    ArtifactSlot.TRANSCRIPT_SRT: 'transcript.srt',
    ArtifactSlot.TRANSCRIPT_JSON: 'transcript.json',
}
_FORMATS = {
    ArtifactSlot.ORIGINAL_VIDEO: ArtifactFormat.VIDEO,
    ArtifactSlot.TRANSCRIPT_TXT: ArtifactFormat.TXT,
    ArtifactSlot.TRANSCRIPT_TIMESTAMPED_TXT: ArtifactFormat.TIMESTAMPED_TXT,
    ArtifactSlot.TRANSCRIPT_READABLE_TXT: ArtifactFormat.READABLE_TXT,
    ArtifactSlot.TRANSCRIPT_SRT: ArtifactFormat.SRT,
    ArtifactSlot.TRANSCRIPT_JSON: ArtifactFormat.JSON,
}
# Default content kind for a worker slot; callers may override explicitly.
_DEFAULT_CONTENT = {
    WorkerSlot.DOWNLOADED_MEDIA: ArtifactSlot.ORIGINAL_VIDEO,
    WorkerSlot.DOWNLOADED_AUDIO: ArtifactSlot.ORIGINAL_VIDEO,
    WorkerSlot.DOWNLOADED_VIDEO: ArtifactSlot.ORIGINAL_VIDEO,
    WorkerSlot.MERGED_VIDEO: ArtifactSlot.ORIGINAL_VIDEO,
    WorkerSlot.TRANSCRIPT_SEGMENTS: ArtifactSlot.TRANSCRIPT_JSON,
    WorkerSlot.PROBE_METADATA: ArtifactSlot.TRANSCRIPT_JSON,
}
_PASS_THROUGH = frozenset({
    ReasonCode.STORAGE_GUARD_TRIPPED, ReasonCode.QUOTA_EXCEEDED,
    ReasonCode.EXPORT_LIMIT_REACHED, ReasonCode.MEDIA_LIMIT_REACHED,
    ReasonCode.STORAGE_UNKNOWN, ReasonCode.STORAGE_STALE,
})


class ArtifactError(DomainValidationError):
    """Closed reason code; the message never contains a path, name or hash."""

    def __init__(self, reason_code: ReasonCode = ReasonCode.ARTIFACT_REJECTED):
        super().__init__(reason_code)

    def __str__(self) -> str:
        return 'Artifact operation failed.'


class Audit(Protocol):
    """Fail-closed sink. Actions: access, publish, reject. Ids and codes only."""

    def __call__(self, action: str, artifact_id: str | None,
                 reason_code: ReasonCode) -> None: ...


@dataclass(frozen=True, slots=True)
class VerifiedFile:
    slot: WorkerSlot
    content: ArtifactSlot
    bytes: int
    sha256: str
    identity: FileIdentity = field(repr=False)
    attempt_dir: str = field(repr=False)
    basename: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    artifact_id: str
    slot: ArtifactSlot
    format: ArtifactFormat
    bytes: int
    sha256: str
    file_name: str
    identity: FileIdentity = field(repr=False)

    def to_artifact(self, attempt_id: str, provenance: Provenance | None) -> Artifact:
        """Public record; the private location is deliberately not recorded."""
        return Artifact(self.artifact_id, self.slot, self.format, self.bytes,
                        self.sha256, True, attempt_id, provenance, None)


def _storage_reason(error: BaseException, default: ReasonCode) -> ReasonCode:
    if isinstance(error, ArtifactError):
        return error.reason_code
    if isinstance(error, (GuardError, DomainValidationError)):
        reason = error.reason_code
        return reason if reason in _PASS_THROUGH else default
    if isinstance(error, OSError):
        if error.errno == errno.ENOSPC:
            return ReasonCode.STORAGE_GUARD_TRIPPED
        if error.errno == getattr(errno, 'EDQUOT', 122):
            return ReasonCode.QUOTA_EXCEEDED
    return default


def _notify(audit: Audit | None, action: str, artifact_id: str | None,
            reason: ReasonCode, *, strict: bool) -> None:
    if audit is None:
        return
    try:
        audit(action, artifact_id, reason)
    except Exception:  # noqa: BLE001 - sink is untrusted; fail closed with a closed code
        if strict:
            raise ArtifactError(ReasonCode.AUDIT_LOSS) from None


def assign_slots(attempt_dir: str, slots: Iterable[WorkerSlot]) -> dict[WorkerSlot, str]:
    """Generate private scratch basenames; the worker never chooses a name."""
    if type(attempt_dir) is not str or not attempt_dir:
        raise ArtifactError(ReasonCode.INVALID_INPUT)
    assigned: dict[WorkerSlot, str] = {}
    for slot in slots:
        slot = WorkerSlot.parse(slot)
        assigned.setdefault(slot, f'{slot.value}-{uuid4().hex}.bin')
    return assigned


def _check_content(content: ArtifactSlot, total: int, decoder, json_buffer) -> None:
    if content is ArtifactSlot.ORIGINAL_VIDEO:
        if total == 0:
            raise ArtifactError()
        return
    try:
        if decoder is not None:
            decoder.decode(b'', final=True)
        if content is ArtifactSlot.TRANSCRIPT_JSON:
            value = json.loads(bytes(json_buffer).decode('utf-8'))
            if type(value) is not dict:
                raise ArtifactError()
    except (UnicodeError, ValueError):
        raise ArtifactError() from None


def _verify(attempt_dir, assignment, slot, claimed_bytes, claimed_sha256, writers_reaped,
            max_bytes, content, adapter) -> VerifiedFile:
    if writers_reaped is not True:
        raise ArtifactError(ReasonCode.WRITER_UNRESOLVED)
    try:
        worker_slot = WorkerSlot.parse(slot)
    except DomainValidationError:
        raise ArtifactError() from None
    name = assignment.get(worker_slot) if isinstance(assignment, dict) else None
    if (type(name) is not str or not _SCRATCH_NAME.fullmatch(name)
            or type(attempt_dir) is not str or type(claimed_bytes) is not int
            or claimed_bytes < 0 or type(claimed_sha256) is not str
            or not _SHA256.fullmatch(claimed_sha256)
            or type(max_bytes) is not int or max_bytes < 0):
        raise ArtifactError()
    kind = _DEFAULT_CONTENT[worker_slot] if content is None else ArtifactSlot.parse(content)
    limit_reason = (ReasonCode.MEDIA_LIMIT_REACHED if kind is ArtifactSlot.ORIGINAL_VIDEO
                    else ReasonCode.EXPORT_LIMIT_REACHED)
    if claimed_bytes > max_bytes:
        raise ArtifactError(limit_reason)
    text = kind is not ArtifactSlot.ORIGINAL_VIDEO
    decoder = codecs.getincrementaldecoder('utf-8')() if text else None
    json_buffer = bytearray() if kind is ArtifactSlot.TRANSCRIPT_JSON else None
    digest, total = hashlib.sha256(), 0
    with open_contained(attempt_dir, name, adapter=adapter) as held:
        identity = held.identity
        while True:
            chunk = held.read(CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ArtifactError(limit_reason)
            if total > claimed_bytes:
                raise ArtifactError()
            digest.update(chunk)
            if decoder is not None:
                decoder.decode(chunk)
            if json_buffer is not None:
                json_buffer.extend(chunk)
        held.verify()
    if total != claimed_bytes or digest.hexdigest() != claimed_sha256:
        raise ArtifactError()
    _check_content(kind, total, decoder, json_buffer)
    return VerifiedFile(worker_slot, kind, total, claimed_sha256, identity,
                        attempt_dir, name)


def verify_artifact(attempt_dir: str, assignment: dict[WorkerSlot, str], slot: object,
                    claimed_bytes: int, claimed_sha256: str, *, writers_reaped: bool,
                    max_bytes: int, content: ArtifactSlot | None = None,
                    adapter: VolumeAdapter | None = None,
                    audit: Audit | None = None) -> VerifiedFile:
    """Re-derive size/hash/content from the file; the event claim is only compared."""
    try:
        verified = _verify(attempt_dir, assignment, slot, claimed_bytes, claimed_sha256,
                           writers_reaped, max_bytes, content,
                           adapter if adapter is not None else native_adapter())
    except (ArtifactError, StorageError, OSError, UnicodeError) as error:
        reason = _storage_reason(error, ReasonCode.ARTIFACT_REJECTED)
        if isinstance(error, (StorageError, OSError, UnicodeError)):
            reason = ReasonCode.ARTIFACT_REJECTED
        _notify(audit, 'reject', None, reason, strict=False)
        raise ArtifactError(reason) from None
    _notify(audit, 'access', None, ReasonCode.OK, strict=True)
    return verified


def _no_clobber_move(source: str, destination: str) -> None:
    """Atomic move that never replaces an existing destination."""
    try:
        if os.name == 'nt':
            os.rename(source, destination)
        else:
            os.link(source, destination)
            os.unlink(source)
    except FileExistsError:
        raise ArtifactError(ReasonCode.DESTINATION_CONFLICT) from None
    except PermissionError:
        # Windows reports some existing-target races as access denied.
        if os.path.lexists(destination):
            raise ArtifactError(ReasonCode.DESTINATION_CONFLICT) from None
        raise


def _sync_directory(path: str) -> None:
    if os.name == 'nt':
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _flush_close(handle) -> None:
    """Durably flush then close a finished copy; runs off the event loop."""
    os.fsync(handle.fileno())
    handle.close()


def _remove_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _final_name(final_slot: ArtifactSlot, video_ext: str) -> str:
    if final_slot is ArtifactSlot.ORIGINAL_VIDEO:
        if video_ext not in VIDEO_EXTENSIONS:
            raise ArtifactError(ReasonCode.INVALID_INPUT)
        return f'video.{video_ext}'
    return _FINAL_NAMES[final_slot]


def _prove_source(verified: VerifiedFile, adapter: VolumeAdapter) -> None:
    with open_contained(verified.attempt_dir, verified.basename,
                        expected=verified.identity, adapter=adapter) as held:
        held.verify()


def _prove_destination(job_path: str, name: str, expected: FileIdentity,
                       adapter: VolumeAdapter) -> None:
    with open_contained(job_path, name, expected=expected, adapter=adapter) as held:
        held.verify()


def _prove_job_volume(output_root: str, job_name: str, job_volume: str,
                      adapter: VolumeAdapter) -> None:
    with open_contained(output_root, job_name, directory=True, adapter=adapter) as job:
        if job.identity.volume != job_volume:
            raise ArtifactError()


def _hash_held(job_path: str, name: str, adapter: VolumeAdapter) -> tuple[str, int, FileIdentity]:
    digest, total = hashlib.sha256(), 0
    with open_contained(job_path, name, adapter=adapter) as held:
        identity = held.identity
        while True:
            chunk = held.read(CHUNK)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), total, identity


def _adopt_existing(job_path: str, name: str, verified: VerifiedFile, job_volume: str,
                    adapter: VolumeAdapter) -> FileIdentity:
    """Prove an existing final file is byte-identical to *verified*; never modify it.

    A retry after an interrupted claim can find its own earlier publication at
    the fixed final name. It is adopted only when it is a contained, no-follow,
    single-link regular file on the job volume whose size and streamed sha256
    match the verified source. Anything else stays a destination conflict.
    """
    try:
        digest, total = hashlib.sha256(), 0
        with open_contained(job_path, name, adapter=adapter) as held:
            identity = held.identity
            if identity.volume != job_volume:
                raise ArtifactError(ReasonCode.DESTINATION_CONFLICT)
            while True:
                chunk = held.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > verified.bytes:
                    raise ArtifactError(ReasonCode.DESTINATION_CONFLICT)
                digest.update(chunk)
            held.verify()
    except (StorageError, OSError):
        raise ArtifactError(ReasonCode.DESTINATION_CONFLICT) from None
    if total != verified.bytes or digest.hexdigest() != verified.sha256:
        raise ArtifactError(ReasonCode.DESTINATION_CONFLICT)
    return identity


async def _move_or_adopt(source: str, job_path: str, name: str, verified: VerifiedFile,
                         job_volume: str, adapter: VolumeAdapter) -> FileIdentity | None:
    """No-clobber move; on conflict adopt an identical destination.

    Returns None when *source* was moved into place, or the adopted
    destination identity (the source is then left for the caller to discard).
    The move is a blocking filesystem call and the adoption check streams the
    whole existing file, so both run off the event loop like the copy path's
    reads.
    """
    try:
        await asyncio.to_thread(_no_clobber_move, source, os.path.join(job_path, name))
    except ArtifactError as error:
        if error.reason_code is not ReasonCode.DESTINATION_CONFLICT:
            raise
        return await asyncio.to_thread(_adopt_existing, job_path, name, verified,
                                       job_volume, adapter)
    return None


async def _copy_publish(verified: VerifiedFile, final_name: str, job_path: str,
                        guard: WriteGuard, adapter: VolumeAdapter,
                        job_volume: str) -> FileIdentity:
    """Copy through *guard* into a private temp, re-hash, then no-clobber move.

    Every blocking step after the chunk loop (flush/fsync, the re-hash and the
    move) runs in a worker thread, one at a time, so a large copy never stalls
    the event loop or the guard's watchdog. The guard is re-checked immediately
    before the move: a watchdog stop that lands after the last accepted chunk
    aborts the publication with the guard's reason and the temp is removed.
    """
    temp_name = f'.{final_name}.{uuid4().hex}.partial'
    temp_path = os.path.join(job_path, temp_name)
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY
             | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0))
    await guard.publication()
    handle = os.fdopen(os.open(temp_path, flags, 0o600), 'wb', buffering=0)
    created = True
    try:
        writer = AsyncFileWriter(handle)
        digest, total = hashlib.sha256(), 0
        with open_contained(verified.attempt_dir, verified.basename,
                            expected=verified.identity, adapter=adapter) as source:
            while True:
                chunk = await asyncio.to_thread(source.read, CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                digest.update(chunk)
                await guard.write(writer, chunk)
        if total != verified.bytes or digest.hexdigest() != verified.sha256:
            raise ArtifactError()
        # Every guarded write has returned, so no other thread holds the handle.
        await asyncio.to_thread(_flush_close, handle)
        sha, size, temp_identity = await asyncio.to_thread(_hash_held, job_path,
                                                           temp_name, adapter)
        if sha != verified.sha256 or size != verified.bytes:
            raise ArtifactError()
        # The watchdog may have stopped the guard while the copy was being
        # flushed or re-hashed; a stopped guard never reaches the move.
        await guard.publication()
        adopted = await _move_or_adopt(temp_path, job_path, final_name, verified,
                                       job_volume, adapter)
        if adopted is not None:
            return adopted  # the private temp copy is discarded below
        created = False
        return temp_identity
    finally:
        if not handle.closed:
            try:
                handle.close()
            except OSError:
                pass
        if created:
            _remove_quietly(temp_path)


async def _publish(verified, final_slot, job_id, output_root, guard, same_volume,
                   video_ext, adapter) -> PublishedArtifact:
    if type(verified) is not VerifiedFile:
        raise ArtifactError(ReasonCode.INVALID_INPUT)
    slot = ArtifactSlot.parse(final_slot)
    validate_uuid_string(job_id)
    if type(output_root) is not str or not output_root:
        raise ArtifactError(ReasonCode.INVALID_INPUT)
    if slot is not verified.content:
        raise ArtifactError()
    name = _final_name(slot, video_ext)
    job_name = f'podmuncher-{job_id}'
    try:
        os.mkdir(os.path.join(output_root, job_name))
    except FileExistsError:
        pass
    job_path = os.path.join(output_root, job_name)
    # Windows refuses renames into a directory we hold (no FILE_SHARE_DELETE
    # analogue for entries), so the job directory is proved before and after the
    # move instead of being held across it.
    with open_contained(output_root, job_name, directory=True, adapter=adapter) as job:
        job_volume = job.identity.volume
    same = same_volume(verified.identity.volume, job_volume)
    await asyncio.to_thread(_prove_source, verified, adapter)
    discard = None
    if same:
        if guard is not None:
            # A guard supplied for the move path is honoured the same way the
            # copy path honours it: a stopped guard never reaches the move.
            await guard.publication()
        source = os.path.join(verified.attempt_dir, verified.basename)
        adopted = await _move_or_adopt(source, job_path, name, verified, job_volume, adapter)
        if adopted is None:
            identity = verified.identity
        else:
            # Identical orphan from an earlier attempt: the private scratch
            # source is discarded exactly as the successful move would have,
            # but only once every final destination check below has passed.
            identity = adopted
            discard = source
    else:
        identity = await _copy_publish(verified, name, job_path, guard, adapter, job_volume)
    await asyncio.to_thread(_prove_destination, job_path, name, identity, adapter)
    await asyncio.to_thread(_prove_job_volume, output_root, job_name, job_volume, adapter)
    if discard is not None:
        _remove_quietly(discard)
    await asyncio.to_thread(_sync_directory, job_path)
    return PublishedArtifact(str(uuid4()), slot, _FORMATS[slot], verified.bytes,
                             verified.sha256, name, identity)


async def publish(verified: VerifiedFile, final_slot: ArtifactSlot, *, job_id: str,
                  output_root: str, guard: WriteGuard | None, audit: Audit,
                  video_ext: str = 'mp4', adapter: VolumeAdapter | None = None,
                  same_volume: Callable[[str, str], bool] | None = None) -> PublishedArtifact:
    """No-clobber publish to <output_root>/podmuncher-<job_id>/.

    Same volume: atomic move. Cross volume: copy through *guard* (already
    started by the caller, who also finishes it) into a private temp, re-hash,
    re-check the guard, then no-clobber rename; a guard stopped after the last
    accepted chunk aborts with the guard's reason and leaves no temp or final
    file. Blocking filesystem steps run off the event loop. The scratch source
    is never deleted here on the cross-volume path and is never touched on
    failure. An existing destination
    that is byte-identical to *verified* is adopted (never rewritten) and
    audited as a normal ``publish``; any other existing destination is a
    ``DESTINATION_CONFLICT``.
    """
    chooser = same_volume if same_volume is not None else (lambda a, b: a == b)
    try:
        published = await _publish(verified, final_slot, job_id, output_root, guard,
                                   chooser, video_ext,
                                   adapter if adapter is not None else native_adapter())
    except (ArtifactError, GuardError, StorageError, OSError, DomainValidationError) as error:
        reason = _storage_reason(error, ReasonCode.PUBLICATION_FAILED)
        if isinstance(error, StorageError):
            reason = ReasonCode.ARTIFACT_REJECTED
        _notify(audit, 'reject', None, reason, strict=False)
        raise ArtifactError(reason) from None
    _notify(audit, 'publish', published.artifact_id, ReasonCode.OK, strict=True)
    return published


def _read_own_manifest(job_dir: str, job_id: str, adapter: VolumeAdapter) -> None:
    """Raise unless the existing manifest is our own for this job."""
    try:
        with open_contained(job_dir, MANIFEST_NAME, adapter=adapter) as held:
            data = bytearray()
            while len(data) <= CHUNK:
                chunk = held.read(CHUNK)
                if not chunk:
                    break
                data.extend(chunk)
        doc = json.loads(bytes(data).decode('utf-8'))
        if (len(data) > CHUNK or type(doc) is not dict
                or doc.get('schema_version') != MANIFEST_SCHEMA
                or doc.get('job_id') != job_id):
            raise ValueError
    except (StorageError, OSError, ValueError):
        raise ArtifactError(ReasonCode.DESTINATION_CONFLICT) from None


def write_manifest(job_dir: str, job_id: str, attempt_id: str,
                   published: list[PublishedArtifact], provenance: Provenance | None,
                   *, adapter: VolumeAdapter | None = None) -> str:
    """Write the sidecar beside the outputs. Returns its fixed basename."""
    adapter = adapter if adapter is not None else native_adapter()
    try:
        validate_uuid_string(job_id)
        validate_uuid_string(attempt_id)
        if (type(job_dir) is not str or type(published) is not list
                or any(type(item) is not PublishedArtifact for item in published)
                or (provenance is not None and type(provenance) is not Provenance)):
            raise ArtifactError(ReasonCode.INVALID_INPUT)
        document = {
            'schema_version': MANIFEST_SCHEMA, 'job_id': job_id, 'attempt_id': attempt_id,
            'artifacts': [{'artifact_id': a.artifact_id, 'slot': a.slot.value,
                           'format': a.format.value, 'bytes': a.bytes, 'sha256': a.sha256,
                           'file_name': a.file_name} for a in published],
            'provenance': provenance.to_dict() if provenance is not None else None,
        }
        payload = json.dumps(document, sort_keys=True).encode('utf-8')
        parent, job_name = os.path.split(job_dir)
        final_path = os.path.join(job_dir, MANIFEST_NAME)
        temp_path = os.path.join(job_dir, f'.{MANIFEST_NAME}.{uuid4().hex}.partial')
        with open_contained(parent, job_name, directory=True, adapter=adapter):
            pass
        exists = os.path.lexists(final_path)
        if exists:
            _read_own_manifest(job_dir, job_id, adapter)
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY
                 | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0))
        created = True
        try:
            with os.fdopen(os.open(temp_path, flags, 0o600), 'wb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if exists:
                os.replace(temp_path, final_path)
            else:
                _no_clobber_move(temp_path, final_path)
            created = False
            _sync_directory(job_dir)
        finally:
            if created:
                _remove_quietly(temp_path)
        with open_contained(parent, job_name, directory=True, adapter=adapter):
            pass
        with open_contained(job_dir, MANIFEST_NAME, adapter=adapter):
            pass
    except (StorageError, OSError, DomainValidationError) as error:
        reason = (error.reason_code if isinstance(error, ArtifactError)
                  else ReasonCode.INVALID_INPUT if isinstance(error, DomainValidationError)
                  and not isinstance(error, StorageError)
                  else _storage_reason(error, ReasonCode.PUBLICATION_FAILED))
        raise ArtifactError(reason) from None
    return MANIFEST_NAME
