"""P7-U4: pure envelope, blocker and record-hash rules of bin/release_benchmarks.py on synthetic records.

No Docker, nvidia-smi or model downloads: ``apply`` runs only on a copy of release.json in tmp_path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "src" / "podmuncher" / "manifests" / "release.json"
GIB = 1024**3


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load("release_benchmarks_under_test", REPO / "bin" / "release_benchmarks.py")


def _record(capability="asr_cpu", model="tiny", codec="wav", seconds=60, outcome="completed", *,
            image="sha256:" + "ab12" * 16, peak=500_000_000, current=True, vram=None):
    measured = outcome == "completed"
    wall = seconds * 0.25
    audio = seconds + 0.02
    if vram is None and capability == "asr_cuda" and measured:
        vram = 300 * 1024**2
    return bench.seal_record({
        "series": "attempt-3", "name": bench.record_name(capability, model, codec, seconds),
        "capability": capability, "platform": "linux-amd64", "bound_image_id": image,
        "bound_image_basis": "record image_id", "bound_to_current_build": current, "model_id": model,
        "codec": codec, "fixture_seconds": seconds, "memory_cap_bytes": 2 * GIB, "cpus": 1.0,
        "storage_sample_budget_seconds": 2.5, "outcome": outcome,
        "load_seconds": 1.5 if measured else None,
        "transcribe_wall_seconds": wall if measured else None,
        "audio_seconds": audio if measured else None,
        "real_time_factor": round(wall / audio, 4) if measured else None,
        "peak_rss_bytes": peak if measured else None,
        "worker_vram_delta_bytes": vram if measured else None,
        "segments": 12 if measured else None, "detected_language": "en" if measured else None,
        "recorded_at": "2026-10-28T01:02:03.000004+00:00",
    })


def _evidence(records):
    return hashlib.sha256("\n".join(sorted(r["record_sha256"] for r in records)).encode("ascii")).hexdigest()


def _synthetic_run(document: dict) -> dict:
    ids = bench.current_image_ids(document)
    fixtures = []
    for seconds in bench.FIXTURE_SECONDS:
        for codec, (extension, _) in bench.CODECS.items():
            name = f"speech-{seconds}.{extension}"
            fixtures.append({"name": name, "codec": codec, "seconds": seconds, "bytes": 1000 + seconds,
                             "sha256": hashlib.sha256(name.encode()).hexdigest()})
    records = [_record("asr_cpu", "small", "mp4_aac", 60, "host_guard_failure:storage_unknown",
                       image=ids["asr_cpu"])]
    for index, (capability, model, codec, seconds) in enumerate(bench.MATRIX):
        records.append(_record(capability, model, codec, seconds, image=ids[capability],
                               peak=400_000_000 + index * 10_000_000))
    probes = json.loads(json.dumps(document["benchmarks"]["probes"]))
    for cap, probe in probes.items():
        probe["bound_image_id"] = ids[cap]
    gpu = json.loads(json.dumps(document["gpu_evidence"]))
    gpu["bound_image_id"] = ids["asr_cuda"]
    gpu["passthrough"] = True
    return {"fixtures": sorted(fixtures, key=lambda f: f["name"]), "records": records, "probes": probes,
            "gpu_evidence": gpu}


def _copy(tmp_path: Path) -> Path:
    path = tmp_path / "release.json"
    shutil.copyfile(MANIFEST_PATH, path)
    return path


def test_record_hash_is_sha256_of_canonical_jsonl_line():
    record = _record()
    assert list(record) == list(bench.RECORD_FIELDS)
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    line = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert bench.record_line(record) == line and "\n" not in line
    assert record["record_sha256"] == hashlib.sha256(line.encode("ascii")).hexdigest()
    assert bench.record_hash(record) == record["record_sha256"]
    changed = dict(record, peak_rss_bytes=record["peak_rss_bytes"] + 1)
    assert bench.record_hash(changed) != record["record_sha256"]
    assert record["name"] == "asr_cpu-tiny-wav-60s-2048m"


def test_envelope_rules_match_release_manifest():
    records = [
        _record(model="base", codec="mp3", seconds=60, peak=600_000_000),
        _record(model="base", codec="wav", seconds=60, peak=650_000_000),
        _record(model="base", codec="mp4_aac", seconds=60, outcome="host_guard_failure:storage_unknown"),
        _record(model="base", codec="mp4_aac", seconds=60, peak=640_000_000),
        _record(model="base", codec="mp3", seconds=600, peak=700_000_000),
        _record(model="base", codec="wav", seconds=60, peak=999_000_000, current=False),  # older build
        _record(model="tiny", codec="wav", seconds=60, peak=300_000_000),  # other model
        _record("asr_cuda", "base", "wav", 60, peak=900_000_000),  # other capability
    ]
    mine = records[:5]
    envelope = bench.model_envelope(records, "base")
    assert envelope == {
        "evidence_sha256": _evidence(mine),
        "codec_matrix": ["mp3", "mp4_aac", "wav"],
        "worker_peak_ram_bytes": 700_000_000,
        "validated_max_duration_seconds": 600,
    }
    assert bench.model_envelope(records, "small") is None  # no evidence: the model is left untouched


def test_validated_duration_stops_below_first_worker_failure():
    records = [
        _record(model="tiny", codec="mp3", seconds=60),
        _record(model="tiny", codec="mp3", seconds=300, outcome="worker_failure:artifact_rejected"),
        _record(model="tiny", codec="mp3", seconds=600, peak=800_000_000),
    ]
    envelope = bench.model_envelope(records, "tiny")
    assert envelope["validated_max_duration_seconds"] == 60
    assert envelope["worker_peak_ram_bytes"] == 800_000_000  # completed records still bound the RAM
    assert envelope["evidence_sha256"] == _evidence(records)
    only_failed = [_record(model="tiny", seconds=60, outcome="worker_failure:model_oom"), _record(seconds=60)]
    with pytest.raises(bench.ToolError):
        bench.model_envelope(only_failed, "tiny")
    assert bench.classify_failure("storage_unknown", True) == "host_guard_failure:storage_unknown"
    assert bench.classify_failure("model_oom", False) == "host_guard_failure:model_oom"
    assert bench.classify_failure("model_oom", True) == "worker_failure:model_oom"


def test_blockers_follow_measured_evidence():
    document = json.loads(MANIFEST_PATH.read_text("utf-8"))
    existing = document["release_status"]["blockers"]
    models = [{"id": "tiny", "validated_max_duration_seconds": 600}, {"id": "medium"}]
    fixed = [(b["code"], b["capability"]) for b in existing if b["code"] not in bench.CONDITIONAL_CODES]

    cpu = [_record(seconds=600)]
    cuda_ok = cpu + [_record("asr_cuda", "tiny", "wav", 60)]
    blockers = bench.rebuild_blockers(existing, cuda_ok, models)
    keys = [(b["code"], b["capability"]) for b in blockers]
    assert keys == fixed + [("long_duration_unvalidated", "asr_cpu"), ("long_duration_unvalidated", "asr_cuda")]
    for blocker in blockers:
        if blocker["code"] not in bench.CONDITIONAL_CODES:
            assert blocker in existing  # unconditional blockers keep their text
    details = {b["capability"]: b["detail"] for b in blockers if b["code"] == "long_duration_unvalidated"}
    assert "measured up to 600 s" in details["asr_cpu"] and "21600 s" in details["asr_cpu"]
    assert "measured up to 60 s" in details["asr_cuda"]

    no_cuda = cpu + [_record("asr_cuda", "tiny", "wav", 60, "host_guard_failure:gpu_unavailable")]
    keys = [(b["code"], b["capability"]) for b in bench.rebuild_blockers(existing, no_cuda, models)]
    assert ("gpu_evidence_missing", "asr_cuda") in keys
    assert ("long_duration_unvalidated", "asr_cuda") not in keys
    assert ("long_duration_unvalidated", "asr_cpu") in keys
    missing = next(b for b in bench.rebuild_blockers(existing, no_cuda, models) if b["code"] == "gpu_evidence_missing")
    assert missing["resolved_by"] == "GPU evidence run" and missing["detail"].endswith(".")

    failing = cuda_ok + [_record(model="tiny", seconds=300, outcome="worker_failure:artifact_rejected"),
                         _record(model="tiny", seconds=300, outcome="worker_failure:model_oom", current=False)]
    blockers = bench.rebuild_blockers(existing, failing, models)
    keys = [(b["code"], b["capability"]) for b in blockers]
    assert ("worker_defect_open", "asr_cpu") in keys and ("worker_defect_open", "asr_cuda") not in keys
    defect = next(b for b in blockers if b["code"] == "worker_defect_open")
    assert defect["resolved_by"] == "worker defect fix"
    assert len(set(keys)) == len(keys)
    # Short fixtures never certify 6 h: the long-duration blockers stay.
    assert set(bench.expected_conditional_blockers(cuda_ok, models)) == {
        ("long_duration_unvalidated", "asr_cpu"), ("long_duration_unvalidated", "asr_cuda")}


def test_apply_on_copy_satisfies_release_manifest_invariants(tmp_path):
    checks = _load("release_manifest_checks_for_benchmarks", REPO / "tests" / "test_release_manifest.py")
    manifest = _copy(tmp_path)
    original = json.loads(manifest.read_text("utf-8"))
    run = _synthetic_run(original)
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(run), encoding="utf-8")
    assert bench.main(["apply", str(run_file), "--manifest", str(manifest)]) == 0
    raw = manifest.read_bytes()
    assert raw.endswith(b"}\n") and b"\r\n" not in raw
    loaded = checks.load_release_manifest(manifest)
    checks.test_envelopes_and_codec_matrix_match_the_measured_records(loaded)
    checks.test_blockers_are_exactly_the_missing_publication_review_and_evidence(loaded)
    document = loaded[0]
    assert document["benchmarks"]["records"] == run["records"]
    assert document["benchmarks"]["fixture_generator"] == bench.FIXTURE_GENERATOR
    assert document["benchmarks"]["method"] == original["benchmarks"]["method"]
    assert document["release_status"]["publishable"] is False
    assert document["model_selection"]["measured_envelopes"]["models"] == ["tiny", "base", "small"]
    amd64 = {m["id"]: m for m in document["runtime_manifests"]["linux-amd64"]["models"]}
    before = {m["id"]: m for m in original["runtime_manifests"]["linux-amd64"]["models"]}
    for model_id, model in amd64.items():
        changed = {k for k in set(model) | set(before[model_id]) if model.get(k) != before[model_id].get(k)}
        allowed = {"evidence_sha256", "codec_matrix", "worker_peak_ram_bytes", "validated_max_duration_seconds"}
        assert changed <= (allowed if model_id in bench.MEASURED_MODELS else set()), (model_id, changed)
        if model_id in bench.MEASURED_MODELS:
            assert model["worker_peak_ram_bytes"] < GIB and model["validated_max_duration_seconds"] == 600
    assert document["runtime_manifests"]["linux-arm64"] == original["runtime_manifests"]["linux-arm64"]


def test_apply_refuses_stale_image_binding_and_leaves_manifest_untouched(tmp_path, capsys):
    manifest = _copy(tmp_path)
    before = manifest.read_bytes()
    document = json.loads(before)
    run = _synthetic_run(document)
    stale = "sha256:" + "cd34" * 16
    run["records"][1] = bench.seal_record(dict(run["records"][1], bound_image_id=stale))
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps(run), encoding="utf-8")
    assert bench.main(["apply", str(run_file), "--manifest", str(manifest)]) == 1
    assert "not the current" in capsys.readouterr().err
    assert manifest.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["release.json", "run.json"]  # no temp file left

    run = _synthetic_run(document)
    run["probes"]["asr_cuda"]["bound_image_id"] = stale
    run_file.write_text(json.dumps(run), encoding="utf-8")
    assert bench.main(["apply", str(run_file), "--manifest", str(manifest)]) == 1
    assert manifest.read_bytes() == before

    run = _synthetic_run(document)
    run["records"][2]["peak_rss_bytes"] += 1  # tampered: record_sha256 no longer recomputes
    run_file.write_text(json.dumps(run), encoding="utf-8")
    assert bench.main(["apply", str(run_file), "--manifest", str(manifest)]) == 1
    assert manifest.read_bytes() == before
