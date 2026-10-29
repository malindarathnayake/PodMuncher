"""Tests for bin/release_candidate.py: pure functions only, no Docker, SSH or network."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
CAPS = {"linux-amd64": ("media", "asr_cpu", "asr_cuda"), "linux-arm64": ("media", "asr_cpu")}


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rc = _load("release_candidate_under_test", "bin/release_candidate.py")
manifest_tests = _load("release_manifest_tests_ref", "tests/test_release_manifest.py")
REAL_MANIFEST = REPO / "src" / "podmuncher" / "manifests" / "release.json"


def _identities(manifest: dict, repo: Path = REPO) -> list[dict]:
    documents = []
    counter = 0
    for platform, caps in CAPS.items():
        entries = {}
        for cap in caps:
            counter += 1
            spec = manifest["local_build_identities"][cap]
            hashes = rc.compute_hashes(cap, spec["build_inputs"], repo)
            entries[cap] = {
                "local_tag": rc.TAGS[cap],
                "local_image_id": "sha256:" + f"{counter:x}" * 64,
                "local_image_kind": "application/vnd.oci.image.index.v1+json",
                "compressed_bytes": 1000 * counter,
                "compressed_bytes_method": rc.COMPRESSED_METHOD,
                "unpacked_bytes": 5000 * counter + (7 if platform == "linux-arm64" else 0),
                "unpacked_bytes_method": "test method " + platform,
                "built_at": f"2026-10-2{counter}T00:00:00.000000001Z",
                "built_on": f"test host, engine {platform}",
                "architecture": rc.DOCKER_ARCH[platform],
                "user": "65532:65532",
                "entrypoint": spec["entrypoint"],
                **hashes,
            }
        documents.append({"platform": platform, "commit": None, "engine_version": "0", "capabilities": entries})
    return documents


def _write_identities(directory: Path, documents: list[dict]) -> list[str]:
    paths = []
    for index, document in enumerate(documents):
        path = directory / f"rc-{index}.json"
        path.write_text(json.dumps(document), "utf-8")
        paths.append(str(path))
    return paths


def test_hash_helpers_match_release_manifest_rules():
    manifest = json.loads(REAL_MANIFEST.read_text("utf-8"))
    for cap in ("media", "asr_cpu", "asr_cuda"):
        files = manifest["local_build_identities"][cap]["build_inputs"]
        expected_inputs = manifest_tests._build_inputs_sha(files)
        expected_requirements = manifest_tests._dockerfile_requirements_sha(cap)
        assert rc.build_inputs_sha(files, REPO) == expected_inputs
        assert rc.dockerfile_requirements_sha(cap, REPO) == expected_requirements
        assert len(expected_inputs) == 64 and len(expected_requirements) == 64
    assert rc.dockerfile_requirements_sha("media", REPO) == "dd8903d5d88c8870f90b5c95aa3a118e9ed77433729bd7c89e1e7067c75b9cd8"
    assert rc.build_inputs_sha(manifest["local_build_identities"]["media"]["build_inputs"], REPO) == hashlib.sha256(
        "\n".join(
            f"{n} {hashlib.sha256((REPO / n).read_bytes()).hexdigest()}"
            for n in manifest["local_build_identities"]["media"]["build_inputs"]
        ).encode("ascii")
    ).hexdigest()


def test_apply_updates_identities_versions_and_rehashes(tmp_path):
    copy = tmp_path / "release.json"
    shutil.copyfile(REAL_MANIFEST, copy)
    before = json.loads(copy.read_text("utf-8"))
    documents = _identities(before)
    paths = _write_identities(tmp_path, documents)

    assert rc.main(["apply", *paths, "--manifest", str(copy)]) == 0

    raw = copy.read_bytes()
    after = json.loads(raw.decode("utf-8"))
    assert raw == (json.dumps(after, indent=1, ensure_ascii=True) + "\n").encode("utf-8")
    assert b"\r" not in raw and raw.endswith(b"}\n")
    assert list(after) == list(before)
    assert json.dumps(after["benchmarks"], sort_keys=True) == json.dumps(before["benchmarks"], sort_keys=True)
    assert after["gpu_evidence"] == before["gpu_evidence"] and after["release_status"] == before["release_status"]

    by_key = {(d["platform"], cap): e for d in documents for cap, e in d["capabilities"].items()}
    for cap in ("media", "asr_cpu", "asr_cuda"):
        record = after["local_build_identities"][cap]
        old = before["local_build_identities"][cap]
        assert list(record) == list(old)
        platforms = [p for p in CAPS if cap in CAPS[p]]
        for platform in platforms:
            for field in rc.PLATFORM_FIELDS:
                assert record["platforms"][platform][field] == by_key[(platform, cap)][field]
        for field in rc.HASH_FIELDS:
            assert record[field] == by_key[(platforms[0], cap)][field]
        assert record["max_compressed_bytes"] == max(by_key[(p, cap)]["compressed_bytes"] for p in platforms)
        assert record["max_unpacked_bytes"] == max(by_key[(p, cap)]["unpacked_bytes"] for p in platforms)

    assert after["release_version"] == "1.0.1"
    assert after["generated_at"].endswith("Z") and len(after["generated_at"]) == 20
    for platform, runtime in after["runtime_manifests"].items():
        assert runtime["release_version"] == "1.0.1"
        for item in runtime["capabilities"]:
            record = after["local_build_identities"][item["id"]]
            assert item["digest"] == by_key[(platform, item["id"])]["local_image_id"]
            assert item["compressed_bytes"] == record["max_compressed_bytes"]
            assert item["unpacked_peak_bytes"] == record["max_unpacked_bytes"]
        canonical = json.dumps(runtime, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        assert after["runtime_manifest_sha256"][platform] == hashlib.sha256(canonical.encode("ascii")).hexdigest()
    assert after["runtime_manifest_sha256"] != before["runtime_manifest_sha256"]


def _mutations():
    def wrong_inputs(docs):
        docs[0]["capabilities"]["media"]["build_inputs_sha256"] = "0" * 64

    def duplicate_ids(docs):
        docs[1]["capabilities"]["media"]["local_image_id"] = docs[0]["capabilities"]["media"]["local_image_id"]

    def missing_cap(docs):
        del docs[0]["capabilities"]["asr_cuda"]

    def missing_platform(docs):
        del docs[1]

    def bad_user(docs):
        docs[0]["capabilities"]["media"]["user"] = "0:0"

    def bad_sizes(docs):
        entry = docs[0]["capabilities"]["media"]
        entry["unpacked_bytes"] = entry["compressed_bytes"] - 1

    return [wrong_inputs, duplicate_ids, missing_cap, missing_platform, bad_user, bad_sizes]


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda f: f.__name__)
def test_apply_refuses_bad_identity_and_leaves_manifest_untouched(tmp_path, mutate, capsys):
    copy = tmp_path / "release.json"
    shutil.copyfile(REAL_MANIFEST, copy)
    original = copy.read_bytes()
    documents = _identities(json.loads(original.decode("utf-8")))
    mutate(documents)
    paths = _write_identities(tmp_path, documents)

    assert rc.main(["apply", *paths, "--manifest", str(copy)]) != 0

    assert copy.read_bytes() == original
    assert "identity validation failed" in capsys.readouterr().err
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".release-")] == []


def test_remote_refuses_dirty_build_context(monkeypatch, capsys, tmp_path):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv[:3] == ["git", "status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout=" M containers/media.Dockerfile\n", stderr="")
        raise AssertionError("unexpected call " + repr(argv))

    def fake_popen(argv, **kwargs):
        raise AssertionError("popen must not be reached")

    monkeypatch.setattr(rc, "run", fake_run)
    monkeypatch.setattr(rc, "popen", fake_popen)
    code = rc.main(["remote", "--host", "podmac", "--platform", "linux-arm64", "--built-on", "mac",
                    "--out", str(tmp_path / "out.json")])
    assert code != 0
    assert "uncommitted" in capsys.readouterr().err
    assert len(calls) == 1 and not any(c[0] == "ssh" for c in calls)
    assert not (tmp_path / "out.json").exists()


def test_ca_path_never_written(monkeypatch, capsys, tmp_path):
    ca = tmp_path / "corp-ca-unique-name.pem"
    ca.write_text("SECRET-CA-BYTES", "utf-8")
    out = tmp_path / "identity.json"
    media_inputs = json.loads(REAL_MANIFEST.read_text("utf-8"))["local_build_identities"]["media"]["entrypoint"]
    builds = []

    def fake_run(argv, **kwargs):
        argv = list(argv)
        if argv[:2] == ["docker", "build"]:
            builds.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[:3] == ["docker", "image", "inspect"]:
            payload = {"Id": "sha256:" + "a" * 64, "Architecture": "amd64", "Size": 9000, "Created": "2026-10-28T00:00:00Z",
                       "Config": {"User": "65532:65532", "Entrypoint": media_inputs}}
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        if argv[:2] == ["docker", "version"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"Version": "29.2.1", "Arch": "amd64"}), stderr="")
        if argv[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        raise AssertionError("unexpected call " + repr(argv))

    class FakeSave:
        def __init__(self):
            self.stdout = io.BytesIO(b"x" * 100)

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(rc, "run", fake_run)
    monkeypatch.setattr(rc, "popen", lambda argv, **kw: FakeSave())
    code = rc.main(["build", "--platform", "linux-amd64", "--built-on", "Test PC", "--out", str(out), "--ca", str(ca)])
    assert code == 0

    assert len(builds) == 3
    for argv in builds:
        assert f"id=build_ca,src={ca}" in argv
        assert argv[argv.index("--platform") + 1] == "linux/amd64"
    assert [b[b.index("--target") + 1] for b in builds if "--target" in b] == ["cpu", "cuda"]

    written = out.read_text("utf-8")
    captured = capsys.readouterr()
    for text in (written, captured.out, captured.err):
        assert str(ca) not in text and "corp-ca-unique-name" not in text and "SECRET-CA-BYTES" not in text
    document = json.loads(written)
    assert set(document["capabilities"]) == {"media", "asr_cpu", "asr_cuda"}
    media = document["capabilities"]["media"]
    assert media["local_image_kind"] == "application/vnd.oci.image.index.v1+json"
    assert media["compressed_bytes"] == 100 and media["unpacked_bytes"] == 9000
    assert media["built_on"] == "Test PC, engine x86_64"
    assert media["unpacked_bytes_method"] == "docker image inspect Size (Docker Engine 29.2.1, containerd snapshotter)"
