"""Distribution contract: hash-locked requirements, wheel contents and clean-install smoke.

Deterministic tests (``-m "not live"``) check the lock against the pyproject pins, the
packaging metadata policy and the contents of a wheel built from the checkout.  The live
smoke (``-m live``) builds the wheel, installs it with the locked dependencies into a
clean virtual environment outside the checkout and runs the CLI and the TUI from there.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "requirements.lock"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
WHEEL_ASSETS = ("podmuncher/assets/app.tcss", "podmuncher/assets/welcome.txt")
HEAVY_MODULES = {"textual", "aiohttp", "psutil", "platformdirs", "opentelemetry", "docker"}
SMOKE_MARKER = "PODMUNCHER_SMOKE_OK"

_FORBIDDEN_SEGMENTS = {"hn844", "research", "docs", "transcripts", "models", "__pycache__",
                       ".venv", ".codex"}
_FORBIDDEN_SUFFIXES = {".pem", ".crt", ".cer", ".key", ".p12", ".pfx", ".safetensors",
                       ".bin", ".onnx", ".pt", ".ckpt", ".gguf", ".mp3", ".wav", ".srt",
                       ".vtt", ".pyc", ".pyo"}
_REQUIRED_EXCLUDES = {"/HN844", "/research", "/Docs", "**/.foreman-*", "**/transcripts",
                      "**/models", "**/*.pem", "**/*.crt", "**/*.key", "**/.env",
                      "**/*.safetensors"}
_LOCK_ENTRY = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;\\]+)")
_LOCK_HASH = re.compile(r"--hash=sha256:(?P<digest>[0-9a-f]{64})\b")


# ----------------------------------------------------------------------------- helpers


def load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def parse_lock(text: str) -> dict[str, dict]:
    """Parse a ``--generate-hashes`` lock into {canonical name: {version, hashes}}."""
    entries: dict[str, dict] = {}
    current: dict | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LOCK_ENTRY.match(line)
        if match:
            name = canonicalize_name(match["name"])
            assert name not in entries, f"duplicate lock entry {name}"
            current = {"version": match["version"], "hashes": []}
            entries[name] = current
            continue
        digest = _LOCK_HASH.search(line)
        if digest:
            assert current is not None, "hash line before any requirement"
            current["hashes"].append(digest["digest"])
            continue
        pytest.fail(f"unexpected lock line (only pinned, hashed requirements allowed): {line}")
    return entries


def pinned_requirements(project: dict) -> dict[str, str]:
    """All runtime and dev requirements from pyproject as {canonical name: exact version}."""
    pins: dict[str, str] = {}
    for value in [*project["dependencies"], *project["optional-dependencies"]["dev"]]:
        requirement = Requirement(value)
        specs = list(requirement.specifier)
        assert len(specs) == 1 and specs[0].operator == "==", f"{value} is not an exact pin"
        pins[canonicalize_name(requirement.name)] = specs[0].version
    return pins


def forbidden_member(name: str) -> bool:
    """True for distribution members that must never ship (research, secrets, weights...)."""
    path = PurePosixPath(name)
    segments = [part.lower() for part in path.parts]
    if any(part in _FORBIDDEN_SEGMENTS or part.startswith(".foreman") for part in segments):
        return True
    if path.name == ".env" or path.name.startswith(".env."):
        return True
    return path.suffix.lower() in _FORBIDDEN_SUFFIXES


def inspect_wheel(wheel: Path) -> dict:
    """Return the wheel's member names, METADATA headers and console entry points."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        dist_info = {PurePosixPath(n).parts[0] for n in names if ".dist-info/" in n}
        assert len(dist_info) == 1, dist_info
        info = dist_info.pop()
        metadata_text = archive.read(f"{info}/METADATA").decode("utf-8")
        entry_points = archive.read(f"{info}/entry_points.txt").decode("utf-8")
    headers: dict[str, list[str]] = {}
    for line in metadata_text.split("\n\n", 1)[0].splitlines():
        key, _, value = line.partition(": ")
        headers.setdefault(key, []).append(value)
    return {
        "names": names,
        "metadata": headers,
        "entry_points": entry_points,
        "forbidden": [n for n in names if forbidden_member(n)],
    }


def assert_wheel_policy(report: dict) -> None:
    for asset in WHEEL_ASSETS:
        assert asset in report["names"], f"missing packaged asset {asset}"
    assert "podmuncher/__init__.py" in report["names"]
    assert report["forbidden"] == [], report["forbidden"]
    assert all(n.startswith(("podmuncher/", "podmuncher-"))
               for n in report["names"]), report["names"]
    metadata = report["metadata"]
    assert metadata["License-Expression"] == ["Apache-2.0"]
    assert metadata["Requires-Python"] == ["<3.15,>=3.12"]
    assert metadata["Version"] == ["1.0.2"]
    assert "[console_scripts]" in report["entry_points"]
    assert "podmuncher = podmuncher:main" in report["entry_points"]


def build_wheel(outdir: Path) -> Path:
    """Build the wheel with ``sys.executable -m build`` from the pinned, installed backend."""
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(outdir),
         str(PROJECT_ROOT)],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    wheels = list(outdir.glob("podmuncher-*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


def make_clean_env(home: Path, **extra: str) -> dict[str, str]:
    """Environment for running the installed package: no checkout paths, isolated homes."""
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP",
                                "PODMUNCHER_STATE_DIR", "PYTHONSAFEPATH"}}
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONIOENCODING="utf-8",
        HOME=str(home), USERPROFILE=str(home), APPDATA=str(home), LOCALAPPDATA=str(home),
        XDG_CONFIG_HOME=str(home), XDG_CACHE_HOME=str(home), XDG_DATA_HOME=str(home),
        XDG_STATE_HOME=str(home),
    )
    env.update(extra)
    return env


def find_uv() -> str:
    found = shutil.which("uv") or shutil.which("uv", path=str(Path.home() / ".local" / "bin"))
    if not found:
        pytest.fail("uv is required for the clean-install smoke and was not found on PATH "
                    "or in ~/.local/bin")
    return found


def run_checked(argv: list, *, cwd: Path, env: dict, timeout: float, what: str
                ) -> subprocess.CompletedProcess:
    result = subprocess.run([str(a) for a in argv], cwd=cwd, env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=timeout,
                            check=False)
    if result.returncode != 0:
        pytest.fail(f"{what} failed (exit {result.returncode}); network or index may be "
                    f"unavailable:\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
    return result


# ------------------------------------------------------------------- deterministic tests


def test_lock_is_resolved_and_hashed_for_every_pin():
    text = LOCK_PATH.read_text(encoding="utf-8")
    header = text.splitlines()[:2]
    assert header[0].startswith("# This file was autogenerated by uv")
    assert "uv pip compile pyproject.toml" in header[1]
    assert "--generate-hashes" in header[1] and "--extra dev" in header[1]
    assert "--universal" in header[1] and "--python-version 3.12" in header[1]
    assert "--index-url" not in text and "--extra-index-url" not in text

    entries = parse_lock(text)
    assert entries, "empty lock"
    for name, entry in entries.items():
        assert entry["hashes"], f"{name} has no sha256 hash"
        assert len(set(entry["hashes"])) == len(entry["hashes"]), name

    pins = pinned_requirements(load_pyproject()["project"])
    for name, version in pins.items():
        assert name in entries, f"pyproject pin {name} missing from requirements.lock"
        assert entries[name]["version"] == version, (
            f"{name}: pyproject pins {version}, lock has {entries[name]['version']}")
    assert pins["hatchling"] == load_pyproject()["build-system"]["requires"][0].split("==")[1]


def test_lock_parser_rejects_unhashed_and_unpinned_lines():
    entries = parse_lock("a==1.0 \\\n    --hash=sha256:" + "0" * 64 + "\n")
    assert entries == {"a": {"version": "1.0", "hashes": ["0" * 64]}}
    with pytest.raises(pytest.fail.Exception):
        parse_lock("a>=1.0\n")
    with pytest.raises(pytest.fail.Exception):
        parse_lock("-e .\n")


def test_metadata_policy_has_no_unresolved_release_claims():
    metadata = load_pyproject()
    project = metadata["project"]
    assert project["license"] == "Apache-2.0"
    # The LICENSE text is owned by P6-U5; until then no license file is claimed.
    assert project["license-files"] in ([], ["LICENSE"])
    assert project["requires-python"] == ">=3.12,<3.15"
    assert project["scripts"] == {"podmuncher": "podmuncher:main"}
    version = Version(project["version"])
    assert version.local is None, project["version"]
    classifiers = set(project["classifiers"])
    assert {f"Programming Language :: Python :: 3.{minor}" for minor in (12, 13, 14)} \
        <= classifiers
    assert not any(re.search(r"Python :: 3\.(?:[0-9]|1[01]|1[5-9])$", c) for c in classifiers)
    assert {"Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS"} <= classifiers
    assert not any("OS Independent" in c for c in classifiers)
    status = [c for c in classifiers if c.startswith("Development Status ::")]
    assert status and all(int(c.split("::")[1].split("-")[0]) <= 3 for c in status), status
    assert "urls" not in project and "Download-URL" not in json.dumps(project)


def test_hatch_config_ships_assets_and_excludes_forbidden_paths():
    build = load_pyproject()["tool"]["hatch"]["build"]
    assert _REQUIRED_EXCLUDES <= set(build["exclude"]), _REQUIRED_EXCLUDES - set(build["exclude"])
    wheel = build["targets"]["wheel"]
    assert wheel["packages"] == ["src/podmuncher"]
    assert "src/podmuncher/assets/**" in wheel.get("artifacts", [])
    sdist = build["targets"]["sdist"]["include"]
    assert "/requirements.lock" in sdist
    assert not any(forbidden_member(path.strip("/")) for path in sdist), sdist
    for asset in WHEEL_ASSETS:
        assert (PROJECT_ROOT / "src" / asset).is_file(), asset


def test_forbidden_member_policy_examples():
    for bad in ("HN844/episode.mp3", "research/notes.txt", "Docs/.foreman-ledger.json",
                "podmuncher/ca.pem", "podmuncher/x.crt", "podmuncher/.env",
                "podmuncher/models/model.safetensors", "podmuncher/transcripts/a.txt",
                "podmuncher/__pycache__/cli.cpython-312.pyc"):
        assert forbidden_member(bad), bad
    for good in (*WHEEL_ASSETS, "podmuncher/model_cache.py", "podmuncher/manifests/release.json",
                 "podmuncher-1.0.2.dist-info/METADATA"):
        assert not forbidden_member(good), good


def test_built_wheel_contents_and_metadata(tmp_path):
    report = inspect_wheel(build_wheel(tmp_path / "dist"))
    assert_wheel_policy(report)


def test_absent_production_manifest_is_not_an_import_error(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c",
         "import podmuncher.manifests, podmuncher.bootstrap, podmuncher.tui; print('ok')"],
        cwd=tmp_path, env=make_clean_env(tmp_path, PYTHONPATH=str(PROJECT_ROOT / "src")),
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "ok"


# ----------------------------------------------------------------------------- live smoke

_TUI_SCRIPT = textwrap.dedent('''
    import asyncio, sys
    from datetime import datetime
    from pathlib import Path
    import podmuncher.tui as tui

    site = Path(sys.argv[1]).resolve()
    assert Path(tui.__file__).resolve().is_relative_to(site), tui.__file__
    SAMPLE = "2026-10-13T10:00:00Z"
    NOW = datetime.fromisoformat("2026-10-13T10:00:02+00:00").timestamp()

    def measure(value, status="ok", reason=None):
        return {"value": value, "status": status, "sample_time": SAMPLE, "reason": reason}

    na = measure(None, "not_applicable", "duration_unknown")
    SNAPSHOT = {
        "schema_version": 1, "sample_time": SAMPLE, "freshness": "ok",
        "service": measure("running"), "docker": measure(True), "worker": measure(True),
        "host_ram": measure(8 * 1024 ** 3), "worker_ram": measure(2 * 1024 ** 3),
        "cpu": measure(12.5), "gpu_capability": measure(False),
        "gpu_utilization": measure(None, "not_applicable", "gpu_unavailable"),
        "gpu_vram": measure(None, "not_applicable", "gpu_unavailable"),
        "model": measure("small"), "storage": [], "budget_confidence": "measured",
        "job_id": None, "attempt_id": None, "stage": None,
        "progress": na, "elapsed_seconds": na, "stage_eta_seconds": na,
    }

    class FakeClient:
        def snapshot(self):
            return {"snapshot": SNAPSHOT}
        def list_jobs(self, limit=None, cursor=None):
            return {"jobs": [], "next_cursor": None}
        def list_cache(self, limit=None, cursor=None):
            return {"entries": [], "next_cursor": None}
        def close(self):
            pass

    async def main():
        app = tui.PodmuncherApp(
            lambda: FakeClient(), poll_seconds=0.05,
            clock=lambda: NOW,
            doctor=lambda: {"read_only": True, "status": "ok", "checks": {}},
            request_builder=lambda text, transcript, video: {"input": text},
        )
        async with app.run_test(size=(80, 24)) as pilot:
            for _ in range(250):
                if app.model.phase == "connected":
                    break
                await pilot.pause(0.02)
            assert app.model.phase == "connected", app.model.phase
            await pilot.pause()
            assert app.size.width == 80 and app.size.height == 24
            readiness = str(app.screen.query_one("#readiness").content)
            tabs = str(app.screen.query_one("#tabs").content)
            assert readiness.startswith("Ready"), readiness
            assert "Overview" in tabs, tabs
        print("TUI_OK", readiness.splitlines()[0][:60])

    asyncio.run(main())
''')


def _venv_paths(venv: Path) -> tuple[Path, Path, Path]:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe", venv / "Scripts" / "podmuncher.exe", \
            venv / "Lib" / "site-packages"
    python = venv / "bin" / "python"
    site = next((venv / "lib").glob("python3.*")) / "site-packages"
    return python, venv / "bin" / "podmuncher", site


@pytest.mark.live
def test_clean_install_smoke_outside_checkout():
    uv = find_uv()
    work = Path(tempfile.mkdtemp(prefix="podmuncher-dist-")).resolve()
    assert not work.is_relative_to(PROJECT_ROOT), "smoke must run outside the checkout"
    try:
        wheel = build_wheel(work / "dist")
        report = inspect_wheel(wheel)
        assert_wheel_policy(report)
        print(f"wheel: {wheel.name} members={len(report['names'])} assets={list(WHEEL_ASSETS)}"
              f" forbidden={report['forbidden']}")

        uv_env = {k: v for k, v in os.environ.items()
                  if k.upper() not in {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"}}
        venv = work / "venv"
        run_checked([uv, "venv", "--native-tls", "--no-project", "--python", sys.executable,
                     venv], cwd=work, env=uv_env, timeout=120, what="uv venv")
        python, script, site = _venv_paths(venv)
        run_checked([uv, "pip", "install", "--native-tls", "--python", python,
                     "--require-hashes", "-r", LOCK_PATH],
                    cwd=work, env=uv_env, timeout=420, what="locked dependency install")
        run_checked([uv, "pip", "install", "--native-tls", "--python", python, "--no-deps",
                     wheel], cwd=work, env=uv_env, timeout=120, what="wheel install")
        run_checked([uv, "pip", "check", "--python", python], cwd=work, env=uv_env,
                    timeout=60, what="uv pip check")

        home = work / "home"
        home.mkdir()
        cwd = work / "cwd"
        cwd.mkdir()
        env = make_clean_env(home)

        for argument in ("--help", "--version"):
            out = run_checked([script, argument], cwd=cwd, env=env, timeout=60,
                              what=f"podmuncher {argument}")
            assert "podmuncher" in out.stdout and out.stderr == "", out.stderr
            print(f"$ podmuncher {argument} -> exit 0: {out.stdout.splitlines()[0]}")
        light = run_checked(
            [python, "-c",
             ("import sys, podmuncher; from podmuncher import main\n"
             "from pathlib import Path\n"
             f"assert Path(podmuncher.__file__).resolve().is_relative_to(Path({str(site)!r}))\n"
             "try:\n    main(['--version'])\nexcept SystemExit as e:\n    assert e.code == 0\n"
             f"heavy = sorted(n for n in sys.modules if n.split('.')[0] in {HEAVY_MODULES!r})\n"
             "assert not heavy, heavy")],
            cwd=cwd, env=env, timeout=60, what="lightweight import check")
        assert light.stdout.strip() == "podmuncher 1.0.2"

        state = work / "state"
        status_env = make_clean_env(home, PODMUNCHER_STATE_DIR=str(state))
        for _ in range(2):  # the second read proves the first did not start the service
            out = run_checked([script, "status", "--json"], cwd=cwd, env=status_env,
                              timeout=60, what="podmuncher status --json")
            doc = json.loads(out.stdout)
            assert doc["ok"] is True and doc["exit_code"] == 0, doc
            assert doc["service_state"] == "stopped", doc
        print(f"$ podmuncher status --json -> exit 0: service_state={doc['service_state']}")

        tui_script = work / "tui_smoke.py"
        tui_script.write_text(_TUI_SCRIPT, encoding="utf-8")
        out = run_checked([python, "-I", tui_script, site], cwd=cwd, env=env, timeout=120,
                          what="TUI 80x24 startup")
        assert "TUI_OK" in out.stdout, out.stdout
        print(f"$ tui 80x24 -> {out.stdout.strip().splitlines()[-1]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    assert not work.exists(), f"temp dir not removed: {work}"
    print(SMOKE_MARKER)
