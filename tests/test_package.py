"""Package contract and side-effect-free command smoke tests."""

import builtins
import importlib
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest

import podmuncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_command_env(tmp_path: Path) -> dict[str, str]:
    """Run the source entry point with isolated homes and disabled bytecode writes."""
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(PROJECT_ROOT / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        HOME=str(tmp_path),
        USERPROFILE=str(tmp_path),
        APPDATA=str(tmp_path),
        LOCALAPPDATA=str(tmp_path),
        XDG_CONFIG_HOME=str(tmp_path),
        XDG_CACHE_HOME=str(tmp_path),
        XDG_DATA_HOME=str(tmp_path),
        TMPDIR=str(tmp_path),
        TEMP=str(tmp_path),
        TMP=str(tmp_path),
    )
    return env


@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_information_commands_have_no_side_effects(argument, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    for key, value in make_command_env(tmp_path).items():
        if os.environ.get(key) != value:
            monkeypatch.setenv(key, value)

    def forbidden(*args, **kwargs):
        pytest.fail("information command attempted Docker, network, or filesystem writes")

    original_open = builtins.open
    original_os_open = os.open

    def read_only_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            forbidden()
        return original_open(file, mode, *args, **kwargs)

    def read_only_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            forbidden()
        return original_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read_only_open)
    monkeypatch.setattr(os, "open", read_only_os_open)
    monkeypatch.setattr(os, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "touch", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(shutil, "which", forbidden)

    with pytest.raises(SystemExit) as result:
        podmuncher.main([argument])
    assert result.value.code == 0
    output = capsys.readouterr()
    assert "podmuncher" in output.out
    assert output.err == ""
    if argument == "--help":
        assert "usage:" in output.out
        assert "--version" in output.out
    else:
        assert output.out.strip() == f"podmuncher {podmuncher.__version__}"
    assert list(tmp_path.iterdir()) == []


def test_package_metadata_policy():
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    assert project["name"] == "podmuncher"
    assert project["version"] == podmuncher.__version__ == "0.1.0.dev0"
    assert project["requires-python"] == ">=3.12,<3.15"
    assert project["license"] == "Apache-2.0"
    assert metadata["build-system"]["build-backend"] == "hatchling.build"
    module_name, function_name = project["scripts"]["podmuncher"].split(":")
    assert (module_name, function_name) == ("podmuncher", "main")
    assert getattr(importlib.import_module(module_name), function_name) is podmuncher.main
    build = metadata["tool"]["hatch"]["build"]
    assert build["targets"]["wheel"]["packages"] == ["src/podmuncher"]

    def dependency_names(requirements):
        parsed = [Requirement(value) for value in requirements]
        assert all(requirement.specifier for requirement in parsed)
        return {canonicalize_name(requirement.name) for requirement in parsed}

    runtime_names = dependency_names(project["dependencies"])
    assert {
        "textual", "aiohttp", "psutil", "platformdirs", "opentelemetry-api",
        "opentelemetry-sdk", "opentelemetry-exporter-otlp-proto-grpc",
    } <= runtime_names
    assert runtime_names.isdisjoint({
        "torch", "torchaudio", "faster-whisper", "openai-whisper", "vosk",
        "speechrecognition", "transformers", "onnxruntime",
    })
    assert {"pytest", "pytest-asyncio", "build", "hatchling", "ruff"} <= dependency_names(
        project["optional-dependencies"]["dev"]
    )
    pytest_config = metadata["tool"]["pytest"]["ini_options"]
    assert pytest_config["pythonpath"] == ["src"]
    assert any(marker.startswith("live:") for marker in pytest_config["markers"])


@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_source_entrypoint_from_isolated_directory(argument, tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; from podmuncher import main; "
            "assert not any(name.split('.')[0] in "
            "{'textual', 'aiohttp', 'psutil', 'platformdirs', 'opentelemetry', 'docker'} "
            "for name in sys.modules); raise SystemExit(main())",
            argument,
        ],
        cwd=tmp_path,
        env=make_command_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "podmuncher" in result.stdout
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []
