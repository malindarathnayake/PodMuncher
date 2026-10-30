"""CLI contract, stable exits, presentation seam and live transport smoke (P5-U3)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from podmuncher import cli, service
from podmuncher.domains import DomainValidationError, JobState, ReasonCode

ROOT = Path(__file__).resolve().parents[1]
JOB = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
CREDENTIAL = "credential-" + "Z9" * 12
HEAVY = {"textual", "aiohttp", "psutil", "platformdirs", "opentelemetry", "docker"}
LATE = ("podmuncher.service", "podmuncher.config", "podmuncher.tui", "podmuncher.setup",
        "podmuncher.presentation")


# -- helpers -------------------------------------------------------------------------


def make_job(state="queued", reason="ok", job_id=JOB):
    return {"job_id": job_id, "state": state, "reason_code": reason, "next_action": "none",
            "source_kind": "local", "output_mode": "transcript", "attempt_id": None,
            "created_at": "2026-10-13T00:00:00Z", "updated_at": "2026-10-13T00:00:00Z",
            "artifacts": []}


class FakeServiceClient:
    """Scripted responses; records calls; closing never cancels."""

    def __init__(self, states=("queued",), error=None):
        self.calls = []
        self.closed = 0
        self.states = list(states)
        self.error = error
        self._credential = CREDENTIAL  # must never reach any output

    def _raise(self):
        if self.error is not None:
            raise self.error

    def _state(self):
        item = self.states[0] if len(self.states) == 1 else self.states.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed += 1

    def submit(self, request, key):
        self.calls.append(("submit", request, key))
        self._raise()
        return {"job": make_job(self._state()), "replayed": False}

    def retry(self, job_id, key):
        self.calls.append(("retry", job_id, key))
        self._raise()
        return {"job": make_job(self._state()), "attempt_id": OTHER, "replayed": False}

    def clone(self, job_id, request, key):
        self.calls.append(("clone", job_id, request, key))
        self._raise()
        return {"job": make_job(self._state(), job_id=OTHER)}

    def get_job(self, job_id):
        self.calls.append(("get_job", job_id))
        self._raise()
        return {"job": make_job(self._state(), job_id=job_id)}

    def list_jobs(self, limit=None, cursor=None):
        self.calls.append(("list_jobs", limit, cursor))
        return {"jobs": [make_job()], "next_cursor": None}

    def cancel(self, job_id):
        self.calls.append(("cancel", job_id))
        self._raise()
        return {"job": make_job("cancelled"), "status": "intent"}

    def recheck(self, job_id):
        self.calls.append(("recheck", job_id))
        self._raise()
        return {"job": make_job("blocked", "storage_insufficient"),
                "preflight": {"result": "ready"}, "ready_to_retry": self.states[0] == "ready"}

    def list_cache(self, limit=None, cursor=None):
        self.calls.append(("list_cache", limit, cursor))
        return {"entries": [{"entry_id": "e1"}], "next_cursor": None}

    def remove_cache(self, entry_id):
        self.calls.append(("remove_cache", entry_id))
        return {"removed": True, "entry_id": entry_id}


class FakeAccess:
    """Records every service touch.  ``client=None`` models a stopped service."""

    def __init__(self, client=None, status=None, stop=None):
        self.client = client
        self.status = status or {"state": "stopped", "snapshot": {"jobs": 3}, "stale": True}
        self.stop_outcome = stop
        self.touched = []
        self.keys = 0

    def connect(self):
        self.touched.append("connect")
        return self.client

    def ensure(self):
        self.touched.append("ensure")
        if self.client is None:
            raise AssertionError("ensure with no client")
        return self.client

    def start(self):
        self.touched.append("start")
        return {"state": "running", "pid": 99}

    def stop(self):
        self.touched.append("stop")
        return self.stop_outcome

    def read_status(self):
        self.touched.append("read_status")
        return self.status

    def new_key(self):
        self.keys += 1
        return f"key-{self.keys}"


@pytest.fixture(autouse=True)
def reset_seams():
    cli.configure_presentation(None)
    cli.set_client_startup(None)
    cli.set_registry_provider(None)
    yield
    cli.configure_presentation(None)
    cli.set_client_startup(None)
    cli.set_registry_provider(None)


HOST_OK = {"read_only": True, "status": "ok", "checks": {
    name: {"status": "ok", "reason_code": "ok", "detail": {}}
    for name in ("paths", "host_capacity", "docker", "backing", "cache")}}


@pytest.fixture(autouse=True)
def host_doctor(monkeypatch):
    """Unit tests never probe the real host: doctor_report is stubbed to all-ok by default."""
    calls = []

    def report(paths=None, **kwargs):
        calls.append(paths)
        return json.loads(json.dumps(HOST_OK))

    monkeypatch.setattr("podmuncher.setup.doctor_report", report)
    return calls


@pytest.fixture
def env(tmp_path):
    return {"PODMUNCHER_CONFIG": str(tmp_path / "absent.toml"), "TERM": "xterm"}


def invoke(argv, access, capsys, env, **kwargs):
    code = cli.main(argv, access=access, sleep=lambda _: None, environ=env, **kwargs)
    out = capsys.readouterr()
    return code, out.out, out.err


def run_cli(*args, env=None, timeout=60, tmp_path=None, stdin=subprocess.DEVNULL):
    """Run the CLI in a subprocess from an isolated working directory."""
    home = str(tmp_path)
    full = os.environ.copy()
    full.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", HOME=home,
                USERPROFILE=home, APPDATA=home, LOCALAPPDATA=home, XDG_CONFIG_HOME=home,
                XDG_CACHE_HOME=home, XDG_DATA_HOME=home, TEMP=home, TMP=home, TMPDIR=home)
    full.update(env or {})
    return subprocess.run(
        [sys.executable, "-c", "import sys, podmuncher; sys.exit(podmuncher.main(sys.argv[1:]))",
         *args],
        cwd=tmp_path, env=full, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
        stdin=stdin, check=False)


class FakeTty(io.StringIO):
    def isatty(self):
        return True


# -- parser --------------------------------------------------------------------------

PARSE_CASES = [
    (["status"], "status"),
    (["jobs", "--limit", "5", "--cursor", "c"], "jobs"),
    (["job", JOB], "job"),
    (["cancel", JOB], "cancel"),
    (["recheck", JOB], "recheck"),
    (["retry", JOB, "--detach"], "retry"),
    (["clone", JOB, "in.mp3", "--video", "--model", "auto", "--device", "cuda"], "clone"),
    (["doctor"], "doctor"),
    (["service", "start"], "service"),
    (["service", "stop"], "service"),
    (["cache", "list", "--limit", "2"], "cache"),
    (["cache", "remove", "entry"], "cache"),
    (["tui"], "tui"),
    (["--setup"], "setup"),
    (["--json", "status"], "status"),
    (["status", "--json", "--no-color"], "status"),
    (["in.mp3"], "submit"),
    (["in.mp3", "--video", "--no-transcript", "--model", "m", "--device", "cpu", "--profile",
      "quality", "--output", "o", "--language", "en", "--detach", "--json", "--no-color"],
     "submit"),
    (["--json", "in.mp3"], "submit"),
    ([], None),
]


@pytest.mark.parametrize("argv,command", PARSE_CASES)
def test_every_command_and_flag_parses(argv, command):
    ns, _ = cli.parse_args(argv)
    assert ns.command == command


def test_flags_land_on_the_namespace():
    ns, _ = cli.parse_args(["in.mp3", "--video", "--device", "cpu", "--profile", "quality",
                            "--output", "o", "--language", "en", "--model", "m", "--detach",
                            "--no-color", "--json"])
    assert (ns.video, ns.no_transcript, ns.device, ns.profile, ns.output, ns.language,
            ns.model, ns.detach, ns.no_color, ns.json) == (
        True, False, "cpu", "quality", "o", "en", "m", True, True, True)
    leading, _ = cli.parse_args(["--json", "--no-color", "status"])
    assert leading.json and leading.no_color


def test_video_only_is_valid_and_builds_video_only_request(env):
    ns, _ = cli.parse_args(["in.mp3", "--video", "--no-transcript"])
    request = cli.build_request(ns, env)
    assert request["outputs"]["video"] is True and request["outputs"]["transcript"] is False
    plain, _ = cli.parse_args(["in.mp3"])
    default = cli.build_request(plain, env)
    assert default["outputs"]["video"] is False and default["outputs"]["transcript"] is True
    assert default["source"]["kind"] == "local"


@pytest.mark.parametrize("argv", [
    ["in.mp3", "--no-transcript"],
    ["clone", JOB, "in.mp3", "--no-transcript"],
    ["in.mp3", "--device", "tpu"],
    ["in.mp3", "--profile", "fast"],
    ["--video"],
    ["--setup", "in.mp3"],
])
def test_invalid_usage_exits_two_with_message_on_stderr(argv, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.parse_args(argv)
    assert stop.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err


def test_no_transcript_alone_exits_two_through_main(capsys, env):
    access = FakeAccess()
    with pytest.raises(SystemExit) as stop:
        cli.main(["in.mp3", "--no-transcript"], access=access, environ=env)
    assert stop.value.code == 2
    assert "--no-transcript" in capsys.readouterr().err
    assert access.touched == []


def test_registry_negatives_are_rejected_for_device_profile_and_reason(capsys):
    registry = json.loads((ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    for negative in registry["negative_examples"]:
        for flag in ("--device", "--profile"):
            with pytest.raises(SystemExit) as stop:
                cli.parse_args(["in.mp3", flag, negative])
            assert stop.value.code == 2
        with pytest.raises(DomainValidationError):
            ReasonCode.parse(negative)
        assert cli.exit_for_reason(negative) == cli.EXIT_FAILED
    capsys.readouterr()
    for value in registry["device"]:
        assert cli.parse_args(["in.mp3", "--device", value])[0].device == value
    for value in registry["profile"]:
        assert cli.parse_args(["in.mp3", "--profile", value])[0].profile == value


# -- rendering -----------------------------------------------------------------------


def test_plain_and_json_render_from_the_same_result(capsys, env):
    stopped = {"state": "stopped", "snapshot": {"jobs": 3}, "stale": True}
    code, out_json, _ = invoke(["status", "--json"], FakeAccess(status=stopped), capsys, env)
    assert code == 0
    result = json.loads(out_json)
    assert result["reason_code"] == "ok" and result["stale"] is True
    assert result["service_state"] == "stopped" and result["next_action"] == cli.START_HINT
    code, out_plain, _ = invoke(["status"], FakeAccess(status=stopped), capsys, env)
    assert code == 0
    assert out_plain.rstrip("\n") == cli.PlainPresentation().render(result, command="status")
    assert "stale: true" in out_plain


def test_hostile_text_is_escaped_in_plain_and_preserved_in_json(capsys, env):
    bs, bidi = chr(92), chr(0x202E)
    esc, c1, nul = chr(0x1B), chr(0x9B), chr(0)
    hostile = esc + "[31m[bold]x[/bold]" + c1 + bidi + nul + bs + "n"
    client = FakeServiceClient()
    code, plain, _ = invoke(["cache", "remove", hostile], FakeAccess(client), capsys, env)
    assert code == 0
    for raw in (esc, c1, bidi, nul):
        assert raw not in plain
    assert bs + "x1b" in plain and bs + "u009b" in plain and bs + "[bold" + bs + "]" in plain
    lines = plain.splitlines()  # no injected line breaks: header plus three indented fields
    assert len(lines) == 4 and all(line.startswith("  ") for line in lines[1:])
    code, raw, _ = invoke(["cache", "remove", hostile, "--json"], FakeAccess(client), capsys, env)
    assert json.loads(raw)["entry_id"] == hostile
    adapter = cli.PlainPresentation()
    assert adapter.escape("a" + chr(10) + "b" + chr(9) + chr(127)) == (
        "a" + bs + "x0ab" + bs + "x09" + bs + "x7f")
    assert adapter.escape("plain text " + chr(0xE9) + chr(0x65E5)) == "plain text " + chr(0xE9) + chr(0x65E5)


def test_progress_lines_go_through_the_adapter(capsys, env, tmp_path):
    client = FakeServiceClient(["queued", "transcribing", "completed"])
    code, out, _ = invoke(["in.mp3", "--output", str(tmp_path)], FakeAccess(client), capsys, env)
    assert code == 0
    lines = out.splitlines()
    assert lines[:4] == ["Transcribing in.mp3", "  Queued...", "  Transcribing...", "Done."]
    assert lines[-1] == f"Job:        {JOB}"


def test_finished_job_reports_where_its_files_are(capsys, env, tmp_path):
    client = FakeServiceClient(["queued", "completed"])
    code, out, _ = invoke(["in.mp3", "--output", str(tmp_path), "--json"], FakeAccess(client), capsys, env)
    assert code == 0
    result = json.loads(out)
    assert result["output_dir"] == os.path.join(str(tmp_path), f"podmuncher-{JOB}")
    assert "Transcribing" not in out  # --json prints only the result


def test_stopped_job_says_what_happened_and_the_next_command():
    plain = cli.PlainPresentation()
    blocked = plain._job_summary({"job": {"state": "blocked"}, "job_id": JOB,
                                  "reason_code": "storage_insufficient", "next_action": "recheck"})
    assert blocked.splitlines()[:2] == [
        "Blocked: not enough free disk space (Podmuncher keeps up to 10 GB free on each drive it uses)"
        " (storage_insufficient)",
        f"Next:       free up disk space, then run: podmuncher recheck {JOB}",
    ]
    failed = plain._job_summary({"job": {"state": "failed"}, "job_id": JOB, "reason_code": "no_audio"})
    assert failed.splitlines()[:2] == ["Failed: the media has no audio (no_audio)",
                                       f"Next:       podmuncher retry {JOB}"]
    unknown = plain._job_summary({"job": {"state": "failed"}, "job_id": JOB, "reason_code": "surprise",
                                  "next_action": "retry_later"})
    assert unknown.splitlines()[:2] == ["Failed (surprise)", f"Next:       podmuncher retry {JOB}"]
    assert set(cli._REASON_HINTS) <= {c.value for c in ReasonCode}


# -- exits ---------------------------------------------------------------------------


def test_reason_table_is_total_and_single_valued():
    assert set(cli.EXIT_FOR_REASON) == set(ReasonCode)
    assert set(cli.EXIT_FOR_REASON.values()) <= {0, 2, 3, 4}
    for code in ReasonCode:
        assert cli.exit_for_reason(code) == cli.exit_for_reason(code.value)
        assert cli.exit_for_reason(code) == cli.EXIT_FOR_REASON[code]
    groups = (cli._EXIT_0, cli._EXIT_2, cli._EXIT_3)
    assert sum(len(g) for g in groups) == len(frozenset().union(*groups))  # disjoint
    expected = {
        0: {"ok", "already_terminal"},
        2: {"invalid_input", "invalid_config", "input_rejected", "source_changed",
            "nesting_limit", "unsupported_schema"},
        3: {"storage_insufficient", "storage_unknown", "storage_stale", "storage_guard_tripped",
            "quota_exceeded", "ram_insufficient", "gpu_unavailable", "docker_unavailable",
            "docker_context_unsupported", "backing_unresolved", "queue_full", "audit_loss"},
    }
    for exit_code, names in expected.items():
        assert {c.value for c in ReasonCode if cli.EXIT_FOR_REASON[c] == exit_code} == names
    for name in ("service_unavailable", "writer_unresolved", "worker_exit_unexpected",
                 "rate_limited", "unauthorized", "model_oom"):
        assert cli.exit_for_reason(name) == 4
    assert cli.exit_for_reason("not-a-code") == 4 and cli.exit_for_reason(None) == 4


@pytest.mark.parametrize("state,exit_code", [
    ("completed", 0), ("partial", 5), ("blocked", 3), ("failed", 4), ("cancelled", 4),
    ("interrupted", 4), ("queued", None), ("checking", None), ("preparing", None),
    ("downloading", None), ("probing", None), ("transcribing", None), ("exporting", None),
    ("surprise", 4),
])
def test_job_state_exits(state, exit_code):
    assert cli.exit_for_job_state(state) == exit_code


def test_every_job_state_is_classified():
    for state in JobState:
        assert cli.exit_for_job_state(state.value) in (None, 0, 3, 4, 5)


@pytest.mark.parametrize("final,exit_code", [
    ("completed", 0), ("partial", 5), ("blocked", 3), ("failed", 4), ("cancelled", 4),
    ("interrupted", 4)])
def test_submit_waits_and_exits_per_final_state(final, exit_code, capsys, env, tmp_path):
    client = FakeServiceClient(["queued", "downloading", final])
    code, out, _ = invoke(["in.mp3", "--json", "--output", str(tmp_path)], FakeAccess(client),
                          capsys, env)
    assert code == exit_code
    result = json.loads(out)
    assert result["job"]["state"] == final and result["exit_code"] == exit_code
    assert [c[0] for c in client.calls] == ["submit", "get_job", "get_job"]
    assert client.closed >= 1


def test_client_errors_map_through_the_reason_table(capsys, env):
    cases = [(ReasonCode.ALREADY_TERMINAL, 0), (ReasonCode.INVALID_INPUT, 2),
             (ReasonCode.DOCKER_UNAVAILABLE, 3), (ReasonCode.SERVICE_UNAVAILABLE, 4),
             (ReasonCode.UNSUPPORTED_SCHEMA, 2)]
    for reason, exit_code in cases:
        client = FakeServiceClient(error=service.ClientError(reason, next_action="none"))
        code, out, _ = invoke(["cancel", JOB, "--json"], FakeAccess(client), capsys, env)
        assert code == exit_code
        result = json.loads(out)
        assert result["reason_code"] == reason.value
        if reason is ReasonCode.UNSUPPORTED_SCHEMA:
            assert result["next_action"] == "upgrade_client"


def test_unexpected_exceptions_exit_four_without_echoing_the_message(capsys, env):
    client = FakeServiceClient(error=RuntimeError(f"boom {CREDENTIAL}"))
    code, out, _ = invoke(["cancel", JOB], FakeAccess(client), capsys, env)
    assert code == 4 and CREDENTIAL not in out and "boom" not in out


def test_bad_job_id_exits_two_before_touching_the_service(capsys, env):
    access = FakeAccess(FakeServiceClient())
    code, out, _ = invoke(["cancel", "../x", "--json"], access, capsys, env)
    assert code == 2 and json.loads(out)["reason_code"] == "invalid_input"
    assert access.touched == []


def test_bad_source_exits_two_before_touching_the_service(capsys, env):
    access = FakeAccess(FakeServiceClient())
    code, _out, _ = invoke(["http://127.0.0.1/a.mp3", "--json"], access, capsys, env)
    assert code == 2 and access.touched == []


def test_recheck_ready_and_blocked(capsys, env):
    code, out, _ = invoke(["recheck", JOB, "--json"], FakeAccess(FakeServiceClient(["ready"])),
                          capsys, env)
    assert code == 0 and json.loads(out)["ready_to_retry"] is True
    code, out, _ = invoke(["recheck", JOB, "--json"], FakeAccess(FakeServiceClient(["blocked"])),
                          capsys, env)
    assert code == 3 and json.loads(out)["reason_code"] == "storage_insufficient"


def test_service_start_and_stop_outcomes(capsys, env):
    ok = SimpleNamespace(ok=True, outcome="stopped", reason_code=ReasonCode.OK)
    code, out, _ = invoke(["service", "stop", "--json"], FakeAccess(stop=ok), capsys, env)
    assert code == 0 and json.loads(out)["outcome"] == "stopped"
    bad = SimpleNamespace(ok=False, outcome="writer_unresolved",
                          reason_code=ReasonCode.WRITER_UNRESOLVED)
    code, out, _ = invoke(["service", "stop", "--json"], FakeAccess(stop=bad), capsys, env)
    assert code == 4 and json.loads(out)["reason_code"] == "writer_unresolved"
    code, out, _ = invoke(["service", "start", "--json"], FakeAccess(), capsys, env)
    assert code == 0 and json.loads(out)["pid"] == 99


# -- detach and Ctrl+C ---------------------------------------------------------------


def test_detach_prints_only_the_job_id(capsys, env, tmp_path):
    client = FakeServiceClient(["queued"])
    code, out, _ = invoke(["in.mp3", "--detach", "--output", str(tmp_path)], FakeAccess(client),
                          capsys, env)
    assert code == 0 and out == JOB + "\n"
    assert [c[0] for c in client.calls] == ["submit"]
    code, out, _ = invoke(["in.mp3", "--detach", "--json", "--output", str(tmp_path)],
                          FakeAccess(client), capsys, env)
    assert code == 0 and json.loads(out)["job_id"] == JOB and json.loads(out)["detached"] is True


def test_ctrl_c_detaches_with_the_job_id_and_never_cancels(capsys, env, tmp_path):
    client = FakeServiceClient(["queued", KeyboardInterrupt()])
    code, out, _ = invoke(["in.mp3", "--output", str(tmp_path)], FakeAccess(client), capsys, env)
    assert code == 130
    assert f"Detached; job {JOB} keeps running" in out
    assert "cancel" not in [c[0] for c in client.calls]
    assert client.closed >= 1
    client = FakeServiceClient(["queued", KeyboardInterrupt()])
    code, out, _ = invoke(["in.mp3", "--json", "--output", str(tmp_path)], FakeAccess(client),
                          capsys, env)
    result = json.loads(out)
    assert code == 130 and result["detached"] is True and result["job_id"] == JOB
    assert result["exit_code"] == 130
    assert "cancel" not in [c[0] for c in client.calls]


def test_ctrl_c_during_the_poll_sleep_also_detaches(capsys, env):
    client = FakeServiceClient(["queued"])

    def interrupt(_):
        raise KeyboardInterrupt

    code = cli.main(["retry", JOB], access=FakeAccess(client), sleep=interrupt, environ=env)
    out = capsys.readouterr().out
    assert code == 130 and JOB in out
    assert [c[0] for c in client.calls] == ["retry"]


def test_observer_transport_error_keeps_the_job_id(capsys, env, tmp_path):
    client = FakeServiceClient(["queued", service.ClientError(ReasonCode.SERVICE_UNAVAILABLE)])
    code, out, _ = invoke(["in.mp3", "--json", "--output", str(tmp_path)], FakeAccess(client),
                          capsys, env)
    result = json.loads(out)
    assert code == 4 and result["job_id"] == JOB and result["reason_code"] == "service_unavailable"


def test_each_invocation_uses_a_fresh_idempotency_key(capsys, env, tmp_path):
    client = FakeServiceClient(["queued"])
    access = FakeAccess(client)
    for _ in range(2):
        invoke(["in.mp3", "--detach", "--output", str(tmp_path)], access, capsys, env)
    invoke(["retry", JOB, "--detach"], access, capsys, env)
    invoke(["clone", JOB, "in.mp3", "--detach", "--output", str(tmp_path)], access, capsys, env)
    assert [c[-1] for c in client.calls] == ["key-1", "key-2", "key-3", "key-4"]
    clone = client.calls[-1]
    assert clone[0] == "clone" and clone[1] == JOB and clone[2]["source"]["input"].endswith("in.mp3")


# -- non-TTY and no-argument behaviour -----------------------------------------------


def test_no_args_without_a_tty_prints_help_and_touches_nothing(tmp_path):
    state = tmp_path / "state"
    for term in ("xterm", "dumb"):
        result = run_cli(env={"TERM": term, "PODMUNCHER_STATE_DIR": str(state)}, tmp_path=tmp_path)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout and "status" in result.stdout
        assert result.stderr == ""
        assert not state.exists()
    assert list(tmp_path.iterdir()) == []


def test_no_args_with_dumb_terminal_prints_help_even_on_a_tty(monkeypatch, env):
    out = FakeTty()
    monkeypatch.setattr(sys, "stdin", FakeTty())
    monkeypatch.setattr(sys, "stdout", out)
    access = FakeAccess()
    code = cli.main([], access=access, environ={**env, "TERM": "dumb"})
    assert code == 0 and "usage:" in out.getvalue() and access.touched == []


def test_no_args_on_a_tty_dispatches_to_the_tui_lazily(monkeypatch, env):
    out = FakeTty()
    monkeypatch.setattr(sys, "stdin", FakeTty())
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setitem(sys.modules, "podmuncher.tui", None)
    access = FakeAccess()
    assert cli.main([], access=access, environ=env) == 4
    assert "implementation pending" in out.getvalue() and access.touched == []


def test_non_tty_commands_never_prompt(tmp_path):
    result = run_cli("status", "--json", env={"PODMUNCHER_STATE_DIR": str(tmp_path / "state")},
                     tmp_path=tmp_path, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["service_state"] == "stopped"
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_help_and_version_are_immediate_and_lazy(argument, tmp_path):
    code = ("import sys, podmuncher; rc = None\n"
            "try:\n    podmuncher.main(sys.argv[1:])\nexcept SystemExit as stop:\n    rc = stop.code\n"
            f"bad = [n for n in sys.modules if n.split('.')[0] in {sorted(HEAVY)!r}]\n"
            f"late = [n for n in {LATE!r} if n in sys.modules]\n"
            "print(rc, bad, late)\n")
    result = subprocess.run([sys.executable, "-c", code, argument], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
                                 "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "0 [] []"


def test_cli_import_is_lazy():
    code = ("import sys, podmuncher.cli\n"
            f"bad = [n for n in sys.modules if n.split('.')[0] in {sorted(HEAVY)!r}]\n"
            f"late = [n for n in {LATE!r} if n in sys.modules]\n"
            "print(bad, late)\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
                                 "PYTHONDONTWRITEBYTECODE": "1"}, timeout=30, check=False)
    assert result.stdout.strip() == "[] []", result.stderr


# -- service handling ----------------------------------------------------------------


def test_read_only_commands_never_start_or_ensure_the_service(capsys, env):
    for argv in (["status"], ["doctor"], ["jobs"], ["job", JOB], ["cache", "list"]):
        access = FakeAccess()
        code, out, _ = invoke([*argv, "--json"], access, capsys, env)
        assert not {"ensure", "start", "stop"} & set(access.touched), argv
        result = json.loads(out)
        assert result["next_action"] == cli.START_HINT and result["stale"] is True
        if argv[0] in ("status", "doctor", "jobs"):
            assert code == 0
        else:
            assert code == 4 and result["reason_code"] == "service_unavailable"


def test_stopped_status_shows_the_stale_persisted_snapshot(capsys, env):
    code, out, _ = invoke(["status", "--json"], FakeAccess(), capsys, env)
    result = json.loads(out)
    assert code == 0 and result["stale"] is True and result["snapshot"] == {"jobs": 3}
    code, out, _ = invoke(["doctor", "--json"], FakeAccess(), capsys, env)
    assert {"name": "service", "status": "stopped"} in json.loads(out)["checks"]
    unreachable = {"state": "unreachable", "snapshot": None, "stale": True}
    code, out, _ = invoke(["status", "--json"], FakeAccess(status=unreachable), capsys, env)
    assert code == 4 and json.loads(out)["reason_code"] == "service_unavailable"


def test_doctor_merges_host_checks_and_never_starts(capsys, env, host_doctor):
    access = FakeAccess()
    code, out, _ = invoke(["doctor", "--json"], access, capsys, env)
    result = json.loads(out)
    assert code == 0 and result["next_action"] == cli.START_HINT
    assert [c["name"] for c in result["checks"]] == [
        "service", "snapshot", "paths", "host_capacity", "docker", "backing", "cache"]
    assert all(c["status"] == "ok" for c in result["checks"][2:])
    assert host_doctor == [None]
    assert not {"ensure", "start", "stop", "connect"} & set(access.touched)


def test_doctor_blocked_host_check_exits_three(monkeypatch, capsys, env):
    report = json.loads(json.dumps(HOST_OK))
    report["status"] = "blocked"
    report["checks"]["docker"] = {"status": "unavailable", "reason_code": "docker_unavailable",
                                  "detail": {}}
    monkeypatch.setattr("podmuncher.setup.doctor_report", lambda paths=None, **kw: report)
    for status in (None, {"state": "running", "snapshot": None, "stale": False}):
        access = FakeAccess(status=status)
        code, out, _ = invoke(["doctor", "--json"], access, capsys, env)
        result = json.loads(out)
        assert code == 3 and result["reason_code"] == "docker_unavailable"
        assert result["blocked_check"] == "docker" and result["next_action"]
        assert {"name": "docker", "status": "unavailable", "reason_code": "docker_unavailable",
                "detail": {}} in result["checks"]
        assert not {"ensure", "start", "stop"} & set(access.touched)
    code, out, _ = invoke(["doctor"], FakeAccess(), capsys, env)
    assert code == 3 and "docker_unavailable" in out


def test_doctor_report_failure_exits_four_closed(monkeypatch, capsys, env):
    def broken(paths=None, **kw):
        raise RuntimeError(CREDENTIAL)

    monkeypatch.setattr("podmuncher.setup.doctor_report", broken)
    access = FakeAccess()
    code, out, err = invoke(["doctor", "--json"], access, capsys, env)
    result = json.loads(out)
    assert code == 4 and result["reason_code"] == "service_unavailable"
    assert result["host_checks"] == "unavailable" and len(result["checks"]) == 2
    assert CREDENTIAL not in out + err and not {"ensure", "start"} & set(access.touched)


def test_running_read_only_commands_use_the_connected_client(capsys, env):
    client = FakeServiceClient(["transcribing"])
    access = FakeAccess(client)
    code, out, _ = invoke(["jobs", "--limit", "3", "--json"], access, capsys, env)
    assert code == 0 and json.loads(out)["jobs"][0]["job_id"] == JOB
    code, out, _ = invoke(["job", JOB, "--json"], access, capsys, env)
    assert code == 0 and json.loads(out)["job"]["state"] == "transcribing"
    code, out, _ = invoke(["cache", "list", "--json"], access, capsys, env)
    assert code == 0 and json.loads(out)["entries"] == [{"entry_id": "e1"}]
    assert access.touched == ["connect", "connect", "connect"]
    assert ("list_jobs", 3, None) in client.calls and client.closed == 3


def test_job_command_reports_the_state_exit(capsys, env):
    code, out, _ = invoke(["job", JOB, "--json"], FakeAccess(FakeServiceClient(["partial"])),
                          capsys, env)
    assert code == 5 and json.loads(out)["job"]["state"] == "partial"


def test_default_access_never_starts_the_service_for_read_only_commands(monkeypatch, capsys,
                                                                       tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("start_service called")

    monkeypatch.setattr(service, "start_service", forbidden)
    monkeypatch.setattr(service, "connect", lambda state_dir=None, **kw: None)
    monkeypatch.setattr(service, "read_status", lambda state_dir=None, **kw: {
        "state": "stopped", "snapshot": {"jobs": 1}, "stale": True})
    env = {"PODMUNCHER_STATE_DIR": str(tmp_path / "state"), "TERM": "xterm"}
    for argv in (["status"], ["jobs"], ["doctor"], ["job", JOB], ["cache", "list"]):
        cli.main([*argv, "--json"], environ=env)
        capsys.readouterr()
    assert not (tmp_path / "state").exists()


def test_default_access_starts_only_for_start_capable_commands(monkeypatch, capsys, tmp_path):
    started = []
    record = SimpleNamespace(pid=77, port=1234, credential=CREDENTIAL, instance_nonce="n")
    client = FakeServiceClient(["queued"])
    monkeypatch.setattr(service, "connect", lambda state_dir=None, **kw: None)
    monkeypatch.setattr(service, "start_service",
                        lambda state_dir=None, **kw: started.append(state_dir) or record)
    monkeypatch.setattr(service, "ServiceClient", lambda rec: client)
    env = {"PODMUNCHER_STATE_DIR": str(tmp_path / "state"), "TERM": "xterm"}
    assert cli.main(["service", "start", "--json"], environ=env) == 0
    out = capsys.readouterr().out
    assert CREDENTIAL not in out and "1234" not in out and json.loads(out)["pid"] == 77
    assert cli.main(["cancel", JOB, "--json"], environ=env) == 0
    assert CREDENTIAL not in capsys.readouterr().out
    assert started == [str(tmp_path / "state")] * 2


def test_service_failure_during_start_maps_through_the_table(monkeypatch, capsys, tmp_path):
    def blocked(state_dir=None, **kw):
        raise service.ServiceError(ReasonCode.STORAGE_UNKNOWN)

    monkeypatch.setattr(service, "connect", lambda state_dir=None, **kw: None)
    monkeypatch.setattr(service, "start_service", blocked)
    env = {"PODMUNCHER_STATE_DIR": str(tmp_path / "state"), "TERM": "xterm",
           "PODMUNCHER_CONFIG": str(tmp_path / "none.toml")}
    assert cli.main(["in.mp3", "--json", "--detach", "--output", str(tmp_path)], environ=env) == 3
    assert json.loads(capsys.readouterr().out)["reason_code"] == "storage_unknown"


def test_credential_never_appears_in_any_output(capsys, env, tmp_path):
    client = FakeServiceClient(["queued", "completed"])
    scenarios = [["status"], ["jobs"], ["job", JOB], ["cancel", JOB], ["recheck", JOB],
                 ["retry", JOB], ["cache", "list"], ["in.mp3", "--output", str(tmp_path)]]
    for argv in scenarios:
        for extra in ([], ["--json"]):
            client.states = ["queued", "completed"]
            _, out, err = invoke([*argv, *extra], FakeAccess(client), capsys, env)
            assert CREDENTIAL not in out + err


# -- precedence ----------------------------------------------------------------------


def test_flag_overrides_config_which_overrides_default(capsys, tmp_path):
    config = tmp_path / "prefs.toml"
    saved_out = tmp_path / "saved-out"
    config.write_text(
        '[selection]\ndevice = "cpu"\nprofile = "quality"\n'
        f'[paths]\noutput = {json.dumps(str(saved_out))}\n', encoding="utf-8")
    env = {"PODMUNCHER_CONFIG": str(config), "TERM": "xterm"}
    client = FakeServiceClient(["queued"])
    flag_out = tmp_path / "flag-out"
    invoke(["in.mp3", "--detach", "--device", "cuda", "--output", str(flag_out)],
           FakeAccess(client), capsys, env)
    request = client.calls[-1][1]
    assert request["selection"]["device"] == "cuda"        # flag over config
    assert request["selection"]["profile"] == "quality"    # config over default
    assert request["selection"]["language"] == "auto"      # default
    assert Path(request["output_root"]) == flag_out.resolve()
    invoke(["in.mp3", "--detach"], FakeAccess(client), capsys, env)
    request = client.calls[-1][1]
    assert request["selection"]["device"] == "cpu"
    assert Path(request["output_root"]) == saved_out.resolve()
    absent = {"PODMUNCHER_CONFIG": str(tmp_path / "none.toml"), "TERM": "xterm"}
    invoke(["in.mp3", "--detach"], FakeAccess(client), capsys, absent)
    assert client.calls[-1][1]["selection"] == {
        "device": "auto", "profile": "balanced", "model": "auto", "language": "auto"}


def test_invalid_saved_preferences_exit_two(capsys, tmp_path):
    config = tmp_path / "prefs.toml"
    config.write_text('[selection]\ndevice = "tpu"\n', encoding="utf-8")
    access = FakeAccess(FakeServiceClient())
    code, out, _ = invoke(["in.mp3", "--json"], access, capsys,
                          {"PODMUNCHER_CONFIG": str(config), "TERM": "xterm"})
    assert code == 2 and json.loads(out)["reason_code"] == "invalid_config"
    assert access.touched == []


def _prefs(tmp_path, text):
    config = tmp_path / "prefs.toml"
    config.write_text(text, encoding="utf-8")
    return {"PODMUNCHER_CONFIG": str(config), "TERM": "xterm"}


def test_registry_provider_admits_verified_model_from_flag_and_prefs(capsys, tmp_path):
    cli.set_registry_provider(lambda: (frozenset({"small"}), frozenset({"en"})))
    client = FakeServiceClient(["queued"])
    absent = {"PODMUNCHER_CONFIG": str(tmp_path / "none.toml"), "TERM": "xterm"}
    code, _, _ = invoke(["in.mp3", "--detach", "--model", "small", "--language", "en",
                         "--output", str(tmp_path)], FakeAccess(client), capsys, absent)
    assert code == 0
    assert client.calls[-1][1]["selection"]["model"] == "small"
    assert client.calls[-1][1]["selection"]["language"] == "en"
    saved = _prefs(tmp_path, '[selection]\nmodel = "small"\n')
    code, _, _ = invoke(["in.mp3", "--detach", "--output", str(tmp_path)], FakeAccess(client),
                        capsys, saved)
    assert code == 0 and client.calls[-1][1]["selection"]["model"] == "small"
    ns, _ = cli.parse_args(["in.mp3", "--output", str(tmp_path)])
    assert cli.build_request(ns, saved)["selection"]["model"] == "small"  # TUI path


def test_without_registry_provider_non_auto_model_is_refused(capsys, tmp_path):
    absent = {"PODMUNCHER_CONFIG": str(tmp_path / "none.toml"), "TERM": "xterm"}
    saved = _prefs(tmp_path, '[selection]\nmodel = "small"\n')
    for argv, environ in ((["in.mp3", "--model", "small"], absent), (["in.mp3"], saved)):
        access = FakeAccess(FakeServiceClient())
        code, out, _ = invoke([*argv, "--json"], access, capsys, environ)
        assert code == 2 and json.loads(out)["reason_code"] == "invalid_config"
        assert access.touched == []
    cli.set_registry_provider(lambda: (frozenset({"tiny"}), None))
    code, _, _ = invoke(["in.mp3", "--model", "small", "--json"], FakeAccess(), capsys, absent)
    assert code == 2  # listed registry, unlisted model


@pytest.mark.parametrize("provider", [
    lambda: (_ for _ in ()).throw(RuntimeError(CREDENTIAL)),
    lambda: [frozenset({"small"}), None],
    lambda: ({"small"}, None),
    lambda: (frozenset({1}), None),
])
def test_registry_provider_fault_is_a_closed_error(provider, capsys, tmp_path):
    cli.set_registry_provider(provider)
    absent = {"PODMUNCHER_CONFIG": str(tmp_path / "none.toml"), "TERM": "xterm"}
    access = FakeAccess(FakeServiceClient())
    code, out, err = invoke(["in.mp3", "--model", "small", "--json"], access, capsys, absent)
    result = json.loads(out)
    assert code == 4 and result["reason_code"] == "service_unavailable"
    assert CREDENTIAL not in out + err and "Traceback" not in err and access.touched == []
    with pytest.raises(TypeError):
        cli.set_registry_provider("nope")


def test_explicit_outputs_mapping_beats_saved_preferences(tmp_path):
    video_only = _prefs(tmp_path, "[outputs]\ntranscript = false\nvideo = true\n")
    base = {"input": "in.mp3", "device": None, "profile": None, "model": None,
            "language": None, "output": str(tmp_path), "video": False, "no_transcript": False}
    request = cli.build_request(
        SimpleNamespace(**base, outputs={"transcript": True, "video": False}), video_only)
    assert (request["outputs"]["transcript"], request["outputs"]["video"]) == (True, False)
    absent = {"PODMUNCHER_CONFIG": str(tmp_path / "none.toml"), "TERM": "xterm"}
    request = cli.build_request(
        SimpleNamespace(**base, outputs={"transcript": False, "video": True}), absent)
    assert (request["outputs"]["transcript"], request["outputs"]["video"]) == (False, True)
    # A malformed mapping is ignored: saved preferences and flags decide as before.
    request = cli.build_request(
        SimpleNamespace(**base, outputs={"transcript": 1, "video": "no"}), video_only)
    assert (request["outputs"]["transcript"], request["outputs"]["video"]) == (False, True)


def test_source_kind_comes_from_classification(capsys, env, tmp_path):
    client = FakeServiceClient(["queued"])
    for value, kind in (("https://example.com/a.mp3", "direct"),
                        ("https://youtu.be/dQw4w9WgXcQ", "youtube"), ("a.mp3", "local")):
        invoke([value, "--detach", "--output", str(tmp_path)], FakeAccess(client), capsys, env)
        assert client.calls[-1][1]["source"]["kind"] == kind


# -- lazy future UI ------------------------------------------------------------------


def test_future_ui_reports_pending_without_importing_it(monkeypatch, capsys, env):
    monkeypatch.setitem(sys.modules, "podmuncher.tui", None)  # simulates an absent module (ImportError)
    access = FakeAccess()
    code, out, _ = invoke(["tui"], access, capsys, env)
    assert code == 4 and "implementation pending" in out and access.touched == []
    code, out, _ = invoke(["tui", "--json"], access, capsys, env)
    result = json.loads(out)
    assert code == 4 and result["pending"] is True and result["command"] == "tui"
    assert result["reason_code"] == "service_unavailable"


def test_future_ui_with_a_present_module_runs_its_main(monkeypatch, capsys, env):
    monkeypatch.setitem(sys.modules, "podmuncher.tui", SimpleNamespace(main=lambda: 0))
    code, out, _ = invoke(["tui"], FakeAccess(), capsys, env)
    assert code == 0 and out == ""


# -- setup through the service -------------------------------------------------------


class SetupServiceClient(FakeServiceClient):
    """The setup routes: a plan, an accepted apply and a scripted terminal status."""

    def __init__(self, status=None):
        super().__init__()
        self.setup_status_doc = status or {"state": "done", "reason_code": "ok", "next_action": "",
                                           "setup_state": "done", "exit_code": 0}

    def setup_plan(self, capabilities, config):
        self.calls.append(("setup_plan", list(capabilities), dict(config)))
        return {"plan": {"plan_id": "plan-1", "capabilities": list(capabilities), "ready": True,
                         "reason_code": "ok", "next_action": "", "size_known": True, "sizes": {}}}

    def setup_apply(self, plan_id, key):
        self.calls.append(("setup_apply", plan_id, key))
        return {"operation": {"operation_id": OTHER, "plan_id": plan_id, "state": "running",
                              "reason_code": "ok", "next_action": ""}}

    def setup_status(self, operation_id):
        self.calls.append(("setup_status", operation_id))
        return {"operation_id": operation_id, **self.setup_status_doc}


def _answers(tmp_path, **overrides):
    data = {"paths": {"cache": str(tmp_path / "cache")}, "capabilities": ["media"], "accept_plan": True}
    data.update(overrides)
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_setup_answers_apply_through_the_running_service(capsys, env, tmp_path):
    """``--setup`` with an answers file: the service is started or attached (it owns the
    provisioning), the plan and apply go through its API and the outcome is the exit code."""
    client = SetupServiceClient()
    access = FakeAccess(client)
    answers = _answers(tmp_path, idempotency_key="setup-key-1")
    code, out, _ = invoke(["--setup"], access, capsys, {**env, "PODMUNCHER_SETUP_ANSWERS": answers})
    assert code == 0 and out == "setup: done (ok)\n"
    assert access.touched == ["ensure"] and client.closed == 1
    assert client.calls[:2] == [("setup_plan", ["media"], {"paths": {"cache": str(tmp_path / "cache")}}),
                                ("setup_apply", "plan-1", "setup-key-1")]
    assert client.calls[2] == ("setup_status", OTHER)
    # A blocked outcome carries the machine's state, reason and next action into the exit.
    client = SetupServiceClient({"state": "failed", "reason_code": "backing_unresolved",
                                 "next_action": "enter the existing Docker data location",
                                 "setup_state": "blocked", "exit_code": 3})
    code, out, _ = invoke(["--setup"], FakeAccess(client), capsys, {**env, "PODMUNCHER_SETUP_ANSWERS": answers})
    assert code == 3 and out == "setup: blocked (backing_unresolved) - enter the existing Docker data location\n"
    # Declined plan: planned, never applied, the same cancellation exit as the local wizard.
    client = SetupServiceClient()
    declined = _answers(tmp_path, accept_plan=False)
    code, out, _ = invoke(["--setup"], FakeAccess(client), capsys, {**env, "PODMUNCHER_SETUP_ANSWERS": declined})
    assert code == 3 and out == "setup: cancelled (cancelled)\n"
    assert [c[0] for c in client.calls] == ["setup_plan"]


def test_setup_service_refusal_maps_through_the_reason_table(capsys, env, tmp_path):
    client = SetupServiceClient()
    client.setup_apply = lambda plan_id, key: (_ for _ in ()).throw(
        service.ClientError(ReasonCode.CONFLICT, status=409))
    answers = _answers(tmp_path)
    code, out, _ = invoke(["--setup"], FakeAccess(client), capsys, {**env, "PODMUNCHER_SETUP_ANSWERS": answers})
    assert code == 4 and out == "setup: failed (conflict)\n" and client.closed == 1


def test_setup_without_answers_or_a_tty_never_starts_the_service(capsys, env, monkeypatch):
    monkeypatch.setattr("podmuncher.setup._is_tty", lambda: False)
    access = FakeAccess()
    code, out, err = invoke(["--setup", "--guided"], access, capsys, env)
    assert code == 2 and "PODMUNCHER_SETUP_ANSWERS" in err and access.touched == []
    # An invalid answers file is refused before the service is touched.
    code, out, _ = invoke(["--setup"], access, capsys, {**env, "PODMUNCHER_SETUP_ANSWERS": "nope.json"})
    assert code == 2 and "invalid answers file" in out and access.touched == []


# -- seams ---------------------------------------------------------------------------


class RecordingAdapter:
    def __init__(self):
        self.rendered = []

    def render(self, result, *, command):
        self.rendered.append(command)
        return f"CUSTOM {command}"

    def escape(self, text):
        return text


def test_configure_presentation_replaces_the_adapter(capsys, env):
    adapter = RecordingAdapter()
    cli.configure_presentation(adapter)
    _, out, _ = invoke(["status"], FakeAccess(), capsys, env)
    assert out == "CUSTOM status\n" and adapter.rendered == ["status"]
    _, out, _ = invoke(["status", "--json"], FakeAccess(), capsys, env)
    assert json.loads(out)["command"] == "status"  # JSON never goes through an adapter
    with pytest.raises(TypeError):
        cli.configure_presentation(object())


def test_presentation_module_is_loaded_lazily_at_invocation(monkeypatch, capsys, env):
    adapter = RecordingAdapter()
    monkeypatch.setitem(sys.modules, "podmuncher.presentation",
                        SimpleNamespace(default_adapter=lambda: adapter))
    _, out, _ = invoke(["status"], FakeAccess(), capsys, env)
    assert out == "CUSTOM status\n"
    cli.configure_presentation(None)
    monkeypatch.setitem(sys.modules, "podmuncher.presentation", None)
    _, out, _ = invoke(["status"], FakeAccess(), capsys, env)
    assert out.startswith("status: ok")


def test_explicit_adapter_wins_over_the_presentation_module(monkeypatch, capsys, env):
    other = RecordingAdapter()
    monkeypatch.setitem(sys.modules, "podmuncher.presentation",
                        SimpleNamespace(default_adapter=lambda: other))
    mine = RecordingAdapter()
    cli.configure_presentation(mine)
    invoke(["status"], FakeAccess(), capsys, env)
    assert mine.rendered == ["status"] and other.rendered == []


def test_broken_adapter_falls_back_to_the_safe_default(capsys, env):
    class Broken(RecordingAdapter):
        def render(self, result, *, command):
            raise RuntimeError("nope")

    cli.configure_presentation(Broken())
    code, out, _ = invoke(["status"], FakeAccess(), capsys, env)
    assert code == 0 and out.startswith("status: ok")


def test_client_startup_hook_runs_for_commands_but_not_help(capsys, env):
    calls = []
    cli.set_client_startup(lambda: calls.append("hook"))
    invoke(["status"], FakeAccess(), capsys, env)
    assert calls == ["hook"]
    with pytest.raises(SystemExit):
        cli.main(["--version"], environ=env)
    with pytest.raises(SystemExit):
        cli.main(["--help"], environ=env)
    invoke([], FakeAccess(), capsys, env)  # non-TTY no-arg help
    capsys.readouterr()
    assert calls == ["hook"]
    with pytest.raises(TypeError):
        cli.set_client_startup("nope")


def test_client_startup_hook_failure_is_a_closed_runtime_error(capsys, env):
    def hook():
        raise RuntimeError(CREDENTIAL)

    cli.set_client_startup(hook)
    code, out, _ = invoke(["status"], FakeAccess(), capsys, env)
    assert code == 4 and CREDENTIAL not in out


# -- live smoke ----------------------------------------------------------------------

FACTORY_SOURCE = '''
import os
import uuid

from podmuncher.api import ApiActionError
from podmuncher.domains import ReasonCode, ServiceState


class FakeActions:
    def __init__(self):
        self.jobs = {}
        self.keys = {}
        self.cancels = 0

    def _job(self, job_id):
        return {"job_id": job_id, "state": "queued", "cancels": self.cancels}

    async def identity(self):
        return {"pid": os.getpid(), "service_state": ServiceState.RUNNING}

    async def snapshot(self):
        return {"schema_version": 1, "jobs": len(self.jobs), "freshness": "fresh"}

    async def list_jobs(self, limit, cursor):
        return [self._job(j) for j in self.jobs], None

    async def get_job(self, job_id):
        if job_id not in self.jobs:
            raise ApiActionError(ReasonCode.INVALID_INPUT, not_found=True)
        return self._job(job_id)

    async def submit(self, request, key):
        if key in self.keys:
            return self._job(self.keys[key]), True
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = request.request_hash
        self.keys[key] = job_id
        return self._job(job_id), False

    async def recheck(self, job_id):
        return self._job(job_id), {"result": "ready"}, True

    async def retry(self, job_id, key):
        return self._job(job_id), str(uuid.uuid4()), False

    async def clone(self, job_id, request, key):
        return self._job(job_id)

    async def cancel(self, job_id):
        self.cancels += 1
        return self._job(job_id), "intent"

    async def open_artifact(self, job_id, artifact_id):
        async def chunks():
            yield b"x" * 10
        return 10, "audio/mpeg", chunks()

    async def list_cache(self, limit, cursor):
        return [], None

    async def remove_cache(self, entry_id):
        return {"entry_id": entry_id}

    async def setup_plan(self, capabilities, config):
        return {"plan_id": "plan-1", "capabilities": [str(c) for c in capabilities], "ready": True,
                "reason_code": "ok", "next_action": "", "size_known": False, "sizes": {}}

    async def setup_apply(self, plan_id, key):
        return {"operation_id": str(uuid.uuid4()), "plan_id": plan_id, "state": "running",
                "reason_code": "ok", "next_action": ""}

    async def setup_status(self, operation_id):
        return {"operation_id": operation_id, "state": "done", "reason_code": "ok", "next_action": "",
                "setup_state": "done", "exit_code": 0}

    async def stop(self):
        return True, "ok"


class Composition:
    def __init__(self, state_dir):
        self.actions = FakeActions()
        self.state_dir = state_dir

    def audit(self, route, code, ids):
        return None

    async def startup(self):
        return None

    async def shutdown(self):
        return True, "ok"


def make_service(state_dir):
    return Composition(state_dir)
'''


def make_service_factory(tmp_path):
    """Write the in-memory factory module; return (module:attr, import_path)."""
    directory = tmp_path / "factory"
    directory.mkdir(exist_ok=True)
    (directory / "fake_service_factory.py").write_text(FACTORY_SOURCE, encoding="utf-8")
    return "fake_service_factory:make_service", str(directory)


def spawn_test_service(state_dir, factory):
    spec, import_path = factory
    return service.start_service(state_dir, factory=spec, import_path=import_path, timeout=30.0)


@pytest.mark.live
def test_own_transport_smoke(tmp_path):
    state = tmp_path / "state"
    outputs = []
    record = None
    children = []
    started = time.monotonic()
    cli_env = {"PODMUNCHER_STATE_DIR": str(state), "PODMUNCHER_CONFIG": str(tmp_path / "none.toml")}

    def run(*args, expect=0):
        result = run_cli(*args, env=cli_env, tmp_path=tmp_path, timeout=60)
        outputs.append(result.stdout + result.stderr)
        assert result.returncode == expect, (args, result.returncode, result.stdout, result.stderr)
        return result

    try:
        record = spawn_test_service(str(state), make_service_factory(tmp_path))
        children = list(service._CHILDREN.values())
        credential = record.credential

        detached = run("in.mp3", "--detach", "--output", str(tmp_path / "out"))
        job_id = detached.stdout.strip()
        assert str(uuid.UUID(job_id)) == job_id

        status = json.loads(run("status", "--json").stdout)
        assert status["service_state"] == "running" and status["stale"] is False
        shown = json.loads(run("job", job_id, "--json").stdout)
        assert shown["job"]["job_id"] == job_id and shown["job"]["state"] == "queued"
        rechecked = json.loads(run("recheck", job_id, "--json").stdout)
        assert rechecked["ready_to_retry"] is True
        cancelled = json.loads(run("cancel", job_id, "--json").stdout)
        assert cancelled["status"] == "intent"
        listed = json.loads(run("jobs", "--json").stdout)
        assert [j["job_id"] for j in listed["jobs"]] == [job_id]

        # --setup with an answers file applies through the running service's setup routes.
        answers = tmp_path / "answers.json"
        answers.write_text(json.dumps({"paths": {}, "capabilities": ["media"], "accept_plan": True}),
                           encoding="utf-8")
        applied = run_cli("--setup", env={**cli_env, "PODMUNCHER_SETUP_ANSWERS": str(answers)},
                          tmp_path=tmp_path, timeout=60)
        outputs.append(applied.stdout + applied.stderr)
        assert applied.returncode == 0 and applied.stdout.strip() == "setup: done (ok)", applied

        stopped = json.loads(run("service", "stop", "--json").stdout)
        assert stopped["outcome"] == "stopped"
        for child in children:
            child.wait(timeout=30)
        assert all(child.poll() is not None for child in children)
        assert service.discover(str(state)) is None

        after = json.loads(run("status", "--json").stdout)  # read-only: never restarts it
        assert after["service_state"] == "stopped" and after["stale"] is True
        assert service.discover(str(state)) is None
        assert all(credential not in text for text in outputs)
        assert time.monotonic() - started < 120
    finally:
        service.kill_launched_services()
        if record is not None:
            service._kill_tree(record.pid)
    print("PODMUNCHER_SMOKE_OK")
