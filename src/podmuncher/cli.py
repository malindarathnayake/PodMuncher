"""Command-line contract, preference precedence, stable exits and the presentation seam.

Phase 5 unit P5-U3.  Contract (Docs/design-summary.md):

* :55 ``podmuncher`` opens the TUI only on a suitable TTY; otherwise it prints help without
  starting Docker.  ``podmuncher tui`` attaches explicitly.  ``--help`` and ``--version`` are
  immediate and side-effect-free.  ``podmuncher --setup`` runs the setup.  No invisible
  prompt in CI.
* :57 ``podmuncher INPUT`` requests transcription.  ``--video`` keeps the original video;
  ``--video --no-transcript`` requests video only; ``--no-transcript`` without another
  requested output is rejected.  Flags override saved preferences: ``--model``,
  ``--device auto|cpu|cuda``, ``--profile balanced|quality``, ``--output``, ``--language``,
  ``--no-color``, ``--json``.
* :59 Submission waits and prints progress by default; ``--detach`` returns the job ID.
  ``status``, ``jobs``, ``cancel JOB``, ``recheck JOB``, ``retry JOB``, ``doctor``,
  ``service start|stop`` and ``cache list/remove`` expose the service actions.  Ctrl+C stops
  the observer (exit 130), reports the retained job ID and never cancels.  Exits: 0 success;
  2 invalid input/config; 3 preflight blocked; 4 runtime failure; 5 partial result; 130
  observer interrupted.  Detailed reason codes appear in JSON.
* :61 submit/retry/TUI/setup/``service start`` may start the service after the state-volume
  preflight.  status/jobs/doctor never start it: they show it stopped with the persisted
  snapshot marked stale.  ``service stop`` succeeds only after writers are reaped.
* :47 ``clone`` creates a new job with a new request hash; ``recheck`` creates no attempt.

Every command produces one result dict.  ``--json`` prints it with the original text; plain
mode renders it through the active :class:`PresentationAdapter`, whose display projection
escapes control characters and markup brackets.  Importing this module touches only the
standard library, ``domains`` and ``requests``; ``config``, ``service``, ``presentation``,
``tui`` and ``setup`` are imported inside the commands that need them.  Runtime records
(which hold the credential) are never rendered.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from .domains import DomainValidationError, JobState, ReasonCode, validate_uuid_string
from .requests import JobRequest, classify_source

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BLOCKED = 3
EXIT_FAILED = 4
EXIT_PARTIAL = 5
EXIT_INTERRUPTED = 130
POLL_INTERVAL_SECONDS = 1.0
START_HINT = "podmuncher service start"

COMMANDS = ("status", "jobs", "job", "cancel", "recheck", "retry", "clone", "doctor",
            "service", "cache", "tui")

# --------------------------------------------------------------------------- exit mapping

_EXIT_0 = frozenset({ReasonCode.OK, ReasonCode.ALREADY_TERMINAL})
_EXIT_2 = frozenset({
    ReasonCode.INVALID_INPUT, ReasonCode.INVALID_CONFIG, ReasonCode.INPUT_REJECTED,
    ReasonCode.SOURCE_CHANGED, ReasonCode.NESTING_LIMIT, ReasonCode.UNSUPPORTED_SCHEMA,
})
_EXIT_3 = frozenset({
    ReasonCode.STORAGE_INSUFFICIENT, ReasonCode.STORAGE_UNKNOWN, ReasonCode.STORAGE_STALE,
    ReasonCode.STORAGE_GUARD_TRIPPED, ReasonCode.QUOTA_EXCEEDED, ReasonCode.RAM_INSUFFICIENT,
    ReasonCode.GPU_UNAVAILABLE, ReasonCode.DOCKER_UNAVAILABLE,
    ReasonCode.DOCKER_CONTEXT_UNSUPPORTED, ReasonCode.BACKING_UNRESOLVED,
    ReasonCode.QUEUE_FULL, ReasonCode.AUDIT_LOSS,
})


def _classify_reason(code: ReasonCode) -> int:
    if code in _EXIT_0:
        return EXIT_OK
    if code in _EXIT_2:
        return EXIT_USAGE
    if code in _EXIT_3:
        return EXIT_BLOCKED
    return EXIT_FAILED  # service unavailable, writer_unresolved, transport and the rest


# One exit per reason code; the table is total over the closed ReasonCode domain.
EXIT_FOR_REASON: Mapping[ReasonCode, int] = {code: _classify_reason(code) for code in ReasonCode}

# Observer outcome per terminal job state.  Non-terminal states keep the observer polling.
JOB_STATE_EXIT: Mapping[JobState, int] = {
    JobState.COMPLETED: EXIT_OK,
    JobState.PARTIAL: EXIT_PARTIAL,
    JobState.BLOCKED: EXIT_BLOCKED,
    JobState.FAILED: EXIT_FAILED,
    JobState.CANCELLED: EXIT_FAILED,
    JobState.INTERRUPTED: EXIT_FAILED,
}
_ACTIVE_STATES = frozenset({
    JobState.QUEUED, JobState.CHECKING, JobState.PREPARING, JobState.DOWNLOADING,
    JobState.PROBING, JobState.TRANSCRIBING, JobState.EXPORTING,
})


def exit_for_reason(reason: object) -> int:
    """Exit code for a reason code (member or wire string).  Unknown values are runtime failures."""
    try:
        return EXIT_FOR_REASON[ReasonCode.parse(reason)]
    except DomainValidationError:
        return EXIT_FAILED


def exit_for_job_state(state: object) -> int | None:
    """Exit code for a terminal job state, or None while the job is still active."""
    try:
        parsed = JobState.parse(state)
    except DomainValidationError:
        return EXIT_FAILED
    if parsed in _ACTIVE_STATES:
        return None
    return JOB_STATE_EXIT[parsed]


# ---------------------------------------------------------------------- presentation seam


class PresentationAdapter(Protocol):
    """Display projection.  Plain/JSON data keep original text; only projections escape."""

    def render(self, result: dict, *, command: str) -> str: ...

    def escape(self, text: str) -> str: ...


class PlainPresentation:
    """Safe default: escapes C0/C1 controls, ESC, format characters and markup brackets."""

    def escape(self, text: str) -> str:
        out: list[str] = []
        for ch in str(text):
            code = ord(ch)
            if ch == "\\":
                out.append("\\\\")
            elif ch in "[]":
                out.append("\\" + ch)
            elif code < 0x20 or code == 0x7F:
                out.append(f"\\x{code:02x}")
            elif 0x80 <= code <= 0x9F:
                out.append(f"\\u{code:04x}")
            elif unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Zl", "Zp"):
                out.append(f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}")
            else:
                out.append(ch)
        return "".join(out)

    def _value(self, value: Any, indent: int, lines: list[str]) -> str | None:
        pad = "  " * indent
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                inner = value[key]
                if isinstance(inner, (dict, list)) and inner:
                    lines.append(f"{pad}{self.escape(str(key))}:")
                    self._value(inner, indent + 1, lines)
                else:
                    lines.append(f"{pad}{self.escape(str(key))}: {self._scalar(inner)}")
            return None
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)) and item:
                    lines.append(f"{pad}-")
                    self._value(item, indent + 1, lines)
                else:
                    lines.append(f"{pad}- {self._scalar(item)}")
            return None
        return self._scalar(value)

    def _scalar(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (dict, list)):
            return "{}" if isinstance(value, dict) else "[]"
        if isinstance(value, (int, float)):
            return str(value)
        return self.escape(str(value))

    def render(self, result: dict, *, command: str) -> str:
        if command == "progress":
            return self.escape(f"{result.get('job_id', '-')} {result.get('state', '-')}")
        job_id = result.get("job_id")
        if result.get("detached"):
            if result.get("exit_code") == EXIT_INTERRUPTED:
                return self.escape(f"Detached; job {job_id} keeps running")
            return self.escape(str(job_id))
        status = "ok" if result.get("ok") else "failed"
        lines = [f"{self.escape(command)}: {status} ({self.escape(str(result.get('reason_code')))})"]
        body = {k: v for k, v in result.items()
                if k not in ("command", "ok", "reason_code") and v is not None}
        self._value(body, 1, lines)
        return "\n".join(lines)


_DEFAULT_ADAPTER = PlainPresentation()
_adapter: PresentationAdapter | None = None
_client_startup: Callable[[], Any] | None = None
RegistryProvider = Callable[[], "tuple[frozenset[str] | None, frozenset[str] | None]"]
_registry_provider: RegistryProvider | None = None


def configure_presentation(adapter: PresentationAdapter | None) -> None:
    """Replace the active adapter (None restores the safe default and lazy discovery)."""
    global _adapter
    if adapter is not None and not (callable(getattr(adapter, "render", None))
                                    and callable(getattr(adapter, "escape", None))):
        raise TypeError("presentation adapter needs render() and escape()")
    _adapter = adapter


def set_client_startup(hook: Callable[[], Any] | None) -> None:
    """Register a hook run at the start of every non-help/version invocation (P6-U2 binding)."""
    global _client_startup
    if hook is not None and not callable(hook):
        raise TypeError("client startup hook must be callable")
    _client_startup = hook


def set_registry_provider(provider: RegistryProvider | None) -> None:
    """Register the verified model/language registry source (SPEC_GAP-10; P6-U2 binding).

    The callable returns ``(verified_models, verified_languages)``, each a frozenset or None.
    It runs inside :func:`build_request`, never at import or on ``--help``/``--version``.
    None restores the auto-only default.
    """
    global _registry_provider
    if provider is not None and not callable(provider):
        raise TypeError("registry provider must be callable")
    _registry_provider = provider


class _RegistryUnavailable(Exception):
    """The registry provider failed or returned a malformed value.  Closed: no message."""

    reason_code = ReasonCode.SERVICE_UNAVAILABLE
    next_action = "podmuncher doctor"


def _registries() -> tuple[frozenset[str] | None, frozenset[str] | None]:
    provider = _registry_provider
    if provider is None:
        return None, None
    try:
        value = provider()
    except Exception:  # noqa: BLE001 - a provider fault must not leak its message
        raise _RegistryUnavailable() from None
    if type(value) is not tuple or len(value) != 2 or any(
            item is not None and (type(item) is not frozenset
                                  or any(type(name) is not str for name in item))
            for item in value):
        raise _RegistryUnavailable()
    return value[0], value[1]


def _active_adapter(as_json: bool) -> PresentationAdapter:
    """The configured adapter; otherwise lazily load ``podmuncher.presentation`` if it exists."""
    global _adapter
    if _adapter is None and not as_json:
        name = f"{__package__}.presentation"
        try:
            module = importlib.import_module(name)
            factory = getattr(module, "default_adapter", None)
            candidate = factory() if callable(factory) else None
        except ImportError:
            candidate = None
        if candidate is not None and callable(getattr(candidate, "render", None)) \
                and callable(getattr(candidate, "escape", None)):
            _adapter = candidate
    return _adapter or _DEFAULT_ADAPTER


# ------------------------------------------------------------------------------ service access


class _ServiceAccess:
    """Lazy bridge to ``podmuncher.service``.  Tests inject a fake with the same methods."""

    def __init__(self, environ: Mapping[str, str]) -> None:
        self._environ = environ

    def _dir(self) -> str:
        from . import service

        return self._environ.get("PODMUNCHER_STATE_DIR") or service.default_state_dir()

    def connect(self) -> Any:
        from . import service

        return service.connect(self._dir())

    def ensure(self) -> Any:
        from . import service

        client = self.connect()
        if client is not None:
            return client
        return service.ServiceClient(service.start_service(self._dir()))

    def start(self) -> dict:
        from . import service

        record = service.start_service(self._dir())
        return {"state": "running", "pid": record.pid}

    def stop(self) -> Any:
        from . import service

        return service.stop_service(self._dir())

    def read_status(self) -> dict:
        from . import service

        return service.read_status(self._dir())

    def new_key(self) -> str:
        from . import service

        return service.new_idempotency_key()


# ----------------------------------------------------------------------------------- parsing


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print the result as JSON")
    common.add_argument("--no-color", action="store_true", help="disable color")
    return common


def _submit_flags() -> argparse.ArgumentParser:
    flags = argparse.ArgumentParser(add_help=False)
    flags.add_argument("--video", action="store_true", help="keep the original video")
    flags.add_argument("--no-transcript", action="store_true",
                       help="skip the transcript (requires --video)")
    flags.add_argument("--model", default=None, help="model name (default: saved preference)")
    flags.add_argument("--device", default=None, choices=("auto", "cpu", "cuda"))
    flags.add_argument("--profile", default=None, choices=("balanced", "quality"))
    flags.add_argument("--output", default=None, help="output directory")
    flags.add_argument("--language", default=None, help="spoken language or auto")
    flags.add_argument("--detach", action="store_true", help="print the job ID and return")
    return flags


def _job_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("job", metavar="JOB")
    return parent


def _build_command_parser() -> argparse.ArgumentParser:
    common, flags, job = _common_parser(), _submit_flags(), _job_parent()
    top = argparse.ArgumentParser(prog="podmuncher")
    sub = top.add_subparsers(dest="command", metavar="COMMAND", required=True)
    sub.add_parser("status", parents=[common], help="service status (never starts it)")
    jobs = sub.add_parser("jobs", parents=[common], help="list jobs")
    jobs.add_argument("--limit", type=int, default=None)
    jobs.add_argument("--cursor", default=None)
    sub.add_parser("job", parents=[common, job], help="show one job")
    sub.add_parser("cancel", parents=[common, job], help="cancel a job explicitly")
    sub.add_parser("recheck", parents=[common, job], help="refresh preflight, no attempt")
    retry = sub.add_parser("retry", parents=[common, job], help="retry a terminal or blocked job")
    retry.add_argument("--detach", action="store_true", help="print the job ID and return")
    clone = sub.add_parser("clone", parents=[common, job, flags],
                           help="create a new job with an edited request")
    clone.add_argument("input", metavar="INPUT", help="source for the new request")
    sub.add_parser("doctor", parents=[common], help="diagnostics (never starts the service)")
    service = sub.add_parser("service", parents=[common], help="start or stop the service")
    service.add_argument("action", choices=("start", "stop"))
    cache = sub.add_parser("cache", parents=[common], help="list or remove cache entries")
    cache_sub = cache.add_subparsers(dest="action", metavar="ACTION", required=True)
    cache_list = cache_sub.add_parser("list", parents=[common])
    cache_list.add_argument("--limit", type=int, default=None)
    cache_list.add_argument("--cursor", default=None)
    cache_remove = cache_sub.add_parser("remove", parents=[common])
    cache_remove.add_argument("entry", metavar="ENTRY")
    sub.add_parser("tui", parents=[common], help="attach the terminal UI")
    return top


def _build_input_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podmuncher", parents=[_common_parser(), _submit_flags()],
        description="Podmuncher podcast transcription. On a TTY with no arguments it opens the "
                    "terminal UI; otherwise it prints this help.",
        epilog="commands: " + ", ".join(COMMANDS) + ". Use 'podmuncher COMMAND --help' for "
               "details. A first argument that is not a command is the INPUT to transcribe; "
               "prefix an input named like a command with './'.")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {_package_version()}")
    parser.add_argument("--setup", action="store_true", help="run the guided setup")
    parser.add_argument("input", nargs="?", metavar="INPUT",
                        help="local media path or public https URL")
    return parser


def _package_version() -> str:
    from . import __version__

    return __version__


def _check_outputs(parser: argparse.ArgumentParser, ns: argparse.Namespace) -> None:
    if getattr(ns, "no_transcript", False) and not getattr(ns, "video", False):
        parser.error("--no-transcript requires --video")


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    """Parse ``argv``.  ``ns.command`` is None for the no-argument path.  May raise SystemExit."""
    tokens = list(argv)
    leading = 0
    while leading < len(tokens) and tokens[leading] in ("--json", "--no-color"):
        leading += 1
    if leading < len(tokens) and tokens[leading] in COMMANDS:
        parser = _build_command_parser()
        ns = parser.parse_args(tokens[leading:])
        for flag in tokens[:leading]:
            setattr(ns, flag[2:].replace("-", "_"), True)
        _check_outputs(parser, ns)
        return ns, parser
    parser = _build_input_parser()
    ns = parser.parse_args(tokens)
    _check_outputs(parser, ns)
    if ns.setup:
        if ns.input is not None:
            parser.error("--setup takes no INPUT")
        ns.command = "setup"
    elif ns.input is not None:
        ns.command = "submit"
    else:
        if ns.video or ns.detach or any(
                getattr(ns, name) is not None for name in ("model", "device", "profile",
                                                           "output", "language")):
            parser.error("INPUT is required")
        ns.command = None
    return ns, parser


# ------------------------------------------------------------------------------ result helpers


def _result(command: str, exit_code: int, reason: object = ReasonCode.OK, **fields: Any) -> dict:
    value = reason.value if isinstance(reason, ReasonCode) else str(reason)
    return {"command": command, "ok": exit_code == EXIT_OK, "exit_code": exit_code,
            "reason_code": value, **fields}


def _error_result(command: str, error: BaseException, **fields: Any) -> dict:
    """Closed mapping of any failure.  Messages are never echoed: they could carry paths."""
    reason = getattr(error, "reason_code", None)
    if not isinstance(reason, ReasonCode):
        reason = ReasonCode.SERVICE_UNAVAILABLE
    next_action = getattr(error, "next_action", None)
    if reason is ReasonCode.UNSUPPORTED_SCHEMA and next_action in (None, "", "none"):
        next_action = "upgrade_client"
    if isinstance(next_action, str) and next_action and next_action != "none":
        fields.setdefault("next_action", next_action)
    return _result(command, exit_for_reason(reason), reason, **fields)


def _job_result(command: str, job: Mapping[str, Any], exit_code: int, **fields: Any) -> dict:
    reason = job.get("reason_code", "ok") if exit_code != EXIT_OK else "ok"
    extra = {"next_action": job["next_action"]} if job.get("next_action") not in (None, "none") else {}
    return _result(command, exit_code, reason, job=dict(job), job_id=job.get("job_id"),
                   **extra, **fields)


class _Context:
    def __init__(self, access: Any, as_json: bool, sleep: Callable[[float], None],
                 interval: float, environ: Mapping[str, str], adapter: PresentationAdapter,
                 no_color: bool) -> None:
        self.access = access
        self.as_json = as_json
        self.sleep = sleep
        self.interval = interval
        self.environ = environ
        self.adapter = adapter
        self.no_color = no_color

    def progress(self, job: Mapping[str, Any]) -> None:
        if self.as_json:
            return
        _write(self.adapter.render({"job_id": job.get("job_id"), "state": job.get("state")},
                                   command="progress"))


def _write(text: str) -> None:
    stream = sys.stdout
    try:
        stream.write(text + "\n")
        stream.flush()
    except UnicodeEncodeError:
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            stream.flush()
            buffer.write((text + "\n").encode("utf-8", "replace"))
            buffer.flush()
        else:
            stream.write((text + "\n").encode("ascii", "backslashreplace").decode("ascii"))


def _emit(ctx: _Context, result: dict) -> None:
    if ctx.as_json:
        _write(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    try:
        text = ctx.adapter.render(result, command=str(result.get("command")))
    except Exception:  # noqa: BLE001 - a broken adapter must not hide the result
        text = _DEFAULT_ADAPTER.render(result, command=str(result.get("command")))
    _write(text)


def _job_id(value: str) -> str:
    return validate_uuid_string(value)


# --------------------------------------------------------------------------- request building


def _read_preferences(environ: Mapping[str, str]) -> bytes | None:
    """Saved preferences (config.toml) or None when the file does not exist."""
    path = environ.get("PODMUNCHER_CONFIG")
    if not path:
        import platformdirs

        path = os.path.join(platformdirs.user_config_dir("podmuncher", appauthor=False),
                            "config.toml")
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except FileNotFoundError:
        return None
    except OSError:
        from .config import ConfigError

        raise ConfigError() from None


def build_request(args: argparse.Namespace, environ: Mapping[str, str]) -> dict:
    """Merge defaults < saved preferences < flags into a validated JobRequest dict.

    Non-auto models and languages are admitted only when the registered registry provider
    lists them (see :func:`set_registry_provider`); without one the request is auto-only.
    An ``outputs`` mapping on ``args`` with bool ``transcript`` and ``video`` (set by the
    dashboard, never by argparse) overrides the saved outputs exactly.
    """
    from .config import Config

    source = classify_source(args.input)
    selection = {name: getattr(args, name, None)
                 for name in ("device", "profile", "model", "language")
                 if getattr(args, name, None) is not None}
    overrides: dict[str, Any] = {}
    if selection:
        overrides["selection"] = selection
    if getattr(args, "output", None) is not None:
        overrides["paths"] = {"output": args.output}
    explicit = getattr(args, "outputs", None)
    if isinstance(explicit, Mapping) and type(explicit.get("transcript")) is bool \
            and type(explicit.get("video")) is bool:
        overrides["outputs"] = {"transcript": explicit["transcript"], "video": explicit["video"]}
    elif getattr(args, "video", False):
        overrides["outputs"] = {"video": True,
                                "transcript": not getattr(args, "no_transcript", False)}
    models, languages = _registries()
    registry = {"verified_models": models, "verified_languages": languages}
    raw = _read_preferences(environ)
    config = Config.from_toml(raw, overrides=overrides, **registry) if raw is not None \
        else Config.from_dict(None, overrides=overrides, **registry)
    data = {
        "schema_version": 1,
        "source": {"kind": source.kind, "input": source.input},
        "outputs": {"transcript": config.outputs.transcript, "video": config.outputs.video,
                    "video_quality": config.outputs.video_quality},
        "selection": {"device": config.selection.device, "profile": config.selection.profile,
                      "model": config.selection.model, "language": config.selection.language},
        "output_root": config.paths.output,
        "limits": {"media_bytes": config.limits.media_bytes,
                   "duration_seconds": config.limits.duration_seconds,
                   "export_bytes": config.limits.export_bytes},
    }
    return JobRequest.from_dict(data, **registry).to_dict()


# ------------------------------------------------------------------------------------ handlers


def _stopped(command: str, **fields: Any) -> dict:
    return _result(command, EXIT_FAILED, ReasonCode.SERVICE_UNAVAILABLE, service_state="stopped",
                   stale=True, next_action=START_HINT, **fields)


def _status_like(ctx: _Context, command: str) -> dict:
    info = ctx.access.read_status()
    state = info.get("state")
    fields: dict[str, Any] = {"service_state": state, "stale": bool(info.get("stale")),
                              "snapshot": info.get("snapshot")}
    if command == "doctor":
        fields["checks"] = [
            {"name": "service", "status": state},
            {"name": "snapshot", "status": "stale" if info.get("stale") else "fresh"
             if info.get("snapshot") is not None else "none"},
        ]
    if state == "running":
        return _result(command, EXIT_OK, **fields)
    if state == "stopped":  # information, not an error: offer to start it
        return _result(command, EXIT_OK, next_action=START_HINT, **fields)
    return _result(command, EXIT_FAILED, ReasonCode.SERVICE_UNAVAILABLE,
                   next_action=START_HINT, **fields)


def _cmd_status(ctx: _Context, args: argparse.Namespace) -> dict:
    return _status_like(ctx, "status")


_DOCTOR_BLOCKING = frozenset({"blocked", "unavailable", "error"})
_RESULT_KEYS = frozenset({"command", "ok", "exit_code", "reason_code", "next_action"})


def _host_checks() -> list[dict]:
    """Run the read-only host diagnostic (no start, pull or write) and flatten its checks."""
    from . import setup

    report = setup.doctor_report(None)
    host = report["checks"]
    if not isinstance(host, Mapping):
        raise TypeError("doctor checks")
    checks = []
    for name, check in host.items():
        entry = {"name": str(name), "status": str(check["status"]),
                 "reason_code": str(check.get("reason_code", ReasonCode.OK.value))}
        if check.get("detail") is not None:
            entry["detail"] = check["detail"]
        if check.get("next_action") not in (None, "", "none"):
            entry["next_action"] = str(check["next_action"])
        checks.append(entry)
    return checks


def _cmd_doctor(ctx: _Context, args: argparse.Namespace) -> dict:
    """Service/snapshot checks, then the host checks.  Never starts, pulls or writes."""
    result = _status_like(ctx, "doctor")
    fields = {k: v for k, v in result.items() if k not in _RESULT_KEYS}
    try:
        fields["checks"] = [*fields["checks"], *_host_checks()]
    except Exception:  # noqa: BLE001 - closed mapping; the diagnostic message is not echoed
        return _result("doctor", EXIT_FAILED, ReasonCode.SERVICE_UNAVAILABLE,
                       host_checks="unavailable", next_action="podmuncher doctor", **fields)
    blocking = next((c for c in fields["checks"] if c["status"] in _DOCTOR_BLOCKING), None)
    if blocking is None:
        result["checks"] = fields["checks"]
        return result
    try:
        reason: object = ReasonCode.parse(blocking.get("reason_code"))
    except DomainValidationError:
        reason = ReasonCode.SERVICE_UNAVAILABLE
    return _result("doctor", EXIT_BLOCKED, reason, blocked_check=blocking["name"],
                   next_action=blocking.get("next_action") or "podmuncher --setup", **fields)


def _cmd_jobs(ctx: _Context, args: argparse.Namespace) -> dict:
    client = ctx.access.connect()
    if client is None:
        info = ctx.access.read_status()
        return _result("jobs", EXIT_OK, service_state="stopped", stale=True, jobs=[],
                       snapshot=info.get("snapshot"), next_action=START_HINT)
    try:
        reply = client.list_jobs(args.limit, args.cursor)
    finally:
        client.close()
    return _result("jobs", EXIT_OK, jobs=reply.get("jobs", []), next_cursor=reply.get("next_cursor"))


def _cmd_job(ctx: _Context, args: argparse.Namespace) -> dict:
    job_id = _job_id(args.job)
    client = ctx.access.connect()
    if client is None:
        return _stopped("job", job_id=job_id)
    try:
        job = client.get_job(job_id)["job"]
    finally:
        client.close()
    code = exit_for_job_state(job.get("state"))
    return _job_result("job", job, EXIT_OK if code is None else code)


def _cmd_cancel(ctx: _Context, args: argparse.Namespace) -> dict:
    job_id = _job_id(args.job)
    client = ctx.access.ensure()
    try:
        reply = client.cancel(job_id)
    finally:
        client.close()
    return _result("cancel", EXIT_OK, job=reply.get("job"), job_id=job_id,
                   status=reply.get("status"))


def _cmd_recheck(ctx: _Context, args: argparse.Namespace) -> dict:
    job_id = _job_id(args.job)
    client = ctx.access.ensure()
    try:
        reply = client.recheck(job_id)
    finally:
        client.close()
    job = reply.get("job") or {}
    ready = bool(reply.get("ready_to_retry"))
    extra = {"preflight": reply.get("preflight"), "ready_to_retry": ready}
    if ready:
        return _result("recheck", EXIT_OK, job=job, job_id=job_id, **extra)
    return _result("recheck", EXIT_BLOCKED, job.get("reason_code", "ok"), job=job, job_id=job_id,
                   **extra)


def _observe(ctx: _Context, command: str, client: Any, job: Mapping[str, Any],
             extra: Mapping[str, Any]) -> dict:
    """Poll until a terminal state.  Ctrl+C detaches the observer; it never cancels."""
    job_id = job.get("job_id")
    last = None
    try:
        while True:
            state = job.get("state")
            if state != last:
                ctx.progress(job)
                last = state
            code = exit_for_job_state(state)
            if code is not None:
                return _job_result(command, job, code, **extra)
            ctx.sleep(ctx.interval)
            job = client.get_job(job_id)["job"]
    except KeyboardInterrupt:
        return _result(command, EXIT_INTERRUPTED, detached=True, job_id=job_id,
                       message="observer detached; job keeps running",
                       next_action=f"podmuncher job {job_id}")
    except Exception as error:  # noqa: BLE001 - keep the retained ID on observer failure
        return _error_result(command, error, job_id=job_id, detached=False)


def _submit_like(ctx: _Context, command: str, args: argparse.Namespace,
                 call: Callable[[Any, str], Mapping[str, Any]]) -> dict:
    key = ctx.access.new_key()
    client = ctx.access.ensure()
    try:
        reply = call(client, key)
        job = reply["job"]
        extra = {k: v for k, v in reply.items() if k != "job"}
        if args.detach:
            return _result(command, EXIT_OK, detached=True, job=dict(job),
                           job_id=job.get("job_id"), **extra)
        return _observe(ctx, command, client, job, extra)
    finally:
        client.close()


def _cmd_submit(ctx: _Context, args: argparse.Namespace) -> dict:
    request = build_request(args, ctx.environ)
    return _submit_like(ctx, "submit", args, lambda client, key: client.submit(request, key))


def _cmd_retry(ctx: _Context, args: argparse.Namespace) -> dict:
    job_id = _job_id(args.job)
    return _submit_like(ctx, "retry", args, lambda client, key: client.retry(job_id, key))


def _cmd_clone(ctx: _Context, args: argparse.Namespace) -> dict:
    job_id = _job_id(args.job)
    request = build_request(args, ctx.environ)
    return _submit_like(ctx, "clone", args,
                        lambda client, key: client.clone(job_id, request, key))


def _cmd_service(ctx: _Context, args: argparse.Namespace) -> dict:
    command = f"service {args.action}"
    if args.action == "start":
        info = ctx.access.start()
        return _result(command, EXIT_OK, service_state=info.get("state", "running"),
                       pid=info.get("pid"))
    outcome = ctx.access.stop()
    if outcome.ok:
        return _result(command, EXIT_OK, service_state="stopped", outcome=outcome.outcome)
    reason = outcome.reason_code
    return _result(command, exit_for_reason(reason), reason, service_state="running",
                   outcome=outcome.outcome)


def _cmd_cache(ctx: _Context, args: argparse.Namespace) -> dict:
    command = f"cache {args.action}"
    if args.action == "list":
        client = ctx.access.connect()
        if client is None:
            return _stopped(command)
        try:
            reply = client.list_cache(args.limit, args.cursor)
        finally:
            client.close()
        return _result(command, EXIT_OK, entries=reply.get("entries", []),
                       next_cursor=reply.get("next_cursor"))
    client = ctx.access.ensure()
    try:
        reply = client.remove_cache(args.entry)
    finally:
        client.close()
    return _result(command, EXIT_OK, removed=bool(reply.get("removed")),
                   entry_id=reply.get("entry_id", args.entry))


def _future_ui(command: str, module_name: str) -> dict:
    """Import the future UI module only now.  Missing module: implementation pending."""
    qualified = f"{__package__}.{module_name}"
    try:
        module = importlib.import_module(qualified)
    except ImportError as error:
        if getattr(error, "name", None) == qualified:
            return _result(command, EXIT_FAILED, ReasonCode.SERVICE_UNAVAILABLE, pending=True,
                           message=f"{command} implementation pending")
        return _result(command, EXIT_FAILED, ReasonCode.SERVICE_UNAVAILABLE, pending=False,
                       message=f"{command} is unavailable")
    entry = getattr(module, "main", None)
    if not callable(entry):
        return _result(command, EXIT_FAILED, ReasonCode.SERVICE_UNAVAILABLE, pending=True,
                       message=f"{command} implementation pending")
    code = entry()
    result = _result(command, code if isinstance(code, int) else EXIT_OK)
    result["_silent"] = True
    return result


def _cmd_tui(ctx: _Context, args: argparse.Namespace) -> dict:
    return _future_ui("tui", "tui")


def _cmd_setup(ctx: _Context, args: argparse.Namespace) -> dict:
    """Guided setup applied by the service, which is started or attached only once the
    setup is going to run: it holds the setup lease, the Docker session and the probe
    port, so provisioning (pulls, the in-worker probe, model acquisition, the smoke) runs
    there.  Prompts and the answers file (``PODMUNCHER_SETUP_ANSWERS``) stay here."""
    from . import setup

    def applier() -> Any:
        return setup.ServiceSetup(ctx.access.ensure(), sleep=ctx.sleep)

    code = setup.main(environ=ctx.environ, applier=applier)
    result = _result("setup", code if isinstance(code, int) else EXIT_OK)
    result["_silent"] = True
    return result


def _dispatch_key(args: argparse.Namespace) -> str:
    return "service" if args.command == "service" else \
        "cache" if args.command == "cache" else args.command


_HANDLERS: Mapping[str, Callable[[_Context, argparse.Namespace], dict]] = {
    "status": _cmd_status, "doctor": _cmd_doctor, "jobs": _cmd_jobs, "job": _cmd_job,
    "cancel": _cmd_cancel, "recheck": _cmd_recheck, "retry": _cmd_retry, "clone": _cmd_clone,
    "service": _cmd_service, "cache": _cmd_cache, "tui": _cmd_tui, "setup": _cmd_setup,
    "submit": _cmd_submit,
}


# ------------------------------------------------------------------------------------- entry


def _interactive(environ: Mapping[str, str]) -> bool:
    try:
        tty = bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False
    return tty and environ.get("TERM", "") != "dumb"


def main(argv: Sequence[str] | None = None, *, access: Any = None,
         sleep: Callable[[float], None] | None = None,
         poll_interval: float = POLL_INTERVAL_SECONDS,
         environ: Mapping[str, str] | None = None) -> int:
    """Run one invocation and return its stable exit code.  ``--help``/``--version`` exit."""
    env = os.environ if environ is None else environ
    tokens = list(sys.argv[1:] if argv is None else argv)
    args, parser = parse_args(tokens)
    as_json = bool(getattr(args, "json", False))
    command = args.command
    if command is None:
        if as_json or not _interactive(env):
            sys.stdout.write(parser.format_help())
            return EXIT_OK
        command = args.command = "tui"
    ctx = _Context(access or _ServiceAccess(env), as_json, sleep or time.sleep, poll_interval,
                   env, _DEFAULT_ADAPTER, bool(getattr(args, "no_color", False)))
    try:
        ctx.adapter = _active_adapter(as_json)
        if _client_startup is not None:
            _client_startup()
        result = _HANDLERS[command](ctx, args)
    except KeyboardInterrupt:
        result = _result(command, EXIT_INTERRUPTED, message="interrupted")
    except Exception as error:  # noqa: BLE001 - every failure maps to a closed reason code
        result = _error_result(command, error)
    if not result.pop("_silent", False):
        _emit(ctx, result)
    return int(result["exit_code"])


__all__ = (
    "COMMANDS", "EXIT_FOR_REASON", "JOB_STATE_EXIT", "PlainPresentation", "PresentationAdapter",
    "build_request", "configure_presentation", "exit_for_job_state", "exit_for_reason", "main",
    "parse_args", "set_client_startup", "set_registry_provider",
)
