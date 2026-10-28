"""Responsive Textual dashboard: Overview, New job, Job detail, Models/Storage, Setup.

Design rules enforced here:

* Every service call runs off the UI loop (``asyncio.to_thread`` inside Textual workers).
  The loop only applies immutable results to the view model and re-renders.
* Snapshots and job lists are polled at 1 Hz; a tick is skipped while the previous one runs.
* All screen state lives in :class:`ViewModel` on the app, never in widgets, so a resize below
  the 80x24 minimum hides the dashboard and shows a notice without losing selection or input.
* Status is always a word.  Unknown/Stale/Error show the reason and are never rendered as 0.
* Untrusted text (titles, reasons, entry names) is escaped before it reaches a widget.
* Detach and exit never cancel anything.  Cancel and cache removal are separate, deliberate,
  named confirmations.

Escaping, width truncation, key policy, terminal profile, the welcome wordmark and terminal
restore come from :mod:`podmuncher.presentation` (shared with the CLI display adapter).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, ClassVar

from rich.cells import cell_len
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.events import Resize
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, OptionList, Static
from textual.widgets.option_list import Option

from . import presentation
from .domains import DomainValidationError, JobState, ReasonCode
from .presentation import TerminalProfile

MIN_WIDTH = 80
MIN_HEIGHT = 24
WIDE_WIDTH = 120
RESOURCES_WIDTH = 56  # resources panel width beside the job list at >= WIDE_WIDTH
POLL_SECONDS = 1.0
RECONNECT_TICKS = 5
# Matches config ``monitoring.stale_after_seconds`` (default 3): a snapshot older than this
# is shown as Stale even when the service still answers.
STALE_AFTER_SECONDS = 3.0
PAGE_SIZE = 50  # jobs/cache page size; the API caps pages at 100
EXIT_OK = 0
EXIT_INTERRUPTED = 130

VIEWS = ("overview", "new", "detail", "models", "setup")
VIEW_LABELS = {"overview": "Overview", "new": "New job", "detail": "Job detail",
               "models": "Models/Storage", "setup": "Setup/Settings"}
TERMINAL_STATES = frozenset({"completed", "partial", "failed", "cancelled", "interrupted"})
RETRYABLE_STATES = frozenset({"failed", "partial", "interrupted", "cancelled"})
STATUS_WORDS = {"ok": "OK", "unknown": "Unknown", "stale": "Stale", "error": "Error",
                "not_applicable": "Not applicable"}

# Plain fallback profile (no colour, no wordmark) when the caller does not detect one.
_PLAIN_PROFILE = TerminalProfile(color="none", theme="auto", reduced_motion=True, unicode=False,
                                 fullscreen_ok=False)
_THEMES = {"light": "textual-light", "dark": "textual-dark"}
# Dashboard actions bound to text keys; presentation.key_action decides when they are text.
_TEXT_KEY_ACTIONS = {"detach": "q", "new_job": "n", "help": "question_mark"}

# App-level single-key actions that only apply on the dashboard itself (not in dialogs).
_DASHBOARD_ACTIONS = frozenset({"detach", "new_job", "help", "back", "view", "cancel_job",
                                "recheck", "retry_job", "clone_job", "remove_cache", "page"})

# Closed reasons that mean the service itself is gone when no HTTP status came back (the
# client is dropped and reconnected).  Every other closed reason, and any answered reply
# other than service_stopped, is a per-request error: the client is kept and the rest of
# the refresh still applies.  See ``_loses_client``.
_LOST_REASONS = frozenset({ReasonCode.SERVICE_UNAVAILABLE, ReasonCode.SERVICE_STOPPED,
                           ReasonCode.SERVICE_LOST})

_RESOURCE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("service", "Service", "text"), ("docker", "Docker", "avail"), ("worker", "Worker", "run"),
    ("host_ram", "Host RAM", "bytes"), ("worker_ram", "Worker RAM", "bytes"),
    ("cpu", "CPU", "pct"), ("gpu_capability", "GPU", "present"),
    ("gpu_utilization", "GPU use", "pct"), ("gpu_vram", "GPU VRAM", "bytes"),
    ("model", "Model", "text"),
)


# ---------------------------------------------------------------------- pure helpers


def safe(text: object) -> str:
    """Escape controls, ESC, bidi controls and markup brackets from untrusted text."""
    return presentation.escape_display("" if text is None else str(text))


def fit(text: str, width: int) -> str:
    """Truncate to ``width`` terminal cells on grapheme boundaries (display width)."""
    return presentation.truncate_display(text, width)


def pad(text: str, width: int) -> str:
    cut = fit(text, width)
    return cut + " " * max(0, width - cell_len(cut))


def freeze(value: Any) -> Any:
    """Recursively convert dict/list payloads into read-only mappings and tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def parse_job_state(value: object) -> JobState | None:
    """The closed job state, or None when the value is not in the domain."""
    try:
        return JobState.parse(value)
    except DomainValidationError:
        return None


def parse_reason(value: object) -> ReasonCode | None:
    try:
        return ReasonCode.parse(value)
    except DomainValidationError:
        return None


def state_word(value: object) -> str:
    state = parse_job_state(value)
    return "Unknown" if state is None else state.value.replace("_", " ").capitalize()


def reason_word(value: object) -> str:
    """Closed reason code text; ``''`` for none/ok; ``unrecognised`` for out-of-domain."""
    if value is None:
        return ""
    reason = parse_reason(value)
    if reason is None:
        return "unrecognised"
    return "" if reason is ReasonCode.OK else reason.value


def human_bytes(value: object) -> str:
    if type(value) not in (int, float) or value < 0:
        return "Unknown"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return "Unknown"


def human_duration(value: object) -> str:
    if type(value) not in (int, float) or value < 0:
        return "Unknown"
    seconds = int(value)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def parse_utc(stamp: object) -> float | None:
    if type(stamp) is not str:
        return None
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except ValueError:
        return None


def age_text(stamp: object, now: float) -> str:
    moment = parse_utc(stamp)
    if moment is None:
        return "age unknown"
    seconds = max(0.0, now - moment)
    return "age " + human_duration(seconds)


def _format_value(kind: str, value: object) -> str:
    if kind == "bytes":
        return human_bytes(value)
    if kind == "pct":
        return f"{value:g}%" if type(value) in (int, float) else "Unknown"
    if kind == "duration":
        return human_duration(value)
    if kind == "avail":
        return "Available" if value is True else "Unavailable" if value is False else "Unknown"
    if kind == "run":
        return "Running" if value is True else "Not running" if value is False else "Unknown"
    if kind == "present":
        return "Present" if value is True else "Absent" if value is False else "Unknown"
    return safe(value) if value is not None else "Unknown"


def measure_text(measure: object, kind: str, *, stale: bool, now: float,
                 default_source: str) -> str:
    """``value - OK`` / ``Unknown (reason)`` plus ``(source, age)``; never a fake zero."""
    if not isinstance(measure, Mapping):
        return "Unknown (no data)"
    status = str(measure.get("status"))
    source = safe(measure.get("source") or default_source)
    tail = f"({source}, {age_text(measure.get('sample_time'), now)})"
    reason = reason_word(measure.get("reason"))
    if status == "ok":
        word = "Stale (last known)" if stale else "OK"
        return f"{_format_value(kind, measure.get('value'))} - {word} {tail}"
    word = STATUS_WORDS.get(status, "Unknown")
    return f"{word}{f' ({reason})' if reason else ''} {tail}"


def job_title(job: Mapping[str, Any], width: int = 60) -> str:
    title = job.get("title")
    text = title if type(title) is str and title else \
        f"{job.get('source_kind', 'job')} {job.get('output_mode', '')}".strip()
    return fit(safe(text), width)


# ------------------------------------------------------------------------- view model


@dataclass(frozen=True)
class Page:
    """Where a paged list stands: the cursor shown, the cursor after it, the way back."""

    cursor: str | None = None          # None is the newest page
    next_cursor: str | None = None     # None when there is no older page
    trail: tuple[str | None, ...] = ()  # cursors of the newer pages, for "newer"
    error: str = ""                    # closed reason when the last refresh failed

    @property
    def number(self) -> int:
        return len(self.trail) + 1


@dataclass(frozen=True)
class ViewModel:
    """Everything the screens render.  Replaced wholesale, never mutated."""

    view: str = "overview"
    phase: str = "connecting"          # connecting | connected | unavailable
    stale: bool = False
    snapshot: Mapping[str, Any] | None = None
    snapshot_source: str = "live service"
    snapshot_error: str = ""
    jobs: tuple = ()
    jobs_loaded: bool = False
    jobs_page: Page = Page()
    cache: tuple = ()
    cache_loaded: bool = False
    cache_page: Page = Page()
    detail: Mapping[str, Any] | None = None
    detail_error: str = ""
    ready_to_retry: str | None = None  # job id whose last recheck said ready
    error: str = ""
    notice: str = ""
    setup_needed: bool = False
    doctor: Mapping[str, Any] | None = None
    doctor_pending: bool = False
    selected_job_id: str | None = None
    selected_cache_id: str | None = None
    new_input: str = ""
    new_transcript: bool = True
    new_video: bool = False
    clone_of: str | None = None
    clone_title: str = ""


# ---------------------------------------------------------------------------- screens


class ConfirmScreen(ModalScreen[bool]):
    """A deliberate confirmation that names its target.  Declining is the default."""

    BINDINGS: ClassVar = [Binding("escape", "decline", "Keep", show=False)]

    def __init__(self, title: str, body: str, confirm_label: str, keep_label: str) -> None:
        super().__init__()
        self._title, self._body = title, body
        self._confirm_label, self._keep_label = confirm_label, keep_label

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self._title, id="dialog-title", markup=False)
            yield Static(self._body, id="dialog-body", markup=False)
            yield Button(self._keep_label, id="confirm-no")
            yield Button(self._confirm_label, id="confirm-yes")

    def on_mount(self) -> None:
        self.query_one("#confirm-no", Button).focus()

    @on(Button.Pressed, "#confirm-yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def _no(self) -> None:
        self.dismiss(False)

    def action_decline(self) -> None:
        self.dismiss(False)


HELP_TEXT = """\
Keys
  Tab / Shift+Tab   move focus
  Arrows            move inside lists and the tab bar
  Enter             open or select
  Esc               back or close
  1-5               switch view
  [ / ]             newer / older page of the list (Overview, Models/Storage)
  n                 new job (outside text inputs)
  q                 detach; running jobs keep running (outside text inputs)
  ?                 this help
  Ctrl+C            exit cleanly; nothing is cancelled
Job detail
  r  recheck: same job, no new attempt; a blocked job can be retried once it says Ready
  t  retry: new attempt of the same job
  l  clone: new job you can edit
  c  cancel the job (asks to confirm, names the job)
Models/Storage
  d  remove the selected cache entry (asks to confirm, names the entry)
Setup
  r  recheck (read-only; starts no download)"""


class HelpScreen(ModalScreen[None]):
    BINDINGS: ClassVar = [Binding("escape", "close", "Close", show=False),
                Binding("question_mark", "close", "Close", show=False)]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Help", id="dialog-title", markup=False)
            yield Static(HELP_TEXT, id="dialog-body", markup=False)
            yield Button("Close (Esc)", id="help-close")

    def on_mount(self) -> None:
        self.query_one("#help-close", Button).focus()

    @on(Button.Pressed, "#help-close")
    def _close(self) -> None:
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


class TabBar(Static, can_focus=True):
    """Compact focusable tab strip; Left/Right change view."""

    BINDINGS: ClassVar = [Binding("left", "step(-1)", "Previous view", show=False),
                Binding("right", "step(1)", "Next view", show=False)]

    def action_step(self, delta: int) -> None:
        app = self.app
        if isinstance(app, PodmuncherApp):
            app.step_view(delta)


# -------------------------------------------------------------------------------- app


class PodmuncherApp(App[int]):
    """The dashboard.  All service access goes through the injected client factory."""

    CSS_PATH = str(Path(__file__).resolve().parent / "assets" / "app.tcss")
    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar = [
        Binding("ctrl+c", "interrupt", "Exit", show=False, priority=True, system=True),
        Binding("q", "detach", "Detach", show=False),
        Binding("n", "new_job", "New job", show=False),
        Binding("question_mark", "help", "Help", show=False),
        Binding("escape", "back", "Back", show=False),
        Binding("1", "view('overview')", show=False),
        Binding("2", "view('new')", show=False),
        Binding("3", "view('detail')", show=False),
        Binding("4", "view('models')", show=False),
        Binding("5", "view('setup')", show=False),
        Binding("c", "cancel_job", show=False),
        Binding("r", "recheck", show=False),
        Binding("t", "retry_job", show=False),
        Binding("l", "clone_job", show=False),
        Binding("d", "remove_cache", show=False),
        Binding("left_square_bracket", "page(-1)", "Newer page", show=False),
        Binding("right_square_bracket", "page(1)", "Older page", show=False),
    ]

    def __init__(self, client_factory: Callable[[], Any], *,
                 clock: Callable[[], float] = time.time,
                 poll_seconds: float = POLL_SECONDS,
                 status_reader: Callable[[], Mapping[str, Any]] | None = None,
                 doctor: Callable[[], Mapping[str, Any]] | None = None,
                 request_builder: Callable[[str, bool, bool], Mapping[str, Any]] | None = None,
                 key_factory: Callable[[], str] | None = None,
                 profile: TerminalProfile | None = None) -> None:
        super().__init__()
        self.profile = profile or _PLAIN_PROFILE
        self._client_factory = client_factory
        self._clock = clock
        self._poll_seconds = poll_seconds
        self._status_reader = status_reader
        self._doctor = doctor or _default_doctor
        self._request_builder = request_builder or _default_request_builder
        self._key_factory = key_factory or _default_key
        self._client: Any = None
        self._io_lock = threading.Lock()
        self._polling = False
        self._ticks_since_attempt = RECONNECT_TICKS
        self._submit_key: str | None = None
        self._parked_focus: Any = None
        self._job_ids: tuple[str, ...] = ()
        self._job_rows: tuple[str, ...] = ()
        # Selection and list highlight as of the last Overview render (see _render_overview).
        self._shown_selection: str | None = None
        self._shown_highlight: int | None = None
        # Explicit selection changes (submit, prune, open) are counted separately from
        # highlight moves the list reported, so a render never undoes the user's navigation.
        self._explicit_selection = 0
        self._shown_explicit = 0
        self._cache_ids: tuple[str, ...] = ()
        self._cache_rows: tuple[str, ...] = ()
        self.model = ViewModel()
        self._home: Any = None
        # No animations: reduced motion is the only mode this dashboard has.
        self.animation_level = "none"

    def _q(self, selector: str, expect_type: Any = None) -> Any:
        """Query the dashboard screen even while a modal dialog is on top."""
        base = self._home if self._home is not None else self.screen
        return base.query_one(selector, expect_type)

    # ------------------------------------------------------------------ composition
    def compose(self) -> ComposeResult:
        with Vertical(id="main"):
            yield Static("", id="readiness", markup=False)
            yield TabBar("", id="tabs", markup=False)
            yield Static("", id="notice", markup=False, classes="-empty")
            with Vertical(id="body"):
                with Vertical(id="view-overview", classes="view -active"):
                    with Vertical(id="jobs-panel"):
                        yield Static("", id="jobs-head", markup=False)
                        yield OptionList(id="jobs")
                    with VerticalScroll(id="resources-panel"):
                        yield Static("", id="resources", markup=False)
                with Vertical(id="view-new", classes="view"), Vertical(id="new-form"):
                    yield Static("", id="new-head", markup=False)
                    yield Input(placeholder="Local path or public HTTPS/YouTube URL",
                                id="source")
                    yield Checkbox("Transcript", True, id="out-transcript")
                    yield Checkbox("Original video", False, id="out-video")
                    yield Static("", id="new-note", markup=False)
                    yield Button("Submit job (Enter in the source field)", id="btn-submit",
                                 classes="-primary")
                with Vertical(id="view-detail", classes="view"):
                    with Vertical(id="actions"):
                        yield Button("Recheck: same job, no new attempt (r)", id="btn-recheck")
                        yield Button("Retry: new attempt of this job (t)", id="btn-retry")
                        yield Button("Clone: new job you can edit (l)", id="btn-clone")
                        yield Button("Cancel this job... (c)", id="btn-cancel")
                    with VerticalScroll(id="detail-scroll"):
                        yield Static("", id="detail", markup=False)
                with Vertical(id="view-models", classes="view"):
                    with VerticalScroll(id="storage-panel"):
                        yield Static("", id="storage", markup=False)
                    yield Static("", id="cache-head", markup=False)
                    yield OptionList(id="cache")
                with Vertical(id="view-setup", classes="view"):
                    with VerticalScroll(id="setup-scroll"):
                        yield Static("", id="setup-body", markup=False)
                    yield Button("Recheck (read-only, starts nothing) (r)", id="btn-setup-recheck")
            yield Static("", id="hint", markup=False)
        yield Static("", id="resize-notice", markup=False)

    def on_mount(self) -> None:
        self._home = self.screen
        theme = _THEMES.get(self.profile.theme)
        if theme is not None:
            # An unknown theme name keeps the terminal default.
            with contextlib.suppress(Exception):
                self.theme = theme
        self.set_interval(self._poll_seconds, self._tick, name="poll")
        self._apply_size(self.size.width, self.size.height)
        self._render()
        self._poll_now()

    async def on_unmount(self) -> None:
        await self.shutdown_client()

    async def shutdown_client(self) -> None:
        """Drop the local connection.  Never cancels or stops anything server-side."""
        client, self._client = self._client, None
        if client is not None:
            await asyncio.to_thread(client.close)

    # ------------------------------------------------------------------ model plumbing
    def _set(self, **changes: Any) -> None:
        if "selected_job_id" in changes:
            self._explicit_selection += 1
        self.model = replace(self.model, **changes)
        self._render()

    async def _io(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run one blocking client call in a worker thread; calls never overlap."""
        def run() -> Any:
            with self._io_lock:
                return fn(*args)
        return await asyncio.to_thread(run)

    # ------------------------------------------------------------------ polling
    def _tick(self) -> None:
        # Age and Stale follow the clock, not a reply: redraw from the current model every
        # tick so a slow call (up to the client timeout) never leaves "age 2s" on screen
        # past the stale window.  Pure rendering; no client call.
        self._render()
        self._poll_now()

    def _poll_now(self) -> None:
        if self._polling:
            return
        self._polling = True
        self.run_worker(self._poll(), group="poll", exit_on_error=False)

    async def _poll(self) -> None:
        try:
            if self._client is None:
                await self._connect()
            if self._client is not None:
                await self._refresh(self._client)
        except Exception as error:  # noqa: BLE001 - never crash the dashboard from a poll
            await self._lost(_error_reason(error))
        finally:
            self._polling = False

    async def _connect(self) -> None:
        self._ticks_since_attempt += 1
        if self._ticks_since_attempt < RECONNECT_TICKS:
            return
        self._ticks_since_attempt = 0
        reason = "service_unavailable"
        client = None
        try:
            client = await asyncio.to_thread(self._client_factory)
        except Exception as error:  # noqa: BLE001
            reason = _error_reason(error)
        if client is not None:
            self._client = client
            return
        snapshot = None
        if self._status_reader is not None:
            try:
                saved = await asyncio.to_thread(self._status_reader)
                snapshot = saved.get("snapshot") if isinstance(saved, Mapping) else None
            except Exception:  # noqa: BLE001
                snapshot = None
        keep = self.model.snapshot if snapshot is None else None
        self._set(phase="unavailable", stale=True, error=reason,
                  snapshot=freeze(snapshot) if snapshot is not None else keep,
                  snapshot_source="saved snapshot" if snapshot is not None
                  else self.model.snapshot_source,
                  jobs_loaded=True,
                  setup_needed=snapshot is None and self.model.snapshot is None)

    async def _lost(self, reason: str) -> None:
        client, self._client = self._client, None
        self._ticks_since_attempt = 0
        if client is not None:
            await asyncio.to_thread(client.close)
        self._set(phase="unavailable", stale=True, error=reason, jobs_loaded=True)

    async def _request(self, fn: Callable[..., Any], *args: Any) -> tuple[Any, BaseException | None]:
        """One refresh request: ``(reply, None)`` or ``(None, error)`` for a per-request
        API error.  Transport and identity failures propagate so the poll drops the client."""
        try:
            return await self._io(fn, *args), None
        except Exception as error:  # classified, never swallowed blindly
            if _loses_client(error):
                raise
            return None, error

    async def _refresh(self, client: Any) -> None:
        """Poll the service.  Each request fails on its own; the successful parts still apply."""
        model = self.model
        changes: dict[str, Any] = {"phase": "connected", "stale": False, "error": "",
                                   "setup_needed": False, "jobs_loaded": True, "detail_error": ""}
        snap, failure = await self._request(client.snapshot)
        if failure is None:
            changes.update(snapshot=freeze(snap.get("snapshot")), snapshot_source="live service")
        changes["snapshot_error"] = "" if failure is None else _error_reason(failure)
        listing, failure = await self._request(client.list_jobs, PAGE_SIZE, model.jobs_page.cursor)
        changes["jobs_page"] = _paged(model.jobs_page, listing, failure)
        if failure is None:
            changes["jobs"] = freeze(listing.get("jobs", []))
        wanted: str | None = None
        if model.view == "detail" and model.selected_job_id:
            wanted = model.selected_job_id
            reply, failure = await self._request(client.get_job, wanted)
            if self.model.selected_job_id != wanted:
                pass  # A newer selection won while this lookup ran; its own poll fetches it.
            elif failure is None:
                changes["detail"] = freeze(reply.get("job"))
            elif _job_gone(failure):
                # Pruned (retention) or unknown id: drop the selection instead of reconnecting.
                changes.update(selected_job_id=None, detail=None, ready_to_retry=None,
                               notice=f"Job {safe(wanted)} no longer exists.")
            else:
                changes["detail_error"] = _error_reason(failure)
        if model.view == "models" or not model.cache_loaded and model.view != "overview":
            reply, failure = await self._request(client.list_cache, PAGE_SIZE,
                                                 model.cache_page.cursor)
            changes["cache_page"] = _paged(model.cache_page, reply, failure)
            if failure is None:
                changes.update(cache=freeze(reply.get("entries", [])), cache_loaded=True)
        # The selection may have changed during any await above (including list_cache):
        # selection-dependent results for an older selection are dropped, never applied.
        if wanted is not None and self.model.selected_job_id != wanted:
            for name in ("detail", "selected_job_id", "ready_to_retry", "notice", "detail_error"):
                changes.pop(name, None)
        # A page change while this tick was in flight wins; its own poll follows.
        repoll = False
        for name in ("jobs", "cache"):
            if getattr(self.model, f"{name}_page").cursor != getattr(model, f"{name}_page").cursor:
                changes.pop(name, None)
                changes.pop(f"{name}_page", None)
                repoll = True
        self._set(**changes)
        if repoll:
            self.call_later(self._poll_now)

    # ------------------------------------------------------------------ rendering
    def _now(self) -> float:
        return self._clock()

    def _effective_stale(self) -> bool:
        """Stale when the service is gone, the snapshot says so, or it is older than the
        configured stale window."""
        model = self.model
        if model.stale or model.snapshot is None:
            return model.stale
        if model.snapshot.get("freshness") == "stale":
            return True
        moment = parse_utc(model.snapshot.get("sample_time"))
        return moment is not None and self._now() - moment > STALE_AFTER_SECONDS

    def _render(self) -> None:
        if not self.is_running:
            return
        self._render_chrome()
        self._render_overview()
        self._render_new()
        self._render_detail()
        self._render_models()
        self._render_setup()

    def _render_chrome(self) -> None:
        model = self.model
        width = max(20, self.size.width - 2)
        self._q("#readiness", Static).update(fit(self._readiness(), width))
        tabs = []
        for index, name in enumerate(VIEWS, 1):
            label = f"{index} {VIEW_LABELS[name]}"
            tabs.append(f">{label}<" if name == model.view else f" {label} ")
        self._q("#tabs", TabBar).update(fit(" ".join(tabs), width))
        lines = []
        if model.error:
            lines.append(f"Error: {safe(model.error)}. Showing last known data as Stale; "
                         "retrying.")
        problems = [f"{what} ({safe(reason)})" for what, reason in (
            ("snapshot", model.snapshot_error), ("jobs", model.jobs_page.error),
            ("job detail", model.detail_error), ("cache", model.cache_page.error)) if reason]
        if problems and not model.error:
            lines.append("Not refreshed: " + ", ".join(problems)
                         + ". Showing the last known data; retrying.")
        if model.notice:
            lines.append(model.notice)
        notice = self._q("#notice", Static)
        notice.update("\n".join(lines))
        notice.set_class(not lines, "-empty")
        hint = "Tab focus  Enter open  Esc back  n new job  [ ] page  ? help  q detach"
        self._q("#hint", Static).update(fit(hint, width))
        for name in VIEWS:
            self._q(f"#view-{name}").set_class(name == model.view, "-active")

    def _readiness(self) -> str:
        model = self.model
        now = self._now()
        if model.phase == "connecting" and model.snapshot is None:
            return "Loading: connecting to the service"
        snapshot = model.snapshot
        if snapshot is None:
            if model.setup_needed:
                return "Setup needed: no service or snapshot yet (press 5 for Setup)"
            return "Unknown: no snapshot available"
        stale = self._effective_stale()
        blocked = any(parse_job_state(j.get("state")) is JobState.BLOCKED for j in model.jobs)
        docker = snapshot.get("docker") or {}
        docker_down = docker.get("status") == "ok" and docker.get("value") is False
        service = snapshot.get("service") or {}
        running = service.get("status") == "ok" and service.get("value") == "running"
        if blocked or docker_down:
            lead = "Blocked"
        elif stale:
            lead = "Stale"
        elif running:
            lead = "Ready"
        else:
            lead = "Unknown"
        parts = [lead, "Service " + _short(service, "text", stale),
                 "Docker " + _short(docker, "avail", stale)]
        storage = snapshot.get("storage") or ()
        if storage:
            parts.append("Storage free " + _short(storage[0].get("free") or {}, "bytes", stale))
        if snapshot.get("job_id"):
            stage = snapshot.get("stage") or "active"
            progress = snapshot.get("progress") or {}
            tail = f" {_format_value('pct', progress.get('value'))}" \
                if progress.get("status") == "ok" else ""
            parts.append(f"Active {safe(stage)}{tail}")
        parts.append(model.snapshot_source + (", " + age_text(snapshot.get("sample_time"), now)))
        return " | ".join(parts)

    def _jobs_rows(self) -> list[str]:
        now = self._now()
        width = max(40, self.size.width - 4)
        if self.size.width >= WIDE_WIDTH:
            width = max(40, self.size.width - RESOURCES_WIDTH - 4)
        extras = [reason_word(job.get("reason_code")) or safe(job.get("next_action") or "")
                  for job in self.model.jobs]
        extra_w = 18 if any(extras) else 0
        title_w = max(10, width - 12 - extra_w - 10 - 3)
        rows = []
        for job, extra in zip(self.model.jobs, extras, strict=True):
            cells = [pad(state_word(job.get("state")), 12), pad(job_title(job, title_w), title_w)]
            if extra_w:
                cells.append(pad(extra, extra_w))
            cells.append(age_text(job.get("updated_at"), now))
            rows.append(" ".join(cells))
        return rows

    def _render_overview(self) -> None:
        model = self.model
        head = self._q("#jobs-head", Static)
        option_list = self._q("#jobs", OptionList)
        if not model.jobs_loaded:
            head.update("Jobs: Loading...")
        elif not model.jobs and model.jobs_page.cursor is None:
            welcome = presentation.welcome_text(self.profile)
            head.update(f"{welcome}\nJobs: none yet. Press n to start a job.")
        else:
            head.update(f"Jobs ({len(model.jobs)}){'  - Stale' if model.stale else ''}"
                        + _page_text(model.jobs_page))
        option_list.display = bool(model.jobs)
        rows = tuple(self._jobs_rows())
        ids = tuple(str(j.get("job_id")) for j in model.jobs)
        # A highlight the user just moved (its OptionHighlighted message still queued, or
        # already acknowledged) must not be undone by this render.  When the list moved
        # since the last render and no explicit selection change happened since, the list
        # is the newer source and the model follows it.  Explicit changes (submit, prune,
        # open) are tracked separately and do move the highlight.
        current = option_list.highlighted
        explicit = self._explicit_selection != self._shown_explicit
        if (current != self._shown_highlight and not explicit
                and current is not None and 0 <= current < len(self._job_ids)):
            model = self.model = replace(model, selected_job_id=self._job_ids[current])
        if rows != self._job_rows or ids != self._job_ids:
            self._job_rows, self._job_ids = rows, ids
            option_list.clear_options()
            option_list.add_options([Option(Text(row)) for row in rows])
        if ids:
            # Keep the highlight on the selected job.  A selection that vanished (pruned) is
            # only replaced by the first row on Overview; other views keep "no job selected".
            if model.selected_job_id in ids:
                wanted: int | None = ids.index(model.selected_job_id)
            else:
                wanted = 0 if model.view == "overview" else None
            if wanted is not None and option_list.highlighted != wanted:
                option_list.highlighted = wanted
                if model.selected_job_id not in ids:
                    self.model = replace(self.model, selected_job_id=ids[0])
        self._shown_selection = self.model.selected_job_id
        self._shown_highlight = option_list.highlighted
        self._shown_explicit = self._explicit_selection
        self._q("#resources", Static).update(self._resources_text())

    def _resources_text(self) -> str:
        model = self.model
        snapshot = model.snapshot
        if snapshot is None:
            return "Resources: Loading..." if model.phase == "connecting" \
                else "Resources: Unknown (no snapshot available)"
        now = self._now()
        stale = self._effective_stale()
        lines = [(f"Resources - {'Stale' if stale else 'Fresh'} - source: "
                  f"{model.snapshot_source} - {age_text(snapshot.get('sample_time'), now)}")]
        for key, label, kind in _RESOURCE_ROWS:
            text = measure_text(snapshot.get(key), kind, stale=stale, now=now,
                                default_source=model.snapshot_source)
            lines.append(f"{label:<11} {text}")
        for entry in snapshot.get("storage") or ():
            lines.append(self._storage_line(entry, stale, now))
        return "\n".join(lines)

    def _storage_line(self, entry: Mapping[str, Any], stale: bool, now: float) -> str:
        label = f"Volume {safe(entry.get('volume_slot'))} ({safe(entry.get('context'))})"
        source = self.model.snapshot_source
        parts = [f"{name}: " + measure_text(entry.get(name), "bytes", stale=stale, now=now,
                                             default_source=source)
                 for name in ("free", "capacity", "reserved", "required", "available_after_job")
                 if entry.get(name) is not None]
        return label + "\n  " + "\n  ".join(parts)

    def _render_new(self) -> None:
        model = self.model
        if model.clone_of:
            head = (f"Clone of {fit(safe(model.clone_title), 40)}: creates a NEW job; the "
                    "original is unchanged.")
        else:
            head = "New job: local path or public HTTPS/YouTube URL"
        self._q("#new-head", Static).update(head)
        self._q("#new-note", Static).update(
            "Model profile: chosen automatically; the reason is shown on the job.\n"
            "Storage estimate: Unknown until the service checks the input.")

    def _reset_input(self) -> None:
        """Clear the source field on an explicit model change (submit done, clone start).

        Polling never writes into the field, so a re-render can't drop typed keys.
        """
        source = self._q("#source", Input)
        if source.value:
            source.value = ""
        self.model = replace(self.model, new_input="")

    def _render_detail(self) -> None:
        model = self.model
        job = self._current_job()
        buttons = {name: self._q(f"#btn-{name}", Button)
                   for name in ("recheck", "retry", "clone", "cancel")}
        if job is None:
            self._q("#detail", Static).update(
                "No job selected. Open one from Overview with Enter.")
            for button in buttons.values():
                button.disabled = True
            return
        state = str(job.get("state"))
        buttons["recheck"].disabled = state != "blocked"
        buttons["retry"].disabled = not self._retry_allowed(job)
        buttons["clone"].disabled = False
        buttons["cancel"].disabled = state in TERMINAL_STATES or parse_job_state(state) is None
        self._q("#detail", Static).update(self._detail_text(job))
        del model

    def _current_job(self) -> Mapping[str, Any] | None:
        model = self.model
        if model.detail is not None and model.detail.get("job_id") == model.selected_job_id:
            return model.detail
        for job in model.jobs:
            if job.get("job_id") == model.selected_job_id:
                return job
        return None

    def _retry_allowed(self, job: Mapping[str, Any]) -> bool:
        """The one rule behind the Retry button and the ``t`` key: terminal states always;
        a blocked job only after a recheck that said Ready to retry."""
        state = str(job.get("state"))
        if state in RETRYABLE_STATES:
            return True
        return state == "blocked" and self.model.ready_to_retry == job.get("job_id")

    def _detail_text(self, job: Mapping[str, Any]) -> str:
        now = self._now()
        model = self.model
        reason = reason_word(job.get("reason_code"))
        lines = [
            f"Title: {job_title(job)}",
            f"Job: {safe(job.get('job_id'))}",
            f"State: {state_word(job.get('state'))}"
            + (f"  (reason: {reason})" if reason else ""),
            f"Source: {safe(job.get('source_kind'))}   Output: {safe(job.get('output_mode'))}",
            f"Attempt: {safe(job.get('attempt_id')) if job.get('attempt_id') else 'none yet'}",
            (f"Created: {age_text(job.get('created_at'), now)}   "
             f"Updated: {age_text(job.get('updated_at'), now)}"),
            f"Next action: {safe(job.get('next_action')) if job.get('next_action') else 'none'}",
        ]
        snapshot = model.snapshot
        if snapshot is not None and snapshot.get("job_id") == job.get("job_id"):
            stale = self._effective_stale()
            lines.append(f"Stage: {safe(snapshot.get('stage') or 'Unknown')}")
            for key, label, kind in (("progress", "Progress", "pct"),
                                     ("elapsed_seconds", "Elapsed", "duration"),
                                     ("stage_eta_seconds", "Stage ETA", "duration")):
                lines.append(f"{label}: " + measure_text(
                    snapshot.get(key), kind, stale=stale, now=now,
                    default_source=model.snapshot_source))
        else:
            lines.append("Live stage telemetry: Unknown (this job is not the active one)")
        artifacts = job.get("artifacts") or ()
        lines.append(f"Artifacts ({len(artifacts)}):")
        for art in artifacts:
            slot = safe(art.get("slot"))
            if art.get("available") is True:
                lines.append(f"  {slot} ({safe(art.get('format'))}, "
                             f"{human_bytes(art.get('bytes'))}) - Available")
            else:
                lines.append(f"  {slot} - Pending validation, not available yet")
        if not artifacts:
            lines.append("  none yet; each artifact appears on its own once validated")
        if model.stale:
            lines.append("Data is Stale: the service is not reachable right now.")
        if str(job.get("state")) == "blocked":
            lines.append("Retry: Ready to retry (the recheck passed)" if self._retry_allowed(job)
                         else "Retry: not yet; run Recheck (r) and retry once it says Ready")
        lines += ["Actions:", "  Recheck = same job, no new attempt",
                  "  Retry = new attempt of the same job",
                  "  Clone = new job; change source or options",
                  "  Cancel = stop this job only (asks to confirm)"]
        return "\n".join(lines)

    def _render_models(self) -> None:
        model = self.model
        now = self._now()
        stale = self._effective_stale()
        snapshot = model.snapshot
        if snapshot is None:
            storage = "Storage: Unknown (no snapshot available)"
        else:
            entries = snapshot.get("storage") or ()
            storage = "Storage - " + (f"{model.snapshot_source}, "
                                      f"{age_text(snapshot.get('sample_time'), now)}")
            storage += "\n" + ("\n".join(self._storage_line(e, stale, now) for e in entries)
                               or "  Unknown (no volumes reported)")
            storage += "\nBudget confidence: " + safe(snapshot.get("budget_confidence"))
        self._q("#storage", Static).update(storage)
        head = self._q("#cache-head", Static)
        option_list = self._q("#cache", OptionList)
        if not model.cache_loaded and model.cache_page.error:
            head.update(f"Model cache: not available ({safe(model.cache_page.error)}); retrying.")
        elif not model.cache_loaded:
            head.update("Model cache: Loading... (open this view to load)")
        elif not model.cache and model.cache_page.cursor is None:
            head.update("Model cache: no entries.")
        else:
            head.update(f"Model cache ({len(model.cache)}) - Enter or d removes the "
                        "selected entry after confirmation" + _page_text(model.cache_page))
        ids = tuple(_cache_id(e) for e in model.cache)
        rows = tuple(_cache_row(e) for e in model.cache)
        if ids != self._cache_ids or rows != self._cache_rows:
            self._cache_ids, self._cache_rows = ids, rows
            option_list.clear_options()
            option_list.add_options([Option(Text(row)) for row in rows])
            if ids:
                option_list.highlighted = ids.index(model.selected_cache_id) \
                    if model.selected_cache_id in ids else 0

    def _render_setup(self) -> None:
        model = self.model
        report = model.doctor
        lines = ["Setup and settings"]
        if model.doctor_pending:
            lines.append("Checking... (read-only)")
        elif report is None:
            lines.append("Not checked yet. Press r to run the read-only check.")
        else:
            overall = "OK" if report.get("status") == "ok" else "Blocked"
            lines.append(f"Overall: {overall}")
            for name, check in sorted((report.get("checks") or {}).items()):
                status = STATUS_WORDS.get(str(check.get("status")), str(check.get("status")))
                if check.get("status") == "blocked":
                    status = "Blocked"
                reason = reason_word(check.get("reason_code"))
                lines.append(f"  {safe(name):<14} {safe(status)}"
                             + (f" ({reason})" if reason else ""))
        lines += ["", "Provisioning, model installs and saved settings are changed with",
                  "`podmuncher --setup`. Nothing downloads when this screen opens."]
        self._q("#setup-body", Static).update("\n".join(lines))

    # ------------------------------------------------------------------ size handling
    def on_resize(self, event: Resize) -> None:
        self._apply_size(event.size.width, event.size.height)
        self._render()

    def _apply_size(self, width: int, height: int) -> None:
        small = width < MIN_WIDTH or height < MIN_HEIGHT
        screen = self._home if self._home is not None else self.screen
        if small and not screen.has_class("-too-small"):
            self._parked_focus = self.focused
        screen.set_class(small, "-too-small")
        if small:
            self._q("#resize-notice", Static).update(
                f"Terminal too small: {width}x{height}. Podmuncher needs at least "
                f"{MIN_WIDTH}x{MIN_HEIGHT}.\nResize the window to continue, or press q to "
                "detach (running jobs keep running).\nYour place and typed text are kept.")
        elif self._parked_focus is not None:
            parked, self._parked_focus = self._parked_focus, None
            if parked.is_attached:
                parked.focus()
        self._q("#view-overview").set_class(width >= WIDE_WIDTH, "-wide")

    # ------------------------------------------------------------------ view switching
    def _guard(self) -> bool:
        """Global single-key actions only apply on the main screen outside text inputs."""
        return not isinstance(self.screen, ModalScreen) and not isinstance(self.focused, Input)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action not in _DASHBOARD_ACTIONS:
            return True
        if isinstance(self.screen, ModalScreen):
            return False
        key = _TEXT_KEY_ACTIONS.get(action)
        if key is not None and presentation.key_action(
                key, in_text_input=isinstance(self.focused, Input)) is None:
            return False  # n / q / ? are plain text inside a text input
        view = self.model.view
        if action in ("detach", "new_job", "view", "cancel_job", "retry_job", "clone_job",
                      "remove_cache", "recheck", "page") and isinstance(self.focused, Input):
            return False
        if action in ("cancel_job", "retry_job", "clone_job"):
            return view == "detail"
        if action == "recheck":
            return view in ("detail", "setup")
        if action == "remove_cache":
            return view == "models"
        if action == "page":
            return view in ("overview", "models")
        return True

    def action_page(self, delta: int) -> None:
        """Step the visible list one page older (``]``) or newer (``[``); fetched off-loop."""
        name = "jobs" if self.model.view == "overview" else "cache"
        page: Page = getattr(self.model, f"{name}_page")
        if delta > 0:
            if page.next_cursor is None:
                self._set(notice="No older entries.")
                return
            moved = Page(cursor=page.next_cursor, trail=page.trail + (page.cursor,))
        else:
            if not page.trail:
                self._set(notice="Already on the newest page.")
                return
            moved = Page(cursor=page.trail[-1], trail=page.trail[:-1])
        self._set(notice=f"Loading page {moved.number}...", **{f"{name}_page": moved})
        self._poll_now()

    def set_view(self, view: str) -> None:
        if view not in VIEWS:
            return
        self._set(view=view, notice="")
        target = {"overview": "#jobs" if self.model.jobs else "#tabs", "new": "#source",
                  "detail": "#detail-scroll", "models": "#cache" if self.model.cache
                  else "#storage-panel", "setup": "#setup-scroll"}[view]
        self._q(target).focus()
        if view == "setup" and self.model.doctor is None:
            self._run_doctor()
        self._poll_now()

    def step_view(self, delta: int) -> None:
        index = (VIEWS.index(self.model.view) + delta) % len(VIEWS)
        self.set_view(VIEWS[index])
        self._q("#tabs").focus()

    def action_view(self, name: str) -> None:
        self.set_view(name)

    def action_new_job(self) -> None:
        self._submit_key = None
        self._set(clone_of=None, clone_title="")
        self.set_view("new")

    def action_back(self) -> None:
        if isinstance(self.screen, ModalScreen):
            return
        if self.model.view == "new" and self.model.clone_of:
            self._set(clone_of=None, clone_title="")
        if self.model.view != "overview":
            self.set_view("overview")

    def action_help(self) -> None:
        if not isinstance(self.screen, ModalScreen):
            self.push_screen(HelpScreen())

    def action_detach(self) -> None:
        """Detach: leave the dashboard.  Running jobs are never touched."""
        self.exit(EXIT_OK, return_code=EXIT_OK)

    def action_interrupt(self) -> None:
        self.exit(EXIT_INTERRUPTED, return_code=EXIT_INTERRUPTED)

    # ------------------------------------------------------------------ list events
    @on(OptionList.OptionHighlighted, "#jobs")
    def _job_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        # A queued message for a highlight the list has already moved past is stale:
        # acknowledging it would snap the selection back during rapid navigation.
        if event.option_index != event.option_list.highlighted:
            return
        if 0 <= event.option_index < len(self._job_ids):
            self.model = replace(self.model, selected_job_id=self._job_ids[event.option_index])

    @on(OptionList.OptionSelected, "#jobs")
    def _job_selected(self, event: OptionList.OptionSelected) -> None:
        if 0 <= event.option_index < len(self._job_ids):
            self.model = replace(self.model, selected_job_id=self._job_ids[event.option_index],
                                 detail=None)
            self.set_view("detail")

    @on(OptionList.OptionHighlighted, "#cache")
    def _cache_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if 0 <= event.option_index < len(self._cache_ids):
            self.model = replace(self.model,
                                 selected_cache_id=self._cache_ids[event.option_index])

    @on(OptionList.OptionSelected, "#cache")
    def _cache_selected(self, event: OptionList.OptionSelected) -> None:
        if 0 <= event.option_index < len(self._cache_ids):
            self.model = replace(self.model,
                                 selected_cache_id=self._cache_ids[event.option_index])
            self.action_remove_cache()

    # ------------------------------------------------------------------ new job form
    @on(Input.Changed, "#source")
    def _source_changed(self, event: Input.Changed) -> None:
        self._submit_key = None
        self.model = replace(self.model, new_input=event.value)

    @on(Checkbox.Changed, "#out-transcript")
    def _transcript_changed(self, event: Checkbox.Changed) -> None:
        self._submit_key = None
        self.model = replace(self.model, new_transcript=event.value)

    @on(Checkbox.Changed, "#out-video")
    def _video_changed(self, event: Checkbox.Changed) -> None:
        self._submit_key = None
        self.model = replace(self.model, new_video=event.value)

    @on(Input.Submitted, "#source")
    def _source_submitted(self) -> None:
        self._submit()

    @on(Button.Pressed, "#btn-submit")
    def _submit_pressed(self) -> None:
        self._submit()

    def _submit(self) -> None:
        model = self.model
        text = model.new_input.strip()
        if not text:
            self._set(notice="Enter a local path or URL first.")
            return
        if not (model.new_transcript or model.new_video):
            self._set(notice="Choose at least one output: Transcript or Original video.")
            return
        if self._submit_key is None:
            self._submit_key = self._key_factory()
        key = self._submit_key
        self._set(notice="Submitting...")
        self.run_worker(self._do_submit(text, model.new_transcript, model.new_video,
                                        model.clone_of, key), group="action",
                        exit_on_error=False)

    async def _do_submit(self, text: str, transcript: bool, video: bool,
                         clone_of: str | None, key: str) -> None:
        client = self._client
        if client is None:
            self._set(notice="Not connected to the service; try again once it is reachable.")
            return
        try:
            request = await asyncio.to_thread(self._request_builder, text, transcript, video)
            if clone_of:
                reply = await self._io(client.clone, clone_of, request, key)
            else:
                reply = await self._io(client.submit, request, key)
        except Exception as error:  # noqa: BLE001
            self._set(notice=f"Submit failed: {safe(_error_reason(error))}")
            return
        job = reply.get("job") or {}
        self._submit_key = None
        self._set(notice=("Cloned as a new job." if clone_of else "Job submitted."),
                  selected_job_id=job.get("job_id"), detail=freeze(job) if job else None,
                  clone_of=None, clone_title="")
        self._reset_input()
        self.set_view("detail")

    # ------------------------------------------------------------------ job actions
    def _target(self) -> tuple[str, str] | None:
        job = self._current_job()
        if job is None:
            return None
        return str(job.get("job_id")), job_title(job, 40)

    @on(Button.Pressed, "#btn-cancel")
    def action_cancel_job(self) -> None:
        target = self._target()
        if target is None:
            return
        job_id, title = target
        job = self._current_job()
        if job is not None and parse_job_state(job.get("state")) is not None \
                and job.get("state") in TERMINAL_STATES:
            self._set(notice=f"{title} is already finished; nothing to cancel.")
            return
        shown = f"{title} ({safe(job_id)})"
        self.push_screen(
            ConfirmScreen(f"Cancel job {shown}?",
                          "This stops this job only. Artifacts already validated are kept. "
                          "Detaching from the dashboard never cancels a job.",
                          f"Yes, cancel job {title}", "No, keep the job running"),
            lambda ok: self._confirmed("cancel", job_id, title) if ok else None)

    @on(Button.Pressed, "#btn-recheck")
    @on(Button.Pressed, "#btn-setup-recheck")
    def action_recheck(self) -> None:
        if self.model.view == "setup":
            self._run_doctor()
            return
        target = self._target()
        if target is not None:
            self._run_action("recheck", *target)

    @on(Button.Pressed, "#btn-retry")
    def action_retry_job(self) -> None:
        job = self._current_job()
        if job is None:
            return
        job_id, label = str(job.get("job_id")), job_title(job, 40)
        if not self._retry_allowed(job):
            why = ("blocked; run Recheck (r) first and retry once it says Ready to retry"
                   if str(job.get("state")) == "blocked"
                   else f"state is {state_word(job.get('state'))}")
            self._set(notice=f"Retry is not available for {label}: {why}.")
            return
        self._run_action("retry", job_id, label)

    @on(Button.Pressed, "#btn-clone")
    def action_clone_job(self) -> None:
        target = self._target()
        if target is None:
            return
        self._submit_key = None
        self._set(clone_of=target[0], clone_title=target[1])
        self._reset_input()
        self.set_view("new")

    def action_remove_cache(self) -> None:
        entry_id = self.model.selected_cache_id
        entry = next((e for e in self.model.cache if _cache_id(e) == entry_id), None)
        if entry is None:
            return
        name = fit(safe(_cache_name(entry)), 50)
        self.push_screen(
            ConfirmScreen(f"Remove cache entry {name}?",
                          "This deletes the cached files for this entry only. Running jobs "
                          "are not cancelled; it may be downloaded again later.",
                          f"Yes, remove {name}", "No, keep the entry"),
            lambda ok: self._confirmed("remove_cache", str(entry_id), name) if ok else None)

    def _confirmed(self, kind: str, target_id: str, label: str) -> None:
        self._run_action(kind, target_id, label)

    def _run_action(self, kind: str, target_id: str, label: str) -> None:
        self._set(notice=f"{kind.replace('_', ' ').capitalize()}: {label} ...")
        self.run_worker(self._do_action(kind, target_id, label), group="action",
                        exit_on_error=False)

    async def _do_action(self, kind: str, target_id: str, label: str) -> None:
        client = self._client
        if client is None:
            self._set(notice="Not connected to the service; nothing was changed.")
            return
        extra: dict[str, Any] = {}
        try:
            if kind == "cancel":
                await self._io(client.cancel, target_id)
                done = f"Cancel requested for job {label}."
            elif kind == "recheck":
                reply = await self._io(client.recheck, target_id)
                done, extra = self._recheck_outcome(reply, target_id, label)
            elif kind == "retry":
                await self._io(client.retry, target_id, self._key_factory())
                done = f"Retry started for {label}: a new attempt of the same job."
                extra["ready_to_retry"] = None
            else:
                await self._io(client.remove_cache, target_id)
                done = f"Removed cache entry {label}."
                extra["cache_loaded"] = False
        except Exception as error:  # noqa: BLE001
            self._set(notice=f"{kind.replace('_', ' ').capitalize()} failed for {label}: "
                             f"{safe(_error_reason(error))}")
            return
        self._set(notice=done, **extra)
        self._poll_now()

    def _recheck_outcome(self, reply: Any, job_id: str, label: str) -> tuple[str, dict[str, Any]]:
        """Keep what the recheck said: the refreshed job and whether Retry is now allowed."""
        doc = reply if isinstance(reply, Mapping) else {}
        preflight = doc.get("preflight") if isinstance(doc.get("preflight"), Mapping) else {}
        if "ready_to_retry" in doc:
            ready = doc.get("ready_to_retry") is True
        else:
            ready = bool(preflight) and parse_reason(preflight.get("reason_code")) is ReasonCode.OK
        extra: dict[str, Any] = {"ready_to_retry": job_id if ready else None}
        job = doc.get("job")
        if isinstance(job, Mapping) and job.get("job_id") == job_id:
            extra["detail"] = freeze(job)
        if ready:
            done = (f"Ready to retry: {label} (recheck passed; same job, no new attempt). "
                    "Press t or Retry to start a new attempt.")
            return done, extra
        reason = reason_word(preflight.get("reason_code")) if preflight else ""
        done = (f"Rechecked {label}: same job, no new attempt. Not ready to retry"
                + (f" ({reason})" if reason else "") + ".")
        return done, extra

    # ------------------------------------------------------------------ setup
    def _run_doctor(self) -> None:
        self._set(doctor_pending=True)
        self.run_worker(self._do_doctor(), group="doctor", exit_on_error=False)

    async def _do_doctor(self) -> None:
        try:
            report = await asyncio.to_thread(self._doctor)
        except Exception as error:  # noqa: BLE001
            self._set(doctor_pending=False, notice=f"Check failed: {safe(_error_reason(error))}")
            return
        self._set(doctor_pending=False, doctor=freeze(report),
                  setup_needed=self.model.setup_needed or report.get("status") != "ok"
                  and self.model.snapshot is None)


# -------------------------------------------------------------------------- utilities


def _short(measure: Mapping[str, Any], kind: str, stale: bool) -> str:
    status = measure.get("status") if isinstance(measure, Mapping) else None
    if status == "ok":
        return _format_value(kind, measure.get("value")) + (" (Stale)" if stale else "")
    return STATUS_WORDS.get(str(status), "Unknown")


def _cache_id(entry: Mapping[str, Any]) -> str:
    return str(entry.get("entry_id"))


def _cache_name(entry: Mapping[str, Any]) -> str:
    for key in ("name", "model", "label"):
        if type(entry.get(key)) is str and entry[key]:
            return entry[key]
    return _cache_id(entry)


def _cache_row(entry: Mapping[str, Any]) -> str:
    size = entry.get("bytes")
    size_text = human_bytes(size) if size is not None else "size Unknown"
    return f"{fit(safe(_cache_name(entry)), 40)}  {size_text}"


def _error_reason(error: BaseException) -> str:
    reason = getattr(error, "reason_code", None)
    return reason.value if isinstance(reason, ReasonCode) else "service_unavailable"


def _loses_client(error: BaseException) -> bool:
    """True for transport and identity failures (drop and reconnect the client); False for
    per-request API errors (keep the client).

    An error that carries an HTTP status came from a live, authenticated service, so it is
    per-request even for a 503 ``service_unavailable``; ``service_stopped`` is the one
    answered reason that still loses the client, since the service is going away."""
    if isinstance(error, OSError) or getattr(error, "identity_mismatch", False) is True:
        return True
    reason = getattr(error, "reason_code", None)
    if not isinstance(reason, ReasonCode):
        return True  # not a closed API error: the reply or transport is unusable
    if reason is ReasonCode.SERVICE_STOPPED:
        return True
    if type(getattr(error, "status", None)) is int:
        return False  # the service answered: keep the client, report the request
    return reason in _LOST_REASONS


def _job_gone(error: BaseException) -> bool:
    """A 404 / check_id / invalid_input answer for a job id: it no longer exists."""
    return (getattr(error, "status", None) == 404
            or getattr(error, "next_action", None) == "check_id"
            or getattr(error, "reason_code", None) is ReasonCode.INVALID_INPUT)


def _paged(page: Page, listing: Any, error: BaseException | None) -> Page:
    """The page after one list request: its next cursor, or its error, or a reset when a
    cursor into an older page is no longer accepted."""
    if error is None:
        cursor = listing.get("next_cursor") if isinstance(listing, Mapping) else None
        return replace(page, next_cursor=cursor if type(cursor) is str and cursor else None,
                       error="")
    if page.cursor is not None and _job_gone(error):
        return Page(error=_error_reason(error))
    return replace(page, error=_error_reason(error))


def _page_text(page: Page) -> str:
    parts = []
    if page.cursor is not None:
        parts.append(f"page {page.number}")
    if page.next_cursor is not None:
        parts.append("] older")
    if page.trail:
        parts.append("[ newer")
    if page.error:
        parts.append(f"not refreshed ({safe(page.error)})")
    return "".join(f"  - {part}" for part in parts)


def _default_key() -> str:
    from .service import new_idempotency_key
    return new_idempotency_key()


def _default_doctor() -> Mapping[str, Any]:
    from .setup import doctor_report
    return doctor_report()


def _default_request_builder(text: str, transcript: bool, video: bool) -> Mapping[str, Any]:
    from .cli import build_request
    # ``outputs`` carries both checkboxes exactly; video/no_transcript stay for older builders.
    args = SimpleNamespace(input=text, device=None, profile=None, model=None, language=None,
                           output=None, video=video, no_transcript=not transcript,
                           outputs={"transcript": transcript, "video": video})
    return build_request(args, os.environ)


def main() -> int:
    """Open the dashboard.  0 on q/detach, 130 on Ctrl+C.  Never cancels a job."""
    from . import service

    state = {"first": True}
    state_dir = os.environ.get("PODMUNCHER_STATE_DIR") or None  # empty means unset, as the CLI

    def factory() -> Any:
        client = service.connect(state_dir)
        if client is None and state["first"]:
            state["first"] = False
            service.start_service(state_dir)
            client = service.connect(state_dir)
        state["first"] = False
        return client

    profile = TerminalProfile.detect(os.environ, stdout=sys.stdout)
    app = PodmuncherApp(factory, status_reader=lambda: service.read_status(state_dir),
                        profile=profile)
    try:
        with presentation.terminal_restore(sys.stdout):
            app.run()
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    finally:
        client, app._client = app._client, None
        if client is not None:
            client.close()
    code = app.return_code
    return EXIT_OK if code is None else code


__all__ = ("Page", "PodmuncherApp", "ViewModel", "main")
