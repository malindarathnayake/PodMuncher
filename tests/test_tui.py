"""Textual dashboard tests: public behaviour through the Pilot and the rendered text."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from textual.widget import Widget
from textual.widgets import Button, Input, OptionList, Static

from podmuncher import tui
from podmuncher.domains import ReasonCode
from podmuncher.service import ClientError

SAMPLE = "2026-10-13T10:00:00Z"
# Two seconds after the sample: inside the 3 s stale window (config stale_after_seconds).
NOW = datetime.fromisoformat("2026-10-13T10:00:02+00:00").timestamp()
LATER = datetime.fromisoformat("2026-10-13T10:00:05+00:00").timestamp()
JOB_A = "11111111-1111-4111-8111-111111111111"
JOB_B = "22222222-2222-4222-8222-222222222222"
ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------ fakes


def measure(value: Any, status: str = "ok", reason: str | None = None) -> dict:
    return {"value": value, "status": status, "sample_time": SAMPLE, "reason": reason}


def make_snapshot(**overrides: Any) -> dict:
    snap = {
        "schema_version": 1, "sample_time": SAMPLE, "freshness": "ok",
        "service": measure("running"), "docker": measure(True), "worker": measure(True),
        "host_ram": measure(8 * 1024 ** 3), "worker_ram": measure(2 * 1024 ** 3),
        "cpu": measure(12.5), "gpu_capability": measure(False),
        "gpu_utilization": measure(None, "not_applicable", "gpu_unavailable"),
        "gpu_vram": measure(None, "not_applicable", "gpu_unavailable"),
        "model": measure("small"), "storage": [], "budget_confidence": "measured",
        "job_id": None, "attempt_id": None, "stage": None,
        "progress": measure(None, "not_applicable", "duration_unknown"),
        "elapsed_seconds": measure(None, "not_applicable", "duration_unknown"),
        "stage_eta_seconds": measure(None, "not_applicable", "duration_unknown"),
    }
    snap.update(overrides)
    return snap


def make_job(job_id: str, state: str = "transcribing", title: str | None = None,
             artifacts: list | None = None, reason: str = "ok") -> dict:
    job = {"job_id": job_id, "source_kind": "local", "output_mode": "transcript",
           "state": state, "reason_code": reason, "next_action": None,
           "attempt_id": None, "created_at": SAMPLE, "updated_at": SAMPLE,
           "artifacts": list(artifacts or [])}
    if title is not None:
        job["title"] = title
    return job


class FakeService:
    """Scripted client.  Records every call and the thread it ran on; close never cancels."""

    def __init__(self, *, snapshot: dict | None = None, jobs: list | None = None,
                 cache: list | None = None) -> None:
        self.loop_thread = threading.get_ident()
        self.snapshot_doc = make_snapshot() if snapshot is None else snapshot
        self.jobs = list(jobs or [])
        self.cache = list(cache or [])
        self.calls: list[tuple[str, tuple]] = []
        self.on_loop: list[str] = []
        self.gate: threading.Event | None = None
        self.gates: dict[str, threading.Event] = {}  # per-route gates; the call blocks until set
        self.fail: Exception | None = None
        self.fail_calls: dict[str, Exception] = {}  # per-route failures; the client survives
        self.ready_to_retry = False
        self.closed = 0

    def _rec(self, name: str, *args: Any) -> None:
        if threading.get_ident() == self.loop_thread:
            self.on_loop.append(name)
        self.calls.append((name, args))
        if self.gate is not None:
            self.gate.wait(5)
        if name in self.gates:
            self.gates[name].wait(5)
        if self.fail is not None and name != "close":
            raise self.fail
        if name in self.fail_calls:
            raise self.fail_calls[name]

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def snapshot(self) -> dict:
        self._rec("snapshot")
        return {"snapshot": self.snapshot_doc}

    @staticmethod
    def _page(items: list, limit: int | None, cursor: str | None) -> tuple[list, str | None]:
        start = int(cursor) if cursor else 0
        size = limit or len(items)
        page = [dict(item) for item in items[start:start + size]]
        return page, (str(start + size) if start + size < len(items) else None)

    def list_jobs(self, limit: int | None = None, cursor: str | None = None) -> dict:
        self._rec("list_jobs", limit, cursor)
        jobs, next_cursor = self._page(self.jobs, limit, cursor)
        return {"jobs": jobs, "next_cursor": next_cursor}

    def get_job(self, job_id: str) -> dict:
        self._rec("get_job", job_id)
        for job in self.jobs:
            if job["job_id"] == job_id:
                return {"job": dict(job)}
        raise ClientError(ReasonCode.INVALID_INPUT, next_action="check_id", status=404)

    def submit(self, request: dict, key: str) -> dict:
        self._rec("submit", request, key)
        job = make_job(JOB_B, "queued")
        self.jobs.append(job)
        return {"job": job, "replayed": False}

    def recheck(self, job_id: str) -> dict:
        self._rec("recheck", job_id)
        reason = "ok" if self.ready_to_retry else "storage_insufficient"
        return {"job": self.get_job(job_id)["job"], "preflight": {"reason_code": reason},
                "ready_to_retry": self.ready_to_retry}

    def retry(self, job_id: str, key: str) -> dict:
        self._rec("retry", job_id, key)
        return {"job": self.get_job(job_id)["job"], "attempt_id": None}

    def clone(self, job_id: str, request: dict, key: str) -> dict:
        self._rec("clone", job_id, request, key)
        return {"job": make_job(JOB_B, "queued")}

    def cancel(self, job_id: str) -> dict:
        self._rec("cancel", job_id)
        return {"job": self.get_job(job_id)["job"], "status": "intent"}

    def list_cache(self, limit: int | None = None, cursor: str | None = None) -> dict:
        self._rec("list_cache", limit, cursor)
        entries, next_cursor = self._page(self.cache, limit, cursor)
        return {"entries": entries, "next_cursor": next_cursor}

    def remove_cache(self, entry_id: str) -> dict:
        self._rec("remove_cache", entry_id)
        self.cache = [e for e in self.cache if e["entry_id"] != entry_id]
        return {"removed": True, "entry_id": entry_id}

    def close(self) -> None:
        self._rec("close")
        self.closed += 1


def make_fake_service(**kwargs: Any) -> FakeService:
    return FakeService(**kwargs)


def make_app(fake: FakeService | None, **kwargs: Any) -> tui.PodmuncherApp:
    loop_thread = threading.get_ident()
    factory_threads: list[int] = []

    def factory() -> Any:
        factory_threads.append(threading.get_ident())
        assert threading.get_ident() != loop_thread
        return fake

    kwargs.setdefault("doctor", lambda: {"read_only": True, "status": "ok", "checks": {}})
    kwargs.setdefault("request_builder", lambda text, transcript, video: {"input": text})
    kwargs.setdefault("clock", lambda: NOW)
    app = tui.PodmuncherApp(factory, poll_seconds=0.05, **kwargs)
    app.factory_threads = factory_threads  # type: ignore[attr-defined]
    return app


# ---------------------------------------------------------------------------- helpers


def _shown(widget: Widget) -> bool:
    node: Any = widget
    while isinstance(node, Widget):
        if not node.display:
            return False
        node = node.parent
    return True


def screen_text(app: tui.PodmuncherApp) -> str:
    """All visible text of the active screen (statics, buttons, list rows, inputs)."""
    parts: list[str] = []
    for widget in app.screen.query("*"):
        if not _shown(widget):
            continue
        if isinstance(widget, Button):
            parts.append(str(widget.label))
        elif isinstance(widget, Static):
            parts.append(str(widget.content))
        elif isinstance(widget, OptionList):
            parts += [str(widget.get_option_at_index(i).prompt)
                      for i in range(widget.option_count)]
        elif isinstance(widget, Input):
            parts.append(widget.value)
    return "\n".join(parts)


def text_of(app: tui.PodmuncherApp, selector: str) -> str:
    return str(app.screen.query_one(selector, Static).content)


async def until(pilot: Any, condition: Callable[[], bool], timeout: float = 4.0) -> None:
    for _ in range(int(timeout / 0.02)):
        if condition():
            return
        await pilot.pause(0.02)
    assert condition(), "condition not reached"


async def ready(pilot: Any) -> None:
    await until(pilot, lambda: pilot.app.model.phase == "connected")
    await pilot.pause()


async def focus_jobs(pilot: Any) -> None:
    for _ in range(10):
        if getattr(pilot.app.focused, "id", None) == "jobs":
            return
        await pilot.press("tab")
    assert getattr(pilot.app.focused, "id", None) == "jobs"


# ------------------------------------------------------------------------------ layout


@pytest.mark.asyncio
async def test_80x24_shows_tabs_and_readiness_line() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A)])
    app = make_app(fake)
    async with app.run_test(size=(80, 24)) as pilot:
        await ready(pilot)
        readiness = text_of(app, "#readiness")
        assert readiness.startswith("Ready")
        assert "Overview" in text_of(app, "#tabs") and "Models/Storage" in text_of(app, "#tabs")
        jobs = app.screen.query_one("#jobs-panel")
        resources = app.screen.query_one("#resources-panel")
        assert not app.screen.query_one("#view-overview").has_class("-wide")
        assert resources.region.y > jobs.region.y
        assert "p o d m u n c h e r" not in screen_text(app)  # no banner on the active view
    assert fake.on_loop == []


@pytest.mark.asyncio
async def test_120x36_puts_jobs_and_resources_side_by_side() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A)])
    app = make_app(fake)
    async with app.run_test(size=(120, 36)) as pilot:
        await ready(pilot)
        jobs = app.screen.query_one("#jobs-panel")
        resources = app.screen.query_one("#resources-panel")
        assert app.screen.query_one("#view-overview").has_class("-wide")
        assert resources.region.x > jobs.region.x
        assert resources.region.y == jobs.region.y
        assert "Host RAM" in screen_text(app)


@pytest.mark.asyncio
async def test_resize_below_minimum_shows_notice_and_restores_state() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A), make_job(JOB_B)])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await focus_jobs(pilot)
        await pilot.press("down")
        assert app.model.selected_job_id == JOB_B
        await pilot.press("n")
        await pilot.press("a", "b", "c")
        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        text = screen_text(app)
        assert "Terminal too small: 60x20" in text
        assert "q to detach" in text
        assert "Jobs" not in text
        await pilot.resize_terminal(100, 30)
        await pilot.pause()
        assert "Terminal too small" not in screen_text(app)
        assert app.model.view == "new"
        assert app.screen.query_one("#source", Input).value == "abc"
        assert app.model.selected_job_id == JOB_B
        await pilot.press("escape")
        assert app.model.view == "overview"
        assert app.model.selected_job_id == JOB_B


# ------------------------------------------------------------------------------ states


@pytest.mark.asyncio
async def test_setup_needed_when_no_service_and_no_snapshot() -> None:
    app = make_app(None)
    async with app.run_test(size=(100, 30)) as pilot:
        await until(pilot, lambda: app.model.phase == "unavailable")
        await pilot.pause()
        assert text_of(app, "#readiness").startswith("Setup needed")
        assert app.factory_threads  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stopped_service_shows_saved_snapshot_as_stale() -> None:
    def reader() -> dict:
        return {"state": "stopped", "snapshot": make_snapshot(), "stale": True}

    app = make_app(None, status_reader=reader)
    async with app.run_test(size=(100, 30)) as pilot:
        await until(pilot, lambda: app.model.snapshot is not None)
        await pilot.pause()
        assert text_of(app, "#readiness").startswith("Stale")
        resources = text_of(app, "#resources")
        assert "saved snapshot" in resources and "Stale" in resources


@pytest.mark.asyncio
async def test_empty_and_loading_states() -> None:
    fake = make_fake_service()
    fake.gate = threading.Event()
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await until(pilot, lambda: "snapshot" in fake.names())
        assert text_of(app, "#readiness").startswith("Loading")
        assert "Loading" in text_of(app, "#jobs-head")
        fake.gate.set()
        fake.gate = None
        await ready(pilot)
        assert "none yet" in text_of(app, "#jobs-head")


@pytest.mark.asyncio
async def test_blocked_partial_completed_rows_and_readiness() -> None:
    jobs = [make_job(JOB_A, "blocked", "Needs space", reason="storage_unknown"),
            make_job(JOB_B, "partial", "Half done"),
            make_job("33333333-3333-4333-8333-333333333333", "completed", "All done")]
    fake = make_fake_service(jobs=jobs)
    app = make_app(fake)
    async with app.run_test(size=(120, 36)) as pilot:
        await ready(pilot)
        text = screen_text(app)
        assert text_of(app, "#readiness").startswith("Blocked")
        for word in ("Blocked", "Partial", "Completed", "storage_unknown"):
            assert word in text


@pytest.mark.asyncio
async def test_error_keeps_last_data_marked_stale() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A)])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        fake.fail = ClientError(ReasonCode.SERVICE_UNAVAILABLE)
        await until(pilot, lambda: app.model.phase == "unavailable")
        await pilot.pause()
        text = screen_text(app)
        assert "Error: service_unavailable" in text
        assert "Stale" in text_of(app, "#readiness")
        assert "Stale" in text_of(app, "#resources")
        assert fake.closed == 1  # the client is dropped; nothing is cancelled
    assert "cancel" not in fake.names()


@pytest.mark.asyncio
async def test_identity_mismatch_drops_the_client() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A)])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        fake.fail = ClientError(ReasonCode.SERVICE_UNAVAILABLE, identity_mismatch=True,
                                next_action="restart_client")
        await until(pilot, lambda: app.model.phase == "unavailable")
        assert fake.closed == 1


@pytest.mark.asyncio
async def test_pruned_detail_job_keeps_connection_and_says_gone() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A, "completed", "Old one"), make_job(JOB_B)])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await focus_jobs(pilot)
        await pilot.press("enter")
        await until(pilot, lambda: "get_job" in fake.names())
        fake.jobs = [j for j in fake.jobs if j["job_id"] != JOB_A]  # retention pruned it
        await until(pilot, lambda: "no longer exists" in text_of(app, "#notice"))
        await pilot.pause()
        assert f"Job {JOB_A} no longer exists." in text_of(app, "#notice")
        assert app.model.phase == "connected" and app.model.selected_job_id is None
        assert "No job selected" in text_of(app, "#detail")
        assert text_of(app, "#readiness").startswith("Ready")
        lookups = fake.names().count("get_job")
        await pilot.pause(0.3)  # several more polls: no retry of the missing id, no reconnect
        assert fake.names().count("get_job") == lookups
        assert fake.closed == 0 and len(app.factory_threads) == 1  # type: ignore[attr-defined]
        assert fake.names().count("snapshot") > 2
    assert fake.on_loop == []


@pytest.mark.asyncio
async def test_per_request_jobs_error_keeps_client_and_last_list() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A, title="Still here")])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        fake.fail_calls["list_jobs"] = ClientError(ReasonCode.CONFLICT, status=409)
        await until(pilot, lambda: "Not refreshed: jobs (conflict)" in text_of(app, "#notice"))
        snapshots = fake.names().count("snapshot")
        await until(pilot, lambda: fake.names().count("snapshot") > snapshots + 1)
        await pilot.pause()
        assert app.model.phase == "connected" and fake.closed == 0
        assert "Still here" in screen_text(app)
        assert "not refreshed (conflict)" in text_of(app, "#jobs-head")
        assert "Error:" not in text_of(app, "#notice")
        del fake.fail_calls["list_jobs"]
        await until(pilot, lambda: "Not refreshed" not in text_of(app, "#notice"))


@pytest.mark.asyncio
async def test_cache_error_keeps_old_entries_marked_not_refreshed() -> None:
    fake = make_fake_service(cache=[{"entry_id": "e1", "name": "whisper-small", "bytes": 1024}])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await pilot.press("4")
        await until(pilot, lambda: app.model.cache_loaded)
        fake.fail_calls["list_cache"] = ClientError(ReasonCode.RATE_LIMITED, retryable=True,
                                                    status=429)
        await until(pilot, lambda: "not refreshed (rate_limited)" in text_of(app, "#cache-head"))
        await pilot.pause()
        assert "whisper-small" in screen_text(app)
        assert "Not refreshed: cache (rate_limited)" in text_of(app, "#notice")
        assert app.model.phase == "connected" and fake.closed == 0
        del fake.fail_calls["list_cache"]
        await until(pilot, lambda: "not refreshed" not in text_of(app, "#cache-head"))


@pytest.mark.asyncio
async def test_snapshot_freshness_stale_renders_stale() -> None:
    fake = make_fake_service(snapshot=make_snapshot(freshness="stale"))
    app = make_app(fake)
    async with app.run_test(size=(120, 36)) as pilot:
        await ready(pilot)
        assert text_of(app, "#readiness").startswith("Stale")
        assert "Resources - Stale" in text_of(app, "#resources")
        assert "Stale (last known)" in text_of(app, "#resources")


@pytest.mark.asyncio
async def test_snapshot_older_than_three_seconds_renders_stale() -> None:
    fake = make_fake_service()
    app = make_app(fake, clock=lambda: LATER)
    async with app.run_test(size=(120, 36)) as pilot:
        await ready(pilot)
        assert text_of(app, "#readiness").startswith("Stale")
        assert "age 5s" in text_of(app, "#resources")
        assert "Resources - Stale" in text_of(app, "#resources")


@pytest.mark.asyncio
async def test_age_and_stale_redraw_while_a_poll_is_blocked() -> None:
    clock = {"now": NOW}
    fake = make_fake_service(jobs=[make_job(JOB_A)])
    app = make_app(fake, clock=lambda: clock["now"])
    async with app.run_test(size=(120, 36)) as pilot:
        await ready(pilot)
        assert text_of(app, "#readiness").startswith("Ready")
        assert "age 2s" in text_of(app, "#readiness")
        gate = fake.gates["snapshot"] = threading.Event()  # the next snapshot call blocks
        seen = fake.names().count("snapshot")
        await until(pilot, lambda: fake.names().count("snapshot") > seen)
        clock["now"] = LATER  # 5 s after the sample: past the 3 s stale window
        await pilot.pause(0.3)  # several ticks while the call is still blocked
        assert fake.names().count("snapshot") == seen + 1  # still blocked, no new call
        assert app.model.phase == "connected"
        readiness = text_of(app, "#readiness")
        assert readiness.startswith("Stale") and "age 5s" in readiness
        resources = text_of(app, "#resources")
        assert "Resources - Stale" in resources and "age 5s" in resources
        assert "age 5s" in str(app.screen.query_one("#jobs", OptionList)
                               .get_option_at_index(0).prompt)  # the job row itself
        gate.set()
        await until(pilot, lambda: fake.names().count("snapshot") > seen + 1)
    assert fake.on_loop == []


@pytest.mark.asyncio
async def test_answered_503_keeps_client_and_applies_snapshot() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A, title="Still here")])
    app = make_app(fake)
    async with app.run_test(size=(120, 36)) as pilot:
        await ready(pilot)
        fake.snapshot_doc = make_snapshot(model=measure("large-v3"))
        fake.fail_calls["list_jobs"] = ClientError(ReasonCode.SERVICE_UNAVAILABLE,
                                                   retryable=True, next_action="retry_later",
                                                   status=503)
        # A poll already past its snapshot call when the doc changed renders the jobs error
        # with the old snapshot; the next poll applies the new one.  Wait for both renders.
        await until(pilot, lambda: "Not refreshed: jobs (service_unavailable)"
                    in text_of(app, "#notice") and "large-v3" in text_of(app, "#resources")
                    and text_of(app, "#readiness").startswith("Ready"))
        assert app.model.phase == "connected" and fake.closed == 0
        assert "large-v3" in text_of(app, "#resources")  # the snapshot still applied
        assert text_of(app, "#readiness").startswith("Ready")
        assert "Still here" in screen_text(app)
        assert "Error:" not in text_of(app, "#notice")
        del fake.fail_calls["list_jobs"]
        await until(pilot, lambda: "Not refreshed" not in text_of(app, "#notice"))
    assert len(app.factory_threads) == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_transport_unavailable_without_status_drops_the_client() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A)])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        fake.fail_calls["list_jobs"] = ClientError(ReasonCode.SERVICE_UNAVAILABLE,
                                                   retryable=True, next_action="retry_later")
        await until(pilot, lambda: app.model.phase == "unavailable")
        del fake.fail_calls["list_jobs"]
        await pilot.pause()
        assert fake.closed == 1
        assert "Error: service_unavailable" in text_of(app, "#notice")
        assert text_of(app, "#readiness").startswith("Stale")


@pytest.mark.asyncio
async def test_service_stopped_with_status_drops_the_client() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A)])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        fake.fail_calls["list_jobs"] = ClientError(ReasonCode.SERVICE_STOPPED, status=503)
        await until(pilot, lambda: app.model.phase == "unavailable")
        del fake.fail_calls["list_jobs"]
        await pilot.pause()
        assert fake.closed == 1
        assert "Error: service_stopped" in text_of(app, "#notice")
        assert text_of(app, "#readiness").startswith("Stale")


@pytest.mark.asyncio
async def test_delayed_404_does_not_clear_a_newer_selection() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A, "completed", "Old one"),
                                   make_job(JOB_B, title="New one")])
    fake.gates["get_job"] = threading.Event()
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await focus_jobs(pilot)
        await pilot.press("enter")  # detail of A; its lookup blocks
        await until(pilot, lambda: ("get_job", (JOB_A,)) in fake.calls)
        await pilot.press("escape")
        await focus_jobs(pilot)
        await pilot.press("down", "enter")
        assert app.model.view == "detail" and app.model.selected_job_id == JOB_B
        fake.jobs = [j for j in fake.jobs if j["job_id"] != JOB_A]  # A pruned: 404 for it
        fake.gates["get_job"].set()
        await until(pilot, lambda: ("get_job", (JOB_B,)) in fake.calls)
        await until(pilot, lambda: "New one" in text_of(app, "#detail"))
        await pilot.pause()
        assert app.model.selected_job_id == JOB_B
        assert app.model.view == "detail" and app.model.phase == "connected"
        assert "no longer exists" not in text_of(app, "#notice")
        assert JOB_B in text_of(app, "#detail")
        assert fake.closed == 0
    assert fake.on_loop == []


@pytest.mark.asyncio
async def test_unknown_and_stale_render_as_words_never_zero() -> None:
    snap = make_snapshot(host_ram=measure(None, "unknown", "storage_unknown"),
                         cpu=measure(None, "stale", "storage_stale"),
                         worker_ram=measure(None, "error", "service_unavailable"))
    fake = make_fake_service(snapshot=snap)
    app = make_app(fake)
    async with app.run_test(size=(120, 36)) as pilot:
        await ready(pilot)
        lines = {line.split()[0] + " " + line.split()[1]: line
                 for line in text_of(app, "#resources").splitlines() if len(line.split()) > 1}
        assert "Unknown (storage_unknown)" in lines["Host RAM"]
        assert "Stale (storage_stale)" in text_of(app, "#resources")
        assert "Error (service_unavailable)" in lines["Worker RAM"]
        for row in (lines["Host RAM"], lines["Worker RAM"]):
            assert "0 B" not in row and "0%" not in row
        assert "live service" in lines["Host RAM"] and "age 2s" in lines["Host RAM"]


# ------------------------------------------------------------------------------- keys


@pytest.mark.asyncio
async def test_tab_cycles_focus() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A)])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        seen = []
        for _ in range(4):
            await pilot.press("tab")
            seen.append(getattr(app.focused, "id", None))
        assert "jobs" in seen and "tabs" in seen
        await pilot.press("shift+tab")
        assert app.focused is not None


@pytest.mark.asyncio
async def test_enter_opens_detail_and_escape_returns() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A, title="Episode one")])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await focus_jobs(pilot)
        await pilot.press("enter")
        assert app.model.view == "detail"
        await until(pilot, lambda: "get_job" in fake.names())
        await pilot.pause()
        detail = text_of(app, "#detail")
        assert JOB_A in detail and "Episode one" in detail
        labels = screen_text(app)
        assert "Recheck: same job, no new attempt" in labels
        assert "Retry: new attempt of this job" in labels
        assert "Clone: new job you can edit" in labels
        assert "pause" not in labels.lower() and "resume" not in labels.lower()
        await pilot.press("escape")
        assert app.model.view == "overview"


@pytest.mark.asyncio
async def test_n_opens_new_job_but_n_inside_input_is_text() -> None:
    fake = make_fake_service()
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await pilot.press("n")
        assert app.model.view == "new"
        assert isinstance(app.focused, Input)
        await pilot.press("n", "q", "question_mark")
        assert app.screen.query_one("#source", Input).value == "nq?"
        assert app.is_running and app.model.view == "new"


@pytest.mark.asyncio
async def test_submit_from_new_job_uses_worker_and_key() -> None:
    fake = make_fake_service()
    app = make_app(fake, key_factory=lambda: "key-1")
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await pilot.press("n")
        await pilot.press(*"clip.mp4")
        await pilot.press("enter")
        await until(pilot, lambda: "submit" in fake.names())
        _name, args = next(c for c in fake.calls if c[0] == "submit")
        assert args == ({"input": "clip.mp4"}, "key-1")
        await until(pilot, lambda: app.model.view == "detail")
    assert fake.on_loop == []


@pytest.mark.asyncio
async def test_q_detaches_without_cancel_and_keeps_job_active() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A, "transcribing")])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await pilot.press("q")
        await pilot.pause()
    assert app.return_code == 0
    assert "cancel" not in fake.names()
    assert fake.closed == 1
    assert fake.jobs[0]["state"] == "transcribing"
    assert fake.on_loop == []


@pytest.mark.asyncio
async def test_ctrl_c_exits_130_without_cancel() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A)])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert app.return_code == 130
    assert "cancel" not in fake.names()


@pytest.mark.asyncio
async def test_cancel_requires_confirmation_naming_the_job() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A, "transcribing", "Long talk")])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await focus_jobs(pilot)
        await pilot.press("enter")
        await pilot.press("c")
        await pilot.pause()
        dialog = screen_text(app)
        assert "Cancel job Long talk" in dialog and JOB_A in dialog
        await pilot.press("escape")
        await pilot.pause()
        assert "cancel" not in fake.names()
        await pilot.press("c")
        await pilot.pause()
        await pilot.press("enter")  # focus starts on "No"
        await pilot.pause()
        assert "cancel" not in fake.names()
        await pilot.press("c")
        await pilot.pause()
        await pilot.press("tab", "enter")
        await until(pilot, lambda: "cancel" in fake.names())
        assert ("cancel", (JOB_A,)) in fake.calls
    assert fake.on_loop == []


@pytest.mark.asyncio
async def test_cache_remove_requires_confirmation_naming_the_entry() -> None:
    fake = make_fake_service(cache=[{"entry_id": "e1", "name": "whisper-small", "bytes": 1024}])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await pilot.press("4")
        await until(pilot, lambda: app.model.cache_loaded)
        await pilot.pause()
        assert "whisper-small" in screen_text(app)
        await pilot.press("d")
        await pilot.pause()
        assert "Remove cache entry whisper-small" in screen_text(app)
        await pilot.press("escape")
        await pilot.pause()
        assert "remove_cache" not in fake.names()
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("tab", "enter")
        await until(pilot, lambda: "remove_cache" in fake.names())
        assert ("remove_cache", ("e1",)) in fake.calls
        assert "cancel" not in fake.names()


@pytest.mark.asyncio
async def test_retry_and_recheck_are_distinct_calls() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A, "failed", "Broken"),
                                   make_job(JOB_B, "blocked", "Waiting")])
    app = make_app(fake, key_factory=lambda: "k")
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await focus_jobs(pilot)
        await pilot.press("enter")
        await pilot.press("t")
        await until(pilot, lambda: "retry" in fake.names())
        assert "new attempt" in text_of(app, "#notice")
        await pilot.press("escape")
        await focus_jobs(pilot)
        await pilot.press("down", "enter")
        await pilot.press("r")
        await until(pilot, lambda: "recheck" in fake.names())
        assert "no new attempt" in text_of(app, "#notice")
        assert ("recheck", (JOB_B,)) in fake.calls and ("retry", (JOB_A, "k")) in fake.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(("toggles", "expected"), [
    ((), (True, False)),                            # defaults: transcript only
    (("out-video",), (True, True)),                 # both outputs
    (("out-transcript", "out-video"), (False, True)),  # video only
])
async def test_submit_passes_both_output_checkboxes(toggles: tuple[str, ...],
                                                    expected: tuple[bool, bool]) -> None:
    seen: list[tuple[str, bool, bool]] = []

    def builder(text: str, transcript: bool, video: bool) -> dict:
        seen.append((text, transcript, video))
        return {"input": text}

    fake = make_fake_service()
    app = make_app(fake, request_builder=builder)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await pilot.press("n")
        await pilot.press(*"clip.mp4")
        for checkbox in ("out-transcript", "out-video"):
            await pilot.press("tab")
            assert getattr(app.focused, "id", None) == checkbox
            if checkbox in toggles:
                await pilot.press("space")
        await pilot.press("tab")
        assert getattr(app.focused, "id", None) == "btn-submit"
        await pilot.press("enter")
        await until(pilot, lambda: "submit" in fake.names())
    assert seen == [("clip.mp4", *expected)]


def test_default_request_builder_passes_outputs_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_build(args: Any, environ: Any) -> dict:
        seen["args"] = args
        return {"input": args.input}

    monkeypatch.setattr("podmuncher.cli.build_request", fake_build)
    assert tui._default_request_builder("clip.mp4", False, True) == {"input": "clip.mp4"}
    args = seen["args"]
    assert args.outputs == {"transcript": False, "video": True}
    assert args.video is True and args.no_transcript is True
    tui._default_request_builder("clip.mp4", True, False)
    assert seen["args"].outputs == {"transcript": True, "video": False}


async def open_blocked_detail(pilot: Any, fake: FakeService) -> Button:
    await ready(pilot)
    await focus_jobs(pilot)
    await pilot.press("enter")
    await until(pilot, lambda: "get_job" in fake.names())
    await pilot.pause()
    return pilot.app.screen.query_one("#btn-retry", Button)


@pytest.mark.asyncio
async def test_blocked_job_retry_enabled_after_ready_recheck() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A, "blocked", "Waiting", reason="storage_unknown")])
    fake.ready_to_retry = True
    app = make_app(fake, key_factory=lambda: "k")
    async with app.run_test(size=(100, 30)) as pilot:
        retry = await open_blocked_detail(pilot, fake)
        assert retry.disabled
        assert "Retry: not yet; run Recheck" in text_of(app, "#detail")
        await pilot.press("r")
        await until(pilot, lambda: "Ready to retry: Waiting" in text_of(app, "#notice"))
        await pilot.pause()
        assert "no new attempt" in text_of(app, "#notice")
        assert "retry" not in fake.names()  # recheck started no attempt
        assert not retry.disabled
        assert "Retry: Ready to retry" in text_of(app, "#detail")
        await pilot.click("#btn-retry")
        await until(pilot, lambda: "retry" in fake.names())
        await pilot.pause()
        assert fake.names().count("retry") == 1
        assert ("retry", (JOB_A, "k")) in fake.calls
        assert ("recheck", (JOB_A,)) in fake.calls
    assert fake.on_loop == []


@pytest.mark.asyncio
async def test_blocked_job_retry_stays_disabled_when_recheck_not_ready() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A, "blocked", "Waiting", reason="storage_unknown")])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        retry = await open_blocked_detail(pilot, fake)
        await pilot.press("r")
        await until(pilot, lambda: "Not ready to retry (storage_insufficient)"
                    in text_of(app, "#notice"))
        await pilot.pause()
        assert "no new attempt" in text_of(app, "#notice")
        assert retry.disabled
        await pilot.press("t")
        await pilot.pause(0.2)
        assert "Retry is not available for Waiting: blocked" in text_of(app, "#notice")
        await pilot.click("#btn-retry")
        await pilot.pause(0.2)
        assert "retry" not in fake.names()
        assert fake.names().count("recheck") == 1


@pytest.mark.asyncio
async def test_paging_reaches_job_150_and_keys_are_text_inside_inputs() -> None:
    jobs = [make_job(f"{index:08d}-0000-4000-8000-000000000000", "completed", f"Episode {index}")
            for index in range(1, 151)]
    fake = make_fake_service(jobs=jobs)
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        assert "Episode 150" not in screen_text(app)
        assert "] older" in text_of(app, "#jobs-head")
        await pilot.press("right_square_bracket")
        await until(pilot, lambda: "page 2" in text_of(app, "#jobs-head"))
        await pilot.press("right_square_bracket")
        await until(pilot, lambda: "Episode 150" in screen_text(app))
        head = text_of(app, "#jobs-head")
        assert "page 3" in head and "[ newer" in head and "] older" not in head
        assert ("list_jobs", (tui.PAGE_SIZE, "100")) in fake.calls
        await pilot.press("right_square_bracket")
        await pilot.pause()
        assert "No older entries" in text_of(app, "#notice")
        await pilot.press("left_square_bracket")
        await until(pilot, lambda: "page 2" in text_of(app, "#jobs-head"))
        await pilot.press("n")
        await pilot.press("right_square_bracket", "left_square_bracket")
        assert app.screen.query_one("#source", Input).value == "][" and app.model.view == "new"
    assert fake.on_loop == []


@pytest.mark.asyncio
async def test_setup_screen_names_the_setup_flag() -> None:
    app = make_app(make_fake_service())
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await pilot.press("5")
        await until(pilot, lambda: app.model.doctor is not None)
        await pilot.pause()
        body = text_of(app, "#setup-body")
        assert "`podmuncher --setup`" in body and "`podmuncher setup`" not in body


@pytest.mark.parametrize(("value", "expected"), [("C:/state/dir", "C:/state/dir"), ("", None)])
def test_main_passes_state_dir_to_service(monkeypatch: pytest.MonkeyPatch, value: str,
                                          expected: str | None) -> None:
    from podmuncher import service

    seen: list[tuple[str, Any]] = []

    class FakeClient:
        def close(self) -> None:
            seen.append(("close", None))

    answers = iter([None, FakeClient()])

    def connect(state_dir: Any = None, **_: Any) -> Any:
        seen.append(("connect", state_dir))
        return next(answers)

    def start_service(state_dir: Any = None, **_: Any) -> dict:
        seen.append(("start_service", state_dir))
        return {}

    def read_status(state_dir: Any = None, **_: Any) -> dict:
        seen.append(("read_status", state_dir))
        return {"state": "stopped"}

    class FakeApp:
        def __init__(self, factory: Callable[[], Any], *, status_reader: Any = None,
                     **_: Any) -> None:
            self._factory, self._reader = factory, status_reader
            self._client: Any = None
            self.return_code = 0

        def run(self) -> None:
            self._client = self._factory()
            self._reader()

    monkeypatch.setattr(service, "connect", connect)
    monkeypatch.setattr(service, "start_service", start_service)
    monkeypatch.setattr(service, "read_status", read_status)
    monkeypatch.setattr(tui, "PodmuncherApp", FakeApp)
    monkeypatch.setenv("PODMUNCHER_STATE_DIR", value)
    assert tui.main() == 0
    assert seen == [("connect", expected), ("start_service", expected), ("connect", expected),
                    ("read_status", expected), ("close", None)]


@pytest.mark.asyncio
async def test_help_opens_and_closes() -> None:
    app = make_app(make_fake_service())
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await pilot.press("question_mark")
        await pilot.pause()
        assert "Ctrl+C" in screen_text(app)
        await pilot.press("escape")
        await pilot.pause()
        assert "Ctrl+C" not in screen_text(app)


# ---------------------------------------------------------------------------- content


@pytest.mark.asyncio
async def test_detail_lists_artifacts_independently_as_they_appear() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A, "transcribing")])
    app = make_app(fake)
    async with app.run_test(size=(100, 30)) as pilot:
        await ready(pilot)
        await focus_jobs(pilot)
        await pilot.press("enter")
        await until(pilot, lambda: "get_job" in fake.names())
        await pilot.pause()
        assert "Artifacts (0)" in text_of(app, "#detail")
        fake.jobs[0] = make_job(JOB_A, "exporting", artifacts=[
            {"artifact_id": "a1", "slot": "transcript", "format": "srt", "bytes": 2048,
             "available": True, "basename": "x.srt"}])
        await until(pilot, lambda: "Artifacts (1)" in text_of(app, "#detail"))
        assert "transcript (srt, 2.0 KB) - Available" in text_of(app, "#detail")
        fake.jobs[0] = make_job(JOB_A, "exporting", artifacts=[
            {"artifact_id": "a1", "slot": "transcript", "format": "srt", "bytes": 2048,
             "available": True, "basename": "x.srt"},
            {"artifact_id": "a2", "slot": "video", "format": "mp4", "bytes": None,
             "available": False, "basename": None}])
        await until(pilot, lambda: "Artifacts (2)" in text_of(app, "#detail"))
        detail = text_of(app, "#detail")
        assert "video - Pending validation" in detail
        assert "transcript (srt, 2.0 KB) - Available" in detail
    assert fake.on_loop == []


@pytest.mark.asyncio
async def test_hostile_title_renders_escaped() -> None:
    hostile = "\x1b[31mEvil[bold]X\x07"
    fake = make_fake_service(jobs=[make_job(JOB_A, title=hostile)])
    app = make_app(fake)
    async with app.run_test(size=(120, 36)) as pilot:
        await ready(pilot)
        rows = screen_text(app)
        assert "\x1b" not in rows and "\x07" not in rows
        assert "\\x1b" in rows and "\\[bold\\]" in rows
        await focus_jobs(pilot)
        await pilot.press("enter")
        await pilot.pause()
        detail = text_of(app, "#detail")
        assert "\x1b" not in detail and "\\[bold\\]X" in detail


def test_domain_negatives_rejected_by_view_model_parsing() -> None:
    contract = json.loads((ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    negatives = list(contract["negative_examples"]) + [None, 7, "COMPLETED", "running"]
    for value in negatives:
        assert tui.parse_job_state(value) is None
        assert tui.state_word(value) == "Unknown"
        assert tui.parse_reason(value) is None
    for value in contract["job_state"] if isinstance(contract["job_state"], list) else ():
        assert tui.parse_job_state(value) is not None
    assert tui.reason_word("__invalid__") == "unrecognised"
    assert tui.reason_word("ok") == ""


@pytest.mark.asyncio
async def test_out_of_domain_job_state_renders_unknown() -> None:
    fake = make_fake_service(jobs=[make_job(JOB_A, "__invalid__", "Odd", reason="__invalid__")])
    app = make_app(fake)
    async with app.run_test(size=(120, 36)) as pilot:
        await ready(pilot)
        rows = screen_text(app)
        assert "Unknown" in rows and "unrecognised" in rows


def test_fit_truncates_by_display_width() -> None:
    wide = "宽" * 20  # each glyph is two cells
    out = tui.fit(wide, 11)
    from rich.cells import cell_len
    assert cell_len(out) <= 11 and out.endswith("…")


def test_textual_not_imported_by_cli_help() -> None:
    import subprocess
    import sys
    code = ("import sys, podmuncher.cli as c\n"
            "try:\n    c.main(['--help'])\nexcept SystemExit:\n    pass\n"
            "print('textual' in sys.modules)")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=ROOT, env=env, timeout=60, check=False)
    assert out.stdout.strip().splitlines()[-1] == "False"
