"""Setup state machine, noninteractive driver, audited save, and read-only doctor."""

import json
import subprocess
from pathlib import Path

import pytest

from podmuncher import setup as su
from podmuncher.domains import Capability, DomainValidationError, ReasonCode
from podmuncher.setup import (
    LocalSetupActions,
    SetupError,
    SetupMachine,
    SetupService,
    StepResult,
    UnboundProvisioning,
)
from podmuncher.setup import (
    SetupState as S,
)

ROOT = Path(__file__).resolve().parents[1]
OK = StepResult(True)
PATHS = {"data": "d", "output": "o", "cache": "c", "scratch": "s"}
FULL = ("media", "asr_cpu")
FORWARD_FULL = [S.PATHS, S.HOST_CHECK, S.DOCKER_CHECK, S.BACKING, S.PLAN, S.CONFIRM, S.PULL, S.HARDWARE,
                S.MODEL_SELECT, S.MODEL_CAPACITY, S.MODEL_ACQUIRE, S.SMOKE, S.SAVE_CONFIG, S.DONE]


OPTIONS = [
    {"id": "small", "peak_ram_bytes": 1_477_000_000, "basis": "published",
     "source": "https://github.com/SYSTRAN/faster-whisper#benchmark", "fits": True},
    {"id": "base", "peak_ram_bytes": 904_921_088, "basis": "measured", "source": "https://example.test/ours",
     "fits": True},
    {"id": "large-v3", "peak_ram_bytes": None, "basis": "estimated"},  # optional keys missing
]
CHOOSE = StepResult(True, data={"choice": "small", "options": OPTIONS})


class FakeProvision:
    def __init__(self, owner):
        self.owner = owner

    def _step(self, name, *args):
        self.owner.calls.append(name if not args else (name, *args))
        result = self.owner.script.get(name, OK)
        return result

    def pull(self, capability):
        return self._step("pull", str(capability))

    def probe_hardware(self):
        return self._step("probe_hardware")

    def choose_model(self, hardware):
        self.owner.calls.append("choose_model")
        return self.owner.script.get("choose_model", CHOOSE)

    def model_capacity(self, choice):
        return self._step("model_capacity", choice)

    def acquire_model(self, choice):
        return self._step("acquire_model", choice)

    def smoke(self, choice):
        return self._step("smoke", choice)

    def bind_paths(self, paths):
        # Recorded apart from ``calls``: binding is not a provisioning step.
        self.owner.bound.append((dict(paths), len(self.owner.calls)))

    def release_cache(self):
        self.owner.released.append(len(self.owner.calls))


class _LegacyProvision:
    """A port from before ``bind_paths`` existed: the machine must still drive it."""

    def __init__(self, inner):
        self._inner = inner

    def pull(self, capability):
        return self._inner.pull(capability)

    def probe_hardware(self):
        return self._inner.probe_hardware()

    def choose_model(self, hardware):
        return self._inner.choose_model(hardware)

    def model_capacity(self, choice):
        return self._inner.model_capacity(choice)

    def acquire_model(self, choice):
        return self._inner.acquire_model(choice)

    def smoke(self, choice):
        return self._inner.smoke(choice)


class FakeSetupActions:
    def __init__(self, **script):
        self.script, self.calls, self.saved, self.cached = script, [], [], None
        self.bound = []  # (paths, number of calls made when bound)
        self.released = []  # number of calls made when the port's cache was released
        self.provision = FakeProvision(self)

    def _r(self, name, *args):
        self.calls.append(name if not args else (name, *args))
        result = self.script.get(name, OK)
        if isinstance(result, Exception):
            raise result
        return result

    def check_paths(self, paths):
        self.calls.append("check_paths")
        return self.script.get("check_paths", StepResult(True, data={"paths": dict(paths)}))

    def host_capacity(self, paths):
        return self._r("host_capacity")

    def docker_status(self):
        return self._r("docker_status")

    def backing(self, manual=None):
        self.calls.append(("backing", manual))
        outcomes = self.script.get("backing", OK)
        if isinstance(outcomes, list):
            return outcomes.pop(0)
        return outcomes

    def plan(self, capabilities, paths):
        return self._r("plan")

    def cached_snapshot(self, choice):
        return self.cached

    def save_config(self, config):
        self.calls.append("save_config")
        self.saved.append(config)
        return self.script.get("save_config", OK)

    def count(self, name):
        return sum(1 for c in self.calls if c == name or (isinstance(c, tuple) and c[0] == name))


def run_to(machine, target):
    while machine.state is not target:
        if machine.state is S.PATHS:
            machine.next(PATHS)
        elif machine.state is S.CONFIRM:
            machine.next(True)
        else:
            machine.next()
        assert machine.state is not S.BLOCKED, machine.reason_code
    return machine


def test_forward_transitions_full():
    fake = FakeSetupActions()
    machine = SetupMachine(fake, capabilities=FULL)
    seen = [machine.state]
    while machine.state is not S.DONE:
        machine.next(PATHS if machine.state is S.PATHS else True if machine.state is S.CONFIRM else None)
        seen.append(machine.state)
    assert seen == FORWARD_FULL
    assert fake.count("save_config") == 1 and fake.calls[-1] == "save_config"
    assert fake.saved[0]["outputs"] == {"transcript": True, "video": False}
    assert fake.saved[0]["selection"] == {"device": "cpu", "model": "auto"}
    assert machine.exit_code == 0


@pytest.mark.parametrize("target", FORWARD_FULL[1:-1])
def test_back_from_each_state(target):
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), target)
    previous = FORWARD_FULL[FORWARD_FULL.index(target) - 1]
    assert machine.back() is True
    assert machine.state is previous
    if FORWARD_FULL.index(target) > FORWARD_FULL.index(S.CONFIRM):
        pulls = fake.count("pull")
        machine.next(True if machine.state is S.CONFIRM else None)
        assert fake.count("pull") == pulls  # persisted provisioning never re-runs


def test_back_from_first_state_is_noop():
    machine = SetupMachine(FakeSetupActions(), capabilities=FULL)
    assert machine.back() is False and machine.state is S.PATHS


def test_back_before_confirm_revokes_confirmation():
    machine = run_to(SetupMachine(FakeSetupActions(), capabilities=FULL), S.PULL)
    assert machine.confirmed
    machine.back()
    machine.back()
    assert machine.state is S.PLAN and not machine.confirmed


def test_recheck_reruns_only_current_check():
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.DOCKER_CHECK)
    before = list(fake.calls)
    machine.recheck()
    assert fake.calls[len(before):] == ["docker_status"]
    assert machine.state is S.DOCKER_CHECK


def test_recheck_does_not_reprovision():
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.PULL)
    machine.recheck()
    assert fake.count("pull") == 2  # media + asr_cpu, once


def test_blocked_backing_never_pulls_and_manual_path_recovers():
    fake = FakeSetupActions(backing=[StepResult(False, ReasonCode.BACKING_UNRESOLVED, next_action="enter path"), OK])
    machine = SetupMachine(fake, capabilities=FULL)
    machine.next(PATHS)
    machine.next()
    machine.next()
    assert machine.state is S.BLOCKED and machine.blocked_from is S.BACKING
    assert machine.exit_code == 3 and fake.count("pull") == 0
    machine.next("/mnt/existing")
    assert machine.state is S.BACKING
    assert ("backing", "/mnt/existing") in fake.calls
    assert machine.paths["docker_backing"] == "/mnt/existing"


def test_paths_docker_backing_is_checked_and_recovery_override_wins():
    fake = FakeSetupActions()
    machine = SetupMachine(fake, capabilities=FULL)
    machine.next({**PATHS, "docker_backing": "/custom/docker"})
    machine.next()
    machine.next()
    assert machine.state is S.BACKING and fake.calls[-1] == ("backing", "/custom/docker")
    assert machine.paths["docker_backing"] == "/custom/docker"
    blocked = FakeSetupActions(backing=[StepResult(False, ReasonCode.BACKING_UNRESOLVED), OK])
    machine = SetupMachine(blocked, capabilities=FULL)
    machine.next({**PATHS, "docker_backing": "/custom/docker"})
    machine.next()
    machine.next()
    assert machine.state is S.BLOCKED and blocked.calls[-1] == ("backing", "/custom/docker")
    machine.next("/override")
    assert machine.state is S.BACKING and blocked.calls[-1] == ("backing", "/override")
    assert machine.paths["docker_backing"] == "/override"


def test_new_paths_after_recovery_override_retire_the_stale_manual_backing():
    fake = FakeSetupActions(backing=[StepResult(False, ReasonCode.BACKING_UNRESOLVED), OK, OK, OK])
    machine = SetupMachine(fake, capabilities=["media"])
    machine.next(PATHS)
    machine.next()
    machine.next()
    assert machine.state is S.BLOCKED and machine.blocked_from is S.BACKING
    machine.next("/stale/override")
    assert machine.state is S.BACKING and fake.calls[-1] == ("backing", "/stale/override")
    # Back to PATHS; paths without docker_backing fall through to auto discovery, not
    # the old override.
    while machine.state is not S.PATHS:
        assert machine.back()
    machine.next(PATHS)
    machine.next()
    machine.next()
    assert machine.state is S.BACKING and fake.calls[-1] == ("backing", None)
    assert "docker_backing" not in machine.paths
    # Back again; new paths carrying their own docker_backing are what gets checked and saved.
    while machine.state is not S.PATHS:
        assert machine.back()
    machine.next({**PATHS, "docker_backing": "/new/docker"})
    machine.next()
    machine.next()
    assert machine.state is S.BACKING and fake.calls[-1] == ("backing", "/new/docker")
    assert machine.paths["docker_backing"] == "/new/docker"
    run_to(machine, S.DONE)
    assert fake.saved[-1]["paths"]["docker_backing"] == "/new/docker"
    fake = FakeSetupActions(backing=[StepResult(False, ReasonCode.BACKING_UNRESOLVED), OK])
    machine = SetupMachine(fake, capabilities=FULL)
    for _ in range(3):
        machine.next(PATHS if machine.state is S.PATHS else None)
    assert machine.state is S.BLOCKED
    machine.recheck()
    assert machine.state is S.BACKING and fake.count("pull") == 0
    machine = SetupMachine(FakeSetupActions(backing=StepResult(False, ReasonCode.BACKING_UNRESOLVED)), capabilities=FULL)
    for _ in range(3):
        machine.next(PATHS if machine.state is S.PATHS else None)
    assert machine.back() and machine.state is S.DOCKER_CHECK


def test_video_only_skips_asr_steps():
    fake = FakeSetupActions()
    machine = SetupMachine(fake, capabilities=["media"])
    assert machine.video_only
    seen = [machine.state]
    while machine.state is not S.DONE:
        machine.next(PATHS if machine.state is S.PATHS else True if machine.state is S.CONFIRM else None)
        seen.append(machine.state)
    assert seen == [S.PATHS, S.HOST_CHECK, S.DOCKER_CHECK, S.BACKING, S.PLAN, S.CONFIRM, S.PULL, S.SAVE_CONFIG, S.DONE]
    assert fake.calls.count(("pull", "media")) == 1
    assert not {"probe_hardware", "choose_model"} & set(fake.calls)
    assert not any(isinstance(c, tuple) and c[0] in ("pull",) and c[1] != "media" for c in fake.calls)
    assert fake.saved[0]["outputs"] == {"transcript": False, "video": True}
    assert fake.saved[0]["selection"]["model"] == "auto"


@pytest.mark.parametrize("target", FORWARD_FULL[:-1])
def test_cancel_from_every_state_never_saves(target):
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), target)
    assert machine.cancel() is S.CANCELLED
    machine.next()
    assert fake.count("save_config") == 0 and machine.state is S.CANCELLED


def test_cancel_from_blocked_never_saves():
    fake = FakeSetupActions(backing=StepResult(False, ReasonCode.BACKING_UNRESOLVED))
    machine = SetupMachine(fake, capabilities=FULL)
    for _ in range(3):
        machine.next(PATHS if machine.state is S.PATHS else None)
    machine.cancel()
    assert machine.state is S.CANCELLED and fake.count("save_config") == 0


def test_declined_confirm_cancels_without_provisioning():
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.CONFIRM)
    machine.next(False)
    assert machine.state is S.CANCELLED
    assert fake.count("pull") == 0 and fake.count("save_config") == 0


def test_confirm_requires_explicit_boolean():
    machine = run_to(SetupMachine(FakeSetupActions(), capabilities=FULL), S.CONFIRM)
    for bad in (None, "yes", 1):
        with pytest.raises(SetupError):
            machine.next(bad)
    assert machine.state is S.CONFIRM


def test_no_acquisition_before_confirm():
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.CONFIRM)
    for state in (S.PULL, S.HARDWARE, S.MODEL_ACQUIRE, S.SMOKE, S.SAVE_CONFIG):
        with pytest.raises(SetupError) as error:
            machine._enter(state)
        assert error.value.reason_code is ReasonCode.CONFLICT
    for name in ("pull", "probe_hardware", "acquire_model", "smoke", "save_config"):
        assert fake.count(name) == 0
    assert not machine.confirmed


@pytest.mark.parametrize("failing", ["pull", "acquire_model", "smoke"])
def test_config_written_only_after_success(failing):
    fake = FakeSetupActions(**{failing: StepResult(False, ReasonCode.UPSTREAM_UNAVAILABLE)})
    machine = SetupMachine(fake, capabilities=FULL)
    while machine.state not in (S.DONE, S.BLOCKED):
        machine.next(PATHS if machine.state is S.PATHS else True if machine.state is S.CONFIRM else None)
    assert machine.state is S.BLOCKED and machine.exit_code == 4
    assert fake.count("save_config") == 0


def test_config_save_guard_requires_prior_steps():
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.PULL)
    machine.results.pop(S.SMOKE, None)
    machine.state = S.SAVE_CONFIG
    machine.next()
    assert machine.state is S.BLOCKED and fake.count("save_config") == 0


def _back_to_model_select(machine):
    while machine.state is not S.MODEL_SELECT:
        assert machine.back()


def test_changed_model_after_back_is_acquired_and_smoked_before_save():
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.SAVE_CONFIG)
    _back_to_model_select(machine)
    mark = len(fake.calls)
    machine.next("base")
    run_to(machine, S.DONE)
    assert fake.calls[mark:] == [("model_capacity", "base"), ("acquire_model", "base"), ("smoke", "base"),
                                 "save_config"]
    assert fake.saved[-1]["selection"]["model"] == "base"


def test_unchanged_model_after_back_reuses_provisioning():
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.SAVE_CONFIG)
    _back_to_model_select(machine)
    mark = len(fake.calls)
    machine.next("small")
    run_to(machine, S.DONE)
    assert fake.calls[mark:] == [("model_capacity", "small"), "save_config"]
    assert fake.count("acquire_model") == 1 and fake.count("smoke") == 1


def test_auto_flow_saves_auto_but_provisions_suggestion():
    fake = FakeSetupActions()
    run_to(SetupMachine(fake, capabilities=FULL), S.DONE)
    assert ("acquire_model", "small") in fake.calls and ("smoke", "small") in fake.calls
    assert fake.saved[-1]["selection"]["model"] == "auto"


def test_explicit_pick_saves_that_id_even_when_it_does_not_fit():
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.MODEL_SELECT)
    machine.next("large-v3")  # fit unknown: the operator decides
    run_to(machine, S.DONE)
    assert ("acquire_model", "large-v3") in fake.calls and ("smoke", "large-v3") in fake.calls
    assert fake.saved[-1]["selection"]["model"] == "large-v3"


def test_explicit_pick_of_the_suggestion_saves_the_id():
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.MODEL_SELECT)
    machine.next("small")
    run_to(machine, S.DONE)
    assert fake.saved[-1]["selection"]["model"] == "small"


def test_auto_to_explicit_invalidates_and_back_to_auto_restores():
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.SAVE_CONFIG)
    _back_to_model_select(machine)
    machine.next("base")
    assert S.MODEL_ACQUIRE not in machine.results and S.SMOKE not in machine.results
    run_to(machine, S.SAVE_CONFIG)
    _back_to_model_select(machine)
    mark = len(fake.calls)
    machine.next()  # Enter: back to auto
    run_to(machine, S.DONE)
    assert fake.calls[mark:] == [("model_capacity", "small"), ("acquire_model", "small"), ("smoke", "small"),
                                 "save_config"]
    assert fake.saved[-1]["selection"]["model"] == "auto"


@pytest.mark.parametrize("bad", ["tiny", "", 3, "SMALL"])
def test_unlisted_model_id_is_rejected_without_moving(bad):
    machine = run_to(SetupMachine(FakeSetupActions(), capabilities=FULL), S.MODEL_SELECT)
    with pytest.raises(SetupError) as error:
        machine.next(bad)
    assert error.value.reason_code is ReasonCode.INVALID_INPUT and machine.state is S.MODEL_SELECT


def test_legacy_string_options_still_accepted():
    fake = FakeSetupActions(choose_model=StepResult(True, data={"choice": "small", "options": ["small", "base"]}))
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.MODEL_SELECT)
    machine.next("base")
    run_to(machine, S.DONE)
    assert fake.saved[-1]["selection"]["model"] == "base"


def test_nothing_fits_auto_blocks_but_explicit_pick_proceeds():
    fake = FakeSetupActions(choose_model=StepResult(True, data={"choice": None, "options": OPTIONS}))
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.MODEL_SELECT)
    machine.next()
    assert machine.state is S.BLOCKED and machine.blocked_from is S.MODEL_SELECT and machine.exit_code == 3
    assert fake.count("model_capacity") == 0
    machine.recheck()
    assert machine.state is S.MODEL_SELECT
    machine.next("base")
    run_to(machine, S.DONE)
    assert fake.saved[-1]["selection"]["model"] == "base"


def test_save_refuses_results_for_another_model():
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.SAVE_CONFIG)
    machine._choice = "base"  # results on record were produced for "small"
    machine.next()
    assert machine.state is S.BLOCKED and machine.reason_code is ReasonCode.CONFLICT
    assert fake.count("save_config") == 0


def test_adapter_exception_blocks_with_closed_reason():
    fake = FakeSetupActions(docker_status=RuntimeError("secret detail"))
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.HOST_CHECK)
    machine.next()
    assert machine.state is S.BLOCKED and machine.reason_code is ReasonCode.SERVICE_UNAVAILABLE
    assert "secret" not in repr(machine.next_action)


def test_cached_snapshot_reuse_skips_download():
    fake = FakeSetupActions()
    fake.cached = "snap-1"
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.MODEL_ACQUIRE)
    assert fake.count("acquire_model") == 0
    assert machine.results[S.MODEL_ACQUIRE].data == {"reused": True, "snapshot": "snap-1"}


def test_uncached_model_is_acquired_once():
    fake = FakeSetupActions()
    run_to(SetupMachine(fake, capabilities=FULL), S.SMOKE)
    assert fake.count("acquire_model") == 1


def test_idempotent_apply_does_not_reprovision():
    fake = FakeSetupActions()
    service = SetupService(fake)
    first = service.apply("key-1", capabilities=FULL, paths=PATHS, accept_plan=True)
    assert first.applied and first.exit_code == 0
    calls = list(fake.calls)
    second = service.apply("key-1", capabilities=FULL, paths=PATHS, accept_plan=True)
    assert second == first and fake.calls == calls
    with pytest.raises(SetupError):
        service.apply("key-1", capabilities=FULL, paths={"data": "other"}, accept_plan=True)


def test_failed_apply_is_not_cached():
    fake = FakeSetupActions(smoke=StepResult(False, ReasonCode.MODEL_LOAD_FAILED))
    service = SetupService(fake)
    assert service.apply("k", capabilities=FULL, paths=PATHS, accept_plan=True).exit_code == 4
    fake.script.pop("smoke")
    assert service.apply("k", capabilities=FULL, paths=PATHS, accept_plan=True).applied


def test_blocked_model_select_takes_the_typed_pick_without_relisting():
    fake = FakeSetupActions(choose_model=StepResult(True, data={"choice": None, "options": OPTIONS}))
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.MODEL_SELECT)
    machine.next()
    assert machine.state is S.BLOCKED and machine.blocked_from is S.MODEL_SELECT
    listed = fake.calls.count("choose_model")
    machine.next("base")  # the pick answers the block; the offered list is not re-run
    assert machine.state is S.MODEL_CAPACITY and fake.calls.count("choose_model") == listed
    assert machine.reason_code is ReasonCode.OK and machine.next_action == ""
    run_to(machine, S.DONE)
    assert fake.saved[-1]["selection"]["model"] == "base"
    # An unlisted id while blocked is rejected the same way as at the prompt.
    machine = run_to(SetupMachine(FakeSetupActions(choose_model=StepResult(
        True, data={"choice": None, "options": OPTIONS})), capabilities=FULL), S.MODEL_SELECT)
    machine.next()
    with pytest.raises(SetupError):
        machine.next("tiny")
    assert machine.state is S.MODEL_SELECT


def test_recovered_block_reports_ok_at_completion():
    fake = FakeSetupActions(backing=[StepResult(False, ReasonCode.BACKING_UNRESOLVED, next_action="enter path"), OK])
    result = su.drive(SetupMachine(fake, capabilities=["media"]), paths=PATHS, accept_plan=True, manual_backing="/x")
    assert result.exit_code == 0 and result.state == "done" and result.applied
    assert result.reason_code == str(ReasonCode.OK) and result.next_action == ""


class _LosingAudit:
    """Audit sink that drops ``setup_done`` while ``losing`` is set."""

    def __init__(self):
        self.losing, self.events = True, []

    def __call__(self, action, target, reason):
        if action == "setup_done" and self.losing:
            raise OSError("disk")
        self.events.append(action)


def test_lost_completion_audit_is_surfaced_and_retry_never_reprovisions():
    fake, audit = FakeSetupActions(), _LosingAudit()
    service = SetupService(fake, audit=audit)
    first = service.apply("k", capabilities=FULL, paths=PATHS, accept_plan=True)
    assert first.exit_code == 4 and first.state == str(S.BLOCKED) and first.reason_code == str(ReasonCode.AUDIT_LOSS)
    assert first.applied and fake.count("save_config") == 1 and first.next_action
    calls = list(fake.calls)
    audit.losing = False
    second = service.apply("k", capabilities=FULL, paths=PATHS, accept_plan=True)
    assert second == first and fake.calls == calls  # committed: nothing is provisioned or saved again
    # On the machine, recheck retries only the record.
    fake, audit = FakeSetupActions(), _LosingAudit()
    machine = SetupMachine(fake, capabilities=["media"], audit=audit)
    assert su.drive(machine, paths=PATHS, accept_plan=True).reason_code == str(ReasonCode.AUDIT_LOSS)
    calls, audit.losing = list(fake.calls), False
    machine.recheck()
    assert machine.state is S.DONE and machine.exit_code == 0 and machine.reason_code is ReasonCode.OK
    assert fake.calls == calls and audit.events == ["setup_apply", "setup_done"]


def test_apply_with_audit_records_ids_only():
    events = []
    service = SetupService(FakeSetupActions(), audit=lambda *a: events.append(a))
    service.apply("k", capabilities=["media"], paths=PATHS, accept_plan=True)
    assert [e[0] for e in events] == ["setup_apply", "setup_done"]
    assert all(isinstance(e[2], ReasonCode) for e in events)


def answers_file(tmp_path, **overrides):
    data = {"paths": {k: str(tmp_path / v) for k, v in PATHS.items()}, "capabilities": ["media"], "accept_plan": True}
    data.update(overrides)
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_noninteractive_answers_file(tmp_path):
    fake = FakeSetupActions()
    path = answers_file(tmp_path)
    assert su.main(actions=fake, environ={su.ANSWERS_ENV: path}, interactive=False) == 0
    assert fake.count("save_config") == 1


@pytest.mark.parametrize(("model", "saved"), [(None, "auto"), ("auto", "auto"), ("base", "base")])
def test_noninteractive_model_answer(tmp_path, model, saved):
    fake = FakeSetupActions()
    extra = {"model": model} if model is not None else {}
    path = answers_file(tmp_path, capabilities=list(FULL), **extra)
    assert su.main(actions=fake, environ={su.ANSWERS_ENV: path}) == 0
    assert fake.saved[-1]["selection"]["model"] == saved
    expected = "small" if saved == "auto" else saved
    assert ("acquire_model", expected) in fake.calls and ("smoke", expected) in fake.calls


def test_noninteractive_unlisted_model_is_invalid_and_saves_nothing(tmp_path):
    fake = FakeSetupActions()
    path = answers_file(tmp_path, capabilities=list(FULL), model="tiny")
    assert su.main(actions=fake, environ={su.ANSWERS_ENV: path}) == 2
    assert fake.count("save_config") == 0 and fake.count("acquire_model") == 0


def test_noninteractive_decline_and_blocked(tmp_path):
    fake = FakeSetupActions()
    assert su.main(actions=fake, environ={su.ANSWERS_ENV: answers_file(tmp_path, accept_plan=False)}) == 3
    assert fake.count("pull") == 0 and fake.count("save_config") == 0
    blocked = FakeSetupActions(backing=StepResult(False, ReasonCode.BACKING_UNRESOLVED))
    assert su.main(actions=blocked, environ={su.ANSWERS_ENV: answers_file(tmp_path)}) == 3
    assert blocked.count("pull") == 0
    recovered = FakeSetupActions(backing=[StepResult(False, ReasonCode.BACKING_UNRESOLVED), OK])
    assert su.main(actions=recovered, environ={su.ANSWERS_ENV: answers_file(tmp_path, manual_backing="/x")}) == 0


@pytest.mark.parametrize("bad", [{"accept_plan": "yes"}, {"capabilities": ["__invalid__"]}, {"capabilities": [""]},
                                 {"capabilities": []}, {"extra": 1}, {"paths": {"bogus": "x"}}])
def test_noninteractive_invalid_answers_exit_2(tmp_path, bad, capsys):
    fake = FakeSetupActions()
    assert su.main(actions=fake, environ={su.ANSWERS_ENV: answers_file(tmp_path, **bad)}) == 2
    assert fake.calls == []


def test_missing_answers_file_exit_2(tmp_path):
    assert su.main(actions=FakeSetupActions(), environ={su.ANSWERS_ENV: str(tmp_path / "nope.json")}) == 2


def test_no_tty_no_answers_refuses_without_blocking(capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("must not prompt"))
    assert su.main(actions=FakeSetupActions(), environ={}, interactive=False) == 2
    assert su.ANSWERS_ENV in capsys.readouterr().err


class ScriptedPrompt:
    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def __call__(self, prompt):
        self.asked.append(prompt)
        if not self.answers:
            raise AssertionError("prompted more than scripted")
        reply = self.answers.pop(0)
        if reply is KeyboardInterrupt:
            raise KeyboardInterrupt
        return reply


def test_interactive_full_flow_with_back_and_recheck():
    fake = FakeSetupActions()
    prompt = ScriptedPrompt("1", "", "back", "", "recheck", "", "", "", "", "yes", *([""] * 7))
    lines = []
    assert su.run_interactive(fake, ask=prompt, out=lines.append) == 0
    assert fake.count("save_config") == 1 and fake.count("pull") == 1


# 2 = CPU transcripts (video only skips model selection). Prompts before MODEL_SELECT:
# capability, paths, host, docker, backing, plan, confirm, pull, hardware.
_TO_MODEL_SELECT = ("2", "", "", "", "", "", "yes", "", "")


def test_interactive_model_prompt_shows_options_and_enter_keeps_auto():
    fake = FakeSetupActions()
    prompt = ScriptedPrompt(*_TO_MODEL_SELECT, "", "", "", "", "")
    lines = []
    assert su.run_interactive(fake, ask=prompt, out=lines.append) == 0
    assert ("  * small: RAM 1.38 GiB, fits yes, basis published, "
            "source https://github.com/SYSTRAN/faster-whisper#benchmark") in lines
    assert "    base: RAM 0.84 GiB, fits yes, basis measured, source https://example.test/ours" in lines
    assert "    large-v3: RAM unknown, fits unknown, basis estimated, source unknown" in lines
    assert any(line.startswith("Enter keeps model auto") for line in lines)
    assert any("model id, or Enter for auto" in asked for asked in prompt.asked)
    assert fake.saved[-1]["selection"]["model"] == "auto"


def test_interactive_invalid_id_reprompts_then_explicit_pick_notes_fit():
    fake = FakeSetupActions()
    prompt = ScriptedPrompt(*_TO_MODEL_SELECT, "tiny", "large-v3", "", "", "", "")
    lines = []
    assert su.run_interactive(fake, ask=prompt, out=lines.append) == 0
    assert "invalid input (invalid_input); type a listed model id, or Enter for auto" in lines
    assert any(line.startswith("note: large-v3 is not known to fit") for line in lines)
    assert fake.saved[-1]["selection"]["model"] == "large-v3"


def test_interactive_pick_while_blocked_at_model_select_proceeds():
    fake = FakeSetupActions(choose_model=StepResult(True, data={"choice": None, "options": OPTIONS}))
    prompt = ScriptedPrompt(*_TO_MODEL_SELECT, "", "large-v3", "", "", "", "")
    lines = []
    assert su.run_interactive(fake, ask=prompt, out=lines.append) == 0
    assert any(line.startswith("blocked at model_select: invalid_input") for line in lines)
    assert any(asked.startswith("model id to choose one") for asked in prompt.asked)
    assert any(line.startswith("note: large-v3 is not known to fit") for line in lines)
    assert fake.calls.count("choose_model") == 1 and fake.saved[-1]["selection"]["model"] == "large-v3"
    assert lines[-1] == "setup: done"


def test_interactive_model_text_is_escaped():
    hostile = [{"id": "small", "peak_ram_bytes": 1, "basis": "published\x1b[2J", "source": "https://x\n\u202e",
                "fits": True}]
    fake = FakeSetupActions(choose_model=StepResult(True, data={"choice": "small", "options": hostile}))
    lines = []
    su.run_interactive(fake, ask=ScriptedPrompt(*_TO_MODEL_SELECT, "cancel"), out=lines.append)
    joined = "\n".join(lines)
    assert "\x1b" not in joined and "\u202e" not in joined
    assert "published\\u001b[2J" in joined and "https://x\\n\\u202e" in joined


def test_interactive_cancel_and_ctrl_c_write_nothing():
    fake = FakeSetupActions()
    assert su.run_interactive(fake, ask=ScriptedPrompt("1", "", "cancel"), out=lambda _: None) == 3
    assert su.run_interactive(fake, ask=ScriptedPrompt("2", "", "", KeyboardInterrupt), out=lambda _: None) == 130
    assert fake.count("save_config") == 0 and fake.count("pull") == 0


def _interrupt_on_done(action, target, reason):
    if action == "setup_done":
        raise KeyboardInterrupt


def test_ctrl_c_after_committed_save_reports_configuration_saved(tmp_path):
    fake = FakeSetupActions()
    fake.audit = _interrupt_on_done
    lines = []
    assert su.run_interactive(fake, ask=ScriptedPrompt("1", "", "", "", "", "", "yes", "", ""), out=lines.append) == 130
    assert fake.count("save_config") == 1
    assert lines[-1] == "setup interrupted; the configuration was already saved"
    fake = FakeSetupActions()
    fake.audit = _interrupt_on_done
    lines = []
    assert su.run_noninteractive(answers_file(tmp_path), fake, out=lines.append) == 130
    assert fake.count("save_config") == 1
    assert lines[-1] == "setup: interrupted; the configuration was already saved"
    before = FakeSetupActions()
    lines = []
    assert su.run_interactive(before, ask=ScriptedPrompt("1", "", KeyboardInterrupt), out=lines.append) == 130
    assert before.count("save_config") == 0 and lines[-1] == "setup interrupted; no configuration written"


def test_interactive_declined_plan_never_pulls():
    fake = FakeSetupActions()
    assert su.run_interactive(fake, ask=ScriptedPrompt("1", "", "", "", "", "", "no"), out=lambda _: None) == 3
    assert fake.count("pull") == 0


def _paths_answered(*lines):
    seen = []
    fake = FakeSetupActions()
    fake.check_paths = lambda paths: (seen.append(dict(paths)), StepResult(True, data={"paths": dict(paths)}))[1]
    out = []
    code = su.run_interactive(fake, ask=ScriptedPrompt("1", *lines, "cancel"), out=out.append)
    return code, seen, out


def test_interactive_paths_keep_spaces_and_strip_quotes():
    code, seen, _ = _paths_answered(r'output=C:\My Podcasts cache="D:\x y"')
    assert code == 3 and seen == [{"output": r"C:\My Podcasts", "cache": r"D:\x y"}]


@pytest.mark.parametrize("bad", ["foo=bar", r"junk output=C:\a", r"output=C:\a output=C:\b", "output=", "output"])
def test_interactive_paths_invalid_input_reprompts(bad):
    code, seen, out = _paths_answered(bad, "")
    assert code == 3 and seen == [{}]  # bad line never reaches the machine; blank line follows
    assert "invalid input (invalid_input)" in out


def test_interactive_blank_paths_line_gives_defaults():
    code, seen, _ = _paths_answered("")
    assert code == 3 and seen == [{}]


def test_parse_paths_line_all_keys():
    line = "data=/d d output='/o' scratch=/s docker_backing=/b x"
    assert su.parse_paths_line(line) == {"data": "/d d", "output": "/o", "scratch": "/s", "docker_backing": "/b x"}


def test_unbound_provisioning_is_service_unavailable():
    port = UnboundProvisioning()
    for name, args in (("pull", (Capability.MEDIA,)), ("probe_hardware", ()), ("choose_model", ({},)),
                       ("model_capacity", ("m",)), ("acquire_model", ("m",)), ("smoke", ("m",))):
        with pytest.raises(SetupError) as error:
            getattr(port, name)(*args)
        assert error.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE
        assert error.value.next_action == "bind provisioning in composition"


def test_default_local_actions_block_at_pull_with_clear_action(tmp_path):
    local = LocalSetupActions(config_path=tmp_path / "cfg" / "config.toml", audit=lambda *a: None,
                              volumes=_FakeVolumes(), docker=_FakeDocker())
    machine = SetupMachine(local, capabilities=["media"], audit=local.audit)
    result = su.drive(machine, paths={k: str(tmp_path / v) for k, v in PATHS.items()}, accept_plan=True)
    assert result.exit_code == 4 and result.reason_code == str(ReasonCode.SERVICE_UNAVAILABLE)
    assert result.next_action == "bind provisioning in composition"
    assert not (tmp_path / "cfg").exists()


def test_capability_contract_negatives_rejected():
    contract = json.loads((ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    for value in contract["negative_examples"]:
        with pytest.raises(DomainValidationError):
            Capability.parse(value)
        with pytest.raises(DomainValidationError):
            su.parse_capabilities([value])
    assert {c.value for c in Capability} == set(contract["capability"])


class _Measure:
    def __init__(self, value, reason=None):
        self.value, self.reason = value, reason


class _Volume:
    def __init__(self, free=10**12, identity="vol-default", writable=True):
        self.free = _Measure(free)
        self.capacity = _Measure(2 * 10**12)
        self.writable = _Measure(writable)
        self.mounted = _Measure(True)
        self.identity = _Measure(identity)


class _FakeVolumes:
    """Per-path volumes; unlisted paths land on one large default volume."""

    def __init__(self, by_path=None):
        self.by_path = {str(Path(k)): v for k, v in (by_path or {}).items()}

    def sample(self, path):
        return self.by_path.get(str(Path(path)), _Volume())

    def sample_many(self, paths):
        return tuple(self.sample(p) for p in paths)


class _Constraint:
    def __init__(self, roles, volume):
        self.roles, self.volume = tuple(roles), volume


class _FakeDocker:
    def __init__(self, constraints=(), error=None):
        self.constraints, self.error, self.samples = tuple(constraints), error, 0
        self.manual_backing = []

    def sample(self, manual_backing=""):
        self.samples += 1
        self.manual_backing.append(manual_backing)
        if self.error is not None:
            raise self.error
        constraints = self.constraints

        class Evidence:
            context = _Measure("desktop-linux")
            desktop_version = _Measure("4.40.0")
            can_pull = True
            reason = None
            backing = ()
        Evidence.constraints = constraints
        return Evidence()


def local_actions(tmp_path, audit):
    return LocalSetupActions(config_path=tmp_path / "cfg" / "config.toml", audit=audit,
                             volumes=_FakeVolumes(), docker=_FakeDocker())


def test_save_config_audited_atomic_write(tmp_path):
    events = []
    actions = local_actions(tmp_path, lambda *a: events.append(a))
    result = actions.save_config({"paths": {k: str(tmp_path / v) for k, v in PATHS.items()},
                                  "selection": {"model": "small"}, "outputs": {"transcript": True, "video": False}})
    assert result.ok
    target = tmp_path / "cfg" / "config.toml"
    assert "small" in target.read_text(encoding="utf-8")
    assert events and events[0][0] == "config_save" and len(events[0][1]) == 16
    assert [p.name for p in target.parent.iterdir()] == ["config.toml"]


def test_audit_failure_on_save_is_audit_loss_and_writes_nothing(tmp_path):
    def broken(*_):
        raise OSError("disk")

    actions = local_actions(tmp_path, broken)
    result = actions.save_config({"paths": {k: str(tmp_path / v) for k, v in PATHS.items()}})
    assert not result.ok and result.reason_code is ReasonCode.AUDIT_LOSS
    assert not (tmp_path / "cfg").exists()


def test_machine_audit_loss_blocks_as_failure(tmp_path):
    def broken(action, *_):
        if action == "config_save":
            raise OSError("disk")

    actions = local_actions(tmp_path, broken)
    machine = SetupMachine(actions, capabilities=["media"])
    actions.provision = FakeProvision(FakeSetupActions())
    result = su.drive(machine, paths={k: str(tmp_path / v) for k, v in PATHS.items()}, accept_plan=True)
    assert result.exit_code == 4 and result.reason_code == str(ReasonCode.AUDIT_LOSS)
    assert not (tmp_path / "cfg" / "config.toml").exists()


# ----------------------------------------------------------------------- setup re-run (merge)

RESTART_LINE = "a running service keeps the configuration it started with; restart it to use the saved one"


def _wizard(tmp_path, suffix="", model="small"):
    return {"paths": {k: str(tmp_path / (v + suffix)) for k, v in PATHS.items()},
            "selection": {"device": "cpu", "model": model}, "outputs": {"transcript": True, "video": False}}


def _load(target, model="small"):
    from podmuncher.config import Config

    return Config.from_toml(target.read_text(encoding="utf-8"),
                            verified_models=frozenset({model}) if model != "auto" else None,
                            verified_languages=frozenset({"en"}))


def test_rerun_keeps_settings_the_wizard_never_asks_about(tmp_path):
    """A corporate CA bundle, telemetry and a tuned limit survive a re-run that changes only
    paths and model; a pinned language the wizard never asks about survives too."""
    actions = local_actions(tmp_path, lambda *a: None)
    target = tmp_path / "cfg" / "config.toml"
    first = actions.save_config(_wizard(tmp_path, model="base"))
    assert first.ok and first.data["existing"] == "missing" and first.data["changed"] is True
    assert first.next_action == RESTART_LINE
    bundle = tmp_path / "corp-ca.pem"
    text = target.read_text(encoding="utf-8")
    text = text.replace('ca_bundle = ""', f"ca_bundle = {json.dumps(str(bundle))}")
    text = text.replace("otlp_enabled = false", "otlp_enabled = true")
    text = text.replace('otlp_endpoint = ""', 'otlp_endpoint = "http://collector.internal:4318"')
    text = text.replace("media_bytes = 10737418240", "media_bytes = 123456789")
    text = text.replace('language = "auto"', 'language = "en"')
    target.write_text(text, encoding="utf-8")
    before = _load(target, "base")
    assert before.tls.ca_bundle == str(bundle) and before.limits.media_bytes == 123456789

    second = actions.save_config(_wizard(tmp_path, "-new", model="small"))
    assert second.ok and second.data["existing"] == "merged" and second.data["changed"] is True
    assert second.next_action == RESTART_LINE
    after = _load(target, "small")
    assert after.paths.cache == str((tmp_path / "c-new").resolve()) and after.selection.model == "small"
    assert after.selection.device == "cpu" and after.selection.language == "en"
    assert after.tls.ca_bundle == str(bundle)
    assert after.telemetry.otlp_enabled is True and after.telemetry.otlp_endpoint == "http://collector.internal:4318"
    assert after.limits.media_bytes == 123456789
    # Everything the wizard never sets is byte-for-byte the earlier value.
    unprompted = {k: v for k, v in after.to_dict().items() if k not in ("paths", "selection", "outputs")}
    assert unprompted == {k: v for k, v in before.to_dict().items() if k not in ("paths", "selection", "outputs")}

    # The same answers again change nothing: no restart guidance.
    third = actions.save_config(_wizard(tmp_path, "-new", model="small"))
    assert third.ok and third.data["changed"] is False and third.next_action == ""


def test_rerun_without_existing_config_starts_from_defaults(tmp_path):
    from podmuncher.config import Limits, Telemetry

    actions = local_actions(tmp_path, lambda *a: None)
    result = actions.save_config(_wizard(tmp_path, model="auto"))
    assert result.ok and result.data["existing"] == "missing" and result.next_action == RESTART_LINE
    saved = _load(tmp_path / "cfg" / "config.toml", "auto")
    assert saved.limits == Limits() and saved.telemetry == Telemetry() and saved.tls.ca_bundle == ""
    assert saved.paths.data == str((tmp_path / "d").resolve()) and saved.selection.model == "auto"


@pytest.mark.parametrize("broken", ["not = [toml", 'schema_version = 2\n[paths]\ndata = "x"\n',
                                    "[limits]\nmedia_bytes = -1\n", "[selection]\nmodel = 7\n"])
def test_rerun_over_an_invalid_config_uses_defaults_and_says_so(tmp_path, broken):
    from podmuncher.config import Limits

    actions = local_actions(tmp_path, lambda *a: None)
    target = tmp_path / "cfg" / "config.toml"
    target.parent.mkdir(parents=True)
    target.write_text(broken, encoding="utf-8")
    result = actions.save_config(_wizard(tmp_path))
    assert result.ok and result.data["existing"] == "invalid"
    assert result.next_action.startswith("the existing configuration could not be read; settings this setup "
                                         "does not ask about were reset to their defaults; ")
    assert result.next_action.endswith(RESTART_LINE)
    saved = _load(target)
    assert saved.limits == Limits() and saved.selection.model == "small"
    assert [p.name for p in target.parent.iterdir()] == ["config.toml"]  # atomic: no temp left behind


def test_rerun_never_writes_when_the_merged_config_is_invalid(tmp_path):
    actions = local_actions(tmp_path, lambda *a: None)
    target = tmp_path / "cfg" / "config.toml"
    assert actions.save_config(_wizard(tmp_path)).ok
    before = target.read_bytes()
    result = actions.save_config({"paths": {"data": "relative\x00bad"}})
    assert not result.ok and result.reason_code is ReasonCode.INVALID_CONFIG
    assert target.read_bytes() == before


def test_done_carries_the_save_guidance(tmp_path):
    fake = FakeSetupActions(save_config=StepResult(True, data={"saved": True}, next_action=RESTART_LINE))
    result = su.drive(SetupMachine(fake, capabilities=FULL), paths=PATHS, accept_plan=True)
    assert result.exit_code == 0 and result.state == "done" and result.applied
    assert result.next_action == RESTART_LINE
    plain = su.drive(SetupMachine(FakeSetupActions(), capabilities=FULL), paths=PATHS, accept_plan=True)
    assert plain.state == "done" and plain.next_action == ""


# ------------------------------------------------------------------- bound provisioning paths


def test_paths_are_bound_before_capacity_and_again_after_a_change():
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.MODEL_ACQUIRE)
    (paths, at), = fake.bound
    assert paths == PATHS and fake.calls[at - 1] == "check_paths"
    assert at < fake.calls.index(("model_capacity", "small"))
    while machine.state is not S.PATHS:
        assert machine.back()
    changed = {**PATHS, "cache": "c2"}
    machine.next(changed)
    run_to(machine, S.MODEL_ACQUIRE)
    assert [p for p, _ in fake.bound] == [PATHS, changed]
    second_at = fake.bound[1][1]
    assert second_at > fake.calls.index(("model_capacity", "small"))  # the first capacity check
    assert second_at < len(fake.calls) - 1 - fake.calls[::-1].index(("model_capacity", "small"))
    assert fake.count("model_capacity") == 2  # the new paths invalidated the first result


def test_ports_without_bind_paths_still_run():
    fake = FakeSetupActions()
    fake.provision = _LegacyProvision(FakeProvision(fake))
    result = su.drive(SetupMachine(fake, capabilities=FULL), paths=PATHS, accept_plan=True)
    assert result.state == "done" and fake.bound == [] and fake.count("save_config") == 1
    # ``back`` below CONFIRM asks nothing of a port without ``release_cache`` either.
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.MODEL_ACQUIRE)
    while machine.state is not S.BACKING:
        assert machine.back()
    assert fake.released == []


def test_back_below_confirm_releases_the_port_cache_before_plan_reruns():
    """Provisioning (MODEL_ACQUIRE) opened the bound cache; going back to BACKING and on
    re-runs PLAN, which samples that cache root: the port releases it first (on Windows a
    held root samples as storage_unknown)."""
    fake = FakeSetupActions()
    machine = run_to(SetupMachine(fake, capabilities=FULL), S.SMOKE)
    assert fake.released == [] and fake.count("acquire_model") == 1
    while machine.state is not S.PLAN:
        assert machine.back()
    (released_at,) = fake.released  # once, when CONFIRM was crossed
    assert released_at == len(fake.calls) and machine.confirmed is False
    assert machine.back()  # to BACKING: PLAN's result is dropped
    machine.next()  # re-runs PLAN after the release
    assert machine.state is S.PLAN and fake.calls[-1] == "plan" and fake.released == [released_at]
    # Stepping back without ever crossing CONFIRM releases nothing more.
    assert machine.back() and machine.back() and fake.released == [released_at]
    run_to(machine, S.DONE)
    assert fake.count("acquire_model") == 2  # provisioned afresh after the release


def test_backing_override_is_bound_for_plan_and_the_pull_admission():
    """The Docker data location BACKING verified (a recovery override, or the typed path)
    reaches the port's bound paths, so PLAN and the pull's admission sample it."""
    fake = FakeSetupActions(backing=[StepResult(False, ReasonCode.BACKING_UNRESOLVED), OK])
    machine = SetupMachine(fake, capabilities=["media"])
    machine.next(PATHS)
    machine.next()
    machine.next()
    assert machine.state is S.BLOCKED and [p for p, _ in fake.bound] == [PATHS]
    machine.next("/mnt/existing")
    assert machine.state is S.BACKING
    assert [p for p, _ in fake.bound] == [PATHS, {**PATHS, "docker_backing": "/mnt/existing"}]
    assert fake.bound[-1][1] == fake.calls.index(("backing", "/mnt/existing")) + 1  # bound right after
    run_to(machine, S.CONFIRM)
    assert [p for p, _ in fake.bound] == [PATHS, {**PATHS, "docker_backing": "/mnt/existing"}]
    # A typed docker_backing is already bound at PATHS: BACKING binds nothing more.
    typed = FakeSetupActions()
    machine = SetupMachine(typed, capabilities=["media"])
    machine.next({**PATHS, "docker_backing": "/custom"})
    run_to(machine, S.CONFIRM)
    assert [p for p, _ in typed.bound] == [{**PATHS, "docker_backing": "/custom"}]


# --------------------------------------------------------------- service-backed apply


class FakeApplier:
    """``SetupApplier`` recording the service calls; the outcome is scripted per apply."""

    def __init__(self, *statuses, plan=None):
        self.statuses = list(statuses)
        self.plan_doc = plan or {"ready": True, "reason_code": "ok", "next_action": "", "size_known": True,
                                 "sizes": {"media": {"download_bytes": 1}}}
        self.plans, self.applies, self.waited, self.closed = [], [], [], 0

    def plan(self, capabilities, config):
        self.plans.append((list(capabilities), json.loads(json.dumps(config))))
        return {"plan_id": f"plan-{len(self.plans)}", "capabilities": list(capabilities), **self.plan_doc}

    def apply(self, plan_id, key):
        self.applies.append((plan_id, key))
        return {"operation_id": f"op-{len(self.applies)}", "plan_id": plan_id, "state": "running",
                "reason_code": "ok", "next_action": ""}

    def wait(self, operation):
        self.waited.append(dict(operation))
        status = self.statuses.pop(0)
        if isinstance(status, BaseException):
            raise status
        return {"operation_id": operation["operation_id"], **status}

    def close(self):
        self.closed += 1


DONE = {"state": "done", "reason_code": "ok", "next_action": "a running service keeps the configuration",
        "setup_state": "done", "exit_code": 0}
BLOCKED_BACKING = {"state": "failed", "reason_code": "backing_unresolved", "setup_state": "blocked",
                   "next_action": "enter the existing Docker data location", "exit_code": 3}


def test_service_setup_wraps_the_client_routes_and_polls_until_settled():
    class Client:
        def __init__(self):
            self.calls, self.closed = [], 0
            self.states = ["running", "running", "done"]

        def setup_plan(self, capabilities, config):
            self.calls.append(("plan", capabilities, config))
            return {"plan": {"plan_id": "p1", "ready": True}}

        def setup_apply(self, plan_id, key):
            self.calls.append(("apply", plan_id, key))
            return {"operation": {"operation_id": "o1", "state": "running"}}

        def setup_status(self, operation_id):
            self.calls.append(("status", operation_id))
            return {"operation_id": operation_id, "state": self.states.pop(0), "exit_code": 0}

        def close(self):
            self.closed += 1

    client, slept = Client(), []
    applier = su.ServiceSetup(client, sleep=slept.append, poll_seconds=0.25)
    assert applier.plan(["media"], {"paths": {}}) == {"plan_id": "p1", "ready": True}
    operation = applier.apply("p1", "k")
    assert applier.wait(operation)["state"] == "done" and slept == [0.25] * 3
    assert client.calls == [("plan", ["media"], {"paths": {}}), ("apply", "p1", "k")] + [("status", "o1")] * 3
    applier.close()
    assert client.closed == 1
    # An operation already settled when apply returned is not polled.
    assert applier.wait({"operation_id": "o2", "state": "done"}) == {"operation_id": "o2", "state": "done"}


def test_remote_noninteractive_answers_apply_through_the_service(tmp_path):
    fake = FakeApplier(DONE)
    lines = []
    path = answers_file(tmp_path, capabilities=list(FULL), model="base", manual_backing="/x",
                        idempotency_key="pinned-key")
    assert su.run_noninteractive(path, applier=lambda: fake, out=lines.append) == 0
    expected_paths = {k: str(tmp_path / v) for k, v in PATHS.items()}
    assert fake.plans == [(["asr_cpu", "media"], {"paths": expected_paths, "manual_backing": "/x", "model": "base"})]
    assert fake.applies == [("plan-1", "pinned-key")] and fake.waited[0]["operation_id"] == "op-1"
    assert lines == ["setup: done (ok) - a running service keeps the configuration"] and fake.closed == 1
    # Without a pinned key every run applies afresh (a fresh key), never a stale replay.
    fresh = FakeApplier(DONE, DONE)
    assert su.main(environ={su.ANSWERS_ENV: answers_file(tmp_path)}, applier=lambda: fresh) == 0
    assert su.main(environ={su.ANSWERS_ENV: answers_file(tmp_path)}, applier=lambda: fresh) == 0
    (_, first), (_, second) = fresh.applies
    assert first != second and fresh.plans[0][1] == {"paths": {k: str(tmp_path / v) for k, v in PATHS.items()}}


def test_remote_noninteractive_outcomes(tmp_path):
    # Declined: planned (the sizes are the operator's information), never applied, exit 3.
    declined = FakeApplier()
    lines = []
    assert su.run_noninteractive(answers_file(tmp_path, accept_plan=False), applier=lambda: declined,
                                 out=lines.append) == 3
    assert len(declined.plans) == 1 and declined.applies == [] and lines == ["setup: cancelled (cancelled)"]
    # Blocked in the service: the machine's state, reason, next action and exit.
    blocked = FakeApplier(BLOCKED_BACKING)
    lines = []
    assert su.run_noninteractive(answers_file(tmp_path), applier=lambda: blocked, out=lines.append) == 3
    assert lines == ["setup: blocked (backing_unresolved) - enter the existing Docker data location"]
    # Failed before the machine ran (a refused lease): a runtime failure with its reason.
    refused = FakeApplier({"state": "failed", "reason_code": "conflict", "next_action": ""})
    lines = []
    assert su.run_noninteractive(answers_file(tmp_path), applier=lambda: refused, out=lines.append) == 4
    assert lines == ["setup: blocked (conflict)"]
    # The service refusing the request (a closed client error) maps through the reason table.
    invalid = FakeApplier(su.SetupError(ReasonCode.INVALID_INPUT))
    lines = []
    assert su.run_noninteractive(answers_file(tmp_path), applier=lambda: invalid, out=lines.append) == 2
    assert lines == ["setup: invalid (invalid_input)"] and invalid.closed == 1
    # Ctrl+C while the service works: the setup continues there.
    interrupted = FakeApplier(KeyboardInterrupt())
    lines = []
    assert su.run_noninteractive(answers_file(tmp_path), applier=lambda: interrupted, out=lines.append) == 130
    assert lines == ["setup: interrupted; a started setup continues in the service"] and interrupted.closed == 1
    # Invalid answers never reach (or start) the service.
    never = []
    assert su.run_noninteractive(answers_file(tmp_path, capabilities=[]), applier=lambda: never.append(1),
                                 out=lambda _: None) == 2
    assert never == []


def _remote_interactive(fake, *answers):
    lines = []
    actions = FakeSetupActions()
    code = su.run_interactive(actions, ask=ScriptedPrompt(*answers), out=lines.append,
                              applier=lambda: fake)
    return code, lines, actions


def test_remote_interactive_prompts_here_and_applies_in_the_service():
    fake = FakeApplier(DONE)
    code, lines, actions = _remote_interactive(fake, "2", "cache=/c docker_backing=/b", "base", "yes")
    assert code == 0 and lines[-1] == "setup: done (ok) - a running service keeps the configuration"
    assert fake.plans == [(["asr_cpu", "media"], {"paths": {"cache": "/c", "docker_backing": "/b"}, "model": "base"})]
    assert len(fake.applies) == 1 and fake.closed == 1
    assert actions.count("check_paths") == 1 and actions.count("pull") == 0  # nothing provisioned here
    assert any(line.startswith('[plan] {"capabilities": ["asr_cpu", "media"]') and '"ready": true' in line
               for line in lines)


def test_remote_interactive_blank_paths_and_auto_model_send_nothing_extra():
    fake = FakeApplier(DONE)
    code, _, _ = _remote_interactive(fake, "1", "", "", "y")
    assert code == 0 and fake.plans == [(["media"], {"paths": {}})]


def test_remote_interactive_block_offers_another_answer_set():
    fake = FakeApplier(BLOCKED_BACKING, DONE)
    code, lines, _ = _remote_interactive(fake, "1", "", "", "yes", "yes", "docker_backing=/mnt/docker", "", "yes")
    assert code == 0
    assert "setup: blocked (backing_unresolved) - enter the existing Docker data location" in lines
    assert [c for c, _ in fake.plans] == [["media"], ["media"]]
    assert fake.plans[1][1] == {"paths": {"docker_backing": "/mnt/docker"}}
    assert fake.applies == [("plan-1", fake.applies[0][1]), ("plan-2", fake.applies[1][1])]
    # Declining another answer set keeps the machine's exit.
    fake = FakeApplier(BLOCKED_BACKING)
    code, _, _ = _remote_interactive(fake, "1", "", "", "yes", "no")
    assert code == 3 and len(fake.applies) == 1


def test_remote_interactive_invalid_paths_reprompt_and_cancel_or_decline_never_apply():
    fake = FakeApplier()
    code, lines, _ = _remote_interactive(fake, "1", "junk", "", "", "no")
    assert code == 3 and "invalid input (invalid_input)" in lines and lines[-1] == "setup: cancelled (cancelled)"
    assert len(fake.plans) == 1 and fake.applies == [] and fake.closed == 1
    fake = FakeApplier()
    assert _remote_interactive(fake, "2", "cancel")[0] == 3 and fake.plans == [] and fake.closed == 1
    fake = FakeApplier()
    assert _remote_interactive(fake, "2", "", "cancel")[0] == 3 and fake.plans == []
    # Cancelling at the choice never starts (or touches) the service.
    never = []
    assert su.run_interactive(FakeSetupActions(), ask=ScriptedPrompt("cancel"), out=lambda _: None,
                              applier=lambda: never.append(1)) == 3
    assert never == []


def test_remote_interactive_ctrl_c_and_service_failures():
    fake = FakeApplier(KeyboardInterrupt())
    code, lines, _ = _remote_interactive(fake, "1", "", "", "yes")
    assert code == 130 and lines[-1] == "setup interrupted; a started setup continues in the service"
    fake = FakeApplier()
    code, lines, _ = _remote_interactive(fake, "1", "", KeyboardInterrupt)
    assert code == 130 and lines[-1] == "setup interrupted; no configuration written" and fake.closed == 1
    fake = FakeApplier(su.SetupError(ReasonCode.CONFLICT))
    code, lines, _ = _remote_interactive(fake, "1", "", "", "yes")
    assert code == 4 and lines[-1] == "setup: failed (conflict)" and fake.closed == 1


# --------------------------------------------------------------------------- plan budgets

GIB = 1024**3


def _plan_actions(tmp_path, *, volumes, sizes, docker=None):
    return LocalSetupActions(config_path=tmp_path / "cfg" / "config.toml", audit=lambda *a: None,
                             volumes=_FakeVolumes(volumes), docker=docker or _FakeDocker(), sizes=sizes)


def _budget(result, *roles):
    return next(b for b in result.data["volumes"] if b["roles"] == sorted(roles))


def test_plan_budgets_each_volume_separately(tmp_path):
    """A small, unrelated output volume no longer blocks a model that lands in the cache."""
    paths = {k: str(tmp_path / v) for k, v in PATHS.items()}
    volumes = {paths["output"]: _Volume(free=1 * GIB, identity="vol-out"),
               paths["cache"]: _Volume(free=10 * GIB, identity="vol-cache"),
               paths["scratch"]: _Volume(free=10 * GIB, identity="vol-scratch"),
               paths["data"]: _Volume(free=10 * GIB, identity="vol-data")}
    sizes = {"asr_cpu": {"download_bytes": 0, "reserve_bytes": 5 * GIB}, "media": {"download_bytes": 0,
                                                                                  "reserve_bytes": GIB // 4}}
    result = _plan_actions(tmp_path, volumes=volumes, sizes=sizes).plan(("asr_cpu", "media"), paths)
    assert result.ok and result.reason_code is ReasonCode.OK and result.data["size_known"] is True
    assert _budget(result, "cache") == {"roles": ["cache"], "free_bytes": 10 * GIB,
                                        "cost_bytes": 5 * GIB + GIB // 4, "fits": True}
    assert _budget(result, "output") == {"roles": ["output"], "free_bytes": GIB, "cost_bytes": 0, "fits": True}
    # The same budget on a cache volume that is too small blocks with the shortfall named.
    volumes[paths["cache"]] = _Volume(free=4 * GIB, identity="vol-cache")
    result = _plan_actions(tmp_path, volumes=volumes, sizes=sizes).plan(("asr_cpu", "media"), paths)
    assert not result.ok and result.reason_code is ReasonCode.STORAGE_INSUFFICIENT
    assert result.next_action.startswith("free space before applying: the volume holding cache needs 1.25 GiB")
    assert _budget(result, "cache")["fits"] is False and _budget(result, "output")["fits"] is True


def test_plan_sums_costs_on_a_shared_volume(tmp_path):
    paths = {k: str(tmp_path / v) for k, v in PATHS.items()}
    sizes = {"asr_cpu": {"model_bytes": 6 * GIB, "scratch_bytes": 5 * GIB, "output_bytes": GIB}}
    separate = {paths["cache"]: _Volume(free=10 * GIB, identity="vol-a"),
                paths["scratch"]: _Volume(free=10 * GIB, identity="vol-b"),
                paths["output"]: _Volume(free=10 * GIB, identity="vol-c")}
    assert _plan_actions(tmp_path, volumes=separate, sizes=sizes).plan(("asr_cpu",), paths).ok
    # cache and scratch are aliases of one volume: 11 GiB against the smaller alias measurement.
    shared = {paths["cache"]: _Volume(free=12 * GIB, identity="vol-a"),
              paths["scratch"]: _Volume(free=10 * GIB, identity="vol-a"),
              paths["output"]: _Volume(free=10 * GIB, identity="vol-c")}
    result = _plan_actions(tmp_path, volumes=shared, sizes=sizes).plan(("asr_cpu",), paths)
    assert not result.ok and result.reason_code is ReasonCode.STORAGE_INSUFFICIENT
    assert _budget(result, "cache", "scratch") == {"roles": ["cache", "scratch"], "free_bytes": 10 * GIB,
                                                   "cost_bytes": 11 * GIB, "fits": False}
    assert _budget(result, "output")["fits"] is True
    assert "cache, scratch needs 1.00 GiB more" in result.next_action


def test_plan_charges_image_pulls_to_docker_volumes(tmp_path):
    paths = {k: str(tmp_path / v) for k, v in PATHS.items()}
    sizes = {"media": {"download_bytes": 3 * GIB, "reserve_bytes": 0}}
    small_vm = _FakeDocker([_Constraint(("docker_data",), _Volume(free=1 * GIB, identity="wsl:vm"))])
    result = _plan_actions(tmp_path, volumes={}, sizes=sizes, docker=small_vm).plan(("media",), paths)
    assert not result.ok and result.reason_code is ReasonCode.STORAGE_INSUFFICIENT and small_vm.samples == 1
    assert _budget(result, "docker") == {"roles": ["docker"], "free_bytes": GIB, "cost_bytes": 3 * GIB,
                                         "fits": False}
    # Docker Desktop for Mac: Docker.raw lives on the host volume the cache uses, so the
    # pull and the cache share one budget with the VM headroom as the tighter measurement.
    mac = _FakeDocker([_Constraint(("docker_data",), _Volume(free=4 * GIB, identity="apfs:data"))])
    volumes = {paths["cache"]: _Volume(free=50 * GIB, identity="apfs:data")}
    sizes["asr_cpu"] = {"download_bytes": 0, "model_bytes": 2 * GIB}
    result = _plan_actions(tmp_path, volumes=volumes, sizes=sizes, docker=mac).plan(("asr_cpu", "media"), paths)
    assert not result.ok
    assert _budget(result, "cache", "docker") == {"roles": ["cache", "docker"], "free_bytes": 4 * GIB,
                                                  "cost_bytes": 5 * GIB, "fits": False}
    # Docker free unknown: BACKING owns Docker admission, so the plan reports instead of blocking.
    broken = _FakeDocker(error=OSError("socket"))
    result = _plan_actions(tmp_path, volumes={}, sizes={"media": {"download_bytes": 3 * GIB}},
                           docker=broken).plan(("media",), paths)
    assert result.ok and _budget(result, "docker") == {"roles": ["docker"], "free_bytes": None,
                                                       "cost_bytes": 3 * GIB, "fits": None}


def test_plan_without_sizes_samples_nothing(tmp_path):
    paths = {k: str(tmp_path / v) for k, v in PATHS.items()}
    docker = _FakeDocker()
    result = _plan_actions(tmp_path, volumes={}, sizes={}, docker=docker).plan(("media",), paths)
    assert result.ok and result.data["size_known"] is False and "volumes" not in result.data
    assert docker.samples == 0
    # Host paths without a verdict still block the plan, with the honest reason.
    unknown = {paths["cache"]: _Volume(free=None, identity=None)}
    result = _plan_actions(tmp_path, volumes=unknown, sizes={"media": {"reserve_bytes": GIB}}).plan(("media",), paths)
    assert not result.ok and result.reason_code is ReasonCode.STORAGE_UNKNOWN


@pytest.mark.parametrize("role", ["docker_vm", "containerd", "docker_backing"])
def test_plan_budgets_every_docker_store(tmp_path, role):
    """The WSL VM disk and the containerd store are Docker stores too: each must hold the pull."""
    paths = {k: str(tmp_path / v) for k, v in PATHS.items()}
    sizes = {"media": {"download_bytes": 3 * GIB, "reserve_bytes": 0}}
    small = _FakeDocker([_Constraint((role,), _Volume(free=1 * GIB, identity=f"store:{role}"))])
    result = _plan_actions(tmp_path, volumes={}, sizes=sizes, docker=small).plan(("media",), paths)
    assert not result.ok and result.reason_code is ReasonCode.STORAGE_INSUFFICIENT
    assert result.next_action == "free space before applying: the volume holding docker needs 2.00 GiB more"
    assert _budget(result, "docker") == {"roles": ["docker"], "free_bytes": GIB, "cost_bytes": 3 * GIB,
                                         "fits": False}
    # Distinct stores are budgeted one by one: a roomy data root does not hide a full VM disk.
    mixed = _FakeDocker([_Constraint(("docker_data",), _Volume(free=50 * GIB, identity="ext4:data")),
                         _Constraint((role,), _Volume(free=1 * GIB, identity=f"store:{role}"))])
    result = _plan_actions(tmp_path, volumes={}, sizes=sizes, docker=mixed).plan(("media",), paths)
    assert not result.ok and result.reason_code is ReasonCode.STORAGE_INSUFFICIENT
    budgets = [b for b in result.data["volumes"] if b["roles"] == ["docker"]]
    assert sorted(b["free_bytes"] for b in budgets) == [GIB, 50 * GIB]
    assert [b["fits"] for b in sorted(budgets, key=lambda b: b["free_bytes"])] == [False, True]


def test_plan_forwards_the_verified_docker_backing(tmp_path):
    paths = {k: str(tmp_path / v) for k, v in PATHS.items()}
    sizes = {"media": {"download_bytes": GIB}}
    docker = _FakeDocker([_Constraint(("docker_data",), _Volume(free=10 * GIB, identity="wsl:vm"))])
    backing = str(tmp_path / "docker-data")
    actions = _plan_actions(tmp_path, volumes={}, sizes=sizes, docker=docker)
    assert actions.plan(("media",), {**paths, "docker_backing": backing}).ok
    assert docker.manual_backing == [backing]
    # Without an operator backing the sample runs on auto discovery, as before.
    assert actions.plan(("media",), paths).ok and docker.manual_backing == [backing, ""]
    # Through the machine: BACKING verifies paths.docker_backing and PLAN samples the same value.
    docker.manual_backing.clear()
    machine = SetupMachine(actions, capabilities=["media"])
    actions.provision = FakeProvision(FakeSetupActions())
    machine.next({**paths, "docker_backing": backing})
    run_to(machine, S.CONFIRM)
    assert docker.manual_backing == ["", backing, backing]  # DOCKER_CHECK, BACKING, PLAN


def test_plan_reports_unknown_host_free_space_as_storage_unknown(tmp_path):
    paths = {k: str(tmp_path / v) for k, v in PATHS.items()}
    sizes = {"media": {"reserve_bytes": GIB}}

    class _Failing(_FakeVolumes):
        def sample_many(self, paths):
            raise OSError("statvfs")

    actions = LocalSetupActions(config_path=tmp_path / "cfg" / "config.toml", audit=lambda *a: None,
                                volumes=_Failing(), docker=_FakeDocker(), sizes=sizes)
    result = actions.plan(("media",), paths)
    assert not result.ok and result.reason_code is ReasonCode.STORAGE_UNKNOWN
    assert result.next_action == "free space before applying"
    # A measured but unwritable host path still blocks as insufficient.
    unwritable = {paths["cache"]: _Volume(free=10 * GIB, identity="vol-ro", writable=False)}
    result = _plan_actions(tmp_path, volumes=unwritable, sizes=sizes).plan(("media",), paths)
    assert not result.ok and result.reason_code is ReasonCode.STORAGE_INSUFFICIENT


# --------------------------------------------------------------------------- darwin file sharing


def _settings(tmp_path, *shared):
    target = tmp_path / "settings-store.json"
    target.write_text(json.dumps({"FilesharingDirectories": list(shared), "SettingsVersion": 46}), encoding="utf-8")
    return target


def test_darwin_paths_outside_docker_shared_folders_warn_without_blocking(tmp_path):
    shared_root, elsewhere = tmp_path / "shared", tmp_path / "elsewhere"
    settings = _settings(tmp_path, str(shared_root))
    actions = LocalSetupActions(config_path=tmp_path / "cfg" / "config.toml", audit=lambda *a: None,
                                volumes=_FakeVolumes(), docker=_FakeDocker(), platform="darwin",
                                docker_settings=settings)
    paths = {"data": str(elsewhere / "data"), "output": str(shared_root / "out"),
             "cache": str(elsewhere / "cache"), "scratch": str(shared_root / "scratch")}
    result = actions.check_paths(paths)
    assert result.ok and result.reason_code is ReasonCode.OK
    assert len(result.data["warnings"]) == 1
    assert result.data["warnings"][0].startswith(f"cache path {Path(paths['cache']).resolve()} is outside")
    assert "Docker Desktop > Settings > Resources > File sharing" in result.data["warnings"][0]
    # The same paths with every one shared, or on another platform, carry no warning at all.
    assert "warnings" not in actions.check_paths({**paths, "cache": str(shared_root / "c")}).data
    windows = LocalSetupActions(config_path=tmp_path / "cfg" / "config.toml", audit=lambda *a: None,
                                volumes=_FakeVolumes(), docker=_FakeDocker(), platform="win32",
                                docker_settings=settings)
    assert "warnings" not in windows.check_paths(paths).data
    # A machine drives through the warning: the step passes and the warning is on record.
    machine = SetupMachine(actions, capabilities=["media"])
    machine.next(paths)
    assert machine.state is S.HOST_CHECK and machine.results[S.PATHS].data["warnings"]


@pytest.mark.parametrize("settings", [None, "not json", json.dumps({"FilesharingDirectories": "/Users"}),
                                      json.dumps({"FilesharingDirectories": ["relative"]}), json.dumps([])])
def test_darwin_unreadable_sharing_settings_warn_nothing(tmp_path, settings):
    target = tmp_path / "settings-store.json"
    if settings is not None:
        target.write_text(settings, encoding="utf-8")
    values = {"output": str(tmp_path / "o"), "cache": str(tmp_path / "c"), "scratch": str(tmp_path / "s")}
    assert su.shared_folder_warnings(values, platform="darwin", settings_path=target) == []


def test_shared_folder_warnings_posix_layout(tmp_path):
    settings = _settings(tmp_path, "/Users", "/Volumes", "/private", "/tmp", "/var/folders")
    values = {"output": "/Users/me/Library/Application Support/podmuncher/outputs",
              "cache": "/Users/me/Library/Caches/podmuncher", "scratch": "/opt/scratch"}
    # Pure path comparison, so the darwin shape is checked on every development host.
    warnings = su.shared_folder_warnings({k: str(Path(v)) for k, v in values.items()}, platform="darwin",
                                         settings_path=settings)
    assert len(warnings) == 1 and warnings[0].startswith("scratch path ")
    assert su.shared_folder_warnings(values, platform="linux", settings_path=settings) == []
    oversized = tmp_path / "big.json"
    oversized.write_bytes(b" " * (su._MAX_SETTINGS_BYTES + 1))
    assert su.shared_folder_warnings(values, platform="darwin", settings_path=oversized) == []


def test_deeply_nested_sharing_settings_warn_nothing_without_crashing(tmp_path):
    """Nested JSON under the byte cap raises RecursionError in json.loads; the advisory stays silent."""
    target = tmp_path / "settings-store.json"
    depth = 100_000
    target.write_text("[" * depth + "]" * depth, encoding="utf-8")
    assert target.stat().st_size <= su._MAX_SETTINGS_BYTES
    values = {"output": str(tmp_path / "o"), "cache": str(tmp_path / "c"), "scratch": str(tmp_path / "s")}
    assert su.shared_folder_warnings(values, platform="darwin", settings_path=target) == []
    actions = LocalSetupActions(config_path=tmp_path / "cfg" / "config.toml", audit=lambda *a: None,
                                volumes=_FakeVolumes(), docker=_FakeDocker(), platform="darwin",
                                docker_settings=target)
    result = actions.check_paths({**values, "data": str(tmp_path / "d")})
    assert result.ok and "warnings" not in result.data


# --------------------------------------------------------------------------- live


@pytest.mark.live
def test_own_transport_smoke(tmp_path):
    from podmuncher import docker_storage

    spawned = []
    real = subprocess.Popen

    def spy(argv, *args, **kwargs):
        spawned.append(tuple(argv))
        return real(argv, *args, **kwargs)

    provider = docker_storage.make_docker_storage_provider(runner=docker_storage.BoundedRunner(popen=spy))
    actions = LocalSetupActions(config_path=tmp_path / "cfg" / "config.toml", docker=provider,
                                audit=lambda *a: pytest.fail("doctor must not audit-write"))
    paths = {k: str(tmp_path / "never-created" / k) for k in ("data", "output", "cache", "scratch")}
    cwd_before = sorted(p.name for p in Path.cwd().iterdir())
    report = su.doctor_report(paths, actions=actions)

    assert report["read_only"] is True and report["status"] in ("ok", "blocked")
    assert set(report["checks"]) == {"paths", "host_capacity", "docker", "backing", "cache"}
    for name, check in report["checks"].items():
        assert check["status"] in ("ok", "blocked", "unavailable", "unknown"), name
        assert isinstance(check["reason_code"], str) and check["reason_code"], name
        ReasonCode(check["reason_code"])
    assert list(tmp_path.iterdir()) == []
    assert sorted(p.name for p in Path.cwd().iterdir()) == cwd_before
    for argv in spawned:
        assert argv[:1] != ("docker",) or argv[1:2] not in (("pull",), ("run",), ("start",), ("create",), ("compose",))
        assert not {"pull", "run", "start", "create"} & set(argv[1:3])
    assert not any("service" in " ".join(a) and "start" in a for a in spawned)
    for name, check in report["checks"].items():
        print(f"doctor {name}: {check['status']} {check['reason_code']}")
    print("PODMUNCHER_SMOKE_OK")
