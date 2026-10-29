"""Guided setup state machine and read-only doctor report.

The machine is pure: every effect goes through ``SetupActions``.  Entering a
state never acquires anything; image pulls, hardware probes, model acquisition
and the inference smoke run only after the plan was explicitly confirmed, and
configuration is written once, last, through ``Config.persist`` with an audited
atomic writer.  Provisioning (image pull, in-worker probe, model acquire,
smoke) is an injected ``ProvisioningPort``; composition binds the real one
(SPEC_GAP-9).  Heavy dependencies are imported lazily inside functions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from .domains import Capability, DomainValidationError, ReasonCode

EXIT_OK, EXIT_INVALID, EXIT_BLOCKED, EXIT_FAILED, EXIT_INTERRUPTED = 0, 2, 3, 4, 130
ANSWERS_ENV = "PODMUNCHER_SETUP_ANSWERS"
_MAX_ANSWERS_BYTES = 65536
_PATH_KEYS = ("data", "output", "cache", "scratch", "docker_backing")

AuditCallable = Callable[[str, str, ReasonCode], None]

# Provisioning costs by destination volume (SPEC_GAP-17 review): image pulls and Docker's
# own data land on the Docker backing; models in the cache; scratch work in scratch; results
# in output.  ``reserve_bytes`` is the older combined worker reserve, counted in the cache
# where the model it mostly consists of lives.  Each volume is budgeted on its own; paths
# that share one volume (same identity) sum their costs on it.
_COST_DESTINATIONS = {"download_bytes": "docker", "docker_bytes": "docker", "model_bytes": "cache",
                      "reserve_bytes": "cache", "scratch_bytes": "scratch", "output_bytes": "output"}
# Docker Desktop for Mac bind-mounts only these shared folders (Settings > Resources > File sharing).
_SHARED_PATH_KEYS = ("output", "cache", "scratch")
_DOCKER_SETTINGS = Path("Library") / "Group Containers" / "group.com.docker" / "settings-store.json"
_MAX_SETTINGS_BYTES = 1048576
_MAX_CONFIG_BYTES = 1048576
_MAX_SHARED_DIRECTORIES = 256


class SetupError(DomainValidationError):
    """Closed-reason setup failure carrying the operator's next action."""

    def __init__(self, reason_code: ReasonCode = ReasonCode.INVALID_INPUT, *, next_action: str = "") -> None:
        super().__init__(reason_code)
        self.next_action = next_action


class SetupState(StrEnum):
    PATHS = "paths"
    HOST_CHECK = "host_check"
    DOCKER_CHECK = "docker_check"
    BACKING = "backing"
    PLAN = "plan"
    CONFIRM = "confirm"
    PULL = "pull"
    HARDWARE = "hardware"
    MODEL_SELECT = "model_select"
    MODEL_CAPACITY = "model_capacity"
    MODEL_ACQUIRE = "model_acquire"
    SMOKE = "smoke"
    SAVE_CONFIG = "save_config"
    DONE = "done"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


S = SetupState
_ORDER = (S.PATHS, S.HOST_CHECK, S.DOCKER_CHECK, S.BACKING, S.PLAN, S.CONFIRM, S.PULL, S.HARDWARE,
          S.MODEL_SELECT, S.MODEL_CAPACITY, S.MODEL_ACQUIRE, S.SMOKE, S.SAVE_CONFIG, S.DONE)
_ASR_ONLY = frozenset({S.HARDWARE, S.MODEL_SELECT, S.MODEL_CAPACITY, S.MODEL_ACQUIRE, S.SMOKE})
# Results of these states survive back(): they are provisioning and are never re-run.
_PERSISTED = frozenset({S.PULL, S.HARDWARE, S.MODEL_ACQUIRE, S.SMOKE})
_NEEDS_CONFIRM = frozenset({S.PULL, S.HARDWARE, S.MODEL_ACQUIRE, S.SMOKE, S.SAVE_CONFIG})
# DONE is here for a lost completion audit: the configuration is committed, the record is not.
_FAILURE_STATES = frozenset({S.PULL, S.MODEL_ACQUIRE, S.SMOKE, S.SAVE_CONFIG, S.DONE})
_TERMINAL = frozenset({S.DONE, S.CANCELLED})
_CONFIRM_INDEX = _ORDER.index(S.CONFIRM)
# Results that are only valid for the model choice they were produced for.
_CHOICE_BOUND = (S.MODEL_CAPACITY, S.MODEL_ACQUIRE, S.SMOKE)
_PATH_SPLIT = re.compile(r"\s+(?=(?:" + "|".join(_PATH_KEYS) + r")=)")


@dataclass(frozen=True, slots=True)
class StepResult:
    ok: bool
    reason_code: ReasonCode = ReasonCode.OK
    data: Mapping[str, Any] = field(default_factory=dict)
    next_action: str = ""


class ProvisioningPort(Protocol):
    def pull(self, capability: Capability) -> StepResult: ...
    def probe_hardware(self) -> StepResult: ...
    def choose_model(self, hardware: Mapping[str, Any]) -> StepResult: ...
    def model_capacity(self, choice: str) -> StepResult: ...
    def acquire_model(self, choice: str) -> StepResult: ...
    def smoke(self, choice: str) -> StepResult: ...

    def bind_paths(self, paths: Mapping[str, str]) -> None:
        """Optional: the accepted wizard paths; capacity, acquisition and smoke use them.

        The machine calls it each time PATHS is accepted (and again when BACKING verifies
        an operator override), before any provisioning step.  Ports without it are still
        valid (the machine looks it up with ``getattr``).
        """

    def release_cache(self) -> None:
        """Optional: release anything the port holds open on the bound paths (a setup-scoped
        model cache).  The machine calls it when ``back`` crosses below CONFIRM, before any
        pre-confirmation check (host capacity, plan) can re-run against a held root.
        """


class SetupActions(Protocol):
    provision: ProvisioningPort

    def check_paths(self, paths: Mapping[str, str]) -> StepResult: ...
    def host_capacity(self, paths: Mapping[str, str]) -> StepResult: ...
    def docker_status(self) -> StepResult: ...
    def backing(self, manual: str | None = None) -> StepResult: ...
    def plan(self, capabilities: tuple[str, ...], paths: Mapping[str, str]) -> StepResult: ...
    def cached_snapshot(self, choice: str) -> str | None: ...
    def save_config(self, config: Mapping[str, Any]) -> StepResult: ...


class UnboundProvisioning:
    """Default port: provisioning is bound by composition (P6-U2)."""

    def _unbound(self, *_args: object) -> StepResult:
        raise SetupError(ReasonCode.SERVICE_UNAVAILABLE, next_action="bind provisioning in composition")

    pull = probe_hardware = choose_model = model_capacity = acquire_model = smoke = _unbound

    def bind_paths(self, paths: Mapping[str, str]) -> None:
        """Binding is not provisioning: accepting paths never needs a bound port."""

    def release_cache(self) -> None:
        """Nothing is held open."""


def parse_capabilities(values: object) -> frozenset[Capability]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise SetupError(ReasonCode.INVALID_INPUT)
    parsed = frozenset(Capability.parse(v) for v in values)
    if not parsed:
        raise SetupError(ReasonCode.INVALID_INPUT)
    return parsed | {Capability.MEDIA}


class SetupMachine:
    """Sequential state machine; ``next``/``back``/``recheck``/``cancel`` are the only inputs."""

    def __init__(self, actions: SetupActions, *, capabilities: object, audit: AuditCallable | None = None) -> None:
        self.actions = actions
        self.capabilities = parse_capabilities(capabilities)
        self.video_only = self.capabilities == {Capability.MEDIA}
        self.order = tuple(s for s in _ORDER if not (self.video_only and s in _ASR_ONLY))
        self._audit = audit
        self.state = S.PATHS
        self.results: dict[SetupState, StepResult] = {}
        self.history: list[SetupState] = []
        self.paths: dict[str, str] = {}
        self.confirmed = False
        # True once ``save_config`` succeeded: the configuration is on disk whatever follows.
        self.committed = False
        self.blocked_from: SetupState | None = None
        self.reason_code = ReasonCode.OK
        self.next_action = ""
        self._paths_answer: Mapping[str, str] | None = None
        self._manual: str | None = None
        # Provisioning target for this machine: the auto suggestion or the operator's pick.
        self._choice = "auto"
        # Only an explicit operator answer at MODEL_SELECT makes the saved model non-auto.
        self._chosen_by_operator = False
        self._suggested: str | None = None
        self._produced_for: dict[SetupState, str] = {}

    # ------------------------------------------------------------------ properties
    @property
    def failed(self) -> bool:
        return self.state is S.BLOCKED and self.blocked_from in _FAILURE_STATES

    @property
    def exit_code(self) -> int:
        if self.state is S.DONE:
            return EXIT_OK
        if self.state is S.BLOCKED:
            return EXIT_FAILED if self.failed else EXIT_BLOCKED
        if self.state is S.CANCELLED:
            return EXIT_BLOCKED
        return EXIT_INVALID

    def _record(self, action: str, target: str, reason: ReasonCode = ReasonCode.OK) -> None:
        if self._audit is not None:
            self._audit(action, target, reason)

    # ------------------------------------------------------------------ transitions
    def next(self, answer: object = None) -> SetupState:
        state = self.state
        if state in _TERMINAL:
            return state
        if state is S.BLOCKED:
            return self._unblock(answer)
        if state is S.PATHS:
            if not isinstance(answer, Mapping) or set(answer) - set(_PATH_KEYS):
                raise SetupError(ReasonCode.INVALID_INPUT)
            return self._run_paths(dict(answer))
        if state is S.CONFIRM:
            if answer is not True and answer is not False:
                raise SetupError(ReasonCode.INVALID_INPUT)
            if answer is False:
                return self.cancel()
            self.confirmed = True
            self._record("setup_apply", "plan", ReasonCode.OK)
        elif state is S.SAVE_CONFIG:
            result = self._call(state, self._work_save)
            if result is None:
                return self.state
            self.results[state] = result
            self.committed = True
        elif state is S.MODEL_SELECT:
            if not self._select(answer):
                return self.state
        return self._advance()

    def back(self) -> bool:
        if self.state in _TERMINAL:
            return False
        if self.state is S.BLOCKED:
            self.state, self.blocked_from = self.blocked_from or S.PATHS, None
            self.reason_code, self.next_action = ReasonCode.OK, ""
        if not self.history:
            return False
        left = self.state
        if left not in _PERSISTED:
            self.results.pop(left, None)
        self.state = self.history.pop()
        if self.order.index(self.state) < _CONFIRM_INDEX and self.confirmed:
            self.confirmed = False
            for key in _PERSISTED:
                self.results.pop(key, None)
            # Provisioning may have opened the bound cache (MODEL_ACQUIRE, SMOKE).  Any
            # check from here on samples that root as a plain directory (held open, the
            # Windows sample is storage_unknown), so the port releases it first.
            self._port_call("release_cache")
        return True

    def recheck(self) -> SetupState:
        if self.state is S.BLOCKED:
            return self._unblock(None)
        if self.state in _TERMINAL or self.state in _PERSISTED or self.state is S.CONFIRM:
            return self.state
        if self.state is S.PATHS:
            return self._run_paths(dict(self._paths_answer)) if self._paths_answer is not None else self.state
        self.results.pop(self.state, None)
        self._enter(self.state)
        return self.state

    def cancel(self) -> SetupState:
        if self.state not in _TERMINAL:
            self.state, self.blocked_from = S.CANCELLED, None
            self.reason_code = ReasonCode.CANCELLED
            self._record("setup_cancel", "setup", ReasonCode.CANCELLED)
        return self.state

    # ------------------------------------------------------------------ internals
    def _unblock(self, answer: object) -> SetupState:
        origin = self.blocked_from or S.PATHS
        # The old failure is stale from here: a re-run that fails again blocks afresh.
        self.reason_code, self.next_action = ReasonCode.OK, ""
        if origin is S.PATHS:
            return self._run_paths(dict(answer) if isinstance(answer, Mapping) else dict(self._paths_answer or {}))
        if origin is S.SAVE_CONFIG:
            self.state, self.blocked_from = S.SAVE_CONFIG, None
            return self.state
        if origin is S.MODEL_SELECT and answer is not None and origin in self.results:
            # Blocked by the choice, not the listing: the offered models stand, so the
            # operator's answer is the pick itself rather than a request to re-list.
            self.state, self.blocked_from = S.MODEL_SELECT, None
            return self.next(answer)
        if origin is S.BACKING and isinstance(answer, str) and answer:
            self._manual = answer
        self.results.pop(origin, None)
        self.blocked_from = None
        self._enter(origin)
        return self.state

    def _run_paths(self, answer: dict) -> SetupState:
        self._paths_answer = answer
        result = self._call(S.PATHS, lambda: self.actions.check_paths(answer))
        if result is None:
            return self.state
        self.results[S.PATHS] = result
        self.paths = dict(result.data.get("paths", answer))
        # Newly accepted paths retire any recovery override typed at an earlier BACKING
        # block: the backing check follows the new paths (or auto discovery) until a
        # later block supplies a fresh override.
        self._manual = None
        self.blocked_from = None
        self.state = S.PATHS
        # Provisioning follows the accepted paths (not the configuration the service
        # started with).  Reaching PATHS again went back() past CONFIRM, which already
        # dropped every provisioning result, so a new binding never serves a stale one.
        self._port_call("bind_paths", dict(self.paths))
        return self._advance()

    def _port_call(self, name: str, *args: object) -> None:
        """Call an optional provisioning-port method (``bind_paths``, ``release_cache``)."""
        method = getattr(self.actions.provision, name, None)
        if callable(method):
            method(*args)

    def _advance(self) -> SetupState:
        index = self.order.index(self.state)
        self.history.append(self.state)
        self._enter(self.order[index + 1])
        return self.state

    def _block(self, origin: SetupState, reason: ReasonCode, next_action: str) -> None:
        self.state, self.blocked_from = S.BLOCKED, origin
        self.reason_code, self.next_action = reason, next_action

    def _call(self, origin: SetupState, work: Callable[[], StepResult]) -> StepResult | None:
        try:
            result = work()
        except DomainValidationError as error:
            self._block(origin, error.reason_code, getattr(error, "next_action", ""))
            return None
        except Exception:  # noqa: BLE001 - closed reason only; never echo adapter text
            self._block(origin, ReasonCode.SERVICE_UNAVAILABLE, "retry after resolving the host problem")
            return None
        if not result.ok:
            self._block(origin, result.reason_code, result.next_action)
            return None
        return result

    def _enter(self, state: SetupState) -> None:
        self.state = state
        if state in self.results:
            return
        if state in _NEEDS_CONFIRM and not self.confirmed:
            raise SetupError(ReasonCode.CONFLICT, next_action="confirm the plan first")
        work = {
            S.HOST_CHECK: lambda: self.actions.host_capacity(self.paths),
            S.DOCKER_CHECK: lambda: self.actions.docker_status(),
            S.BACKING: self._work_backing,
            S.PLAN: lambda: self.actions.plan(tuple(sorted(self.capabilities)), self.paths),
            S.PULL: self._work_pull,
            S.HARDWARE: lambda: self.actions.provision.probe_hardware(),
            S.MODEL_SELECT: self._work_select,
            S.MODEL_CAPACITY: lambda: self.actions.provision.model_capacity(self._choice),
            S.MODEL_ACQUIRE: self._work_acquire,
            S.SMOKE: lambda: self.actions.provision.smoke(self._choice),
        }.get(state)
        if work is None:
            if state is S.DONE:
                # The save step's guidance (restart a running service, reset settings)
                # is the completion guidance; a lost completion audit replaces it.
                saved = self.results.get(S.SAVE_CONFIG)
                self.next_action = saved.next_action if saved is not None else ""
                self._record_done()
            return
        result = self._call(state, work)
        if result is not None:
            self.results[state] = result
            if state in _CHOICE_BOUND:
                self._produced_for[state] = self._choice

    def _record_done(self) -> None:
        """The completion audit runs after the configuration is committed.

        A lost record is a failure to report, never a reason to provision or save
        again: ``recheck`` retries only this record (same AUDIT_LOSS closure as save).
        """
        try:
            self._record("setup_done", "setup", ReasonCode.OK)
        except Exception:  # noqa: BLE001 - audit sink is untrusted; closed reason only
            self._block(S.DONE, ReasonCode.AUDIT_LOSS,
                        "configuration was saved; recheck to record the completion audit")

    def _select(self, answer: object) -> bool:
        """None or "auto" accepts the suggestion; a listed id is the operator's explicit pick.

        An explicit pick is never fit-checked (the operator decides); it must only be one
        of the offered (pinned) models.  Returns False when the machine blocked instead.
        """
        if answer is None or answer == "auto":
            if not self._suggested:
                self._block(S.MODEL_SELECT, ReasonCode.INVALID_INPUT,
                            "no model fits this machine automatically; choose a model id")
                return False
            self._chosen_by_operator = False
            self._set_choice(self._suggested)
            return True
        if type(answer) is not str or answer not in option_ids(self.results[S.MODEL_SELECT].data):
            raise SetupError(ReasonCode.INVALID_INPUT, next_action="type a listed model id, or Enter for auto")
        self._chosen_by_operator = True
        self._set_choice(answer)
        return True

    def _set_choice(self, choice: str) -> None:
        """A changed model invalidates every result produced for the previous one."""
        if choice != self._choice:
            for state in _CHOICE_BOUND:
                self.results.pop(state, None)
                self._produced_for.pop(state, None)
        self._choice = choice

    def _work_backing(self) -> StepResult:
        # A recovery override typed at a BACKING block wins; otherwise the operator's
        # paths.docker_backing (when given) is the Docker data location to verify.
        manual = self._manual or self.paths.get("docker_backing") or None
        result = self.actions.backing(manual)
        if result.ok and manual and self.paths.get("docker_backing") != manual:
            # The verified override is now one of the accepted paths: PLAN and the pull's
            # admission sample the same Docker data location BACKING verified.
            self.paths["docker_backing"] = manual
            self._port_call("bind_paths", dict(self.paths))
        return result

    def _work_pull(self) -> StepResult:
        pulled = []
        for capability in sorted(self.capabilities):
            result = self.actions.provision.pull(capability)
            if not result.ok:
                return result
            pulled.append(str(capability))
        return StepResult(True, data={"pulled": pulled})

    def _work_select(self) -> StepResult:
        result = self.actions.provision.choose_model(self.results[S.HARDWARE].data)
        if result.ok:
            self._suggested = suggested_id(result.data)
        return result

    def _work_acquire(self) -> StepResult:
        cached = self.actions.cached_snapshot(self._choice)
        if cached:
            return StepResult(True, data={"reused": True, "snapshot": cached})
        return self.actions.provision.acquire_model(self._choice)

    def config_data(self) -> dict:
        caps = self.capabilities
        device = "cuda" if Capability.ASR_CUDA in caps else "cpu" if Capability.ASR_CPU in caps else "auto"
        return {
            "paths": dict(self.paths),
            "selection": {"device": device,
                          "model": self._choice if self._chosen_by_operator and not self.video_only else "auto"},
            "outputs": {"transcript": not self.video_only, "video": self.video_only},
        }

    def _work_save(self) -> StepResult:
        if not self.confirmed:
            raise SetupError(ReasonCode.CONFLICT, next_action="confirm the plan first")
        required = [s for s in self.order if s in _PERSISTED or s in (S.MODEL_CAPACITY, S.PLAN)]
        if not all(s in self.results and self.results[s].ok for s in required):
            raise SetupError(ReasonCode.CONFLICT, next_action="complete every earlier step")
        if not self.video_only and any(self._produced_for.get(s) != self._choice for s in _CHOICE_BOUND):
            raise SetupError(ReasonCode.CONFLICT, next_action="complete every earlier step")
        return self.actions.save_config(self.config_data())


def _option_id(option: object) -> str | None:
    """Options are ``{"id": ...}`` mappings (older ports sent bare id strings)."""
    value = option.get("id") if isinstance(option, Mapping) else option
    return value if type(value) is str and value else None


def _options(data: Mapping[str, Any]) -> list:
    options = data.get("options")
    return list(options) if isinstance(options, (list, tuple)) else []


def option_ids(data: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(i for i in map(_option_id, _options(data)) if i is not None)


def suggested_id(data: Mapping[str, Any]) -> str | None:
    """The auto suggestion; ``None`` when nothing fits automatically."""
    choice = data.get("choice")
    return _option_id(choice) if choice is not None else None


# ------------------------------------------------------------------------ idempotent apply


@dataclass(frozen=True, slots=True)
class SetupResult:
    exit_code: int
    state: str
    reason_code: str
    next_action: str = ""
    # The configuration was written; True even when only the completion audit was lost.
    applied: bool = False


def drive(machine: SetupMachine, *, paths: Mapping[str, str], accept_plan: bool,
          manual_backing: str | None = None, model: str | None = None) -> SetupResult:
    """Run a machine to a stop using explicit answers only (never prompts)."""
    manual_pending = manual_backing
    while machine.state not in _TERMINAL:
        state = machine.state
        if state is S.BLOCKED:
            if machine.blocked_from is S.BACKING and manual_pending:
                answer, manual_pending = manual_pending, None
                machine.next(answer)
                continue
            break
        if state is S.PATHS:
            machine.next(paths)
        elif state is S.CONFIRM:
            machine.next(bool(accept_plan))
        elif state is S.MODEL_SELECT and model is not None:
            machine.next(model)
        else:
            machine.next()
    return SetupResult(machine.exit_code, str(machine.state), str(machine.reason_code),
                       machine.next_action, applied=machine.committed)


class SetupService:
    """apply(key): once a key committed its configuration the stored result is returned, no re-provisioning.

    A committed run whose completion audit was lost is stored as that failure: a retry
    reports the loss again instead of provisioning and saving a second time.
    """

    def __init__(self, actions: SetupActions, *, audit: AuditCallable | None = None,
                 store: dict[str, tuple[str, SetupResult]] | None = None) -> None:
        self.actions, self.audit = actions, audit
        self.store = store if store is not None else {}
        # The machine of the most recent apply(): lets a caller interrupted mid-run
        # (KeyboardInterrupt escapes drive) tell whether the configuration was committed.
        self.machine: SetupMachine | None = None

    def apply(self, key: str, *, capabilities: object, paths: Mapping[str, str], accept_plan: bool,
              manual_backing: str | None = None, model: str | None = None) -> SetupResult:
        if type(key) is not str or not key or len(key) > 128:
            raise SetupError(ReasonCode.INVALID_INPUT)
        fingerprint = hashlib.sha256(json.dumps(
            [sorted(str(c) for c in parse_capabilities(capabilities)), dict(paths), bool(accept_plan),
             manual_backing, model], sort_keys=True).encode()).hexdigest()
        stored = self.store.get(key)
        if stored is not None:
            if stored[0] != fingerprint:
                raise SetupError(ReasonCode.CONFLICT)
            return stored[1]
        machine = SetupMachine(self.actions, capabilities=capabilities, audit=self.audit)
        self.machine = machine
        result = drive(machine, paths=paths, accept_plan=accept_plan, manual_backing=manual_backing, model=model)
        if result.applied:
            self.store[key] = (fingerprint, result)
        return result


# ------------------------------------------------------------------- service-backed apply


class SetupApplier(Protocol):
    """Setup applied by the running service: provisioning (image pulls, the in-worker probe,
    model acquisition, the smoke) runs there, under the service's setup lease, Docker
    session and probe port.  ``service.ServiceClient`` is wrapped by :class:`ServiceSetup`.
    """

    def plan(self, capabilities: Sequence[str], config: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def apply(self, plan_id: str, key: str) -> Mapping[str, Any]: ...
    def wait(self, operation: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


class ServiceSetup:
    """``SetupApplier`` over the service API (``/v1/setup/plan``, ``/v1/setup/apply`` and
    ``/v1/setup/{operation_id}``), polling the operation until it leaves ``running``."""

    def __init__(self, client: Any, *, sleep: Callable[[float], None] | None = None,
                 poll_seconds: float = 1.0) -> None:
        import time

        self._client = client
        self._sleep = sleep if sleep is not None else time.sleep
        self._poll = float(poll_seconds)

    def plan(self, capabilities: Sequence[str], config: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(self._client.setup_plan(list(capabilities), dict(config))["plan"])

    def apply(self, plan_id: str, key: str) -> Mapping[str, Any]:
        return dict(self._client.setup_apply(plan_id, key)["operation"])

    def wait(self, operation: Mapping[str, Any]) -> Mapping[str, Any]:
        status = dict(operation)
        while status.get("state") == "running":
            self._sleep(self._poll)
            status = dict(self._client.setup_status(str(status["operation_id"])))
        return status

    def close(self) -> None:
        self._client.close()


def _remote_result(status: Mapping[str, Any]) -> SetupResult:
    """The service's operation status as the same ``SetupResult`` ``drive`` returns.

    ``setup_state`` and ``exit_code`` are the machine's own when the service ran it; an
    operation that failed before the machine ran (a refused lease, a lost audit) has
    neither and is a runtime failure with its closed reason.
    """
    done = status.get("state") == "done"
    reason = status.get("reason_code")
    reason = reason if type(reason) is str and reason else str(ReasonCode.OK if done else ReasonCode.SERVICE_UNAVAILABLE)
    state = status.get("setup_state")
    state = state if type(state) is str and state else str(S.DONE if done else S.BLOCKED)
    exit_code = status.get("exit_code")
    if type(exit_code) is not int:
        exit_code = EXIT_OK if done else EXIT_FAILED
    action = status.get("next_action")
    return SetupResult(exit_code, state, reason, action if type(action) is str else "", applied=done)


def apply_remote(applier: SetupApplier, *, capabilities: object, paths: Mapping[str, str],
                 accept_plan: bool, manual_backing: str | None = None, model: str | None = None,
                 key: str | None = None) -> SetupResult:
    """``drive`` through the service: plan, then (if accepted) apply and wait for the outcome.

    The service runs the machine with these explicit answers (``SetupService.apply``), so a
    declined plan is the same cancellation, and a block or failure carries the machine's
    state, reason and next action.  ``key`` is the apply's idempotency key; a fresh one per
    call unless the operator pins it.
    """
    plan = applier.plan(sorted(str(c) for c in parse_capabilities(capabilities)),
                        _remote_config(paths, manual_backing=manual_backing, model=model))
    if not accept_plan:
        return SetupResult(EXIT_BLOCKED, str(S.CANCELLED), str(ReasonCode.CANCELLED))
    return _apply_plan(applier, plan, key)


def _remote_config(paths: Mapping[str, str], *, manual_backing: str | None = None,
                   model: str | None = None) -> dict[str, Any]:
    """The service's setup config: only what was answered (it merges over its own paths)."""
    config: dict[str, Any] = {"paths": {k: v for k, v in paths.items() if type(v) is str}}
    if manual_backing is not None:
        config["manual_backing"] = manual_backing
    if model is not None:
        config["model"] = model
    return config


def _apply_plan(applier: SetupApplier, plan: Mapping[str, Any], key: str | None = None) -> SetupResult:
    import uuid

    operation = applier.apply(str(plan["plan_id"]), key or str(uuid.uuid4()))
    return _remote_result(applier.wait(operation))


def _closed_reason(error: BaseException) -> ReasonCode:
    reason = getattr(error, "reason_code", None)
    return reason if isinstance(reason, ReasonCode) else ReasonCode.SERVICE_UNAVAILABLE


def _exit_for(reason: ReasonCode) -> int:
    from .cli import exit_for_reason

    return exit_for_reason(reason)


def _report(result: SetupResult) -> str:
    return (f"setup: {result.state} ({result.reason_code})"
            + (f" - {result.next_action}" if result.next_action else ""))


# ------------------------------------------------------------------------ local (real) adapters


def _file_audit(path: Path) -> AuditCallable:
    def record(action: str, target_id: str, reason: ReasonCode) -> None:
        from datetime import datetime
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": datetime.now(UTC).isoformat(), "action": action,
                           "target_id": target_id, "reason_code": str(reason)}) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
    return record


class LocalConfigWriter:
    """Real AuditedConfigWriter: volume admission, injected audit, temp + os.replace."""

    def __init__(self, volumes: Any, audit: AuditCallable) -> None:
        self.volumes, self.audit_fn = volumes, audit

    def admit(self, target: Path) -> None:
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise SetupError(ReasonCode.STATE_WRITE_FAILED)
        evidence = self.volumes.sample(target.parent)
        if evidence.writable.value is not True or evidence.mounted.value is False:
            raise SetupError(ReasonCode.STORAGE_UNKNOWN)

    def audit(self, target: Path, payload: bytes) -> None:
        self.audit_fn("config_save", hashlib.sha256(payload).hexdigest()[:16], ReasonCode.OK)

    def atomic_replace(self, target: Path, payload: bytes) -> None:
        import tempfile
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if os.name == "posix":
                os.chmod(temp, 0o600)
            os.replace(temp, target)
        except BaseException:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise


def default_config_path() -> Path:
    import platformdirs
    return Path(platformdirs.user_config_dir("podmuncher", appauthor=False)) / "config.toml"


def _reason_of(error: BaseException, fallback: ReasonCode) -> ReasonCode:
    return error.reason_code if isinstance(error, DomainValidationError) else fallback


def _capacity_result(evidence: Mapping[str, Any]) -> StepResult:
    data: dict[str, Any] = {}
    reason = ReasonCode.OK
    for key, ev in evidence.items():
        data[key] = {"free_bytes": ev.free.value, "capacity_bytes": ev.capacity.value,
                     "writable": ev.writable.value}
        if ev.free.value is None or ev.writable.value is None:
            reason = ReasonCode.STORAGE_UNKNOWN
        elif ev.writable.value is not True or ev.free.value <= 0:
            reason = ReasonCode.STORAGE_INSUFFICIENT
    return StepResult(reason is ReasonCode.OK, reason, data,
                      "free space or choose another location" if reason is not ReasonCode.OK else "")


def _destination_costs(sizes: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Sum every capability's positive byte counts onto the volume role each one lands on."""
    costs: dict[str, int] = {}
    for size in sizes:
        for key, destination in _COST_DESTINATIONS.items():
            value = size.get(key)
            if type(value) is int and value > 0:
                costs[destination] = costs.get(destination, 0) + value
    return costs


def _volume_budgets(volumes: Iterable[tuple[str, str | None, int | None]],
                    costs: Mapping[str, int]) -> list[dict[str, Any]]:
    """Group (role, identity, free) samples by volume identity and budget each group.

    Aliases of one volume sum their roles' costs against the smallest free measurement
    among them (every alias's constraint applies).  Unknown identities stay separate, and
    an unknown free space reports ``fits: None`` instead of a verdict.
    """
    groups: dict[object, dict[str, Any]] = {}
    for index, (role, identity, free) in enumerate(volumes):
        key: object = identity if type(identity) is str and identity else ("unknown", index)
        group = groups.setdefault(key, {"roles": [], "free_bytes": None, "cost_bytes": 0, "fits": None})
        group["roles"].append(role)
        group["cost_bytes"] += int(costs.get(role, 0))
        if type(free) is int and (group["free_bytes"] is None or free < group["free_bytes"]):
            group["free_bytes"] = free
    for group in groups.values():
        group["roles"] = sorted(group["roles"])
        if group["free_bytes"] is not None:
            group["fits"] = group["cost_bytes"] <= group["free_bytes"]
    return sorted(groups.values(), key=lambda g: g["roles"])


def _shared_directories(settings_path: Path) -> tuple[Path, ...] | None:
    """Docker Desktop's ``FilesharingDirectories`` (bounded, read-only); None when unknown."""
    try:
        with settings_path.open("rb") as stream:
            raw = stream.read(_MAX_SETTINGS_BYTES + 1)
        if len(raw) > _MAX_SETTINGS_BYTES:
            return None
        settings = json.loads(raw)
    except (OSError, ValueError, UnicodeError, RecursionError):
        # RecursionError: deeply nested JSON under the byte cap; the advisory stays silent.
        return None
    listed = settings.get("FilesharingDirectories") if type(settings) is dict else None
    if (type(listed) is not list or len(listed) > _MAX_SHARED_DIRECTORIES
            or not all(type(d) is str and "\x00" not in d and (d.startswith("/") or Path(d).is_absolute())
                       for d in listed)):
        return None
    shared: list[Path] = []
    for entry in listed:
        path = Path(entry)
        shared.append(path)
        try:
            resolved = path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved != path:
            shared.append(resolved)
    return tuple(shared)


def _within_any(path: Path, roots: Iterable[Path]) -> bool:
    parents = set(path.parents)
    return any(root == path or root in parents for root in roots)


def shared_folder_warnings(values: Mapping[str, str], *, platform: str,
                           settings_path: Path | None = None) -> list[str]:
    """Darwin only: name each output/cache/scratch path outside Docker Desktop's shared folders.

    Bind mounts outside ``FilesharingDirectories`` fail at run time, so the paths step says
    so with the fix; it never blocks (the operator decides).  The default platformdirs
    locations under /Users are normally shared.  An unreadable settings file warns nothing.
    """
    if not platform.startswith("darwin"):
        return []
    shared = _shared_directories(settings_path if settings_path is not None else Path.home() / _DOCKER_SETTINGS)
    if not shared:
        return []
    warnings: list[str] = []
    for key in _SHARED_PATH_KEYS:
        value = values.get(key)
        if value and not _within_any(Path(value), shared):
            warnings.append(f"{key} path {value} is outside Docker Desktop's shared folders; add it in "
                            "Docker Desktop > Settings > Resources > File sharing, or choose a path under "
                            "one of them")
    return warnings


class LocalSetupActions:
    """Read-only checks through the real adapters plus audited config persistence."""

    def __init__(self, *, provisioning: ProvisioningPort | None = None, config_path: str | Path | None = None,
                 audit: AuditCallable | None = None, volumes: Any = None, docker: Any = None,
                 snapshot_source: Callable[[str], str | None] | None = None,
                 sizes: Mapping[str, Mapping[str, int]] | None = None,
                 platform: str | None = None, docker_settings: str | Path | None = None) -> None:
        self.provision = provisioning if provisioning is not None else UnboundProvisioning()
        self.config_path = Path(config_path) if config_path is not None else default_config_path()
        self.audit = audit if audit is not None else _file_audit(self.config_path.parent / "setup-audit.log")
        self._volumes, self._docker = volumes, docker
        self._snapshot_source = snapshot_source
        self._sizes = dict(sizes or {})
        self._platform = platform if platform is not None else sys.platform
        # Docker Desktop's settings file (shared folders); None means the per-user default.
        self._docker_settings = Path(docker_settings) if docker_settings is not None else None

    @property
    def volumes(self) -> Any:
        if self._volumes is None:
            from .volumes import make_volume_provider
            self._volumes = make_volume_provider()
        return self._volumes

    @property
    def docker(self) -> Any:
        if self._docker is None:
            from .docker_storage import make_docker_storage_provider
            self._docker = make_docker_storage_provider()
        return self._docker

    def check_paths(self, paths: Mapping[str, str]) -> StepResult:
        from .config import Config, ConfigError
        try:
            resolved = Config.from_dict({"paths": {k: v for k, v in paths.items() if v != "" or k == "docker_backing"}}).paths
        except ConfigError as error:
            return StepResult(False, error.reason_code, next_action="enter absolute, valid paths")
        values = {k: getattr(resolved, k) for k in _PATH_KEYS}
        for key in ("data", "output", "cache", "scratch"):
            if Path(values[key]).exists() and not Path(values[key]).is_dir():
                return StepResult(False, ReasonCode.INVALID_INPUT, next_action=f"{key} path is not a directory")
        data: dict[str, Any] = {"paths": values}
        warnings = shared_folder_warnings(values, platform=self._platform, settings_path=self._docker_settings)
        if warnings:
            data["warnings"] = warnings
        return StepResult(True, data=data)

    def _host_evidence(self, paths: Mapping[str, str]) -> dict[str, Any]:
        """One volume sample per configured host path, in path-key order (may raise)."""
        keys = [k for k in ("data", "output", "cache", "scratch") if paths.get(k)]
        return dict(zip(keys, self.volumes.sample_many(tuple(paths[k] for k in keys))))

    def host_capacity(self, paths: Mapping[str, str]) -> StepResult:
        try:
            evidence = self._host_evidence(paths)
        except Exception as error:  # noqa: BLE001
            return StepResult(False, _reason_of(error, ReasonCode.STORAGE_UNKNOWN))
        return _capacity_result(evidence)

    def _sample_docker(self, manual: str = "") -> Any:
        return self.docker.sample(manual_backing=manual)

    def _docker_volumes(self, manual: str = "") -> list[tuple[str, str | None, int | None]]:
        """Every Docker store (VM data root, host backing, containerd) as (role, identity, free).

        A pull must fit each store Docker writes through, so every constraint carries the
        full docker cost; one sharing a host path's volume identity sums with it in the
        budget.  ``manual`` is the operator's verified Docker backing, the same value
        BACKING checked.  BACKING already admitted Docker; here only its free space
        matters, so an unreadable sample yields one unknown-free entry that reports rather
        than blocks.
        """
        try:
            evidence = self._sample_docker(manual)
        except Exception:  # noqa: BLE001 - closed handling; the budget shows free as unknown
            return [("docker", None, None)]
        found: list[tuple[str, str | None, int | None]] = []
        for constraint in getattr(evidence, "constraints", None) or ():
            volume = constraint.volume
            found.append(("docker", volume.identity.value, volume.free.value))
        return found or [("docker", None, None)]

    def docker_status(self) -> StepResult:
        try:
            ev = self._sample_docker()
        except Exception as error:  # noqa: BLE001
            return StepResult(False, _reason_of(error, ReasonCode.DOCKER_UNAVAILABLE),
                              next_action="install or start Docker Desktop yourself; setup never installs it")
        if ev.context.value is None:
            return StepResult(False, ev.context.reason or ReasonCode.DOCKER_UNAVAILABLE,
                              next_action="install or start Docker Desktop yourself; setup never installs it")
        return StepResult(True, data={"context": ev.context.value, "desktop_version": ev.desktop_version.value})

    def backing(self, manual: str | None = None) -> StepResult:
        try:
            ev = self._sample_docker(manual or "")
        except Exception as error:  # noqa: BLE001
            return StepResult(False, _reason_of(error, ReasonCode.BACKING_UNRESOLVED),
                              next_action="enter the existing Docker data location to verify its association")
        if not ev.can_pull:
            return StepResult(False, ev.reason or ReasonCode.BACKING_UNRESOLVED,
                              next_action="enter the existing Docker data location to verify its association")
        return StepResult(True, data={"backing_associations": len(ev.backing), "manual": bool(manual)})

    def plan(self, capabilities: tuple[str, ...], paths: Mapping[str, str]) -> StepResult:
        """Budget every provisioning cost on the volume it lands on, one volume at a time.

        ``data["volumes"]`` lists each distinct volume with the roles on it, its free space
        and the summed cost; a volume shared by several paths carries all their costs.
        """
        sizes = {c: dict(self._sizes.get(str(c), {})) for c in capabilities}
        costs = _destination_costs(sizes.values())
        data: dict[str, Any] = {"capabilities": list(map(str, capabilities)), "sizes": sizes,
                                "size_known": any(bool(s) for s in sizes.values())}
        if not costs:
            return StepResult(True, ReasonCode.OK, data)
        try:
            evidence = self._host_evidence(paths)
        except Exception as error:  # noqa: BLE001 - closed reason; the host check names the detail
            return StepResult(False, _reason_of(error, ReasonCode.STORAGE_UNKNOWN), data, "free space before applying")
        capacity = _capacity_result(evidence)
        if not capacity.ok:
            # storage_unknown when a host path's free space turned unknown since HOST_CHECK.
            return StepResult(False, capacity.reason_code, data, "free space before applying")
        volumes = [(key, ev.identity.value, ev.free.value) for key, ev in evidence.items()]
        if costs.get("docker"):
            # The operator's verified backing (paths.docker_backing, set by BACKING) drives the sample.
            volumes.extend(self._docker_volumes(paths.get("docker_backing") or ""))
        budgets = _volume_budgets(volumes, costs)
        data["volumes"] = budgets
        short = [b for b in budgets if b["fits"] is False]
        if short:
            detail = "; ".join(f"the volume holding {', '.join(b['roles'])} needs "
                               f"{(b['cost_bytes'] - b['free_bytes']) / _GIB:.2f} GiB more" for b in short)
            return StepResult(False, ReasonCode.STORAGE_INSUFFICIENT, data, f"free space before applying: {detail}")
        return StepResult(True, ReasonCode.OK, data)

    def cached_snapshot(self, choice: str) -> str | None:
        return self._snapshot_source(choice) if self._snapshot_source is not None else None

    def _existing_config(self) -> tuple[Any, str]:
        """``(config, state)``: the configuration at ``config_path`` (state ``merged``), or
        the defaults when the file is ``missing`` or ``invalid``.

        ``invalid`` means a file exists but cannot be read or validated: setup is being
        re-run to fix things, so the defaults stand in and the save goes ahead.  A pinned
        model or language is trusted as written (the registry is not known here); the
        wizard's own values replace them anyway.
        """
        from .config import Config, ConfigError
        try:
            raw = self.config_path.read_bytes()
        except FileNotFoundError:
            return Config.from_dict(None), "missing"
        except OSError:
            return Config.from_dict(None), "invalid"
        try:
            if len(raw) > _MAX_CONFIG_BYTES:
                raise ConfigError()
            data = tomllib.loads(raw.decode("utf-8"))
            selection = data.get("selection") if type(data) is dict else None
            pinned = {key: selection.get(key) for key in ("model", "language")} if type(selection) is dict else {}
            registries = {f"verified_{key}s": frozenset({value}) for key, value in pinned.items()
                          if type(value) is str and value and value != "auto"}
            return Config.from_dict(data, **registries), "merged"
        except (ConfigError, UnicodeError, ValueError, RecursionError):
            return Config.from_dict(None), "invalid"

    def save_config(self, config: Mapping[str, Any]) -> StepResult:
        """Layer the wizard's values over the existing configuration and persist the result.

        Only the keys the wizard sets change; ``tls.ca_bundle``, telemetry, limits and every
        other setting the wizard never asks about survive a re-run.
        """
        from .config import Config, ConfigError
        model = config.get("selection", {}).get("model", "auto")
        try:
            existing, state = self._existing_config()
            language = existing.selection.language
            built = Config.from_dict(existing.to_dict(), overrides=dict(config),
                                     verified_models=frozenset({model}) if model != "auto" else None,
                                     verified_languages=frozenset({language}) if language != "auto" else None)
            built.persist(self.config_path, writer=LocalConfigWriter(self.volumes, self.audit))
        except ConfigError as error:
            return StepResult(False, error.reason_code, next_action="no configuration was written")
        guidance = []
        if state == "invalid":
            guidance.append("the existing configuration could not be read; settings this setup "
                            "does not ask about were reset to their defaults")
        changed = built.to_dict() != existing.to_dict()
        if changed:
            guidance.append("a running service keeps the configuration it started with; "
                            "restart it to use the saved one")
        return StepResult(True, data={"saved": True, "existing": state, "changed": changed},
                          next_action="; ".join(guidance))


# ------------------------------------------------------------------------ doctor (read only)


def _status_for(reason: ReasonCode) -> str:
    return "unavailable" if reason in (ReasonCode.DOCKER_UNAVAILABLE, ReasonCode.DOCKER_CONTEXT_UNSUPPORTED) else \
        "unknown" if reason in (ReasonCode.STORAGE_UNKNOWN, ReasonCode.BACKING_UNRESOLVED) else "blocked"


def _check(result: StepResult) -> dict:
    return {"status": "ok" if result.ok else _status_for(result.reason_code),
            "reason_code": str(result.reason_code), "detail": dict(result.data)}


def _cache_check(cache: str) -> dict:
    """Read-only cache look: top-level entry count only; never opens or verifies a store."""
    try:
        path = Path(cache)
        count = sum(1 for _ in path.iterdir()) if path.is_dir() else 0
    except OSError:
        return {"status": "unknown", "reason_code": str(ReasonCode.CACHE_UNKNOWN), "detail": {}}
    return {"status": "ok", "reason_code": str(ReasonCode.OK), "detail": {"present": count > 0, "entries": count}}


def doctor_report(paths: Mapping[str, str] | None = None, *, actions: LocalSetupActions | None = None) -> dict:
    """Strictly read-only: no start, no pull, no download, no writes."""
    actions = actions if actions is not None else LocalSetupActions()
    checks: dict[str, dict] = {}
    path_result = actions.check_paths(dict(paths or {}))
    checks["paths"] = _check(path_result)
    resolved = dict(path_result.data.get("paths", {})) if path_result.ok else {}
    checks["host_capacity"] = _check(actions.host_capacity(resolved)) if resolved else \
        {"status": "blocked", "reason_code": str(ReasonCode.INVALID_INPUT), "detail": {}}
    checks["docker"] = _check(actions.docker_status())
    checks["backing"] = _check(actions.backing(resolved.get("docker_backing") or None))
    checks["cache"] = _cache_check(resolved["cache"]) if resolved else \
        {"status": "unknown", "reason_code": str(ReasonCode.CACHE_UNKNOWN), "detail": {}}
    overall = "ok" if all(c["status"] == "ok" for c in checks.values()) else "blocked"
    return {"read_only": True, "status": overall, "checks": checks}


# ------------------------------------------------------------------------ interaction


def load_answers(path: str) -> dict:
    try:
        raw = Path(path).read_bytes()
        if len(raw) > _MAX_ANSWERS_BYTES:
            raise SetupError(ReasonCode.REQUEST_TOO_LARGE)
        data = json.loads(raw)
    except (OSError, ValueError):
        raise SetupError(ReasonCode.INVALID_INPUT) from None
    allowed = {"paths", "capabilities", "accept_plan", "manual_backing", "idempotency_key", "model"}
    if (type(data) is not dict or not data.keys() <= allowed or type(data.get("accept_plan")) is not bool
            or type(data.get("paths")) is not dict or "capabilities" not in data
            or set(data["paths"]) - set(_PATH_KEYS) or any(type(v) is not str for v in data["paths"].values())
            or any(k in data and type(data[k]) is not str for k in ("manual_backing", "idempotency_key", "model"))):
        raise SetupError(ReasonCode.INVALID_INPUT)
    parse_capabilities(data["capabilities"])
    return data


def _run_noninteractive_remote(answers: Mapping[str, Any], applier: SetupApplier, *,
                               out: Callable[[str], None]) -> int:
    """The answers file applied by the service; the same report lines as the local run."""
    try:
        result = apply_remote(applier, capabilities=answers["capabilities"], paths=answers["paths"],
                              accept_plan=answers["accept_plan"], manual_backing=answers.get("manual_backing"),
                              model=answers.get("model"), key=answers.get("idempotency_key"))
    except KeyboardInterrupt:
        out("setup: interrupted; a started setup continues in the service")
        return EXIT_INTERRUPTED
    except Exception as error:  # noqa: BLE001 - closed reason only; the service's text is not echoed
        reason = _closed_reason(error)
        code = _exit_for(reason)
        out(f"setup: {'invalid' if code == EXIT_INVALID else 'failed'} ({reason})")
        return code
    finally:
        applier.close()
    out(_report(result))
    return result.exit_code


def _nvidia_gpu_present() -> bool:
    """True when ``nvidia-smi`` lists a GPU on this host (never raises)."""
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        result = subprocess.run(("nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"),
                                capture_output=True, timeout=15, check=False,
                                stdin=subprocess.DEVNULL, **flags)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip().startswith(b"GPU-")


def default_output_dir() -> str:
    """Where transcripts go unless the user picks a folder: ``Podmuncher`` in their home folder."""
    return str(Path.home() / "Podmuncher")


def auto_answers(*, device: str | None = None, output: str | None = None, model: str | None = None,
                 gpu: bool | None = None) -> dict:
    """The best settings for this machine, with optional overrides from the command line.

    With an NVIDIA GPU, transcription runs on the GPU and the CPU image is kept as the
    fallback; without one (or with ``device="cpu"``), on the CPU.  Transcripts go to
    ``output`` or :func:`default_output_dir`; the other paths keep their defaults.  The model
    is chosen automatically unless named.
    """
    gpu = _nvidia_gpu_present() if gpu is None else gpu
    use_gpu = device == "cuda" or (device in (None, "auto") and gpu)
    answers: dict[str, Any] = {"capabilities": ["asr_cuda", "asr_cpu"] if use_gpu else ["asr_cpu"],
                               "paths": {"output": output or default_output_dir()}, "accept_plan": True}
    if model and model != "auto":
        answers["model"] = model
    return answers


def run_noninteractive(answers_path: str, actions: SetupActions | None = None, *,
                       out: Callable[[str], None] = print,
                       applier: Callable[[], SetupApplier] | None = None) -> int:
    """Apply an answers file: through the service when ``applier`` (a factory, called only
    now that the answers are valid) is given, else locally through ``actions``."""
    try:
        answers = load_answers(answers_path)
    except DomainValidationError as error:
        out(f"setup: invalid answers file ({error.reason_code})")
        return EXIT_INVALID
    return run_answers(answers, actions, out=out, applier=applier)


def run_answers(answers: Mapping[str, Any], actions: SetupActions | None = None, *,
                out: Callable[[str], None] = print,
                applier: Callable[[], SetupApplier] | None = None) -> int:
    """Apply validated answers (from a file or :func:`auto_answers`)."""
    if applier is not None:
        return _run_noninteractive_remote(answers, applier(), out=out)
    actions = actions if actions is not None else LocalSetupActions()
    audit = getattr(actions, "audit", None)
    key = answers.get("idempotency_key") or hashlib.sha256(json.dumps(answers, sort_keys=True).encode()).hexdigest()[:32]
    service = SetupService(actions, audit=audit)
    try:
        result = service.apply(key, capabilities=answers["capabilities"], paths=answers["paths"],
                               accept_plan=answers["accept_plan"], manual_backing=answers.get("manual_backing"),
                               model=answers.get("model"))
    except KeyboardInterrupt:
        if service.machine is not None and service.machine.committed:
            out("setup: interrupted; the configuration was already saved")
        else:
            out("setup: interrupted; no configuration written")
        return EXIT_INTERRUPTED
    except DomainValidationError as error:
        out(f"setup: invalid ({error.reason_code})")
        return EXIT_INVALID
    out(f"setup: {result.state} ({result.reason_code})" + (f" - {result.next_action}" if result.next_action else ""))
    return result.exit_code


def parse_paths_line(line: str) -> dict[str, str]:
    """Parse ``key=value`` pairs whose values may contain spaces; split only before known keys."""
    text = line.strip()
    if not text:
        return {}
    parsed: dict[str, str] = {}
    for segment in _PATH_SPLIT.split(text):
        key, sep, value = segment.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not sep or key not in _PATH_KEYS or key in parsed or not value:
            raise SetupError(ReasonCode.INVALID_INPUT)
        parsed[key] = value
    return parsed


_COMMANDS = ("back", "recheck", "cancel")
_GIB = 1024 ** 3


def _safe(value: object) -> str:
    """Escape untrusted text for the terminal (control characters, non-ASCII)."""
    return json.dumps(str(value), ensure_ascii=True)[1:-1]


def _requirement(option: Mapping[str, Any]) -> str:
    for key, label in (("peak_vram_bytes", "GPU"), ("peak_ram_bytes", "RAM")):
        if key in option:
            value = option.get(key)
            if type(value) is int and value > 0:
                return f"{label} {value / _GIB:.2f} GiB"
            return f"{label} unknown"
    return "requirement unknown"


def model_choice_lines(data: Mapping[str, Any]) -> list[str]:
    """The MODEL_SELECT guide: every offered model with its table numbers and source."""
    suggested = suggested_id(data)
    lines = ["[model_select] models for this machine (highest rank first; * = suggested):"]
    for option in _options(data):
        model_id = _option_id(option)
        if model_id is None:
            continue
        row = option if isinstance(option, Mapping) else {}
        fits = row.get("fits")
        fit_text = "yes" if fits is True else "no" if fits is False else "unknown"
        mark = "*" if model_id == suggested else " "
        lines.append(f"  {mark} {_safe(model_id)}: {_requirement(row)}, fits {fit_text}, "
                     f"basis {_safe(row.get('basis', 'unknown'))}, source {_safe(row.get('source', 'unknown'))}")
    if suggested:
        lines.append(f"Enter keeps model auto: each job re-picks the largest model that fits "
                     f"(here: {_safe(suggested)}). Type a model id to pin it.")
    else:
        lines.append("No model fits automatically; type a model id to choose one.")
    return lines


def _fit_note(data: Mapping[str, Any], model_id: str) -> str | None:
    for option in _options(data):
        if isinstance(option, Mapping) and _option_id(option) == model_id and option.get("fits") is not True:
            return (f"note: {_safe(model_id)} is not known to fit this machine's memory; "
                    "it will be used as chosen")
    return None

_CHOICES = (("video only", ("media",)), ("transcripts on CPU", ("media", "asr_cpu")),
            ("transcripts on NVIDIA GPU", ("media", "asr_cuda")))


def _ask_choice(ask: Callable[[str], str], out: Callable[[str], None]) -> tuple[str, ...] | None:
    """The capability choice, or None when the operator cancels."""
    out("Choose what to set up:")
    for number, (label, _) in enumerate(_CHOICES, 1):
        out(f"  {number}. {label}")
    while True:
        pick = ask("number (cancel to quit): ").strip()
        if pick == "cancel":
            return None
        if pick.isdigit() and 1 <= int(pick) <= len(_CHOICES):
            return _CHOICES[int(pick) - 1][1]


_REMOTE_PATHS_PROMPT = ("paths as key=value (data,output,cache,scratch,docker_backing) or blank for "
                        "the service's current paths [cancel]: ")


def _run_interactive_remote(capabilities: tuple[str, ...], applier: SetupApplier, actions: SetupActions, *,
                            ask: Callable[[str], str], out: Callable[[str], None]) -> int:
    """Prompts answered here, the machine run by the service (``apply_remote``).

    Paths are checked here for shape only (``check_paths``, read-only); the service merges
    them over its current configuration and runs every check and provisioning step.  A
    block or failure is reported with the machine's next action and the operator may
    answer again (a Docker data location after a backing block, a model id when nothing
    fits automatically); every answer set is a fresh plan and apply.
    """
    started = False
    try:
        while True:
            while True:
                line = ask(_REMOTE_PATHS_PROMPT).strip()
                if line == "cancel":
                    return EXIT_BLOCKED
                try:
                    paths = parse_paths_line(line)
                    checked = actions.check_paths(paths)
                except DomainValidationError as error:
                    checked = StepResult(False, error.reason_code, next_action=getattr(error, "next_action", ""))
                if checked.ok:
                    for warning in checked.data.get("warnings") or ():
                        out(f"note: {_safe(warning)}")
                    break
                out(f"invalid input ({checked.reason_code})" + (f"; {checked.next_action}" if checked.next_action else ""))
            model = ask("model id to pin, or Enter for auto [cancel]: ").strip()
            if model == "cancel":
                return EXIT_BLOCKED
            plan = applier.plan(sorted(str(c) for c in parse_capabilities(capabilities)),
                                _remote_config(paths, model=model if model and model != "auto" else None))
            out("[plan] " + json.dumps({k: plan.get(k) for k in ("capabilities", "size_known", "sizes",
                                                                 "ready", "reason_code", "next_action")},
                                       default=str))
            if ask("apply this plan and download what it lists? yes/no [cancel]: ").strip().lower() not in ("y", "yes"):
                out(f"setup: {S.CANCELLED} ({ReasonCode.CANCELLED})")
                return EXIT_BLOCKED
            started = True
            result = _apply_plan(applier, plan)
            out(_report(result))
            if result.applied:
                return result.exit_code
            if ask("answer again? yes/no: ").strip().lower() not in ("y", "yes"):
                return result.exit_code
    except KeyboardInterrupt:
        out("setup interrupted; a started setup continues in the service" if started
            else "setup interrupted; no configuration written")
        return EXIT_INTERRUPTED
    except Exception as error:  # noqa: BLE001 - closed reason only; the service's text is not echoed
        reason = _closed_reason(error)
        out(f"setup: failed ({reason})")
        return _exit_for(reason)
    finally:
        applier.close()


def run_interactive(actions: SetupActions | None = None, *, ask: Callable[[str], str] = input,
                    out: Callable[[str], None] = print,
                    applier: Callable[[], SetupApplier] | None = None) -> int:
    """Plain sequential prompts; every prompt accepts back, recheck or cancel.

    With ``applier`` (a factory, called once a choice is made) the service applies the
    setup: see ``_run_interactive_remote``.
    """
    actions = actions if actions is not None else LocalSetupActions()
    machine: SetupMachine | None = None
    try:
        capabilities = _ask_choice(ask, out)
        if capabilities is None:
            return EXIT_BLOCKED
        if applier is not None:
            return _run_interactive_remote(capabilities, applier(), actions, ask=ask, out=out)
        machine = SetupMachine(actions, capabilities=capabilities, audit=getattr(actions, "audit", None))
        while machine.state not in _TERMINAL:
            state = machine.state
            if state is S.BLOCKED:
                out(f"blocked at {machine.blocked_from}: {machine.reason_code}; {machine.next_action}")
            elif state is S.MODEL_SELECT and state in machine.results:
                for text in model_choice_lines(machine.results[state].data):
                    out(text)
            else:
                out(f"[{state}] " + json.dumps(dict(machine.results[state].data), default=str)
                    if state in machine.results else f"[{state}]")
            # Blocked at the model choice, the answer is still a model id (the list is on record).
            choosing = state is S.MODEL_SELECT or (state is S.BLOCKED and machine.blocked_from is S.MODEL_SELECT)
            prompt = {S.PATHS: "paths as key=value (data,output,cache,scratch) or blank for defaults",
                      S.CONFIRM: "apply this plan and download what it lists? yes/no",
                      S.MODEL_SELECT: "model id, or Enter for auto",
                      S.BLOCKED: "existing Docker data path (backing), or recheck/back/cancel"}.get(state, "Enter to continue")
            if choosing and state is S.BLOCKED:
                prompt = "model id to choose one, or recheck/back/cancel"
            line = ask(f"{prompt} [back/recheck/cancel]: ").strip()
            if line in _COMMANDS:
                {"back": machine.back, "recheck": machine.recheck, "cancel": machine.cancel}[line]()
                continue
            try:
                if state is S.PATHS:
                    machine.next(parse_paths_line(line))
                elif state is S.CONFIRM:
                    machine.next(line.lower() in ("y", "yes"))
                elif choosing:
                    data = machine.results[S.MODEL_SELECT].data if S.MODEL_SELECT in machine.results else {}
                    note = _fit_note(data, line) if line and line != "auto" else None
                    machine.next(line or None)
                    if note and machine.state is not S.MODEL_SELECT:
                        out(note)
                elif state is S.BLOCKED:
                    machine.next(line or None)
                else:
                    machine.next()
            except DomainValidationError as error:
                hint = getattr(error, "next_action", "")
                out(f"invalid input ({error.reason_code})" + (f"; {hint}" if hint else ""))
    except KeyboardInterrupt:
        if machine is not None:
            machine.cancel()
        if machine is not None and machine.committed:
            out("setup interrupted; the configuration was already saved")
        else:
            out("setup interrupted; no configuration written")
        return EXIT_INTERRUPTED
    out(f"setup: {machine.state}" + (f" ({machine.reason_code})" if machine.state is not S.DONE else ""))
    return machine.exit_code


def _is_tty() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def main(argv: list[str] | None = None, *, actions: SetupActions | None = None,
         environ: Mapping[str, str] | None = None, interactive: bool | None = None,
         applier: Callable[[], SetupApplier] | None = None,
         auto: Mapping[str, Any] | None = None) -> int:
    """cli entry. Never blocks on hidden input.

    ``applier`` is the CLI's service binding (a factory for :class:`ServiceSetup`): the
    service is started or attached only once setup is going to run, never for a refusal.
    ``auto`` (from :func:`auto_answers`) applies the best settings without any question.
    """
    env = os.environ if environ is None else environ
    answers = env.get(ANSWERS_ENV, "")
    if answers:
        return run_noninteractive(answers, actions, applier=applier)
    if auto is not None:
        where = "the NVIDIA GPU (CPU as fallback)" if "asr_cuda" in auto["capabilities"] else "the CPU"
        print(f"Setting up Podmuncher: transcription on {where}. Downloading what it needs; "
              "this can take a few minutes the first time.")
        lines: list[str] = []
        code = run_answers(auto, actions, out=lines.append, applier=applier)
        if code == EXIT_OK:
            # Each job carries the saved output folder and device, so no restart is needed.
            print(f"Ready. Transcripts go to {auto['paths']['output']}")
            print("Transcribe with: podmuncher <file or link>")
        else:
            for line in lines:
                print(line)
        return code
    if (_is_tty() if interactive is None else interactive):
        return run_interactive(actions, applier=applier)
    print(f"setup: no terminal and no answers; set {ANSWERS_ENV} to a JSON answers file", file=sys.stderr)
    return EXIT_INVALID


__all__ = (
    "ANSWERS_ENV", "LocalConfigWriter", "LocalSetupActions", "ProvisioningPort", "ServiceSetup",
    "SetupActions", "SetupApplier", "SetupError", "SetupMachine", "SetupResult", "SetupService",
    "SetupState", "StepResult", "UnboundProvisioning", "apply_remote", "doctor_report", "drive",
    "auto_answers", "load_answers", "main", "parse_capabilities", "run_answers", "shared_folder_warnings",
)
