"""Presentation helpers: escaping, display-width truncation, profiles, restore, keys, wordmark."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.cells import cell_len

from podmuncher import cli, presentation
from podmuncher.presentation import (
    KEY_POLICY,
    TerminalProfile,
    default_adapter,
    escape_display,
    key_action,
    terminal_restore,
    truncate_display,
    welcome_text,
)

RESET = "\x1b[0m\x1b[?25h\x1b[?1049l"
ELLIPSIS = "\u2026"


class FakeTerminal(io.StringIO):
    def __init__(self, isatty: bool, encoding: str) -> None:
        super().__init__()
        self._tty = isatty
        self._enc = encoding

    def isatty(self) -> bool:
        return self._tty

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._enc


def make_terminal(isatty=True, encoding="utf-8", environ=None):
    return FakeTerminal(isatty, encoding), dict(environ or {})


def render_case(isatty=True, encoding="utf-8", environ=None) -> TerminalProfile:
    term, env = make_terminal(isatty, encoding, environ)
    return TerminalProfile.detect(env, stdout=term)


# ------------------------------------------------------------------------- escaping


@pytest.mark.parametrize("raw, expected", [
    ("\x1b[31mred\x1b[0m", "\\x1b\\[31mred\\x1b\\[0m"),
    ("\x1b]0;title\x07", "\\x1b\\]0;title\\x07"),
    ("a\u202eb", "a\\u202eb"),
    ("a\u2066b\u2069", "a\\u2066b\\u2069"),
    ("[bold]x[/bold]", "\\[bold\\]x\\[/bold\\]"),
    ("\x7f\x85", "\\x7f\\u0085"),
])
def test_escape_visible(raw, expected):
    original = str(raw)
    assert escape_display(raw) == expected
    assert raw == original


def test_escape_matches_cli_and_keeps_rtl():
    rtl = "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd"
    assert escape_display(rtl) == rtl
    sample = "x\x1b[1m\u202d[a]\\\u2028\U000e0001"
    assert escape_display(sample) == cli.PlainPresentation().escape(sample)


# ------------------------------------------------------------------------- truncation


CASES = [
    "x" * 80,
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059" * 4,
    "\U0001f600\U0001f601\U0001f602\U0001f923" * 6,
    "e\u0301" * 30,
    "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd " * 8,
]


@pytest.mark.parametrize("text", CASES)
@pytest.mark.parametrize("width", [1, 2, 5, 11, 20])
def test_truncate_by_display_width(text, width):
    original = str(text)
    out = truncate_display(text, width)
    assert cell_len(out) <= width
    assert text == original
    body = out[:-1] if out.endswith(ELLIPSIS) else out
    assert text.startswith(body)  # logical-order prefix: nothing reversed
    assert not body.startswith("\u0301")


def test_truncate_no_orphan_combining_mark():
    text = "e\u0301e\u0301e\u0301"
    assert truncate_display(text, 2, ellipsis="") == "e\u0301e\u0301"
    assert truncate_display(text, 1, ellipsis="") == "e\u0301"
    assert truncate_display(text, 2) == "e\u0301" + ELLIPSIS


def test_truncate_short_untouched_and_edges():
    assert truncate_display("short", 10) == "short"
    assert truncate_display("\u65e5\u672c", 4) == "\u65e5\u672c"
    assert truncate_display("\u65e5\u672c\u8a9e", 4) == "\u65e5" + ELLIPSIS
    assert truncate_display("anything", 0) == ""
    assert truncate_display("anything", 1, ellipsis="...") == "a"


# ------------------------------------------------------------------------- keys


@pytest.mark.parametrize("key", ["tab", "shift+tab", "up", "down", "left", "right", "enter",
                                 "escape", "ctrl+c"])
@pytest.mark.parametrize("in_text", [True, False])
def test_keys_act_everywhere(key, in_text):
    assert key_action(key, in_text_input=in_text) == KEY_POLICY[key]


@pytest.mark.parametrize("key, action", [("n", "new_job"), ("question_mark", "help"),
                                         ("q", "detach")])
def test_text_keys_only_outside_inputs(key, action):
    assert key_action(key, in_text_input=False) == action
    assert key_action(key, in_text_input=True) is None


def test_key_misc():
    assert key_action("?", in_text_input=False) == "help"
    assert key_action("?", in_text_input=True) is None
    assert key_action("f13", in_text_input=False) is None
    assert KEY_POLICY["ctrl+c"] == "exit"


# ------------------------------------------------------------------------- restore


def test_restore_normal_exception_and_interrupt():
    term, env = make_terminal(environ={"TERM": "xterm"})
    with terminal_restore(term, environ=env):
        pass
    assert term.getvalue() == RESET
    for exc in (RuntimeError, KeyboardInterrupt):
        term, env = make_terminal(environ={"TERM": "xterm"})
        with pytest.raises(exc), terminal_restore(term, environ=env):
            raise exc("boom")
        assert term.getvalue() == RESET


def test_restore_silent_for_pipe_and_dumb():
    for isatty, env in ((False, {"TERM": "xterm"}), (True, {"TERM": "dumb"})):
        term, environ = make_terminal(isatty=isatty, environ=env)
        with terminal_restore(term, environ=environ):
            pass
        assert term.getvalue() == ""


# ------------------------------------------------------------------------- profiles


def test_profiles():
    assert render_case(environ={"TERM": "xterm-256color"}).color == "full"
    assert render_case(environ={"TERM": "xterm", "COLORTERM": "truecolor"}).color == "full"
    assert render_case(environ={"TERM": "xterm"}).color == "16"
    assert render_case(environ={"TERM": "xterm-256color", "NO_COLOR": "1"}).color == "none"
    assert render_case(environ={"TERM": "dumb"}).color == "none"
    assert render_case(isatty=False, environ={"TERM": "xterm-256color"}).color == "none"
    assert render_case(environ={"PODMUNCHER_THEME": "light"}).theme == "light"
    assert render_case(environ={"PODMUNCHER_THEME": "dark"}).theme == "dark"
    assert render_case(environ={"PODMUNCHER_THEME": "neon"}).theme == "auto"
    assert render_case().theme == "auto"
    assert render_case(environ={"PODMUNCHER_REDUCED_MOTION": "1"}).reduced_motion
    assert not render_case().reduced_motion


def test_fullscreen_and_unicode():
    assert render_case(environ={"TERM": "xterm"}).fullscreen_ok
    assert not render_case(isatty=False).fullscreen_ok
    assert not render_case(environ={"TERM": "dumb"}).fullscreen_ok
    latin = render_case(encoding="cp1252", environ={"TERM": "xterm"})
    assert not latin.unicode and not latin.fullscreen_ok


def test_welcome_text_gating():
    art = welcome_text(render_case(environ={"TERM": "xterm-256color"}))
    assert "p o d m u n c h e r" in art and "\n" in art
    for profile in (render_case(encoding="cp1252", environ={"TERM": "xterm-256color"}),
                    render_case(environ={"TERM": "dumb"}),
                    render_case(environ={"NO_COLOR": "1"}),
                    render_case(isatty=False)):
        assert welcome_text(profile) == "podmuncher"


def test_welcome_asset():
    path = Path(presentation.__file__).parent / "assets" / "welcome.txt"
    text = path.read_text(encoding="ascii")
    lines = text.splitlines()
    assert 0 < len(lines) <= 5
    assert max(len(line) for line in lines) <= 60
    assert "podmuncher" in text.replace(" ", "")


# ------------------------------------------------------------------------- adapter


def test_default_adapter_conforms_and_renders():
    adapter = default_adapter()
    assert callable(adapter.render) and callable(adapter.escape)
    try:
        cli.configure_presentation(adapter)
    finally:
        cli.configure_presentation(None)
    assert adapter.escape("[x]\x1b") == "\\[x\\]\\x1b"
    result = {"ok": True, "reason_code": "ok", "note": "[bold]\x1b[31m"}
    out = adapter.render(result, command="status")
    assert "\\[bold\\]\\x1b\\[31m" in out and "\x1b" not in out
    assert result["note"] == "[bold]\x1b[31m"


def test_render_truncates_only_on_tty(monkeypatch):
    note = "\u65e5\u672c\u8a9e" * 40
    result = {"ok": True, "reason_code": "ok", "note": note}
    monkeypatch.setattr("shutil.get_terminal_size",
                        lambda *a, **k: type("S", (), {"columns": 30})())
    tty, _ = make_terminal(isatty=True)
    pipe, _ = make_terminal(isatty=False)
    short = presentation._DisplayAdapter(tty).render(result, command="x")
    full = presentation._DisplayAdapter(pipe).render(result, command="x")
    assert all(cell_len(line) <= 30 for line in short.split("\n"))
    assert note in full
    assert result["note"] == note


def test_no_textual_import():
    src = Path(presentation.__file__).read_text(encoding="utf-8")
    assert "import textual" not in src and "from textual" not in src
