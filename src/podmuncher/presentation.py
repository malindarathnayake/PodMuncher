"""Terminal-safe presentation helpers (display-only escaping, width truncation, profiles, keys).

Everything here is a projection: original text is never mutated, never reversed (RTL stays in
logical order) and measured with rich's framework cell widths.  Only rich is imported, never
textual.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Any, TextIO

from rich.cells import cell_len, split_graphemes

__all__ = (
    "KEY_POLICY", "TerminalProfile", "default_adapter", "escape_display", "key_action",
    "terminal_restore", "truncate_display", "welcome_text",
)

_RESET_SEQUENCE = "\x1b[0m\x1b[?25h\x1b[?1049l"  # SGR reset, show cursor, leave the alt screen
_WORDMARK_FALLBACK = "podmuncher"
_THEMES = ("auto", "light", "dark")


# ----------------------------------------------------------------------------- escaping


def escape_display(text: str) -> str:
    """Visible escape of terminal controls, bidi spoofing controls and Rich markup brackets.

    Identical to ``cli.PlainPresentation.escape`` so the CLI and the TUI escape the same way:
    backslash, ``[`` and ``]``, C0/DEL (``\\xNN``), C1, ESC, and the Cc/Cf/Cs/Zl/Zp categories
    (which include the bidi overrides U+202A-U+202E and isolates U+2066-U+2069).  Ordinary RTL
    letters are untouched.  Returns a new string.
    """
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


def truncate_display(text: str, width: int, *, ellipsis: str = "\u2026") -> str:
    """Truncate to at most ``width`` terminal cells, cutting only on grapheme boundaries."""
    text = str(text)
    if width <= 0:
        return ""
    spans, total = split_graphemes(text)
    if total <= width:
        return text
    budget = width - cell_len(ellipsis)
    if budget < 0:  # ellipsis does not fit: hard cut, no marker
        ellipsis, budget = "", width
    used = 0
    end = 0
    for _start, stop, cells in spans:
        if used + cells > budget:
            break
        used += cells
        end = stop
    return text[:end] + ellipsis


# ----------------------------------------------------------------------------- profile


def _is_tty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


@dataclass(frozen=True)
class TerminalProfile:
    color: str  # "none" | "16" | "full"
    theme: str  # "auto" | "light" | "dark"
    reduced_motion: bool
    unicode: bool
    fullscreen_ok: bool

    @classmethod
    def detect(cls, environ: Mapping[str, str], *, stdout: Any) -> TerminalProfile:
        term = environ.get("TERM", "")
        dumb = term.lower() == "dumb"
        tty = _is_tty(stdout)
        encoding = str(getattr(stdout, "encoding", "") or "").lower().replace("_", "-")
        utf8 = encoding in ("utf-8", "utf8")
        if environ.get("NO_COLOR") or dumb or not tty:
            color = "none"
        elif environ.get("COLORTERM", "").lower() in ("truecolor", "24bit") or "256color" in term:
            color = "full"
        else:
            color = "16"
        theme = environ.get("PODMUNCHER_THEME", "auto").strip().lower()
        if theme not in _THEMES:
            theme = "auto"
        return cls(
            color=color,
            theme=theme,
            reduced_motion=environ.get("PODMUNCHER_REDUCED_MOTION", "") == "1",
            unicode=utf8,
            fullscreen_ok=tty and utf8 and not dumb,
        )


@contextlib.contextmanager
def terminal_restore(stream: TextIO, *, environ: Mapping[str, str] | None = None) -> Iterator[None]:
    """Restore the terminal on every exit (normal, exception, KeyboardInterrupt); re-raises."""
    try:
        yield
    finally:
        env = os.environ if environ is None else environ
        if _is_tty(stream) and env.get("TERM", "").lower() != "dumb":
            with contextlib.suppress(OSError, ValueError):
                stream.write(_RESET_SEQUENCE)
                stream.flush()


# ----------------------------------------------------------------------------- keys

KEY_POLICY: dict[str, str] = {
    "tab": "focus_next",
    "shift+tab": "focus_previous",
    "up": "move_up",
    "down": "move_down",
    "left": "move_left",
    "right": "move_right",
    "enter": "open",
    "escape": "back",
    "n": "new_job",
    "question_mark": "help",
    "q": "detach",
    "ctrl+c": "exit",
}
_TEXT_KEYS = frozenset({"n", "question_mark", "q"})
_KEY_ALIASES = {"?": "question_mark", "esc": "escape", "return": "enter"}


def key_action(key: str, *, in_text_input: bool) -> str | None:
    """Action for a key; ``n``/``q``/``?`` are plain text inside text inputs."""
    name = _KEY_ALIASES.get(key, key)
    if in_text_input and name in _TEXT_KEYS:
        return None
    return KEY_POLICY.get(name)


# ----------------------------------------------------------------------------- adapter, branding


class _DisplayAdapter:
    """``cli.PresentationAdapter``: plain ``key: value`` lines, width-truncated on a TTY."""

    def __init__(self, stdout: Any = None) -> None:
        self._stdout = stdout

    def escape(self, text: str) -> str:
        return escape_display(text)

    def render(self, result: dict, *, command: str) -> str:
        from .cli import PlainPresentation

        rendered = PlainPresentation().render(result, command=command)
        stream = self._stdout if self._stdout is not None else sys.stdout
        if not _is_tty(stream):
            return rendered
        width = shutil.get_terminal_size().columns
        return "\n".join(truncate_display(line, width) for line in rendered.split("\n"))


def default_adapter() -> _DisplayAdapter:
    return _DisplayAdapter()


def welcome_text(profile: TerminalProfile) -> str:
    """Original text wordmark for welcome/empty screens only; plain fallback otherwise."""
    if not (profile.unicode and profile.fullscreen_ok and profile.color != "none"):
        return _WORDMARK_FALLBACK
    try:
        text = resources.files("podmuncher").joinpath("assets", "welcome.txt").read_text("utf-8")
    except (OSError, ModuleNotFoundError):
        return _WORDMARK_FALLBACK
    return text.rstrip("\n") or _WORDMARK_FALLBACK
