"""Terminal presentation helpers — tables, progress, colour, target scoring.

Deliberately dependency-free and SLURM-aware. Log files are not TTYs, so colour codes
and carriage-return progress bars turn a cluster log into unreadable noise; everything
here degrades to plain, line-buffered output when stdout is redirected.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from typing import Iterable, Iterator, Optional, Sequence

_FORCE = os.environ.get("SW_FORCE_COLOR") == "1"
_NO_COLOR = os.environ.get("NO_COLOR") is not None


def supports_color() -> bool:
    if _NO_COLOR:
        return False
    if _FORCE:
        return True
    return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


class _C:
    def __init__(self) -> None:
        on = supports_color()
        self.reset = "\033[0m" if on else ""
        self.bold = "\033[1m" if on else ""
        self.dim = "\033[2m" if on else ""
        self.red = "\033[91m" if on else ""
        self.green = "\033[92m" if on else ""
        self.yellow = "\033[93m" if on else ""
        self.blue = "\033[94m" if on else ""
        self.magenta = "\033[95m" if on else ""
        self.cyan = "\033[96m" if on else ""


C = _C()

PASS = f"{C.green}PASS{C.reset}"
FAIL = f"{C.red}FAIL{C.reset}"
WARN = f"{C.yellow}WARN{C.reset}"


def term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default


def rule(title: str = "", char: str = "─") -> str:
    w = min(term_width(), 100)
    if not title:
        return char * w
    pad = max(w - len(title) - 3, 0)
    return f"{C.bold}{char * 2} {title} {char * pad}{C.reset}"


def header(title: str, subtitle: str = "") -> str:
    out = [rule(title, "═")]
    if subtitle:
        out.append(f"{C.dim}{subtitle}{C.reset}")
    return "\n".join(out)


def table(
    rows: Sequence[Sequence[object]],
    headers: Sequence[str],
    aligns: Optional[Sequence[str]] = None,
    indent: str = "  ",
) -> str:
    """Fixed-width table. ``aligns`` entries are 'l' or 'r' (default right except col 0)."""
    if not rows:
        return f"{indent}(no rows)"
    aligns = list(aligns or ["l"] + ["r"] * (len(headers) - 1))
    cells = [[_plain(str(c)) for c in row] for row in rows]
    widths = [
        max(len(_plain(h)), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)
    ]

    def fmt(vals: Sequence[str], raw: Sequence[object]) -> str:
        parts = []
        for i, v in enumerate(vals):
            pad = widths[i] - len(v)
            s = str(raw[i])
            parts.append((" " * pad + s) if aligns[i] == "r" else (s + " " * pad))
        return indent + "  ".join(parts)

    out = [
        f"{C.bold}{fmt([_plain(h) for h in headers], headers)}{C.reset}",
        indent + "  ".join("─" * w for w in widths),
    ]
    out += [fmt(c, r) for c, r in zip(cells, rows)]
    return "\n".join(out)


def _plain(s: str) -> str:
    """Length of a string ignoring ANSI escapes, for column alignment."""
    out, i = [], 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def bar(value: float, width: int = 24, lo: float = 0.0, hi: float = 1.0) -> str:
    """Unicode meter, coloured by magnitude."""
    frac = 0.0 if hi <= lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(frac * width)
    colour = C.red if frac > 0.7 else (C.yellow if frac > 0.4 else C.green)
    return f"{colour}{'█' * filled}{C.reset}{C.dim}{'░' * (width - filled)}{C.reset}"


def status(ok: Optional[bool]) -> str:
    if ok is None:
        return f"{C.dim} -- {C.reset}"
    return PASS if ok else FAIL


class Progress:
    """Minimal progress reporter.

    On a TTY it redraws one line. Redirected to a SLURM log it prints a new line only
    at intervals, so an 8-hour job produces a readable handful of lines instead of
    tens of thousands of carriage returns.
    """

    def __init__(self, total: int, desc: str = "", every: float = 30.0) -> None:
        self.total = max(int(total), 1)
        self.desc = desc
        self.every = every
        self.n = 0
        self._t0 = time.monotonic()
        self._last = 0.0
        self._tty = sys.stdout.isatty()
        self._rendered_at = -1  # guards against printing the final line twice

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.monotonic()
        if self._tty:
            if now - self._last < 0.1 and self.n < self.total:
                return
        elif now - self._last < self.every and self.n < self.total:
            return
        self._last = now
        self._render()

    def _render(self) -> None:
        if self._rendered_at == self.n:
            return
        self._rendered_at = self.n
        elapsed = time.monotonic() - self._t0
        frac = self.n / self.total
        rate = self.n / max(elapsed, 1e-9)
        eta = (self.total - self.n) / max(rate, 1e-9)
        msg = (f"{self.desc} {self.n}/{self.total} ({frac*100:5.1f}%) "
               f"{rate:.1f}/s  elapsed {_hms(elapsed)}  eta {_hms(eta)}")
        if self._tty:
            print(f"\r{' ' * min(term_width(), 120)}\r  {bar(frac)} {msg}", end="", flush=True)
        else:
            print(f"  {msg}", flush=True)

    def close(self) -> None:
        self._render()  # no-ops if the final state was already printed
        if self._tty:
            print(flush=True)


def track(it: Iterable, total: Optional[int] = None, desc: str = "") -> Iterator:
    """Wrap an iterable in a Progress reporter."""
    if total is None:
        try:
            total = len(it)  # type: ignore[arg-type]
        except TypeError:
            total = 0
    p = Progress(total or 1, desc)
    try:
        for item in it:
            yield item
            p.update()
    finally:
        p.close()


def _hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


if __name__ == "__main__":
    print(header("ShadowWeave console", "presentation helpers"))
    print()
    print(table(
        [["iou_5s", "0.612", f"{PASS}"], ["collision_rate", "0.184", f"{FAIL}"]],
        headers=["metric", "value", "status"],
    ))
    print()
    for v in (0.1, 0.5, 0.9):
        print(f"  {v:.1f} {bar(v)}")
    print()
    p = Progress(30, "demo")
    for _ in range(30):
        time.sleep(0.005)
        p.update()
    p.close()
    print(f"\ncolour enabled: {supports_color()}  (piped output degrades to plain text)")
    print("console OK")
