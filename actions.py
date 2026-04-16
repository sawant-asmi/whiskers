# Whiskers Desktop Pet - Action executor (Step 5)
#
# Takes the structured Action from brain.py and actually does the thing on macOS.
# Called from the listener thread after brain.process() returns — blocking here
# is fine because the listener is already paused until the next wake word.

from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
import time
import urllib.parse
from typing import List, Optional

import pyautogui

from brain import Action


# ── App-name aliases ────────────────────────────────────────────────────────
# macOS `open -a` is case-insensitive but needs the app's display name.
# Map what Whisper typically hears → the canonical app bundle name.

APP_ALIASES = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "brave browser": "Brave Browser",
    "brave": "Brave Browser",
    "edge": "Microsoft Edge",
    "microsoft edge": "Microsoft Edge",
    "firefox": "Firefox",
    "safari": "Safari",
    "vs code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "code": "Visual Studio Code",
    "visual studio code": "Visual Studio Code",
    "music": "Music",
    "apple music": "Music",
    "spotify": "Spotify",
    "zoom": "zoom.us",
    "iterm": "iTerm",
    "terminal": "Terminal",
    "finder": "Finder",
    "mail": "Mail",
    "messages": "Messages",
    "calendar": "Calendar",
    "notes": "Notes",
    "reminders": "Reminders",
    "photos": "Photos",
    "slack": "Slack",
    "discord": "Discord",
    "system settings": "System Settings",
    "system preferences": "System Preferences",
}


def _resolve_app_name(name: str) -> str:
    return APP_ALIASES.get(name.lower().strip(), name.strip())


# Keep references to running timers so they survive GC.
_pending_timers: List[threading.Timer] = []


# ── Public entry point ──────────────────────────────────────────────────────

def execute(action: Action, cat=None) -> None:
    """Execute a brain Action on the host machine. `cat` is the CatWindow
    (needed for the `move` action to emit the Qt signal)."""
    handler = _DISPATCH.get(action.action)
    if handler is None:
        print(f"[actions] no handler for {action.action!r}")
        return
    try:
        handler(action, cat)
    except Exception as e:
        print(f"[actions] {action.action} failed: {e!r}")


# ── Handlers ────────────────────────────────────────────────────────────────

def _open_app(action: Action, cat) -> None:
    name = (action.argument or "").strip()
    if not name:
        print("[actions] open_app: no app name")
        return
    resolved = _resolve_app_name(name)
    print(f"[actions] open -a {resolved!r}")
    result = subprocess.run(
        ["open", "-a", resolved],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[actions] open_app failed ({result.returncode}): {result.stderr.strip()}")


def _type_text(action: Action, cat) -> None:
    text = action.argument or ""
    if not text:
        print("[actions] type_text: no text")
        return
    # Give the user's focused window a beat to settle (wake-word UI just fired).
    time.sleep(0.25)
    print(f"[actions] typing: {text!r}")
    # typewrite handles printable ASCII + common punctuation; non-ASCII will
    # raise KeyError which we swallow with write() as a fallback.
    try:
        pyautogui.typewrite(text, interval=0.02)
    except KeyError:
        pyautogui.write(text, interval=0.02)


def _search_google(action: Action, cat) -> None:
    q = (action.argument or "").strip()
    if not q:
        return
    url = f"https://www.google.com/search?q={urllib.parse.quote_plus(q)}"
    print(f"[actions] open {url}")
    subprocess.run(["open", url], check=False)


def _open_youtube(action: Action, cat) -> None:
    q = (action.argument or "").strip()
    if q:
        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(q)}"
    else:
        url = "https://www.youtube.com/"
    print(f"[actions] open {url}")
    subprocess.run(["open", url], check=False)


def _screenshot(action: Action, cat) -> None:
    ts = time.strftime("%Y-%m-%d-%H%M%S")
    path = os.path.expanduser(f"~/Desktop/Whiskers-{ts}.png")
    print(f"[actions] screencapture -> {path}")
    subprocess.run(["screencapture", path], check=False)


def _volume(action: Action, cat) -> None:
    arg = (action.argument or "").lower().strip()
    if arg == "mute":
        _osa("set volume with output muted")
    elif arg == "unmute":
        _osa("set volume without output muted")
    elif arg == "up":
        _osa("set volume output volume (output volume of (get volume settings) + 10) --100%")
    elif arg == "down":
        _osa("set volume output volume (output volume of (get volume settings) - 10) --100%")
    elif arg.isdigit():
        level = max(0, min(100, int(arg)))
        _osa(f"set volume output volume {level} --100%")
    else:
        print(f"[actions] volume: unrecognized argument {arg!r}")


def _set_timer(action: Action, cat) -> None:
    raw = (action.argument or "").strip()
    seconds = _parse_duration(raw)
    if not seconds:
        print(f"[actions] set_timer: couldn't parse duration {raw!r}")
        return
    label = raw or f"{seconds} seconds"
    print(f"[actions] timer: {seconds}s ({label})")

    def _fire():
        msg = f"Your {label} timer is done!"
        # Use shlex-quoted AppleScript — safer than f-string with arbitrary text.
        script = (
            f'display notification {shlex.quote(msg)} '
            f'with title "Whiskers Timer" '
            f'sound name "Glass"'
        )
        subprocess.run(["osascript", "-e", script], check=False)
        print(f"[actions] timer fired ({label})")

    t = threading.Timer(seconds, _fire)
    t.daemon = True
    t.start()
    _pending_timers.append(t)


def _move(action: Action, cat) -> None:
    direction = (action.argument or "").lower().strip()
    if direction not in ("left", "right"):
        print(f"[actions] move: bad direction {direction!r}")
        return
    if cat is None:
        print("[actions] move: no cat window reference")
        return
    # Signals are thread-safe across Qt; this hops onto the GUI thread.
    cat.move_signal.emit(direction)


def _chat(action: Action, cat) -> None:
    """Chat responses are spoken + displayed — nothing to execute."""
    print(f"[actions] chat: {action.response!r}")


def _unknown(action: Action, cat) -> None:
    print("[actions] unknown action — nothing to execute")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _osa(script: str) -> None:
    """Run AppleScript via osascript."""
    subprocess.run(["osascript", "-e", script], check=False)


def _parse_duration(s: str) -> Optional[int]:
    """'5 minutes' / '30 sec' / '1 hour' -> seconds."""
    m = re.search(r"(\d+)\s*(second|sec|minute|min|hour|hr)s?\b", s, re.I)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("sec"):
        return n
    if unit.startswith("min"):
        return n * 60
    if unit.startswith("h"):
        return n * 3600
    return None


_DISPATCH = {
    "chat": _chat,
    "open_app": _open_app,
    "type_text": _type_text,
    "search_google": _search_google,
    "open_youtube": _open_youtube,
    "screenshot": _screenshot,
    "volume": _volume,
    "set_timer": _set_timer,
    "move": _move,
    "unknown": _unknown,
}
