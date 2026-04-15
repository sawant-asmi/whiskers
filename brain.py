# Whiskers Desktop Pet - Local regex-based intent router
#
# Step 4 (local version): given a transcribed voice command, classify it into
# a structured Action using pattern matching. Zero network, zero API cost.
#
# Why regex, not an LLM:
#   - The command space is small and fixed.
#   - Latency is under a millisecond — matters for voice feedback.
#   - No API key, no internet, no downloads, works offline.
#
# Trade-off: rigid. If a phrase isn't anticipated by any pattern we return
# `unknown` and Whiskers says she didn't catch it. Step 5 can add a local-LLM
# fallback for unknowns later if it turns out to be a problem.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Pattern


# ── Action schema (same shape Step 5 will execute) ──────────────────────────

@dataclass
class Action:
    action: str                  # one of the labels in ACTION_LABELS below
    argument: Optional[str]      # action-specific argument, or None
    response: str                # short spoken confirmation


ACTION_LABELS = {
    "open_app", "search_google", "open_youtube", "type_text",
    "screenshot", "volume", "set_timer", "move", "unknown",
}


# ── Rules ───────────────────────────────────────────────────────────────────
# Each rule: (compiled_regex, action_name, arg_extractor, response_builder)
# - arg_extractor:   match -> Optional[str]
# - response_builder: arg  -> str (spoken confirmation)
# Patterns are tried in order; first match wins. ORDER MATTERS — more specific
# patterns come first so e.g. "open youtube" hits the youtube rule before the
# generic open_app catch-all.

Rule = tuple[Pattern[str], str, Callable[[re.Match], Optional[str]], Callable[[Optional[str]], str]]

RULES: list[Rule] = [
    # ── MOVE (Whiskers himself) ──
    (
        re.compile(r"\b(?:move|scoot|go|slide|shift)\b[^a-z]*\b(left|right)\b", re.I),
        "move",
        lambda m: m.group(1).lower(),
        lambda a: f"Scooting {a}.",
    ),
    (
        re.compile(r"\b(?:to the |on the )(left|right)\b.*\b(?:side|please)?\b", re.I),
        "move",
        lambda m: m.group(1).lower(),
        lambda a: f"Scooting {a}.",
    ),

    # ── SCREENSHOT ──
    (
        re.compile(r"\b(?:take (?:a )?)?screen ?shot\b", re.I),
        "screenshot",
        lambda m: None,
        lambda a: "Taking a screenshot.",
    ),

    # ── VOLUME ──
    (
        re.compile(r"\bun[ -]?mute\b", re.I),
        "volume",
        lambda m: "unmute",
        lambda a: "Unmuting.",
    ),
    (
        re.compile(r"\b(?:mute|silence)\b", re.I),
        "volume",
        lambda m: "mute",
        lambda a: "Muting.",
    ),
    (
        re.compile(r"\b(?:volume|sound|audio)\b[^a-z]*(?:to\s+)?(\d{1,3})\b", re.I),
        "volume",
        lambda m: m.group(1),
        lambda a: f"Setting volume to {a}.",
    ),
    (
        re.compile(r"\b(?:turn (?:it |the (?:volume|sound) )?up|louder|volume up|raise (?:the )?volume)\b", re.I),
        "volume",
        lambda m: "up",
        lambda a: "Volume up.",
    ),
    (
        re.compile(r"\b(?:turn (?:it |the (?:volume|sound) )?down|quieter|softer|volume down|lower (?:the )?volume)\b", re.I),
        "volume",
        lambda m: "down",
        lambda a: "Volume down.",
    ),

    # ── TIMER ──
    (
        re.compile(
            r"\b(?:set (?:a |an )?timer|timer)\b[^a-z0-9]*(?:for\s+)?"
            r"(\d+\s*(?:second|minute|hour|sec|min|hr)s?)",
            re.I,
        ),
        "set_timer",
        lambda m: m.group(1).strip(),
        lambda a: f"Timer set for {a}.",
    ),
    (
        re.compile(
            r"\b(\d+\s*(?:second|minute|hour|sec|min|hr)s?)\s+(?:timer|countdown)\b",
            re.I,
        ),
        "set_timer",
        lambda m: m.group(1).strip(),
        lambda a: f"Timer set for {a}.",
    ),

    # ── YOUTUBE (before the generic open_app rule) ──
    (
        re.compile(
            r"\b(?:open|launch|go to|play on)?\s*you\s*tube\b"
            r"(?:\s+(?:and\s+)?(?:search(?:\s+for)?|play|watch|find)\s+(.+))?",
            re.I,
        ),
        "open_youtube",
        lambda m: (m.group(1).strip().rstrip(".!?") if m.group(1) else None),
        lambda a: f"Opening YouTube{' and searching for ' + a if a else ''}.",
    ),

    # ── GOOGLE SEARCH ──
    (
        re.compile(
            r"\b(?:search(?:\s+(?:google|the web|online))?\s+(?:for\s+)?|google\s+(?:for\s+)?|look\s+up\s+)"
            r"(.+)",
            re.I,
        ),
        "search_google",
        lambda m: m.group(1).strip().rstrip(".!?"),
        lambda a: f"Searching for {a}.",
    ),

    # ── TYPE TEXT ──
    (
        re.compile(r"\btype\s+(.+)", re.I),
        "type_text",
        lambda m: m.group(1).strip().rstrip(".!?"),
        lambda a: "Typing.",
    ),

    # ── OPEN APP (catch-all — keep last among matchers) ──
    (
        re.compile(r"\b(?:open|launch|start|run)\s+(.+)", re.I),
        "open_app",
        lambda m: m.group(1).strip().rstrip(".!?"),
        lambda a: f"Opening {a}.",
    ),
]


def process(text: str) -> Optional[Action]:
    """
    Classify a transcribed utterance into an Action. Prints the result;
    Step 5 will execute it. Returns None only when there is no utterance.
    """
    if not text or not text.strip():
        print("[brain] (empty transcript — nothing to do)")
        return None

    cleaned = text.strip()

    for pattern, action_name, arg_fn, resp_fn in RULES:
        m = pattern.search(cleaned)
        if m:
            arg = arg_fn(m)
            resp = resp_fn(arg)
            action = Action(action=action_name, argument=arg, response=resp)
            print(
                f"[brain] action={action.action!r}  "
                f"argument={action.argument!r}  "
                f"response={action.response!r}"
            )
            return action

    action = Action(
        action="unknown",
        argument=None,
        response="Hmm, I didn't catch that.",
    )
    print(f"[brain] no rule matched {cleaned!r} -> unknown")
    return action
