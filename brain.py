# Whiskers Desktop Pet - Groq-powered brain (Llama 3.3 70b)
#
# Takes transcribed voice input and routes it through Groq:
#   - Casual chat → conversational reply (displayed + spoken)
#   - Computer command → structured JSON action for actions.py

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL


# ── Action schema ───────────────────────────────────────────────────────────

@dataclass
class Action:
    action: str                  # "chat", "open_app", "search_google", etc.
    argument: Optional[str]      # action-specific parameter, or None
    response: str                # short spoken/displayed confirmation


ACTION_LABELS = {
    "chat", "open_app", "search_google", "open_youtube", "type_text",
    "screenshot", "volume", "set_timer", "move", "unknown",
}


# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are **Whiskers**, a witty, warm, and slightly cat-like desktop assistant \
who lives on a macOS screen. You're a small animated cat character. You keep \
replies SHORT and snappy — 1-2 sentences max. A subtle cat reference now and \
then is fine, but don't overdo it.

You handle two kinds of input:

## 1. Casual chat
Greetings, questions, jokes, feelings, compliments, "how are you", etc. \
Reply naturally in your cat personality.

## 2. Computer commands
When the user wants to DO something on their Mac. Recognise these intents:

| action | argument | examples |
|---|---|---|
| open_app | app name ("Safari", "Brave", "Spotify") | "open Safari", "launch Brave" |
| search_google | search query | "search for pasta recipes", "Google best pizza" |
| open_youtube | query or null | "YouTube lo-fi beats", "open YouTube" |
| type_text | text to type | "type hello how are you" |
| screenshot | null | "take a screenshot", "screenshot" |
| volume | "up"/"down"/"mute"/"unmute" or 0-100 | "volume up", "mute", "set volume to 40" |
| set_timer | duration string | "set a timer for 5 minutes" |
| move | "left" or "right" | "move left", "scoot to the right" |

If the command doesn't match anything above, use action "unknown".

## Response format — ALWAYS valid JSON

For casual chat:
{"type": "chat", "response": "your witty reply"}

For commands:
{"type": "command", "action": "<action>", "argument": "<value or null>", "response": "short confirmation"}

## Examples

User: "hey whiskers"
{"type": "chat", "response": "Hey there! Just lounging on your screen as usual."}

User: "how are you"
{"type": "chat", "response": "Purring along just fine! What's up?"}

User: "open safari"
{"type": "command", "action": "open_app", "argument": "Safari", "response": "Opening Safari for you!"}

User: "search for best pizza near me"
{"type": "command", "action": "search_google", "argument": "best pizza near me", "response": "Let me Google that!"}

User: "take a screenshot"
{"type": "command", "action": "screenshot", "argument": null, "response": "Say cheese!"}

User: "you're cute"
{"type": "chat", "response": "Aw, you're making me purr! Thanks!"}

User: "blah blah random gibberish"
{"type": "command", "action": "unknown", "argument": null, "response": "Hmm, I didn't quite catch that."}

Always respond with ONLY the JSON object. No markdown, no backticks, no extra text.\
"""


# ── Client ──────────────────────────────────────────────────────────────────

_client: Optional[Groq] = None
_client_status: str = "unknown"  # "ok" | "no_key" | "error"


def _get_client() -> Optional[Groq]:
    global _client, _client_status
    if _client is not None:
        return _client
    if _client_status == "no_key":
        return None

    if not GROQ_API_KEY:
        print("[brain] GROQ_API_KEY not set — brain disabled.")
        print("[brain] export GROQ_API_KEY=gsk_... and restart Whiskers.")
        _client_status = "no_key"
        return None

    try:
        _client = Groq(api_key=GROQ_API_KEY)
        _client_status = "ok"
        print(f"[brain] Groq client ready (model: {GROQ_MODEL})")
        return _client
    except Exception as e:
        print(f"[brain] Groq init failed: {e!r}")
        _client_status = "error"
        return None


# ── Public API ──────────────────────────────────────────────────────────────

def process(text: str) -> Optional[Action]:
    """
    Send transcribed text to Groq, parse the JSON response, return an Action.
    Returns None only when there's no text or the API is unavailable.
    """
    if not text or not text.strip():
        print("[brain] (empty transcript — nothing to do)")
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text.strip()},
            ],
            temperature=0.7,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        print(f"[brain] Groq API error: {e!r}")
        return Action(
            action="unknown", argument=None,
            response="Sorry, my brain glitched for a second.",
        )

    raw = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    if usage:
        print(
            f"[brain] groq tokens: prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} "
            f"total={usage.total_tokens}"
        )

    return _parse_response(raw)


def _parse_response(raw: str) -> Action:
    """Parse Groq's JSON into an Action. Gracefully handles bad JSON."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[brain] bad JSON from Groq: {raw!r}")
        # Treat raw text as a chat response
        return Action(action="chat", argument=None, response=raw[:200])

    msg_type = data.get("type", "chat")
    resp = data.get("response", "")

    if msg_type == "chat":
        action = Action(action="chat", argument=None, response=resp)
    elif msg_type == "command":
        act = data.get("action", "unknown")
        arg = data.get("argument")
        # Normalise null / "null" / "" to None
        if arg in (None, "null", ""):
            arg = None
        if act not in ACTION_LABELS:
            act = "unknown"
        action = Action(action=act, argument=arg, response=resp)
    else:
        action = Action(action="chat", argument=None, response=resp)

    print(
        f"[brain] type={msg_type!r}  action={action.action!r}  "
        f"argument={action.argument!r}  response={action.response!r}"
    )
    return action
