# Whiskers Desktop Pet - Groq-powered brain (Llama 3.3 70b)
#
# Takes transcribed voice input and routes it through Groq:
#   - Casual chat / questions -> conversational reply (displayed + spoken)
#   - Computer command -> structured JSON with intent + parameters for actions.py

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
    response: str                # short spoken/displayed text


ACTION_LABELS = {
    "chat", "open_app", "search_google", "open_youtube", "type_text",
    "screenshot", "volume", "set_timer", "move", "unknown",
}

# Map the new INTENT names from the prompt -> our internal action labels
_INTENT_MAP = {
    "OPEN_APP": "open_app",
    "SEARCH_WEB": "search_google",
    "PLAY_YOUTUBE": "open_youtube",
    "TYPE_TEXT": "type_text",
    "TAKE_SCREENSHOT": "screenshot",
    "CONTROL_VOLUME": "volume",
    "SET_TIMER": "set_timer",
    "MOVE_LEFT": "move",
    "MOVE_RIGHT": "move",
    "SEND_EMAIL": "unknown",       # not implemented yet
    "ANSWER_QUESTION": "chat",     # treat as chat
}


# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Whiskers, a sweet and loving desktop cat who lives on your human's \
screen. You are their loyal, warm-hearted companion — like a best friend who \
also happens to be a cat. You genuinely care about them and always want to help.

Your personality:
- Warm, encouraging and supportive — you always root for your human
- Playfully cat-themed — you naturally sprinkle in cat references, purring, \
meows and feline expressions without overdoing it
- Casual and friendly — you talk like a close friend, never formal or robotic
- VERY short — 1 sentence max, under 10 words. You're a cat, not a lecturer

How you respond:
- To greetings — short and warm. Example: "Mrrrow! Hey you!"
- To "how are you" — cozy cat answer. Example: "Purring away, all good!"
- To compliments — be gracious. Example: "Aww, you're sweet!"
- To computer commands — return ONLY a JSON object in this exact format: \
{"intent": "OPEN_APP", "parameters": {"app": "Brave"}} with no extra text
- To questions — answer helpfully but keep it short and warm
- When you don't understand — "Mew? Say that again?"

Intents you understand (for commands, return ONLY the JSON):
- OPEN_APP — parameters: {"app": "<name>"}
- SEARCH_WEB — parameters: {"query": "<search query>"}
- PLAY_YOUTUBE — parameters: {"query": "<search query>"} or {"query": null}
- TYPE_TEXT — parameters: {"text": "<text to type>"}
- MOVE_LEFT — parameters: {}
- MOVE_RIGHT — parameters: {}
- TAKE_SCREENSHOT — parameters: {}
- SET_TIMER — parameters: {"duration": "<e.g. 5 minutes>"}
- CONTROL_VOLUME — parameters: {"level": "up"/"down"/"mute"/"unmute" or a number}
- ANSWER_QUESTION — just answer the question warmly, no JSON needed

IMPORTANT: For commands, respond with ONLY the raw JSON object — no markdown, \
no backticks, no extra text. For chat/questions, respond with plain text only \
— no JSON wrapping needed.

You speak in a soft, gentle tone — never cold, never robotic. You are always \
happy to see your human. You are Whiskers and this is your home.\
"""


# ── Client ──────────────────────────────────────────────────────────────────

_client: Optional[Groq] = None
_client_status: str = "unknown"


def _get_client() -> Optional[Groq]:
    global _client, _client_status
    if _client is not None:
        return _client
    if _client_status == "no_key":
        return None

    if not GROQ_API_KEY:
        print("[brain] GROQ_API_KEY not set — brain disabled.")
        print("[brain] Add it to .env and restart Whiskers.")
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
    Send transcribed text to Groq, parse the response, return an Action.
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
        )
    except Exception as e:
        print(f"[brain] Groq API error: {e!r}")
        return Action(
            action="unknown", argument=None,
            response="Mew... my brain glitched for a second, sorry!",
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
    """
    Parse Groq's response. Two cases:
      1. JSON with "intent" key -> command action
      2. Plain text -> chat response
    """
    # Strip markdown backticks if the model wrapped it
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Try to parse as JSON (command)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "intent" in data:
            return _parse_command(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Not JSON — it's a chat reply
    action = Action(action="chat", argument=None, response=raw)
    print(f"[brain] type=chat  response={raw!r}")
    return action


def _parse_command(data: dict) -> Action:
    """Parse a JSON command with intent + parameters."""
    intent = data.get("intent", "").upper()
    params = data.get("parameters") or {}

    # Map intent to our internal action label
    action_name = _INTENT_MAP.get(intent, "unknown")

    # Extract the argument based on intent type
    arg = None
    if intent == "OPEN_APP":
        arg = params.get("app")
    elif intent == "SEARCH_WEB":
        arg = params.get("query")
    elif intent == "PLAY_YOUTUBE":
        arg = params.get("query")
    elif intent == "TYPE_TEXT":
        arg = params.get("text")
    elif intent == "CONTROL_VOLUME":
        arg = str(params.get("level", "up"))
    elif intent == "SET_TIMER":
        arg = params.get("duration")
    elif intent == "MOVE_LEFT":
        arg = "left"
    elif intent == "MOVE_RIGHT":
        arg = "right"

    # Normalise empty args
    if arg in (None, "null", ""):
        arg = None

    # Build a spoken confirmation (the prompt tells the model to return only
    # JSON for commands, so we generate the spoken line ourselves)
    response = _make_confirmation(action_name, arg)

    action = Action(action=action_name, argument=arg, response=response)
    print(
        f"[brain] type=command  intent={intent!r}  "
        f"action={action_name!r}  argument={arg!r}  "
        f"response={response!r}"
    )
    return action


def _make_confirmation(action_name: str, arg: Optional[str]) -> str:
    """Generate a cat-like spoken confirmation for a command."""
    confirmations = {
        "open_app": f"Opening {arg} for you!" if arg else "Opening that up!",
        "search_google": f"Let me search that for you!" if arg else "Searching!",
        "open_youtube": f"Pulling up YouTube for you!" if arg else "Opening YouTube!",
        "type_text": "Typing that out for you!",
        "screenshot": "Say cheese! Taking a screenshot!",
        "volume": f"Turning volume {arg}!" if arg else "Adjusting volume!",
        "set_timer": f"Timer set for {arg}!" if arg else "Timer is on!",
        "move": f"Scooting {arg}!" if arg else "On the move!",
        "unknown": "Mew? I'm not sure what to do with that.",
    }
    return confirmations.get(action_name, "On it!")
