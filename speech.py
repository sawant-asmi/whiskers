# Whiskers Desktop Pet - Text-to-speech (Step 6)
#
# pyttsx3 has a well-known quirk: calling `engine.runAndWait()` multiple times
# from different threads (or even from the same thread with bad timing) can
# hang or deadlock, especially on macOS where the driver is NSSpeechSynthesizer.
#
# The reliable pattern is:
#   1. Init one engine on a dedicated worker thread.
#   2. Feed it phrases via a queue.
#   3. Each `say()` + `runAndWait()` happens serially on that thread.
#
# The listener thread that calls `speaker.speak(...)` just pushes to the queue
# and returns — never blocks on the TTS engine.

from __future__ import annotations

import queue
import threading
from typing import Optional

import pyttsx3

from config import TTS_ENABLED, TTS_RATE, TTS_VOLUME, TTS_VOICE


class Speaker:
    """Non-blocking speech queue. Call `speak(text)` from any thread."""

    _SHUTDOWN = object()

    def __init__(self,
                 rate: int = TTS_RATE,
                 volume: float = TTS_VOLUME,
                 voice: Optional[str] = TTS_VOICE,
                 enabled: bool = TTS_ENABLED):
        self.enabled = enabled
        self._rate = rate
        self._volume = volume
        self._voice_hint = voice
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None

        if self.enabled:
            self._thread = threading.Thread(
                target=self._run, name="WhiskersTTS", daemon=True
            )
            self._thread.start()

    # ── Public API ──

    def speak(self, text: str) -> None:
        """Queue text to be spoken. Returns immediately."""
        if not self.enabled:
            return
        text = (text or "").strip()
        if not text:
            return
        self._queue.put(text)

    def shutdown(self) -> None:
        """Stop the worker thread cleanly. Safe to call multiple times."""
        if self._thread and self._thread.is_alive():
            self._queue.put(self._SHUTDOWN)

    # ── Worker thread ──

    def _run(self) -> None:
        try:
            engine = pyttsx3.init()
        except Exception as e:
            print(f"[speech] pyttsx3 init failed: {e!r} — TTS disabled")
            self.enabled = False
            return

        # Apply properties
        try:
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            if self._voice_hint:
                hint = self._voice_hint.lower()
                for v in engine.getProperty("voices"):
                    if hint in (v.name or "").lower() or hint in (v.id or "").lower():
                        engine.setProperty("voice", v.id)
                        print(f"[speech] using voice: {v.name}")
                        break
                else:
                    print(f"[speech] voice matching {self._voice_hint!r} not found; using default")
        except Exception as e:
            print(f"[speech] setProperty failed: {e!r} (continuing with defaults)")

        while True:
            item = self._queue.get()
            if item is self._SHUTDOWN:
                break
            text: str = item
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[speech] error while speaking {text!r}: {e!r}")

        try:
            engine.stop()
        except Exception:
            pass
