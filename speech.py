# Whiskers Desktop Pet - Kokoro TTS (Step 6 rebuilt)
#
# Uses Kokoro neural TTS with voice af_heart.
# Streams audio chunks as they generate — plays the first chunk while
# the rest is still being synthesised, so perceived latency is low.
# Playback via sounddevice (already installed with moonshine-voice).

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from config import KOKORO_VOICE, KOKORO_SPEED, KOKORO_SAMPLE_RATE


class Speaker:
    """Non-blocking TTS queue backed by Kokoro. Call speak(text) from any thread."""

    _SHUTDOWN = object()

    def __init__(self, speaking_lock: threading.Event = None):
        self._queue: queue.Queue = queue.Queue()
        self._pipe = None
        self._thread: Optional[threading.Thread] = None
        # Shared flag — set while audio is playing so the mic loop can pause.
        self._speaking_lock = speaking_lock

        self._thread = threading.Thread(
            target=self._run, name="WhiskersTTS", daemon=True
        )
        self._thread.start()

    def speak(self, text: str) -> None:
        """Queue text to be spoken. Returns immediately."""
        text = (text or "").strip()
        if text:
            self._queue.put(text)

    def shutdown(self) -> None:
        if self._thread and self._thread.is_alive():
            self._queue.put(self._SHUTDOWN)

    def _run(self) -> None:
        # Lazy-load Kokoro pipeline on the worker thread (heavy first init)
        try:
            from kokoro import KPipeline
            self._pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
            print(f"[speech] Kokoro ready (voice={KOKORO_VOICE})")
        except Exception as e:
            print(f"[speech] Kokoro init failed: {e!r} — TTS disabled")
            return

        while True:
            item = self._queue.get()
            if item is self._SHUTDOWN:
                break
            self._speak_streaming(item)

    def _speak_streaming(self, text: str) -> None:
        """Generate and stream audio chunk by chunk."""
        if self._speaking_lock:
            self._speaking_lock.set()  # tell mic loop to pause
        try:
            for result in self._pipe(
                text,
                voice=KOKORO_VOICE,
                speed=KOKORO_SPEED,
            ):
                audio = result.audio.numpy() if hasattr(result.audio, 'numpy') else np.array(result.audio)
                sd.play(audio, samplerate=KOKORO_SAMPLE_RATE)
                sd.wait()
        except Exception as e:
            print(f"[speech] Kokoro error: {e!r}")
        finally:
            if self._speaking_lock:
                self._speaking_lock.clear()  # mic loop resumes
