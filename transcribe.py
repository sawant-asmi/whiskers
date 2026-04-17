# Whiskers Desktop Pet - Groq Whisper API transcription
#
# Sends captured PCM audio to Groq's Whisper endpoint (whisper-large-v3-turbo).
# Groq runs Whisper on LPU hardware — transcription returns in under a second.
# No local Whisper model, no torch inference, no 140MB download.

from __future__ import annotations

import io
import wave
import numpy as np
from groq import Groq

from config import GROQ_API_KEY, AUDIO_SAMPLE_RATE

WHISPER_MODEL = "whisper-large-v3-turbo"


class GroqWhisperTranscriber:
    """
    Sends int16 PCM to Groq Whisper API and returns the transcribed text.
    Reuses a single Groq client.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            print("[transcribe] GROQ_API_KEY not set — transcription disabled.")
            self._client = None
        else:
            self._client = Groq(api_key=GROQ_API_KEY)
            print(f"[transcribe] Groq Whisper ready (model: {WHISPER_MODEL})")

    def transcribe(self, pcm_int16: np.ndarray) -> str:
        """
        Convert int16 PCM to WAV in memory, send to Groq Whisper, return text.
        """
        if self._client is None:
            return ""
        if pcm_int16 is None or len(pcm_int16) == 0:
            return ""

        # Build WAV in memory (Groq needs a file-like object with a name)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(AUDIO_SAMPLE_RATE)
            wf.writeframes(pcm_int16.tobytes())
        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"  # Groq API needs a filename

        try:
            result = self._client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=wav_buffer,
                language="en",
                response_format="text",
            )
            text = result.strip() if isinstance(result, str) else str(result).strip()
            print(f"[transcribe] groq whisper: {text!r}")
            return text
        except Exception as e:
            print(f"[transcribe] Groq Whisper error: {e!r}")
            return ""
