# Whiskers Desktop Pet - Local Whisper transcription

import wave
import numpy as np
import whisper
from config import WHISPER_MODEL, WHISPER_LANGUAGE, AUDIO_SAMPLE_RATE


# Bias Whisper toward short voice commands (cuts long-form hallucinations).
_COMMAND_PROMPT = (
    "Voice command to a desktop assistant. Examples: open Brave, "
    "search Google for pasta recipe, take a screenshot, set a timer "
    "for five minutes, turn the volume up."
)

# If Whisper's no_speech_prob for every segment is above this, we treat the
# utterance as silence/noise and return "".
_NO_SPEECH_PROB_THRESHOLD = 0.6

# Debug: where to dump the last captured PCM so you can listen back and tell
# whether the problem is the mic or Whisper.
_DEBUG_WAV_PATH = "/tmp/whiskers_last.wav"


class WhisperTranscriber:
    """
    Thin wrapper around openai-whisper. Model loads once at startup (slow),
    transcribe() is called per utterance.

    We feed a numpy int16 array directly so Whisper never invokes ffmpeg —
    no system ffmpeg binary required.
    """

    def __init__(self, model_name=WHISPER_MODEL, language=WHISPER_LANGUAGE):
        print(f"Loading Whisper model '{model_name}' (first run downloads ~140MB)...")
        self.model = whisper.load_model(model_name)
        self.language = language
        print("Whisper model loaded.")

    def transcribe(self, pcm_int16: np.ndarray) -> str:
        if pcm_int16 is None or len(pcm_int16) == 0:
            return ""

        # Dump recording for debugging (play back with `afplay /tmp/whiskers_last.wav`).
        try:
            with wave.open(_DEBUG_WAV_PATH, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes(pcm_int16.tobytes())
        except Exception as e:
            print(f"[transcribe] could not write debug wav: {e}")

        # int16 PCM -> float32 in [-1.0, 1.0] (what whisper expects internally)
        audio = pcm_int16.astype(np.float32) / 32768.0

        result = self.model.transcribe(
            audio,
            fp16=False,                      # CPU path, no GPU half-precision
            language=self.language,          # skip language detection if set
            initial_prompt=_COMMAND_PROMPT,  # bias toward short commands
            condition_on_previous_text=False,  # don't feed past hallucinations back in
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            temperature=0.0,                 # deterministic; reduces creative drift
        )

        # Post-filter: if every segment thinks it's silence, throw the text out.
        segments = result.get("segments") or []
        if segments and all(
            seg.get("no_speech_prob", 0.0) > _NO_SPEECH_PROB_THRESHOLD
            for seg in segments
        ):
            print("[transcribe] rejected — all segments flagged no_speech")
            return ""

        text = result.get("text", "").strip()

        # Print diagnostics so you can see why something got rejected/accepted.
        if segments:
            nsp = [f"{seg.get('no_speech_prob', 0.0):.2f}" for seg in segments]
            print(f"[transcribe] no_speech_probs={nsp} text={text!r}")

        return text
