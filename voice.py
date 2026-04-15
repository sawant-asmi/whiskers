# Whiskers Desktop Pet - Wake Word Detection + Recording (OpenWakeWord + PyAudio)

import threading
import numpy as np
import pyaudio
from openwakeword.model import Model
from config import (
    WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD,
    AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE,
    SILENCE_RMS_THRESHOLD, SILENCE_DURATION_MS,
    MAX_RECORDING_SECONDS, MIN_RECORDING_SECONDS,
)


class WakeWordListener:
    """
    Owns the microphone. Single loop that alternates between:
      1) listening for the wake word
      2) recording until the user stops talking
      3) handing PCM audio to a transcriber and firing callbacks

    Callbacks (any may be None) run in the listener thread — they should be
    quick or dispatch to the UI thread via Qt signals.
    """

    def __init__(self, on_wake=None, on_recording_complete=None,
                 on_transcript=None, transcriber=None):
        self.on_wake = on_wake                                # wake word fired
        self.on_recording_complete = on_recording_complete    # recording finished, about to transcribe
        self.on_transcript = on_transcript                    # transcribed text ready (may be "")
        self.transcriber = transcriber                        # obj with .transcribe(pcm_int16) -> str

        self.running = False
        self._thread = None

        # Load openwakeword model
        self.model = Model(
            wakeword_models=[WAKE_WORD_MODEL],
            inference_framework="onnx",
        )

        self.audio = pyaudio.PyAudio()

    def start(self):
        """Start listening for wake word in background thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print(f"Listening for wake word ({WAKE_WORD_MODEL})...")

    def stop(self):
        """Stop listening."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _listen_loop(self):
        """Main listening loop — runs in background thread."""
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK_SIZE,
        )

        try:
            while self.running:
                # ── WAKE-WORD PHASE ──
                audio_data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                self.model.predict(audio_array)

                detected = False
                for _model_name, scores in self.model.prediction_buffer.items():
                    if scores[-1] > WAKE_WORD_THRESHOLD:
                        print(f"Wake word detected! (score: {scores[-1]:.2f})")
                        self.model.reset()
                        detected = True
                        break
                if not detected:
                    continue

                # ── CALLBACK: ON WAKE ──
                self._safe_call(self.on_wake)

                # ── RECORDING PHASE ──
                pcm = self._record_until_silence(stream)

                # ── CALLBACK: RECORDING DONE (UI switches to "Thinking...") ──
                self._safe_call(self.on_recording_complete)

                # ── TRANSCRIBE ──
                text = ""
                if self.transcriber is not None and pcm is not None and len(pcm) > 0:
                    try:
                        text = self.transcriber.transcribe(pcm)
                    except Exception as e:
                        print(f"[voice] transcribe error: {e}")

                # ── CALLBACK: TRANSCRIPT READY ──
                self._safe_call(self.on_transcript, text)

                # Clear any wake-model state so our own audio doesn't re-trigger
                self.model.reset()
        finally:
            stream.stop_stream()
            stream.close()

    def _record_until_silence(self, stream):
        """
        Read from the already-open stream until we detect sustained silence
        (or hit the max-duration cap). Returns a flat int16 numpy array.
        """
        chunks = []
        silent_chunks = 0

        chunks_per_sec = AUDIO_SAMPLE_RATE / AUDIO_CHUNK_SIZE
        max_chunks = int(MAX_RECORDING_SECONDS * chunks_per_sec)
        min_chunks = int(MIN_RECORDING_SECONDS * chunks_per_sec)
        silence_chunks_needed = max(1, int((SILENCE_DURATION_MS / 1000.0) * chunks_per_sec))

        for i in range(max_chunks):
            if not self.running:
                break
            data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
            arr = np.frombuffer(data, dtype=np.int16)
            chunks.append(arr)

            # RMS in int16 space
            rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
            if rms < SILENCE_RMS_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if i >= min_chunks and silent_chunks >= silence_chunks_needed:
                break

        if not chunks:
            return None
        return np.concatenate(chunks)

    @staticmethod
    def _safe_call(cb, *args):
        if cb is None:
            return
        try:
            cb(*args)
        except Exception as e:
            print(f"[voice] callback error: {e}")

    def cleanup(self):
        """Clean up audio resources."""
        self.stop()
        self.audio.terminate()
