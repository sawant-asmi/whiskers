# Whiskers Desktop Pet - Wake Word + Silero VAD Recording + Moonshine STT
#
# Architecture:
#   1. Wake-word phase: continuously feed audio to openwakeword (PyAudio)
#   2. On wake: close PyAudio, switch to Silero VAD recording
#   3. VAD waits up to 6 seconds for speech to START
#   4. Once speech starts, capture frames; stop ~300ms after speech ends
#   5. Feed captured PCM to Moonshine for local transcription
#   6. Fire callbacks, reopen PyAudio for wake word
#
# Moonshine's Transcriber.transcribe_without_streaming() is used — it takes
# a float32 audio buffer and returns text. No mic conflict since we capture
# audio ourselves with PyAudio and pass the buffer.

from __future__ import annotations

import threading
import numpy as np
import pyaudio
import torch
from silero_vad import load_silero_vad
from openwakeword.model import Model
from moonshine_voice.transcriber import Transcriber
from moonshine_voice import get_model_for_language, ModelArch

from config import (
    WAKE_WORD_MODEL, WAKE_WORD_THRESHOLD,
    AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE,
    VAD_THRESHOLD, VAD_SILENCE_LIMIT_CHUNKS, VAD_PRE_SPEECH_TIMEOUT,
    VAD_MAX_DURATION_SEC, VAD_MIN_SPEECH_CHUNKS, VAD_CHUNK_SIZE,
    MOONSHINE_MODEL,
)


class WakeWordListener:
    """
    Owns the microphone. Alternates between wake-word detection and
    Silero VAD recording + Moonshine transcription.
    """

    def __init__(self, on_wake=None, on_recording_complete=None,
                 on_transcript=None, speaking_lock=None):
        self.on_wake = on_wake
        self.on_recording_complete = on_recording_complete
        self.on_transcript = on_transcript
        self._speaking_lock = speaking_lock  # set while TTS is playing

        self.running = False
        self._thread = None

        # Wake word model
        self.ww_model = Model(
            wakeword_models=[WAKE_WORD_MODEL],
            inference_framework="onnx",
        )

        # Silero VAD
        print("[voice] Loading Silero VAD...")
        self.vad_model = load_silero_vad()
        print("[voice] Silero VAD ready.")

        # Moonshine STT
        print(f"[voice] Loading Moonshine ({MOONSHINE_MODEL})...")
        model_name = MOONSHINE_MODEL.replace("moonshine-", "") + "-en"
        model_path, arch = get_model_for_language("en", ModelArch.BASE)
        self.moonshine = Transcriber(model_path, arch)
        print("[voice] Moonshine ready.")

        self.audio = pyaudio.PyAudio()

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print(f"Listening for wake word ({WAKE_WORD_MODEL})...")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _listen_loop(self):
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
                # Skip processing while Kokoro is speaking (prevents feedback loop)
                if self._speaking_lock and self._speaking_lock.is_set():
                    stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                    continue
                audio_data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                self.ww_model.predict(audio_array)

                detected = False
                for _name, scores in self.ww_model.prediction_buffer.items():
                    if scores[-1] > WAKE_WORD_THRESHOLD:
                        print(f"Wake word detected! (score: {scores[-1]:.2f})")
                        self.ww_model.reset()
                        detected = True
                        break
                if not detected:
                    continue

                # ── ON WAKE ──
                self._safe_call(self.on_wake)

                # Close wake stream, open VAD stream
                stream.stop_stream()
                stream.close()

                vad_stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=AUDIO_SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=VAD_CHUNK_SIZE,
                )

                # ── VAD RECORDING ──
                pcm_float = self._vad_record(vad_stream)

                vad_stream.stop_stream()
                vad_stream.close()

                # ── RECORDING DONE ──
                self._safe_call(self.on_recording_complete)

                # ── MOONSHINE TRANSCRIPTION ──
                text = ""
                if pcm_float is not None and len(pcm_float) > 0:
                    try:
                        result = self.moonshine.transcribe_without_streaming(
                            pcm_float.tolist(), sample_rate=AUDIO_SAMPLE_RATE
                        )
                        # Extract text from transcript lines
                        text = str(result).strip()
                        print(f"[voice] moonshine: {text!r}")
                    except Exception as e:
                        print(f"[voice] moonshine error: {e}")

                # ── TRANSCRIPT CALLBACK ──
                self._safe_call(self.on_transcript, text)

                # Reopen wake word stream
                stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=AUDIO_SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=AUDIO_CHUNK_SIZE,
                )
                self.ww_model.reset()
        finally:
            stream.stop_stream()
            stream.close()

    def _vad_record(self, stream) -> np.ndarray | None:
        """
        Silero VAD recording. Waits up to VAD_PRE_SPEECH_TIMEOUT for speech
        to start, then captures until ~300ms of silence after speech ends.
        Returns float32 numpy array normalised to [-1, 1].
        """
        self.vad_model.reset_states()

        all_chunks = []
        silence_count = 0
        speech_detected = False
        total_speech_chunks = 0

        # Phase 1: wait for speech to start (up to 6 seconds)
        pre_speech_max = int(VAD_PRE_SPEECH_TIMEOUT * AUDIO_SAMPLE_RATE / VAD_CHUNK_SIZE)
        max_chunks = int(VAD_MAX_DURATION_SEC * AUDIO_SAMPLE_RATE / VAD_CHUNK_SIZE)

        for i in range(max_chunks):
            if not self.running:
                break

            data = stream.read(VAD_CHUNK_SIZE, exception_on_overflow=False)
            chunk_int16 = np.frombuffer(data, dtype=np.int16)
            chunk_float = torch.from_numpy(
                chunk_int16.astype(np.float32) / 32768.0
            )

            speech_prob = self.vad_model(chunk_float, AUDIO_SAMPLE_RATE).item()

            if speech_prob >= VAD_THRESHOLD:
                # Speech frame
                speech_detected = True
                total_speech_chunks += 1
                silence_count = 0
                all_chunks.append(chunk_int16)
            else:
                if speech_detected:
                    # Silence after speech started — count toward end
                    silence_count += 1
                    all_chunks.append(chunk_int16)

                    if silence_count >= VAD_SILENCE_LIMIT_CHUNKS:
                        # ~320ms of silence → done
                        break
                else:
                    # Still waiting for speech to start
                    if i >= pre_speech_max:
                        print("[voice] no speech detected within timeout")
                        return None

        if not speech_detected or total_speech_chunks < VAD_MIN_SPEECH_CHUNKS:
            print("[voice] no speech detected in VAD window")
            return None

        pcm_int16 = np.concatenate(all_chunks)
        duration_ms = len(pcm_int16) / AUDIO_SAMPLE_RATE * 1000
        print(f"[voice] VAD captured {duration_ms:.0f}ms "
              f"({total_speech_chunks} speech frames)")

        # Return float32 normalised (Moonshine expects this)
        return pcm_int16.astype(np.float32) / 32768.0

    @staticmethod
    def _safe_call(cb, *args):
        if cb is None:
            return
        try:
            cb(*args)
        except Exception as e:
            print(f"[voice] callback error: {e}")

    def cleanup(self):
        self.stop()
        try:
            self.moonshine.close()
        except Exception:
            pass
        self.audio.terminate()
