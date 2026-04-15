#!/usr/bin/env python3
# Whiskers Desktop Pet - Main Entry Point

from window import CatWindow
from voice import WakeWordListener
from transcribe import WhisperTranscriber
from speech import Speaker
import brain
import actions


def main():
    print("Starting Whiskers...")
    cat = CatWindow()

    # Load Whisper up-front so the first wake-word response isn't slow.
    transcriber = WhisperTranscriber()

    # TTS runs on its own thread; speak() is non-blocking.
    speaker = Speaker()

    def on_transcript(text):
        print(f"[main] transcript: {text!r}")
        cat.transcript_signal.emit(text)
        action = brain.process(text)
        if action is not None:
            actions.execute(action, cat)
            speaker.speak(action.response)

    listener = WakeWordListener(
        on_wake=lambda: cat.wake_signal.emit(),
        on_recording_complete=lambda: cat.thinking_signal.emit(),
        on_transcript=on_transcript,
        transcriber=transcriber,
    )
    listener.start()

    try:
        cat.run()
    finally:
        listener.cleanup()
        speaker.shutdown()


if __name__ == "__main__":
    main()
