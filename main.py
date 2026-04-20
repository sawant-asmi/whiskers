#!/usr/bin/env python3
# Whiskers Desktop Pet - Main Entry Point

import threading

from window import CatWindow
from voice import WakeWordListener
from speech import Speaker
import brain
import actions


def main():
    print("Starting Whiskers...")
    cat = CatWindow()

    # Shared flag: set while Kokoro is playing audio, so the mic loop pauses.
    speaking_lock = threading.Event()

    # TTS runs on its own thread; speak() is non-blocking.
    speaker = Speaker(speaking_lock=speaking_lock)

    def on_transcript(text):
        print(f"[main] transcript: {text!r}")
        cat.transcript_signal.emit(text)

        action = brain.process(text)
        if action is None:
            return

        # Show Groq's reply in the bubble
        cat.response_signal.emit(action.response)

        # Execute command actions; chat/unknown just pass through
        actions.execute(action, cat)

        # Speak the response aloud via Kokoro
        speaker.speak(action.response)

    listener = WakeWordListener(
        on_wake=lambda: cat.wake_signal.emit(),
        on_recording_complete=lambda: cat.thinking_signal.emit(),
        on_transcript=on_transcript,
        speaking_lock=speaking_lock,
    )
    listener.start()

    try:
        cat.run()
    finally:
        listener.cleanup()
        speaker.shutdown()


if __name__ == "__main__":
    main()
