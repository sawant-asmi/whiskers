#!/usr/bin/env python3
# Whiskers Desktop Pet - Main Entry Point

from window import CatWindow
from voice import WakeWordListener
from speech import Speaker
import brain
import actions


def main():
    print("Starting Whiskers...")
    cat = CatWindow()

    # TTS runs on its own thread; speak() is non-blocking.
    speaker = Speaker()

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
    )
    listener.start()

    try:
        cat.run()
    finally:
        listener.cleanup()
        speaker.shutdown()


if __name__ == "__main__":
    main()
