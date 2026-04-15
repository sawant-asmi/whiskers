# Whiskers Desktop Pet - Configuration

import os

# API Keys
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")

# Window settings
CAT_SIZE = 150          # Size of the cat image in pixels
BOB_AMPLITUDE = 5       # How many pixels the cat bobs up and down
BOB_SPEED = 80          # Milliseconds between animation frames
MARGIN_RIGHT = 30       # Pixels from right edge of screen
MARGIN_BOTTOM = 30      # Pixels from bottom edge of screen
MOVE_STEP = 100         # Pixels to move on "move left"/"move right"
MOVE_ANIM_SPEED = 16    # ms between movement animation frames (~60fps)
MOVE_ANIM_STEPS = 30    # number of frames for smooth slide

# Wake word settings
WAKE_WORD_MODEL = "hey_jarvis_v0.1"  # placeholder until custom "hey whiskers" model
WAKE_WORD_THRESHOLD = 0.5            # detection confidence threshold (0-1)
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 1280              # ~80ms at 16kHz

# Recording + transcription (Step 3)
WHISPER_MODEL = "base"               # whisper model size: tiny, base, small, medium, large
WHISPER_LANGUAGE = "en"              # force English; set None to auto-detect
SILENCE_RMS_THRESHOLD = 500          # int16 RMS below this counts as silence
SILENCE_DURATION_MS = 1000           # consecutive silence required to stop recording
MAX_RECORDING_SECONDS = 12           # hard cap on recording length
MIN_RECORDING_SECONDS = 0.5          # minimum before silence can trigger stop

# Text-to-speech (Step 6)
TTS_ENABLED = True                   # master switch
TTS_RATE = 185                       # words per minute (default ~200; lower = slower)
TTS_VOLUME = 1.0                     # 0.0 - 1.0
TTS_VOICE = None                     # case-insensitive name substring match (e.g. "Samantha",
                                     # "Karen", "Alex", "Daniel"); None = system default

# Personality settings
IDLE_SLEEP_TIMEOUT = 300000   # 5 minutes in ms before Whiskers falls asleep
ZOOMIE_MIN_INTERVAL = 600000  # 10 minutes minimum between zoomies
ZOOMIE_MAX_INTERVAL = 900000  # 15 minutes maximum between zoomies
ZOOMIE_SPEED = 12             # ms per frame during zoomies (~80fps)

# Paths
CAT_IMAGE = os.path.join(os.path.dirname(__file__), "cat.png")
