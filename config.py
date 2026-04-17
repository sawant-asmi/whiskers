# Whiskers Desktop Pet - Configuration

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file into os.environ

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

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

# Silero VAD settings
VAD_THRESHOLD = 0.5                  # speech probability threshold
VAD_SILENCE_LIMIT_CHUNKS = 10        # ~320ms of no-speech to stop (10 * 32ms)
VAD_PRE_SPEECH_TIMEOUT = 6.0         # seconds to wait for speech to start after wake
VAD_MAX_DURATION_SEC = 15            # hard cap on recording length
VAD_MIN_SPEECH_CHUNKS = 3            # minimum speech chunks before we accept it
VAD_CHUNK_SIZE = 512                 # 32ms per frame at 16kHz

# Moonshine STT
MOONSHINE_MODEL = "moonshine-base"

# Kokoro TTS
KOKORO_VOICE = "af_heart"            # warm, natural voice
KOKORO_SPEED = 1.0                   # 1.0 = normal speed
KOKORO_SAMPLE_RATE = 24000           # Kokoro outputs at 24kHz

# Personality settings
IDLE_SLEEP_TIMEOUT = 300000   # 5 minutes in ms before Whiskers falls asleep
ZOOMIE_MIN_INTERVAL = 600000  # 10 minutes minimum between zoomies
ZOOMIE_MAX_INTERVAL = 900000  # 15 minutes maximum between zoomies
ZOOMIE_SPEED = 12             # ms per frame during zoomies (~80fps)

# Paths
CAT_IMAGE = os.path.join(os.path.dirname(__file__), "cat.png")
