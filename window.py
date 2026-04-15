# Whiskers Desktop Pet - Window, Animations & Personality (PyQt6)

import math
import random
import sys
import time
from PyQt6.QtCore import Qt, QTimer, QPoint, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QTransform, QCursor
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from config import (
    CAT_SIZE, BOB_AMPLITUDE, BOB_SPEED, MARGIN_RIGHT, MARGIN_BOTTOM, CAT_IMAGE,
    MOVE_STEP, MOVE_ANIM_SPEED, MOVE_ANIM_STEPS,
    IDLE_SLEEP_TIMEOUT, ZOOMIE_MIN_INTERVAL, ZOOMIE_MAX_INTERVAL, ZOOMIE_SPEED,
)


class CatWindow(QWidget):
    # Signals for thread-safe UI updates
    wake_signal = pyqtSignal()
    move_signal = pyqtSignal(str)           # "left" or "right"
    thinking_signal = pyqtSignal()          # recording stopped, whisper running
    transcript_signal = pyqtSignal(str)     # transcription ready (may be empty)

    def __init__(self):
        self.app = QApplication(sys.argv)
        super().__init__()

        # Frameless, transparent, always-on-top, no taskbar icon
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        # Load cat images (normal and flipped for direction)
        self.pixmap_right = QPixmap(CAT_IMAGE).scaled(
            CAT_SIZE, CAT_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.pixmap_left = self.pixmap_right.transformed(QTransform().scale(-1, 1))

        # Cat label
        self.cat_label = QLabel(self)
        self.cat_label.setPixmap(self.pixmap_right)
        self.cat_label.setFixedSize(CAT_SIZE, CAT_SIZE)
        self.cat_label.setStyleSheet("background: transparent;")
        self.facing = "right"

        # Window size (extra room for bobbing + bubble)
        self.win_w = CAT_SIZE
        self.win_h = CAT_SIZE + BOB_AMPLITUDE * 2 + 40
        self.setFixedSize(self.win_w, self.win_h)

        # Position bottom-right
        self.screen_geo = self.app.primaryScreen().availableGeometry()
        x = self.screen_geo.width() - self.win_w - MARGIN_RIGHT
        y = self.screen_geo.height() - self.win_h - MARGIN_BOTTOM
        self.move(x, y)

        # Center cat label initially
        self.cat_base_y = (self.win_h - CAT_SIZE) // 2 + 10
        self.cat_label.move(0, self.cat_base_y)

        # Status bubble (hidden by default)
        self.bubble = QLabel(self)
        self.bubble.setStyleSheet(
            "background-color: rgba(0,0,0,180); color: white; border-radius: 10px;"
            "padding: 5px 10px; font-size: 12px;"
        )
        self.bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bubble.hide()

        # --- State machine ---
        self.frame = 0
        self.state = "idle"  # idle, alert, sleeping, zoomie, moving
        self._alert_frames = 0
        self._sleep_opacity = 1.0
        self._zzz_frame = 0

        # --- Movement ---
        self._move_target_x = 0
        self._move_dx = 0
        self._move_frames_left = 0

        # --- Personality timers ---
        self._last_interaction = time.time()
        self._wake_timestamps = []

        # Main animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate)
        self.timer.start(BOB_SPEED)

        # Idle/sleep checker (every 10s)
        self._idle_timer = QTimer()
        self._idle_timer.timeout.connect(self._check_idle)
        self._idle_timer.start(10000)

        # Zoomie timer — random interval
        self._zoomie_timer = QTimer()
        self._zoomie_timer.setSingleShot(True)
        self._zoomie_timer.timeout.connect(self._trigger_zoomie)
        self._schedule_next_zoomie()

        # Connect signals
        self.wake_signal.connect(self._on_wake)
        self.move_signal.connect(self._on_move)
        self.thinking_signal.connect(self._on_thinking)
        self.transcript_signal.connect(self._on_transcript)

        self.show()

        # macOS native: stay on top, all spaces, never hide, never steal focus
        self._ns_window = None
        try:
            from AppKit import NSApp, NSFloatingWindowLevel
            self._ns_window = NSApp.windows()[-1]
            self._ns_window.setLevel_(NSFloatingWindowLevel)
            self._ns_window.setCollectionBehavior_(
                1 << 0 | 1 << 3  # canJoinAllSpaces + fullScreenAuxiliary
            )
            self._ns_window.setHidesOnDeactivate_(False)
        except Exception as e:
            print(f"Could not set window level: {e}")

        self._enforce_timer = QTimer()
        self._enforce_timer.timeout.connect(self._enforce_top)
        self._enforce_timer.start(500)

    # ── Always on top ──

    def _enforce_top(self):
        try:
            if self._ns_window:
                self._ns_window.orderFrontRegardless()
        except Exception:
            pass

    # ── Animation dispatcher ──

    def _animate(self):
        self.frame += 1

        if self.state == "idle":
            offset_y = int(math.sin(self.frame * 0.15) * BOB_AMPLITUDE)
            self.cat_label.move(0, self.cat_base_y + offset_y)

        elif self.state == "alert":
            self._alert_frames += 1
            bounce = int(math.sin(self._alert_frames * 0.5) * BOB_AMPLITUDE * 3)
            self.cat_label.move(0, self.cat_base_y + bounce)
            if self._alert_frames > 30:
                self.set_state("idle")

        elif self.state == "sleeping":
            # Slow, deep breathing bob
            offset_y = int(math.sin(self.frame * 0.05) * BOB_AMPLITUDE * 0.5)
            self.cat_label.move(0, self.cat_base_y + offset_y)
            # Show zzz bubble
            self._zzz_frame += 1
            dots = "z" * (1 + (self._zzz_frame // 15) % 3)
            self.bubble.setText(dots)
            self.bubble.adjustSize()
            bx = (self.win_w - self.bubble.width()) // 2
            self.bubble.move(bx, 0)
            self.bubble.show()

        elif self.state == "moving":
            if self._move_frames_left > 0:
                self._move_frames_left -= 1
                # Ease-out movement
                progress = 1 - (self._move_frames_left / MOVE_ANIM_STEPS)
                ease = 1 - (1 - progress) ** 3  # cubic ease-out
                current_pos = self.pos()
                target_x = self._move_start_x + int(self._move_dx * ease)
                self.move(target_x, current_pos.y())
                # Bob while moving
                offset_y = int(math.sin(self.frame * 0.3) * BOB_AMPLITUDE * 2)
                self.cat_label.move(0, self.cat_base_y + offset_y)
            else:
                self.set_state("idle")

        elif self.state == "zoomie":
            self._zoomie_step()

    # ── Wake word ──

    def _on_wake(self):
        self._last_interaction = time.time()
        now = time.time()

        # Track rapid wake calls for annoyance
        self._wake_timestamps.append(now)
        self._wake_timestamps = [t for t in self._wake_timestamps if now - t < 10]

        if len(self._wake_timestamps) >= 4:
            # Annoyed! Too many calls in 10 seconds
            self.show_bubble("*looks away*", duration=2000)
            self._wake_timestamps.clear()
            return

        # Small chance Whiskers ignores you (10%)
        if random.random() < 0.10:
            self.show_bubble("...", duration=2000)
            self._face("left" if self.facing == "right" else "right")
            return

        if self.state == "sleeping":
            self.bubble.hide()

        self.set_state("alert")
        self.show_bubble("Listening...", duration=15000)  # held until recording ends

    def _on_thinking(self):
        """Recording finished — show 'Thinking...' while Whisper runs."""
        self._last_interaction = time.time()
        self.show_bubble("Thinking...", duration=15000)

    def _on_transcript(self, text):
        """Transcription ready — display it (truncated) then fall back to idle."""
        self._last_interaction = time.time()
        if text:
            display = text if len(text) <= 60 else text[:57] + "..."
            self.show_bubble(display, duration=4000)
        else:
            self.show_bubble("?", duration=2000)
        self.set_state("idle")

    # ── Movement ──

    def _on_move(self, direction):
        if self.state in ("zoomie", "moving"):
            return
        self._last_interaction = time.time()

        current_x = self.pos().x()
        if direction == "left":
            target_x = max(0, current_x - MOVE_STEP)
            self._face("left")
        else:
            target_x = min(self.screen_geo.width() - self.win_w, current_x + MOVE_STEP)
            self._face("right")

        self._move_start_x = current_x
        self._move_dx = target_x - current_x
        self._move_frames_left = MOVE_ANIM_STEPS
        self.set_state("moving")

    def _face(self, direction):
        if direction == "left" and self.facing != "left":
            self.cat_label.setPixmap(self.pixmap_left)
            self.facing = "left"
        elif direction == "right" and self.facing != "right":
            self.cat_label.setPixmap(self.pixmap_right)
            self.facing = "right"

    # ── Personality: Sleep ──

    def _check_idle(self):
        if self.state in ("sleeping", "zoomie", "moving", "alert"):
            return
        elapsed = (time.time() - self._last_interaction) * 1000
        if elapsed > IDLE_SLEEP_TIMEOUT:
            self.set_state("sleeping")
            self._zzz_frame = 0
            print("Whiskers fell asleep...")

    # ── Personality: Zoomies ──

    def _schedule_next_zoomie(self):
        interval = random.randint(ZOOMIE_MIN_INTERVAL, ZOOMIE_MAX_INTERVAL)
        self._zoomie_timer.start(interval)

    def _trigger_zoomie(self):
        if self.state == "sleeping":
            self._schedule_next_zoomie()
            return
        print("ZOOMIES!")
        self.show_bubble("!!!", duration=1500)
        self._zoomie_phase = "run_left"
        self._zoomie_orig_x = self.pos().x()
        self._zoomie_target_x = max(50, self._zoomie_orig_x - 400)
        self._face("left")
        self.set_state("zoomie")

    def _zoomie_step(self):
        current_x = self.pos().x()
        speed = 8

        if self._zoomie_phase == "run_left":
            new_x = current_x - speed
            if new_x <= self._zoomie_target_x:
                self._zoomie_phase = "run_back"
                self._face("right")
                new_x = self._zoomie_target_x
            self.move(new_x, self.pos().y())
            bounce = int(math.sin(self.frame * 0.8) * BOB_AMPLITUDE * 3)
            self.cat_label.move(0, self.cat_base_y + bounce)

        elif self._zoomie_phase == "run_back":
            new_x = current_x + speed
            if new_x >= self._zoomie_orig_x:
                new_x = self._zoomie_orig_x
                self.move(new_x, self.pos().y())
                self.set_state("idle")
                self._schedule_next_zoomie()
                return
            self.move(new_x, self.pos().y())
            bounce = int(math.sin(self.frame * 0.8) * BOB_AMPLITUDE * 3)
            self.cat_label.move(0, self.cat_base_y + bounce)

    # ── State ──

    def set_state(self, state):
        old_state = self.state
        self.state = state
        if state == "alert":
            self._alert_frames = 0
        if old_state == "sleeping" and state != "sleeping":
            self.bubble.hide()

    def show_bubble(self, text, duration=3000):
        self.bubble.setText(text)
        self.bubble.adjustSize()
        bx = (self.win_w - self.bubble.width()) // 2
        self.bubble.move(bx, 0)
        self.bubble.show()
        QTimer.singleShot(duration, self.bubble.hide)

    def run(self):
        sys.exit(self.app.exec())

    def stop(self):
        self.app.quit()
