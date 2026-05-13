"""
DubStudio Pro — Video Player widget.
Supports original and dubbed video toggle, playback speed, seek.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QSlider, QSizePolicy, QComboBox,
                              QStackedWidget)

from app.ui.design import (ACCENT, BG, BG_CARD, BG_PANEL, BG_INPUT,
                            BORDER, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
                            FONT_SM, FONT_MD)
from app.ui.widgets import AnimatedButton


class VideoPlayer(QWidget):
    """
    Embedded video player with:
    - Original / Dubbed toggle
    - Play / Pause / Seek
    - Playback speed selector (0.5x … 2x)
    - Position signal for timeline sync
    """

    position_changed = pyqtSignal(int)   # ms
    duration_changed = pyqtSignal(int)   # ms

    SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── paths (initialised to empty — MUST init in __init__) ──
        self._original_path: str = ""
        self._dubbed_path: str = ""
        self._showing_dubbed: bool = False
        self._dragging_slider: bool = False

        self._build_ui()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── video surface ──
        self._video_widget = QVideoWidget()
        self._video_widget.setStyleSheet("background: #000;")
        self._video_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._video_widget, 1)

        # ── media player ──
        self._player = QMediaPlayer()
        self._audio = QAudioOutput()
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_widget)
        self._audio.setVolume(1.0)

        # ── seek bar ──
        self._seek_bar = QSlider(Qt.Orientation.Horizontal)
        self._seek_bar.setRange(0, 1000)
        self._seek_bar.setFixedHeight(18)
        self._seek_bar.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: #2A2A2A;
                height: 4px; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT};
                border-radius: 2px;
            }}
        """)
        root.addWidget(self._seek_bar)

        # ── controls bar ──
        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(8, 6, 8, 6)
        ctrl.setSpacing(8)

        # play/pause
        self._btn_play = AnimatedButton("▶  Play", "primary")
        self._btn_play.setFixedWidth(90)
        ctrl.addWidget(self._btn_play)

        # time label
        self._lbl_time = QLabel("0:00 / 0:00")
        self._lbl_time.setStyleSheet(
            f"color:{TEXT_SECONDARY}; font-size:{FONT_SM}px; background:transparent;")
        self._lbl_time.setFixedWidth(110)
        ctrl.addWidget(self._lbl_time)

        ctrl.addStretch()

        # speed label + combo
        speed_lbl = QLabel("Speed:")
        speed_lbl.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:{FONT_SM}px; background:transparent;")
        ctrl.addWidget(speed_lbl)

        self._speed_combo = QComboBox()
        for s in self.SPEEDS:
            self._speed_combo.addItem(f"{s}×", s)
        self._speed_combo.setCurrentIndex(2)   # 1.0×
        self._speed_combo.setFixedWidth(68)
        ctrl.addWidget(self._speed_combo)

        ctrl.addSpacing(12)

        # original / dubbed toggle
        self._btn_toggle = AnimatedButton("🎬 Original", "accent-outline")
        self._btn_toggle.setFixedWidth(110)
        self._btn_toggle.setEnabled(False)
        ctrl.addWidget(self._btn_toggle)

        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl)
        ctrl_widget.setStyleSheet(f"background:{BG_PANEL}; border-top:1px solid {BORDER};")
        root.addWidget(ctrl_widget)

        self.setStyleSheet(f"background:{BG};")

    # ──────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────
    def _connect_signals(self):
        self._btn_play.clicked.connect(self._toggle_play)
        self._seek_bar.sliderPressed.connect(self._on_slider_pressed)
        self._seek_bar.sliderReleased.connect(self._on_slider_released)
        self._seek_bar.sliderMoved.connect(self._on_slider_moved)
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self._btn_toggle.clicked.connect(self._toggle_version)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def load_job(self, original_path: str, dubbed_path: str = "") -> None:
        self._original_path = original_path
        self._dubbed_path = dubbed_path
        self._showing_dubbed = False
        self._btn_toggle.setEnabled(bool(dubbed_path))
        self._btn_toggle.setText("🎬 Original")
        self._load_current()

    def seek_to_ms(self, ms: int) -> None:
        self._player.setPosition(ms)

    def get_position_ms(self) -> int:
        return self._player.position()

    def get_duration_ms(self) -> int:
        return self._player.duration()

    def play(self):
        self._player.play()

    def pause(self):
        self._player.pause()

    def clear(self):
        self._player.stop()
        self._player.setSource(QUrl())
        self._original_path = ""
        self._dubbed_path = ""
        self._lbl_time.setText("0:00 / 0:00")
        self._seek_bar.setValue(0)
        self._btn_toggle.setEnabled(False)

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────
    def _load_current(self):
        path = self._dubbed_path if self._showing_dubbed else self._original_path
        if not path:
            return
        self._player.setSource(QUrl.fromLocalFile(path))

    def _toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _toggle_version(self):
        if not self._dubbed_path:
            return
        pos = self._player.position()
        was_playing = (self._player.playbackState() ==
                       QMediaPlayer.PlaybackState.PlayingState)
        self._player.pause()
        self._showing_dubbed = not self._showing_dubbed
        self._btn_toggle.setText("🎬 Dubbed" if self._showing_dubbed else "🎬 Original")
        self._load_current()
        # restore position after load
        def _restore():
            self._player.setPosition(pos)
            if was_playing:
                self._player.play()
        QTimer.singleShot(300, _restore)

    def _on_speed_changed(self, idx: int):
        rate = self._speed_combo.itemData(idx)
        if rate:
            self._player.setPlaybackRate(float(rate))

    def _on_slider_pressed(self):
        self._dragging_slider = True

    def _on_slider_released(self):
        self._dragging_slider = False
        dur = self._player.duration()
        if dur > 0:
            pos = int(self._seek_bar.value() / 1000 * dur)
            self._player.setPosition(pos)

    def _on_slider_moved(self, val: int):
        dur = self._player.duration()
        if dur > 0:
            ms = int(val / 1000 * dur)
            self._update_time_label(ms, dur)

    def _on_position_changed(self, pos_ms: int):
        if not self._dragging_slider:
            dur = self._player.duration()
            if dur > 0:
                self._seek_bar.setValue(int(pos_ms / dur * 1000))
            self._update_time_label(pos_ms, dur)
        self.position_changed.emit(pos_ms)

    def _on_duration_changed(self, dur_ms: int):
        self.duration_changed.emit(dur_ms)

    def _on_playback_state(self, state):
        playing = (state == QMediaPlayer.PlaybackState.PlayingState)
        self._btn_play.setText("⏸  Pause" if playing else "▶  Play")

    def _update_time_label(self, pos_ms: int, dur_ms: int):
        self._lbl_time.setText(
            f"{_fmt(pos_ms)} / {_fmt(dur_ms)}"
        )


def _fmt(ms: int) -> str:
    s = ms // 1000
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"