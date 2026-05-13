"""
DubStudio Pro — Speaker Manager Dialog.
Rename speakers, assign colors, confirm profiles, reassign dialogues.
"""
from __future__ import annotations

from typing import Optional, Dict, Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QBrush
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QPushButton, QWidget, QScrollArea,
                              QFrame, QColorDialog, QMessageBox, QProgressBar,
                              QSizePolicy)

from app.models.job_model import Job
from app.models.speaker_model import DEFAULT_COLORS
from app.ui.design import (ACCENT, BG_PANEL, BG_CARD, BORDER, TEXT_PRIMARY,
                            TEXT_SECONDARY, TEXT_DIM, SUCCESS, WARNING, ERROR,
                            FONT_SM, FONT_XS, FONT_MD)
from app.ui.widgets import AnimatedButton, SectionHeader, Divider


class ColorSwatch(QWidget):
    """Clickable colour swatch."""

    clicked = pyqtSignal()

    def __init__(self, color: str = "#E8A020", parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(28, 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_color(self, color: str):
        self._color = color
        self.update()

    def color(self) -> str:
        return self._color

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(self._color)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 5, 5)
        p.end()

    def mousePressEvent(self, _):
        self.clicked.emit()


class SpeakerRow(QWidget):
    """One row per speaker: ID, color swatch, name edit, dialogue count."""

    def __init__(self, speaker_id: str, name: str, color: str,
                 dialogue_count: int, parent=None):
        super().__init__(parent)
        self._speaker_id = speaker_id
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(10)

        # colour swatch
        self._swatch = ColorSwatch(color)
        self._swatch.clicked.connect(self._pick_color)
        lay.addWidget(self._swatch)

        # id label
        id_lbl = QLabel(speaker_id)
        id_lbl.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:{FONT_XS}px; background:transparent;")
        id_lbl.setFixedWidth(90)
        lay.addWidget(id_lbl)

        # name edit
        self._name_edit = QLineEdit(name)
        self._name_edit.setPlaceholderText("Display name…")
        lay.addWidget(self._name_edit, 1)

        # dialogue count
        cnt_lbl = QLabel(f"{dialogue_count} lines")
        cnt_lbl.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:{FONT_XS}px; background:transparent;")
        cnt_lbl.setFixedWidth(60)
        lay.addWidget(cnt_lbl)

        self.setStyleSheet(f"""
            SpeakerRow {{
                background: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 5px;
            }}
        """)
        self.setObjectName("SpeakerRow")

    def _pick_color(self):
        dlg = QColorDialog(QColor(self._swatch.color()), self)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        if dlg.exec():
            self._swatch.set_color(dlg.selectedColor().name())

    def speaker_id(self) -> str:
        return self._speaker_id

    def display_name(self) -> str:
        return self._name_edit.text().strip() or self._speaker_id

    def color(self) -> str:
        return self._swatch.color()


class SpeakerManagerDialog(QDialog):
    """
    Modal dialog for managing speakers.
    - Rename each speaker
    - Assign color
    - Confirm speaker profiles (extract audio clips)
    """

    profiles_confirmed = pyqtSignal()   # re-emitted after profile extraction

    def __init__(self, job: Job, parent=None):
        super().__init__(parent)
        self._job = job
        self.setWindowTitle("Speaker Manager")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumWidth(520)
        self.setStyleSheet(f"background:{BG_PANEL};")
        self._rows: list[SpeakerRow] = []
        self._build_ui()
        self._center_on_parent(parent)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        root.addWidget(SectionHeader("Speaker Manager",
                                     "Rename speakers and confirm voice profiles"))
        root.addWidget(Divider())

        # ── speaker rows ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner.setStyleSheet(f"background:{BG_PANEL};")
        inner_lay = QVBoxLayout(inner)
        inner_lay.setSpacing(6)
        inner_lay.setContentsMargins(0, 0, 0, 0)

        # count dialogues per speaker
        counts: Dict[str, int] = {}
        for d in self._job.dialogues:
            counts[d.speaker_id] = counts.get(d.speaker_id, 0) + 1

        speaker_ids = sorted(set(d.speaker_id for d in self._job.dialogues))
        for i, spk in enumerate(speaker_ids):
            color = self._job.get_speaker_color(spk)
            name = self._job.get_speaker_name(spk)
            row = SpeakerRow(spk, name, color, counts.get(spk, 0))
            self._rows.append(row)
            inner_lay.addWidget(row)

        inner_lay.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll)

        root.addWidget(Divider())

        # ── profile status ──
        self._lbl_profile_status = QLabel(
            "Speaker profiles are built from dialogue audio clips.")
        self._lbl_profile_status.setStyleSheet(
            f"color:{TEXT_SECONDARY}; font-size:{FONT_XS}px; background:transparent;")
        self._lbl_profile_status.setWordWrap(True)
        root.addWidget(self._lbl_profile_status)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: #2A2A2A;
                border: 1px solid {BORDER};
                border-radius: 4px;
                height: 8px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: {ACCENT};
                border-radius: 4px;
            }}
        """)
        root.addWidget(self._progress)

        # ── buttons ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_confirm = AnimatedButton("✓ Confirm Profiles", "primary")
        self._btn_confirm.clicked.connect(self._confirm_profiles)
        btn_row.addWidget(self._btn_confirm)

        btn_row.addStretch()

        btn_cancel = AnimatedButton("Cancel", "ghost")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_save = AnimatedButton("Save Names & Colors", "accent-outline")
        btn_save.clicked.connect(self._save_and_close)
        btn_row.addWidget(btn_save)

        root.addLayout(btn_row)

    def _confirm_profiles(self):
        """Extract speaker profile WAVs from original audio."""
        self._apply_names_colors()
        self._btn_confirm.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(self._rows))
        self._progress.setValue(0)

        from app.job_manager import extract_speaker_profiles
        from PyQt6.QtWidgets import QApplication

        total_speakers = len(self._rows)

        def progress_cb(i, n, spk):
            self._progress.setValue(i + 1)
            self._lbl_profile_status.setText(f"Extracting: {spk}…")
            QApplication.processEvents()

        try:
            durations = extract_speaker_profiles(self._job, progress_cb)
            self._progress.setValue(total_speakers)
            msgs = []
            for spk, dur in durations.items():
                warn = " ⚠ < 6s" if dur < 6.0 else ""
                name = self._job.get_speaker_name(spk)
                msgs.append(f"  {name}: {dur:.1f}s{warn}")
            self._lbl_profile_status.setText(
                "Profiles confirmed:\n" + "\n".join(msgs))
            self.profiles_confirmed.emit()
        except FileNotFoundError as ex:
            self._lbl_profile_status.setText(f"⚠ {ex}")
            msg = QMessageBox(self)
            msg.setWindowModality(Qt.WindowModality.ApplicationModal)
            msg.setWindowTitle("Audio Not Found")
            msg.setText(str(ex))
            msg.exec()
        except Exception as ex:
            self._lbl_profile_status.setText(f"Error: {ex}")
        finally:
            self._btn_confirm.setEnabled(True)
            self._progress.setVisible(False)

    def _apply_names_colors(self):
        for row in self._rows:
            spk = row.speaker_id()
            self._job.speaker_names[spk] = row.display_name()
            self._job.speaker_colors[spk] = row.color()

    def _save_and_close(self):
        self._apply_names_colors()
        from app.job_manager import save_ui_state
        save_ui_state(self._job)
        self.accept()

    def _center_on_parent(self, parent):
        if parent:
            geo = parent.geometry()
            self.adjustSize()
            self.move(
                geo.center().x() - self.width() // 2,
                geo.center().y() - self.height() // 2,
            )