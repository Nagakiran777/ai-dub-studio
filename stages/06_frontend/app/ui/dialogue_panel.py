"""
DubStudio Pro — Dialogue Editor Panel.
Undo/redo stack, prev/next navigation, reset, speaker reassign.
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Optional, Callable, List, Deque, Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QLineEdit, QTextEdit, QCheckBox, QComboBox,
                              QPushButton, QGroupBox, QSizePolicy, QFrame,
                              QScrollArea)

from app.models.job_model import Dialogue, Job
from app.ui.design import (ACCENT, BG_PANEL, BG_CARD, BG_INPUT, BORDER,
                            TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM, FONT_SM,
                            FONT_MD, FONT_XS, SUCCESS, WARNING, ERROR)
from app.ui.widgets import AnimatedButton, SectionHeader, Divider, StatusBadge


MAX_UNDO = 50


class _UndoEntry:
    def __init__(self, dialogue_id: str, before: dict, after: dict):
        self.dialogue_id = dialogue_id
        self.before = before
        self.after = after


class DialoguePanel(QWidget):
    """
    Right-panel dialogue editor.
    Signals:
      dialogue_changed(id) — something was edited
      selection_changed(id) — user clicked prev/next
    """

    dialogue_changed = pyqtSignal(str)
    selection_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job: Optional[Job] = None
        self._current_id: Optional[str] = None
        self._updating_ui = False

        self._undo_stack: Deque[_UndoEntry] = deque(maxlen=MAX_UNDO)
        self._redo_stack: Deque[_UndoEntry] = deque(maxlen=MAX_UNDO)

        # store original values from JSON for reset
        self._originals: dict = {}   # dialogue_id → snapshot dict

        self._build_ui()

    # ──────────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)

        # ── header ──
        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        self._lbl_header = SectionHeader("Dialogue Editor")
        header_row.addWidget(self._lbl_header)
        root.addLayout(header_row)

        # ── nav row ──
        nav = QHBoxLayout()
        nav.setSpacing(6)
        self._btn_prev = AnimatedButton("◀ Prev", "ghost")
        self._btn_prev.setFixedHeight(26)
        self._btn_next = AnimatedButton("Next ▶", "ghost")
        self._btn_next.setFixedHeight(26)
        self._lbl_idx = QLabel("— / —")
        self._lbl_idx.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_idx.setStyleSheet(
            f"color:{TEXT_SECONDARY}; font-size:{FONT_XS}px; background:transparent;")
        self._lbl_idx.setFixedWidth(60)
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._lbl_idx)
        nav.addWidget(self._btn_next)
        nav.addStretch()

        # undo / redo / reset
        self._btn_undo = AnimatedButton("↩ Undo", "ghost")
        self._btn_undo.setFixedHeight(26)
        self._btn_undo.setEnabled(False)
        self._btn_redo = AnimatedButton("↪ Redo", "ghost")
        self._btn_redo.setFixedHeight(26)
        self._btn_redo.setEnabled(False)
        self._btn_reset = AnimatedButton("↺ Reset", "ghost")
        self._btn_reset.setFixedHeight(26)
        nav.addWidget(self._btn_undo)
        nav.addWidget(self._btn_redo)
        nav.addWidget(self._btn_reset)
        root.addLayout(nav)

        root.addWidget(Divider())

        # ── scroll area for fields ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background:transparent; border:none;")

        inner = QWidget()
        inner.setStyleSheet("background:transparent;")
        form = QVBoxLayout(inner)
        form.setContentsMargins(0, 0, 4, 0)
        form.setSpacing(8)

        # ── dialogue id / speaker ──
        id_row = QHBoxLayout()
        self._lbl_id = QLabel("—")
        self._lbl_id.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:{FONT_XS}px; background:transparent;")
        id_row.addWidget(self._lbl_id)
        id_row.addStretch()
        self._badge_status = StatusBadge("pending")
        id_row.addWidget(self._badge_status)
        form.addLayout(id_row)

        # speaker
        spk_row = QHBoxLayout()
        spk_lbl = QLabel("Speaker:")
        spk_lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{FONT_XS}px; background:transparent;")
        spk_lbl.setFixedWidth(70)
        self._combo_speaker = QComboBox()
        self._combo_speaker.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        spk_row.addWidget(spk_lbl)
        spk_row.addWidget(self._combo_speaker)
        form.addLayout(spk_row)

        # timing
        time_row = QHBoxLayout()
        time_row.setSpacing(6)
        t_lbl = QLabel("Timing:")
        t_lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{FONT_XS}px; background:transparent;")
        t_lbl.setFixedWidth(70)
        self._edit_start = QLineEdit()
        self._edit_start.setPlaceholderText("start ms")
        self._edit_start.setFixedWidth(80)
        dash = QLabel("–")
        dash.setStyleSheet(f"color:{TEXT_DIM}; background:transparent;")
        self._edit_end = QLineEdit()
        self._edit_end.setPlaceholderText("end ms")
        self._edit_end.setFixedWidth(80)
        time_row.addWidget(t_lbl)
        time_row.addWidget(self._edit_start)
        time_row.addWidget(dash)
        time_row.addWidget(self._edit_end)
        time_row.addStretch()
        form.addLayout(time_row)

        # emotion / intensity
        em_row = QHBoxLayout()
        em_lbl = QLabel("Emotion:")
        em_lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{FONT_XS}px; background:transparent;")
        em_lbl.setFixedWidth(70)
        self._lbl_emotion = QLabel("—")
        self._lbl_emotion.setStyleSheet(
            f"color:{ACCENT}; font-size:{FONT_XS}px; background:transparent;")
        em_row.addWidget(em_lbl)
        em_row.addWidget(self._lbl_emotion)
        em_row.addStretch()
        self._lbl_intensity = QLabel("")
        self._lbl_intensity.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:{FONT_XS}px; background:transparent;")
        em_row.addWidget(self._lbl_intensity)
        form.addLayout(em_row)

        form.addWidget(Divider())

        # original text
        form.addWidget(self._field_label("Original text:"))
        self._edit_text = QTextEdit()
        self._edit_text.setFixedHeight(70)
        self._edit_text.setPlaceholderText("Original transcribed text…")
        form.addWidget(self._edit_text)

        # translation
        form.addWidget(self._field_label("Translation / Dub text:"))
        self._edit_translation = QTextEdit()
        self._edit_translation.setFixedHeight(70)
        self._edit_translation.setPlaceholderText("Translation or dubbed text…")
        form.addWidget(self._edit_translation)

        # lang row
        lang_row = QHBoxLayout()
        lang_row.setSpacing(6)
        self._combo_src_lang = QComboBox()
        self._combo_src_lang.setFixedWidth(80)
        arrow_lbl = QLabel("→")
        arrow_lbl.setStyleSheet(f"color:{TEXT_DIM}; background:transparent;")
        self._combo_tgt_lang = QComboBox()
        self._combo_tgt_lang.setFixedWidth(80)
        for combo in [self._combo_src_lang, self._combo_tgt_lang]:
            for lang in ["en", "es", "fr", "de", "it", "pt", "ja", "zh", "ko",
                         "ar", "hi", "ru", "tr", "nl", "pl"]:
                combo.addItem(lang)
        lang_row.addWidget(QLabel("Lang:"))
        lang_row.addWidget(self._combo_src_lang)
        lang_row.addWidget(arrow_lbl)
        lang_row.addWidget(self._combo_tgt_lang)
        lang_row.addStretch()
        form.addLayout(lang_row)

        # dub enabled
        dub_row = QHBoxLayout()
        self._chk_dub = QCheckBox("Include in dub")
        self._chk_dub.setChecked(True)
        dub_row.addWidget(self._chk_dub)
        dub_row.addStretch()
        form.addLayout(dub_row)

        form.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        self.setStyleSheet(f"background:{BG_PANEL};")

        # ── connect ──
        self._btn_prev.clicked.connect(self._go_prev)
        self._btn_next.clicked.connect(self._go_next)
        self._btn_undo.clicked.connect(self.undo)
        self._btn_redo.clicked.connect(self.redo)
        self._btn_reset.clicked.connect(self._reset_current)
        self._edit_start.editingFinished.connect(self._on_timing_changed)
        self._edit_end.editingFinished.connect(self._on_timing_changed)
        self._edit_text.textChanged.connect(self._on_text_changed)
        self._edit_translation.textChanged.connect(self._on_translation_changed)
        self._combo_speaker.currentIndexChanged.connect(self._on_speaker_changed)
        self._combo_src_lang.currentIndexChanged.connect(self._on_lang_changed)
        self._combo_tgt_lang.currentIndexChanged.connect(self._on_lang_changed)
        self._chk_dub.toggled.connect(self._on_dub_toggled)

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{TEXT_SECONDARY}; font-size:{FONT_XS}px; background:transparent;")
        return lbl

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def set_job(self, job: Job):
        self._job = job
        self._originals = {d.id: self._snapshot(d) for d in job.dialogues}
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_undo_buttons()
        # populate speaker combo
        self._updating_ui = True
        self._combo_speaker.clear()
        for spk in sorted(set(d.speaker_id for d in job.dialogues)):
            name = job.get_speaker_name(spk)
            self._combo_speaker.addItem(f"{name} ({spk})", spk)
        self._updating_ui = False

    def select_dialogue(self, dialogue_id: str):
        if not self._job:
            return
        d = self._job.get_dialogue(dialogue_id)
        if not d:
            return
        self._current_id = dialogue_id
        self._populate(d)
        self._update_nav_label()

    def clear(self):
        self._job = None
        self._current_id = None
        self._lbl_id.setText("—")
        self._edit_text.clear()
        self._edit_translation.clear()
        self._lbl_emotion.setText("—")
        self._lbl_intensity.setText("")

    # ── undo / redo ──
    def push_undo(self, dialogue_id: str, before: dict, after: dict):
        self._undo_stack.append(_UndoEntry(dialogue_id, before, after))
        self._redo_stack.clear()
        self._update_undo_buttons()

    def undo(self):
        if not self._undo_stack or not self._job:
            return
        entry = self._undo_stack.pop()
        d = self._job.get_dialogue(entry.dialogue_id)
        if d:
            self._apply_snapshot(d, entry.before)
            self._redo_stack.append(entry)
            if entry.dialogue_id == self._current_id:
                self._populate(d)
            self.dialogue_changed.emit(entry.dialogue_id)
        self._update_undo_buttons()

    def redo(self):
        if not self._redo_stack or not self._job:
            return
        entry = self._redo_stack.pop()
        d = self._job.get_dialogue(entry.dialogue_id)
        if d:
            self._apply_snapshot(d, entry.after)
            self._undo_stack.append(entry)
            if entry.dialogue_id == self._current_id:
                self._populate(d)
            self.dialogue_changed.emit(entry.dialogue_id)
        self._update_undo_buttons()

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────
    def _populate(self, d: Dialogue):
        self._updating_ui = True
        self._lbl_id.setText(d.id)
        self._edit_start.setText(str(d.start_ms))
        self._edit_end.setText(str(d.end_ms))
        self._edit_text.setPlainText(d.text)
        self._edit_translation.setPlainText(d.translation or d.text)
        self._lbl_emotion.setText(d.emotion or "—")
        self._lbl_intensity.setText(f"{d.intensity:.2f}" if d.emotion else "")
        self._chk_dub.setChecked(d.dub_enabled)
        # speaker combo
        for i in range(self._combo_speaker.count()):
            if self._combo_speaker.itemData(i) == d.speaker_id:
                self._combo_speaker.setCurrentIndex(i)
                break
        # lang combos
        for combo, val in [(self._combo_src_lang, d.source_lang),
                           (self._combo_tgt_lang, d.target_lang)]:
            idx = combo.findText(val)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        # status badge
        if d.dub_enabled:
            self._badge_status.set_status("completed" if d.translation else "pending")
        else:
            self._badge_status.set_status("skipped")
        self._updating_ui = False

    def _update_nav_label(self):
        if not self._job or not self._current_id:
            self._lbl_idx.setText("— / —")
            return
        ids = [d.id for d in self._job.dialogues]
        try:
            idx = ids.index(self._current_id)
            self._lbl_idx.setText(f"{idx+1} / {len(ids)}")
        except ValueError:
            self._lbl_idx.setText("— / —")

    def _go_prev(self):
        if not self._job or not self._current_id:
            return
        ids = [d.id for d in self._job.dialogues]
        try:
            idx = ids.index(self._current_id)
            if idx > 0:
                self.selection_changed.emit(ids[idx - 1])
        except ValueError:
            pass

    def _go_next(self):
        if not self._job or not self._current_id:
            return
        ids = [d.id for d in self._job.dialogues]
        try:
            idx = ids.index(self._current_id)
            if idx < len(ids) - 1:
                self.selection_changed.emit(ids[idx + 1])
        except ValueError:
            pass

    def _reset_current(self):
        if not self._job or not self._current_id:
            return
        d = self._job.get_dialogue(self._current_id)
        if not d:
            return
        orig = self._originals.get(self._current_id)
        if not orig:
            return
        before = self._snapshot(d)
        self._apply_snapshot(d, orig)
        after = self._snapshot(d)
        self.push_undo(d.id, before, after)
        self._populate(d)
        self.dialogue_changed.emit(d.id)

    def _update_undo_buttons(self):
        self._btn_undo.setEnabled(bool(self._undo_stack))
        self._btn_redo.setEnabled(bool(self._redo_stack))

    # ── change handlers ──
    def _on_timing_changed(self):
        if self._updating_ui or not self._job or not self._current_id:
            return
        d = self._job.get_dialogue(self._current_id)
        if not d:
            return
        before = self._snapshot(d)
        try:
            d.start_ms = int(self._edit_start.text())
        except ValueError:
            pass
        try:
            d.end_ms = int(self._edit_end.text())
        except ValueError:
            pass
        after = self._snapshot(d)
        self.push_undo(d.id, before, after)
        self.dialogue_changed.emit(d.id)

    def _on_text_changed(self):
        if self._updating_ui or not self._job or not self._current_id:
            return
        d = self._job.get_dialogue(self._current_id)
        if not d:
            return
        before = self._snapshot(d)
        d.text = self._edit_text.toPlainText()
        after = self._snapshot(d)
        self.push_undo(d.id, before, after)
        self.dialogue_changed.emit(d.id)

    def _on_translation_changed(self):
        if self._updating_ui or not self._job or not self._current_id:
            return
        d = self._job.get_dialogue(self._current_id)
        if not d:
            return
        before = self._snapshot(d)
        d.translation = self._edit_translation.toPlainText()
        after = self._snapshot(d)
        self.push_undo(d.id, before, after)
        self.dialogue_changed.emit(d.id)

    def _on_speaker_changed(self, _):
        if self._updating_ui or not self._job or not self._current_id:
            return
        d = self._job.get_dialogue(self._current_id)
        if not d:
            return
        before = self._snapshot(d)
        new_spk = self._combo_speaker.currentData()
        if new_spk and new_spk != d.speaker_id:
            d.speaker_id = new_spk
            after = self._snapshot(d)
            self.push_undo(d.id, before, after)
            self.dialogue_changed.emit(d.id)

    def _on_lang_changed(self, _):
        if self._updating_ui or not self._job or not self._current_id:
            return
        d = self._job.get_dialogue(self._current_id)
        if not d:
            return
        d.source_lang = self._combo_src_lang.currentText()
        d.target_lang = self._combo_tgt_lang.currentText()
        self.dialogue_changed.emit(d.id)

    def _on_dub_toggled(self, checked: bool):
        if self._updating_ui or not self._job or not self._current_id:
            return
        d = self._job.get_dialogue(self._current_id)
        if not d:
            return
        d.dub_enabled = checked
        self._badge_status.set_status("completed" if checked and d.translation else
                                      "pending" if checked else "skipped")
        self.dialogue_changed.emit(d.id)

    # ── snapshot helpers ──
    @staticmethod
    def _snapshot(d: Dialogue) -> dict:
        return {
            "start_ms": d.start_ms,
            "end_ms": d.end_ms,
            "text": d.text,
            "translation": d.translation,
            "speaker_id": d.speaker_id,
            "dub_enabled": d.dub_enabled,
            "source_lang": d.source_lang,
            "target_lang": d.target_lang,
            "emotion": d.emotion,
            "intensity": d.intensity,
        }

    @staticmethod
    def _apply_snapshot(d: Dialogue, snap: dict):
        d.start_ms = snap["start_ms"]
        d.end_ms = snap["end_ms"]
        d.text = snap["text"]
        d.translation = snap["translation"]
        d.speaker_id = snap["speaker_id"]
        d.dub_enabled = snap["dub_enabled"]
        d.source_lang = snap["source_lang"]
        d.target_lang = snap["target_lang"]
        d.emotion = snap.get("emotion", d.emotion)
        d.intensity = snap.get("intensity", d.intensity)