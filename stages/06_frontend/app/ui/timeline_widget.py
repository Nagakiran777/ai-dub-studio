"""
DubStudio Pro — Timeline widget.
- Pinned speaker label column (never scrolls)
- Scrollable dialogue blocks with waveform drawn inside
- Golden playhead
- Right-click context menu: Merge with next, Split at midpoint
- Colour-coded speaker rows
"""
from __future__ import annotations

import math
from typing import Optional, List, Dict, Callable

from PyQt6.QtCore import (Qt, QRect, QPoint, QRectF, pyqtSignal, QTimer,
                           QPointF)
from PyQt6.QtGui import (QPainter, QPen, QBrush, QColor, QFont,
                         QFontMetrics, QLinearGradient, QPolygon,
                         QContextMenuEvent, QMouseEvent)
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout,
                              QScrollArea, QScrollBar, QMenu, QSizePolicy,
                              QAbstractScrollArea)

from app.models.job_model import Dialogue, Job
from app.ui.design import (ACCENT, BG, BORDER, TEXT_PRIMARY, TEXT_SECONDARY,
                            TEXT_DIM, TIMELINE_BG, PLAYHEAD_COLOR,
                            BG_PANEL, BG_CARD, FONT_SM, FONT_XS)


ROW_H = 56          # px per speaker row
RULER_H = 24        # px for time ruler
LABEL_W = 110       # px for pinned label column
PX_PER_SEC = 80.0   # horizontal scale (pixels per second of video)
MIN_BLOCK_W = 4     # minimum block width in px


# ──────────────────────────────────────────────────────────────────────────────
# Ruler widget (top of scroll area)
# ──────────────────────────────────────────────────────────────────────────────
class RulerWidget(QWidget):
    def __init__(self, total_ms: int = 60000, parent=None):
        super().__init__(parent)
        self._total_ms = total_ms
        self.setFixedHeight(RULER_H)

    def set_total_ms(self, ms: int):
        self._total_ms = ms
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        total_s = self._total_ms / 1000

        p.fillRect(self.rect(), QColor(BG_PANEL))

        pen = QPen(QColor(BORDER))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawLine(0, RULER_H - 1, w, RULER_H - 1)

        # choose tick interval
        tick_s = 1
        for t in [1, 2, 5, 10, 30, 60]:
            if PX_PER_SEC * t >= 40:
                tick_s = t
                break

        font = QFont("Segoe UI", FONT_XS)
        p.setFont(font)
        p.setPen(QColor(TEXT_DIM))

        s = 0
        while s <= total_s + tick_s:
            x = int(s * PX_PER_SEC)
            if x > w:
                break
            p.setPen(QColor(BORDER))
            p.drawLine(x, RULER_H - 8, x, RULER_H - 1)
            if s % max(tick_s, 5) == 0 or tick_s <= 2:
                p.setPen(QColor(TEXT_DIM))
                lbl = _fmt_s(s)
                p.drawText(x + 3, RULER_H - 6, lbl)
            s += tick_s
        p.end()


# ──────────────────────────────────────────────────────────────────────────────
# DialogueBlock drawing canvas
# ──────────────────────────────────────────────────────────────────────────────
class TimelineCanvas(QWidget):
    """
    Draws all dialogue blocks. Emits signals on click, drag, right-click.
    """

    dialogue_clicked = pyqtSignal(str)           # dialogue_id
    dialogue_drag_end = pyqtSignal(str, int, int)  # id, new_start_ms, new_end_ms
    merge_requested = pyqtSignal(str)             # dialogue_id (merge with next)
    split_requested = pyqtSignal(str)             # dialogue_id (split at midpoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job: Optional[Job] = None
        self._speaker_order: List[str] = []
        self._waveform_data: Dict[str, List[float]] = {}   # dialogue_id → samples
        self._selected_id: Optional[str] = None
        self._playhead_ms: int = 0
        self._total_ms: int = 60000
        self._drag_id: Optional[str] = None
        self._drag_edge: Optional[str] = None   # "left" | "right" | "move"
        self._drag_start_x: int = 0
        self._drag_orig_start: int = 0
        self._drag_orig_end: int = 0
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ── public API ────────────────────────────────────────────────────────
    def set_job(self, job: Job):
        self._job = job
        self._rebuild_speaker_order()
        self._recalc_size()
        self.update()

    def set_selected(self, dialogue_id: Optional[str]):
        self._selected_id = dialogue_id
        self.update()

    def set_playhead(self, ms: int):
        self._playhead_ms = ms
        self.update()

    def set_waveform(self, dialogue_id: str, samples: List[float]):
        self._waveform_data[dialogue_id] = samples
        self.update()

    def selected_id(self) -> Optional[str]:
        return self._selected_id

    # ── internals ─────────────────────────────────────────────────────────
    def _rebuild_speaker_order(self):
        if not self._job:
            return
        seen = []
        for d in self._job.dialogues:
            if d.speaker_id not in seen:
                seen.append(d.speaker_id)
        self._speaker_order = seen
        self._total_ms = max(
            (d.end_ms for d in self._job.dialogues), default=60000
        ) + 2000

    def _recalc_size(self):
        w = max(800, int(self._total_ms / 1000 * PX_PER_SEC) + 100)
        h = max(200, len(self._speaker_order) * ROW_H + 40)
        self.setMinimumSize(w, h)

    def _row_y(self, speaker_id: str) -> int:
        idx = self._speaker_order.index(speaker_id) if speaker_id in self._speaker_order else 0
        return idx * ROW_H

    def _ms_to_x(self, ms: int) -> int:
        return int(ms / 1000 * PX_PER_SEC)

    def _x_to_ms(self, x: int) -> int:
        return max(0, int(x / PX_PER_SEC * 1000))

    def _block_rect(self, d: Dialogue) -> QRect:
        x = self._ms_to_x(d.start_ms)
        y = self._row_y(d.speaker_id) + 4
        w = max(MIN_BLOCK_W, self._ms_to_x(d.end_ms) - x)
        h = ROW_H - 8
        return QRect(x, y, w, h)

    def paintEvent(self, _):
        if not self._job:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(TIMELINE_BG))

        # row backgrounds
        for i, spk in enumerate(self._speaker_order):
            ry = i * ROW_H
            alt = QColor("#0F0F0F") if i % 2 == 0 else QColor("#131313")
            p.fillRect(0, ry, self.width(), ROW_H, alt)
            # separator line
            p.setPen(QPen(QColor(BORDER), 1))
            p.drawLine(0, ry + ROW_H - 1, self.width(), ry + ROW_H - 1)

        # dialogue blocks
        for d in self._job.dialogues:
            if d.speaker_id not in self._speaker_order:
                continue
            self._draw_block(p, d)

        # skipped / disabled
        for d in self._job.dialogues:
            if not d.dub_enabled and d.speaker_id in self._speaker_order:
                self._draw_block_disabled(p, d)

        # playhead
        ph_x = self._ms_to_x(self._playhead_ms)
        p.setPen(QPen(QColor(PLAYHEAD_COLOR), 2))
        p.drawLine(ph_x, 0, ph_x, self.height())
        # triangle tip
        tri = QPolygon([
            QPoint(ph_x, 0),
            QPoint(ph_x - 6, -10),
            QPoint(ph_x + 6, -10),
        ])
        p.setBrush(QBrush(QColor(PLAYHEAD_COLOR)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(tri)
        p.end()

    def _draw_block(self, p: QPainter, d: Dialogue):
        rect = self._block_rect(d)
        color_hex = self._job.get_speaker_color(d.speaker_id) if self._job else ACCENT
        color = QColor(color_hex)
        selected = (d.id == self._selected_id)

        # background gradient
        grad = QLinearGradient(QPointF(rect.left(), rect.top()),
                               QPointF(rect.left(), rect.bottom()))
        grad.setColorAt(0, color.lighter(130 if selected else 110))
        grad.setColorAt(1, color.darker(150))
        p.setBrush(QBrush(grad))

        border_color = QColor(ACCENT) if selected else color.lighter(160)
        pen = QPen(border_color, 2 if selected else 1)
        p.setPen(pen)
        p.drawRoundedRect(rect, 4, 4)

        # waveform
        samples = self._waveform_data.get(d.id, [])
        if samples and rect.width() > 20:
            self._draw_waveform(p, rect, samples, color)

        # label
        if rect.width() > 30:
            p.setPen(QColor("#FFFFFF" if selected else "#CCCCCC"))
            font = QFont("Segoe UI", FONT_XS)
            font.setBold(selected)
            p.setFont(font)
            fm = QFontMetrics(font)
            text = d.text
            text = fm.elidedText(text, Qt.TextElideMode.ElideRight, rect.width() - 8)
            p.drawText(rect.adjusted(4, 4, -4, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, text)

    def _draw_block_disabled(self, p: QPainter, d: Dialogue):
        rect = self._block_rect(d)
        p.setBrush(QBrush(QColor("#1A1A1A")))
        p.setPen(QPen(QColor(BORDER), 1, Qt.PenStyle.DashLine))
        p.drawRoundedRect(rect, 4, 4)
        if rect.width() > 30:
            p.setPen(QColor(TEXT_DIM))
            font = QFont("Segoe UI", FONT_XS)
            p.setFont(font)
            fm = QFontMetrics(font)
            text = fm.elidedText(d.text, Qt.TextElideMode.ElideRight, rect.width() - 8)
            # strikethrough via line
            p.drawText(rect.adjusted(4, 4, -4, -4),
                       Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, text)
            mid_y = rect.top() + rect.height() // 2
            p.setPen(QPen(QColor(TEXT_DIM), 1))
            p.drawLine(rect.left() + 4, mid_y, rect.right() - 4, mid_y)

    def _draw_waveform(self, p: QPainter, rect: QRect, samples: List[float], color: QColor):
        """
        BUG4 FIX: samples are already proportional to block width (1 sample ≈ 1 px).
        We map n samples across w pixels using linear interpolation — no stretching.
        """
        mid_y = rect.top() + rect.height() // 2
        half_h = (rect.height() - 8) // 2
        n = len(samples)
        if n == 0:
            return
        w = rect.width() - 4
        if w <= 0:
            return

        wave_color = QColor(255, 255, 255, 60)
        p.setPen(QPen(wave_color, 1))
        x0 = rect.left() + 2

        # Draw one vertical bar per pixel column, sampling from proportional positions
        for i in range(w):
            # Linear map: pixel i → sample index
            frac = i / max(w - 1, 1)
            idx = int(frac * (n - 1))
            idx = max(0, min(idx, n - 1))
            val = samples[idx]
            bar_h = max(1, int(abs(val) * half_h))
            p.drawLine(x0 + i, mid_y - bar_h, x0 + i, mid_y + bar_h)

    # ── mouse ─────────────────────────────────────────────────────────────
    def _dialogue_at(self, pos: QPoint) -> Optional[Dialogue]:
        if not self._job:
            return None
        for d in reversed(self._job.dialogues):
            if d.speaker_id not in self._speaker_order:
                continue
            if self._block_rect(d).contains(pos):
                return d
        return None

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            d = self._dialogue_at(e.position().toPoint())
            if d:
                self._selected_id = d.id
                self.dialogue_clicked.emit(d.id)
                rect = self._block_rect(d)
                px = e.position().x()
                # detect edge drag
                if px - rect.left() < 8:
                    self._drag_edge = "left"
                elif rect.right() - px < 8:
                    self._drag_edge = "right"
                else:
                    self._drag_edge = "move"
                self._drag_id = d.id
                self._drag_start_x = int(e.position().x())
                self._drag_orig_start = d.start_ms
                self._drag_orig_end = d.end_ms
                self.update()

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._drag_id and self._job:
            dx = int(e.position().x()) - self._drag_start_x
            delta_ms = int(dx / PX_PER_SEC * 1000)
            d = self._job.get_dialogue(self._drag_id)
            if d:
                if self._drag_edge == "left":
                    d.start_ms = max(0, self._drag_orig_start + delta_ms)
                    d.start_ms = min(d.start_ms, d.end_ms - 100)
                elif self._drag_edge == "right":
                    d.end_ms = max(self._drag_orig_end + delta_ms, d.start_ms + 100)
                elif self._drag_edge == "move":
                    shift = delta_ms
                    d.start_ms = max(0, self._drag_orig_start + shift)
                    d.end_ms = max(d.start_ms + 100, self._drag_orig_end + shift)
                self.update()
        else:
            # cursor feedback
            d = self._dialogue_at(e.position().toPoint())
            if d:
                rect = self._block_rect(d)
                px = e.position().x()
                if px - rect.left() < 8 or rect.right() - px < 8:
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                else:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e: QMouseEvent):
        if self._drag_id and self._job:
            d = self._job.get_dialogue(self._drag_id)
            if d:
                self.dialogue_drag_end.emit(d.id, d.start_ms, d.end_ms)
        self._drag_id = None
        self._drag_edge = None

    def contextMenuEvent(self, e: QContextMenuEvent):
        d = self._dialogue_at(e.pos())
        if not d:
            return
        menu = QMenu(self)
        act_merge = menu.addAction("⊕ Merge with next dialogue")
        act_split = menu.addAction("✂ Split at midpoint")
        chosen = menu.exec(e.globalPos())
        if chosen == act_merge:
            self.merge_requested.emit(d.id)
        elif chosen == act_split:
            self.split_requested.emit(d.id)


# ──────────────────────────────────────────────────────────────────────────────
# Pinned speaker label column
# ──────────────────────────────────────────────────────────────────────────────
class SpeakerLabelWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._speaker_order: List[str] = []
        self._job: Optional[Job] = None
        self.setFixedWidth(LABEL_W)

    def set_job(self, job: Job, speaker_order: List[str]):
        self._job = job
        self._speaker_order = speaker_order
        h = max(200, len(speaker_order) * ROW_H + 40)
        self.setMinimumHeight(h)
        self.update()

    def set_scroll_offset(self, offset: int):
        self._offset = offset
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(BG_PANEL))

        # ruler placeholder
        p.fillRect(0, 0, LABEL_W, RULER_H, QColor(BG_PANEL))
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawLine(0, RULER_H - 1, LABEL_W, RULER_H - 1)

        if not self._job:
            p.end()
            return

        offset = getattr(self, "_offset", 0)
        for i, spk in enumerate(self._speaker_order):
            ry = RULER_H + i * ROW_H - offset
            if ry + ROW_H < 0 or ry > self.height():
                continue
            color_hex = self._job.get_speaker_color(spk)
            color = QColor(color_hex)

            # accent strip
            p.fillRect(0, ry, 3, ROW_H, color)
            # row bg
            alt = QColor("#1C1C1C") if i % 2 == 0 else QColor("#1F1F1F")
            p.fillRect(3, ry, LABEL_W - 3, ROW_H, alt)
            # separator
            p.setPen(QPen(QColor(BORDER), 1))
            p.drawLine(0, ry + ROW_H - 1, LABEL_W, ry + ROW_H - 1)

            # name
            name = self._job.get_speaker_name(spk)
            p.setPen(QColor(color_hex))
            font = QFont("Segoe UI", FONT_XS)
            font.setBold(True)
            p.setFont(font)
            fm = QFontMetrics(font)
            name = fm.elidedText(name, Qt.TextElideMode.ElideRight, LABEL_W - 14)
            p.drawText(8, ry + ROW_H // 2 - 6, name)

            # speaker id small
            p.setPen(QColor(TEXT_DIM))
            font2 = QFont("Segoe UI", 8)
            p.setFont(font2)
            p.drawText(8, ry + ROW_H // 2 + 7, spk)

        p.end()


# ──────────────────────────────────────────────────────────────────────────────
# Full timeline widget (label column + scroll area)
# ──────────────────────────────────────────────────────────────────────────────
class TimelineWidget(QWidget):
    """
    Full timeline: pinned labels on left, scrollable canvas on right.
    """

    dialogue_selected = pyqtSignal(str)
    dialogue_moved = pyqtSignal(str, int, int)   # id, start_ms, end_ms
    merge_requested = pyqtSignal(str)
    split_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job: Optional[Job] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QHBoxLayout()
        body.setSpacing(0)
        body.setContentsMargins(0, 0, 0, 0)

        # ── pinned label column ──
        self._label_col = SpeakerLabelWidget()
        body.addWidget(self._label_col)

        # ── scrollable area ──
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: {TIMELINE_BG};
            }}
        """)

        # canvas inside scroll
        self._canvas_container = QWidget()
        canvas_layout = QVBoxLayout(self._canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)

        self._ruler = RulerWidget()
        canvas_layout.addWidget(self._ruler)

        self._canvas = TimelineCanvas()
        self._canvas.dialogue_clicked.connect(self._on_dialogue_clicked)
        self._canvas.dialogue_drag_end.connect(self._on_drag_end)
        self._canvas.merge_requested.connect(self.merge_requested)
        self._canvas.split_requested.connect(self.split_requested)
        canvas_layout.addWidget(self._canvas)
        canvas_layout.addStretch()

        self._scroll.setWidget(self._canvas_container)
        body.addWidget(self._scroll, 1)

        root.addLayout(body)

        # sync vertical scroll to label column
        self._scroll.verticalScrollBar().valueChanged.connect(
            lambda v: self._label_col.set_scroll_offset(v)
        )
        self.setStyleSheet(f"background:{TIMELINE_BG};")

    # ── public API ────────────────────────────────────────────────────────
    def set_job(self, job: Job):
        self._job = job
        self._canvas.set_job(job)
        self._ruler.set_total_ms(
            max((d.end_ms for d in job.dialogues), default=60000) + 2000
        )
        speaker_order = self._canvas._speaker_order
        self._label_col.set_job(job, speaker_order)
        self._canvas_container.setMinimumSize(
            self._canvas.minimumWidth(),
            self._canvas.minimumHeight() + RULER_H
        )

    def set_playhead(self, ms: int):
        self._canvas.set_playhead(ms)
        # auto-scroll to playhead if out of view
        x = int(ms / 1000 * PX_PER_SEC)
        hsb = self._scroll.horizontalScrollBar()
        vp_w = self._scroll.viewport().width()
        if x < hsb.value() or x > hsb.value() + vp_w - 40:
            hsb.setValue(max(0, x - vp_w // 3))

    def set_selected(self, dialogue_id: Optional[str]):
        self._canvas.set_selected(dialogue_id)
        # scroll to block
        if dialogue_id and self._job:
            d = self._job.get_dialogue(dialogue_id)
            if d:
                x = int(d.start_ms / 1000 * PX_PER_SEC)
                hsb = self._scroll.horizontalScrollBar()
                vp_w = self._scroll.viewport().width()
                if x < hsb.value() or x > hsb.value() + vp_w - 40:
                    hsb.setValue(max(0, x - 60))

    def set_waveform(self, dialogue_id: str, samples: list):
        self._canvas.set_waveform(dialogue_id, samples)

    def refresh(self):
        if self._job:
            self.set_job(self._job)

    def _on_dialogue_clicked(self, did: str):
        self.dialogue_selected.emit(did)

    def _on_drag_end(self, did: str, start_ms: int, end_ms: int):
        self.dialogue_moved.emit(did, start_ms, end_ms)


def _fmt_s(s: int) -> str:
    m, s = divmod(s, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"