"""
DubStudio Pro — Reusable widgets.
AnimatedButton, SectionHeader, StatusBadge, Divider, LogTerminal.
"""
from __future__ import annotations

from PyQt6.QtCore import (Qt, QPropertyAnimation, QEasingCurve,
                           pyqtProperty, QPoint)
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PyQt6.QtWidgets import (QWidget, QLabel, QHBoxLayout, QVBoxLayout,
                              QPushButton, QSizePolicy, QPlainTextEdit,
                              QFrame)

from app.ui.design import (ACCENT, ACCENT_DIM, BG_CARD, BG_PANEL, BORDER,
                            BORDER_LIGHT, TEXT_PRIMARY, TEXT_SECONDARY,
                            TEXT_DIM, SUCCESS, WARNING, ERROR, RUNNING,
                            SKIPPED, STATUS_COLORS, STATUS_LABELS,
                            FONT_SM, FONT_MD, FONT_LG, FONT_XS)


# ──────────────────────────────────────────────────────────────────────────────
# AnimatedButton
# ──────────────────────────────────────────────────────────────────────────────
class AnimatedButton(QPushButton):
    """Button with hover glow and press animation."""

    def __init__(self, text: str = "", variant: str = "default",
                 icon_text: str = "", parent=None):
        super().__init__(text, parent)
        self._variant = variant
        self._icon_text = icon_text
        self._glow = 0.0
        self._pressed_anim = None
        self._setup_style()

    def _setup_style(self):
        v = self._variant
        if v == "primary":
            bg, hover_bg, text_col = ACCENT, "#FFB840", "#000000"
        elif v == "danger":
            bg, hover_bg, text_col = "#C0392B", "#E74C3C", TEXT_PRIMARY
        elif v == "success":
            bg, hover_bg, text_col = "#27AE60", "#2ECC71", TEXT_PRIMARY
        elif v == "ghost":
            bg, hover_bg, text_col = "transparent", BG_CARD, TEXT_SECONDARY
        elif v == "accent-outline":
            bg, hover_bg, text_col = "transparent", f"{ACCENT}22", ACCENT
        else:
            bg, hover_bg, text_col = BG_CARD, "#2A2A2A", TEXT_PRIMARY

        border_col = ACCENT if v == "accent-outline" else BORDER

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: {text_col};
                border: 1px solid {border_col};
                border-radius: 5px;
                padding: 5px 14px;
                font-size: {FONT_SM}px;
                font-weight: 500;
                letter-spacing: 0.3px;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
                border-color: {ACCENT if v != 'danger' else '#E74C3C'};
                color: {"#000" if v == "primary" else ACCENT if v != 'danger' else TEXT_PRIMARY};
            }}
            QPushButton:pressed {{
                background-color: {ACCENT_DIM if v == 'primary' else '#1A1A1A'};
                border-color: {ACCENT_DIM};
            }}
            QPushButton:disabled {{
                background-color: #1A1A1A;
                color: #404040;
                border-color: #282828;
            }}
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


# ──────────────────────────────────────────────────────────────────────────────
# SectionHeader
# ──────────────────────────────────────────────────────────────────────────────
class SectionHeader(QWidget):
    """Labelled section header with optional accent line."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        accent_bar = QFrame()
        accent_bar.setFixedWidth(3)
        accent_bar.setStyleSheet(f"background: {ACCENT}; border-radius: 2px;")
        lay.addWidget(accent_bar)

        text_col = QVBoxLayout()
        text_col.setSpacing(1)

        title_lbl = QLabel(title.upper())
        title_lbl.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            font-size: {FONT_MD}px;
            font-weight: 600;
            letter-spacing: 1px;
            background: transparent;
        """)
        text_col.addWidget(title_lbl)

        if subtitle:
            sub_lbl = QLabel(subtitle)
            sub_lbl.setStyleSheet(f"""
                color: {TEXT_DIM};
                font-size: {FONT_XS}px;
                background: transparent;
            """)
            text_col.addWidget(sub_lbl)

        lay.addLayout(text_col)
        lay.addStretch()
        self.setFixedHeight(36 if subtitle else 24)


# ──────────────────────────────────────────────────────────────────────────────
# StatusBadge
# ──────────────────────────────────────────────────────────────────────────────
class StatusBadge(QLabel):
    """Coloured status pill label."""

    def __init__(self, status: str = "pending", parent=None):
        super().__init__(parent)
        self.set_status(status)

    def set_status(self, status: str):
        color = STATUS_COLORS.get(status, TEXT_DIM)
        label = STATUS_LABELS.get(status, status.upper())
        self.setText(label)
        self.setStyleSheet(f"""
            QLabel {{
                color: {color};
                background: {color}22;
                border: 1px solid {color}55;
                border-radius: 3px;
                padding: 1px 7px;
                font-size: {FONT_XS}px;
                font-weight: 700;
                letter-spacing: 0.8px;
            }}
        """)


# ──────────────────────────────────────────────────────────────────────────────
# Divider
# ──────────────────────────────────────────────────────────────────────────────
class Divider(QFrame):
    """Horizontal separator line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {BORDER}; border: none;")


# ──────────────────────────────────────────────────────────────────────────────
# LogTerminal
# ──────────────────────────────────────────────────────────────────────────────
class LogTerminal(QPlainTextEdit):
    """Dark terminal-style log output widget."""

    MAX_LINES = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: #0D0D0D;
                color: #B0C4B0;
                border: 1px solid {BORDER};
                border-radius: 4px;
                font-family: "Consolas", "Courier New", monospace;
                font-size: {FONT_SM}px;
                padding: 6px;
                selection-background-color: {ACCENT}55;
            }}
        """)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

    def append_line(self, text: str, color: str = ""):
        """Append a line, optionally coloured (HTML)."""
        self.moveCursor(self.textCursor().MoveOperation.End)
        if color:
            self.appendHtml(f'<span style="color:{color}">{text}</span>')
        else:
            self.appendPlainText(text)
        # auto-scroll
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        # trim
        doc = self.document()
        while doc.blockCount() > self.MAX_LINES:
            cursor = self.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def clear_log(self):
        self.clear()