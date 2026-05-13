"""
DubStudio Pro — Design system.
Colors, fonts, global Qt stylesheet.
Dark professional theme — DaVinci Resolve inspired.
"""
from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────────────────────────────────────
BG          = "#141414"
BG_PANEL    = "#1C1C1C"
BG_CARD     = "#222222"
BG_INPUT    = "#2A2A2A"
BG_HOVER    = "#2E2E2E"
BG_SELECTED = "#333333"

ACCENT      = "#E8A020"      # golden
ACCENT_DIM  = "#A06808"
ACCENT_GLOW = "#FFB84080"

BORDER      = "#333333"
BORDER_MID  = "#3A3A3A"
BORDER_LIGHT= "#484848"

TEXT_PRIMARY   = "#F0F0F0"
TEXT_SECONDARY = "#A0A0A0"
TEXT_DIM       = "#606060"
TEXT_DISABLED  = "#505050"

SUCCESS   = "#4CAF50"
WARNING   = "#FF9800"
ERROR     = "#F44336"
INFO      = "#2196F3"
RUNNING   = "#29B6F6"
SKIPPED   = "#757575"

SCROLLBAR_BG    = "#1C1C1C"
SCROLLBAR_HANDLE= "#3A3A3A"
SCROLLBAR_HOVER = "#505050"

TIMELINE_BG     = "#111111"
PLAYHEAD_COLOR  = "#E8A020"

# ──────────────────────────────────────────────────────────────────────────────
# Status → colour mapping
# ──────────────────────────────────────────────────────────────────────────────
STATUS_COLORS = {
    "pending":   TEXT_DIM,
    "running":   RUNNING,
    "completed": SUCCESS,
    "failed":    ERROR,
    "skipped":   SKIPPED,
}

STATUS_LABELS = {
    "pending":   "PENDING",
    "running":   "RUNNING",
    "completed": "COMPLETE",
    "failed":    "FAILED",
    "skipped":   "SKIPPED",
}

# ──────────────────────────────────────────────────────────────────────────────
# Font sizes
# ──────────────────────────────────────────────────────────────────────────────
FONT_XS  = 9
FONT_SM  = 10
FONT_MD  = 11
FONT_LG  = 13
FONT_XL  = 16
FONT_XXL = 20

# ──────────────────────────────────────────────────────────────────────────────
# Global stylesheet
# ──────────────────────────────────────────────────────────────────────────────
GLOBAL_STYLESHEET = f"""
/* ── Base ── */
QWidget {{
    background-color: {BG};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "DejaVu Sans", sans-serif;
    font-size: {FONT_MD}px;
    selection-background-color: {ACCENT};
    selection-color: #000000;
}}

/* ── Main window ── */
QMainWindow {{
    background-color: {BG};
}}

/* ── Splitter ── */
QSplitter::handle {{
    background-color: {BORDER};
}}
QSplitter::handle:hover {{
    background-color: {ACCENT_DIM};
}}
QSplitter::handle:horizontal {{
    width: 4px;
}}
QSplitter::handle:vertical {{
    height: 4px;
}}

/* ── Scroll bars ── */
QScrollBar:vertical {{
    background: {SCROLLBAR_BG};
    width: 8px;
    margin: 0;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {SCROLLBAR_HANDLE};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {SCROLLBAR_HOVER};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {SCROLLBAR_BG};
    height: 8px;
    margin: 0;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {SCROLLBAR_HANDLE};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {SCROLLBAR_HOVER};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── LineEdit / TextEdit ── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT_PRIMARY};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid {ACCENT};
}}

/* ── ComboBox ── */
QComboBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 8px;
    color: {TEXT_PRIMARY};
    min-width: 80px;
}}
QComboBox:hover {{
    border: 1px solid {BORDER_LIGHT};
}}
QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: #000000;
    outline: none;
}}

/* ── CheckBox ── */
QCheckBox {{
    spacing: 6px;
    color: {TEXT_PRIMARY};
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER_LIGHT};
    border-radius: 3px;
    background-color: {BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

/* ── Labels ── */
QLabel {{
    background: transparent;
    color: {TEXT_PRIMARY};
}}

/* ── GroupBox ── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 8px;
    font-size: {FONT_SM}px;
    color: {TEXT_SECONDARY};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_SECONDARY};
    font-size: {FONT_SM}px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* ── ToolTip ── */
QToolTip {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_MID};
    color: {TEXT_PRIMARY};
    padding: 4px 8px;
    border-radius: 4px;
    font-size: {FONT_SM}px;
}}

/* ── Menu ── */
QMenu {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 20px;
    border-radius: 4px;
    color: {TEXT_PRIMARY};
}}
QMenu::item:selected {{
    background-color: {BG_HOVER};
    color: {ACCENT};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 0;
}}

/* ── MessageBox ── */
QMessageBox {{
    background-color: {BG_PANEL};
}}
QMessageBox QLabel {{
    color: {TEXT_PRIMARY};
    min-width: 300px;
}}

/* ── Dialog ── */
QDialog {{
    background-color: {BG_PANEL};
}}

/* ── Slider ── */
QSlider::groove:horizontal {{
    background: {BG_INPUT};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
"""