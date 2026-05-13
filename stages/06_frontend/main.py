"""
DubStudio Pro — Entry point.
Run via:
  conda run -n dub_frontend python stages/06_frontend/main.py
"""
import sys
import os

# Ensure project root is in path
_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_STAGE_DIR))
if _STAGE_DIR not in sys.path:
    sys.path.insert(0, _STAGE_DIR)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from app.ui.design import GLOBAL_STYLESHEET, BG, FONT_MD
from app.ui.main_window import MainWindow


def main():
    # High-DPI scaling
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("DubStudio Pro")
    app.setOrganizationName("DubStudio")

    # Global stylesheet
    app.setStyleSheet(GLOBAL_STYLESHEET)

    # Default font
    font = QFont("Segoe UI", FONT_MD)
    app.setFont(font)

    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()