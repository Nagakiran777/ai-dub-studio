"""
DubStudio Pro — Pipeline Panel.
Stage cards with Run / Skip buttons, live log terminal at bottom.
Stage subprocess is monitored via QTimer, output streamed to log.
"""
from __future__ import annotations

import subprocess
from typing import Optional, Dict

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QScrollArea, QFrame, QSizePolicy, QSplitter)

from app.models.job_model import Job, StageStatus
from app.pipeline_api import (STAGE_DEFINITIONS, REVIEW_GATES,
                               get_stage_description, is_review_gate,
                               get_review_message, run_stage)
from app.ui.design import (ACCENT, BG_PANEL, BG_CARD, BORDER, BORDER_LIGHT,
                            TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
                            SUCCESS, ERROR, RUNNING, WARNING, SKIPPED,
                            STATUS_COLORS, FONT_SM, FONT_XS, FONT_MD)
from app.ui.widgets import AnimatedButton, SectionHeader, Divider, LogTerminal, StatusBadge


class StageRow(QWidget):
    """Single stage card with status badge, run/skip/cancel buttons."""

    run_clicked    = pyqtSignal(str)   # stage_id
    skip_clicked   = pyqtSignal(str)
    cancel_clicked = pyqtSignal(str)

    def __init__(self, stage_def: dict, parent=None):
        super().__init__(parent)
        self._stage_id = stage_def["id"]
        self._status = "pending"
        self._running = False
        self._build_ui(stage_def)

    def _build_ui(self, sd: dict):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(4)

        # ── top row: badge + name ──
        top = QHBoxLayout()
        top.setSpacing(8)

        self._badge = StatusBadge("pending")
        top.addWidget(self._badge)

        name_col = QVBoxLayout()
        name_col.setSpacing(1)
        lbl = QLabel(sd["label"])
        lbl.setStyleSheet(
            f"color:{TEXT_PRIMARY}; font-size:{FONT_MD}px; font-weight:600; background:transparent;")
        name_col.addWidget(lbl)
        desc = QLabel(get_stage_description(sd["id"]))
        desc.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:{FONT_XS}px; background:transparent;")
        desc.setWordWrap(True)
        name_col.addWidget(desc)
        top.addLayout(name_col, 1)

        # gate indicator
        if is_review_gate(sd["id"]):
            gate_lbl = QLabel("⚑ Review gate")
            gate_lbl.setStyleSheet(
                f"color:{WARNING}; font-size:{FONT_XS}px; background:transparent;")
            top.addWidget(gate_lbl)

        root.addLayout(top)

        # ── button row ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.setContentsMargins(0, 2, 0, 0)

        self._btn_run = AnimatedButton("▶ Run", "primary")
        self._btn_run.setFixedHeight(24)
        self._btn_run.setFixedWidth(72)
        btn_row.addWidget(self._btn_run)

        self._btn_skip = AnimatedButton("⊘ Skip", "ghost")
        self._btn_skip.setFixedHeight(24)
        self._btn_skip.setFixedWidth(66)
        btn_row.addWidget(self._btn_skip)

        self._btn_cancel = AnimatedButton("■ Cancel", "danger")
        self._btn_cancel.setFixedHeight(24)
        self._btn_cancel.setFixedWidth(76)
        self._btn_cancel.setVisible(False)
        btn_row.addWidget(self._btn_cancel)

        btn_row.addStretch()
        root.addLayout(btn_row)

        self.setStyleSheet(f"""
            StageRow {{
                background: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 6px;
            }}
        """)
        self.setObjectName("StageRow")

        self._btn_run.clicked.connect(lambda: self.run_clicked.emit(self._stage_id))
        self._btn_skip.clicked.connect(lambda: self.skip_clicked.emit(self._stage_id))
        self._btn_cancel.clicked.connect(lambda: self.cancel_clicked.emit(self._stage_id))

    def set_status(self, status: str, running: bool = False):
        self._status = status
        self._running = running
        self._badge.set_status(status)

        can_run = status in ("pending", "failed")
        can_skip = status in ("pending", "failed")
        self._btn_run.setVisible(not running)
        self._btn_skip.setVisible(not running)
        self._btn_cancel.setVisible(running)
        self._btn_run.setEnabled(can_run and not running)
        self._btn_skip.setEnabled(can_skip and not running)

        # highlight card when running
        if running:
            self.setStyleSheet(f"""
                StageRow {{
                    background: {BG_CARD};
                    border: 1px solid {RUNNING};
                    border-radius: 6px;
                }}
            """)
        elif status == "completed":
            self.setStyleSheet(f"""
                StageRow {{
                    background: {BG_CARD};
                    border: 1px solid {SUCCESS}55;
                    border-radius: 6px;
                }}
            """)
        elif status == "failed":
            self.setStyleSheet(f"""
                StageRow {{
                    background: {BG_CARD};
                    border: 1px solid {ERROR}55;
                    border-radius: 6px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                StageRow {{
                    background: {BG_CARD};
                    border: 1px solid {BORDER};
                    border-radius: 6px;
                }}
            """)


class PipelinePanel(QWidget):
    """
    Full pipeline panel: stage cards list + live log terminal.
    Signals:
      stage_completed(stage_id)
      stage_failed(stage_id)
      review_gate_reached(stage_id, message)
    """

    stage_completed       = pyqtSignal(str)
    stage_failed          = pyqtSignal(str)
    review_gate_reached   = pyqtSignal(str, str)
    auto_save_profiles    = pyqtSignal()   # emitted before 02_emotion runs

    def __init__(self, parent=None):
        super().__init__(parent)
        self._job: Optional[Job] = None
        self._rows: Dict[str, StageRow] = {}
        self._running_stage: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._poll_timer = QTimer()
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_process)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(4)

        # ── stage cards scroll area ──
        stage_widget = QWidget()
        stage_widget.setStyleSheet(f"background:{BG_PANEL};")
        stage_layout = QVBoxLayout(stage_widget)
        stage_layout.setContentsMargins(6, 6, 6, 6)
        stage_layout.setSpacing(6)

        stage_layout.addWidget(SectionHeader("Pipeline Stages", "Run or skip each stage"))
        stage_layout.addWidget(Divider())

        for sd in STAGE_DEFINITIONS:
            row = StageRow(sd)
            row.run_clicked.connect(self._on_run)
            row.skip_clicked.connect(self._on_skip)
            row.cancel_clicked.connect(self._on_cancel)
            self._rows[sd["id"]] = row
            stage_layout.addWidget(row)

        stage_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(stage_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"background:{BG_PANEL}; border:none;")
        splitter.addWidget(scroll)

        # ── log terminal ──
        log_widget = QWidget()
        log_widget.setStyleSheet(f"background:{BG_PANEL};")
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(6, 4, 6, 4)
        log_layout.setSpacing(4)

        log_hdr = QHBoxLayout()
        log_hdr.addWidget(SectionHeader("Live Log"))
        btn_clear = AnimatedButton("Clear", "ghost")
        btn_clear.setFixedHeight(22)
        btn_clear.setFixedWidth(50)
        btn_clear.clicked.connect(lambda: self._log.clear_log())
        log_hdr.addWidget(btn_clear)
        log_layout.addLayout(log_hdr)

        self._log = LogTerminal()
        log_layout.addWidget(self._log)
        splitter.addWidget(log_widget)

        splitter.setSizes([500, 200])
        root.addWidget(splitter)
        self.setStyleSheet(f"background:{BG_PANEL};")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def set_job(self, job: Job):
        self._job = job
        # reset ALL rows first, then apply status
        for sid, row in self._rows.items():
            row.set_status("pending", running=False)
        for sid, ss in job.stages.items():
            if sid in self._rows:
                self._rows[sid].set_status(ss.status)

    def log(self, text: str, color: str = ""):
        self._log.append_line(text, color)

    def clear_log(self):
        self._log.clear_log()

    # ──────────────────────────────────────────────────────────────────────
    # Stage execution
    # ──────────────────────────────────────────────────────────────────────
    def _on_run(self, stage_id: str):
        if not self._job:
            return
        if self._running_stage:
            self.log(f"[WARN] Stage '{self._running_stage}' already running.", WARNING)
            return
        sd = next((s for s in STAGE_DEFINITIONS if s["id"] == stage_id), None)
        if not sd:
            return

        # BUG3 FIX: auto-save speaker profiles before 02_emotion runs,
        # in case user forgot to confirm them via the Speaker Manager dialog.
        if stage_id == "02_emotion":
            self.log("⚙ Auto-saving speaker profiles before Emotion Analysis...", WARNING)
            self.auto_save_profiles.emit()   # handled by main_window

        self._running_stage = stage_id
        self._rows[stage_id].set_status("running", running=True)
        self.log(f"\n▶ Starting stage: {sd['label']} [{stage_id}]", ACCENT)
        self.log(f"  Job: {self._job.job_id}", TEXT_DIM)

        try:
            self._proc = run_stage(stage_id, self._job.job_id, sd["env"])
            self._poll_timer.start()
        except Exception as ex:
            self.log(f"[ERROR] Failed to launch: {ex}", ERROR)
            self._rows[stage_id].set_status("failed")
            self._running_stage = None

    def _on_skip(self, stage_id: str):
        if not self._job:
            return
        self._rows[stage_id].set_status("skipped")
        self.log(f"⊘ Skipped stage: {stage_id}", SKIPPED)
        from app.job_manager import update_manifest_stage
        update_manifest_stage(self._job.job_id, stage_id, "skipped")
        self._job.stages[stage_id] = type(self._job.stages.get(
            stage_id, object()))()
        if stage_id in self._job.stages:
            self._job.stages[stage_id].status = "skipped"
        self.stage_completed.emit(stage_id)

    def _on_cancel(self, stage_id: str):
        if self._proc:
            self.log(f"■ Cancelling stage: {stage_id}", WARNING)
            self._proc.kill()
            self._proc = None
        self._poll_timer.stop()
        self._running_stage = None
        self._rows[stage_id].set_status("failed")
        self.log(f"Stage {stage_id} cancelled.", WARNING)

    def _poll_process(self):
        if not self._proc:
            self._poll_timer.stop()
            return

        # read available stdout
        try:
            line = self._proc.stdout.readline()
            if line:
                self._log.append_line(line.rstrip())
        except Exception:
            pass

        ret = self._proc.poll()
        if ret is not None:
            # drain remaining output
            try:
                remaining = self._proc.stdout.read()
                if remaining:
                    for ln in remaining.splitlines():
                        self._log.append_line(ln)
            except Exception:
                pass

            self._poll_timer.stop()
            sid = self._running_stage
            self._proc = None
            self._running_stage = None

            if ret == 0:
                self._rows[sid].set_status("completed")
                self.log(f"\n✓ Stage '{sid}' completed successfully.", SUCCESS)
                from app.job_manager import update_manifest_stage
                from datetime import datetime, timezone
                update_manifest_stage(
                    self._job.job_id, sid, "completed",
                    completed_at=datetime.now(timezone.utc).isoformat()
                )
                # check review gate
                if is_review_gate(sid):
                    msg = get_review_message(sid)
                    self.log(f"⚑ Review gate: {msg}", WARNING)
                    self.review_gate_reached.emit(sid, msg)
                else:
                    self.stage_completed.emit(sid)
            else:
                self._rows[sid].set_status("failed")
                self.log(f"\n✗ Stage '{sid}' failed (exit code {ret}).", ERROR)
                from app.job_manager import update_manifest_stage
                from datetime import datetime, timezone
                update_manifest_stage(
                    self._job.job_id, sid, "failed",
                    failed_at=datetime.now(timezone.utc).isoformat()
                )
                self.stage_failed.emit(sid)