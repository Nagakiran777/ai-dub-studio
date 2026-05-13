"""
DubStudio Pro — Main Window.
Layout:
  Left:  Video player (top) + Timeline (bottom, full width)
  Right: Vertical splitter → Pipeline stages | Dialogue editor | Live log
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QKeySequence, QShortcut, QAction, QFont
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QSplitter, QLabel, QPushButton, QFileDialog,
                              QMessageBox, QComboBox, QToolBar, QStatusBar,
                              QApplication, QSizePolicy)

from app.ui.design import (ACCENT, BG, BG_PANEL, BG_CARD, BORDER, TEXT_PRIMARY,
                            TEXT_SECONDARY, TEXT_DIM, SUCCESS, WARNING, ERROR,
                            FONT_SM, FONT_MD, FONT_XL)
from app.ui.widgets import AnimatedButton, SectionHeader, Divider, LogTerminal
from app.ui.video_player import VideoPlayer
from app.ui.timeline_widget import TimelineWidget
from app.ui.dialogue_panel import DialoguePanel
from app.ui.pipeline_panel import PipelinePanel
from app.ui.speaker_manager import SpeakerManagerDialog
from app.models.job_model import Job, Dialogue
from app import job_manager

PROJECT_ROOT = "/mnt/d/$/my/dubbing_V2"


# ──────────────────────────────────────────────────────────────────────────────
# Waveform loader thread
# ──────────────────────────────────────────────────────────────────────────────
class WaveformThread(QThread):
    waveform_ready = pyqtSignal(str, list)   # dialogue_id, samples

    def __init__(self, job: Job, parent=None):
        super().__init__(parent)
        self._job = job

    def run(self):
        try:
            import soundfile as sf
            import numpy as np
            wav_path = job_manager.original_wav_path(self._job.job_id)
            if not wav_path.exists():
                return
            data, sr = sf.read(str(wav_path), always_2d=True)
            mono = data.mean(axis=1)
            for d in self._job.dialogues:
                start_s = int(d.start_ms / 1000 * sr)
                end_s = int(d.end_ms / 1000 * sr)
                clip = mono[start_s:end_s]
                if len(clip) == 0:
                    continue
                # BUG4 FIX: proportional sample count based on dialogue duration.
                # Each sample represents one pixel at PX_PER_SEC=80 scale.
                # This means short blocks get fewer points (not stretched).
                duration_s = (d.end_ms - d.start_ms) / 1000.0
                # At 80 px/sec, a 1s dialogue = 80 px wide → 80 samples
                n_out = max(4, int(duration_s * 80))
                n_out = min(n_out, len(clip))  # can't have more samples than audio
                if n_out == 0:
                    continue
                # Use fixed-size buckets: average absolute amplitude per bucket
                bucket_size = len(clip) / n_out
                samples = []
                for i in range(n_out):
                    lo = int(i * bucket_size)
                    hi = int((i + 1) * bucket_size)
                    chunk = clip[lo:hi]
                    if len(chunk):
                        samples.append(float(np.abs(chunk).mean()))
                    else:
                        samples.append(0.0)
                if samples:
                    mx = max(samples) or 1.0
                    samples = [s / mx for s in samples]
                self.waveform_ready.emit(d.id, samples)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Job combo list widget
# ──────────────────────────────────────────────────────────────────────────────
class JobSelector(QWidget):
    job_selected = pyqtSignal(str)
    new_job_requested = pyqtSignal()
    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lbl = QLabel("Job:")
        lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:{FONT_SM}px; background:transparent;")
        lay.addWidget(lbl)

        self._combo = QComboBox()
        self._combo.setMinimumWidth(260)
        self._combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay.addWidget(self._combo, 1)

        btn_refresh = AnimatedButton("⟳", "ghost")
        btn_refresh.setFixedWidth(32)
        btn_refresh.setToolTip("Refresh job list")
        btn_refresh.clicked.connect(self.refresh_requested)
        lay.addWidget(btn_refresh)

        btn_new = AnimatedButton("+ New Job", "primary")
        btn_new.clicked.connect(self.new_job_requested)
        lay.addWidget(btn_new)

        self._combo.currentIndexChanged.connect(self._on_changed)

    def populate(self, job_ids: list, current: str = ""):
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem("— select job —", "")
        for jid in job_ids:
            self._combo.addItem(jid, jid)
        if current:
            idx = self._combo.findData(current)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)

    def _on_changed(self, idx: int):
        jid = self._combo.itemData(idx)
        if jid:
            self.job_selected.emit(jid)


# ──────────────────────────────────────────────────────────────────────────────
# Main Window
# ──────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self._job: Optional[Job] = None
        self._waveform_thread: Optional[WaveformThread] = None

        self.setWindowTitle("DubStudio Pro")
        self.setMinimumSize(1200, 700)
        # No max size limits — freely resizable
        self.resize(1600, 950)

        self._build_ui()
        self._setup_shortcuts()
        self._refresh_jobs()

    # ──────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── toolbar ──
        self._build_toolbar()

        # ── central widget ──
        central = QWidget()
        central.setStyleSheet(f"background:{BG};")
        self.setCentralWidget(central)

        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # ── horizontal splitter: left (video+timeline) | right panel ──
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(4)

        # ── LEFT: video (top) + timeline (bottom) ──
        left_widget = QWidget()
        left_widget.setStyleSheet(f"background:{BG};")
        left_lay = QVBoxLayout(left_widget)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)

        v_split_left = QSplitter(Qt.Orientation.Vertical)
        v_split_left.setHandleWidth(4)

        # video player
        self._video = VideoPlayer()
        self._video.setMinimumHeight(200)
        v_split_left.addWidget(self._video)

        # timeline — fills remaining vertical space
        self._timeline = TimelineWidget()
        self._timeline.setMinimumHeight(160)
        v_split_left.addWidget(self._timeline)

        v_split_left.setSizes([400, 260])
        left_lay.addWidget(v_split_left)

        h_split.addWidget(left_widget)

        # ── RIGHT: vertical splitter (pipeline | dialogue | log) ──
        self._right_panel = QSplitter(Qt.Orientation.Vertical)
        self._right_panel.setHandleWidth(4)
        self._right_panel.setStyleSheet(f"background:{BG_PANEL};")
        self._right_panel.setMinimumWidth(340)

        self._pipeline = PipelinePanel()
        self._right_panel.addWidget(self._pipeline)

        self._dialogue_panel = DialoguePanel()
        self._right_panel.addWidget(self._dialogue_panel)

        self._right_panel.setSizes([420, 380])
        h_split.addWidget(self._right_panel)

        h_split.setSizes([1100, 420])
        main_lay.addWidget(h_split)

        # ── status bar ──
        self._status = QStatusBar()
        self._status.setStyleSheet(
            f"background:{BG_PANEL}; color:{TEXT_DIM}; "
            f"font-size:{FONT_SM}px; border-top:1px solid {BORDER};")
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — open a job or create a new one.")

        # ── wire signals ──
        self._video.position_changed.connect(self._on_video_position)
        self._timeline.dialogue_selected.connect(self._on_dialogue_selected)
        self._timeline.dialogue_moved.connect(self._on_dialogue_moved)
        self._timeline.merge_requested.connect(self._on_merge)
        self._timeline.split_requested.connect(self._on_split)
        self._dialogue_panel.dialogue_changed.connect(self._on_dialogue_edited)
        self._dialogue_panel.selection_changed.connect(self._on_dialogue_selected)
        self._pipeline.stage_completed.connect(self._on_stage_completed)
        self._pipeline.stage_failed.connect(self._on_stage_failed)
        self._pipeline.review_gate_reached.connect(self._on_review_gate)
        self._pipeline.auto_save_profiles.connect(self._auto_save_profiles)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setStyleSheet(f"""
            QToolBar {{
                background: {BG_PANEL};
                border-bottom: 1px solid {BORDER};
                padding: 4px 8px;
                spacing: 8px;
            }}
        """)
        self.addToolBar(tb)

        # Logo label
        logo = QLabel("  🎬 DubStudio Pro")
        logo.setStyleSheet(
            f"color:{ACCENT}; font-size:{FONT_XL}px; font-weight:700; "
            f"letter-spacing:1px; background:transparent;")
        tb.addWidget(logo)

        spacer = QWidget()
        spacer.setFixedWidth(24)
        spacer.setStyleSheet("background:transparent;")
        tb.addWidget(spacer)

        # Job selector
        self._job_selector = JobSelector()
        self._job_selector.job_selected.connect(self._load_job)
        self._job_selector.new_job_requested.connect(self._new_job)
        self._job_selector.refresh_requested.connect(self._refresh_jobs)
        tb.addWidget(self._job_selector)

        sep = QWidget()
        sep.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sep.setStyleSheet("background:transparent;")
        tb.addWidget(sep)

        # Speaker manager button
        self._btn_speakers = AnimatedButton("👤 Speakers", "accent-outline")
        self._btn_speakers.setEnabled(False)
        self._btn_speakers.clicked.connect(self._open_speaker_manager)
        tb.addWidget(self._btn_speakers)

        # Save button
        self._btn_save = AnimatedButton("💾 Save", "ghost")
        self._btn_save.setEnabled(False)
        self._btn_save.setToolTip("Save all edits (Ctrl+S)")
        self._btn_save.clicked.connect(self._save_all)
        tb.addWidget(self._btn_save)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(
            self._dialogue_panel.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(
            self._dialogue_panel.redo)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(
            self._save_all)
        QShortcut(QKeySequence("Space"), self).activated.connect(
            self._toggle_play)

    # ──────────────────────────────────────────────────────────────────────
    # Job management
    # ──────────────────────────────────────────────────────────────────────
    def _refresh_jobs(self):
        current = self._job.job_id if self._job else ""
        ids = job_manager.list_jobs()
        self._job_selector.populate(ids, current)

    def _new_job(self):
        """Browse for a video file and create a new job."""
        self._status.showMessage("Opening file picker...")
        QApplication.processEvents()
        path = self._browse_video()
        if not path:
            self._status.showMessage("No video selected.")
            return
        self._status.showMessage(f"Creating job for: {path}")
        QApplication.processEvents()
        try:
            job_id = job_manager.create_job(path)
            self._refresh_jobs()
            self._load_job(job_id)
        except Exception as ex:
            self._show_error("Failed to create job", str(ex) + "\n\nPath: " + str(path))

    def _browse_video(self) -> str:
        """
        Open a video file picker.
        1. Try PowerShell native Windows dialog (works best from WSL2).
        2. Fall back to Qt file dialog if PowerShell fails.
        """
        # ── attempt PowerShell native dialog ──
        # Use a hidden TopMost owner form so dialog appears in front
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "Add-Type -AssemblyName System.Drawing; "
            "$owner = New-Object System.Windows.Forms.Form; "
            "$owner.TopMost = $true; "
            "$owner.StartPosition = 'CenterScreen'; "
            "$owner.Size = New-Object System.Drawing.Size(1,1); "
            "$owner.Show(); "
            "$owner.Activate(); "
            "$f = New-Object System.Windows.Forms.OpenFileDialog; "
            "$f.Title = 'Select video file for DubStudio'; "
            "$f.Filter = 'Video files|*.mp4;*.mkv;*.avi;*.mov;*.webm|All files|*.*'; "
            "$f.InitialDirectory = [Environment]::GetFolderPath('MyVideos'); "
            "$null = $f.ShowDialog($owner); "
            "$owner.Dispose(); "
            "Write-Output $f.FileName"
        )
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive",
                 "-Command", ps_script],
                capture_output=True, text=True, timeout=120
            )
            win_path = result.stdout.strip()
            # powershell returns empty string if user cancelled
            if win_path and Path(_win_to_wsl(win_path)).suffix.lower() in (
                    ".mp4", ".mkv", ".avi", ".mov", ".webm"):
                wsl_path = _win_to_wsl(win_path)
                self._status.showMessage(f"Selected: {wsl_path}")
                return wsl_path
            # user cancelled — don't fall through to Qt dialog
            if win_path == "" and result.returncode == 0:
                return ""
        except FileNotFoundError:
            # powershell.exe not found — pure Linux, use Qt
            pass
        except subprocess.TimeoutExpired:
            self._show_error("Timeout", "File dialog timed out.")
            return ""
        except Exception as ex:
            # log but continue to Qt fallback
            self._status.showMessage(f"PowerShell error: {ex} — using Qt dialog")

        # ── Qt fallback ──
        start_dir = "/mnt/d" if Path("/mnt/d").exists() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", start_dir,
            "Video files (*.mp4 *.mkv *.avi *.mov *.webm);;All files (*)"
        )
        return path or ""

    def _load_job(self, job_id: str):
        self._status.showMessage(f"Loading job: {job_id}…")
        QApplication.processEvents()
        job = job_manager.load_job(job_id)
        if not job:
            self._show_error("Job not found", f"Could not load job: {job_id}")
            return
        self._job = job
        self._load_job_into_ui(job)

    def _load_job_into_ui(self, job: Job):
        """Full UI reload from Job object."""
        # video
        original = job.meta.input_video
        dubbed = str(job_manager.dubbed_output_path(job.job_id))
        if not Path(dubbed).exists():
            dubbed = ""
        self._video.load_job(original, dubbed)

        # pipeline
        self._pipeline.set_job(job)

        # timeline
        self._timeline.set_job(job)

        # dialogue panel
        self._dialogue_panel.set_job(job)

        # enable controls
        self._btn_speakers.setEnabled(True)
        self._btn_save.setEnabled(True)

        # start waveform extraction
        self._start_waveform_load(job)

        self._status.showMessage(
            f"Job loaded: {job.job_id}  |  "
            f"{len(job.dialogues)} dialogues  |  "
            f"{len(set(d.speaker_id for d in job.dialogues))} speakers"
        )

        # select first dialogue if any
        if job.dialogues:
            self._on_dialogue_selected(job.dialogues[0].id)

    # ──────────────────────────────────────────────────────────────────────
    # Waveform
    # ──────────────────────────────────────────────────────────────────────
    def _start_waveform_load(self, job: Job):
        if self._waveform_thread and self._waveform_thread.isRunning():
            self._waveform_thread.quit()
            self._waveform_thread.wait(500)
        self._waveform_thread = WaveformThread(job, self)
        self._waveform_thread.waveform_ready.connect(self._timeline.set_waveform)
        self._waveform_thread.start()

    # ──────────────────────────────────────────────────────────────────────
    # Signal handlers
    # ──────────────────────────────────────────────────────────────────────
    def _on_video_position(self, ms: int):
        self._timeline.set_playhead(ms)

    def _on_dialogue_selected(self, dialogue_id: str):
        if not self._job:
            return
        self._timeline.set_selected(dialogue_id)
        self._dialogue_panel.select_dialogue(dialogue_id)
        # seek video to dialogue start
        d = self._job.get_dialogue(dialogue_id)
        if d:
            self._video.seek_to_ms(d.start_ms)

    def _on_dialogue_edited(self, dialogue_id: str):
        # Selective refresh — do NOT call _load_job_into_ui
        self._timeline.refresh()
        self._status.showMessage(f"Edited: {dialogue_id}")

    def _on_dialogue_moved(self, dialogue_id: str, start_ms: int, end_ms: int):
        if not self._job:
            return
        d = self._job.get_dialogue(dialogue_id)
        if d:
            before = {"start_ms": d.start_ms, "end_ms": d.end_ms}
            d.start_ms = start_ms
            d.end_ms = end_ms
            after = {"start_ms": start_ms, "end_ms": end_ms}
            self._dialogue_panel.select_dialogue(dialogue_id)
            self._timeline.refresh()
            self._status.showMessage(
                f"Moved {dialogue_id}: {start_ms}ms → {end_ms}ms")

    # ── merge / split ──
    def _on_merge(self, dialogue_id: str):
        if not self._job:
            return
        ids = [d.id for d in self._job.dialogues]
        try:
            idx = ids.index(dialogue_id)
        except ValueError:
            return
        if idx >= len(ids) - 1:
            self._show_info("Cannot merge", "No next dialogue to merge with.")
            return
        d1 = self._job.dialogues[idx]
        d2 = self._job.dialogues[idx + 1]
        # merge
        d1.end_ms = d2.end_ms
        d1.text = d1.text.rstrip() + " " + d2.text.lstrip()
        d1.translation = (d1.translation or "").rstrip() + " " + (d2.translation or "").lstrip()
        self._job.dialogues.pop(idx + 1)
        self._timeline.set_job(self._job)
        self._dialogue_panel.set_job(self._job)
        self._dialogue_panel.select_dialogue(dialogue_id)
        self._status.showMessage(f"Merged {dialogue_id} with {d2.id}")

    def _on_split(self, dialogue_id: str):
        if not self._job:
            return
        ids = [d.id for d in self._job.dialogues]
        try:
            idx = ids.index(dialogue_id)
        except ValueError:
            return
        d = self._job.dialogues[idx]
        mid = (d.start_ms + d.end_ms) // 2
        if mid <= d.start_ms or mid >= d.end_ms:
            self._show_info("Cannot split", "Dialogue is too short to split.")
            return
        # create new id
        new_id = f"{d.id}_b"
        new_d = Dialogue(
            id=new_id,
            speaker_id=d.speaker_id,
            start_ms=mid,
            end_ms=d.end_ms,
            start_time="",
            end_time="",
            text=d.text,
            translation=d.translation,
            dub_enabled=d.dub_enabled,
            source_lang=d.source_lang,
            target_lang=d.target_lang,
            emotion=d.emotion,
            intensity=d.intensity,
        )
        d.end_ms = mid
        self._job.dialogues.insert(idx + 1, new_d)
        self._timeline.set_job(self._job)
        self._dialogue_panel.set_job(self._job)
        self._dialogue_panel.select_dialogue(dialogue_id)
        self._status.showMessage(f"Split {dialogue_id} at {mid}ms")

    # ── pipeline ──
    def _on_stage_completed(self, stage_id: str):
        """Selective reload — reload only new data, not full UI."""
        if not self._job:
            return

        if stage_id in ("01_asr", "01b_diarization"):
            new_job = job_manager.load_job(self._job.job_id)
            if new_job:
                # BUG2 FIX: carry over existing speaker names/colors so they
                # are not wiped when the fresh JSON has no ui_state yet
                for spk, name in self._job.speaker_names.items():
                    if spk not in new_job.speaker_names or not new_job.speaker_names[spk]:
                        new_job.speaker_names[spk] = name
                for spk, color in self._job.speaker_colors.items():
                    if spk not in new_job.speaker_colors or not new_job.speaker_colors[spk]:
                        new_job.speaker_colors[spk] = color
                # Ensure every dialogue speaker has a name and color assigned
                from app.models.speaker_model import SpeakerRegistry, DEFAULT_COLORS
                _color_idx = len(new_job.speaker_colors)
                for d in new_job.dialogues:
                    spk = d.speaker_id
                    if spk not in new_job.speaker_names:
                        new_job.speaker_names[spk] = spk.replace("_", " ").title()
                    if spk not in new_job.speaker_colors:
                        new_job.speaker_colors[spk] = DEFAULT_COLORS[_color_idx % len(DEFAULT_COLORS)]
                        _color_idx += 1
                self._job = new_job
                self._timeline.set_job(new_job)
                self._dialogue_panel.set_job(new_job)
                self._start_waveform_load(new_job)
                # Auto-save the speaker assignments so ui_state.json is populated
                job_manager.save_ui_state(new_job)
                self._status.showMessage(
                    f"Stage {stage_id} done — "
                    f"{len(set(d.speaker_id for d in new_job.dialogues))} speakers detected, "
                    f"{len(new_job.dialogues)} dialogues loaded.")
        elif stage_id == "03_translation":
            new_job = job_manager.load_job(self._job.job_id)
            if new_job:
                # carry over speaker data here too
                new_job.speaker_names = dict(self._job.speaker_names)
                new_job.speaker_colors = dict(self._job.speaker_colors)
                self._job = new_job
                self._timeline.set_job(new_job)
                self._dialogue_panel.set_job(new_job)
                self._status.showMessage(f"Stage {stage_id} done — translations loaded.")
        elif stage_id == "05_assembly":
            dubbed = str(job_manager.dubbed_output_path(self._job.job_id))
            if Path(dubbed).exists():
                self._video.load_job(self._job.meta.input_video, dubbed)
            self._status.showMessage(f"Stage {stage_id} done — dubbed video ready.")
        else:
            self._status.showMessage(f"Stage completed: {stage_id}")

    def _on_stage_failed(self, stage_id: str):
        self._status.showMessage(f"Stage FAILED: {stage_id}")

    def _auto_save_profiles(self):
        """
        BUG3 FIX: Called automatically before 02_emotion runs.
        Extracts speaker profiles if they don't already exist,
        so the user doesn't have to remember to do it manually.
        """
        if not self._job:
            return
        profiles_dir = job_manager.speaker_profiles_dir(self._job.job_id)
        speakers = list(set(d.speaker_id for d in self._job.dialogues))
        missing = [s for s in speakers
                   if not (profiles_dir / f"{s}.wav").exists()]
        if not missing:
            self._pipeline.log(
                f"  Speaker profiles already exist for: {', '.join(speakers)}", "")
            return
        self._pipeline.log(
            f"  Auto-extracting profiles for: {', '.join(missing)}", "")
        try:
            from app.job_manager import extract_speaker_profiles
            def _cb(i, n, spk):
                self._pipeline.log(f"  Extracting {spk}...", "")
                from PyQt6.QtWidgets import QApplication
                QApplication.processEvents()
            durations = extract_speaker_profiles(self._job, _cb)
            job_manager.save_ui_state(self._job)
            for spk, dur in durations.items():
                warn = " ⚠ < 6s" if dur < 6.0 else ""
                name = self._job.get_speaker_name(spk)
                self._pipeline.log(f"  ✓ {name}: {dur:.1f}s{warn}", "")
            self._status.showMessage(
                f"Auto-saved {len(durations)} speaker profile(s).")
        except FileNotFoundError:
            self._pipeline.log(
                "  ⚠ Original WAV not found — skipping profile extraction.", "")
        except Exception as ex:
            self._pipeline.log(f"  ⚠ Profile extraction error: {ex}", "")

    def _on_review_gate(self, stage_id: str, message: str):
        dlg = QMessageBox(self)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setWindowTitle("Review Gate")
        dlg.setText(f"Stage '{stage_id}' completed.\n\n{message}")
        dlg.setIcon(QMessageBox.Icon.Information)
        self._center_dialog(dlg)
        dlg.exec()

    # ── toolbar actions ──
    def _open_speaker_manager(self):
        if not self._job:
            return
        dlg = SpeakerManagerDialog(self._job, self)
        dlg.profiles_confirmed.connect(self._on_profiles_confirmed)
        dlg.exec()
        # refresh timeline labels after rename/color
        if self._job:
            self._timeline.set_job(self._job)

    def _on_profiles_confirmed(self):
        self._status.showMessage("Speaker profiles confirmed.")

    def _save_all(self):
        if not self._job:
            return
        try:
            job_manager.save_ui_state(self._job)
            job_manager.save_translations(self._job)
            job_manager.save_transcription(self._job)
            self._status.showMessage("All changes saved.")
        except Exception as ex:
            self._show_error("Save failed", str(ex))

    def _toggle_play(self):
        from PyQt6.QtMultimedia import QMediaPlayer
        if hasattr(self._video, '_player'):
            p = self._video._player
            if p.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                p.pause()
            else:
                p.play()

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    def _show_error(self, title: str, text: str):
        dlg = QMessageBox(self)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setWindowTitle(title)
        dlg.setText(text)
        dlg.setIcon(QMessageBox.Icon.Critical)
        self._center_dialog(dlg)
        dlg.exec()

    def _show_info(self, title: str, text: str):
        dlg = QMessageBox(self)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setWindowTitle(title)
        dlg.setText(text)
        dlg.setIcon(QMessageBox.Icon.Information)
        self._center_dialog(dlg)
        dlg.exec()

    def _center_dialog(self, dlg):
        dlg.adjustSize()
        geo = self.geometry()
        dlg.move(
            geo.center().x() - dlg.width() // 2,
            geo.center().y() - dlg.height() // 2,
        )


def _win_to_wsl(win_path: str) -> str:
    """Convert Windows path (e.g. D:\\$\\my\\file.mp4) to WSL path."""
    # Try wslpath first
    try:
        result = subprocess.run(
            ["wslpath", win_path], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    # Manual fallback
    import re
    p = win_path.replace("\\", "/")
    m = re.match(r"^([A-Za-z]):(.*)", p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2)
        return f"/mnt/{drive}{rest}"
    return win_path