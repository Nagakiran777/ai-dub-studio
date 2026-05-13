"""
DubStudio Pro — Job data model (pure dataclasses, no Qt dependency).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class WordToken:
    word: str
    start_ms: int
    end_ms: int
    speaker_id: str = ""


@dataclass
class Dialogue:
    id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    start_time: str
    end_time: str
    text: str
    is_nonlexical: bool = False
    words: List[WordToken] = field(default_factory=list)
    # translation fields (from 03_translations.json)
    emotion: str = ""
    intensity: float = 0.0
    translation: str = ""
    is_passthrough: bool = False
    dub_enabled: bool = True
    source_lang: str = "en"
    target_lang: str = "en"

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class StageStatus:
    stage_id: str
    status: str = "pending"   # pending | running | completed | failed | skipped
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None


@dataclass
class TTSEntry:
    id: str
    speaker_id: str
    wav_path: str
    original_duration_ms: float
    synthesized_duration_ms: float
    target_duration_ms: float
    final_duration_ms: float
    stretch_ratio: float
    emotion: str
    intensity: float
    tts_text: str
    success: bool
    reference_audio: str = ""
    stretch_within_limit: bool = True


@dataclass
class JobMeta:
    job_id: str
    input_video: str
    input_file: str
    created_at: str


@dataclass
class Job:
    meta: JobMeta
    dialogues: List[Dialogue] = field(default_factory=list)
    stages: Dict[str, StageStatus] = field(default_factory=dict)
    tts_entries: Dict[str, TTSEntry] = field(default_factory=dict)
    # ui_state sidecar
    speaker_names: Dict[str, str] = field(default_factory=dict)
    speaker_colors: Dict[str, str] = field(default_factory=dict)
    dialogue_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def job_id(self) -> str:
        return self.meta.job_id

    def get_dialogue(self, dialogue_id: str) -> Optional[Dialogue]:
        for d in self.dialogues:
            if d.id == dialogue_id:
                return d
        return None

    def get_speaker_name(self, speaker_id: str) -> str:
        return self.speaker_names.get(speaker_id, speaker_id)

    def get_speaker_color(self, speaker_id: str) -> str:
        return self.speaker_colors.get(speaker_id, "#E8A020")