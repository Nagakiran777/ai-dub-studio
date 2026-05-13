"""
DubStudio Pro — Speaker color and naming helpers.
"""
from __future__ import annotations
from typing import Dict, Optional

# Default speaker palette — 10 distinct colors
DEFAULT_COLORS = [
    "#E8A020",  # golden (primary)
    "#4FC3F7",  # sky blue
    "#81C784",  # sage green
    "#F06292",  # rose
    "#CE93D8",  # lavender
    "#FFB74D",  # amber
    "#4DB6AC",  # teal
    "#FF8A65",  # coral
    "#A1887F",  # warm brown
    "#90A4AE",  # steel
]


class SpeakerRegistry:
    """Manages speaker display names and colors for a job."""

    def __init__(self) -> None:
        self._names: Dict[str, str] = {}
        self._colors: Dict[str, str] = {}
        self._index: int = 0

    def load(self, names: Dict[str, str], colors: Dict[str, str]) -> None:
        self._names = dict(names)
        self._colors = dict(colors)

    def get_name(self, speaker_id: str) -> str:
        return self._names.get(speaker_id, _pretty(speaker_id))

    def set_name(self, speaker_id: str, name: str) -> None:
        self._names[speaker_id] = name

    def get_color(self, speaker_id: str) -> str:
        if speaker_id not in self._colors:
            self._colors[speaker_id] = DEFAULT_COLORS[self._index % len(DEFAULT_COLORS)]
            self._index += 1
        return self._colors[speaker_id]

    def set_color(self, speaker_id: str, color: str) -> None:
        self._colors[speaker_id] = color

    def all_speaker_ids(self) -> list:
        return list(self._names.keys())

    def to_dicts(self):
        return dict(self._names), dict(self._colors)

    def ensure_speaker(self, speaker_id: str) -> None:
        """Register speaker if not already known."""
        if speaker_id not in self._names:
            self._names[speaker_id] = _pretty(speaker_id)
        if speaker_id not in self._colors:
            self.get_color(speaker_id)


def _pretty(speaker_id: str) -> str:
    """Convert 'speaker_01' → 'Speaker 01'."""
    parts = speaker_id.replace("_", " ").title()
    return parts