"""
Dub speed matching — single source of truth.

Used by BOTH the web server (to show/preview the speed) and Stage 05 assembly
(to actually apply it when exporting), so the preview and the final video can
never disagree about how fast a dub plays. Standard library only, so it can be
imported from the base conda env and from the dub_assembly env alike.

Semantics: "speed" is a duration-rate change applied with pitch-preserving time-stretch.
A rate of 1.25 makes the clip about 0.8x as long while aiming to retain its original
pitch; 0.8 makes it longer. Extreme rates can still introduce audio artifacts.
"""

from __future__ import annotations
from typing import Optional

# Automatic matching stays inside a range that still sounds like natural speech.
MIN_AUTO_SPEED = 0.5
MAX_AUTO_SPEED = 1.5
# The manual slider is allowed a wider range.
MIN_MANUAL_SPEED = 0.25
MAX_MANUAL_SPEED = 2.0
# Within +/-5% of the original length it's not worth touching the audio.
DEADBAND = 0.05


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def auto_speed(dub_ms: Optional[float], original_ms: Optional[float]) -> float:
    """Speed that makes the dub as long as the original dialogue (clamped).

    dub longer than original  -> > 1.0 (speed up to fit)
    dub shorter than original -> < 1.0 (slow down to fill)
    """
    try:
        dub = float(dub_ms or 0)
        orig = float(original_ms or 0)
    except (TypeError, ValueError):
        return 1.0
    if dub <= 0 or orig <= 0:
        return 1.0
    ratio = dub / orig
    if abs(ratio - 1.0) <= DEADBAND:
        return 1.0
    return round(_clamp(ratio, MIN_AUTO_SPEED, MAX_AUTO_SPEED), 2)


def effective_speed(manual: Optional[float], dub_ms: Optional[float],
                    original_ms: Optional[float]) -> float:
    """The speed actually used: the user's manual value if they set one,
    otherwise the automatic length-matching value."""
    if manual is not None:
        try:
            return round(_clamp(float(manual), MIN_MANUAL_SPEED, MAX_MANUAL_SPEED), 2)
        except (TypeError, ValueError):
            pass
    return auto_speed(dub_ms, original_ms)
