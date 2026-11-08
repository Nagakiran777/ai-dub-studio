"""
Dub speed + trim geometry — single source of truth.

Used by BOTH the web server (to show/preview a dub) and Stage 05 assembly
(to actually build it when exporting), so the preview and the final video can
never disagree about where a dub sits, how long it is, or how fast it plays.
Standard library only, so it can be imported from the base conda env and from
the dub_assembly env alike.

Speed semantics: "speed" is a duration-rate change applied with a
pitch-preserving time-stretch. A rate of 1.25 makes the clip about 0.8x as
long while aiming to retain its pitch; 0.8 makes it longer. Extreme rates can
still introduce audio artifacts.

DEFAULT SPEED IS 1.0 (normal). Automatic length-matching is NOT applied
silently any more; `auto_speed()` is only the value behind the "Fit to
original" button, which the user can press per dialogue.

Dub geometry (all stored in 03_translations.json, all optional):
  dub_start_ms       where the dub starts on the timeline (null = same as the
                     original line's start_ms)
  dub_trim_start_ms  cut this much off the START of the synthesized clip
  dub_trim_end_ms    keep the clip only up to this point (null = whole clip)
                     (both trim values are positions inside the NATURAL clip)
  dub_speed          manual speed (null = 1.0)
  dub_deleted        true = this dub is removed; the original voice plays
"""

from __future__ import annotations
from typing import Optional

# "Fit to original" stays inside a range that still sounds like natural speech.
MIN_AUTO_SPEED = 0.5
MAX_AUTO_SPEED = 1.5
# The manual slider is allowed a wider range.
MIN_MANUAL_SPEED = 0.25
MAX_MANUAL_SPEED = 2.0
# Within +/-5% of the original length it's not worth touching the audio.
DEADBAND = 0.05
# A trimmed dub can never be shorter than this.
MIN_KEPT_MS = 100.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _num(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def auto_speed(dub_ms: Optional[float], original_ms: Optional[float]) -> float:
    """Speed that would make the dub as long as the original dialogue (clamped).

    dub longer than original  -> > 1.0 (speed up to fit)
    dub shorter than original -> < 1.0 (slow down to fill)

    Only used for the "Fit to original" button — never applied automatically.
    """
    dub = _num(dub_ms, 0.0) or 0.0
    orig = _num(original_ms, 0.0) or 0.0
    if dub <= 0 or orig <= 0:
        return 1.0
    ratio = dub / orig
    if abs(ratio - 1.0) <= DEADBAND:
        return 1.0
    return round(_clamp(ratio, MIN_AUTO_SPEED, MAX_AUTO_SPEED), 2)


def effective_speed(manual: Optional[float], dub_ms: Optional[float] = None,
                    original_ms: Optional[float] = None) -> float:
    """The speed actually used: the user's manual value, otherwise normal (1.0).

    `dub_ms` / `original_ms` are accepted only so older callers keep working.
    """
    m = _num(manual)
    if m is None:
        return 1.0
    return round(_clamp(m, MIN_MANUAL_SPEED, MAX_MANUAL_SPEED), 2)


def dub_parts_geometry(natural_ms: Optional[float], trans_entry: Optional[dict],
                       orig_start_ms: float, orig_end_ms: float) -> list:
    """All pieces a dub has been cut into, each with its own geometry.

    Piece "p0" is the dialogue's own dub (the dub_* fields). Splitting a dub in the Dubbed
    timeline adds more pieces to `dub_parts` (a list of {id, dub_start_ms, dub_trim_start_ms,
    dub_trim_end_ms, dub_speed, dub_deleted}); each one plays its own part of the SAME
    synthesized clip, so it can be moved, trimmed, sped up or deleted on its own.
    """
    t = trans_entry or {}
    first = dub_geometry(natural_ms, t, orig_start_ms, orig_end_ms)
    first["part_id"] = "p0"
    out = [first]
    for raw in (t.get("dub_parts") or []):
        if not isinstance(raw, dict):
            continue
        piece = {k: raw.get(k) for k in ("dub_start_ms", "dub_trim_start_ms", "dub_trim_end_ms", "dub_speed", "dub_deleted")}
        g = dub_geometry(natural_ms, piece, orig_start_ms, orig_end_ms)
        g["part_id"] = str(raw.get("id") or f"p{len(out)}")
        out.append(g)
    return out


def dub_geometry(natural_ms: Optional[float], trans_entry: Optional[dict],
                 orig_start_ms: float, orig_end_ms: float) -> dict:
    """Where a dub sits, which part of the clip is kept and how long it ends up.

    natural_ms     length of the synthesized clip as written by Stage 04
    trans_entry    the dialogue's entry in 03_translations.json (may be None)
    orig_start_ms  the ORIGINAL line's start/end (used only as defaults)
    """
    t = trans_entry or {}
    natural = max(0.0, _num(natural_ms, 0.0) or 0.0)

    tin = _clamp(_num(t.get("dub_trim_start_ms"), 0.0) or 0.0, 0.0, max(0.0, natural - MIN_KEPT_MS))
    tout_raw = _num(t.get("dub_trim_end_ms"))
    tout = natural if tout_raw is None else _clamp(tout_raw, tin + MIN_KEPT_MS, max(natural, tin + MIN_KEPT_MS))
    tout = min(tout, natural) if natural > 0 else tout
    kept = max(0.0, tout - tin)

    speed = effective_speed(t.get("dub_speed"))
    start = _num(t.get("dub_start_ms"))
    if start is None:
        start = float(orig_start_ms)
    start = max(0.0, start)
    length = kept / speed if speed > 0 else kept
    orig_len = max(0.0, float(orig_end_ms) - float(orig_start_ms))

    return {
        "natural_ms":  round(natural, 1),
        "trim_in_ms":  round(tin, 1),
        "trim_out_ms": round(tout, 1),
        "kept_ms":     round(kept, 1),
        "speed":       speed,
        "speed_fit":   auto_speed(kept, orig_len),
        "dub_start_ms": round(start, 1),
        "dub_len_ms":  round(length, 1),
        "dub_end_ms":  round(start + length, 1),
        "original_ms": round(orig_len, 1),
        "deleted":     bool(t.get("dub_deleted")),
        "moved":       t.get("dub_start_ms") is not None,
        "trimmed":     tin > 0.5 or (natural > 0 and tout < natural - 0.5),
    }
