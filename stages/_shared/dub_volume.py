"""
Per-dialogue volume matching — standard library only, so the web server can use the SAME rule
as Stage 05 and the preview sounds as loud as the export.

    gain = loudness(original line) / loudness(dubbed clip)

"Loudness" is the RMS of the speaking part only: audio is cut into 25 ms frames and frames
quieter than 10 % of the loudest frame (pauses, breaths, tails) are ignored. The original
line's loudness is measured on ITS OWN separated vocals slice — not on the whole video —
so a whisper stays a whisper and a shout stays a shout.
"""

from __future__ import annotations

import wave
from array import array
from math import sqrt
from typing import Optional

FRAME_MS     = 25
KEEP_RATIO   = 0.1      # frames below 10 % of the loudest frame don't count
RMS_FLOOR    = 1e-4
GAIN_MIN     = 0.1
GAIN_MAX     = 8.0


def read_mono(path, start_ms: float = 0.0, end_ms: Optional[float] = None, target_sr: int = 16000):
    """(samples as floats −1..1 at roughly target_sr, sr) for a PCM wav slice, or (None, 0)."""
    try:
        with wave.open(str(path), "rb") as wf:
            sr, nch, sw, n = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
            a = max(0, int(start_ms / 1000.0 * sr))
            b = n if end_ms is None else min(n, int(end_ms / 1000.0 * sr))
            if b <= a:
                return None, 0
            wf.setpos(a)
            raw = wf.readframes(b - a)
    except Exception:
        return None, 0
    if sw == 2:
        arr = array("h"); arr.frombytes(raw); scale = 32768.0
    elif sw == 4:
        arr = array("i"); arr.frombytes(raw); scale = 2147483648.0
    elif sw == 1:
        arr = array("B"); arr.frombytes(raw); scale = 128.0
        arr = array("h", [(x - 128) for x in arr])
    else:
        return None, 0
    step = max(1, sr // target_sr)
    frames = len(arr) // nch
    out = []
    for i in range(0, frames, step):
        base = i * nch
        v = arr[base] if nch == 1 else (arr[base] + arr[base + 1]) / 2.0
        out.append(v / scale)
    return out, sr // step


def active_rms(samples, sr: int) -> float:
    """Same algorithm as Stage 05's numpy version."""
    if not samples:
        return 0.0
    n = max(1, int(sr * FRAME_MS / 1000))
    m = len(samples) // n
    if m < 1:
        return sqrt(sum(x * x for x in samples) / len(samples))
    sums = []
    for f in range(m):
        seg = samples[f * n:(f + 1) * n]
        sums.append(sum(x * x for x in seg))
    frame_rms = [sqrt(s / n) for s in sums]
    peak = max(frame_rms)
    if peak < RMS_FLOOR:
        return 0.0
    keep = [s for s, r in zip(sums, frame_rms) if r >= KEEP_RATIO * peak]
    return sqrt(sum(keep) / (len(keep) * n))


def dialogue_gain(vocals_wav, start_ms: float, end_ms: float, dub_wav) -> Optional[dict]:
    """{"gain", "orig_rms", "dub_rms"} or None when either file can't be read (float wavs …)."""
    ref, rsr = read_mono(vocals_wav, start_ms, end_ms)
    dub, dsr = read_mono(dub_wav)
    if ref is None or dub is None:
        return None
    r, d = active_rms(ref, rsr), active_rms(dub, dsr)
    if r < 1e-4 or d < RMS_FLOOR:
        return {"gain": 1.0, "orig_rms": round(r, 5), "dub_rms": round(d, 5)}
    return {"gain": round(max(GAIN_MIN, min(GAIN_MAX, r / d)), 3), "orig_rms": round(r, 5), "dub_rms": round(d, 5)}
