// Play the KEPT part of a dub clip at its own speed, the way the export builds it:
// start at the trim-in point, stop at the trim-out point, pitch-preserving rate, and the
// same per-dialogue loudness gain the export applies (so preview == export).

let _ctx = null
const getCtx = () => {
  if (!_ctx) {
    const C = window.AudioContext || window.webkitAudioContext
    if (!C) return null
    _ctx = new C()
  }
  return _ctx
}

/** element → gain (layer/dialogue level) → mute (scheduled silence) → speakers */
function ensureGraph(el) {
  if (el._gainNode) return true
  const ctx = getCtx()
  if (!ctx) return false
  const src = ctx.createMediaElementSource(el)
  el._gainNode = ctx.createGain()
  el._muteNode = ctx.createGain()
  src.connect(el._gainNode); el._gainNode.connect(el._muteNode); el._muteNode.connect(ctx.destination)
  return true
}

/**
 * Per-element level through WebAudio (an <audio>'s own volume can't go above 1, but the Mixer
 * and per-dialogue volume go up to 300 %). Only wires the graph up when a level other than 1
 * is needed, so plain playback stays untouched.
 */
export function setElementGain(el, gain) {
  const g = Number.isFinite(gain) && gain >= 0 ? gain : 1
  try {
    if (!el._gainNode && Math.abs(g - 1) < 0.005) return
    if (!ensureGraph(el)) return
    el._gainNode.gain.value = g
    if (_ctx.state === 'suspended') _ctx.resume()
  } catch { /* no WebAudio → plain playback */ }
}

/**
 * The VOCALS layer in the dubbed preview: the original vocals, silenced exactly where a dub
 * replaces a dialogue — screams, breaths and every other vocal sound stay audible, like in
 * the export. `windows` = [{s,e}] in ms (merged, sorted). Gain ramps (10 ms, like the export's
 * fades) are scheduled on the audio clock from the element's current position, so the mutes
 * land on time; call again after play / seek / speed change (and now and then to absorb drift).
 */
export function scheduleMute(el, windows, rate = 1, fadeMs = 10) {
  try {
    if (!ensureGraph(el)) return
    const ctx = _ctx
    if (ctx.state === 'suspended') ctx.resume()
    const p = el._muteNode.gain
    const now = ctx.currentTime
    const t = el.currentTime * 1000
    const f = fadeMs / 1000
    const at = (ms) => now + (ms - t) / 1000 / (rate || 1)
    p.cancelScheduledValues(0)
    p.setValueAtTime(windows.some(w => t >= w.s && t < w.e) ? 0 : 1, now)
    for (const w of windows) {
      if (w.e <= t) continue
      if (w.s > t + 90000) break                       // schedule only the next ~90 s of media
      if (w.s > t) {
        p.setValueAtTime(1, Math.max(now, at(w.s) - f))
        p.linearRampToValueAtTime(0, Math.max(now + f, at(w.s)))
      }
      p.setValueAtTime(0, Math.max(now, at(w.e)))
      p.linearRampToValueAtTime(1, Math.max(now + f, at(w.e) + f))
    }
  } catch { /* ignore */ }
}

/** Where the original voice is replaced: the ORIGINAL bar of every dialogue that has a (non-deleted) dub. */
export function silenceWindows(dialogues, ttsMap) {
  const raw = []
  for (const d of dialogues || []) {
    const e = ttsMap?.[d.id]
    if (e?.success && (e.parts || []).some(p => !p.deleted)) raw.push({ s: d.start_ms, e: d.end_ms })
  }
  raw.sort((a, b) => a.s - b.s)
  const out = []
  for (const w of raw) {
    const last = out[out.length - 1]
    if (last && w.s <= last.e) last.e = Math.max(last.e, w.e)
    else out.push({ ...w })
  }
  return out
}

// `au` is a reused <audio>; loading a new src resets its playbackRate, so the rate is set
// both as the default and after the src (see VideoPlayer.setRate).
export function playDubClip(au, url, { speed = 1, trimInMs = 0, trimOutMs = null, naturalMs = 0, gain = 1 } = {}) {
  clearTimeout(au._dubTimer)
  au.pause()
  const rate  = speed || 1
  const outMs = trimOutMs == null ? naturalMs : trimOutMs
  const keptMs = Math.max(0, (outMs || 0) - trimInMs)
  au.defaultPlaybackRate = rate
  au.src = url
  au.playbackRate = rate
  au.preservesPitch = true; au.webkitPreservesPitch = true
  setElementGain(au, gain)
  const go = () => {
    try { au.currentTime = trimInMs / 1000 } catch {}
    au.play().catch(() => {})
    if (keptMs > 0) au._dubTimer = setTimeout(() => au.pause(), keptMs / rate + 40)
  }
  if (trimInMs > 0) au.addEventListener('loadedmetadata', go, { once: true })
  else go()
}

export function stopDubClip(au) {
  clearTimeout(au._dubTimer)
  au.pause()
}
