// Waveform peaks, computed in the browser from the same audio the app already
// serves (original voice slice / synthesized dub) — no extra backend work and
// any format the browser can decode works.
//
//   getPeaks(url, key) -> Promise<{ peaks: Float32Array, durationMs: number }>
//
// peaks[i] is the loudest sample (0..1, normalised per clip) in the i-th
// ~10 ms slice. Results are cached by `key`; the same request in flight is
// shared; at most MAX_PARALLEL decodes run at once so a timeline with
// hundreds of bars doesn't hammer the server.

const BUCKET_MS    = 10
const MAX_BUCKETS  = 4000
const MIN_BUCKETS  = 32
const MAX_PARALLEL = 3

const cache    = new Map()   // key -> Promise
const queue    = []
let   running  = 0

function pump() {
  while (running < MAX_PARALLEL && queue.length) {
    const job = queue.shift()
    running++
    job().finally(() => { running--; pump() })
  }
}

async function decode(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const buf = await res.arrayBuffer()
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext
  const ctx = new Ctx(1, 1, 22050)           // only used to decode; resampled to 22.05 kHz
  const audio = await new Promise((resolve, reject) => {
    const p = ctx.decodeAudioData(buf, resolve, reject)
    if (p && p.catch) p.catch(reject)
  })
  const ch = []
  for (let c = 0; c < audio.numberOfChannels; c++) ch.push(audio.getChannelData(c))
  const durationMs = audio.duration * 1000
  const n = Math.max(MIN_BUCKETS, Math.min(MAX_BUCKETS, Math.ceil(durationMs / BUCKET_MS)))
  const peaks = new Float32Array(n)
  const per = audio.length / n
  let clipMax = 0
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * per), b = Math.max(a + 1, Math.floor((i + 1) * per))
    let m = 0
    for (const d of ch) for (let j = a; j < b && j < d.length; j++) { const v = Math.abs(d[j]); if (v > m) m = v }
    peaks[i] = m
    if (m > clipMax) clipMax = m
  }
  const norm = Math.max(clipMax, 0.02)       // don't blow silence up into a full-height wave
  for (let i = 0; i < n; i++) peaks[i] = Math.min(1, peaks[i] / norm)
  return { peaks, durationMs }
}

export function getPeaks(url, key = url) {
  if (cache.has(key)) return cache.get(key)
  const p = new Promise((resolve, reject) => {
    queue.push(() => decode(url).then(resolve, reject))
    pump()
  })
  cache.set(key, p)
  p.catch(() => cache.delete(key))           // failed → allow a retry later
  return p
}

export function clearPeaks() { cache.clear() }
