/**
 * Read-only transliteration for the editor's "view original in another script".
 * (This is the opposite direction from TransliterationInput, which turns Latin
 * typing INTO a native script.)
 *
 * - Fully offline: no network, no rate limits, instant.
 * - The source script is detected from the text itself (Unicode ranges), so it
 *   works for auto-detected jobs and mixed-script lines without a language code.
 * - Indic scripts use @indic-transliteration/sanscript, then a small clean-up so
 *   the Latin output reads like everyday spelling ("chala rojulu ayindi kada")
 *   instead of scholarly IAST ("cālā rojulu ayiṃdi kadā").
 * - Other scripts use any-ascii, loaded lazily on first use (separate chunk).
 * - Results are cached in memory AND localStorage, so toggling the view on and
 *   off repeatedly (or reloading) never recomputes the same line twice.
 */
import Sanscript from '@indic-transliteration/sanscript'

// ── Indic scripts ────────────────────────────────────────────────────────────
export const INDIC_SCRIPTS = {
  devanagari: { scheme: 'devanagari', range: /[\u0900-\u097F]/g, label: 'Devanagari (Hindi / Marathi)' },
  bengali:    { scheme: 'bengali',    range: /[\u0980-\u09FF]/g, label: 'Bengali' },
  gurmukhi:   { scheme: 'gurmukhi',   range: /[\u0A00-\u0A7F]/g, label: 'Gurmukhi (Punjabi)' },
  gujarati:   { scheme: 'gujarati',   range: /[\u0A80-\u0AFF]/g, label: 'Gujarati' },
  oriya:      { scheme: 'oriya',      range: /[\u0B00-\u0B7F]/g, label: 'Odia' },
  tamil:      { scheme: 'tamil',      range: /[\u0B80-\u0BFF]/g, label: 'Tamil' },
  telugu:     { scheme: 'telugu',     range: /[\u0C00-\u0C7F]/g, label: 'Telugu' },
  kannada:    { scheme: 'kannada',    range: /[\u0C80-\u0CFF]/g, label: 'Kannada' },
  malayalam:  { scheme: 'malayalam',  range: /[\u0D00-\u0D7F]/g, label: 'Malayalam' },
}

// ── Other scripts (romanized via any-ascii) ──────────────────────────────────
const OTHER_SCRIPTS = {
  kana:     { range: /[\u3040-\u30FF]/g,                 label: 'Japanese' },
  hangul:   { range: /[\uAC00-\uD7AF\u1100-\u11FF]/g,    label: 'Korean' },
  han:      { range: /[\u4E00-\u9FFF]/g,                 label: 'Chinese' },
  cyrillic: { range: /[\u0400-\u04FF]/g,                 label: 'Cyrillic (Russian etc.)' },
  greek:    { range: /[\u0370-\u03FF]/g,                 label: 'Greek' },
  arabic:   { range: /[\u0600-\u06FF]/g,                 label: 'Arabic / Persian / Urdu' },
  hebrew:   { range: /[\u0590-\u05FF]/g,                 label: 'Hebrew' },
  thai:     { range: /[\u0E00-\u0E7F]/g,                 label: 'Thai' },
}
// Scripts where Latin output is only a rough guide (vowels unwritten, or CJK
// readings that need a dictionary to be right) — the UI flags these.
const APPROXIMATE = new Set(['kana', 'hangul', 'han', 'arabic', 'hebrew', 'thai'])

const count = (text, re) => (text.match(re) || []).length

/** Dominant script of a piece of text: an INDIC_SCRIPTS key, an OTHER_SCRIPTS
 *  key, 'latin' (letters only, nothing to convert), or null (no letters). */
export function detectScript(text) {
  if (!text) return null
  let best = null, bestN = 0
  for (const [id, s] of Object.entries(INDIC_SCRIPTS)) {
    const n = count(text, s.range); if (n > bestN) { best = id; bestN = n }
  }
  if (best) return best
  // Japanese text mixes kana with kanji; treat any kana as Japanese, bare Han as Chinese
  for (const id of ['kana', 'hangul', 'han', 'cyrillic', 'greek', 'arabic', 'hebrew', 'thai']) {
    if (count(text, OTHER_SCRIPTS[id].range) > 0) return id
  }
  return /\p{L}/u.test(text) ? 'latin' : null
}

export function scriptLabel(script) {
  return INDIC_SCRIPTS[script]?.label || OTHER_SCRIPTS[script]?.label || (script === 'latin' ? 'Latin letters' : '')
}

/** Target scripts that make sense for text written in `script`. */
export function targetOptionsFor(script) {
  if (!script || script === 'latin') return []
  const opts = [{ id: 'latin', label: 'English (Latin letters)' }]
  if (INDIC_SCRIPTS[script]) {
    for (const [id, s] of Object.entries(INDIC_SCRIPTS)) if (id !== script) opts.push({ id, label: s.label })
  }
  return opts
}

export const isApproximate = (script) => APPROXIMATE.has(script)

// ── Latin clean-up for Indic text ────────────────────────────────────────────
const MALAYALAM_CHILLU = {
  '\u0D7A': '\u0D23\u0D4D', '\u0D7B': '\u0D28\u0D4D', '\u0D7C': '\u0D30\u0D4D',
  '\u0D7D': '\u0D32\u0D4D', '\u0D7E': '\u0D33\u0D4D', '\u0D7F': '\u0D15\u0D4D',
}
const PLAIN = {
  'ā': 'a', 'ī': 'i', 'ū': 'u', 'ṝ': 'ri', 'ṛ': 'ri', 'ḹ': 'li', 'ḷ': 'l',
  'ē': 'e', 'ō': 'o', 'è': 'e', 'ò': 'o', 'ṭ': 't', 'ḍ': 'd', 'ṇ': 'n',
  'ś': 'sh', 'ṣ': 'sh', 'ḥ': 'h', 'ṅ': 'n', 'ñ': 'n', 'ṟ': 'r', 'ṉ': 'n', 'ḻ': 'zh',
}

// Northern languages drop the written-but-silent final "a" (राम = "raam", not
// "raama") and end nasal words in "n" (हैं = "hain"). The southern ones (Telugu,
// Kannada, Malayalam, Tamil) pronounce final vowels and end nasals in "m".
const NORTHERN = new Set(['devanagari', 'bengali', 'gurmukhi', 'gujarati', 'oriya'])

function simplifyIast(s, script) {
  s = s.normalize('NFC')
  const northern = NORTHERN.has(script)
  if (northern) {
    s = s.replace(/([^aāiīuūṛṝḷḹeēoōèò\s\p{P}\d])a(?=[\s\p{P}]|$)/gu, '$1')   // silent final "a"
  }
  s = s.replace(/~/g, 'n')                       // chandrabindu (nasalised vowel), e.g. हूँ -> hun
  // anusvara: 'm' before p/b/m; at the end of a word 'n' (north) / 'm' (south); otherwise 'n'
  s = s.replace(/[ṃṁ](?=[pbm])/g, 'm').replace(/[ṃṁ](?=[^\p{L}]|$)/gu, northern ? 'n' : 'm').replace(/[ṃṁ]/g, 'n')
  // IAST 'c' is the "ch" sound and 'ch' the aspirated one
  s = s.replace(/ch/g, '\u0001').replace(/c/g, 'ch').replace(/\u0001/g, 'chh')
  s = s.replace(/[āīūṝṛḹḷēōèòṭḍṇśṣḥṅñṟṉḻ]/g, (c) => PLAIN[c])
  // anything left over (stray combining marks) is dropped
  return s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').normalize('NFC')
}

/** Source Indic text -> IAST, with the per-script fixes the library needs. */
function toIast(text, script) {
  let src = text
  if (script === 'malayalam') src = src.replace(/[\u0D7A-\u0D7F]/g, (c) => MALAYALAM_CHILLU[c])
  let iast = Sanscript.t(src, INDIC_SCRIPTS[script].scheme, 'iast')
  if (script === 'tamil') {
    // Tamil has one letter per stop consonant; the library renders each as its
    // voiced/aspirated cousin (ப -> bha, க -> gha ...). Undo that.
    iast = iast.replace(/jh/g, 'c').replace(/gh/g, 'k').replace(/ḍh/g, 'ṭ').replace(/dh/g, 't').replace(/bh/g, 'p')
  }
  return iast
}

// ── any-ascii (lazy) ─────────────────────────────────────────────────────────
let anyAscii = null
let anyAsciiLoading = null
const readyListeners = new Set()
/** Called when a lazily-loaded engine finishes loading, so the UI can redraw. */
export const onEngineReady = (cb) => { readyListeners.add(cb); return () => readyListeners.delete(cb) }

function loadAnyAscii() {
  if (!anyAsciiLoading) {
    anyAsciiLoading = import('any-ascii').then((m) => {
      anyAscii = m.default || m.anyAscii
      readyListeners.forEach((cb) => cb())
    }).catch(() => { anyAsciiLoading = null })
  }
  return anyAsciiLoading
}

function romanizeOther(text, script) {
  const words = text.split(/(\s+)/)
  return words.map((w) => {
    if (!w || /^\s+$/.test(w)) return w
    const out = anyAscii(w)
    if (script === 'hangul') return out.toLowerCase()                       // join syllables: annyeonghaseyo
    if (script === 'han') return out.replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase()  // Ni Hao -> ni hao
    return out
  }).join('')
}

// ── cache ────────────────────────────────────────────────────────────────────
const STORAGE_KEY = 'dubstudio.translit.v1'
const MAX_ENTRIES = 3000
const cache = new Map()
let saveTimer = null
let hits = 0, misses = 0

try {
  if (typeof localStorage !== 'undefined') {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    for (const [k, v] of Object.entries(saved)) cache.set(k, v)
  }
} catch { /* a corrupt or unavailable cache just means starting empty */ }

function scheduleSave() {
  if (typeof localStorage === 'undefined') return
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => {
    try {
      while (cache.size > MAX_ENTRIES) cache.delete(cache.keys().next().value) // oldest first
      localStorage.setItem(STORAGE_KEY, JSON.stringify(Object.fromEntries(cache)))
    } catch { /* storage full or blocked — the in-memory cache still works */ }
  }, 400)
}

export const cacheStats = () => ({ size: cache.size, hits, misses })
export const clearTransliterationCache = () => { cache.clear(); try { localStorage.removeItem(STORAGE_KEY) } catch {} }

// ── public API ───────────────────────────────────────────────────────────────
/**
 * Convert `text` into the `target` script ('latin' or an INDIC_SCRIPTS key).
 * Returns the text unchanged when there is nothing to convert, the target isn't
 * available for this script, or a lazy engine is still loading (listen with
 * onEngineReady to redraw once it is).
 */
export function transliterate(text, target = 'latin') {
  if (!text) return text || ''
  const key = `${target}\u0000${text}`
  const cached = cache.get(key)
  if (cached !== undefined) { hits++; return cached }

  const script = detectScript(text)
  if (!script || script === 'latin') return text
  const options = targetOptionsFor(script)
  const t = options.some((o) => o.id === target) ? target : 'latin'
  if (t !== target) {                          // unsupported target for this script: fall back to Latin
    const fallback = transliterate(text, 'latin')
    return fallback
  }

  let out
  if (INDIC_SCRIPTS[script]) {
    out = target === 'latin'
      ? simplifyIast(toIast(text, script), script)
      : Sanscript.t(script === 'tamil' || script === 'malayalam' ? toIast(text, script) : text,
                    script === 'tamil' || script === 'malayalam' ? 'iast' : INDIC_SCRIPTS[script].scheme,
                    INDIC_SCRIPTS[target].scheme)
  } else {
    if (!anyAscii) { loadAnyAscii(); return text }   // not ready yet: show original, redraw when loaded
    out = romanizeOther(text, script)
  }
  misses++
  cache.set(key, out)
  scheduleSave()
  return out
}
