import React, { useState, useRef, useEffect } from 'react'
import { TRANSLITERATION_ITC, IME_ONLY_LANGUAGES, SPEECH_RECOGNITION_LOCALE } from '../languages'

const BOUNDARY_KEYS = { ' ': ' ', 'Enter': '\n', '.': '.', ',': ',', '!': '!', '?': '?' }
const WORD_RE = /^[a-zA-Z']+$/

/**
 * Textarea that, for languages with a Google Input Tools transliteration
 * code, lets the user type phonetically in Latin letters and see the
 * native script appear live (word-committed on space/punctuation/blur) —
 * e.g. typing "chala rojulu iyyindhi" produces the Telugu script as you go.
 * Falls back to plain typing for unsupported languages or when toggled off.
 * Also offers mic dictation via the browser's SpeechRecognition API when
 * available. The field's value is always exactly what gets saved — there is
 * no hidden raw-vs-converted state the parent has to know about.
 */
export default function TransliterationInput({
  value, onChange, language, placeholder, rows = 3, className, style,
}) {
  const itc = TRANSLITERATION_ITC[language]
  const canTransliterate = !!itc
  const imeOnlyName = IME_ONLY_LANGUAGES[language]
  const [translitOn, setTranslitOn] = useState(canTransliterate)
  const [listening, setListening] = useState(false)
  const [convertErr, setConvertErr] = useState(false)
  const taRef = useRef(null)
  const recognitionRef = useRef(null)

  useEffect(() => { setTranslitOn(canTransliterate) }, [language, canTransliterate])
  useEffect(() => () => { try { recognitionRef.current?.stop() } catch {} }, [])

  const convertWordAt = async (text, boundaryIdx) => {
    let start = boundaryIdx
    while (start > 0 && /[a-zA-Z']/.test(text[start - 1])) start--
    const word = text.slice(start, boundaryIdx)
    if (!word || !WORD_RE.test(word)) return null
    try {
      const res = await fetch(
        `https://inputtools.google.com/request?text=${encodeURIComponent(word)}&itc=${itc}&num=1&cp=0&cs=1&ie=utf-8&oe=utf-8`
      )
      const data = await res.json()
      const candidate = data?.[1]?.[0]?.[1]?.[0]
      setConvertErr(false)
      if (!candidate) return null
      return { start, end: boundaryIdx, replacement: candidate }
    } catch {
      setConvertErr(true)
      return null
    }
  }

  const handleKeyDown = async (e) => {
    if (!canTransliterate || !translitOn) return
    const boundaryChar = BOUNDARY_KEYS[e.key]
    if (boundaryChar === undefined) return
    const ta = taRef.current
    if (!ta) return
    e.preventDefault()
    const cursor = ta.selectionStart
    const text = value || ''
    const result = await convertWordAt(text, cursor)
    let newText, newCursor
    if (result) {
      newText = text.slice(0, result.start) + result.replacement + boundaryChar + text.slice(result.end)
      newCursor = result.start + result.replacement.length + boundaryChar.length
    } else {
      newText = text.slice(0, cursor) + boundaryChar + text.slice(cursor)
      newCursor = cursor + boundaryChar.length
    }
    onChange(newText)
    requestAnimationFrame(() => {
      if (taRef.current) taRef.current.selectionStart = taRef.current.selectionEnd = newCursor
    })
  }

  const handleBlur = async () => {
    if (!canTransliterate || !translitOn) return
    const text = value || ''
    const result = await convertWordAt(text, text.length)
    if (result) onChange(text.slice(0, result.start) + result.replacement + text.slice(result.end))
  }

  const micSupported = typeof window !== 'undefined' && (window.SpeechRecognition || window.webkitSpeechRecognition)

  const toggleMic = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) return
    if (listening) {
      try { recognitionRef.current?.stop() } catch {}
      setListening(false)
      return
    }
    const rec = new SR()
    rec.lang = SPEECH_RECOGNITION_LOCALE[language] || (language ? `${language}-${language.toUpperCase()}` : 'en-US')
    rec.continuous = true
    rec.interimResults = true
    const base = value || ''
    let committed = base
    rec.onresult = (ev) => {
      let interim = ''
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const t = ev.results[i][0].transcript
        if (ev.results[i].isFinal) {
          committed += (committed && !/\s$/.test(committed) ? ' ' : '') + t.trim()
        } else {
          interim = t
        }
      }
      onChange(committed + (interim ? (committed && !/\s$/.test(committed) ? ' ' : '') + interim : ''))
    }
    rec.onend  = () => setListening(false)
    rec.onerror = () => setListening(false)
    recognitionRef.current = rec
    rec.start()
    setListening(true)
  }

  return (
    <div>
      <textarea
        ref={taRef}
        value={value || ''}
        onChange={e => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
        placeholder={placeholder}
        rows={rows}
        className={className}
        style={style}
      />
      {(canTransliterate || micSupported || imeOnlyName) && (
        <div style={{ display: 'flex', gap: 6, marginTop: 5, alignItems: 'center', flexWrap: 'wrap' }}>
          {imeOnlyName && (
            <span style={{ fontSize: 9, color: '#555' }} title={`${imeOnlyName} needs a full dictionary-based input method, not simple phonetic typing`}>
              Type-in-English isn't available for {imeOnlyName} yet
            </span>
          )}
          {canTransliterate && (
            <button type="button" onClick={() => setTranslitOn(v => !v)}
              title="Type phonetically in English letters, get native script"
              style={{
                fontSize: 9, padding: '3px 8px', borderRadius: 5, cursor: 'pointer', fontWeight: 600,
                background: translitOn ? 'rgba(245,166,35,0.15)' : 'rgba(255,255,255,0.04)',
                border: `1px solid ${translitOn ? 'rgba(245,166,35,0.35)' : '#2a2a2a'}`,
                color: translitOn ? '#f5a623' : '#555',
              }}>
              {translitOn ? '✓ Type-in-English ON' : 'Type-in-English OFF'}
            </button>
          )}
          {micSupported && (
            <button type="button" onClick={toggleMic}
              title="Dictate this field with your microphone"
              style={{
                fontSize: 9, padding: '3px 8px', borderRadius: 5, cursor: 'pointer', fontWeight: 600,
                display: 'flex', alignItems: 'center', gap: 4,
                background: listening ? 'rgba(239,68,68,0.15)' : 'rgba(255,255,255,0.04)',
                border: `1px solid ${listening ? 'rgba(239,68,68,0.4)' : '#2a2a2a'}`,
                color: listening ? '#f87171' : '#555',
              }}>
              {listening && <span className="pulse-accent" style={{ width: 6, height: 6 }} />}
              {listening ? '● Listening…' : '🎤 Speak'}
            </button>
          )}
          {convertErr && (
            <span style={{ fontSize: 8, color: '#555' }}>couldn't reach transliteration — typed as-is</span>
          )}
        </div>
      )}
    </div>
  )
}
