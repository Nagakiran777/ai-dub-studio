import React from 'react'
import { useStore } from '../store'
import {
  detectScript, targetOptionsFor, scriptLabel, isApproximate, transliterate,
} from '../transliterate'

/**
 * Header for the "Original Text" field: the label, a switch that shows the
 * original dialogue in another script (English letters by default), and — when
 * on — a live reading box. The editable textarea itself stays untouched below,
 * so you can read a Telugu line as "chala rojulu ayindi kada" while still
 * editing the real Telugu (e.g. with Type-in-English). One switch covers every
 * dialogue; the converted text is cached, so flipping it repeatedly is free.
 */
export default function OriginalReader({ text }) {
  // translitVersion is read so we redraw when a lazily-loaded engine finishes
  const { translitOn, translitTarget, setTranslitOn, setTranslitTarget, translitVersion } = useStore()
  void translitVersion

  const script  = detectScript(text)
  const options = targetOptionsFor(script)
  const canConvert = options.length > 0
  const target  = options.some((o) => o.id === translitTarget) ? translitTarget : 'latin'
  const showBox = translitOn && canConvert && !!text?.trim()
  const out     = showBox ? transliterate(text, target) : ''
  const loading = showBox && out === text            // a lazy engine is still loading
  const approx  = showBox && target === 'latin' && isApproximate(script)

  const title = canConvert
    ? `Show this dialogue written in another script (${scriptLabel(script)} → ${options.find((o) => o.id === target)?.label})`
    : (script === 'latin' ? 'Already in Latin letters — nothing to convert' : 'Nothing to convert for this dialogue')

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '4px 8px', marginBottom: 5 }}>
        <span style={{ fontSize: 9, color: '#3a3a3a', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Original Text</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {translitOn && canConvert && (
            <select
              value={target}
              onChange={(e) => setTranslitTarget(e.target.value)}
              aria-label="Show original in"
              className="ds-input"
              style={{ width: 'auto', fontSize: 9.5, padding: '2px 6px', borderRadius: 6 }}
            >
              {options.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
            </select>
          )}
          <button
            type="button"
            role="switch"
            aria-checked={translitOn}
            disabled={!canConvert && !translitOn}
            title={title}
            onClick={() => setTranslitOn(!translitOn)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6, padding: '3px 8px 3px 6px', borderRadius: 20,
              cursor: (!canConvert && !translitOn) ? 'not-allowed' : 'pointer', fontSize: 9.5, fontWeight: 600,
              background: translitOn ? 'rgba(96,165,250,0.12)' : 'rgba(255,255,255,0.04)',
              border: `1px solid ${translitOn ? 'rgba(96,165,250,0.35)' : '#262626'}`,
              color: translitOn ? '#60a5fa' : '#666',
              opacity: (!canConvert && !translitOn) ? 0.5 : 1,
            }}
          >
            <span style={{
              width: 22, height: 12, borderRadius: 7, position: 'relative', flexShrink: 0,
              background: translitOn ? '#3b82f6' : '#333', transition: 'background 0.18s',
            }}>
              <span style={{
                position: 'absolute', top: 1, left: translitOn ? 11 : 1, width: 10, height: 10,
                borderRadius: '50%', background: '#fff', transition: 'left 0.18s cubic-bezier(0.22,1,0.36,1)',
              }} />
            </span>
            Transliterate
          </button>
        </div>
      </div>

      {showBox && (
        <div className="fade-up" style={{
          marginBottom: 6, padding: '8px 10px', borderRadius: 8,
          background: 'rgba(96,165,250,0.06)', border: '1px solid rgba(96,165,250,0.18)',
        }}>
          <div style={{
            fontSize: 12.5, lineHeight: 1.5, color: loading ? '#555' : '#cfe3ff',
            whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          }}>
            {loading ? 'Loading…' : out}
          </div>
          {approx && (
            <div style={{ fontSize: 8.5, color: '#7a8aa3', marginTop: 4 }}
                 title="Romanizing this script is only a rough guide — unwritten vowels or a need for a dictionary (e.g. Japanese kanji) limit accuracy">
              ≈ approximate for {scriptLabel(script)}
            </div>
          )}
        </div>
      )}
    </>
  )
}
