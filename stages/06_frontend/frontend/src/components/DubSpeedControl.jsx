import React from 'react'

const MIN = 0.25, MAX = 2.0, STEP = 0.05
const fmtX   = (v) => `${(+v).toFixed(2).replace(/0$/, '')}×`
const fmtSec = (ms) => `${(ms / 1000).toFixed(1)}s`

/**
 * Speed of one dub clip. By default the speed is chosen automatically so the
 * dub takes about as long as the original line; drag the slider to override it,
 * "Auto" puts it back. Speed changes use pitch-preserving time-stretch in the
 * export and pitch-preserving playback in the preview — no re-synthesis needed,
 * so changing it never makes a dub "stale". Extreme rates can still add artifacts.
 *
 * Controlled by the parent: `value` is what's shown/auditioned right now
 * (a draft while dragging), `onDraft` reports drags, `onCommit` saves
 * (a number, or null to go back to automatic).
 */
export default function DubSpeedControl({ value, isManual, autoSpeed, dubMs, originalMs, onDraft, onCommit }) {
  const fitted = dubMs > 0 ? dubMs / value : 0
  const diff   = originalMs > 0 ? fitted - originalMs : 0
  const fits   = originalMs > 0 && Math.abs(diff) <= originalMs * 0.06
  const oneAt  = ((1 - MIN) / (MAX - MIN)) * 100

  const commit = (e) => onCommit(parseFloat(e.currentTarget.value))

  return (
    <div style={{ padding: '8px 10px 9px', borderRadius: 8, background: 'rgba(255,255,255,0.025)', border: '1px solid #1c1c1c' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontSize: 9, color: '#3a3a3a', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Dub speed</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{
            fontSize: 8.5, fontWeight: 700, padding: '1px 6px', borderRadius: 8,
            background: isManual ? 'rgba(245,166,35,0.12)' : 'rgba(255,255,255,0.05)',
            color: isManual ? '#f5a623' : '#666', border: `1px solid ${isManual ? 'rgba(245,166,35,0.3)' : '#262626'}`,
          }}>{isManual ? 'Manual' : 'Auto'}</span>
          <span style={{ fontSize: 13, fontWeight: 700, color: '#e8e8e8', fontFamily: 'monospace', minWidth: 42, textAlign: 'right' }}>{fmtX(value)}</span>
        </div>
      </div>

      <div style={{ position: 'relative', padding: '2px 0 10px' }}>
        <input
          type="range" min={MIN} max={MAX} step={STEP} value={value}
          aria-label="Dub speed"
          onChange={(e) => onDraft(parseFloat(e.target.value))}
          onPointerUp={commit} onKeyUp={commit} onBlur={commit}
          style={{ width: '100%', accentColor: '#f5a623', cursor: 'pointer', margin: 0 }}
        />
        {/* tick at 1× so "unchanged" is easy to find */}
        <span style={{ position: 'absolute', left: `calc(${oneAt}% + ${(0.5 - oneAt / 100) * 14}px)`, bottom: 0, transform: 'translateX(-50%)', fontSize: 8, color: '#444', pointerEvents: 'none' }}>1×</span>
        <span style={{ position: 'absolute', left: 0, bottom: 0, fontSize: 8, color: '#333', pointerEvents: 'none' }}>{fmtX(MIN)}</span>
        <span style={{ position: 'absolute', right: 0, bottom: 0, fontSize: 8, color: '#333', pointerEvents: 'none' }}>{fmtX(MAX)}</span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <span style={{ fontSize: 9, lineHeight: 1.5, color: '#555' }}>
          dub {fmtSec(dubMs)} → <b style={{ color: fits ? '#4ade80' : '#fbbf24', fontWeight: 600 }}>{fmtSec(fitted)}</b> · original line {fmtSec(originalMs)}
          {!fits && originalMs > 0 && (
            <span style={{ color: '#7a6a3a' }}> ({diff > 0 ? '+' : ''}{(diff / 1000).toFixed(1)}s)</span>
          )}
        </span>
        {isManual && (
          <button type="button" onClick={() => { onDraft(null); onCommit(null) }}
            title={`Go back to the automatic speed (${fmtX(autoSpeed)})`}
            style={{ fontSize: 9, padding: '2px 8px', borderRadius: 6, cursor: 'pointer', flexShrink: 0,
              background: 'rgba(255,255,255,0.04)', border: '1px solid #262626', color: '#888' }}>
            ↺ Auto
          </button>
        )}
      </div>
    </div>
  )
}
