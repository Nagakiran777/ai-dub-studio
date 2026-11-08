import React, { useEffect } from 'react'
import { useStore } from '../store'

/**
 * Small one-time reminder shown when speaker detection finishes and some dialogues are
 * shorter than 1 second. Those can still be dubbed (their reference is repeated 3× so it
 * is long enough), but a good reference gives a better voice. OK or ✕ closes it.
 */
export default function ShortDialogueNotice() {
  const { shortWarning, dismissShortWarning } = useStore()
  useEffect(() => {
    if (!shortWarning) return
    const onKey = (e) => { if (e.key === 'Escape' || e.key === 'Enter') dismissShortWarning() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [shortWarning, dismissShortWarning])
  if (!shortWarning) return null
  const n = shortWarning.count
  return (
    <div role="dialog" aria-label="Short dialogues" onClick={dismissShortWarning}
      style={{ position: 'fixed', inset: 0, zIndex: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(3px)' }}>
      <div onClick={e => e.stopPropagation()}
        style={{ width: 380, maxWidth: '90vw', padding: '18px 20px', borderRadius: 12, background: '#121212', border: '1px solid rgba(251,191,36,0.35)', boxShadow: '0 20px 60px rgba(0,0,0,0.6)', position: 'relative' }}>
        <button onClick={dismissShortWarning} aria-label="Close"
          style={{ position: 'absolute', top: 8, right: 10, background: 'none', border: 'none', color: '#666', fontSize: 16, cursor: 'pointer' }}>✕</button>
        <div style={{ fontSize: 22, marginBottom: 6 }}>⚠️</div>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#fbbf24', marginBottom: 6 }}>
          {n} {n === 1 ? 'dialogue is' : 'dialogues are'} shorter than 1 second
        </div>
        <div style={{ fontSize: 11.5, lineHeight: 1.55, color: '#999', marginBottom: 14 }}>
          Very short lines don't give the voice model much to copy. They will still be dubbed, but for a
          better result give them a good reference audio (look for the ⚠ on those bars and pick or upload a reference in the editor).
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button onClick={dismissShortWarning} autoFocus
            style={{ padding: '6px 18px', borderRadius: 8, fontSize: 12, fontWeight: 700, cursor: 'pointer', background: 'rgba(245,166,35,0.15)', border: '1px solid rgba(245,166,35,0.4)', color: '#f5a623' }}>OK</button>
        </div>
      </div>
    </div>
  )
}
