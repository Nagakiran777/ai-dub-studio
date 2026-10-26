import React from 'react'

/**
 * Shared slide-over shell: covers the DialogueEditor area, sliding in from
 * the right with a soft shadow and a subtle scale-in for a more polished
 * feel than a flat translateX. Only one of these is ever "active" at a time
 * (driven by store.activeOverlay), so they all stack at the same position —
 * whichever is active is the one that's interactive and visible.
 */
export default function SlideOverPanel({ active, onClose, title, icon, accent = '#f5a623', children }) {
  return (
    <div style={{
      position: 'absolute', inset: 0, zIndex: 50,
      pointerEvents: active ? 'auto' : 'none',
      opacity: active ? 1 : 0,
      transform: `translateX(${active ? '0%' : '100%'}) scale(${active ? 1 : 0.98})`,
      // visibility flips AFTER the slide-out finishes (and immediately on open) so a
      // closed panel's buttons can't be reached with Tab while it's invisible.
      visibility: active ? 'visible' : 'hidden',
      transition: `transform 0.34s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.28s ease, visibility 0s linear ${active ? '0s' : '0.34s'}`,
      background: '#0a0a0a', borderLeft: `1px solid ${accent}22`,
      display: 'flex', flexDirection: 'column',
      boxShadow: active ? `-20px 0 50px rgba(0,0,0,0.55), -1px 0 0 ${accent}18` : 'none',
    }}>
      <div className="panel-head" style={{
        padding: '10px 14px', borderBottom: '1px solid #1a1a1a', flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: accent, display: 'flex', alignItems: 'center', gap: 7, letterSpacing: '0.02em' }}>
          <span style={{ fontSize: 13 }}>{icon}</span>{title}
        </span>
        <button onClick={onClose}
          style={{
            width: 25, height: 25, borderRadius: 7, background: 'rgba(255,255,255,0.04)',
            border: '1px solid #262626', color: '#666', cursor: 'pointer', fontSize: 12,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'background 0.15s, color 0.15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)'; e.currentTarget.style.color = '#aaa' }}
          onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; e.currentTarget.style.color = '#666' }}
        >✕</button>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
        {children}
      </div>
    </div>
  )
}
