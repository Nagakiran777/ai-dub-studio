import React from 'react'
import { useStore } from '../store'

const CONFIGS = {
  confirm_translation: {
    icon:  '🌐',
    title: 'Translation Complete',
    color: '#3b82f6',
    done:  'All dialogues have been translated.',
    instruction: 'Review translations in the Dialogue Editor. Edit any line, then click Resume.',
  },
  confirm_tts: {
    icon:  '🔊',
    title: 'Voice Synthesis Complete',
    color: '#22c55e',
    done:  'All dialogues have been dubbed.',
    instruction: 'Click any dialogue to hear it. Edit translation and Re-dub if needed. When satisfied, click Resume.',
  },
}

export default function CheckpointModal({ checkpoint }) {
  const { resumePipeline, dismissCheckpointModal } = useStore()

  // Speaker checkpoint handled inline in PipelinePanel — not here
  if (checkpoint === 'confirm_speakers') return null

  const cfg = CONFIGS[checkpoint]
  if (!cfg) return null

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 400,
      background: 'rgba(0,0,0,0.7)',
      backdropFilter: 'blur(6px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 24,
    }} onClick={dismissCheckpointModal}>
      <div onClick={e => e.stopPropagation()} style={{
        background: 'linear-gradient(180deg, #1a1a1a 0%, #141414 100%)',
        border: `1px solid ${cfg.color}33`,
        borderRadius: 16,
        boxShadow: `0 24px 64px rgba(0,0,0,0.9), 0 0 30px ${cfg.color}0d`,
        width: '100%', maxWidth: 380,
        overflow: 'hidden',
        position: 'relative',
      }}>
        <button onClick={dismissCheckpointModal} aria-label="Close — review before resuming"
          style={{
            position: 'absolute', top: 10, right: 10, width: 24, height: 24, borderRadius: 7,
            background: 'rgba(255,255,255,0.05)', border: '1px solid #2a2a2a', color: '#666',
            cursor: 'pointer', fontSize: 13, display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>✕</button>
        {/* Header */}
        <div style={{
          padding: '18px 20px 14px',
          borderBottom: `1px solid ${cfg.color}22`,
          background: `linear-gradient(135deg, ${cfg.color}08 0%, transparent 100%)`,
          display: 'flex', alignItems: 'center', gap: 12,
        }}>
          <div style={{
            width: 38, height: 38, borderRadius: 10, flexShrink: 0,
            background: `${cfg.color}18`, border: `1px solid ${cfg.color}33`,
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18,
          }}>{cfg.icon}</div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h2 style={{ fontSize: 14, fontWeight: 700, color: '#f0f0f0' }}>{cfg.title}</h2>
              <span style={{
                fontSize: 9, padding: '2px 7px', borderRadius: 10, fontWeight: 700,
                background: `${cfg.color}20`, color: cfg.color, border: `1px solid ${cfg.color}33`,
              }}>✓ DONE</span>
            </div>
            <p style={{ fontSize: 11, color: '#555', marginTop: 3 }}>{cfg.done}</p>
          </div>
        </div>

        {/* Body */}
        <div style={{ padding: '14px 20px 18px' }}>
          <p style={{ fontSize: 11, color: '#666', lineHeight: 1.6, marginBottom: 14 }}>
            {cfg.instruction}
          </p>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={dismissCheckpointModal}
              style={{
                padding: '10px 16px', borderRadius: 10, fontSize: 12, fontWeight: 600, cursor: 'pointer',
                background: 'rgba(255,255,255,0.04)', border: '1px solid #2a2a2a', color: '#888',
              }}
            >
              Review First
            </button>
            <button
              onClick={resumePipeline}
              style={{
                flex: 1, padding: '10px', borderRadius: 10, fontSize: 12,
                fontWeight: 700, cursor: 'pointer', border: 'none',
                background: `linear-gradient(135deg, ${cfg.color}, ${cfg.color}cc)`,
                color: '#fff',
                boxShadow: `0 4px 16px ${cfg.color}44`,
                transition: 'all 0.2s',
              }}
              onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-1px)'}
              onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
            >
              ▶ Resume Pipeline
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}