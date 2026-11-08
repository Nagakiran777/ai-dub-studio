import React from 'react'
import axios from 'axios'
import { useStore } from '../store'

const API = '/api'

// 05_assembly is intentionally not a visible row — it's folded into the
// single Export action once TTS is done. Labels only, no stage numbers.
const STAGES = [
  { id: '00_vocals',       label: 'Vocals Extraction',  icon: '🎵' },
  { id: '01_asr',          label: 'Speech Recognition', icon: '🎙' },
  { id: '01b_diarization', label: 'Speaker Detection',  icon: '👥' },
  { id: '03_translation',  label: 'Translation',        icon: '🌐' },
  { id: '04_tts',          label: 'Voice Synthesis',    icon: '🔊' },
]

function StageRow({ stage, status, isActive, onRunOnly, canRunOnly }) {
  return (
    <div className="stage-row" style={{
      display: 'flex', alignItems: 'center', gap: 10, padding: '6px 10px',
      borderRadius: 8,
      background: isActive ? 'rgba(245,166,35,0.08)' :
                  status === 'completed' ? 'rgba(34,197,94,0.04)' : 'transparent',
      border: isActive ? '1px solid rgba(245,166,35,0.2)' :
              status === 'completed' ? '1px solid rgba(34,197,94,0.1)' : '1px solid transparent',
      position: 'relative',
    }}>
      {isActive && (
        <div style={{
          position: 'absolute', left: 0, top: '20%', bottom: '20%',
          width: 2, borderRadius: 1,
          background: 'linear-gradient(to bottom, #f5a623, #ff6b35)',
        }} />
      )}
      <span style={{ fontSize: 13, opacity: status === 'pending' ? 0.25 : 1, paddingLeft: isActive ? 4 : 0 }}>
        {stage.icon}
      </span>
      <span style={{
        fontSize: 11.5, flex: 1,
        color: isActive ? '#f5a623' : status === 'completed' ? '#4ade80' :
               status === 'failed' ? '#f87171' : '#555',
        fontWeight: isActive ? 600 : 400,
      }}>{stage.label}</span>
      {isActive  && <div className="spinner" style={{ width: 11, height: 11 }} />}
      {!isActive && status === 'completed' && <span style={{ fontSize: 10, color: '#4ade80' }}>✓</span>}
      {!isActive && status === 'failed'    && <span style={{ fontSize: 10, color: '#f87171' }}>✗</span>}
      {!isActive && status === 'pending'   && <div style={{ width: 5, height: 5, borderRadius: '50%', background: '#222' }} />}
      {canRunOnly && !isActive && (
        <button onClick={onRunOnly} title={status === 'completed' ? `Re-run ${stage.label} (later stages will need re-running too)` : `Run ${stage.label}`}
          style={{
            width: 19, height: 19, borderRadius: 5, flexShrink: 0, marginLeft: 4,
            background: 'rgba(255,255,255,0.04)', border: '1px solid #262626',
            color: '#555', cursor: 'pointer', fontSize: 9,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'background 0.15s, color 0.15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.background='rgba(245,166,35,0.1)'; e.currentTarget.style.color='#f5a623' }}
          onMouseLeave={e => { e.currentTarget.style.background='rgba(255,255,255,0.04)'; e.currentTarget.style.color='#555' }}
        >{status === 'completed' ? '↻' : '▶'}</button>
      )}
    </div>
  )
}

/* ── One consistent status/action bar for every pipeline state ── */
function StatusBar({ manifest, onRunAll, onExport, onStop, onDownload, onOpenTranslation }) {
  const { pipelineRunning, pipelineStage, readyForExport, waitingOnTranslation, currentJobId } = useStore()
  const stages = manifest?.stages || {}
  const anyCompleted = STAGES.some(s => stages[s.id]?.status === 'completed')
  const assemblyDone = stages['05_assembly']?.status === 'completed'
  const stageLabel = (id) => STAGES.find(s => s.id === id)?.label || id

  let content
  if (!currentJobId) {
    content = { dot: '#444', text: 'Select or create a job to begin' }
  } else if (assemblyDone) {
    content = {
      dot: '#4ade80', text: '✓ Complete', hint: 'Download anytime, or re-export if you have made more changes since.',
      action: { label: '⬇ Download', onClick: onDownload, color: '#22c55e' },
      secondaryAction: { label: '↻ Re-export', onClick: onExport, color: '#60a5fa' },
    }
  } else if (pipelineRunning) {
    content = { dot: '#f5a623', text: `Running: ${stageLabel(pipelineStage)}`, spinner: true, action: { label: '■ Stop', onClick: onStop, color: '#ef4444' } }
  } else if (readyForExport) {
    content = { dot: '#4ade80', text: '✓ Voices ready', hint: 'Dialogues are still fully editable — export whenever you are ready.', action: { label: '⬇ Export & Download', onClick: onExport, color: '#22c55e' } }
  } else if (waitingOnTranslation) {
    content = { dot: '#60a5fa', text: 'Waiting on translation', hint: 'Type it yourself or import an AI response, then continue.', action: { label: '🌐 Open Translation', onClick: onOpenTranslation, color: '#60a5fa' } }
  } else if (anyCompleted) {
    content = { dot: '#888', text: 'Stopped partway', hint: 'Pick up where it left off.', action: { label: '▶ Continue', onClick: onRunAll, color: '#3b82f6' } }
  } else {
    content = { dot: '#444', text: 'Not started', action: { label: '▶ Run All', onClick: onRunAll, color: '#f5a623' } }
  }

  return (
    <div className="status-bar" style={{
      margin: '8px 10px', padding: '9px 12px', borderRadius: 11,
      display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '8px 10px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 9, flex: '1 1 76px', minWidth: 0 }}>
        {content.spinner
          ? <div className="spinner" style={{ width: 12, height: 12, flexShrink: 0 }} />
          : <div style={{ width: 8, height: 8, borderRadius: '50%', background: content.dot, flexShrink: 0, boxShadow: `0 0 6px ${content.dot}88` }} />}
        <span title={content.hint || content.text} style={{ fontSize: 11.5, color: '#b5b5b5', lineHeight: 1.35 }}>{content.text}</span>
      </div>
      {(content.secondaryAction || content.action) && (
        <div style={{ display: 'flex', gap: 6, marginLeft: 'auto', flexShrink: 0 }}>
          {content.secondaryAction && (
            <button onClick={content.secondaryAction.onClick}
              style={{
                padding: '6px 10px', borderRadius: 8, fontSize: 11, fontWeight: 600, cursor: 'pointer',
                background: 'transparent', border: `1px solid ${content.secondaryAction.color}44`,
                color: content.secondaryAction.color, whiteSpace: 'nowrap',
              }}>{content.secondaryAction.label}</button>
          )}
          {content.action && (
            <button onClick={content.action.onClick}
              style={{
                padding: '6px 11px', borderRadius: 8, fontSize: 11, fontWeight: 700, cursor: 'pointer',
                background: `${content.action.color}22`, border: `1px solid ${content.action.color}55`,
                color: content.action.color, whiteSpace: 'nowrap',
                boxShadow: `inset 0 1px 0 ${content.action.color}33`,
              }}>{content.action.label}</button>
          )}
        </div>
      )}
    </div>
  )
}

// GLOBAL speech-recognition model — used for ALL dialogues the next time this stage runs.
// (Re-transcribing a single dialogue has its own model choice in the dialogue editor.)
const ASR_CHOICES = [
  { id: 'medium',   label: 'Medium',   note: 'recommended' },
  { id: 'large-v3', label: 'Large-v3', note: 'if the transcription is poor' },
]
function AsrModelPicker({ disabled }) {
  const { asrModel, setAsrModel, currentJobId } = useStore()
  if (!currentJobId) return null
  return (
    <div style={{ margin: '2px 6px 6px 28px', padding: '6px 8px', borderRadius: 8, background: 'rgba(255,255,255,0.025)', border: '1px solid #1c1c1c' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontSize: 9, color: '#555', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Select ASR model · all dialogues</span>
      </div>
      <div role="radiogroup" aria-label="ASR model" style={{ display: 'flex', gap: 4 }}>
        {ASR_CHOICES.map(c => {
          const on = asrModel === c.id
          return (
            <button key={c.id} role="radio" aria-checked={on} disabled={disabled}
              title={`${c.label} — ${c.note}. Used for every dialogue the next time Speech Recognition runs (▶ / ↻).`}
              onClick={() => setAsrModel(c.id)}
              style={{
                flex: 1, padding: '4px 6px', borderRadius: 6, fontSize: 10, fontWeight: 600, cursor: disabled ? 'not-allowed' : 'pointer',
                background: on ? 'rgba(245,166,35,0.16)' : 'transparent',
                border: `1px solid ${on ? 'rgba(245,166,35,0.45)' : '#262626'}`,
                color: on ? '#f5a623' : '#666', opacity: disabled ? 0.5 : 1, lineHeight: 1.25,
              }}>
              {c.label}
              <div style={{ fontSize: 8, fontWeight: 500, color: on ? '#c98a1c' : '#444' }}>{c.note}</div>
            </button>
          )
        })}
      </div>
      <div style={{ fontSize: 8.5, color: '#3d3d3d', marginTop: 4, lineHeight: 1.4 }}>
        Applies the next time Speech Recognition runs — press ↻ on it to redo the transcription with this model.
      </div>
    </div>
  )
}

export default function PipelinePanel() {
  const {
    manifest, currentJobId,
    pipelineRunning, pipelineStage,
    startPipeline, runSingleStage, exportVideo,
    activeOverlay, toggleOverlay,
  } = useStore()

  const stages  = manifest?.stages || {}
  const getStatus = (id) => {
    if (pipelineStage === id && pipelineRunning) return 'running'
    return stages[id]?.status || 'pending'
  }

  const stopPipeline = async () => {
    await axios.post(`${API}/jobs/${currentJobId}/pipeline/stop`).catch(() => {})
  }

  const download = () => window.open(`${API}/jobs/${currentJobId}/download`, '_blank')

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: '#0e0e0e', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        padding: '9px 14px 8px', borderBottom: '1px solid #1a1a1a', flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'linear-gradient(180deg, #161616 0%, #111111 100%)',
      }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: '#555', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Pipeline
        </span>
        <button onClick={() => toggleOverlay('log')}
          style={{
            display: 'flex', alignItems: 'center', gap: 5, padding: '4px 10px', borderRadius: 7,
            background: activeOverlay === 'log' ? 'rgba(245,166,35,0.1)' : 'rgba(255,255,255,0.03)',
            border: `1px solid ${activeOverlay === 'log' ? 'rgba(245,166,35,0.25)' : '#222'}`,
            color: activeOverlay === 'log' ? '#f5a623' : '#666', cursor: 'pointer', fontSize: 10, fontWeight: 600,
            transition: 'background 0.15s, color 0.15s, border-color 0.15s',
          }}>
          {pipelineRunning && activeOverlay !== 'log' && <div className="pulse-accent" style={{ width: 5, height: 5 }} />}
          📜 {activeOverlay === 'log' ? 'Hide Log' : 'View Log'}
        </button>
      </div>

      {/* Unified status/action bar — same shape and position for every state */}
      <StatusBar
        manifest={manifest}
        onRunAll={startPipeline}
        onExport={exportVideo}
        onStop={stopPipeline}
        onDownload={download}
        onOpenTranslation={() => toggleOverlay('translation')}
      />

      {/* Stage rows only — nothing else. No run button on Translation: it no
          longer runs automatically (Google Translate was too unreliable),
          handled entirely from the Translation panel instead. */}
      <div style={{ padding: '5px 10px 10px', flex: 1, minHeight: 0, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {STAGES.map(s => (
          <React.Fragment key={s.id}>
            <StageRow stage={s} status={getStatus(s.id)}
              isActive={pipelineStage === s.id && pipelineRunning}
              canRunOnly={!!currentJobId && !pipelineRunning && s.id !== '03_translation'}
              onRunOnly={() => runSingleStage(s.id)}
            />
            {s.id === '01_asr' && <AsrModelPicker disabled={pipelineRunning && pipelineStage === '01_asr'} />}
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}
