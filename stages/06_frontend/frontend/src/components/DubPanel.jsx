import React, { useEffect, useRef, useState } from 'react'
import { useStore } from '../store'
import { useAudio } from '../useAudio'
import Waveform from './Waveform'
import DubSpeedControl from './DubSpeedControl'
import { playDubClip, stopDubClip } from '../dubPlayback'
import { writePart, splitPart, joinPart, previousPartOf, partOf, MIN_KEPT_MS } from '../dubParts'

const API = '/api'
const fmtS = (ms) => `${(ms / 1000).toFixed(2)}s`

/**
 * The Dubbed-view editor for ONE piece of a dub: see its waveform, trim it, cut it in two,
 * change its speed, remove it. None of this touches the original line or re-synthesizes
 * anything — it only decides which part of the finished clip is used, how fast it plays and
 * where it sits. A dub that has been cut shows one tab per piece.
 */
export default function DubPanel({ dialogue, entry, color }) {
  const { currentJobId, updateDialogueFields, updateDialogue, translations, ttsEpoch, ttsVersion,
          selectedPartId, selectDub, playheadMs, mix } = useStore()
  const au = useAudio()
  const trackRef = useRef(null)
  const ctx = { updateDialogueFields, translations }

  const parts = entry.parts || []
  const part  = partOf(entry, selectedPartId)
  const pid   = part?.part_id || 'p0'
  const many  = parts.length > 1
  const nat   = entry.natural_ms || 0

  const [trim, setTrim]             = useState(null)   // {tin,tout} while dragging / typing
  const [speedDraft, setSpeedDraft] = useState(null)
  const [playing, setPlaying]       = useState(false)
  const [volDraft, setVolDraft]     = useState(null)

  useEffect(() => { setTrim(null) }, [dialogue.id, pid, part?.trim_in_ms, part?.trim_out_ms, nat])
  useEffect(() => { setSpeedDraft(null) }, [dialogue.id, pid, part?.speed])
  useEffect(() => { setVolDraft(null) }, [dialogue.id, entry.dub_volume])
  useEffect(() => () => stopDubClip(au.current), [dialogue.id])  // eslint-disable-line react-hooks/exhaustive-deps

  if (!part) return null
  const volume = volDraft ?? entry.dub_volume ?? 1
  const commitVol = () => { if (volDraft != null) { updateDialogue(dialogue.id, 'dub_volume', Math.abs(volDraft - 1) < 0.005 ? null : volDraft) } }
  const tin   = trim ? trim.tin  : part.trim_in_ms
  const tout  = trim ? trim.tout : part.trim_out_ms
  const speed = speedDraft ?? part.speed ?? 1
  const kept  = Math.max(0, tout - tin)
  const finalMs = speed > 0 ? kept / speed : kept
  const gone  = !!part.deleted
  const trimmed = tin > 0.5 || tout < nat - 0.5

  const url  = `${API}/jobs/${currentJobId}/tts_audio/${dialogue.id}?v=${ttsEpoch}-${ttsVersion[dialogue.id] || 0}`
  const wkey = `d:${currentJobId}:${dialogue.id}:${ttsEpoch}:${ttsVersion[dialogue.id] || 0}:${nat}`

  const commitTrim = (t) => {
    const a = Math.round(Math.max(0, Math.min(t.tin, nat - MIN_KEPT_MS)))
    const b = Math.round(Math.max(a + MIN_KEPT_MS, Math.min(t.tout, nat)))
    writePart(ctx, dialogue.id, pid, {
      dub_trim_start_ms: a <= 0 ? null : a,
      dub_trim_end_ms:   b >= nat - 1 ? null : b,
    })
  }

  // drag a trim handle on the waveform
  const startHandle = (which) => (ev) => {
    ev.preventDefault(); ev.stopPropagation()
    const rect = trackRef.current.getBoundingClientRect()
    let cur = { tin, tout }
    const toMs = (clientX) => Math.max(0, Math.min(nat, ((clientX - rect.left) / rect.width) * nat))
    const move = (e2) => {
      const ms = toMs(e2.clientX)
      cur = which === 'in'
        ? { tin: Math.min(ms, cur.tout - MIN_KEPT_MS), tout: cur.tout }
        : { tin: cur.tin, tout: Math.max(ms, cur.tin + MIN_KEPT_MS) }
      setTrim(cur)
    }
    const up = () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      commitTrim(cur)
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  // ✂ cut this piece in two at a position inside the clip
  const cutAt = async (natMs) => {
    const res = splitPart(ctx, dialogue.id, entry, part, natMs)
    if (!res) return
    await updateDialogueFields(dialogue.id, res.fields)
    selectDub(dialogue.id, res.newPartId)
  }
  const cutAtPlayhead = () => cutAt(part.trim_in_ms + (playheadMs - part.dub_start_ms) * part.speed)
  const playheadInside = playheadMs > part.dub_start_ms + MIN_KEPT_MS / speed && playheadMs < part.dub_end_ms - MIN_KEPT_MS / speed
  const onWaveDouble = (ev) => {
    if (gone) return
    const rect = trackRef.current.getBoundingClientRect()
    cutAt(((ev.clientX - rect.left) / rect.width) * nat)
  }

  const prev = pid !== 'p0' ? previousPartOf(entry, part) : null

  const play = () => {
    if (playing) { stopDubClip(au.current); setPlaying(false); return }
    const el = au.current
    playDubClip(el, `${url}&t=${Date.now()}`, { speed, trimInMs: tin, trimOutMs: tout, naturalMs: nat, gain: (entry.dub_volume ?? 1) * (mix?.dub ?? 1) })
    setPlaying(true)
    el.onpause = () => setPlaying(false)
    el.onended = () => setPlaying(false)
  }

  const numInput = (label, val, onCommit) => (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 2, flex: 1 }}>
      <span style={{ fontSize: 8.5, color: '#3a3a3a', textTransform: 'uppercase', letterSpacing: '0.07em' }}>{label}</span>
      <input type="number" step="0.01" min="0" max={(nat / 1000).toFixed(2)} value={(val / 1000).toFixed(2)}
        onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) onCommit(Math.round(v * 1000)) }}
        onBlur={() => trim && commitTrim(trim)}
        onKeyDown={e => { if (e.key === 'Enter' && trim) commitTrim(trim) }}
        className="ds-input" style={{ fontSize: 10, fontFamily: 'monospace', padding: '4px 6px' }} />
    </label>
  )

  const btn = (danger) => ({
    padding: '4px 10px', borderRadius: 6, fontSize: 10, cursor: 'pointer', background: 'transparent',
    border: `1px solid ${danger ? 'rgba(239,68,68,0.25)' : '#262626'}`, color: danger ? '#f87171' : '#888',
  })

  // where the OTHER pieces of this clip start/end — thin cut lines on the waveform
  const cuts = parts.filter(p => p.part_id !== pid).flatMap(p => [p.trim_in_ms, p.trim_out_ms]).filter(v => v > 1 && v < nat - 1)

  return (
    <div style={{ padding: 10, borderRadius: 9, background: 'rgba(245,166,35,0.04)', border: '1px solid rgba(245,166,35,0.18)', display: 'flex', flexDirection: 'column', gap: 9, opacity: gone ? 0.75 : 1 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 10, fontWeight: 700, color: '#f5a623' }}>🔊 Dubbed clip</span>
        <span style={{ fontSize: 9, fontFamily: 'monospace', color: gone ? '#f87171' : '#888' }}>
          {gone ? 'removed — original voice plays' : <>length <b style={{ color: '#e8e8e8' }}>{fmtS(finalMs)}</b> · at {fmtS(part.dub_start_ms)}</>}
        </span>
      </div>

      {many && (
        <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
          {parts.map((p, i) => (
            <button key={p.part_id} onClick={() => selectDub(dialogue.id, p.part_id)}
              style={{ fontSize: 9, padding: '2px 9px', borderRadius: 12, cursor: 'pointer',
                background: p.part_id === pid ? 'rgba(245,166,35,0.18)' : 'transparent',
                border: `1px solid ${p.part_id === pid ? 'rgba(245,166,35,0.5)' : '#262626'}`,
                color: p.part_id === pid ? '#f5a623' : (p.deleted ? '#7a4a4a' : '#777'),
                textDecoration: p.deleted ? 'line-through' : 'none' }}>
              Part {i + 1}
            </button>
          ))}
        </div>
      )}

      {/* waveform + trim handles (double-click = cut here) */}
      <div ref={trackRef} onDoubleClick={onWaveDouble} title="Double-click to cut here"
        style={{ position: 'relative', height: 64, borderRadius: 7, background: '#0a0a0a', border: '1px solid #1c1c1c', overflow: 'hidden', touchAction: 'none' }}>
        <Waveform url={url} cacheKey={wkey} color={color || '#f5a623'} alpha={0.85} lazy={false} />
        <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${(tin / (nat || 1)) * 100}%`, background: 'rgba(0,0,0,0.7)', pointerEvents: 'none' }} />
        <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: `${(1 - tout / (nat || 1)) * 100}%`, background: 'rgba(0,0,0,0.7)', pointerEvents: 'none' }} />
        {cuts.map((c, i) => (
          <div key={i} style={{ position: 'absolute', top: 0, bottom: 0, left: `${(c / (nat || 1)) * 100}%`, width: 0, borderLeft: '1px dashed rgba(255,255,255,0.35)', pointerEvents: 'none' }} />
        ))}
        {!gone && ['in', 'out'].map(which => (
          <div key={which} role="slider" aria-label={which === 'in' ? 'Trim start' : 'Trim end'}
            onPointerDown={startHandle(which)} onDoubleClick={e => e.stopPropagation()}
            style={{ position: 'absolute', top: 0, bottom: 0, width: 12, marginLeft: -6, cursor: 'ew-resize', left: `${((which === 'in' ? tin : tout) / (nat || 1)) * 100}%`, zIndex: 3, display: 'flex', justifyContent: 'center' }}>
            <div style={{ width: 3, height: '100%', background: '#f5a623', boxShadow: '0 0 6px rgba(245,166,35,0.7)', borderRadius: 2 }} />
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
        {numInput('Play from (s)', tin, (v) => setTrim({ tin: Math.min(v, tout - MIN_KEPT_MS), tout }))}
        {numInput('Play until (s)', tout, (v) => setTrim({ tin, tout: Math.max(v, tin + MIN_KEPT_MS) }))}
        <div style={{ flex: 1, fontSize: 9, color: '#555', lineHeight: 1.5, paddingBottom: 5 }}>
          clip {fmtS(nat)}{trimmed && <> · plays <b style={{ color: '#ccc' }}>{fmtS(kept)}</b></>}
        </div>
      </div>

      {!gone && (
        <DubSpeedControl
          value={speed}
          isManual={speedDraft != null ? speedDraft !== 1 : Math.abs((part.speed || 1) - 1) > 0.004}
          fitSpeed={(() => {
            // same rule as stages/_shared/dub_speed.auto_speed, on the part being played right now
            const r = (entry.original_ms > 0 && kept > 0) ? kept / entry.original_ms : 1
            return Math.abs(r - 1) <= 0.05 ? 1 : Math.max(0.5, Math.min(1.5, Math.round(r * 100) / 100))
          })()}
          keptMs={kept}
          originalMs={entry.original_ms}
          onDraft={setSpeedDraft}
          onCommit={(v) => writePart(ctx, dialogue.id, pid, { dub_speed: v })}
        />
      )}

      {/* Volume of THIS dubbed dialogue (all its parts). 100 % = the dub's natural level; nothing is normalized. */}
      <div style={{ padding: '8px 10px 9px', borderRadius: 8, background: 'rgba(255,255,255,0.025)', border: '1px solid #1c1c1c' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 5 }}>
          <span style={{ fontSize: 9, color: '#3a3a3a', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Volume of this dialogue</span>
          <span style={{ fontSize: 13, fontWeight: 700, color: Math.abs(volume - 1) < 0.005 ? '#e8e8e8' : '#f5a623', fontFamily: 'monospace' }}>{Math.round(volume * 100)}%</span>
        </div>
        <input type="range" aria-label="Dialogue volume" min={0} max={3} step={0.05} value={volume}
          onChange={e => { setVolDraft(parseFloat(e.target.value)) }}
          onPointerUp={commitVol} onKeyUp={commitVol} onBlur={commitVol}
          style={{ width: '100%', accentColor: '#f5a623', cursor: 'pointer', margin: 0 }} />
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginTop: 5, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 9, color: '#555', lineHeight: 1.5 }}>100% = natural level · the export uses exactly this</span>
          <span style={{ display: 'flex', gap: 5 }}>
            {entry.volume_gain != null && Math.abs(entry.volume_gain - volume) > 0.02 && (
              <button type="button" onClick={() => { setVolDraft(null); updateDialogue(dialogue.id, 'dub_volume', entry.volume_gain) }} style={btn(false)}
                title={`Set the volume that makes this dub as loud as its original line (original ${(+entry.orig_rms).toFixed(3)} · dub ${(+entry.dub_rms).toFixed(3)})`}>≈ Match original (×{(+entry.volume_gain).toFixed(2)})</button>
            )}
            {Math.abs(volume - 1) > 0.005 && (
              <button type="button" onClick={() => { setVolDraft(null); updateDialogue(dialogue.id, 'dub_volume', null) }} style={btn(false)}>↺ 100%</button>
            )}
          </span>
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {!gone && (
          <button onClick={play} style={{ ...btn(false), color: '#4ade80', borderColor: 'rgba(34,197,94,0.3)', background: 'rgba(34,197,94,0.1)', fontWeight: 700 }}>
            {playing ? '⏹ Stop' : '▶ Play'}
          </button>
        )}
        {!gone && (
          <button onClick={cutAtPlayhead} disabled={!playheadInside} style={{ ...btn(false), opacity: playheadInside ? 1 : 0.4, cursor: playheadInside ? 'pointer' : 'not-allowed' }}
            title="Cut this dub in two where the playhead is (or double-click the waveform)">✂ Split at playhead</button>
        )}
        {!gone && trimmed && (
          <button onClick={() => writePart(ctx, dialogue.id, pid, { dub_trim_start_ms: null, dub_trim_end_ms: null })} style={btn(false)}>↺ Reset trim</button>
        )}
        {!gone && pid === 'p0' && part.moved && (
          <button onClick={() => writePart(ctx, dialogue.id, pid, { dub_start_ms: null })} style={btn(false)} title="Put the dub back at the original line's start">⇤ Original start</button>
        )}
        {!gone && pid !== 'p0' && (
          <button onClick={() => joinPart(ctx, dialogue.id, entry, part)} style={btn(false)}
            title={prev ? 'Merge this piece back into the previous one' : 'Remove this piece'}>⤺ {prev ? 'Join with previous' : 'Remove part'}</button>
        )}
        {gone
          ? <button onClick={() => writePart(ctx, dialogue.id, pid, { dub_deleted: null })} style={btn(false)}>↺ Restore</button>
          : <button onClick={() => writePart(ctx, dialogue.id, pid, { dub_deleted: true })} style={btn(true)} title="Remove this piece from the export — where nothing else is dubbed, the original voice plays">🗑 Delete {many ? 'part' : 'dub'}</button>}
      </div>
    </div>
  )
}
