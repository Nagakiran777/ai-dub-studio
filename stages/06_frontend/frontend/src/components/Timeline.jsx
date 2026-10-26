import React, { useRef, useEffect, useState, useCallback } from 'react'
import { useAudio } from '../useAudio'
import { transliterate } from '../transliterate'
import { useStore } from '../store'
import ContextMenu from './ContextMenu'

const API        = '/api'
const LABEL_W    = 120
const ROW_H      = 44
const RULER_H    = 26
const MIN_DUR_MS = 1000
const MIN_BAR_PX = 4

export default function Timeline() {
  const {
    transcription, translations, ttsManifest,
    manifest, speakers, speakerColors,
    selectedDialogueId, selectDialogue,
    playheadMs, setPlayheadMs,
    timelineZoom, setZoom,
    currentJobId, updateDialogue, updateDialogueFields, addDialogue,
    translitOn, translitTarget, translitVersion, videoPlaying, dubbedView,
    mergeDialogues, deleteDialogue, splitDialogue,
  } = useStore()

  const scrollRef  = useRef(null)
  const audioRef   = useAudio()
  const dragRef    = useRef(null)
  const justDraggedRef = useRef(false) // swallow the click that follows a real drag
  const [ctxMenu, setCtxMenu]     = useState(null)  // { x, y, items }
  const [, forceUpdate]           = useState(0)

  const asr_done    = manifest?.stages?.['01_asr']?.status    === 'completed'
  const transl_done = manifest?.stages?.['03_translation']?.status === 'completed'
  const tts_done    = manifest?.stages?.['04_tts']?.status    === 'completed'

  const dialogues = transcription?.dialogues || []
  const translMap = {}
  ;(translations?.dialogues || []).forEach(d => { translMap[d.id] = d })
  const ttsMap = {}
  ;(ttsManifest?.dialogues || []).forEach(d => { ttsMap[d.id] = d })

  const speakerOrder = speakers.map(s => s.id)
  const spkGroups = {}
  speakerOrder.forEach(s => { spkGroups[s] = [] })
  dialogues.forEach(d => {
    const s = d.speaker_id || 'speaker_00'
    if (!spkGroups[s]) spkGroups[s] = []
    spkGroups[s].push(d)
  })

  const totalMs = dialogues.length > 0 ? Math.max(...dialogues.map(d => d.end_ms)) + 8000 : 60000
  const pxPerMs = timelineZoom / 1000
  const msToPx  = ms => ms * pxPerMs
  const pxToMs  = px => px / pxPerMs

  // Ruler ticks
  const tickMs = timelineZoom < 30 ? 30000 : timelineZoom < 60 ? 10000 : timelineZoom < 120 ? 5000 : 1000
  const ticks  = []
  for (let ms = 0; ms <= totalMs; ms += tickMs) ticks.push(ms)
  const fmtMs = ms => {
    const s = Math.floor(ms/1000), m = Math.floor(s/60), ss = s%60
    return m > 0 ? `${m}:${ss.toString().padStart(2,'0')}` : `${ss}s`
  }

  // Wheel zoom
  const onWheel = useCallback(e => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault()
      setZoom(timelineZoom * (e.deltaY < 0 ? 1.15 : 0.87))
    }
  }, [timelineZoom, setZoom])
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [onWheel])

  // Auto-scroll playhead
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const px = msToPx(playheadMs)
    const sl = el.scrollLeft, sw = el.clientWidth
    if (px < sl + 60 || px > sl + sw - 60) el.scrollLeft = px - sw * 0.4
  }, [playheadMs])

  // Ruler click to seek
  const onRulerClick = e => {
    const rect = scrollRef.current.getBoundingClientRect()
    const x    = e.clientX - rect.left + scrollRef.current.scrollLeft
    setPlayheadMs(Math.max(0, Math.round(pxToMs(x))))
  }

  // Play dialogue audio (exact slice)
  const playDialogue = (dlg) => {
    // While the video is playing the mix already supplies the audio at the new
    // position — a click-preview on top would just double it up.
    if (videoPlaying) return
    const au = audioRef.current
    au.pause()
    // A dub plays at its own speed with pitch preservation, matching Stage 05.
    // The original voice stays at normal speed. Set the rate every time because
    // this audio element is reused and loading a new src resets its playback rate.
    const isDub = dubbedView && tts_done && ttsMap[dlg.id]?.success   // Original view previews the original voice
    const rate  = isDub ? (ttsMap[dlg.id].speed ?? 1) : 1
    au.defaultPlaybackRate = rate
    au.src = isDub
      ? `${API}/jobs/${currentJobId}/tts_audio/${dlg.id}?t=${Date.now()}`   // cache-bust: a re-dub overwrites the same URL
      : `${API}/jobs/${currentJobId}/vocals_audio/${dlg.id}`
    au.playbackRate = rate
    au.preservesPitch = true; au.webkitPreservesPitch = true
    au.currentTime = 0
    au.play().catch(() => {})
  }

  // Add dialogue at position
  const addDialogueAt = async (startMs, speakerId) => {
    const endMs = startMs + 3000
    const id    = `dialogue_${Date.now()}`
    const newDlg = {
      id, speaker_id: speakerId || 'speaker_00',
      start_ms: startMs, end_ms: endMs,
      text: 'New dialogue',
    }
    await addDialogue(newDlg)
    selectDialogue(id)
  }

  // Drag to move/resize bar
  const getBarBounds = (dlg) => {
    if (dragRef.current?.dlg.id === dlg.id) {
      return {
        start_ms: dragRef.current.previewStart ?? dlg.start_ms,
        end_ms:   dragRef.current.previewEnd   ?? dlg.end_ms,
      }
    }
    return { start_ms: dlg.start_ms, end_ms: dlg.end_ms }
  }

  const startBarDrag = (e, dlg, mode) => {
    e.stopPropagation(); e.preventDefault()
    const startX = e.clientX, startY = e.clientY
    const origStart = dlg.start_ms, origEnd = dlg.end_ms
    const origRow = Math.max(0, speakerOrder.indexOf(dlg.speaker_id))
    const DRAG_THRESHOLD_PX = 4   // below this it's a click, not a drag
    const AXIS_RELEASE_PX   = 12  // sticky axis lock lets go past this
    let dragging = false, axis = null
    dragRef.current = { dlg, mode, origStart, origEnd, origRow, targetRow: origRow, dy: 0, dragging: false }

    const onMove = ev => {
      if (!dragRef.current) return
      const dx = ev.clientX - startX, dy = ev.clientY - startY
      if (!dragging) {
        if (Math.abs(dx) < DRAG_THRESHOLD_PX && Math.abs(dy) < DRAG_THRESHOLD_PX) return
        dragging = true
        // Decide which axis the user meant, so a mostly-vertical speaker
        // change doesn't also nudge the timing by a few ms of hand jitter.
        axis = (mode === 'move' && Math.abs(dy) > Math.abs(dx)) ? 'y' : 'x'
        dragRef.current.dragging = true
      }
      const effDx = (axis === 'y' && Math.abs(dx) < AXIS_RELEASE_PX) ? 0 : dx
      const effDy = (axis === 'x' && Math.abs(dy) < AXIS_RELEASE_PX) ? 0 : dy
      const dms = Math.round(pxToMs(effDx))
      if (mode === 'move') {
        const dur = origEnd - origStart
        dragRef.current.previewStart = Math.max(0, origStart + dms)
        dragRef.current.previewEnd   = Math.max(0, origStart + dms) + dur
        const maxRow = Math.max(0, speakerOrder.length - 1)
        const targetRow = Math.min(maxRow, Math.max(0, origRow + Math.round(effDy / ROW_H)))
        dragRef.current.targetRow = targetRow
        // keep the bar inside the rows area while it follows the pointer
        dragRef.current.dy = Math.min((maxRow - origRow) * ROW_H, Math.max(-origRow * ROW_H, effDy))
      } else if (mode === 'resize-right') {
        dragRef.current.previewEnd = Math.max(origEnd + dms, origStart + 200)
      } else if (mode === 'resize-left') {
        dragRef.current.previewStart = Math.min(origStart + dms, origEnd - 200)
      }
      forceUpdate(n => n + 1)
    }

    const onUp = async () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      if (!dragRef.current) return
      const { dlg: d, previewStart, previewEnd, targetRow } = dragRef.current
      dragRef.current = null
      forceUpdate(n => n + 1)
      if (!dragging) return
      justDraggedRef.current = true
      setTimeout(() => { justDraggedRef.current = false }, 60)
      selectDialogue(d.id)

      // One combined update (time and/or speaker) = one undo step, one save.
      const fields = {}
      if (previewStart !== undefined && previewStart !== d.start_ms) fields.start_ms = previewStart
      if (previewEnd   !== undefined && previewEnd   !== d.end_ms)   fields.end_ms   = previewEnd
      if (mode === 'move' && targetRow !== origRow && speakerOrder[targetRow]) {
        fields.speaker_id = speakerOrder[targetRow]
      }
      if (Object.keys(fields).length) await updateDialogueFields(d.id, fields)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  // Right-click on dialogue bar
  const onBarContextMenu = (e, dlg, rowIdx) => {
    e.preventDefault(); e.stopPropagation()
    const dlgs = [...dialogues].sort((a, b) => a.start_ms - b.start_ms)
    const idx  = dlgs.findIndex(d => d.id === dlg.id)
    const dur  = dlg.end_ms - dlg.start_ms
    setCtxMenu({
      x: e.clientX, y: e.clientY,
      items: [
        { label: 'Play Audio',     icon: '▶',  action: () => playDialogue(dlg) },
        { label: 'Select',         icon: '✦',  action: () => selectDialogue(dlg.id) },
        { divider: true },
        { label: 'Split at Playhead', icon: '⊘',
          action: () => splitDialogue(dlg.id, playheadMs),
          disabled: playheadMs <= dlg.start_ms || playheadMs >= dlg.end_ms,
        },
        { label: 'Split at Middle', icon: '⊘',
          action: () => splitDialogue(dlg.id, Math.round((dlg.start_ms + dlg.end_ms) / 2)),
        },
        { label: 'Merge with Next', icon: '⊕',
          action: () => idx < dlgs.length-1 && mergeDialogues(dlg.id, dlgs[idx+1].id),
          disabled: idx >= dlgs.length - 1,
        },
        { divider: true },
        { label: 'Delete',  icon: '🗑', danger: true,
          action: () => { deleteDialogue(dlg.id); selectDialogue(null) },
        },
      ],
    })
  }

  // Right-click on speaker label
  const onSpeakerContextMenu = (e, spkId) => {
    e.preventDefault()
    setCtxMenu({
      x: e.clientX, y: e.clientY,
      items: [
        { label: 'Rename Speaker',  icon: '✏️',
          action: () => {
            const newName = prompt(`Rename ${spkId}:`, spkId)
            if (newName && newName.trim() && newName.trim() !== spkId) {
              // Use store rename
              const { renameSpeaker, refreshAll } = useStore.getState()
              renameSpeaker(spkId, newName.trim()).then(() => refreshAll(currentJobId))
            }
          }
        },
        { label: 'Select All Lines', icon: '☑',
          action: () => {
            const dlgs = dialogues.filter(d => d.speaker_id === spkId)
            if (dlgs.length) selectDialogue(dlgs[0].id)
          }
        },
      ],
    })
  }

  // Right-click on empty timeline row (not on a bar)
  const onRowContextMenu = (e, spkId, rowMs) => {
    e.preventDefault()
    setCtxMenu({
      x: e.clientX, y: e.clientY,
      items: [
        { label: 'Add Dialogue Here', icon: '＋',
          action: () => addDialogueAt(rowMs, spkId),
        },
        { label: 'Seek Playhead Here', icon: '▷',
          action: () => setPlayheadMs(rowMs),
        },
      ],
    })
  }

  if (!currentJobId) return (
    <div style={{ height:'100%', display:'flex', alignItems:'center', justifyContent:'center', color:'#333', fontSize:13 }}>
      Select a job
    </div>
  )

  if (!asr_done) return (
    <div style={{ height:'100%', display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:8, color:'#333' }}>
      <span style={{ fontSize:28 }}>🎙</span>
      <span style={{ fontSize:12 }}>Timeline available after Speech Recognition</span>
    </div>
  )

  return (
    <div style={{ height:'100%', display:'flex', overflow:'hidden', background:'#080808', position:'relative', userSelect:'none' }}
      onContextMenu={e => e.preventDefault()}>

      {/* Context menu */}
      {ctxMenu && <ContextMenu x={ctxMenu.x} y={ctxMenu.y} items={ctxMenu.items} onClose={() => setCtxMenu(null)} />}

      {/* Speaker labels */}
      <div style={{ width: LABEL_W, flexShrink:0, background:'#0e0e0e', borderRight:'1px solid #1a1a1a', display:'flex', flexDirection:'column', zIndex:10 }}>
        <div style={{ height: RULER_H, borderBottom:'1px solid #1a1a1a' }} />
        {speakerOrder.map((spk, i) => {
          const color = speakerColors[spk] || '#888'
          return (
            <div key={spk}
              onContextMenu={e => onSpeakerContextMenu(e, spk)}
              style={{
                height: ROW_H, display:'flex', flexDirection:'column',
                justifyContent:'center', padding:'0 12px',
                borderBottom:'1px solid #141414', flexShrink:0, cursor:'context-menu',
              }}>
              <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                <div style={{ width:7, height:7, borderRadius:'50%', background:color, flexShrink:0, boxShadow:`0 0 5px ${color}66` }} />
                <span style={{ fontSize:11, fontWeight:600, color, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                  {spk.replace('speaker_','Speaker ').replace(/Speaker (\d+)/, (_,n) => `Speaker ${n.padStart(2,'0')}`)}
                </span>
              </div>
              <span style={{ fontSize:9, color:'#2a2a2a', paddingLeft:13, marginTop:1, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                {spk}
              </span>
            </div>
          )
        })}
      </div>

      {/* Scrollable */}
      <div ref={scrollRef} style={{ flex:1, overflowX:'auto', overflowY:'hidden', position:'relative' }}>
        <div style={{ width: Math.max(msToPx(totalMs), 800), position:'relative', minHeight:'100%' }}>

          {/* Ruler */}
          <div onClick={onRulerClick} style={{
            height: RULER_H, position:'sticky', top:0, zIndex:15,
            background:'linear-gradient(180deg, #161616 0%, #111111 100%)',
            borderBottom:'1px solid #1d1d1d', cursor:'crosshair',
          }}>
            {ticks.map(ms => (
              <div key={ms} style={{ position:'absolute', left: msToPx(ms), top:0, display:'flex', flexDirection:'column', alignItems:'flex-start' }}>
                <div style={{ width:1, height:7, background:'#282828', marginTop:5 }} />
                <span style={{ fontSize:9, color:'#3a3a3a', marginTop:2, marginLeft:3, fontFamily:'monospace' }}>{fmtMs(ms)}</span>
              </div>
            ))}
            {/* Playhead on ruler */}
            <div style={{ position:'absolute', top:0, left: msToPx(playheadMs), width:2, height:'100%', background:'rgba(245,166,35,0.6)', pointerEvents:'none' }}>
              <div style={{ position:'absolute', top:2, left:'50%', transform:'translateX(-50%)', width:8, height:8, borderRadius:2, background:'#f5a623' }} />
            </div>
          </div>

          {/* Row backgrounds */}
          {speakerOrder.map((spk, i) => {
            const dr = dragRef.current
            const dropTarget = dr?.dragging && dr.mode === 'move' && dr.targetRow === i && dr.targetRow !== dr.origRow
            const tint = speakerColors[spk] || '#888'
            return (
            <div key={spk} style={{
              position:'absolute', top: RULER_H + i * ROW_H, left:0, right:0, height: ROW_H,
              background: dropTarget ? `${tint}1f` : (i%2===0 ? 'rgba(255,255,255,0.01)' : 'transparent'),
              borderBottom:'1px solid #141414',
              boxShadow: dropTarget ? `inset 0 0 0 1px ${tint}66` : 'none',
              transition: 'background 0.12s, box-shadow 0.12s',
            }}
            onContextMenu={e => {
              const rect = scrollRef.current.getBoundingClientRect()
              const x    = e.clientX - rect.left + scrollRef.current.scrollLeft
              onRowContextMenu(e, spk, Math.round(pxToMs(x)))
            }}
            />
            )
          })}

          {/* Playhead */}
          <div style={{
            position:'absolute', top: RULER_H, bottom:0, left: msToPx(playheadMs),
            width:2, background:'linear-gradient(to bottom, #f5a623, rgba(245,166,35,0.2))',
            pointerEvents:'none', zIndex:20,
          }}>
            <div style={{ position:'absolute', top:0, left:'50%', transform:'translateX(-50%)', width:0, height:0, borderLeft:'5px solid transparent', borderRight:'5px solid transparent', borderTop:'7px solid #f5a623' }} />
          </div>

          {/* Dialogue bars */}
          {speakerOrder.map((spk, rowIdx) => {
            const color = speakerColors[spk] || '#888'
            return (spkGroups[spk] || []).map(dlg => {
              const { start_ms, end_ms } = getBarBounds(dlg)
              const x   = msToPx(start_ms)
              const w   = Math.max(MIN_BAR_PX, msToPx(end_ms - start_ms))
              const dr = dragRef.current
              const draggingThis = dr?.dlg.id === dlg.id && dr.dragging
              const top = RULER_H + rowIdx * ROW_H + (draggingThis && dr.mode === 'move' ? dr.dy : 0)
              const sel = dlg.id === selectedDialogueId
              const short = (dlg.end_ms - dlg.start_ms) < MIN_DUR_MS
              const hasTTS = tts_done && ttsMap[dlg.id]?.success
              const transl = translMap[dlg.id]

              return (
                <div key={dlg.id} className="dlg-bar"
                  style={{
                    position:'absolute', top: top+5, height: ROW_H-10, left: x, width: w,
                    borderRadius: 6,
                    background: sel
                      ? `linear-gradient(135deg, ${color}cc, ${color}88)`
                      : hasTTS ? `linear-gradient(135deg, ${color}55, ${color}33)`
                      : `linear-gradient(135deg, ${color}33, ${color}22)`,
                    border: sel ? `1.5px solid ${color}` : `1px solid ${color}44`,
                    overflow:'hidden', display:'flex', alignItems:'center',
                    transition: draggingThis ? 'none' : 'box-shadow 0.15s',
                    zIndex: draggingThis ? 30 : (sel ? 8 : 5),
                    opacity: draggingThis ? 0.92 : 1,
                    boxShadow: draggingThis ? `0 6px 20px rgba(0,0,0,0.55), 0 0 12px ${color}55` : (sel ? `0 0 12px ${color}44` : 'none'),
                    cursor: draggingThis ? 'grabbing' : 'pointer',
                  }}
                  onClick={() => {
                    if (justDraggedRef.current) return // that click was the end of a drag
                    selectDialogue(dlg.id); setPlayheadMs(dlg.start_ms); playDialogue(dlg)
                  }}
                  onContextMenu={e => onBarContextMenu(e, dlg, rowIdx)}
                  onMouseDown={e => { if (e.button===0 && !e.altKey) startBarDrag(e, dlg, 'move') }}
                >
                  {/* Left resize */}
                  <div style={{ position:'absolute', left:0, top:0, bottom:0, width:6, cursor:'ew-resize', zIndex:10 }}
                    onMouseDown={e => { e.stopPropagation(); startBarDrag(e, dlg, 'resize-left') }} />

                  {/* Text */}
                  {w > 32 && (
                    <span style={{ fontSize:9, color: sel ? '#fff' : 'rgba(255,255,255,0.65)', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', padding:'0 8px', pointerEvents:'none', flex:1 }}>
                      {transl?.translation || (translitOn ? transliterate(dlg.text, translitTarget) : dlg.text) || ''}
                    </span>
                  )}

                  {short && w > 14 && <span style={{ position:'absolute', top:1, right:2, fontSize:8, color:'#fbbf24', pointerEvents:'none' }}>⚠</span>}
                  {hasTTS && w > 20 && <span style={{ position:'absolute', bottom:1, right:2, fontSize:7, color:'#4ade80', pointerEvents:'none' }}>✓</span>}

                  {/* Right resize */}
                  <div style={{ position:'absolute', right:0, top:0, bottom:0, width:6, cursor:'ew-resize', zIndex:10 }}
                    onMouseDown={e => { e.stopPropagation(); startBarDrag(e, dlg, 'resize-right') }} />
                </div>
              )
            })
          })}
        </div>
      </div>

      {/* Zoom controls */}
      <div style={{
        position:'absolute', bottom:8, right:12, zIndex:30,
        display:'flex', alignItems:'center', gap:4,
        background:'rgba(14,14,14,0.95)', border:'1px solid #222',
        borderRadius:8, padding:'4px 8px', backdropFilter:'blur(8px)',
      }}>
        <button onClick={() => setZoom(timelineZoom * 0.75)} style={{ background:'none', border:'none', color:'#555', cursor:'pointer', fontSize:13, padding:'0 3px' }}>−</button>
        <span style={{ fontSize:9, color:'#444', fontFamily:'monospace', minWidth:42, textAlign:'center' }}>{Math.round(timelineZoom)}px/s</span>
        <button onClick={() => setZoom(timelineZoom * 1.33)} style={{ background:'none', border:'none', color:'#555', cursor:'pointer', fontSize:13, padding:'0 3px' }}>+</button>
      </div>
    </div>
  )
}