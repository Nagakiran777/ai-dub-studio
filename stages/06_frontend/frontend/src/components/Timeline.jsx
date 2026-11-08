import React, { useRef, useEffect, useState, useCallback } from 'react'
import { useAudio } from '../useAudio'
import { transliterate } from '../transliterate'
import { useStore } from '../store'
import ContextMenu from './ContextMenu'
import Waveform from './Waveform'
import { setElementGain } from '../dubPlayback'
import { writePart, splitPart, joinPart, previousPartOf } from '../dubParts'

const API        = '/api'
const LABEL_W    = 120
const ROW_H      = 44
const RULER_H    = 26
const MIN_DUR_MS = 1000
const MIN_BAR_PX = 4
const MIN_KEPT_MS = 100    // a trimmed dub never gets shorter than this (same as stages/_shared/dub_speed.py)

export default function Timeline() {
  const {
    transcription, translations, ttsManifest,
    manifest, speakers, speakerColors,
    selectedDialogueId, selectDialogue, selectedPartId, selectDub,
    playheadMs, setPlayheadMs,
    timelineZoom, setZoom,
    currentJobId, updateDialogue, updateDialogueFields, addDialogue,
    translitOn, translitTarget, translitVersion, videoPlaying,
    mergeDialogues, deleteDialogue, splitDialogue,
    timelineMode, setTimelineMode, ttsEpoch, ttsVersion, mix,
  } = useStore()

  const scrollRef  = useRef(null)
  const labelsRef  = useRef(null)      // speaker-name column; follows the rows' vertical scroll
  const [sbH, setSbH] = useState(0)    // height of the horizontal scrollbar (keeps both columns the same length)
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

  // Two independent views of the same dialogues:
  //   original — the recognised line (where the original voice is, what gets silenced)
  //   dubbed   — the synthesized clip (where it is placed, trimmed, how fast)
  // Changing one never changes the other. The dubbed view needs finished TTS.
  const dubMode = tts_done && timelineMode === 'dubbed'
  const playTimerRef = useRef(null)
  const dubEntryOf = (dlg) => {
    const e = ttsMap[dlg.id]
    return (e && e.success && e.dub_len_ms != null) ? e : null
  }

  const speakerOrder = speakers.map(s => s.id)
  const spkGroups = {}
  speakerOrder.forEach(s => { spkGroups[s] = [] })
  dialogues.forEach(d => {
    const s = d.speaker_id || 'speaker_00'
    if (!spkGroups[s]) spkGroups[s] = []
    spkGroups[s].push(d)
  })

  const lastDubEnd = Math.max(0, ...(ttsManifest?.dialogues || []).filter(e => e.success).flatMap(e => (e.parts || []).map(p => p.dub_end_ms)))
  const totalMs = dialogues.length > 0 ? Math.max(Math.max(...dialogues.map(d => d.end_ms)), lastDubEnd) + 8000 : 60000
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

  // Vertical scrolling: when there are more speakers than fit, swipe / wheel / drag the
  // scrollbar up and down — the speaker names stay lined up with their rows. Works the
  // same in the Original and Dubbed views (they share these rows).
  const onRowsScroll = () => {
    if (labelsRef.current && scrollRef.current) labelsRef.current.scrollTop = scrollRef.current.scrollTop
  }
  const onLabelsWheel = (e) => {            // wheel over the names scrolls the rows too
    if (scrollRef.current && !e.ctrlKey && !e.metaKey) scrollRef.current.scrollTop += e.deltaY
  }
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const measure = () => setSbH(Math.max(0, el.offsetHeight - el.clientHeight))
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(measure); ro.observe(el)
    return () => ro.disconnect()
  })
  // Bring the selected dialogue's speaker row into view (e.g. after picking it elsewhere)
  useEffect(() => {
    const el = scrollRef.current
    if (!el || !selectedDialogueId) return
    const d = dialogues.find(x => x.id === selectedDialogueId)
    const row = d ? speakerOrder.indexOf(d.speaker_id) : -1
    if (row < 0) return
    const top = row * ROW_H, bottom = top + ROW_H
    const viewTop = el.scrollTop, viewBottom = el.scrollTop + el.clientHeight - RULER_H
    if (top < viewTop) el.scrollTop = top
    else if (bottom > viewBottom) el.scrollTop = bottom - (el.clientHeight - RULER_H)
  }, [selectedDialogueId])   // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-scroll playhead
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const px = msToPx(playheadMs)
    const sl = el.scrollLeft, sw = el.clientWidth
    if (px < sl + 60 || px > sl + sw - 60) el.scrollLeft = px - sw * 0.4
  }, [playheadMs])

  // While the video plays, follow along: the dialogue under the playhead (in the CURRENT
  // view — original bars or dub bars) gets selected and scrolled to the middle of the
  // timeline, vertically to its speaker row too, so you always see what is being said.
  const followRef = useRef(null)
  useEffect(() => {
    if (!videoPlaying) { followRef.current = null; return }
    let active = null, activeStart = -1, aS = 0, aLen = 0, aPart = 'p0'
    for (const d of dialogues) {
      if (dubMode) {
        const en = dubEntryOf(d)
        if (!en) continue
        for (const p of (en.parts || [])) {
          if (p.deleted) continue
          if (playheadMs >= p.dub_start_ms && playheadMs < p.dub_start_ms + p.dub_len_ms && p.dub_start_ms > activeStart) {
            active = d; activeStart = p.dub_start_ms; aS = p.dub_start_ms; aLen = p.dub_len_ms; aPart = p.part_id
          }
        }
      } else {
        const s0 = d.start_ms, len = d.end_ms - d.start_ms
        if (playheadMs >= s0 && playheadMs < s0 + len && s0 > activeStart) { active = d; activeStart = s0; aS = s0; aLen = len }
      }
    }
    const key = active ? `${dubMode ? 'd' : 'o'}:${active.id}:${aPart}` : null
    if (key === followRef.current) return
    followRef.current = key
    if (!active) return
    if (dubMode) selectDub(active.id, aPart); else selectDialogue(active.id)
    const el = scrollRef.current
    if (!el) return
    const barW = msToPx(aLen)
    const left = barW > el.clientWidth * 0.8 ? msToPx(aS) - 40 : msToPx(aS) + barW / 2 - el.clientWidth / 2
    const row  = Math.max(0, speakerOrder.indexOf(active.speaker_id))
    const top  = row * ROW_H + ROW_H / 2 - (el.clientHeight - RULER_H) / 2
    el.scrollTo({ left: Math.max(0, left), top: Math.max(0, top) })
  }, [playheadMs, videoPlaying, dubMode])   // eslint-disable-line react-hooks/exhaustive-deps

  // Ruler click to seek
  const onRulerClick = e => {
    const rect = scrollRef.current.getBoundingClientRect()
    const x    = e.clientX - rect.left + scrollRef.current.scrollLeft
    setPlayheadMs(Math.max(0, Math.round(pxToMs(x))))
  }

  // Play dialogue audio (exact slice)
  const playDialogue = (dlg, part = null) => {
    // While the video is playing the mix already supplies the audio at the new
    // position — a click-preview on top would just double it up.
    if (videoPlaying) return
    const au = audioRef.current
    au.pause()
    clearTimeout(playTimerRef.current)
    // The Original view previews the original voice; the Dubbed view previews the
    // dub — only its KEPT (trimmed) part, at its own speed, like the export.
    // Set the rate every time because this audio element is reused and loading a
    // new src resets its playback rate.
    const e0    = dubMode ? dubEntryOf(dlg) : null
    const pt    = e0 ? (part || e0.parts?.[0]) : null
    const e     = pt                                   // the piece being previewed (kept range, speed, length)
    const isDub = !!e0 && !!pt && !pt.deleted
    const rate  = isDub ? (pt.speed ?? 1) : 1
    au.defaultPlaybackRate = rate
    au.src = isDub
      ? `${API}/jobs/${currentJobId}/tts_audio/${dlg.id}?t=${Date.now()}`   // cache-bust: a re-dub overwrites the same URL
      : `${API}/jobs/${currentJobId}/vocals_audio/${dlg.id}`
    au.playbackRate = rate
    au.preservesPitch = true; au.webkitPreservesPitch = true
    setElementGain(au, isDub ? (e0.dub_volume ?? 1) * (mix?.dub ?? 1) : 1)      // this dialogue's volume × Dubbed layer — same as the export
    const startAt = isDub ? (e.trim_in_ms || 0) / 1000 : 0
    const seekAndPlay = () => {
      try { au.currentTime = startAt } catch {}
      au.play().catch(() => {})
      if (isDub) playTimerRef.current = setTimeout(() => au.pause(), (e.dub_len_ms || 0) + 40)
    }
    if (startAt > 0) au.addEventListener('loadedmetadata', seekAndPlay, { once: true })
    else seekAndPlay()
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
    if (dragRef.current?.kind !== 'dub' && dragRef.current?.dlg.id === dlg.id) {
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

  // Where a DUB piece is drawn (its own geometry, never the original bar's)
  const getDubBounds = (dlg, part) => {
    const dr = dragRef.current
    if (dr?.kind === 'dub' && dr.dlg.id === dlg.id && dr.partId === part.part_id && dr.previewStart != null) {
      return { start: dr.previewStart, len: dr.previewLen, tin: dr.previewTin, tout: dr.previewTout }
    }
    return { start: part.dub_start_ms, len: part.dub_len_ms, tin: part.trim_in_ms, tout: part.trim_out_ms }
  }

  // Drag a dub piece: body = move it, edges = trim the clip's start / end.
  // Edge-dragging never re-synthesizes anything — it only changes which part of the
  // already-rendered clip this piece plays (and, for the left edge, where it starts).
  const startDubDrag = (ev, dlg, entry, part, mode) => {
    ev.stopPropagation(); ev.preventDefault()
    const startX = ev.clientX
    const speed = part.speed || 1
    const o = { start: part.dub_start_ms, tin: part.trim_in_ms, tout: part.trim_out_ms, natural: entry.natural_ms }
    const DRAG_THRESHOLD_PX = 4
    let dragging = false
    dragRef.current = { kind: 'dub', dlg, partId: part.part_id, mode, dragging: false }

    const onMove = e2 => {
      if (!dragRef.current) return
      const dx = e2.clientX - startX
      if (!dragging) {
        if (Math.abs(dx) < DRAG_THRESHOLD_PX) return
        dragging = true; dragRef.current.dragging = true
      }
      const dms = Math.round(pxToMs(dx))
      const dr = dragRef.current
      if (mode === 'move') {
        dr.previewStart = Math.max(0, o.start + dms)
        dr.previewTin = o.tin; dr.previewTout = o.tout
        dr.previewLen = (o.tout - o.tin) / speed
      } else if (mode === 'resize-right') {
        const wantLen = Math.max(MIN_KEPT_MS / speed, (o.tout - o.tin) / speed + dms)
        const tout = Math.min(o.natural, Math.max(o.tin + MIN_KEPT_MS, o.tin + wantLen * speed))
        dr.previewStart = o.start; dr.previewTin = o.tin; dr.previewTout = tout
        dr.previewLen = (tout - o.tin) / speed
      } else if (mode === 'resize-left') {
        const tin = Math.min(o.tout - MIN_KEPT_MS, Math.max(0, o.tin + dms * speed))
        const shift = (tin - o.tin) / speed
        dr.previewStart = Math.max(0, o.start + shift)
        dr.previewTin = tin; dr.previewTout = o.tout
        dr.previewLen = (o.tout - tin) / speed
      }
      forceUpdate(n => n + 1)
    }

    const onUp = async () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      const dr = dragRef.current
      dragRef.current = null
      forceUpdate(n => n + 1)
      if (!dr || !dragging) return
      justDraggedRef.current = true
      setTimeout(() => { justDraggedRef.current = false }, 60)
      selectDub(dlg.id, part.part_id)
      const r = v => Math.round(v)
      const fields = {}
      if (Math.abs(dr.previewStart - o.start) >= 1) fields.dub_start_ms = r(dr.previewStart)
      if (Math.abs(dr.previewTin  - o.tin)   >= 1) fields.dub_trim_start_ms = r(dr.previewTin)
      if (Math.abs(dr.previewTout - o.tout)  >= 1) fields.dub_trim_end_ms   = r(dr.previewTout)
      if (fields.dub_trim_end_ms != null && fields.dub_trim_end_ms >= o.natural - 1) fields.dub_trim_end_ms = null
      if (fields.dub_trim_start_ms != null && fields.dub_trim_start_ms <= 0) fields.dub_trim_start_ms = null
      if (Object.keys(fields).length) await writePart({ updateDialogueFields, translations }, dlg.id, part.part_id, fields)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  // Cut a dub piece in two at a timeline position (ms)
  const splitDubAt = async (dlg, entry, part, timelineMs) => {
    const cut = part.trim_in_ms + (timelineMs - part.dub_start_ms) * part.speed
    const res = splitPart({ translations }, dlg.id, entry, part, cut)
    if (!res) return
    await updateDialogueFields(dlg.id, res.fields)
    selectDub(dlg.id, res.newPartId)
  }
  const clientXToMs = (clientX) => {
    const el = scrollRef.current
    return pxToMs(clientX - el.getBoundingClientRect().left + el.scrollLeft)
  }

  // Right-click on a dub piece (Dubbed view)
  const onDubContextMenu = (ev, dlg, e, part) => {
    ev.preventDefault(); ev.stopPropagation()
    const w = { updateDialogueFields, translations }
    const fit = part.speed_fit
    const here = clientXToMs(ev.clientX)
    const canSplitHere = here > part.dub_start_ms && here < part.dub_end_ms
    const canSplitHead = playheadMs > part.dub_start_ms && playheadMs < part.dub_end_ms
    const isFirst = part.part_id === 'p0'
    const prev = isFirst ? null : previousPartOf(e, part)
    setCtxMenu({
      x: ev.clientX, y: ev.clientY,
      items: part.deleted ? [
        { label: 'Restore this part', icon: '↺', action: () => writePart(w, dlg.id, part.part_id, { dub_deleted: null }) },
        { label: 'Select',            icon: '✦', action: () => selectDub(dlg.id, part.part_id) },
      ] : [
        { label: 'Play', icon: '▶', action: () => playDialogue(dlg, part) },
        { label: 'Select', icon: '✦', action: () => selectDub(dlg.id, part.part_id) },
        { divider: true },
        { label: 'Split here', icon: '✂', disabled: !canSplitHere, action: () => splitDubAt(dlg, e, part, here) },
        { label: 'Split at playhead', icon: '✂', disabled: !canSplitHead, action: () => splitDubAt(dlg, e, part, playheadMs) },
        !isFirst && { label: prev ? 'Join with previous part' : 'Remove this part', icon: '⤺', action: () => joinPart(w, dlg.id, e, part) },
        { divider: true },
        { label: 'Reset trim', icon: '✂', disabled: !part.trimmed,
          action: () => writePart(w, dlg.id, part.part_id, { dub_trim_start_ms: null, dub_trim_end_ms: null }) },
        { label: 'Move back to original start', icon: '⇤', disabled: !(isFirst && part.moved),
          action: () => writePart(w, dlg.id, part.part_id, { dub_start_ms: null }) },
        { label: `Fit to original length (${fit}×)`, icon: '⇔', disabled: Math.abs(fit - (part.speed || 1)) < 0.005,
          action: () => writePart(w, dlg.id, part.part_id, { dub_speed: fit === 1 ? null : fit }) },
        { label: 'Normal speed (1×)', icon: '↺', disabled: Math.abs((part.speed || 1) - 1) < 0.005,
          action: () => writePart(w, dlg.id, part.part_id, { dub_speed: null }) },
        { divider: true },
        { divider: true },
        { label: `Volume +10%  (now ${Math.round((e.dub_volume ?? 1) * 100)}%)`, icon: '🔊',
          action: () => updateDialogueFields(dlg.id, { dub_volume: Math.min(3, Math.round(((e.dub_volume ?? 1) + 0.1) * 100) / 100) }) },
        { label: 'Volume −10%', icon: '🔉',
          action: () => updateDialogueFields(dlg.id, { dub_volume: Math.max(0, Math.round(((e.dub_volume ?? 1) - 0.1) * 100) / 100) }) },
        { label: 'Volume 100%', icon: '↺', disabled: Math.abs((e.dub_volume ?? 1) - 1) < 0.005,
          action: () => updateDialogueFields(dlg.id, { dub_volume: null }) },
        { divider: true },
        { label: 'Delete this part', icon: '🗑', danger: true,
          action: () => writePart(w, dlg.id, part.part_id, { dub_deleted: true }) },
      ].filter(Boolean),
    })
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
        !dubMode && { label: 'Add Dialogue Here', icon: '＋',
          action: () => addDialogueAt(rowMs, spkId),
        },
        { label: 'Seek Playhead Here', icon: '▷',
          action: () => setPlayheadMs(rowMs),
        },
      ].filter(Boolean),
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
        <div style={{ height: RULER_H, borderBottom:'1px solid #1a1a1a', display:'flex', alignItems:'center', justifyContent:'center', padding:'0 6px' }}>
          <div role="tablist" aria-label="Timeline view"
            style={{ display:'flex', width:'100%', padding:2, borderRadius:7, background:'rgba(255,255,255,0.04)', border:'1px solid #1f1f1f' }}>
            {[['original','Original'],['dubbed','Dubbed']].map(([m, label]) => {
              const active = (m === 'dubbed') === dubMode
              const locked = m === 'dubbed' && !tts_done
              return (
                <button key={m} role="tab" aria-selected={active} disabled={locked}
                  title={locked ? 'Available after Voice Synthesis' : (m === 'dubbed' ? 'Dubbed clips: move, trim, speed, delete' : 'Original lines: recognition and silencing')}
                  onClick={() => setTimelineMode(m)}
                  style={{
                    flex:1, padding:'2px 0', fontSize:9, fontWeight:700, borderRadius:5, border:'none',
                    cursor: locked ? 'not-allowed' : 'pointer', transition:'all 0.15s',
                    background: active ? 'rgba(245,166,35,0.18)' : 'transparent',
                    color: active ? '#f5a623' : (locked ? '#2a2a2a' : '#555'),
                    boxShadow: active ? 'inset 0 0 0 1px rgba(245,166,35,0.35)' : 'none',
                  }}>{label}</button>
              )
            })}
          </div>
        </div>
        <div ref={labelsRef} onWheel={onLabelsWheel} style={{ flex:1, minHeight:0, overflow:'hidden' }}>
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
        <div style={{ height: sbH, flexShrink:0 }} />
        </div>
      </div>

      {/* Scrollable (both directions) */}
      <div ref={scrollRef} onScroll={onRowsScroll} style={{ flex:1, overflowX:'auto', overflowY:'auto', position:'relative', overscrollBehaviorY:'contain' }}>
        <div style={{ width: Math.max(msToPx(totalMs), 800), height: RULER_H + speakerOrder.length * ROW_H, position:'relative', minHeight:'100%' }}>

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

          {/* Dialogue bars — ORIGINAL view: the recognised lines (what gets silenced) */}
          {!dubMode && speakerOrder.map((spk, rowIdx) => {
            const color = speakerColors[spk] || '#888'
            return (spkGroups[spk] || []).map(dlg => {
              const { start_ms, end_ms } = getBarBounds(dlg)
              const x   = msToPx(start_ms)
              const w   = Math.max(MIN_BAR_PX, msToPx(end_ms - start_ms))
              const dr = dragRef.current
              const draggingThis = dr?.kind !== 'dub' && dr?.dlg.id === dlg.id && dr.dragging
              const top = RULER_H + rowIdx * ROW_H + (draggingThis && dr.mode === 'move' ? dr.dy : 0)
              const sel = dlg.id === selectedDialogueId
              const short = (dlg.end_ms - dlg.start_ms) < MIN_DUR_MS
              const hasTTS = tts_done && ttsMap[dlg.id]?.success

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
                    // selecting only — the audio plays from the ▶ buttons in the dialogue editor
                    selectDialogue(dlg.id); setPlayheadMs(dlg.start_ms)
                  }}
                  onContextMenu={e => onBarContextMenu(e, dlg, rowIdx)}
                  onMouseDown={e => { if (e.button===0 && !e.altKey) startBarDrag(e, dlg, 'move') }}
                >
                  {/* Waveform of the ORIGINAL voice (separated vocals) for this line */}
                  <Waveform
                    url={`${API}/jobs/${currentJobId}/vocals_audio/${dlg.id}?s=${dlg.start_ms}-${dlg.end_ms}`}
                    cacheKey={`o:${currentJobId}:${dlg.id}:${dlg.start_ms}:${dlg.end_ms}`}
                    color="#ffffff" alpha={sel ? 0.7 : 0.45} />

                  {/* Left resize */}
                  <div style={{ position:'absolute', left:0, top:0, bottom:0, width:6, cursor:'ew-resize', zIndex:10 }}
                    onMouseDown={e => { e.stopPropagation(); startBarDrag(e, dlg, 'resize-left') }} />

                  {/* Transcription (what was recognised) */}
                  {w > 32 && (
                    <span style={{ position:'relative', fontSize:9, color: sel ? '#fff' : 'rgba(255,255,255,0.8)', textShadow:'0 1px 2px rgba(0,0,0,0.9)', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', padding:'0 8px', pointerEvents:'none', flex:1 }}>
                      {(translitOn ? transliterate(dlg.text, translitTarget) : dlg.text) || ''}
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

          {/* DUBBED view: one bar per piece of each synthesized clip, exactly as long as it sounds
              (trim + speed applied). Independent of the original bars. A dub can be cut in two
              (right-click → Split) — every piece then moves, trims and deletes on its own. */}
          {dubMode && speakerOrder.map((spk, rowIdx) => {
            const color = speakerColors[spk] || '#888'
            return (spkGroups[spk] || []).map(dlg => {
              const e = dubEntryOf(dlg)
              if (!e) return null
              const top = RULER_H + rowIdx * ROW_H
              const nat = e.natural_ms || 1
              const wlabel = translMap[dlg.id]?.translation || ''
              const many = (e.parts || []).length > 1
              return (
                <React.Fragment key={dlg.id}>
                  {/* thin marker: where the ORIGINAL line sits, for reference only */}
                  <div style={{ position:'absolute', top: top + ROW_H - 5, height: 2, left: msToPx(dlg.start_ms), width: Math.max(2, msToPx(dlg.end_ms - dlg.start_ms)), background: `${color}55`, borderRadius: 2, pointerEvents:'none', zIndex: 2 }} />

                  {(e.parts || []).map((part, pi) => {
                    const { start, len, tin, tout } = getDubBounds(dlg, part)
                    const x = msToPx(start)
                    const w = Math.max(MIN_BAR_PX, msToPx(len))
                    const dr = dragRef.current
                    const draggingThis = dr?.kind === 'dub' && dr.dlg.id === dlg.id && dr.partId === part.part_id && dr.dragging
                    const sel = dlg.id === selectedDialogueId && (part.part_id === selectedPartId || (!many))
                    const gone = !!part.deleted
                    const wkey = `d:${currentJobId}:${dlg.id}:${ttsEpoch}:${ttsVersion[dlg.id] || 0}:${nat}`
                    return (
                      <div key={part.part_id} className="dlg-bar"
                        style={{
                          position:'absolute', top: top+5, height: ROW_H-10, left: x, width: w,
                          borderRadius: 6,
                          background: gone ? 'transparent'
                            : sel ? `linear-gradient(135deg, ${color}cc, ${color}88)`
                            : `linear-gradient(135deg, ${color}55, ${color}33)`,
                          border: gone ? `1px dashed ${color}66` : sel ? `1.5px solid ${color}` : `1px solid ${color}66`,
                          overflow:'hidden', display:'flex', alignItems:'center',
                          transition: draggingThis ? 'none' : 'box-shadow 0.15s',
                          zIndex: draggingThis ? 30 : (sel ? 8 : 5),
                          opacity: gone ? 0.5 : (draggingThis ? 0.92 : 1),
                          boxShadow: draggingThis ? `0 6px 20px rgba(0,0,0,0.55), 0 0 12px ${color}55` : (sel ? `0 0 12px ${color}44` : 'none'),
                          cursor: gone ? 'default' : (draggingThis ? 'grabbing' : 'pointer'),
                        }}
                        onClick={() => {
                          if (justDraggedRef.current) return
                          selectDub(dlg.id, part.part_id)
                          if (gone) return
                          setPlayheadMs(Math.round(start))   // no audio on click — use ▶ in the dialogue editor
                        }}
                        onDoubleClick={ev => { if (!gone) splitDubAt(dlg, e, part, clientXToMs(ev.clientX)) }}
                        onContextMenu={ev => onDubContextMenu(ev, dlg, e, part)}
                        onMouseDown={ev => { if (ev.button === 0 && !gone) startDubDrag(ev, dlg, e, part, 'move') }}
                      >
                        {!gone && (
                          <Waveform
                            url={`${API}/jobs/${currentJobId}/tts_audio/${dlg.id}?v=${ttsEpoch}-${ttsVersion[dlg.id] || 0}`}
                            cacheKey={wkey}
                            from={tin / nat} to={tout / nat}
                            color="#ffffff" alpha={sel ? 0.75 : 0.5} />
                        )}

                        {!gone && (
                          <div title="Drag to trim the start"
                            style={{ position:'absolute', left:0, top:0, bottom:0, width:7, cursor:'ew-resize', zIndex:10, background: sel ? 'rgba(255,255,255,0.28)' : 'rgba(255,255,255,0.10)', borderRight:'1px solid rgba(255,255,255,0.25)' }}
                            onMouseDown={ev => startDubDrag(ev, dlg, e, part, 'resize-left')} />
                        )}

                        {w > 40 && (
                          <span style={{ position:'relative', fontSize:9, color: sel ? '#fff' : 'rgba(255,255,255,0.8)', textShadow:'0 1px 2px rgba(0,0,0,0.9)', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', padding:'0 10px', pointerEvents:'none', flex:1 }}>
                            {gone ? 'part removed' : (pi === 0 ? wlabel : '')}
                          </span>
                        )}

                        {!gone && w > 30 && (
                          <span style={{ position:'absolute', bottom:1, right:9, fontSize:7.5, color:'rgba(255,255,255,0.8)', textShadow:'0 1px 2px #000', fontFamily:'monospace', pointerEvents:'none' }}>
                            {many ? `${pi + 1}/${e.parts.length} · ` : ''}{(len / 1000).toFixed(1)}s{(part.speed && Math.abs(part.speed - 1) > 0.004) ? ` · ${(+part.speed).toFixed(2).replace(/0$/, '')}×` : ''}{(part.trimmed && !many) ? ' ✂' : ''}
                          </span>
                        )}

                        {!gone && (
                          <div title="Drag to trim the end"
                            style={{ position:'absolute', right:0, top:0, bottom:0, width:7, cursor:'ew-resize', zIndex:10, background: sel ? 'rgba(255,255,255,0.28)' : 'rgba(255,255,255,0.10)', borderLeft:'1px solid rgba(255,255,255,0.25)' }}
                            onMouseDown={ev => startDubDrag(ev, dlg, e, part, 'resize-right')} />
                        )}
                      </div>
                    )
                  })}
                </React.Fragment>
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