import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { useStore } from '../store'
import ContextMenu from './ContextMenu'
import { WHISPER_LANGUAGES, TARGET_LANGUAGES } from '../languages'
import TransliterationInput from './TransliterationInput'
import { useAudio } from '../useAudio'
import OriginalReader from './OriginalReader'
import DubPanel from './DubPanel'
import { playDubClip } from '../dubPlayback'

const API        = '/api'
const MIN_DUR_MS = 1000

export default function DialogueEditor() {
  const {
    transcription, translations, ttsManifest,
    manifest, speakers, speakerColors, speakerProfiles,
    selectedDialogueId, selectDialogue,
    currentJobId, updateDialogue, deleteDialogue,
    mergeDialogues, splitDialogue, rerunTTS, refreshAll,
    sourceLang, targetLang, playheadMs, autoSaveEnabled, ttsRerunStatus, asrRerunStatus, rerunASR, asrModel,
    timelineMode, setTimelineMode, ttsEpoch, ttsVersion, mix,
  } = useStore()

  const [editText,   setEditText]   = useState('')
  const [editTransl, setEditTransl] = useState('')
  const [editSpk,    setEditSpk]    = useState('')
  const [editSrcLang,setEditSrcLang]= useState('')
  const [editTgtLang,setEditTgtLang]= useState('')
  const [dirty,      setDirty]      = useState(false)
  const touchedRef = useRef(new Set()) // which fields the user actually edited this session
  const [saving,     setSaving]     = useState(false)
  const [rerunning,  setRerunning]  = useState(false)
  const [selRef,     setSelRef]     = useState('')
  const [speakerRefs,setSpeakerRefs]= useState([])
  const [loadingRefs,setLoadingRefs]= useState(false)
  const [prevPlaying,setPrevPlaying]= useState(false)
  const [showAdd,    setShowAdd]    = useState(false)
  const [addText,    setAddText]    = useState('')
  const [addSpk,     setAddSpk]     = useState('')
  const [ctxMenu,    setCtxMenu]    = useState(null)

  const audioRef    = useAudio()
  const ttsAudioRef = useAudio()

  const asr_done    = manifest?.stages?.['01_asr']?.status    === 'completed'
  const transl_done = manifest?.stages?.['03_translation']?.status === 'completed'
  const tts_done    = manifest?.stages?.['04_tts']?.status    === 'completed'

  const dialogues = transcription?.dialogues || []
  const translMap = {}
  ;(translations?.dialogues || []).forEach(d => { translMap[d.id] = d })
  const ttsMap = {}
  ;(ttsManifest?.dialogues || []).forEach(d => { ttsMap[d.id] = d })

  const idx      = dialogues.findIndex(d => d.id === selectedDialogueId)
  const dialogue = dialogues[idx]
  const transl   = translMap[selectedDialogueId]
  const ttsEntry = ttsMap[selectedDialogueId]
  const rerunStatus = dialogue ? (ttsRerunStatus[dialogue.id] || null) : null
  const asrStatus    = dialogue ? (asrRerunStatus[dialogue.id] || null) : null

  // INDIVIDUAL ASR model: the model used to re-transcribe THIS dialogue. Independent of the job's
  // (global) model in the pipeline panel; remembered between sessions. Medium is the recommended default.
  const [oneAsrModel, setOneAsrModel] = useState(() => {
    try { const v = localStorage.getItem('dubstudio.asrModelOne'); return (v === 'large-v3' || v === 'medium') ? v : 'medium' } catch { return 'medium' }
  })
  const pickOneAsrModel = (m) => { setOneAsrModel(m); try { localStorage.setItem('dubstudio.asrModelOne', m) } catch {} }

  // Uploaded reference voices (any audio file) — usable as the ref for any line
  const [userRefs, setUserRefs]     = useState([])
  const [uploading, setUploading]   = useState(false)
  const [uploadErr, setUploadErr]   = useState('')
  const [uploadText, setUploadText] = useState('')
  const fileInputRef = useRef(null)
  const loadUserRefs = () => {
    if (!currentJobId) { setUserRefs([]); return Promise.resolve() }
    return axios.get(`${API}/jobs/${currentJobId}/user_refs`).then(r => setUserRefs(r.data)).catch(() => setUserRefs([]))
  }
  useEffect(() => { loadUserRefs() }, [currentJobId])   // eslint-disable-line react-hooks/exhaustive-deps

  const uploadRef = async (file) => {
    if (!file || !currentJobId) return
    setUploading(true); setUploadErr('')
    try {
      const fd = new FormData(); fd.append('file', file); fd.append('text', uploadText)
      const { data } = await axios.post(`${API}/jobs/${currentJobId}/user_refs`, fd)
      await loadUserRefs()
      setUploadText('')
      await handleSetRef(data.value)          // use it for this dialogue right away
    } catch (e) {
      setUploadErr(e?.response?.data?.detail || 'Upload failed')
    } finally { setUploading(false); if (fileInputRef.current) fileInputRef.current.value = '' }
  }
  const deleteUserRef = async (ref) => {
    if (!confirm(`Remove "${ref.name}"? Lines using it go back to Auto.`)) return
    await axios.delete(`${API}/jobs/${currentJobId}/user_refs/${ref.id}`).catch(() => {})
    if (selRef === ref.value) setSelRef('')
    loadUserRefs()
  }
  const applyRefToSpeaker = async (refVal) => {
    if (!dialogue) return
    await axios.post(`${API}/jobs/${currentJobId}/dialogue/set_ref_speaker`, { speaker_id: dialogue.speaker_id, ref_audio: refVal }).catch(() => {})
  }

  // Load speaker-specific ref dialogues for short dialogues
  useEffect(() => {
    if (!dialogue?.speaker_id || !currentJobId) { setSpeakerRefs([]); return }
    const dur = dialogue.end_ms - dialogue.start_ms
    if (dur >= MIN_DUR_MS && !tts_done) { setSpeakerRefs([]); return }
    setLoadingRefs(true)
    axios.get(`${API}/jobs/${currentJobId}/speaker_dialogues/${dialogue.speaker_id}`)
      .then(r => setSpeakerRefs(r.data.filter(ref => ref.id !== dialogue.id)))
      .catch(() => setSpeakerRefs([]))
      .finally(() => setLoadingRefs(false))
  }, [dialogue?.speaker_id, dialogue?.id, currentJobId, tts_done])

  useEffect(() => {
    if (!dialogue) return
    setEditText(dialogue.text || '')
    setEditSpk(dialogue.speaker_id || '')
    setEditTransl(transl?.translation || dialogue.text || '')
    const resolvedSrc = dialogue.source_lang || sourceLang || 'auto'
    setEditSrcLang(resolvedSrc === 'auto' ? (transcription?.detected_language || 'auto') : resolvedSrc)
    setEditTgtLang(dialogue.target_lang || targetLang || 'en')
    setSelRef(transl?.manual_ref_audio || '')
    setDirty(false)
    touchedRef.current = new Set()
  }, [selectedDialogueId, dialogue?.text, transl?.translation, transl?.manual_ref_audio])

  const mark = (setter, fieldName) => val => {
    setter(val)
    touchedRef.current.add(fieldName)
    setDirty(true)
  }

  const save = async () => {
    if (!dialogue || !dirty) return
    setSaving(true)
    try {
      const touched = touchedRef.current
      if (touched.has('text')       && editText    !== dialogue.text)        await updateDialogue(dialogue.id, 'text',        editText)
      if (touched.has('speaker_id') && editSpk     !== dialogue.speaker_id)  await updateDialogue(dialogue.id, 'speaker_id',  editSpk)
      if (touched.has('source_lang')&& editSrcLang !== (dialogue.source_lang||'')) await updateDialogue(dialogue.id, 'source_lang', editSrcLang)
      if (touched.has('target_lang')&& editTgtLang !== (dialogue.target_lang||'')) await updateDialogue(dialogue.id, 'target_lang', editTgtLang)
      if (touched.has('translation')&& transl_done && editTransl !== (transl?.translation || ''))
        await updateDialogue(dialogue.id, 'translation', editTransl)
      touchedRef.current = new Set()
      setDirty(false)
    } finally { setSaving(false) }
  }

  // Auto-save: debounce a bit after the user stops typing, only for fields
  // they actually touched (per the touched-set above), so nothing gets
  // written that wasn't intentionally changed.
  useEffect(() => {
    if (!autoSaveEnabled || !dirty) return
    const t = setTimeout(() => { save() }, 1400)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoSaveEnabled, dirty, editText, editTransl, editSpk, editSrcLang, editTgtLang])

  // Re-dub: save any pending edits, set ref if picked, kick off rerun.
  // Completion is reported via WS (tts_rerun_started/complete/failed →
  // store.ttsRerunStatus) rather than polling tts_manifest — polling was
  // reporting "done" on the very first tick because the OLD entry from the
  // original bulk TTS run was already present before the rerun even started.
  const handleRerun = async () => {
    if (!dialogue || rerunning) return
    setRerunning(true)

    if (dirty && touchedRef.current.has('translation') && transl_done && editTransl !== (transl?.translation || '')) {
      await updateDialogue(dialogue.id, 'translation', editTransl)
      touchedRef.current.delete('translation')
      if (touchedRef.current.size === 0) setDirty(false)
    }

    if (selRef) {
      await axios.post(`${API}/jobs/${currentJobId}/dialogue/set_ref`, {
        dialogue_id: dialogue.id, ref_audio: selRef,
      }).catch(() => {})
    }

    try {
      await axios.post(`${API}/jobs/${currentJobId}/tts/rerun`, { dialogue_id: dialogue.id })
    } catch {
      useStore.setState(s => ({ ttsRerunStatus: { ...s.ttsRerunStatus, [dialogue.id]: 'failed' } }))
    } finally {
      setRerunning(false)
    }
  }

  const handleSetRef = async (refVal) => {
    setSelRef(refVal)
    if (!dialogue) return
    await axios.post(`${API}/jobs/${currentJobId}/dialogue/set_ref`, {
      dialogue_id: dialogue.id, ref_audio: refVal,
    }).catch(() => {})
  }

  // Play original dialogue slice
  const playOriginal = () => {
    const au = audioRef.current
    if (prevPlaying) { au.pause(); setPrevPlaying(false); return }
    au.src = `${API}/jobs/${currentJobId}/vocals_audio/${dialogue.id}`
    au.currentTime = 0
    au.onended = () => setPrevPlaying(false)
    au.play().then(() => setPrevPlaying(true)).catch(() => setPrevPlaying(false))
  }

  // Play TTS dub
  const playTTS = () => {
    const au = ttsAudioRef.current
    au.pause()
    // Cache-bust: after a redub, the file at this exact URL changed but the
    // URL itself didn't — re-assigning an unchanged .src is a no-op in most
    // browsers, so without this the old buffered audio would just replay.
    // Plays what the export will use: only the kept (trimmed) part, at the dub's
    // speed, pitch preserved.
    playDubClip(au, `${API}/jobs/${currentJobId}/tts_audio/${dialogue.id}?t=${Date.now()}`, {
      speed: ttsEntry?.speed ?? 1, trimInMs: ttsEntry?.trim_in_ms ?? 0,
      trimOutMs: ttsEntry?.trim_out_ms ?? null, naturalMs: ttsEntry?.natural_ms ?? 0,
      gain: (ttsEntry?.dub_volume ?? 1) * (mix?.dub ?? 1),   // this dialogue's volume × Dubbed layer — same as the export
    })
  }

  const playRefPreview = (url) => {
    const au = audioRef.current
    au.src = url; au.currentTime = 0; au.play().catch(() => {})
  }

  const nav = delta => {
    const ni = idx + delta
    if (ni >= 0 && ni < dialogues.length) selectDialogue(dialogues[ni].id)
  }

  // Add new dialogue
  const handleAddDialogue = async () => {
    if (!addText.trim()) return
    const spk = addSpk || speakers[0]?.id || 'speaker_00'
    // Place after selected dialogue or at end
    const startMs = dialogue ? dialogue.end_ms + 500 : (dialogues.length > 0 ? dialogues[dialogues.length-1].end_ms + 500 : 0)
    const endMs   = startMs + Math.max(2000, addText.length * 80)
    const newId   = `dialogue_${Date.now()}`
    // Inject directly into transcription JSON via update endpoint
    await axios.post(`${API}/jobs/${currentJobId}/dialogue/add`, {
      id: newId, speaker_id: spk,
      start_ms: startMs, end_ms: endMs,
      text: addText.trim(),
    }).catch(() => {})
    setAddText(''); setShowAdd(false)
    await refreshAll(currentJobId)
    selectDialogue(newId)
  }

  const fmtMs = ms => {
    if (!ms && ms !== 0) return '—'
    const s = ms/1000, m = Math.floor(s/60), ss = (s%60).toFixed(2)
    return m > 0 ? `${m}:${ss.padStart(5,'0')}` : `${ss}s`
  }

  const color   = speakerColors[dialogue?.speaker_id] || '#888'
  const dur     = dialogue ? dialogue.end_ms - dialogue.start_ms : 0
  const isShort = dur < MIN_DUR_MS
  const ttsOk    = tts_done && ttsEntry?.success && !ttsEntry?.stale
  const ttsStale = tts_done && ttsEntry?.success && ttsEntry?.stale
  const ttsFail  = tts_done && ttsEntry && !ttsEntry.success

  // Right-click on editor panel
  const onEditorContextMenu = (e) => {
    if (!dialogue) return
    e.preventDefault()
    const dlgs = [...dialogues].sort((a, b) => a.start_ms - b.start_ms)
    const i    = dlgs.findIndex(d => d.id === dialogue.id)
    setCtxMenu({
      x: e.clientX, y: e.clientY,
      items: [
        { label: 'Play Original',  icon: '▶', action: playOriginal },
        (ttsOk || ttsStale) && { label: 'Play Dub', icon: '🔊', action: playTTS },
        { divider: true },
        { label: 'Split at Playhead', icon: '⊘',
          action: () => splitDialogue(dialogue.id, playheadMs),
          disabled: playheadMs <= dialogue.start_ms || playheadMs >= dialogue.end_ms,
        },
        { label: 'Split at Middle', icon: '⊘',
          action: () => splitDialogue(dialogue.id, Math.round((dialogue.start_ms+dialogue.end_ms)/2)),
        },
        { label: 'Merge with Next', icon: '⊕',
          action: () => i < dlgs.length-1 && mergeDialogues(dialogue.id, dlgs[i+1].id),
          disabled: i >= dlgs.length-1,
        },
        { divider: true },
        { label: 'Add Dialogue After', icon: '＋', action: () => setShowAdd(true) },
        { label: 'Delete', icon: '🗑', danger: true,
          action: () => { deleteDialogue(dialogue.id); selectDialogue(null) }
        },
      ].filter(Boolean),
    })
  }

  if (!currentJobId || !asr_done) return (
    <div style={{ height:'100%', display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:8, background:'#0e0e0e' }}>
      <span style={{ fontSize:28 }}>✏️</span>
      <span style={{ fontSize:11, color:'#2a2a2a' }}>{!currentJobId ? 'No job selected' : 'Available after Speech Recognition'}</span>
    </div>
  )

  return (
    <div style={{ height:'100%', display:'flex', flexDirection:'column', background:'#0e0e0e', overflow:'hidden' }}
      onContextMenu={onEditorContextMenu}>
      <audio ref={audioRef} />
      <audio ref={ttsAudioRef} />

      {ctxMenu && <ContextMenu x={ctxMenu.x} y={ctxMenu.y} items={ctxMenu.items} onClose={() => setCtxMenu(null)} />}

      {/* Nav */}
      <div style={{
        display:'flex', alignItems:'center', gap:6, padding:'7px 12px',
        borderBottom:'1px solid #1a1a1a', flexShrink:0,
        background:'linear-gradient(180deg, #161616 0%, #0f0f0f 100%)',
      }}>
        <button onClick={() => nav(-1)} disabled={idx <= 0}
          style={{ padding:'3px 8px', borderRadius:5, border:'1px solid #1e1e1e', background:'transparent', color: idx<=0?'#222':'#666', cursor: idx<=0?'not-allowed':'pointer', fontSize:10 }}>◀</button>

        <span style={{ flex:1, textAlign:'center', fontSize:10, color:'#3a3a3a', fontFamily:'monospace' }}>
          {dialogue ? `${idx+1} / ${dialogues.length}` : `— / ${dialogues.length}`}
        </span>

        {dialogue && (
          <button onClick={playOriginal}
            style={{
              padding:'3px 9px', borderRadius:5, border:`1px solid ${prevPlaying?color+'55':'#222'}`,
              background: prevPlaying?color+'18':'transparent',
              color: prevPlaying?color:'#555', cursor:'pointer', fontSize:10,
            }}>
            {prevPlaying ? '⏹' : '▶'} Original
          </button>
        )}

        {/* Add Dialogue button */}
        <button onClick={() => setShowAdd(true)}
          style={{
            padding:'3px 9px', borderRadius:5, border:'1px solid rgba(245,166,35,0.3)',
            background:'rgba(245,166,35,0.08)', color:'#f5a623', cursor:'pointer', fontSize:10, fontWeight:600,
          }}>＋ Add</button>

        <button onClick={() => nav(1)} disabled={idx >= dialogues.length-1}
          style={{ padding:'3px 8px', borderRadius:5, border:'1px solid #1e1e1e', background:'transparent', color: idx>=dialogues.length-1?'#222':'#666', cursor: idx>=dialogues.length-1?'not-allowed':'pointer', fontSize:10 }}>▶</button>
      </div>

      {/* Add Dialogue panel */}
      {showAdd && (
        <div style={{ padding:'10px 12px', borderBottom:'1px solid #1a1a1a', background:'rgba(245,166,35,0.04)', flexShrink:0 }}>
          <div style={{ fontSize:10, color:'#f5a623', fontWeight:600, marginBottom:7 }}>＋ Add New Dialogue</div>
          <textarea value={addText} onChange={e => setAddText(e.target.value)}
            placeholder="Type dialogue text..." className="ds-textarea" rows={2}
            style={{ marginBottom:6 }} />
          <div style={{ display:'flex', gap:6 }}>
            <select value={addSpk} onChange={e => setAddSpk(e.target.value)}
              className="ds-input" style={{ flex:1, fontSize:10 }}>
              <option value="">— speaker —</option>
              {speakers.map(s => <option key={s.id} value={s.id}>{s.id}</option>)}
            </select>
            <button onClick={handleAddDialogue} disabled={!addText.trim()}
              style={{
                padding:'6px 14px', borderRadius:7, fontSize:10, fontWeight:700,
                background: addText.trim() ? 'linear-gradient(135deg,#f5a623,#e8921a)' : 'rgba(255,255,255,0.04)',
                color: addText.trim() ? '#000' : '#333', border:'none', cursor: addText.trim()?'pointer':'not-allowed',
              }}>Add</button>
            <button onClick={() => { setShowAdd(false); setAddText('') }}
              style={{ padding:'6px 10px', borderRadius:7, fontSize:10, background:'transparent', border:'1px solid #222', color:'#555', cursor:'pointer' }}>✕</button>
          </div>
        </div>
      )}

      {/* No dialogue selected */}
      {!dialogue && (
        <div style={{ flex:1, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:6 }}>
          <span style={{ fontSize:24 }}>👆</span>
          <span style={{ fontSize:11, color:'#2a2a2a' }}>Click a dialogue in the timeline</span>
        </div>
      )}

      {/* Dialogue content */}
      {dialogue && (
        <div style={{ flex:1, overflowY:'auto', padding:'10px 12px', display:'flex', flexDirection:'column', gap:10 }}>

          {/* ID + badges */}
          <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
            <span style={{ fontSize:9, fontFamily:'monospace', color:'#2a2a2a' }}>{dialogue.id}</span>
            <div style={{ display:'flex', gap:3 }}>
              {ttsOk    && <Badge color="#22c55e">TTS ✓</Badge>}
              {ttsStale && <Badge color="#f59e0b">⚠ TTS stale — re-dub</Badge>}
              {ttsFail  && <Badge color="#ef4444">TTS ✗</Badge>}
              {isShort && <Badge color="#f59e0b">⚠ &lt;1s</Badge>}
              {rerunStatus==='done'   && <Badge color="#22c55e">Re-dubbed ✓</Badge>}
              {rerunStatus==='failed' && <Badge color="#ef4444">Re-dub failed</Badge>}
              {dialogue?.asr_model && <Badge color="#a78bfa">🎙 {dialogue.asr_model}</Badge>}
              {asrStatus==='done'   && <Badge color="#22c55e">Re-transcribed ✓</Badge>}
              {asrStatus==='failed' && <Badge color="#ef4444">Re-ASR failed</Badge>}
            </div>
          </div>

          {/* Dubbed view: the clip itself — waveform, trim, speed, delete */}
          {timelineMode === 'dubbed' && tts_done && ttsEntry?.success && ttsEntry.dub_len_ms != null && (
            <DubPanel dialogue={dialogue} entry={ttsEntry} color={color} />
          )}

          {/* Speaker */}
          <div>
            <Label>Speaker</Label>
            <div style={{ position:'relative' }}>
              <div style={{ position:'absolute', left:9, top:'50%', transform:'translateY(-50%)', width:7, height:7, borderRadius:'50%', background:color, boxShadow:`0 0 5px ${color}88`, zIndex:1 }} />
              <select value={editSpk} onChange={e => mark(setEditSpk, 'speaker_id')(e.target.value)}
                className="ds-input" style={{ paddingLeft:24, color }}>
                {speakers.map(s => <option key={s.id} value={s.id}>{s.id}</option>)}
              </select>
            </div>
          </div>

          {/* Timing */}
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:6 }}>
            {[
              { label:'Start', val: fmtMs(dialogue.start_ms) },
              { label:'End',   val: fmtMs(dialogue.end_ms) },
              { label:'Dur',   val: fmtMs(dur), warn: isShort },
            ].map(({ label, val, warn }) => (
              <div key={label}>
                <Label>{label}</Label>
                <div style={{ padding:'5px 7px', borderRadius:6, fontFamily:'monospace', fontSize:10, background: warn?'rgba(245,158,11,0.07)':'rgba(255,255,255,0.02)', border:`1px solid ${warn?'rgba(245,158,11,0.2)':'#1e1e1e'}`, color: warn?'#fbbf24':'#666' }}>{val}</div>
              </div>
            ))}
          </div>

          {/* Language overrides */}
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:6 }}>
            <div>
              <Label>Source Lang</Label>
              <select value={editSrcLang} onChange={e => mark(setEditSrcLang, 'source_lang')(e.target.value)} className="ds-input" style={{ fontSize:10 }}>
                {WHISPER_LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.name}</option>)}
              </select>
            </div>
            <div>
              <Label>Target Lang</Label>
              <select value={editTgtLang} onChange={e => mark(setEditTgtLang, 'target_lang')(e.target.value)} className="ds-input" style={{ fontSize:10 }}>
                {TARGET_LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.name}</option>)}
              </select>
            </div>
          </div>

          {/* Ref audio — shown for short dialogues AND at TTS stage for all dialogues */}
          {(isShort || tts_done || transl_done) && (
            <div style={{ padding:10, borderRadius:8, background: isShort?'rgba(245,158,11,0.05)':'rgba(59,130,246,0.05)', border:`1px solid ${isShort?'rgba(245,158,11,0.2)':'rgba(59,130,246,0.15)'}` }}>
              <p style={{ fontSize:10, color: isShort?'#fbbf24':'#93c5fd', marginBottom:7 }}>
                {isShort ? '⚠ Under 1s — pick a ref audio:' : '🎙 Reference audio for this dialogue:'}
              </p>
              <div style={{ display:'flex', flexDirection:'column', gap:3, maxHeight:170, overflowY:'auto' }}>
                {/* Auto (use self) option — always available */}
                <div onClick={() => handleSetRef('')}
                  style={{
                    display:'flex', alignItems:'center', gap:6, padding:'5px 8px', borderRadius:6, cursor:'pointer',
                    background: !selRef?`${color}15`:'rgba(255,255,255,0.02)',
                    border:`1px solid ${!selRef?color+'44':'#1e1e1e'}`,
                  }}>
                  <div style={{ width:8, height:8, borderRadius:'50%', flexShrink:0, background: !selRef?color:'#2a2a2a', border:`1px solid ${!selRef?color:'#333'}` }} />
                  <span style={{ fontSize:9, color: !selRef?'#ccc':'#444', fontStyle:'italic' }}>Auto (use dialogue slice itself)</span>
                </div>

                {/* Voices the user uploaded */}
                {userRefs.map(ref => (
                  <div key={ref.id} onClick={() => handleSetRef(ref.value)}
                    style={{
                      display:'flex', alignItems:'center', gap:6, padding:'5px 8px', borderRadius:6, cursor:'pointer',
                      background: selRef===ref.value?`${color}15`:'rgba(255,255,255,0.02)',
                      border:`1px solid ${selRef===ref.value?color+'44':'#1e1e1e'}`,
                    }}>
                    <div style={{ width:8, height:8, borderRadius:'50%', flexShrink:0, background: selRef===ref.value?color:'#2a2a2a', border:`1px solid ${selRef===ref.value?color:'#333'}` }} />
                    <span style={{ fontSize:8, color:'#f5a623', flexShrink:0 }}>📁</span>
                    <span title={ref.name} style={{ flex:1, fontSize:9, color: selRef===ref.value?'#ccc':'#555', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{ref.name}</span>
                    <span style={{ fontSize:8, color:'#444', flexShrink:0 }}>{(ref.duration_ms/1000).toFixed(1)}s</span>
                    <button onClick={e => { e.stopPropagation(); playRefPreview(ref.url) }} title="Play"
                      style={{ width:17, height:17, borderRadius:4, border:'1px solid #2a2a2a', background:'rgba(255,255,255,0.05)', color:'#555', cursor:'pointer', fontSize:8, display:'flex', alignItems:'center', justifyContent:'center' }}>▶</button>
                    <button onClick={e => { e.stopPropagation(); deleteUserRef(ref) }} title="Remove this uploaded voice"
                      style={{ width:17, height:17, borderRadius:4, border:'1px solid #2a2a2a', background:'rgba(255,255,255,0.05)', color:'#a55', cursor:'pointer', fontSize:8, display:'flex', alignItems:'center', justifyContent:'center' }}>✕</button>
                  </div>
                ))}

                {/* Other lines of the same speaker */}
                {loadingRefs
                  ? <div style={{ fontSize:10, color:'#555', display:'flex', alignItems:'center', gap:5, padding:'4px 2px' }}><div className="spinner" style={{width:10,height:10}} /> Loading...</div>
                  : speakerRefs.length === 0
                  ? <div style={{ fontSize:9, color:'#333', padding:'3px 2px' }}>No other dialogues ≥1s from this speaker</div>
                  : speakerRefs.map(ref => (
                    <div key={ref.id} onClick={() => handleSetRef(ref.id)}
                      style={{
                        display:'flex', alignItems:'center', gap:6, padding:'5px 8px', borderRadius:6, cursor:'pointer',
                        background: selRef===ref.id?`${color}15`:'rgba(255,255,255,0.02)',
                        border:`1px solid ${selRef===ref.id?color+'44':'#1e1e1e'}`,
                      }}>
                      <div style={{ width:8, height:8, borderRadius:'50%', flexShrink:0, background: selRef===ref.id?color:'#2a2a2a', border:`1px solid ${selRef===ref.id?color:'#333'}` }} />
                      <span style={{ flex:1, fontSize:9, color: selRef===ref.id?'#ccc':'#555', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{ref.text}</span>
                      <span style={{ fontSize:8, color:'#444', flexShrink:0 }}>{ref.duration}s</span>
                      <button onClick={e => { e.stopPropagation(); playRefPreview(ref.url) }}
                        style={{ width:17, height:17, borderRadius:4, border:'1px solid #2a2a2a', background:'rgba(255,255,255,0.05)', color:'#555', cursor:'pointer', fontSize:8, display:'flex', alignItems:'center', justifyContent:'center' }}>▶</button>
                    </div>
                  ))
                }
              </div>

              {/* Upload your own voice reference — any audio (or video) file */}
              <div style={{ marginTop:8, paddingTop:8, borderTop:'1px solid #1a1a1a', display:'flex', flexDirection:'column', gap:5 }}>
                <input value={uploadText} onChange={e => setUploadText(e.target.value)}
                  placeholder="(optional) what is said in the clip — improves accuracy"
                  className="ds-input" style={{ fontSize:10 }} />
                <div style={{ display:'flex', gap:6, flexWrap:'wrap' }}>
                  <input ref={fileInputRef} type="file" accept="audio/*,video/*,.wav,.mp3,.m4a,.flac,.ogg,.opus,.aac,.wma,.aiff,.amr,.mp4,.mkv,.webm" style={{ display:'none' }}
                    onChange={e => uploadRef(e.target.files?.[0])} />
                  <button onClick={() => fileInputRef.current?.click()} disabled={uploading}
                    style={{ padding:'4px 10px', borderRadius:6, fontSize:10, fontWeight:600, cursor: uploading?'wait':'pointer', background:'rgba(245,166,35,0.1)', border:'1px solid rgba(245,166,35,0.3)', color:'#f5a623' }}>
                    {uploading ? 'Uploading…' : '📁 Upload voice reference'}
                  </button>
                  {selRef?.startsWith('user:') && (
                    <button onClick={() => applyRefToSpeaker(selRef)} title="Use this uploaded voice for every line of this speaker"
                      style={{ padding:'4px 10px', borderRadius:6, fontSize:10, cursor:'pointer', background:'transparent', border:'1px solid #262626', color:'#888' }}>
                      Use for all of this speaker
                    </button>
                  )}
                </div>
                {uploadErr && <div style={{ fontSize:9, color:'#f87171' }}>{uploadErr}</div>}
                <div style={{ fontSize:8.5, color:'#3a3a3a' }}>Any audio file works (wav, mp3, m4a, flac, ogg…). 3–10 s of clean speech is best. Press 🔄 Re-dub to use it.</div>
              </div>
            </div>
          )}

          {/* Original text */}
          <div>
            <OriginalReader text={editText} />
            <TransliterationInput
              value={editText}
              onChange={v => mark(setEditText, 'text')(v)}
              language={editSrcLang}
              className="ds-textarea" rows={3}
            />
          </div>

          {/* Translation */}
          <div>
            <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:5 }}>
              <Label>Dub Text</Label>
              {!transl_done && <span style={{ fontSize:9, color:'#2a2a2a' }}>Run Translation first</span>}
            </div>
            {transl_done
              ? <TransliterationInput
                  value={editTransl}
                  onChange={v => mark(setEditTransl, 'translation')(v)}
                  language={editTgtLang}
                  className="ds-textarea" rows={3} placeholder="Enter or paste translation..."
                />
              : <div style={{ padding:'7px 10px', borderRadius:7, background:'rgba(255,255,255,0.02)', border:'1px solid #181818', fontSize:11, color:'#222', fontStyle:'italic' }}>Not available yet</div>
            }
          </div>

          {/* TTS preview */}
          {ttsOk && (
            <div style={{ display:'flex', alignItems:'center', gap:7, padding:'7px 9px', borderRadius:7, background:'rgba(34,197,94,0.06)', border:'1px solid rgba(34,197,94,0.15)' }}>
              <button onClick={playTTS}
                style={{ padding:'4px 10px', borderRadius:5, fontSize:10, fontWeight:700, cursor:'pointer', background:'rgba(34,197,94,0.15)', border:'1px solid rgba(34,197,94,0.3)', color:'#4ade80' }}>
                ▶ Dub
              </button>
              <span style={{ fontSize:9, color:'#4ade80', flex:1, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{ttsEntry?.tts_text || ''}</span>
            </div>
          )}
          {ttsStale && (
            <div style={{ display:'flex', alignItems:'center', gap:7, padding:'7px 9px', borderRadius:7, background:'rgba(245,158,11,0.06)', border:'1px solid rgba(245,158,11,0.2)' }}>
              <button onClick={playTTS}
                style={{ padding:'4px 10px', borderRadius:5, fontSize:10, fontWeight:700, cursor:'pointer', background:'rgba(245,158,11,0.15)', border:'1px solid rgba(245,158,11,0.3)', color:'#fbbf24' }}>
                ▶ Old Dub
              </button>
              <span style={{ fontSize:9, color:'#b45309', flex:1 }}>Dialogue changed since this was synthesized — re-dub to match</span>
            </div>
          )}
          {(ttsOk || ttsStale) && timelineMode !== 'dubbed' && (
            <button onClick={() => setTimelineMode('dubbed')}
              style={{ textAlign:'left', padding:'6px 9px', borderRadius:7, fontSize:10, cursor:'pointer', background:'transparent', border:'1px dashed #262626', color:'#666' }}>
              ⚙ Speed, trim &amp; position of the dub → open the <b style={{ color:'#f5a623' }}>Dubbed</b> timeline view
            </button>
          )}
          {ttsFail && (
            <div style={{ padding:'6px 9px', borderRadius:7, background:'rgba(239,68,68,0.06)', border:'1px solid rgba(239,68,68,0.2)', fontSize:10, color:'#f87171' }}>
              TTS failed for this dialogue
            </div>
          )}

          {/* Actions */}
          <div style={{ display:'flex', flexWrap:'wrap', gap:5 }}>
            <button onClick={save} disabled={!dirty||saving}
              style={{
                padding:'5px 12px', borderRadius:6, fontSize:10, fontWeight:700, cursor: dirty?'pointer':'not-allowed',
                background: dirty?'linear-gradient(135deg,#f5a623,#e8921a)':'rgba(255,255,255,0.04)',
                color: dirty?'#000':'#2a2a2a', border: dirty?'none':'1px solid #1e1e1e',
                boxShadow: dirty?'0 2px 10px rgba(245,166,35,0.3)':'none', transition:'all 0.2s',
              }}>
              {saving ? '...' : dirty ? (autoSaveEnabled ? '⏳ Auto-saving…' : '💾 Save') : '✓ Saved'}
            </button>

            {tts_done && transl_done && (
              <button onClick={handleRerun} disabled={rerunning || rerunStatus === 'running'}
                style={{
                  padding:'5px 12px', borderRadius:6, fontSize:10, fontWeight:600, cursor: (rerunning||rerunStatus==='running')?'not-allowed':'pointer',
                  background:'rgba(59,130,246,0.1)', border:'1px solid rgba(59,130,246,0.25)',
                  color:'#60a5fa', opacity: (rerunning||rerunStatus==='running')?0.6:1,
                  display:'flex', alignItems:'center', gap:4,
                }}>
                {(rerunning||rerunStatus==='running')?<><div className="spinner" style={{width:9,height:9}}/> Re-dubbing...</>:'🔄 Re-dub'}
              </button>
            )}

            {dialogue && (
              <select value={oneAsrModel} onChange={e => pickOneAsrModel(e.target.value)} disabled={asrStatus === 'running'}
                aria-label="Select model for this dialogue"
                title="Select model — used only when you re-transcribe THIS dialogue (the job's model for all dialogues is set in the pipeline panel)"
                style={{ padding:'4px 6px', borderRadius:6, fontSize:10, background:'rgba(168,85,247,0.06)', border:'1px solid rgba(168,85,247,0.25)', color:'#c084fc', cursor:'pointer' }}>
                <option value="medium">Medium (recommended)</option>
                <option value="large-v3">Large-v3</option>
              </select>
            )}

            {dialogue && (
              <button onClick={() => rerunASR(dialogue.id, oneAsrModel)} disabled={asrStatus === 'running'}
                title="Re-run speech recognition on this dialogue's current audio slice — useful after dragging its boundaries, or to fix a mis-heard line"
                style={{
                  padding:'5px 12px', borderRadius:6, fontSize:10, fontWeight:600, cursor: asrStatus==='running'?'not-allowed':'pointer',
                  background:'rgba(168,85,247,0.1)', border:'1px solid rgba(168,85,247,0.25)',
                  color:'#c084fc', opacity: asrStatus==='running'?0.6:1,
                  display:'flex', alignItems:'center', gap:4,
                }}>
                {asrStatus==='running'?<><div className="spinner" style={{width:9,height:9}}/> Transcribing...</>:'🎙 Re-transcribe'}
              </button>
            )}

            <button
              onClick={() => {
                const sorted = [...dialogues].sort((a,b)=>a.start_ms-b.start_ms)
                const i = sorted.findIndex(d=>d.id===dialogue.id)
                i<sorted.length-1 && mergeDialogues(dialogue.id, sorted[i+1].id)
              }}
              disabled={(() => {
                const sorted = [...dialogues].sort((a,b)=>a.start_ms-b.start_ms)
                return sorted.findIndex(d=>d.id===dialogue?.id) >= sorted.length-1
              })()}
              style={{ padding:'5px 10px', borderRadius:6, fontSize:10, cursor: idx<dialogues.length-1?'pointer':'not-allowed', background:'transparent', border:'1px solid #1e1e1e', color: idx<dialogues.length-1?'#666':'#222' }}>
              ⊕ Merge
            </button>

            <button
              onClick={() => splitDialogue(dialogue.id, playheadMs > dialogue.start_ms && playheadMs < dialogue.end_ms ? playheadMs : Math.round((dialogue.start_ms+dialogue.end_ms)/2))}
              style={{ padding:'5px 10px', borderRadius:6, fontSize:10, cursor:'pointer', background:'transparent', border:'1px solid #1e1e1e', color:'#666' }}>
              ⊘ Split {playheadMs>dialogue.start_ms&&playheadMs<dialogue.end_ms?'at ▶':'at mid'}
            </button>

            <button
              onClick={() => { if(confirm('Delete?')){ deleteDialogue(dialogue.id); selectDialogue(null) } }}
              style={{ padding:'5px 10px', borderRadius:6, fontSize:10, cursor:'pointer', background:'transparent', border:'1px solid rgba(239,68,68,0.2)', color:'#f87171' }}>
              🗑 Delete
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function Label({ children }) {
  return <div style={{ fontSize:9, color:'#3a3a3a', textTransform:'uppercase', letterSpacing:'0.07em', marginBottom:4 }}>{children}</div>
}
function Badge({ color, children }) {
  return <span style={{ fontSize:9, padding:'2px 6px', borderRadius:4, background:`${color}18`, color, border:`1px solid ${color}33` }}>{children}</span>
}