import { create } from 'zustand'
import axios from 'axios'
import { onEngineReady } from './transliterate'

const API = '/api'

export const SPEAKER_COLORS = [
  '#4CAF50','#2196F3','#FF9800','#E91E63',
  '#9C27B0','#00BCD4','#FF5722','#8BC34A',
  '#3F51B5','#F44336','#009688','#FFC107',
  '#607D8B','#795548','#FF4081','#69F0AE',
]

const WS_RECONNECT_BASE_MS = 1000
const WS_RECONNECT_MAX_MS  = 15000

export const useStore = create((set, get) => ({
  jobs: [], currentJobId: null, jobMeta: null, manifest: null,
  transcription: null, translations: null, ttsManifest: null,
  speakers: [], speakerColors: {}, speakerProfiles: [],
  selectedDialogueId: null, timelineZoom: 80, playheadMs: 0,
  pipelineRunning: false, pipelineStage: null,
  pipelinePaused: false, pausedCheckpoint: null,
  readyForExport: false,
  waitingOnTranslation: false,
  activeOverlay: null, // null | 'log' | 'speakers' | 'translation'
  logs: [],
  undoStack: [], redoStack: [],
  sourceLang: 'auto', targetLang: 'en',
  autoSaveEnabled: true,
  // "View original in another script" — one switch for every dialogue. Off by
  // default each session (original language first); only the chosen target
  // script is remembered. translitVersion bumps when a lazily-loaded engine
  // finishes loading so already-rendered text redraws.
  dubbedView: false,     // Original / Dubbed audio choice (shared so the timeline's click-preview follows it)
  videoPlaying: false,   // true while the video is playing — the timeline skips its click-preview then (it would double up with the mix)
  translitOn: false,
  translitTarget: (() => { try { return localStorage.getItem('dubstudio.translitTarget') || 'latin' } catch { return 'latin' } })(),
  translitVersion: 0,
  ttsRerunStatus: {}, // dialogue_id -> 'running' | 'done' | 'failed'
  asrRerunStatus: {},
  ws: null, wsReconnectAttempt: 0, wsReconnectTimer: null,

  loadJobs: async () => {
    const { data } = await axios.get(`${API}/jobs`)
    set({ jobs: data })
  },

  selectJob: async (jobId) => {
    const { wsReconnectTimer } = get()
    if (wsReconnectTimer) clearTimeout(wsReconnectTimer)
    set({ currentJobId: jobId, logs: [], selectedDialogueId: null,
          transcription: null, translations: null, ttsManifest: null,
          pipelineRunning: false, pipelinePaused: false, pausedCheckpoint: null, readyForExport: false,
          undoStack: [], redoStack: [],
          wsReconnectAttempt: 0 })
    get().connectWS(jobId)
    await get().refreshAll(jobId)
  },

  refreshAll: async (jobId) => {
    const id = jobId || get().currentJobId
    if (!id) return
    const [meta, manifest, speakers, profiles] = await Promise.all([
      axios.get(`${API}/jobs/${id}/meta`).then(r=>r.data).catch(()=>null),
      axios.get(`${API}/jobs/${id}/manifest`).then(r=>r.data).catch(()=>null),
      axios.get(`${API}/jobs/${id}/speakers`).then(r=>r.data).catch(()=>[]),
      axios.get(`${API}/jobs/${id}/speaker_profiles`).then(r=>r.data).catch(()=>[]),
    ])
    const colors = {}
    speakers.forEach((s,i) => { colors[s.id] = SPEAKER_COLORS[i % SPEAKER_COLORS.length] })
    set({ jobMeta: meta, manifest, speakers, speakerColors: colors, speakerProfiles: profiles })
    if (meta?.source_lang) set({ sourceLang: meta.source_lang })
    if (meta?.target_lang) set({ targetLang: meta.target_lang })
    const stages = manifest?.stages || {}
    const updates = {}
    if (stages['01_asr']?.status === 'completed') {
      const t = await axios.get(`${API}/jobs/${id}/transcription`).then(r=>r.data).catch(()=>null)
      updates.transcription = t
    }
    if (stages['03_translation']?.status === 'completed') {
      const t = await axios.get(`${API}/jobs/${id}/translations`).then(r=>r.data).catch(()=>null)
      updates.translations = t
    }
    if (stages['04_tts']?.status === 'completed') {
      const t = await axios.get(`${API}/jobs/${id}/tts_manifest`).then(r=>r.data).catch(()=>null)
      updates.ttsManifest = t
    }
    if (!get().pipelineRunning) {
      updates.readyForExport = stages['04_tts']?.status === 'completed' && stages['05_assembly']?.status !== 'completed'
      updates.waitingOnTranslation = stages['01b_diarization']?.status === 'completed' && stages['03_translation']?.status !== 'completed'
    }
    if (Object.keys(updates).length) set(updates)
  },

  connectWS: (jobId) => {
    const { ws, wsReconnectTimer } = get()
    if (wsReconnectTimer) clearTimeout(wsReconnectTimer)
    if (ws) { try { ws.close() } catch {} }
    const host = window.location.hostname
    const socket = new WebSocket(`ws://${host}:8000/ws/${jobId}`)
    socket.onmessage = (e) => get().handleWS(JSON.parse(e.data))
    socket.onopen = () => set({ wsReconnectAttempt: 0 })
    socket.onclose = () => {
      set({ ws: null })
      // Reconnect with backoff as long as this job is still selected —
      // a dropped connection (backend restart, network blip) used to just
      // kill live updates silently until the user reselected the job.
      const { currentJobId, wsReconnectAttempt } = get()
      if (currentJobId !== jobId) return
      const attempt = wsReconnectAttempt + 1
      const delay   = Math.min(WS_RECONNECT_BASE_MS * 2 ** (attempt - 1), WS_RECONNECT_MAX_MS)
      const timer   = setTimeout(() => {
        if (get().currentJobId === jobId) get().connectWS(jobId)
      }, delay)
      set({ wsReconnectAttempt: attempt, wsReconnectTimer: timer })
    }
    set({ ws: socket })
  },

  handleWS: (msg) => {
    const { refreshAll, currentJobId } = get()
    switch (msg.type) {
      case 'log':
        set(s => ({ logs: [...s.logs.slice(-800), msg] })); break
      case 'stage_started':
        set({ pipelineRunning: true, pipelineStage: msg.stage, readyForExport: false, waitingOnTranslation: false }); break
      case 'stage_completed':
      case 'stage_failed':
      case 'stage_skipped':
        refreshAll(currentJobId); break
      case 'pipeline_paused':
        set({ pipelinePaused: true, pausedCheckpoint: msg.checkpoint, pipelineRunning: false,
              pipelineStage: null })
        refreshAll(currentJobId); break
      case 'pipeline_resumed':
        set({ pipelinePaused: false, pausedCheckpoint: null, pipelineRunning: true }); break
      case 'waiting_on_translation':
        set({ pipelineRunning: false, pipelineStage: null, waitingOnTranslation: true })
        refreshAll(currentJobId); break
      case 'ready_for_export':
        // The whole point of stopping here: TTS is done, nothing runs stage 5
        // automatically. This is the event the backend sends at that point —
        // it used to have no handler at all, which is why pipelineRunning
        // stayed stuck true forever ("keeps showing running") after TTS.
        set({ pipelineRunning: false, pipelineStage: null, pipelinePaused: false,
              pausedCheckpoint: null, readyForExport: true })
        refreshAll(currentJobId); break
      case 'pipeline_complete':
        set({ pipelineRunning: false, pipelineStage: null, pipelinePaused: false })
        refreshAll(currentJobId); break
      case 'pipeline_stopped':
        set({ pipelineRunning: false, pipelineStage: null })
        refreshAll(currentJobId); break
      case 'pipeline_state_lost':
        set({ pipelineRunning: false, pipelineStage: null, pipelinePaused: false, pausedCheckpoint: null }); break
      case 'pipeline_state_recovered':
        set({ pipelineRunning: true, pipelinePaused: false, pausedCheckpoint: null }); break
      case 'dialogue_updated':
      case 'dialogues_merged':
      case 'dialogue_deleted':
      case 'dialogue_split':
      case 'dialogue_added':
      case 'speaker_renamed':
        refreshAll(currentJobId); break
      case 'profiles_built':
        set({ pipelineRunning: false })
        refreshAll(currentJobId); break
      case 'export_started':
        set({ pipelineRunning: true, pipelineStage: '05_assembly', readyForExport: false }); break
      case 'export_complete':
        set({ pipelineRunning: false, readyForExport: false })
        refreshAll(currentJobId)
        // "it will automatically assemble and download" — trigger the
        // download the moment assembly finishes, no second click needed.
        if (msg.download_url) window.open(msg.download_url, '_blank')
        break
      case 'export_failed':
        set({ pipelineRunning: false }); break
      case 'tts_rerun_started':
        set(s => ({ ttsRerunStatus: { ...s.ttsRerunStatus, [msg.dialogue_id]: 'running' } })); break
      case 'tts_rerun_complete':
        set(s => ({ ttsRerunStatus: { ...s.ttsRerunStatus, [msg.dialogue_id]: 'done' } }))
        refreshAll(currentJobId); break
      case 'tts_rerun_failed':
        set(s => ({ ttsRerunStatus: { ...s.ttsRerunStatus, [msg.dialogue_id]: 'failed' } })); break
      case 'asr_rerun_started':
        set(s => ({ asrRerunStatus: { ...s.asrRerunStatus, [msg.dialogue_id]: 'running' } })); break
      case 'asr_rerun_complete':
        set(s => ({ asrRerunStatus: { ...s.asrRerunStatus, [msg.dialogue_id]: 'done' } }))
        refreshAll(currentJobId); break
      case 'asr_rerun_failed':
        set(s => ({ asrRerunStatus: { ...s.asrRerunStatus, [msg.dialogue_id]: 'failed' } })); break
    }
  },

  startPipeline: async () => {
    const { currentJobId } = get()
    await axios.post(`${API}/jobs/${currentJobId}/pipeline/run`)
    set({ pipelineRunning: true, pipelinePaused: false, logs: [] })
  },

  runSingleStage: async (stageId) => {
    const { currentJobId } = get()
    await axios.post(`${API}/jobs/${currentJobId}/pipeline/run_stage`, { stage_id: stageId })
    set({ pipelineRunning: true, pipelinePaused: false, logs: [] })
  },

  resumePipeline: async () => {
    const { currentJobId } = get()
    // No optimistic state set here on purpose: the backend fires
    // pipeline_resumed then (often within the same tick) pipeline_paused
    // again if the very next stage is the assembly/export checkpoint. If we
    // optimistically set pipelineRunning:true here, that write can land
    // AFTER those WS messages already correctly set it back to false,
    // permanently overwriting it — this was the "resume just keeps loading"
    // bug. The WS handler is the single source of truth for this state.
    await axios.post(`${API}/jobs/${currentJobId}/pipeline/resume`).catch(() => {})
  },

  exportVideo: async () => {
    const { currentJobId } = get()
    await axios.post(`${API}/jobs/${currentJobId}/export`)
    set({ pipelineRunning: true })
  },

  selectDialogue: (id) => set({ selectedDialogueId: id }),
  setPlayheadMs:  (ms) => set({ playheadMs: ms }),
  setZoom: (z) => set({ timelineZoom: Math.max(15, Math.min(600, z)) }),
  setAutoSave: (v) => set({ autoSaveEnabled: v }),
  setVideoPlaying: (v) => set({ videoPlaying: !!v }),
  setDubbedView: (v) => set({ dubbedView: !!v }),
  setTranslitOn: (v) => set({ translitOn: !!v }),
  setTranslitTarget: (t) => {
    try { localStorage.setItem('dubstudio.translitTarget', t) } catch {}
    set({ translitTarget: t })
  },

  setSourceLang: async (lang) => {
    set({ sourceLang: lang })
    const { currentJobId, targetLang } = get()
    if (currentJobId) {
      await axios.post(`${API}/jobs/${currentJobId}/language`, {
        source_lang: lang, target_lang: targetLang || 'en',
      }).catch(() => {})
    }
  },

  setTargetLang: async (lang) => {
    set({ targetLang: lang })
    const { currentJobId, sourceLang } = get()
    if (currentJobId) {
      await axios.post(`${API}/jobs/${currentJobId}/language`, {
        source_lang: sourceLang || 'auto', target_lang: lang,
      }).catch(() => {})
    }
  },

  // Snapshot current transcription+translations onto the undo stack. Call
  // this BEFORE making a change so the pre-change state is what gets
  // restored on undo.
  _snapshot: () => {
    const { transcription, translations, undoStack } = get()
    const snap = {
      transcription: transcription ? JSON.parse(JSON.stringify(transcription)) : null,
      translations:  translations  ? JSON.parse(JSON.stringify(translations))  : null,
    }
    set({ undoStack: [...undoStack, snap], redoStack: [] })
  },

  updateDialogue: async (dialogueId, field, value) => {
    const { currentJobId, transcription, translations } = get()
    get()._snapshot()
    if (transcription) {
      const t = JSON.parse(JSON.stringify(transcription))
      t.dialogues?.forEach(d => { if (d.id === dialogueId) d[field] = value })
      set({ transcription: t })
    }
    if (translations) {
      const t = JSON.parse(JSON.stringify(translations))
      t.dialogues?.forEach(d => { if (d.id === dialogueId) d[field] = value })
      set({ translations: t })
    }
    await axios.post(`${API}/jobs/${currentJobId}/dialogue/update`, { dialogue_id: dialogueId, field, value })
  },

  updateDialogueFields: async (dialogueId, fields) => {
    const { currentJobId, transcription, translations } = get()
    get()._snapshot()
    const apply = (obj) => {
      if (!obj) return obj
      const t = JSON.parse(JSON.stringify(obj))
      t.dialogues?.forEach(d => { if (d.id === dialogueId) Object.assign(d, fields) })
      return t
    }
    set({ transcription: apply(transcription), translations: apply(translations) })
    await axios.post(`${API}/jobs/${currentJobId}/dialogue/update_many`, { dialogue_id: dialogueId, fields })
  },

  addDialogue: async (dlg) => {
    const { currentJobId } = get()
    get()._snapshot()
    await axios.post(`${API}/jobs/${currentJobId}/dialogue/add`, dlg)
    await get().refreshAll(currentJobId)
    return dlg.id
  },

  mergeDialogues: async (idA, idB) => {
    const { currentJobId } = get()
    get()._snapshot()
    await axios.post(`${API}/jobs/${currentJobId}/dialogue/merge`, { dialogue_id_a: idA, dialogue_id_b: idB })
  },

  deleteDialogue: async (dialogueId) => {
    const { currentJobId } = get()
    get()._snapshot()
    await axios.post(`${API}/jobs/${currentJobId}/dialogue/delete`, { dialogue_id: dialogueId })
  },

  splitDialogue: async (dialogueId, splitMs) => {
    const { currentJobId } = get()
    get()._snapshot()
    await axios.post(`${API}/jobs/${currentJobId}/dialogue/split`, { dialogue_id: dialogueId, split_ms: splitMs })
  },

  renameSpeaker: async (oldId, newId) => {
    const { currentJobId } = get()
    await axios.post(`${API}/jobs/${currentJobId}/speakers/rename`, { old_id: oldId, new_id: newId })
  },

  addSpeaker: async () => {
    const { currentJobId } = get()
    const { data } = await axios.post(`${API}/jobs/${currentJobId}/speakers/add`)
    await get().refreshAll(currentJobId)
    return data.speaker_id
  },

  buildProfiles: async () => {
    const { currentJobId } = get()
    await axios.post(`${API}/jobs/${currentJobId}/speakers/build_profiles`)
  },

  skipTranslation: async () => {
    const { currentJobId } = get()
    await axios.post(`${API}/jobs/${currentJobId}/translation/skip`)
    await get().refreshAll(currentJobId)
  },

  exportTranslationFile: () => {
    const { currentJobId } = get()
    window.open(`${API}/jobs/${currentJobId}/translation/export_file`, '_blank')
  },

  importTranslationFile: async (file) => {
    const { currentJobId } = get()
    const form = new FormData()
    form.append('file', file)
    const { data } = await axios.post(`${API}/jobs/${currentJobId}/translation/import_file`, form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    await get().refreshAll(currentJobId)
    return data // { matched, unmatched_ids, total_dialogues }
  },

  rerunTTS: async (dialogueId) => {
    const { currentJobId } = get()
    await axios.post(`${API}/jobs/${currentJobId}/tts/rerun`, { dialogue_id: dialogueId })
  },

  rerunASR: async (dialogueId) => {
    const { currentJobId } = get()
    await axios.post(`${API}/jobs/${currentJobId}/asr/rerun`, { dialogue_id: dialogueId })
  },

  toggleOverlay: (name) => set(s => ({ activeOverlay: s.activeOverlay === name ? null : name })),
  closeOverlay: () => set({ activeOverlay: null }),

  refreshAll_tts: async () => {
    const { currentJobId } = get()
    const t = await axios.get(`${API}/jobs/${currentJobId}/tts_manifest`).then(r=>r.data).catch(()=>null)
    if (t) set({ ttsManifest: t })
  },

  // Undo/redo restore local state AND push it back to the server in one
  // call, so server state can't drift from what's shown after an undo.
  undo: async () => {
    const { currentJobId, undoStack, redoStack, transcription, translations } = get()
    if (!undoStack.length) return
    const prev = undoStack[undoStack.length-1]
    set({ undoStack: undoStack.slice(0,-1), redoStack: [...redoStack,{transcription,translations}],
          transcription: prev.transcription, translations: prev.translations })
    await axios.post(`${API}/jobs/${currentJobId}/dialogue/bulk_set`, {
      transcription_dialogues: prev.transcription?.dialogues ?? null,
      translations_dialogues:  prev.translations?.dialogues  ?? null,
    }).catch(() => {})
  },

  redo: async () => {
    const { currentJobId, undoStack, redoStack, transcription, translations } = get()
    if (!redoStack.length) return
    const next = redoStack[redoStack.length-1]
    set({ redoStack: redoStack.slice(0,-1), undoStack: [...undoStack,{transcription,translations}],
          transcription: next.transcription, translations: next.translations })
    await axios.post(`${API}/jobs/${currentJobId}/dialogue/bulk_set`, {
      transcription_dialogues: next.transcription?.dialogues ?? null,
      translations_dialogues:  next.translations?.dialogues  ?? null,
    }).catch(() => {})
  },
}))

// redraw transliterated text once the lazily-loaded engine is ready
onEngineReady(() => useStore.setState((s) => ({ translitVersion: s.translitVersion + 1 })))
