import React, { useRef, useEffect, useState } from 'react'
import axios from 'axios'
import { setElementGain, scheduleMute, silenceWindows } from '../dubPlayback'
import { useStore } from '../store'
import { useAudio } from '../useAudio'

const API = '/api'

// Safety ceiling only — stops a dialogue counting as "active" forever if
// something is wrong; real lines are far shorter than this.
const MAX_DUB_MS = 15000

/**
 * Set an audio element's playback rate in a way that survives changing its
 * src (loading a new source resets playbackRate to defaultPlaybackRate, which
 * is why setting only playbackRate once leaves later clips at 1x), and keep the
 * pitch-preservation behaviour explicit for browsers that support it.
 */
function setRate(el, rate, keepPitch) {
  if (!el) return
  el.defaultPlaybackRate = rate
  el.playbackRate = rate
  el.preservesPitch = keepPitch
  el.webkitPreservesPitch = keepPitch
}

const fmt = (s) => {
  if (!s || isNaN(s)) return '0:00'
  const m = Math.floor(s / 60), sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

export default function VideoPlayer() {
  const { currentJobId, setPlayheadMs, playheadMs, transcription, ttsManifest, manifest, setVideoPlaying, dubbedView, setDubbedView, assembled, mix, setMixLayer } = useStore()
  const containerRef = useRef(null)
  const videoRef   = useRef(null)
  const barRef     = useRef(null)
  const bgAudioRef  = useAudio()
  const ttsAudioRef = useAudio()
  const vocAudioRef = useAudio()           // the VOCALS layer (original vocals, silenced under the dubbed dialogues)
  const mixRef      = useRef(mix)
  mixRef.current    = mix
  const lastMuteSchedRef = useRef(0)
  const [mixerOpen, setMixerOpen] = useState(false)
  const [playing,  setPlaying]  = useState(false)
  const [duration, setDuration] = useState(0)
  const [speed,    setSpeed]    = useState(1)
  const [vol,      setVol]      = useState(1)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [scrubbing, setScrubbing] = useState(false)
  const [hoverFrac, setHoverFrac] = useState(null)
  const activeTtsIdRef  = useRef(null)
  const activePartRef   = useRef(null)    // {dlg, trimOut, speed} of the piece now playing
  const activeDubSpeedRef = useRef(1)   // per-dialogue dub speed of the clip now playing
  const speedRef = useRef(1)            // latest global speed, for listeners
  speedRef.current = speed
  const scrubRef = useRef(false)        // gesture state (a ref: pointermove can fire before a re-render)
  const wasPlayingRef = useRef(false)

  const videoUrl = currentJobId ? `${API}/jobs/${currentJobId}/video` : null

  // After an export the player shows the REAL assembled video (its audio already is the dub).
  // As soon as anything is edited the export is out of date and the live preview takes over.
  const [viewPreview, setViewPreview] = useState(false)           // user chose to look at the preview anyway
  useEffect(() => { setViewPreview(false) }, [currentJobId, assembled?.version, assembled?.stale])
  const assembledFresh = !!(assembled?.available && !assembled?.stale)
  const showAssembled  = assembledFresh && !viewPreview
  const mixOn = dubbedView && !showAssembled          // the in-browser dub mix only runs for the preview
  const assembledUrl = currentJobId ? `${API}/jobs/${currentJobId}/assembled_video?v=${assembled?.version || 0}` : null

  // Some formats (webm, mkv, avi, HEVC …) can't play in a browser as-is; the server
  // converts them once into a preview copy. Ask until it's ready.
  const [pv, setPv] = useState({ state: 'checking', error: '' })
  useEffect(() => {
    if (!currentJobId) return
    let dead = false, timer = null
    setPv({ state: 'checking', error: '' })
    const poll = async () => {
      try {
        const { data } = await axios.get(`${API}/jobs/${currentJobId}/video_status`)
        if (dead) return
        setPv({ state: data.state, error: data.error || '' })
        if (data.state === 'processing') timer = setTimeout(poll, 2000)
      } catch { if (!dead) setPv({ state: 'failed', error: 'Could not reach the server' }) }
    }
    poll()
    return () => { dead = true; clearTimeout(timer) }
  }, [currentJobId])
  const lastPlayRef = useRef({ t: 0, playing: false })
  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    const tu = () => { lastPlayRef.current.t = v.currentTime }
    const pl = () => { lastPlayRef.current.playing = true }
    const pa = () => { lastPlayRef.current.playing = false }
    v.addEventListener('timeupdate', tu); v.addEventListener('play', pl); v.addEventListener('pause', pa)
    return () => { v.removeEventListener('timeupdate', tu); v.removeEventListener('play', pl); v.removeEventListener('pause', pa) }
  }, [currentJobId])
  useEffect(() => {          // after the picture source changes, continue from the same moment
    const v = videoRef.current
    if (!v) return
    const { t, playing } = lastPlayRef.current
    if (t < 0.1) return
    const go = () => { try { v.currentTime = t } catch {} ; if (playing) v.play().catch(() => {}) }
    v.addEventListener('loadedmetadata', go, { once: true })
    return () => v.removeEventListener('loadedmetadata', go)
  }, [showAssembled, assembled?.version])   // eslint-disable-line react-hooks/exhaustive-deps

  const retryPreview = async () => {
    setPv({ state: 'processing', error: '' })
    await axios.post(`${API}/jobs/${currentJobId}/video_status/retry`).catch(() => {})
    setPv({ state: 'checking', error: '' })
    // re-run the poll by toggling the job effect
    const { data } = await axios.get(`${API}/jobs/${currentJobId}/video_status`).catch(() => ({ data: { state: 'failed' } }))
    setPv({ state: data.state, error: data.error || '' })
  }
  const ttsDone  = manifest?.stages?.['04_tts']?.status === 'completed'
  const ttsMap = {}
  ;(ttsManifest?.dialogues || []).forEach(d => { ttsMap[d.id] = d })
  const dialogues = transcription?.dialogues || []

  // Dubbed/Original: locked to Original until Voice Synthesis is done, then
  // switches to Dubbed by itself (and starts on Dubbed for jobs already past
  // that point). After that it's the user's choice.
  useEffect(() => { setDubbedView(ttsDone) }, [ttsDone, currentJobId])

  useEffect(() => {
    const v = videoRef.current; if (!v) return
    const onTime = () => setPlayheadMs(Math.round(v.currentTime * 1000))
    const onMeta = () => { setDuration(v.duration); setRate(v, speedRef.current, true) }
    v.addEventListener('timeupdate', onTime); v.addEventListener('loadedmetadata', onMeta)
    return () => { v.removeEventListener('timeupdate', onTime); v.removeEventListener('loadedmetadata', onMeta) }
  }, [currentJobId])

  // Mixer sliders / per-dialogue volume changed → apply to what is playing right now
  useEffect(() => {
    if (!mixOn) return
    setElementGain(bgAudioRef.current, mix.no_vocals)
    setElementGain(vocAudioRef.current, mix.vocals)
    const key = activeTtsIdRef.current
    if (key) {
      const id = key.split(':')[0]
      setElementGain(ttsAudioRef.current, (ttsMap[id]?.dub_volume ?? 1) * mix.dub)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mix, mixOn, ttsManifest])

  // Sync playhead from timeline click
  useEffect(() => {
    if (!videoRef.current) return
    if (Math.abs((videoRef.current.currentTime * 1000) - playheadMs) > 500) {
      videoRef.current.currentTime = playheadMs / 1000
    }
  }, [playheadMs])

  // Fullscreen state tracking
  useEffect(() => {
    const onChange = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  const toggleFullscreen = () => {
    if (document.fullscreenElement) { document.exitFullscreen().catch(() => {}); return }
    containerRef.current?.requestFullscreen?.().catch(() => {})
  }

  // ── Playback speed: video, background audio and dubbed audio follow the
  // global rate. Each dub also has its own per-dialogue speed. Always request
  // pitch preservation for the combined rate so preview playback stays natural,
  // matching the pitch-preserving time-stretch used by the export.
  useEffect(() => {
    setRate(videoRef.current, speed, true)
    setRate(bgAudioRef.current, speed, true)
    setRate(vocAudioRef.current, speed, true)
    setRate(ttsAudioRef.current, speed * activeDubSpeedRef.current, true)
    if (mixOn && vocAudioRef.current.getAttribute('src')) scheduleMute(vocAudioRef.current, silenceWindows(dialogues, ttsMap), speed)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [speed])

  // ── Dubbed view: mute the video's own audio, play the no_vocals track as the
  // background (kept time-synced), and trigger each dialogue's dub as the
  // playhead reaches it. A live client-side mix for review — the final export
  // still only happens through Stage 05.
  useEffect(() => {
    const v = videoRef.current
    const bg = bgAudioRef.current
    const tts = ttsAudioRef.current
    const voc = vocAudioRef.current
    if (!v || !bg || !tts) return
    if (mixOn) {
      v.muted = true
      bg.src = `${API}/jobs/${currentJobId}/audio_track/no_vocals`
      setRate(bg, speedRef.current, true)
      bg.currentTime = v.currentTime
      bg.volume = vol
      tts.volume = vol
      // vocals layer: everything the original voices did except the replaced dialogues
      voc.src = `${API}/jobs/${currentJobId}/audio_track/vocals`
      setRate(voc, speedRef.current, true)
      voc.currentTime = v.currentTime
      voc.volume = vol
      setElementGain(bg, mixRef.current.no_vocals)
      setElementGain(voc, mixRef.current.vocals)
      scheduleMute(voc, silenceWindows(dialogues, ttsMap), speedRef.current)
      activeTtsIdRef.current = null
      if (playing) { bg.play().catch(() => {}); voc.play().catch(() => {}) }
    } else {
      // Fully tear down rather than just pause — clearing .src (and calling
      // .load()) drops any buffered playback state, so nothing from a previous
      // dubbed session can stay audible once you switch back to the original.
      bg.pause(); bg.removeAttribute('src'); bg.load()
      voc.pause(); voc.removeAttribute('src'); voc.load()
      tts.pause(); tts.removeAttribute('src'); tts.load()
      activeTtsIdRef.current = null
      v.muted = false
      v.volume = vol
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mixOn, currentJobId])

  useEffect(() => {
    const v = videoRef.current
    if (!v || !mixOn) return
    const bg = bgAudioRef.current, tts = ttsAudioRef.current, voc = vocAudioRef.current
    const resched = () => scheduleMute(voc, silenceWindows(dialogues, ttsMap), speedRef.current)
    const onPlay = () => {
      bg.play().catch(() => {})
      voc.play().catch(() => {}); resched()
      // pick the dub up where it was paused (pause -> play inside a line used to leave it silent)
      if (activeTtsIdRef.current && tts.getAttribute('src') && !tts.ended) tts.play().catch(() => {})
    }
    const onPause = () => { bg.pause(); voc.pause(); tts.pause() }
    const onSeeked = () => resched()
    const onSeeking = () => {
      bg.currentTime = v.currentTime
      voc.currentTime = v.currentTime
      // A seek can land inside the clip that's "already active" — drop it so
      // the next tick restarts it at the right offset instead of mid-sentence.
      tts.pause(); activeTtsIdRef.current = null
    }
    v.addEventListener('play', onPlay); v.addEventListener('pause', onPause); v.addEventListener('seeking', onSeeking); v.addEventListener('seeked', onSeeked)
    return () => { v.removeEventListener('play', onPlay); v.removeEventListener('pause', onPause); v.removeEventListener('seeking', onSeeking); v.removeEventListener('seeked', onSeeked) }
  }, [mixOn, dialogues, ttsManifest])

  useEffect(() => {
    const v = videoRef.current
    if (!v || !mixOn) return
    const bg = bgAudioRef.current, tts = ttsAudioRef.current, voc = vocAudioRef.current
    const onTime = () => {
      // The mix only follows the video while it is actually PLAYING. A paused
      // video still fires 'timeupdate' after every seek (e.g. clicking a
      // dialogue moves the playhead), which used to start a second copy of the
      // dub on top of the timeline's own click-preview — the "echo".
      if (v.paused) return
      if (Math.abs(bg.currentTime - v.currentTime) > 0.3) bg.currentTime = v.currentTime
      if (voc.getAttribute('src')) {
        if (Math.abs(voc.currentTime - v.currentTime) > 0.3) { voc.currentTime = v.currentTime; scheduleMute(voc, silenceWindows(dialogues, ttsMap), speedRef.current) }
        else if (Date.now() - lastMuteSchedRef.current > 4000) {           // re-anchor the scheduled mutes to absorb clock drift
          lastMuteSchedRef.current = Date.now()
          scheduleMute(voc, silenceWindows(dialogues, ttsMap), speedRef.current)
        }
      }
      const ms = v.currentTime * 1000
      // The clip that should be playing is whichever dialogue's start we most
      // recently passed — NOT gated on the original line's end, so a dub that
      // runs longer than the original line plays out in full, like the export.
      // Each dub has its OWN position/length (moved, trimmed, sped up in the
      // Dubbed timeline view) — independent of the original line's bar. Deleted
      // dubs are skipped, exactly like the export.
      // A dub may be cut into several pieces; each piece has its own position, trim and speed.
      let target = null, targetStart = 0, targetPart = null
      for (const d of dialogues) {
        const e = ttsMap[d.id]
        if (!e?.success) continue
        for (const p of (e.parts || [])) {
          if (p.deleted) continue
          if (ms >= p.dub_start_ms && ms < p.dub_end_ms && (!target || p.dub_start_ms > targetStart)) {
            target = d; targetStart = p.dub_start_ms; targetPart = p
          }
        }
      }
      const targetKey = target ? `${target.id}:${targetPart.part_id}` : null
      if (targetKey !== activeTtsIdRef.current) {
        // Two pieces of the same clip that touch (a fresh split) play straight through — no reload, no gap.
        const prevPiece = activePartRef.current
        if (targetKey && prevPiece && prevPiece.dlg === target.id && Math.abs(prevPiece.trimOut - (targetPart.trim_in_ms || 0)) < 2
            && prevPiece.speed === targetPart.speed && !tts.paused && !tts.ended) {
          activeTtsIdRef.current = targetKey
          activePartRef.current = { dlg: target.id, trimOut: targetPart.trim_out_ms, speed: targetPart.speed }
          return
        }
        activeTtsIdRef.current = targetKey
        activePartRef.current = targetKey ? { dlg: target.id, trimOut: targetPart.trim_out_ms, speed: targetPart.speed } : null
        if (targetKey) {
          const dubSpeed = targetPart.speed ?? 1
          activeDubSpeedRef.current = dubSpeed
          tts.src = `${API}/jobs/${currentJobId}/tts_audio/${target.id}?t=${Date.now()}`
          setRate(tts, speedRef.current * dubSpeed, true)   // after src: it resets the rate
          tts.volume = vol
          // level = this dialogue's own volume × the Dubbed layer slider — exactly what the export uses (no auto-normalization)
          setElementGain(tts, (ttsMap[target.id]?.dub_volume ?? 1) * mixRef.current.dub)
          // Arriving part-way into a piece (a seek) starts part-way into the clip; the
          // piece's kept stretch begins at its trim-in point.
          const offset = (targetPart.trim_in_ms || 0) / 1000 + Math.max(0, (ms - targetStart) / 1000 * dubSpeed)
          tts.currentTime = offset > 0.05 ? offset : 0
          tts.play().catch(() => {})
        } else {
          activeDubSpeedRef.current = 1
          tts.pause()
        }
      }
    }
    v.addEventListener('timeupdate', onTime)
    return () => v.removeEventListener('timeupdate', onTime)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mixOn, dialogues, ttsManifest, currentJobId, vol])

  const togglePlay = () => {
    if (!videoRef.current) return
    if (playing) { videoRef.current.pause(); setPlaying(false) }
    else { videoRef.current.play(); setPlaying(true) }
  }

  const changeVol = v => {
    setVol(v)
    if (videoRef.current && !mixOn) videoRef.current.volume = v
    if (bgAudioRef.current) bgAudioRef.current.volume = v
    if (ttsAudioRef.current) ttsAudioRef.current.volume = v
    if (vocAudioRef.current) vocAudioRef.current.volume = v
  }

  // ── Seek bar: click, or press-hold-and-drag. The video is paused while
  // dragging so the frame under the pointer is what you see, then resumes
  // from where you let go if it was playing.
  const fracFromEvent = (e) => {
    const r = barRef.current.getBoundingClientRect()
    return Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
  }
  const seekToFrac = (f, precise) => {
    const v = videoRef.current
    if (!v || !duration) return
    const t = f * duration
    if (!precise && typeof v.fastSeek === 'function') v.fastSeek(t); else v.currentTime = t
    setPlayheadMs(Math.round(t * 1000))
  }
  const onBarDown = (e) => {
    const v = videoRef.current
    if (!v || !duration) return
    e.currentTarget.setPointerCapture(e.pointerId)
    wasPlayingRef.current = !v.paused
    v.pause()
    scrubRef.current = true; setScrubbing(true)
    seekToFrac(fracFromEvent(e))
  }
  const onBarMove = (e) => {
    const f = fracFromEvent(e)
    setHoverFrac(f)
    if (scrubRef.current) seekToFrac(f)
  }
  const endScrub = (e) => {
    if (!scrubRef.current) return
    scrubRef.current = false; setScrubbing(false)
    seekToFrac(fracFromEvent(e), true)
    if (wasPlayingRef.current) videoRef.current?.play()
  }
  const onBarKey = (e) => {
    const v = videoRef.current
    if (!v || !duration) return
    const step = { ArrowLeft: -5, ArrowRight: 5, ArrowDown: -5, ArrowUp: 5 }[e.key]
    if (step !== undefined) { e.preventDefault(); seekToFrac(Math.min(1, Math.max(0, (v.currentTime + step) / duration)), true) }
    else if (e.key === 'Home') { e.preventDefault(); seekToFrac(0, true) }
    else if (e.key === 'End')  { e.preventDefault(); seekToFrac(1, true) }
  }

  const progress = duration > 0 ? Math.min(1, (playheadMs / 1000) / duration) : 0
  const bubbleFrac = scrubbing ? progress : hoverFrac
  const trackH = (scrubbing || hoverFrac !== null) ? 6 : 3

  return (
    <div ref={containerRef} style={{ height:'100%', display:'flex', flexDirection:'column', background:'#000' }}>
      <div style={{ flex:1, display:'flex', alignItems:'center', justifyContent:'center', overflow:'hidden', position:'relative' }}>
        {videoUrl
          ? <video ref={videoRef} src={showAssembled ? assembledUrl : (pv.state === 'ready' ? videoUrl : undefined)} style={{ maxHeight:'100%', maxWidth:'100%', objectFit:'contain' }}
              onPlay={() => { setPlaying(true); setVideoPlaying(true) }}
              onPause={() => { setPlaying(false); setVideoPlaying(false) }}
              onEnded={() => { setPlaying(false); setVideoPlaying(false) }} />
          : <div style={{ color:'#222', fontSize:13, display:'flex', flexDirection:'column', alignItems:'center', gap:8 }}>
              <span style={{ fontSize:32 }}>🎬</span><span>Select a job to preview</span>
            </div>
        }
        {videoUrl && !showAssembled && (pv.state === 'processing' || pv.state === 'checking' || pv.state === 'failed') && (
          <div style={{ position:'absolute', inset:0, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:10, color:'#888', fontSize:12, textAlign:'center', padding:24, background:'rgba(0,0,0,0.6)', zIndex:5 }}>
            {pv.state === 'failed' ? (
              <>
                <span style={{ fontSize:26 }}>⚠️</span>
                <span style={{ color:'#f87171' }}>This video couldn't be prepared for preview</span>
                <span style={{ fontSize:10, color:'#666', maxWidth:360 }}>{pv.error}</span>
                <button onClick={retryPreview} style={{ padding:'5px 14px', borderRadius:7, fontSize:11, cursor:'pointer', background:'rgba(245,166,35,0.1)', border:'1px solid rgba(245,166,35,0.3)', color:'#f5a623' }}>Try again</button>
              </>
            ) : (
              <>
                <div className="spinner" style={{ width:22, height:22 }} />
                <span>{pv.state === 'processing' ? 'Converting this video so your browser can play it…' : 'Checking video…'}</span>
                {pv.state === 'processing' && <span style={{ fontSize:10, color:'#555' }}>Only for the preview — dubbing and export use your original file.</span>}
              </>
            )}
          </div>
        )}
        {/* The preview is only an approximation — say so until the real video has been assembled */}
        {videoUrl && ttsDone && !showAssembled && pv.state === 'ready' && (
          <div style={{ position:'absolute', left:8, bottom:8, maxWidth:'62%', zIndex:6, padding:'5px 9px', borderRadius:8, fontSize:9.5, lineHeight:1.45, color:'#e9c46a', background:'rgba(0,0,0,0.62)', border:'1px solid rgba(245,166,35,0.3)', backdropFilter:'blur(4px)' }}>
            {assembled?.available && assembled?.stale
              ? <>You changed something after the last export, so this is the <b>preview</b> again. Export to assemble the video and see the real output.</>
              : assembled?.available
              ? <>Showing the <b>preview</b>. <button onClick={() => setViewPreview(false)} style={{ background:'none', border:'none', color:'#4ade80', cursor:'pointer', fontSize:9.5, padding:0, textDecoration:'underline' }}>Show the assembled output</button></>
              : <>This is only a <b>preview</b> and does not show the real output — Export to assemble the video and see it.</>}
          </div>
        )}
        {videoUrl && (
          <div style={{ position:'absolute', top:8, right:8, display:'flex', gap:6, alignItems:'center', zIndex:6 }}>
            {showAssembled && (
              <div style={{ display:'flex', alignItems:'center', gap:8, padding:'4px 10px', borderRadius:9, background:'rgba(34,197,94,0.16)', border:'1px solid rgba(34,197,94,0.45)', backdropFilter:'blur(4px)' }}>
                <span style={{ fontSize:10, fontWeight:700, color:'#4ade80' }}>✓ ASSEMBLED OUTPUT</span>
                <button onClick={() => setViewPreview(true)} title="Look at the live editing preview instead"
                  style={{ fontSize:9, padding:'2px 7px', borderRadius:6, cursor:'pointer', background:'rgba(0,0,0,0.35)', border:'1px solid rgba(255,255,255,0.15)', color:'#bbb' }}>Show preview</button>
              </div>
            )}
            {/* Original | Dubbed — locked to Original until Voice Synthesis is done */}
            {!showAssembled && <div role="group" aria-label="Audio" title={ttsDone ? 'Switch between the original audio and the dubbed mix (a live preview — export for the final file)' : 'Available after Voice Synthesis'}
              style={{
                display:'flex', padding:2, borderRadius:9, gap:2,
                background:'rgba(0,0,0,0.55)', border:'1px solid rgba(255,255,255,0.15)', backdropFilter:'blur(4px)',
                opacity: ttsDone ? 1 : 0.55,
              }}>
              {[['original','Original'],['dubbed','Dubbed']].map(([key,label]) => {
                const active = (key === 'dubbed') === dubbedView
                return (
                  <button key={key} type="button" disabled={!ttsDone} aria-pressed={active}
                    onClick={() => setDubbedView(key === 'dubbed')}
                    style={{
                      padding:'4px 11px', borderRadius:7, fontSize:10, fontWeight:700,
                      cursor: ttsDone ? 'pointer' : 'not-allowed', border:'none',
                      background: active ? (key === 'dubbed' ? 'linear-gradient(135deg,#f5a623,#e8921a)' : 'rgba(255,255,255,0.16)') : 'transparent',
                      color: active ? (key === 'dubbed' ? '#1a1100' : '#fff') : '#8a8a8a',
                      boxShadow: active && key === 'dubbed' ? '0 1px 8px rgba(245,166,35,0.4)' : 'none',
                    }}>
                    {label}
                  </button>
                )
              })}
            </div>}
            <button onClick={toggleFullscreen} title="Fullscreen"
              style={{
                width:28, height:28, borderRadius:7, cursor:'pointer',
                background:'rgba(0,0,0,0.55)', border:'1px solid rgba(255,255,255,0.15)', color:'#ccc',
                backdropFilter:'blur(4px)', display:'flex', alignItems:'center', justifyContent:'center', fontSize:12,
              }}>
              {isFullscreen ? '⤡' : '⤢'}
            </button>
          </div>
        )}
      </div>

      {/* Seek bar — click, or hold and drag. The invisible padding makes the thin track easy to grab. */}
      <div
        ref={barRef} role="slider" tabIndex={0}
        aria-label="Seek" aria-valuemin={0} aria-valuemax={Math.round(duration)} aria-valuenow={Math.round(playheadMs / 1000)}
        onPointerDown={onBarDown} onPointerMove={onBarMove} onPointerUp={endScrub} onPointerCancel={endScrub}
        onPointerLeave={() => { if (!scrubRef.current) setHoverFrac(null) }}
        onKeyDown={onBarKey}
        style={{ height:16, display:'flex', alignItems:'center', cursor: duration ? 'pointer' : 'default', position:'relative', flexShrink:0, touchAction:'none' }}
      >
        <div style={{ position:'relative', width:'100%', height:trackH, background:'#1a1a1a', borderRadius:3, transition:'height 0.12s' }}>
          <div style={{ height:'100%', width:`${progress * 100}%`, background:'linear-gradient(90deg,#f5a623,#ff6b35)', borderRadius:3 }} />
          {duration > 0 && (
            <div style={{
              position:'absolute', top:'50%', left:`${progress * 100}%`,
              transform:`translate(-${progress * 100}%,-50%) scale(${scrubbing ? 1.35 : 1})`,
              width:10, height:10, borderRadius:'50%', background:'#f5a623',
              boxShadow: scrubbing ? '0 0 12px rgba(245,166,35,0.95)' : '0 0 8px rgba(245,166,35,0.8)',
              transition: scrubbing ? 'none' : 'transform 0.12s',
            }} />
          )}
        </div>
        {bubbleFrac !== null && duration > 0 && (
          <div style={{
            position:'absolute', bottom:20, left:`${bubbleFrac * 100}%`, transform:`translateX(-${bubbleFrac * 100}%)`,
            padding:'2px 7px', borderRadius:6, fontSize:10, fontFamily:'monospace', color:'#ddd', pointerEvents:'none',
            background:'rgba(20,20,20,0.92)', border:'1px solid #2a2a2a', whiteSpace:'nowrap',
          }}>{fmt(bubbleFrac * duration)}</div>
        )}
      </div>

      {/* Controls */}
      <div style={{ display:'flex', alignItems:'center', gap:10, padding:'6px 12px', background:'linear-gradient(180deg,#0e0e0e,#0a0a0a)', borderTop:'1px solid #161616', flexShrink:0 }}>
        <button onClick={togglePlay} disabled={!videoUrl}
          style={{ width:30, height:30, borderRadius:8, background: playing?'rgba(245,166,35,0.15)':'linear-gradient(135deg,#f5a623,#e8921a)', border: playing?'1px solid rgba(245,166,35,0.3)':'none', color: playing?'#f5a623':'#000', cursor: videoUrl?'pointer':'not-allowed', fontSize:12, fontWeight:800, display:'flex', alignItems:'center', justifyContent:'center', opacity: videoUrl?1:0.3 }}>
          {playing?'⏸':'▶'}
        </button>
        <span style={{ fontSize:11, color:'#666', fontFamily:'monospace' }}>{fmt(playheadMs/1000)} / {fmt(duration)}</span>
        {showAssembled ? <span style={{ fontSize:9, color:'#4ade80', fontWeight:700 }}>● ASSEMBLED</span> : mixOn && <span style={{ fontSize:9, color:'#f5a623', fontWeight:700 }}>● DUBBED PREVIEW</span>}
        <div style={{ flex:1 }} />
        {/* Mixer: how loud each LAYER is — saved with the job and used by the export too */}
        <div style={{ position:'relative' }}>
          <button onClick={() => setMixerOpen(o => !o)} title="Mixer — volume of the background, vocals and dubbed layers"
            style={{ padding:'3px 8px', borderRadius:6, fontSize:10, cursor:'pointer', background: mixerOpen?'rgba(245,166,35,0.15)':'transparent',
                     color: (mix.no_vocals!==1||mix.vocals!==1||mix.dub!==1) ? '#f5a623' : '#666', border:`1px solid ${mixerOpen?'rgba(245,166,35,0.35)':'#262626'}` }}>🎚 Mixer</button>
          {mixerOpen && (
            <div role="dialog" aria-label="Mixer" style={{ position:'absolute', bottom:'calc(100% + 8px)', right:0, width:250, padding:'10px 12px', borderRadius:10, background:'#121212', border:'1px solid #2a2a2a', boxShadow:'0 12px 40px rgba(0,0,0,0.6)', zIndex:50 }}>
              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:8 }}>
                <span style={{ fontSize:10, fontWeight:700, color:'#f5a623', textTransform:'uppercase', letterSpacing:'0.07em' }}>Mixer</span>
                <button onClick={() => { setMixLayer('no_vocals',1); setMixLayer('vocals',1); setMixLayer('dub',1) }} style={{ fontSize:9, padding:'1px 7px', borderRadius:5, cursor:'pointer', background:'transparent', border:'1px solid #262626', color:'#777' }}>↺ Reset all</button>
              </div>
              {[['no_vocals','Background (non-vocals)'],['vocals','Vocals (screams, sounds)'],['dub','Dubbed voices']].map(([k,label]) => (
                <div key={k} style={{ marginBottom:9 }}>
                  <div style={{ display:'flex', justifyContent:'space-between', fontSize:10, color:'#999', marginBottom:2 }}>
                    <span>{label}</span><b style={{ color: mix[k]===1?'#888':'#f5a623', fontFamily:'monospace' }}>{Math.round(mix[k]*100)}%</b>
                  </div>
                  <input type="range" aria-label={label} min={0} max={3} step={0.05} value={mix[k]} onChange={e => setMixLayer(k, parseFloat(e.target.value))}
                    onDoubleClick={() => setMixLayer(k, 1)} style={{ width:'100%', accentColor:'#f5a623', cursor:'pointer', margin:0 }} />
                </div>
              ))}
              <div style={{ fontSize:9, color:'#444', lineHeight:1.5 }}>Double-click a slider for 100 %. The same levels are used when you export. Each dubbed line also has its own volume in the Dubbed view.</div>
            </div>
          )}
        </div>
        <span style={{ fontSize:11, color:'#444' }}>{vol===0?'🔇':'🔊'}</span>
        <input type="range" min={0} max={1} step={0.05} value={vol} onChange={e => changeVol(parseFloat(e.target.value))}
          style={{ width:55, accentColor:'#f5a623', cursor:'pointer' }} />
        <div style={{ display:'flex', gap:2 }}>
          {[0.5,1,1.5,2].map(s => (
            <button key={s} onClick={() => setSpeed(s)}
              style={{ padding:'3px 7px', borderRadius:5, fontSize:10, cursor:'pointer', background: speed===s?'rgba(245,166,35,0.15)':'transparent', color: speed===s?'#f5a623':'#444', border: speed===s?'1px solid rgba(245,166,35,0.3)':'1px solid transparent', fontWeight: speed===s?700:400, transition:'all 0.15s' }}>
              {s}x
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
