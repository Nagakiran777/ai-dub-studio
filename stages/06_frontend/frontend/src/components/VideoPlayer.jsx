import React, { useRef, useEffect, useState } from 'react'
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
  const { currentJobId, setPlayheadMs, playheadMs, transcription, ttsManifest, manifest, setVideoPlaying, dubbedView, setDubbedView } = useStore()
  const containerRef = useRef(null)
  const videoRef   = useRef(null)
  const barRef     = useRef(null)
  const bgAudioRef  = useAudio()
  const ttsAudioRef = useAudio()
  const [playing,  setPlaying]  = useState(false)
  const [duration, setDuration] = useState(0)
  const [speed,    setSpeed]    = useState(1)
  const [vol,      setVol]      = useState(1)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [scrubbing, setScrubbing] = useState(false)
  const [hoverFrac, setHoverFrac] = useState(null)
  const activeTtsIdRef  = useRef(null)
  const activeDubSpeedRef = useRef(1)   // per-dialogue dub speed of the clip now playing
  const speedRef = useRef(1)            // latest global speed, for listeners
  speedRef.current = speed
  const scrubRef = useRef(false)        // gesture state (a ref: pointermove can fire before a re-render)
  const wasPlayingRef = useRef(false)

  const videoUrl = currentJobId ? `${API}/jobs/${currentJobId}/video` : null
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
    setRate(ttsAudioRef.current, speed * activeDubSpeedRef.current, true)
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
    if (!v || !bg || !tts) return
    if (dubbedView) {
      v.muted = true
      bg.src = `${API}/jobs/${currentJobId}/audio_track/no_vocals`
      setRate(bg, speedRef.current, true)
      bg.currentTime = v.currentTime
      bg.volume = vol
      tts.volume = vol
      activeTtsIdRef.current = null
      if (playing) bg.play().catch(() => {})
    } else {
      // Fully tear down rather than just pause — clearing .src (and calling
      // .load()) drops any buffered playback state, so nothing from a previous
      // dubbed session can stay audible once you switch back to the original.
      bg.pause(); bg.removeAttribute('src'); bg.load()
      tts.pause(); tts.removeAttribute('src'); tts.load()
      activeTtsIdRef.current = null
      v.muted = false
      v.volume = vol
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dubbedView, currentJobId])

  useEffect(() => {
    const v = videoRef.current
    if (!v || !dubbedView) return
    const bg = bgAudioRef.current, tts = ttsAudioRef.current
    const onPlay = () => {
      bg.play().catch(() => {})
      // pick the dub up where it was paused (pause -> play inside a line used to leave it silent)
      if (activeTtsIdRef.current && tts.getAttribute('src') && !tts.ended) tts.play().catch(() => {})
    }
    const onPause = () => { bg.pause(); tts.pause() }
    const onSeeking = () => {
      bg.currentTime = v.currentTime
      // A seek can land inside the clip that's "already active" — drop it so
      // the next tick restarts it at the right offset instead of mid-sentence.
      tts.pause(); activeTtsIdRef.current = null
    }
    v.addEventListener('play', onPlay); v.addEventListener('pause', onPause); v.addEventListener('seeking', onSeeking)
    return () => { v.removeEventListener('play', onPlay); v.removeEventListener('pause', onPause); v.removeEventListener('seeking', onSeeking) }
  }, [dubbedView])

  useEffect(() => {
    const v = videoRef.current
    if (!v || !dubbedView) return
    const bg = bgAudioRef.current, tts = ttsAudioRef.current
    const onTime = () => {
      // The mix only follows the video while it is actually PLAYING. A paused
      // video still fires 'timeupdate' after every seek (e.g. clicking a
      // dialogue moves the playhead), which used to start a second copy of the
      // dub on top of the timeline's own click-preview — the "echo".
      if (v.paused) return
      if (Math.abs(bg.currentTime - v.currentTime) > 0.3) bg.currentTime = v.currentTime
      const ms = v.currentTime * 1000
      // The clip that should be playing is whichever dialogue's start we most
      // recently passed — NOT gated on the original line's end, so a dub that
      // runs longer than the original line plays out in full, like the export.
      let target = null
      for (const d of dialogues) {
        if (!ttsMap[d.id]?.success) continue
        if (ms >= d.start_ms && ms < d.start_ms + MAX_DUB_MS && (!target || d.start_ms > target.start_ms)) target = d
      }
      const targetId = target?.id || null
      if (targetId !== activeTtsIdRef.current) {
        activeTtsIdRef.current = targetId
        if (targetId) {
          const dubSpeed = ttsMap[targetId]?.speed ?? 1
          activeDubSpeedRef.current = dubSpeed
          tts.src = `${API}/jobs/${currentJobId}/tts_audio/${targetId}?t=${Date.now()}`
          setRate(tts, speedRef.current * dubSpeed, true)   // after src: it resets the rate
          tts.volume = vol
          // In the export the clip starts at the dialogue's start, so if we
          // arrive part-way into the line (a seek) start part-way into the clip.
          const offset = Math.max(0, (ms - target.start_ms) / 1000 * dubSpeed)
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
  }, [dubbedView, dialogues, ttsManifest, currentJobId, vol])

  const togglePlay = () => {
    if (!videoRef.current) return
    if (playing) { videoRef.current.pause(); setPlaying(false) }
    else { videoRef.current.play(); setPlaying(true) }
  }

  const changeVol = v => {
    setVol(v)
    if (videoRef.current && !dubbedView) videoRef.current.volume = v
    if (bgAudioRef.current) bgAudioRef.current.volume = v
    if (ttsAudioRef.current) ttsAudioRef.current.volume = v
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
          ? <video ref={videoRef} src={videoUrl} style={{ maxHeight:'100%', maxWidth:'100%', objectFit:'contain' }}
              onPlay={() => { setPlaying(true); setVideoPlaying(true) }}
              onPause={() => { setPlaying(false); setVideoPlaying(false) }}
              onEnded={() => { setPlaying(false); setVideoPlaying(false) }} />
          : <div style={{ color:'#222', fontSize:13, display:'flex', flexDirection:'column', alignItems:'center', gap:8 }}>
              <span style={{ fontSize:32 }}>🎬</span><span>Select a job to preview</span>
            </div>
        }
        {videoUrl && (
          <div style={{ position:'absolute', top:8, right:8, display:'flex', gap:6, alignItems:'center' }}>
            {/* Original | Dubbed — locked to Original until Voice Synthesis is done */}
            <div role="group" aria-label="Audio" title={ttsDone ? 'Switch between the original audio and the dubbed mix (a live preview — export for the final file)' : 'Available after Voice Synthesis'}
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
            </div>
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
        {dubbedView && <span style={{ fontSize:9, color:'#f5a623', fontWeight:700 }}>● DUBBED</span>}
        <div style={{ flex:1 }} />
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
