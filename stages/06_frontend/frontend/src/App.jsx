import React, { useEffect, useRef, useState, useCallback } from 'react'
import { useStore } from './store'
import TopBar from './components/TopBar'
import VideoPlayer from './components/VideoPlayer'
import Timeline from './components/Timeline'
import PipelinePanel from './components/PipelinePanel'
import DialogueEditor from './components/DialogueEditor'
import LogPanel from './components/LogPanel'
import SpeakerPanel from './components/SpeakerPanel'
import TranslationPanel from './components/TranslationPanel'
import ShortDialogueNotice from './components/ShortDialogueNotice'

export default function App() {
  const { loadJobs, activeOverlay, closeOverlay } = useStore()
  const [rightW,    setRightW]    = useState(360)
  const [videoH,    setVideoH]    = useState(0.46)
  const [pipelineH, setPipelineH] = useState(0.5)
  const containerRef = useRef(null)
  const draggingRef  = useRef(null)

  useEffect(() => { loadJobs() }, [])

  // Esc closes whichever slide-over (Log / Speakers / Translation) is open
  useEffect(() => {
    if (!activeOverlay) return
    const onKey = (e) => { if (e.key === 'Escape') closeOverlay() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [activeOverlay, closeOverlay])

  const startDrag = useCallback((type) => (e) => {
    e.preventDefault()
    draggingRef.current = { type, startX: e.clientX, startY: e.clientY, startRightW: rightW, startVideoH: videoH, startPipelineH: pipelineH }
    document.body.style.cursor  = type === 'vertical' ? 'col-resize' : 'row-resize'
    document.body.style.userSelect = 'none'
    const onMove = (ev) => {
      const dr = draggingRef.current; if (!dr) return
      const cont = containerRef.current?.getBoundingClientRect(); if (!cont) return
      if (dr.type === 'vertical') {
        setRightW(Math.max(280, Math.min(600, dr.startRightW - (ev.clientX - dr.startX))))
      } else if (dr.type === 'video') {
        const leftH = cont.height - 44
        setVideoH(Math.max(0.2, Math.min(0.75, dr.startVideoH + (ev.clientY - dr.startY) / leftH)))
      } else if (dr.type === 'pipeline') {
        const rightH = cont.height - 44
        setPipelineH(Math.max(0.2, Math.min(0.75, dr.startPipelineH + (ev.clientY - dr.startY) / rightH)))
      }
    }
    const onUp = () => {
      draggingRef.current = null
      document.body.style.cursor = ''; document.body.style.userSelect = ''
      window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove); window.addEventListener('mouseup', onUp)
  }, [rightW, videoH, pipelineH])

  return (
    <div ref={containerRef} style={{ display:'flex', flexDirection:'column', height:'100vh', width:'100vw', overflow:'hidden', background:'#080808' }}>
      <TopBar />
      <ShortDialogueNotice />
      <div style={{ display:'flex', flex:1, overflow:'hidden' }}>
        {/* Left */}
        <div style={{ display:'flex', flexDirection:'column', flex:1, overflow:'hidden', minWidth:0 }}>
          <div style={{ flexShrink:0, overflow:'hidden', height:`${videoH*100}%` }}>
            <VideoPlayer />
          </div>
          <div className="resize-handle-h" onMouseDown={startDrag('video')} />
          <div style={{ flex:1, overflow:'hidden', minHeight:0 }}>
            <Timeline />
          </div>
        </div>
        {/* Vertical divider */}
        <div className="resize-handle" onMouseDown={startDrag('vertical')} />
        {/* Right */}
        <div style={{ display:'flex', flexDirection:'column', overflow:'hidden', flexShrink:0, width:rightW, position:'relative' }}>
          <div style={{ flexShrink:0, overflow:'hidden', height:`${pipelineH*100}%` }}>
            <PipelinePanel />
          </div>
          <div className="resize-handle-h" onMouseDown={startDrag('pipeline')} />
          <div style={{ flex:1, overflow:'hidden', minHeight:0, position:'relative' }}>
            <DialogueEditor />
            <LogPanel />
          </div>
          {/* Speakers/Translation need real room, so they cover the whole right column */}
          <SpeakerPanel />
          <TranslationPanel />
        </div>
      </div>
    </div>
  )
}