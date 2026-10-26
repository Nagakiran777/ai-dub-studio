import React, { useRef, useEffect } from 'react'
import { useStore } from '../store'
import SlideOverPanel from './SlideOverPanel'

export default function LogPanel() {
  const { activeOverlay, closeOverlay, logs } = useStore()
  const active = activeOverlay === 'log'
  const logRef = useRef(null)

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs, active])

  return (
    <SlideOverPanel active={active} onClose={closeOverlay} title="Pipeline Log" icon="📜" accent="#f5a623">
      <div ref={logRef} style={{
        fontFamily: 'monospace', fontSize: 10.5, lineHeight: '1.5', padding: '10px 14px',
      }}>
        {logs.length === 0
          ? <span style={{ color: '#2a2a2a' }}>Waiting for pipeline...</span>
          : logs.map((log, i) => (
            <div key={i} style={{
              color: log.text?.includes('ERROR')   ? '#f87171'
                   : log.text?.includes('WARNING') ? '#fbbf24'
                   : log.text?.includes('✓') || log.text?.toLowerCase().includes('complete') ? '#4ade80'
                   : '#444',
              marginBottom: 1, wordBreak: 'break-word',
            }}>{log.text}</div>
          ))
        }
      </div>
    </SlideOverPanel>
  )
}
