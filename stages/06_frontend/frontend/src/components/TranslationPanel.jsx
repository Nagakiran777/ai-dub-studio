import React, { useRef, useState } from 'react'
import { useStore } from '../store'
import SlideOverPanel from './SlideOverPanel'

export default function TranslationPanel() {
  const { activeOverlay, closeOverlay, manifest, skipTranslation, exportTranslationFile, importTranslationFile } = useStore()
  const active = activeOverlay === 'translation'
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const fileInputRef = useRef(null)
  const done = manifest?.stages?.['03_translation']?.status === 'completed'

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setBusy(true); setResult(null)
    try {
      const r = await importTranslationFile(file)
      setResult(r)
    } catch {
      setResult({ error: true })
    } finally { setBusy(false) }
  }

  return (
    <SlideOverPanel active={active} onClose={closeOverlay} title="Translation" icon="🌐" accent="#60a5fa">
      <div style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{
          padding: '8px 10px', borderRadius: 8, fontSize: 10, lineHeight: 1.5,
          background: done ? 'rgba(34,197,94,0.08)' : 'rgba(255,255,255,0.03)',
          border: `1px solid ${done ? 'rgba(34,197,94,0.2)' : '#262626'}`,
          color: done ? '#4ade80' : '#666',
        }}>
          {done ? '✓ Translation is done for this job.' : 'Not started yet. Two ways to do it:'}
        </div>

        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: '#60a5fa', marginBottom: 6 }}>Option A — Any AI, via a file</div>
          <button onClick={exportTranslationFile}
            style={{ width: '100%', padding: '9px 12px', borderRadius: 8, fontSize: 11, fontWeight: 600, cursor: 'pointer', background: 'rgba(96,165,250,0.1)', border: '1px solid rgba(96,165,250,0.25)', color: '#60a5fa', textAlign: 'left', marginBottom: 6 }}>
            ⬇ Get translation file
          </button>
          <div style={{ fontSize: 9, color: '#444', marginBottom: 6, lineHeight: 1.5 }}>
            Paste the whole file into ChatGPT, Claude, Gemini — any chat AI — then come back and upload its reply.
          </div>
          <button onClick={() => fileInputRef.current?.click()} disabled={busy}
            style={{ width: '100%', padding: '9px 12px', borderRadius: 8, fontSize: 11, fontWeight: 600, cursor: busy ? 'not-allowed' : 'pointer', background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.25)', color: '#4ade80', textAlign: 'left', opacity: busy ? 0.6 : 1 }}>
            {busy ? 'Importing...' : "⬆ Upload the AI's response"}
          </button>
          <input ref={fileInputRef} type="file" accept=".txt" style={{ display: 'none' }} onChange={handleUpload} />
          {result && !result.error && (
            <div style={{ fontSize: 9, color: result.unmatched_ids?.length ? '#fbbf24' : '#4ade80', lineHeight: 1.5, marginTop: 6 }}>
              Matched {result.matched}/{result.total_dialogues}.
              {result.unmatched_ids?.length ? ` Couldn't match: ${result.unmatched_ids.slice(0,8).join(', ')}${result.unmatched_ids.length>8?'…':''}` : ' All lines matched.'}
            </div>
          )}
          {result?.error && <div style={{ fontSize: 9, color: '#f87171', marginTop: 6 }}>Import failed — check the file format.</div>}
        </div>

        <div style={{ height: 1, background: '#1a1a1a' }} />

        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: '#888', marginBottom: 6 }}>Option B — Type it yourself</div>
          <button onClick={skipTranslation} disabled={busy}
            style={{ width: '100%', padding: '9px 12px', borderRadius: 8, fontSize: 11, fontWeight: 600, cursor: 'pointer', background: 'transparent', border: '1px solid #262626', color: '#888', textAlign: 'left' }}>
            ⏭ Skip — I'll type each line in the editor
          </button>
        </div>
      </div>
    </SlideOverPanel>
  )
}
