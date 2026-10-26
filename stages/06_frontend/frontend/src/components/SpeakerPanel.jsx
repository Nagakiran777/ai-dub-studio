import React, { useRef, useState } from 'react'
import { useStore, SPEAKER_COLORS } from '../store'
import SlideOverPanel from './SlideOverPanel'
import { useAudio } from '../useAudio'

const API = '/api'

export default function SpeakerPanel() {
  const {
    activeOverlay, closeOverlay, currentJobId, speakers, transcription,
    renameSpeaker, refreshAll, updateDialogue, addSpeaker,
  } = useStore()
  const active = activeOverlay === 'speakers'
  const [renameMap, setRenameMap] = useState({})
  const [playingId, setPlayingId] = useState(null)
  const [adding, setAdding] = useState(false)
  const audioRef = useAudio()
  const dialogues = transcription?.dialogues || []

  const playDlg = (dlg) => {
    const au = audioRef.current
    au.pause()
    setPlayingId(dlg.id)
    au.src = `${API}/jobs/${currentJobId}/vocals_audio/${dlg.id}`
    au.onended = () => setPlayingId(null)
    au.play().catch(() => setPlayingId(null))
  }

  const commitRename = async (oldId) => {
    const newId = (renameMap[oldId] || '').trim()
    if (!newId || newId === oldId) return
    await renameSpeaker(oldId, newId)
    await refreshAll(currentJobId)
  }

  const handleAddSpeaker = async () => {
    setAdding(true)
    try { await addSpeaker() } finally { setAdding(false) }
  }

  return (
    <SlideOverPanel active={active} onClose={closeOverlay} title="Speakers" icon="👥" accent="#c084fc">
      <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, color: '#c084fc' }}>{speakers.length} Speaker{speakers.length !== 1 ? 's' : ''}</div>
            <div style={{ fontSize: 9, color: '#555', marginTop: 2 }}>Rename anytime, saves as you type. Drag a dialogue bar onto a speaker's row on the Timeline to reassign it.</div>
          </div>
          <button onClick={handleAddSpeaker} disabled={adding}
            style={{
              padding: '6px 12px', borderRadius: 7, fontSize: 10, fontWeight: 700, cursor: adding ? 'not-allowed' : 'pointer',
              background: 'rgba(192,132,252,0.12)', border: '1px solid rgba(192,132,252,0.3)', color: '#c084fc',
              flexShrink: 0, opacity: adding ? 0.6 : 1, whiteSpace: 'nowrap',
            }}>
            {adding ? '...' : '+ Add Speaker'}
          </button>
        </div>

        {speakers.length === 0 && (
          <div style={{ fontSize: 10, color: '#444', padding: '10px 0' }}>No speakers yet — run Speaker Detection, or add one manually.</div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {speakers.map((spk, i) => {
            const color   = SPEAKER_COLORS[i % SPEAKER_COLORS.length]
            const nameVal = renameMap[spk.id] ?? spk.id
            const spkDlgs = dialogues.filter(d => d.speaker_id === spk.id)
            return (
              <div key={spk.id} style={{ borderRadius: 8, border: '1px solid #222', background: 'rgba(0,0,0,0.3)', overflow: 'hidden' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px' }}>
                  <div style={{ width: 9, height: 9, borderRadius: '50%', background: color, flexShrink: 0, boxShadow: `0 0 6px ${color}77` }} />
                  <input
                    value={nameVal}
                    onChange={e => setRenameMap(m => ({ ...m, [spk.id]: e.target.value }))}
                    onBlur={() => commitRename(spk.id)}
                    onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }}
                    placeholder={spk.id}
                    style={{ flex: 1, background: 'transparent', border: 'none', color, fontSize: 12, fontWeight: 600, outline: 'none' }}
                  />
                  <span style={{ fontSize: 9, color: '#444' }}>{spkDlgs.length} line{spkDlgs.length !== 1 ? 's' : ''}</span>
                </div>
                {spkDlgs.length > 0 && (
                  <div style={{ maxHeight: 140, overflowY: 'auto', borderTop: '1px solid #1a1a1a' }}>
                    {spkDlgs.slice(0, 10).map(dlg => (
                      <div key={dlg.id} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '5px 10px', borderBottom: '1px solid #141414', background: playingId === dlg.id ? `${color}10` : 'transparent' }}>
                        <button onClick={() => playingId === dlg.id ? (audioRef.current.pause(), setPlayingId(null)) : playDlg(dlg)}
                          style={{ width: 18, height: 18, borderRadius: 4, flexShrink: 0, background: playingId === dlg.id ? color + '33' : 'rgba(255,255,255,0.05)', border: `1px solid ${playingId === dlg.id ? color + '55' : '#2a2a2a'}`, color: playingId === dlg.id ? color : '#555', cursor: 'pointer', fontSize: 8, fontWeight: 700, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          {playingId === dlg.id ? '■' : '▶'}
                        </button>
                        <span style={{ flex: 1, fontSize: 9.5, color: '#666', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{dlg.text || '—'}</span>
                        <select value={dlg.speaker_id}
                          onChange={e => updateDialogue(dlg.id, 'speaker_id', e.target.value).then(() => refreshAll(currentJobId))}
                          style={{ background: '#111', border: '1px solid #222', borderRadius: 4, padding: '2px 5px', fontSize: 8, color: '#777', cursor: 'pointer' }}>
                          {speakers.map(s => <option key={s.id} value={s.id}>{renameMap[s.id] || s.id}</option>)}
                        </select>
                      </div>
                    ))}
                    {spkDlgs.length > 10 && <div style={{ padding: '4px 10px', fontSize: 9, color: '#333' }}>+{spkDlgs.length - 10} more</div>}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </SlideOverPanel>
  )
}
