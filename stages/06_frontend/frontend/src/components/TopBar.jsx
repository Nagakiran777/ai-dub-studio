import React, { useState, useRef } from 'react'
import axios from 'axios'
import { useStore } from '../store'
import { WHISPER_LANGUAGES, TARGET_LANGUAGES } from '../languages'
const API = '/api'
export default function TopBar() {
  const { jobs,currentJobId,manifest,loadJobs,selectJob,undoStack,redoStack,undo,redo,sourceLang,setSourceLang,targetLang,setTargetLang,autoSaveEnabled,setAutoSave,transcription,activeOverlay,toggleOverlay } = useStore()
  const [showJobs,setShowJobs]=useState(false),[showNew,setShowNew]=useState(false),[showLang,setShowLang]=useState(false)
  const [videos,setVideos]=useState([]),[selVideo,setSelVideo]=useState(''),[uploading,setUploading]=useState(false),[creating,setCreating]=useState(false)
  const fileRef = useRef(null)
  const openNew = async () => { const {data}=await axios.get(`${API}/input_videos`); setVideos(data); setShowNew(true); setSelVideo('') }
  const handleUpload = async e => {
    const file=e.target.files?.[0]; if(!file) return; setUploading(true)
    try { const form=new FormData(); form.append('file',file); const {data}=await axios.post(`${API}/upload_video`,form,{headers:{'Content-Type':'multipart/form-data'}}); setSelVideo(data.path); setVideos(v=>[...v.filter(x=>x.path!==data.path),data]) } finally { setUploading(false); e.target.value='' }
  }
  const createJob = async () => {
    if(!selVideo) return; setCreating(true)
    try { const {data}=await axios.post(`${API}/jobs/create`,{input_video:selVideo}); await axios.post(`${API}/jobs/${data.job_id}/language`,{source_lang:sourceLang||'auto',target_lang:targetLang||'en'}).catch(()=>{}); await loadJobs(); await selectJob(data.job_id); setShowNew(false) } finally { setCreating(false) }
  }
  const stages=manifest?.stages||{}
  const detectedCode=transcription?.detected_language
  const detectedName=detectedCode && (WHISPER_LANGUAGES.find(l=>l.code===detectedCode)?.name || detectedCode)
  const srcName=(sourceLang&&sourceLang!=='auto')
    ?(WHISPER_LANGUAGES.find(l=>l.code===sourceLang)?.name||sourceLang)
    :(detectedName?`Auto (${detectedName})`:'Auto')
  const tgtName=TARGET_LANGUAGES.find(l=>l.code===(targetLang||'en'))?.name||'English'
  return (
    <>
      <div style={{display:'flex',alignItems:'center',height:44,padding:'0 14px',background:'linear-gradient(180deg,#181818,#111111)',borderBottom:'1px solid #1d1d1d',gap:8,flexShrink:0,zIndex:30,position:'relative'}}>
        <div style={{display:'flex',alignItems:'center',gap:8,marginRight:8,flexShrink:0}}>
          <div style={{width:26,height:26,borderRadius:7,background:'linear-gradient(135deg,#f5a623,#ff6b35)',display:'flex',alignItems:'center',justifyContent:'center',fontSize:13,fontWeight:800,color:'#000',boxShadow:'0 2px 10px rgba(245,166,35,0.45)'}}>D</div>
          <span className="gradient-text" style={{fontWeight:700,fontSize:13,letterSpacing:'-0.3px'}}>DubStudio Pro</span>
        </div>
        <div style={{position:'relative'}}>
          <button onClick={()=>setShowJobs(v=>!v)} style={{display:'flex',alignItems:'center',gap:8,padding:'5px 12px',background:'rgba(255,255,255,0.04)',border:'1px solid #222',borderRadius:8,cursor:'pointer',minWidth:190,fontSize:11,color:'#aaa'}}>
            <span style={{color:'#3a3a3a',fontSize:9}}>JOB</span>
            <span style={{flex:1,textAlign:'left',color:currentJobId?'#ddd':'#333',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{currentJobId||'— select —'}</span>
            <span style={{color:'#2a2a2a',fontSize:9}}>▾</span>
          </button>
          {showJobs&&<div style={{position:'absolute',top:'calc(100% + 4px)',left:0,background:'#161616',border:'1px solid #222',borderRadius:10,minWidth:280,maxHeight:280,overflowY:'auto',boxShadow:'0 12px 40px rgba(0,0,0,0.85)',zIndex:200}}>
            {jobs.length===0?<div style={{padding:'12px 16px',fontSize:11,color:'#333'}}>No jobs</div>
            :jobs.map(j=><button key={j.job_id} onClick={()=>{selectJob(j.job_id);setShowJobs(false)}} style={{width:'100%',textAlign:'left',padding:'9px 16px',fontSize:11,color:j.job_id===currentJobId?'#f5a623':'#888',background:j.job_id===currentJobId?'rgba(245,166,35,0.06)':'transparent',border:'none',cursor:'pointer',display:'block',borderBottom:'1px solid #181818'}}>{j.job_id}</button>)}
          </div>}
        </div>
        <button className="btn-accent" onClick={openNew} style={{borderRadius:8,padding:'5px 13px',fontSize:11,flexShrink:0}}>+ New Job</button>
        <div style={{flex:1}}/>
        <div style={{position:'relative',flexShrink:0}}>
          <button onClick={()=>setShowLang(v=>!v)} style={{display:'flex',alignItems:'center',gap:6,padding:'5px 10px',background:'rgba(255,255,255,0.04)',border:'1px solid #222',borderRadius:8,cursor:'pointer',fontSize:10,color:'#888'}}>
            <span style={{color:'#555'}}>{srcName}</span><span style={{color:'#333'}}>→</span><span style={{color:'#f5a623'}}>{tgtName}</span><span style={{color:'#2a2a2a',fontSize:9}}>▾</span>
          </button>
          {showLang&&<div style={{position:'absolute',top:'calc(100% + 4px)',right:0,background:'#161616',border:'1px solid #222',borderRadius:10,padding:14,minWidth:300,boxShadow:'0 12px 40px rgba(0,0,0,0.85)',zIndex:200}}>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
              <div><label style={{fontSize:9,color:'#444',display:'block',marginBottom:5,textTransform:'uppercase',letterSpacing:'0.07em'}}>Source (ASR)</label>
                <select value={sourceLang||'auto'} onChange={e=>setSourceLang(e.target.value)} style={{width:'100%',background:'#111',border:'1px solid #222',borderRadius:6,padding:'6px 8px',fontSize:11,color:'#aaa',outline:'none',cursor:'pointer'}}>
                  {WHISPER_LANGUAGES.map(l=><option key={l.code} value={l.code}>{l.name}</option>)}</select></div>
              <div><label style={{fontSize:9,color:'#444',display:'block',marginBottom:5,textTransform:'uppercase',letterSpacing:'0.07em'}}>Target (TTS)</label>
                <select value={targetLang||'en'} onChange={e=>setTargetLang(e.target.value)} style={{width:'100%',background:'#111',border:'1px solid #222',borderRadius:6,padding:'6px 8px',fontSize:11,color:'#f5a623',outline:'none',cursor:'pointer'}}>
                  {TARGET_LANGUAGES.map(l=><option key={l.code} value={l.code}>{l.name}</option>)}</select></div>
            </div>
            <button onClick={()=>setShowLang(false)} style={{width:'100%',marginTop:8,padding:'6px',borderRadius:7,background:'rgba(255,255,255,0.05)',border:'1px solid #2a2a2a',color:'#888',fontSize:10,cursor:'pointer'}}>Done</button>
          </div>}
        </div>
        {currentJobId && [
          {key:'speakers',icon:'👥',label:'Speakers',color:'#c084fc'},
          {key:'translation',icon:'🌐',label:'Translation',color:'#60a5fa'},
        ].map(({key,icon,label,color})=>(
          <button key={key} onClick={()=>toggleOverlay(key)}
            style={{display:'flex',alignItems:'center',gap:5,padding:'5px 10px',borderRadius:8,cursor:'pointer',fontSize:10,fontWeight:600,flexShrink:0,
              background:activeOverlay===key?`${color}1a`:'rgba(255,255,255,0.04)',
              border:`1px solid ${activeOverlay===key?color+'55':'#222'}`,
              color:activeOverlay===key?color:'#888',
              transition:'background 0.15s, color 0.15s, border-color 0.15s'}}>
            <span style={{fontSize:12}}>{icon}</span>{label}
          </button>
        ))}
        {[{label:'↩',action:undo,disabled:!undoStack.length},{label:'↪',action:redo,disabled:!redoStack.length}].map(({label,action,disabled})=>(
          <button key={label} onClick={action} disabled={disabled} style={{width:28,height:28,borderRadius:7,border:'1px solid #1e1e1e',background:'transparent',color:disabled?'#2a2a2a':'#666',cursor:disabled?'not-allowed':'pointer',fontSize:14,display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>{label}</button>
        ))}
        <button onClick={()=>setAutoSave(!autoSaveEnabled)} title="Auto-save dialogue edits as you type"
          style={{display:'flex',alignItems:'center',gap:6,padding:'4px 10px',borderRadius:20,border:`1px solid ${autoSaveEnabled?'rgba(34,197,94,0.3)':'#222'}`,background:autoSaveEnabled?'rgba(34,197,94,0.08)':'rgba(255,255,255,0.03)',cursor:'pointer',fontSize:9,color:autoSaveEnabled?'#4ade80':'#555',flexShrink:0}}>
          <span style={{width:24,height:13,borderRadius:8,background:autoSaveEnabled?'#22c55e':'#333',position:'relative',transition:'background 0.15s',flexShrink:0}}>
            <span style={{position:'absolute',top:1,left:autoSaveEnabled?12:1,width:11,height:11,borderRadius:'50%',background:'#fff',transition:'left 0.15s'}}/>
          </span>
          Auto-save
        </button>
      </div>
      {showNew&&<div style={{position:'fixed',inset:0,background:'rgba(0,0,0,0.8)',display:'flex',alignItems:'center',justifyContent:'center',zIndex:300,backdropFilter:'blur(6px)'}}>
        <div style={{background:'#161616',border:'1px solid #252525',borderRadius:16,padding:26,width:440,boxShadow:'0 24px 64px rgba(0,0,0,0.9)'}}>
          <h2 style={{fontSize:15,fontWeight:700,marginBottom:4}}>New Dubbing Job</h2>
          <p style={{fontSize:11,color:'#444',marginBottom:20}}>Browse from your computer or select from input folder</p>
          <input ref={fileRef} type="file" accept="video/*,.mkv,.avi,.mov,.mp4,.webm,.flv,.m4v,.ts,.wmv" style={{display:'none'}} onChange={handleUpload}/>
          <button onClick={()=>fileRef.current?.click()} disabled={uploading} style={{width:'100%',padding:'10px',borderRadius:8,fontSize:11,background:'rgba(255,255,255,0.04)',border:'1px dashed #333',color:'#888',cursor:'pointer',marginBottom:12,display:'flex',alignItems:'center',justifyContent:'center',gap:8}}>
            {uploading?<><div className="spinner" style={{width:12,height:12}}/> Uploading...</>:<><span style={{fontSize:16}}>📁</span> Browse from computer (any video format)</>}
          </button>
          {videos.length>0&&<><div style={{display:'flex',alignItems:'center',gap:8,marginBottom:10}}><div style={{flex:1,height:1,background:'#1e1e1e'}}/><span style={{fontSize:10,color:'#333'}}>or from data/input/</span><div style={{flex:1,height:1,background:'#1e1e1e'}}/></div>
            <select value={selVideo} onChange={e=>setSelVideo(e.target.value)} style={{width:'100%',background:'rgba(255,255,255,0.04)',border:'1px solid #222',borderRadius:8,padding:'8px 12px',fontSize:11,color:selVideo?'#ddd':'#555',marginBottom:14,outline:'none',cursor:'pointer'}}>
              <option value="">— select from folder —</option>{videos.map(v=><option key={v.path} value={v.path}>{v.name}</option>)}</select></>}
          {selVideo&&<div style={{padding:'7px 10px',borderRadius:7,marginBottom:14,background:'rgba(245,166,35,0.07)',border:'1px solid rgba(245,166,35,0.2)',fontSize:10,color:'#f5a623',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>✓ {selVideo.split('/').pop()}</div>}
          <div style={{display:'flex',gap:8,justifyContent:'flex-end'}}>
            <button onClick={()=>setShowNew(false)} style={{borderRadius:8,padding:'7px 16px',fontSize:11,background:'transparent',border:'1px solid #222',color:'#666',cursor:'pointer'}}>Cancel</button>
            <button className="btn-accent" onClick={createJob} disabled={!selVideo||creating} style={{borderRadius:8,padding:'7px 18px',fontSize:11}}>{creating?'Creating...':'Create Job'}</button>
          </div>
        </div>
      </div>}
    </>
  )
}