import React, { useEffect, useRef } from 'react'
export default function ContextMenu({ x, y, items, onClose }) {
  const ref = useRef(null)
  useEffect(() => {
    const h = e => { if (ref.current && !ref.current.contains(e.target)) onClose() }
    const k = e => { if (e.key==='Escape') onClose() }
    document.addEventListener('mousedown', h); document.addEventListener('keydown', k)
    return () => { document.removeEventListener('mousedown', h); document.removeEventListener('keydown', k) }
  }, [onClose])
  const vw=window.innerWidth, vh=window.innerHeight, estW=180, estH=items.length*30+12
  const cx=x+estW>vw?vw-estW-8:x, cy=y+estH>vh?vh-estH-8:y
  return (
    <div ref={ref} style={{ position:'fixed', left:cx, top:cy, zIndex:1000, background:'linear-gradient(180deg,#1e1e1e,#191919)', border:'1px solid #2e2e2e', borderRadius:10, boxShadow:'0 8px 32px rgba(0,0,0,0.8)', padding:'5px 0', minWidth:160, backdropFilter:'blur(8px)', userSelect:'none' }} onContextMenu={e=>e.preventDefault()}>
      {items.map((item,i) => {
        if (item.divider) return <div key={i} style={{ height:1, background:'#252525', margin:'4px 0' }} />
        return (
          <button key={i} onClick={() => { item.action?.(); onClose() }} disabled={item.disabled}
            style={{ display:'flex', alignItems:'center', gap:9, width:'100%', padding:'6px 14px', background:'transparent', border:'none', color:item.danger?'#f87171':item.disabled?'#333':'#ccc', fontSize:11, cursor:item.disabled?'not-allowed':'pointer', textAlign:'left', transition:'background 0.1s' }}
            onMouseEnter={e => { if (!item.disabled) e.currentTarget.style.background=item.danger?'rgba(239,68,68,0.1)':'rgba(255,255,255,0.06)' }}
            onMouseLeave={e => e.currentTarget.style.background='transparent'}>
            {item.icon && <span style={{ fontSize:12, width:16, textAlign:'center', flexShrink:0 }}>{item.icon}</span>}
            <span style={{ flex:1 }}>{item.label}</span>
            {item.shortcut && <span style={{ fontSize:9, color:'#444', flexShrink:0 }}>{item.shortcut}</span>}
          </button>
        )
      })}
    </div>
  )
}