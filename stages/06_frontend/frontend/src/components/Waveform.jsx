import React, { useEffect, useRef, useState } from 'react'
import { getPeaks } from '../waveform'

/**
 * Draws a clip's waveform filling its parent box (position:absolute; inset:0).
 *
 *   url / cacheKey   the audio to analyse (cacheKey must change when the audio changes)
 *   from / to        fraction of the clip to draw, 0..1 — lets a TRIMMED dub bar show
 *                    only the part that is kept
 *   color            wave colour
 *   lazy             only fetch/decode once the bar scrolls into view
 *
 * The canvas is stretched to the bar, so while a bar is being dragged/resized
 * the wave simply follows it; it is recomputed when the audio itself changes.
 */
export default function Waveform({ url, cacheKey, from = 0, to = 1, color = '#ffffff', alpha = 0.55, lazy = true, onData }) {
  const wrapRef   = useRef(null)
  const canvasRef = useRef(null)
  const [data, setData]       = useState(null)
  const [visible, setVisible] = useState(!lazy)
  const [size, setSize]       = useState({ w: 0, h: 0 })

  // lazy: wait until the bar is actually on screen
  useEffect(() => {
    if (!lazy || visible) return
    const el = wrapRef.current
    if (!el || typeof IntersectionObserver === 'undefined') { setVisible(true); return }
    const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { setVisible(true); io.disconnect() } }, { rootMargin: '200px' })
    io.observe(el)
    return () => io.disconnect()
  }, [lazy, visible])

  useEffect(() => {
    if (!visible || !url) return
    let dead = false
    setData(null)
    getPeaks(url, cacheKey || url).then(d => { if (!dead) { setData(d); onData && onData(d) } }).catch(() => {})
    return () => { dead = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, url, cacheKey])

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const measure = () => setSize({ w: el.clientWidth, h: el.clientHeight })
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const cv = canvasRef.current
    if (!cv || !data || size.w < 2 || size.h < 2) return
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    cv.width  = Math.max(1, Math.floor(size.w * dpr))
    cv.height = Math.max(1, Math.floor(size.h * dpr))
    const g = cv.getContext('2d')
    g.clearRect(0, 0, cv.width, cv.height)
    g.globalAlpha = alpha
    g.fillStyle = color
    const { peaks } = data
    const n  = peaks.length
    const i0 = Math.max(0, Math.floor(from * n)), i1 = Math.min(n, Math.max(i0 + 1, Math.ceil(to * n)))
    const span = i1 - i0
    const mid  = cv.height / 2
    const colW = Math.max(1, Math.round(dpr))          // one device-pixel-ish column
    const cols = Math.max(1, Math.floor(cv.width / colW))
    for (let c = 0; c < cols; c++) {
      const a = i0 + Math.floor((c / cols) * span)
      const b = Math.max(a + 1, i0 + Math.floor(((c + 1) / cols) * span))
      let m = 0
      for (let i = a; i < b && i < n; i++) if (peaks[i] > m) m = peaks[i]
      const h = Math.max(1, m * (cv.height - 2 * dpr))
      g.fillRect(c * colW, mid - h / 2, colW, h)
    }
  }, [data, size.w, size.h, from, to, color, alpha])

  return (
    <div ref={wrapRef} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
    </div>
  )
}
