import { useRef } from 'react'

/**
 * A stable <audio> element created exactly once per component instance.
 * `useRef(new Audio())` looks equivalent but isn't: the argument is evaluated
 * on EVERY render (only the first result is kept), so a component that
 * re-renders several times a second during playback would allocate and
 * discard a fresh media element each time.
 */
export function useAudio() {
  const ref = useRef(null)
  if (ref.current === null && typeof Audio !== 'undefined') ref.current = new Audio()
  return ref
}
