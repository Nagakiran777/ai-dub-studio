// A dubbed dialogue can be CUT into pieces ("parts"). Every part plays its own stretch of the
// same synthesized clip and can be moved, trimmed, sped up or deleted on its own.
//   p0       the dialogue's own dub      → fields live directly on the translation entry
//   p1, p2…  pieces made by splitting    → stored in entry.dub_parts = [{id, dub_start_ms, …}]
// The server annotates every dub with `parts` (resolved geometry); this file edits the raw data.

const FIELDS = ['dub_start_ms', 'dub_trim_start_ms', 'dub_trim_end_ms', 'dub_speed', 'dub_deleted']
export const MIN_KEPT_MS = 100

export const rawPartsOf = (translations, dlgId) =>
  ((translations?.dialogues || []).find(d => d.id === dlgId)?.dub_parts || []).map(p => ({ ...p }))

export const partOf = (entry, partId) =>
  (entry?.parts || []).find(p => p.part_id === partId) || entry?.parts?.[0] || null

/** Save field changes for one part (null removes a field's override). */
export function writePart({ updateDialogueFields, translations }, dlgId, partId, fields) {
  if (!partId || partId === 'p0') return updateDialogueFields(dlgId, fields)
  const raw = rawPartsOf(translations, dlgId)
  const i = raw.findIndex(p => p.id === partId)
  if (i < 0) return Promise.resolve()
  const next = { ...raw[i] }
  Object.entries(fields).forEach(([k, v]) => { if (FIELDS.includes(k)) { if (v === null || v === undefined) delete next[k]; else next[k] = v } })
  raw[i] = next
  return updateDialogueFields(dlgId, { dub_parts: raw })
}

/**
 * Cut `part` at `cutNatMs` (a position inside the synthesized clip). The part keeps everything
 * before the cut; a NEW part gets everything after it and is placed exactly where it would
 * have played, so nothing audible changes until you move or trim one of them.
 * Returns the fields to save, or null if the cut is too close to an edge.
 */
export function splitPart({ translations }, dlgId, entry, part, cutNatMs) {
  const c = Math.round(cutNatMs)
  if (!part || c <= part.trim_in_ms + MIN_KEPT_MS || c >= part.trim_out_ms - MIN_KEPT_MS) return null
  const raw = rawPartsOf(translations, dlgId)
  const n = Math.max(0, ...raw.map(p => parseInt(String(p.id).replace(/\D/g, ''), 10) || 0), (entry.parts || []).length - 1) + 1
  const id = `p${n}`
  const piece = {
    id,
    dub_start_ms: Math.round(part.dub_start_ms + (c - part.trim_in_ms) / part.speed),
    dub_trim_start_ms: c,
  }
  if (part.trim_out_ms < entry.natural_ms - 1) piece.dub_trim_end_ms = Math.round(part.trim_out_ms)
  if (Math.abs(part.speed - 1) > 0.004) piece.dub_speed = part.speed
  const fields = {}
  if (part.part_id === 'p0') {
    fields.dub_trim_end_ms = c
    fields.dub_parts = [...raw, piece]
  } else {
    const i = raw.findIndex(p => p.id === part.part_id)
    if (i < 0) return null
    raw[i] = { ...raw[i], dub_trim_end_ms: c }
    fields.dub_parts = [...raw, piece]
  }
  return { fields, newPartId: id }
}

/** The part that ends exactly where `part` begins (its other half after a split), if any. */
export const previousPartOf = (entry, part) =>
  (entry.parts || []).find(p => p.part_id !== part.part_id && Math.abs(p.trim_out_ms - part.trim_in_ms) < 1.5)

/** Delete a split piece, giving its stretch back to the previous part when they touch. */
export function joinPart({ updateDialogueFields, translations }, dlgId, entry, part) {
  if (part.part_id === 'p0') return null
  const raw = rawPartsOf(translations, dlgId).filter(p => p.id !== part.part_id)
  const prev = previousPartOf(entry, part)
  const fields = { dub_parts: raw.length ? raw : null }
  const tout = part.trim_out_ms >= entry.natural_ms - 1 ? null : Math.round(part.trim_out_ms)
  if (prev) {
    if (prev.part_id === 'p0') fields.dub_trim_end_ms = tout
    else {
      const i = raw.findIndex(p => p.id === prev.part_id)
      if (i >= 0) { const q = { ...raw[i] }; if (tout === null) delete q.dub_trim_end_ms; else q.dub_trim_end_ms = tout; raw[i] = q; fields.dub_parts = raw }
    }
  }
  return updateDialogueFields(dlgId, fields)
}
