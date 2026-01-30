import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ref, get, set, push } from 'firebase/database'
import { db, auth } from '../firebase.js'

function loadImageData(url) {
  return new Promise((resolve) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = img.width
      canvas.height = img.height
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0)
      resolve({ data: ctx.getImageData(0, 0, img.width, img.height), width: img.width, height: img.height })
    }
    img.src = url
  })
}

function toGreyscale(imageData) {
  const d = imageData.data
  const grey = new Float32Array(d.length / 4)
  for (let i = 0; i < d.length; i += 4) {
    grey[i / 4] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
  }
  return grey
}

function rotatePoint(x, y, cx, cy, angle) {
  const rad = angle * Math.PI / 180
  const cos = Math.cos(rad), sin = Math.sin(rad)
  const nx = cos * (x - cx) - sin * (y - cy) + cx
  const ny = sin * (x - cx) + cos * (y - cy) + cy
  return [nx, ny]
}

function getBbox(points) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of points) {
    minX = Math.min(minX, x); minY = Math.min(minY, y)
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y)
  }
  return { x: Math.floor(minX), y: Math.floor(minY), w: Math.ceil(maxX - minX), h: Math.ceil(maxY - minY) }
}

function createRotatedMaskedImage(greyData, width, height, contour, angle, clipPercent = 100) {
  const cx = width / 2, cy = height / 2
  const rotatedContour = contour.map(([x, y]) => rotatePoint(x, y, cx, cy, angle))
  const bbox = getBbox(rotatedContour)
  const offsetContour = rotatedContour.map(([x, y]) => [x - bbox.x, y - bbox.y])
  
  const outCanvas = document.createElement('canvas')
  outCanvas.width = bbox.w
  outCanvas.height = bbox.h
  const outCtx = outCanvas.getContext('2d')
  const outData = outCtx.createImageData(bbox.w, bbox.h)
  
  const rad = -angle * Math.PI / 180
  const cos = Math.cos(rad), sin = Math.sin(rad)
  
  for (let dy = 0; dy < bbox.h; dy++) {
    for (let dx = 0; dx < bbox.w; dx++) {
      const rx = dx + bbox.x, ry = dy + bbox.y
      const sx = cos * (rx - cx) - sin * (ry - cy) + cx
      const sy = sin * (rx - cx) + cos * (ry - cy) + cy
      const sxi = Math.floor(sx), syi = Math.floor(sy)
      if (sxi >= 0 && sxi < width && syi >= 0 && syi < height) {
        const val = greyData[syi * width + sxi]
        const idx = (dy * bbox.w + dx) * 4
        outData.data[idx] = val
        outData.data[idx + 1] = val
        outData.data[idx + 2] = val
        outData.data[idx + 3] = 255
      }
    }
  }
  outCtx.putImageData(outData, 0, 0)
  
  const maskCanvas = document.createElement('canvas')
  maskCanvas.width = bbox.w
  maskCanvas.height = bbox.h
  const maskCtx = maskCanvas.getContext('2d')
  maskCtx.fillStyle = 'white'
  maskCtx.beginPath()
  maskCtx.moveTo(offsetContour[0][0], offsetContour[0][1])
  for (let i = 1; i < offsetContour.length; i++) maskCtx.lineTo(offsetContour[i][0], offsetContour[i][1])
  maskCtx.closePath()
  maskCtx.fill()
  const maskData = maskCtx.getImageData(0, 0, bbox.w, bbox.h)
  
  const maskedVals = []
  for (let i = 0; i < outData.data.length; i += 4) {
    if (maskData.data[i] > 0) maskedVals.push(outData.data[i])
  }
  maskedVals.sort((a, b) => a - b)
  const minVal = maskedVals[0] || 0
  const maxIdx = Math.floor(maskedVals.length * clipPercent / 100) - 1
  const maxVal = maskedVals[Math.max(0, maxIdx)] || 255
  const range = maxVal - minVal || 1
  
  for (let i = 0; i < outData.data.length; i += 4) {
    if (maskData.data[i] === 0) {
      outData.data[i] = 0
      outData.data[i + 1] = 0
      outData.data[i + 2] = 0
    } else {
      const scaled = Math.min(255, Math.round((outData.data[i] - minVal) / range * 255))
      outData.data[i] = scaled
      outData.data[i + 1] = scaled
      outData.data[i + 2] = scaled
    }
    outData.data[i + 3] = 255
  }
  outCtx.putImageData(outData, 0, 0)
  return outCanvas.toDataURL()
}

export default function DataViewer() {
  const { folderName } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [zoom, setZoom] = useState(1)
  const [selectedImg, setSelectedImg] = useState(null)
  const [unifiedUrl, setUnifiedUrl] = useState(null)
  const [rotatedImages, setRotatedImages] = useState([])
  const [unifiedExpanded, setUnifiedExpanded] = useState(false)
  const [metaExpanded, setMetaExpanded] = useState(false)
  const [auditExpanded, setAuditExpanded] = useState(false)
  const [brightness, setBrightness] = useState(100)
  const [imageData, setImageData] = useState(null)
  const [modalZoom, setModalZoom] = useState(1)
  const [shiftPressed, setShiftPressed] = useState(false)
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 })
  const [dragging, setDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [didDrag, setDidDrag] = useState(false)
  const [notes, setNotes] = useState('')
  const [saveStatus, setSaveStatus] = useState(null)
  const saveTimeoutRef = useRef(null)
  const [label, setLabel] = useState('')
  const [availableLabels, setAvailableLabels] = useState([])
  const [labelSaveStatus, setLabelSaveStatus] = useState(null)

  const saveNotes = useCallback((value) => {
    setSaveStatus('saving')
    const now = new Date().toISOString()
    Promise.all([
      set(ref(db, `folders/${folderName}/notes`), value),
      set(ref(db, `folders/${folderName}/last_update`), now)
    ]).then(() => {
      setSaveStatus('saved')
      setTimeout(() => setSaveStatus(null), 2000)
    })
  }, [folderName])

  const saveLabel = useCallback((newLabel) => {
    const oldLabel = label
    if (newLabel === oldLabel) return
    setLabelSaveStatus('saving')
    const now = new Date().toISOString()
    const user = auth.currentUser
    const auditEntry = {
      user: user?.displayName || user?.email || 'unknown',
      timestamp: now,
      from: oldLabel,
      to: newLabel
    }
    Promise.all([
      set(ref(db, `folders/${folderName}/User Input`), newLabel),
      set(ref(db, `folders/${folderName}/last_update`), now),
      push(ref(db, `folders/${folderName}/audit`), auditEntry)
    ]).then(() => {
      setLabel(newLabel)
      setLabelSaveStatus('saved')
      setTimeout(() => setLabelSaveStatus(null), 2000)
    })
  }, [folderName, label])

  const handleNotesChange = e => {
    const value = e.target.value
    setNotes(value)
    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current)
    saveTimeoutRef.current = setTimeout(() => saveNotes(value), 1000)
  }

  useEffect(() => {
    const down = e => {
      if (e.key === 'Shift') setShiftPressed(true)
      if (e.key === 'Escape') { setSelectedImg(null); setModalZoom(1); setPanOffset({ x: 0, y: 0 }) }
    }
    const up = e => e.key === 'Shift' && setShiftPressed(false)
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up) }
  }, [])

  const handleMouseDown = e => {
    e.preventDefault()
    setDragging(true)
    setDidDrag(false)
    setDragStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y })
  }
  const handleMouseMove = e => {
    if (!dragging) return
    setDidDrag(true)
    setPanOffset({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y })
  }
  const handleMouseUp = () => setDragging(false)

  useEffect(() => {
    get(ref(db, 'folders')).then(snap => {
      if (snap.exists()) {
        const all = snap.val()
        const labels = [...new Set(Object.values(all).map(f => f['User Input'] || f.label || '').filter(Boolean))].sort()
        setAvailableLabels(labels)
        const d = all[folderName]
        if (d) {
          setData(d)
          setNotes(d.notes || '')
          setLabel(d['User Input'] || d.label || '')
        }
      }
    })
  }, [folderName])

  useEffect(() => {
    if (!data?.download_links) return
    const links = data.download_links
    if (!links['-1'] || !links['0']) return
    
    const contour = typeof data.Contour === 'string' ? JSON.parse(data.Contour) : data.Contour
    const angles = data['Chosen Facet PD'] || []
    
    Promise.all([loadImageData(links['-1']), loadImageData(links['0'])]).then(([img1, img2]) => {
      const grey1 = toGreyscale(img1.data)
      const grey2 = toGreyscale(img2.data)
      const unified = new Float32Array(grey1.length)
      for (let i = 0; i < grey1.length; i++) unified[i] = (grey1[i] + grey2[i]) / 2
      
      const canvas = document.createElement('canvas')
      canvas.width = img1.width
      canvas.height = img1.height
      const ctx = canvas.getContext('2d')
      const out = ctx.createImageData(img1.width, img1.height)
      for (let i = 0; i < unified.length; i++) {
        out.data[i * 4] = unified[i]
        out.data[i * 4 + 1] = unified[i]
        out.data[i * 4 + 2] = unified[i]
        out.data[i * 4 + 3] = 255
      }
      ctx.putImageData(out, 0, 0)
      setUnifiedUrl(canvas.toDataURL())
      setImageData({ unified, width: img1.width, height: img1.height, contour, angles })
    })
  }, [data])

  useEffect(() => {
    if (!imageData) return
    const { unified, width, height, contour, angles } = imageData
    if (contour && angles.length > 0) {
      const rotated = angles.map(angle => ({
        angle,
        url: createRotatedMaskedImage(unified, width, height, contour, angle, brightness)
      }))
      setRotatedImages(rotated)
    }
  }, [imageData, brightness])

  if (!data) return <div className="loading">Loading...</div>

  const images = data.download_links || {}
  const meta = { ...data }
  delete meta.download_links
  delete meta.Contour

  return (
    <div className="data-viewer-page">
      <header>
        <button onClick={() => navigate('/all_items')}>&larr; Back</button>
        <h1>Folder: {folderName} {label && `- ${label}`}</h1>
        <div className="label-edit">
          <label>Label:</label>
          <select value={label} onChange={e => saveLabel(e.target.value)}>
            <option value="">-- Select --</option>
            {availableLabels.map(l => <option key={l} value={l}>{l}</option>)}
          </select>
          {labelSaveStatus === 'saving' && <span className="save-status updating">updating...</span>}
          {labelSaveStatus === 'saved' && <span className="save-status done">✓</span>}
        </div>
      </header>

      <div className="content">
        <section className="images-section">
          <h2>Images</h2>
          <div className="zoom-controls">
            <button onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}>-</button>
            <span>{Math.round(zoom * 100)}%</span>
            <button onClick={() => setZoom(z => Math.min(4, z + 0.25))}>+</button>
          </div>
          <div className="images-grid">
            {Object.entries(images).map(([name, url]) => (
              <div key={name} className="image-card" onClick={() => setSelectedImg({ name, url })}>
                <img src={url} alt={name} style={{ transform: `scale(${zoom})` }} />
                <span>{name}</span>
              </div>
            ))}
          </div>
          {unifiedUrl && (
            <div className="unified-section">
              <h3 className="foldable-header" onClick={() => setUnifiedExpanded(!unifiedExpanded)}>
                {unifiedExpanded ? '▼' : '▶'} Unified (Mean Greyscale)
              </h3>
              {unifiedExpanded && (
                <div className="image-card unified-card" onClick={() => setSelectedImg({ name: 'Unified', url: unifiedUrl })}>
                  <img src={unifiedUrl} alt="Unified" style={{ transform: `scale(${zoom})` }} />
                </div>
              )}
            </div>
          )}
          {rotatedImages.length > 0 && (
            <div className="rotated-section">
              <h3>Rotated & Masked (by angle)</h3>
              <div className="brightness-control">
                <label>Brightness (clip top %): {brightness}%</label>
                <input type="range" min="50" max="100" value={brightness} onChange={e => setBrightness(Number(e.target.value))} />
              </div>
              <div className="rotated-grid">
                {rotatedImages.map(({ angle, url }) => (
                  <div key={angle} className="image-card" onClick={() => setSelectedImg({ name: `Angle ${angle}°`, url })}>
                    <img src={url} alt={`Angle ${angle}`} style={{ transform: `scale(${zoom})` }} />
                    <span>{angle.toFixed(2)}°</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        <section className="notes-section">
          <h2>Notes {saveStatus === 'saving' && <span className="save-status updating">updating...</span>}
            {saveStatus === 'saved' && <span className="save-status done">✓</span>}
          </h2>
          <textarea 
            value={notes} 
            onChange={handleNotesChange} 
            placeholder="Add notes here..."
            rows={4}
          />
        </section>

        <section className="audit-section">
          <h2 className="foldable-header" onClick={() => setAuditExpanded(!auditExpanded)}>
            {auditExpanded ? '▼' : '▶'} Audit Log
          </h2>
          {auditExpanded && data.audit && (
            <table className="audit-table">
              <thead>
                <tr><th>Time</th><th>User</th><th>Change</th></tr>
              </thead>
              <tbody>
                {Object.values(data.audit).sort((a, b) => b.timestamp.localeCompare(a.timestamp)).map((entry, i) => (
                  <tr key={i}>
                    <td>{new Date(entry.timestamp).toLocaleString()}</td>
                    <td>{entry.user}</td>
                    <td>{entry.from} → {entry.to}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {auditExpanded && !data.audit && <p>No audit entries yet.</p>}
        </section>

        <section className="meta-section">
          <h2 className="foldable-header" onClick={() => setMetaExpanded(!metaExpanded)}>
            {metaExpanded ? '▼' : '▶'} Metadata
          </h2>
          {metaExpanded && (
            <table className="meta-table">
              <tbody>
                {Object.entries(meta).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>

      {selectedImg && (
        <div className="modal" onClick={() => { setSelectedImg(null); setModalZoom(1); setPanOffset({ x: 0, y: 0 }) }}>
          <div className="modal-content" onClick={e => e.stopPropagation()} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
            <button className="close-btn" onClick={() => { setSelectedImg(null); setModalZoom(1); setPanOffset({ x: 0, y: 0 }) }}>&times;</button>
            <h3>{selectedImg.name} ({Math.round(modalZoom * 100)}%)</h3>
            <div className="modal-image-container" onWheel={e => {
                e.preventDefault()
                setPanOffset(p => ({ x: p.x - e.deltaX, y: p.y - e.deltaY }))
              }}>
              <img 
                src={selectedImg.url} 
                alt={selectedImg.name}
                style={{ 
                  transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${modalZoom})`, 
                  cursor: dragging ? 'grabbing' : shiftPressed ? 'zoom-out' : 'zoom-in' 
                }}
                onMouseDown={handleMouseDown}
                onClick={e => {
                  e.stopPropagation()
                  if (!didDrag) setModalZoom(z => shiftPressed ? Math.max(0.25, z / 1.5) : Math.min(10, z * 1.5))
                }}
                draggable={false}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
