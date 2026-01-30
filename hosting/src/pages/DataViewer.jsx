import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ref, get } from 'firebase/database'
import { db } from '../firebase.js'

export default function DataViewer() {
  const { folderName } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [zoom, setZoom] = useState(1)
  const [selectedImg, setSelectedImg] = useState(null)

  useEffect(() => {
    get(ref(db, `folders/${folderName}`)).then(snap => {
      if (snap.exists()) setData(snap.val())
    })
  }, [folderName])

  if (!data) return <div className="loading">Loading...</div>

  const images = data.download_links || {}
  const meta = { ...data }
  delete meta.download_links
  delete meta.Contour

  return (
    <div className="data-viewer-page">
      <header>
        <button onClick={() => navigate('/all_items')}>&larr; Back</button>
        <h1>Folder: {folderName}</h1>
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
        </section>

        <section className="meta-section">
          <h2>Metadata</h2>
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
        </section>
      </div>

      {selectedImg && (
        <div className="modal" onClick={() => setSelectedImg(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="close-btn" onClick={() => setSelectedImg(null)}>&times;</button>
            <h3>{selectedImg.name}</h3>
            <div className="modal-image-container">
              <img src={selectedImg.url} alt={selectedImg.name} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
