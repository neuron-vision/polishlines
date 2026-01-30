import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { ref, get } from 'firebase/database'
import { signOut } from 'firebase/auth'
import { db, auth } from '../firebase.js'

export default function AllItems() {
  const [folders, setFolders] = useState([])
  const [filter, setFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    get(ref(db, 'folders')).then(snap => {
      if (snap.exists()) {
        const data = snap.val()
        setFolders(Object.keys(data).map(k => ({ name: k, ...data[k] })))
      }
      setLoading(false)
    })
  }, [])

  const filtered = folders.filter(f => f.name.includes(filter))

  return (
    <div className="all-items-page">
      <header>
        <h1>All Items</h1>
        <button onClick={() => signOut(auth)}>Logout</button>
      </header>
      <input
        type="text"
        placeholder="Filter by folder name..."
        value={filter}
        onChange={e => setFilter(e.target.value)}
        className="filter-input"
      />
      {loading ? <p>Loading...</p> : (
        <table className="items-table">
          <thead>
            <tr>
              <th>Folder</th>
              <th>Label</th>
              <th>Stone Name</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(f => (
              <tr key={f.name} onClick={() => navigate(`/data_viewer/${f.name}`)}>
                <td>{f.name}</td>
                <td>{f["User Input"] || f.label || '-'}</td>
                <td>{f["Stone Name"] || '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
