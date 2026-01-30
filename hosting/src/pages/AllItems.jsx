import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { ref, get } from 'firebase/database'
import { signOut } from 'firebase/auth'
import { db, auth } from '../firebase.js'

export default function AllItems() {
  const [folders, setFolders] = useState([])
  const [filter, setFilter] = useState('')
  const [labelFilter, setLabelFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const [sortBy, setSortBy] = useState('name')
  const [sortDir, setSortDir] = useState('asc')
  const navigate = useNavigate()

  const toggleSort = col => {
    if (sortBy === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortBy(col); setSortDir('asc') }
  }
  const sortIcon = col => sortBy === col ? (sortDir === 'asc' ? ' ▲' : ' ▼') : ''

  useEffect(() => {
    get(ref(db, 'folders')).then(snap => {
      if (snap.exists()) {
        const data = snap.val()
        setFolders(Object.keys(data).map(k => ({ name: k, ...data[k] })))
      }
      setLoading(false)
    })
  }, [])

  const getLabel = f => f["User Input"] || f.label || ''
  const getNotes = f => f.notes || ''
  const getLastUpdate = f => f.last_update || ''
  const formatDate = d => d ? new Date(d).toLocaleString() : '-'
  const uniqueLabels = [...new Set(folders.map(getLabel))].filter(Boolean).sort()
  const filtered = folders
    .filter(f => {
      const matchesName = f.name.includes(filter)
      const matchesLabel = !labelFilter || getLabel(f) === labelFilter
      return matchesName && matchesLabel
    })
    .sort((a, b) => {
      let va, vb
      if (sortBy === 'name') { va = a.name; vb = b.name }
      else if (sortBy === 'label') { va = getLabel(a); vb = getLabel(b) }
      else if (sortBy === 'notes') { va = getNotes(a); vb = getNotes(b) }
      else if (sortBy === 'last_update') { va = getLastUpdate(a); vb = getLastUpdate(b) }
      else { va = a.name; vb = b.name }
      const cmp = va.localeCompare(vb)
      return sortDir === 'asc' ? cmp : -cmp
    })

  return (
    <div className="all-items-page">
      <header>
        <h1>All Items</h1>
        <button onClick={() => signOut(auth)}>Logout</button>
      </header>
      <div className="filters-row">
        <input
          type="text"
          placeholder="Filter by folder name..."
          value={filter}
          onChange={e => setFilter(e.target.value)}
          className="filter-input"
        />
        <select value={labelFilter} onChange={e => setLabelFilter(e.target.value)} className="label-filter">
          <option value="">All Labels</option>
          {uniqueLabels.map(l => <option key={l} value={l}>{l}</option>)}
        </select>
      </div>
      {loading ? <p>Loading...</p> : (
        <table className="items-table">
          <thead>
            <tr>
              <th className="sortable" onClick={() => toggleSort('name')}>Folder{sortIcon('name')}</th>
              <th className="sortable" onClick={() => toggleSort('label')}>Label{sortIcon('label')}</th>
              <th>Stone Name</th>
              <th className="sortable" onClick={() => toggleSort('notes')}>Notes{sortIcon('notes')}</th>
              <th className="sortable" onClick={() => toggleSort('last_update')}>Last Update{sortIcon('last_update')}</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(f => (
              <tr key={f.name} onClick={() => navigate(`/data_viewer/${f.name}`)}>
                <td>{f.name}</td>
                <td>{f["User Input"] || f.label || '-'}</td>
                <td>{f["Stone Name"] || '-'}</td>
                <td className="notes-cell">{f.notes || '-'}</td>
                <td>{formatDate(f.last_update)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
