import { useState, useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { onAuthStateChanged } from 'firebase/auth'
import { auth } from './firebase.js'
import Login from './pages/Login.jsx'
import AllItems from './pages/AllItems.jsx'
import DataViewer from './pages/DataViewer.jsx'

export default function App() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    return onAuthStateChanged(auth, (u) => {
      setUser(u)
      setLoading(false)
    })
  }, [])

  if (loading) return <div className="loading">Loading...</div>

  return (
    <Routes>
      <Route path="/login" element={user ? <Navigate to="/all_items" /> : <Login />} />
      <Route path="/all_items" element={user ? <AllItems /> : <Navigate to="/login" />} />
      <Route path="/data_viewer/:folderName" element={user ? <DataViewer /> : <Navigate to="/login" />} />
      <Route path="*" element={<Navigate to={user ? "/all_items" : "/login"} />} />
    </Routes>
  )
}
