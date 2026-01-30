import { useState, useEffect } from 'react'
import { ref, get } from 'firebase/database'
import { db } from '../firebase.js'
import * as ort from 'onnxruntime-web'
import toast from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'

// Configure WASM paths to look in the onnx-bin folder
ort.env.wasm.wasmPaths = window.location.origin + '/onnx-bin/'
ort.env.wasm.numThreads = 1 // Use single thread for stability

export default function Train() {
  const [latestEmbedder, setLatestEmbedder] = useState(null)
  const [loading, setLoading] = useState(true)
  const [sanityResult, setSanityResult] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    get(ref(db, 'metadata/embedders/latest')).then(snap => {
      if (snap.exists()) setLatestEmbedder(snap.val())
      setLoading(false)
    }).catch(err => {
      toast.error(`Failed to fetch metadata: ${err.message}`)
      setLoading(false)
    })
  }, [])

  const runSanity = async () => {
    if (!latestEmbedder?.url) return
    const t = toast.loading('Downloading model and running sanity check...')
    try {
      setSanityResult(null)
      // 1. Download model
      const res = await fetch(latestEmbedder.url)
      if (!res.ok) throw new Error('Download failed')
      const arrayBuffer = await res.arrayBuffer()
      toast.success('Model downloaded', { id: t })
      
      // 2. Load and Infer
      const t2 = toast.loading('Initializing ONNX session (CPU)...', { id: t })
      const session = await ort.InferenceSession.create(arrayBuffer, {
        executionProviders: ['wasm']
      })
      
      const inputName = session.inputNames[0]
      const inputShape = [1, 3, 224, 224]
      const inputSize = inputShape.reduce((a, b) => a * b)
      const inputData = new Float32Array(inputSize).fill(0.5)
      const inputTensor = new ort.Tensor('float32', inputData, inputShape)
      
      const t3 = toast.loading('Running inference...', { id: t })
      const outputs = await session.run({ [inputName]: inputTensor })
      const output = outputs[session.outputNames[0]]
      
      setSanityResult({
        shape: output.dims,
        size: output.data.length,
        mean: Array.from(output.data.slice(0, 10)).reduce((a, b) => a + b, 0) / 10 // small sample mean
      })
      toast.success('Sanity check passed!', { id: t })
    } catch (err) {
      toast.error(`Sanity check failed: ${err.message}`, { id: t })
    }
  }

  return (
    <div className="train-page">
      <header>
        <button onClick={() => navigate('/all_items')}>&larr; Back</button>
        <h1>Train & Embedder Management</h1>
      </header>
      
      <main className="content">
        <section className="embedder-section">
          <h2>Latest Embedder</h2>
          {loading ? <p>Loading metadata...</p> : latestEmbedder ? (
            <div className="embedder-card">
              <p><strong>Name:</strong> {latestEmbedder.name}</p>
              <p><strong>Last Updated:</strong> {new Date(latestEmbedder.timestamp).toLocaleString()}</p>
              <div className="actions">
                <button className="primary-btn" onClick={runSanity}>Download & Sanity Check</button>
                <a href={latestEmbedder.url} className="secondary-link" download={`embedder_${latestEmbedder.name}.onnx`}>Direct Download Link</a>
              </div>
            </div>
          ) : <p>No embedder metadata found.</p>}

          {sanityResult && (
            <div className="sanity-results">
              <h3>Sanity Results</h3>
              <p><strong>Output Shape:</strong> {JSON.stringify(sanityResult.shape)}</p>
              <p><strong>Vector Size:</strong> {sanityResult.size}</p>
              <p><strong>Sample Mean (first 10):</strong> {sanityResult.mean.toFixed(4)}</p>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
