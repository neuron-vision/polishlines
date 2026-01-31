import { useState, useEffect } from 'react'
import { ref as dbRef, get } from 'firebase/database'
import { doc, setDoc, getDoc, collection } from 'firebase/firestore'
import { db, fs } from '../firebase.js'
import * as ort from 'onnxruntime-web'
import toast, { Toaster } from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'

// Configure WASM paths
ort.env.wasm.wasmPaths = window.location.origin + '/onnx-bin/'
ort.env.wasm.numThreads = 1

// --- Helper Functions (Copied from DataViewer for self-containment) ---
function loadImageData(url) {
  return new Promise((resolve, reject) => {
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
    img.onerror = reject
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
  return [cos * (x - cx) - sin * (y - cy) + cx, sin * (x - cx) + cos * (y - cy) + cy]
}

function getBbox(points) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of points) {
    minX = Math.min(minX, x); minY = Math.min(minY, y)
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y)
  }
  return { x: Math.floor(minX), y: Math.floor(minY), w: Math.ceil(maxX - minX), h: Math.ceil(maxY - minY) }
}

function createRotatedMaskedImage(greyData, width, height, contour, angle) {
  const cx = width / 2, cy = height / 2
  const rotatedContour = contour.map(([x, y]) => rotatePoint(x, y, cx, cy, angle))
  const bbox = getBbox(rotatedContour)
  const offsetContour = rotatedContour.map(([x, y]) => [x - bbox.x, y - bbox.y])
  const outCanvas = document.createElement('canvas')
  outCanvas.width = bbox.w; outCanvas.height = bbox.h
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
        outData.data[idx] = val; outData.data[idx+1] = val; outData.data[idx+2] = val; outData.data[idx+3] = 255
      }
    }
  }
  outCtx.putImageData(outData, 0, 0)
  const maskCanvas = document.createElement('canvas')
  maskCanvas.width = bbox.w; maskCanvas.height = bbox.h
  const maskCtx = maskCanvas.getContext('2d')
  maskCtx.fillStyle = 'white'; maskCtx.beginPath()
  maskCtx.moveTo(offsetContour[0][0], offsetContour[0][1])
  for (let i = 1; i < offsetContour.length; i++) maskCtx.lineTo(offsetContour[i][0], offsetContour[i][1])
  maskCtx.closePath(); maskCtx.fill()
  const maskData = maskCtx.getImageData(0, 0, bbox.w, bbox.h)
  for (let i = 0; i < outData.data.length; i += 4) {
    if (maskData.data[i] === 0) { outData.data[i] = 0; outData.data[i+1] = 0; outData.data[i+2] = 0 }
  }
  outCtx.putImageData(outData, 0, 0)
  return outCanvas
}

function preprocessForModel(canvas) {
  const resizeCanvas = document.createElement('canvas')
  resizeCanvas.width = 224; resizeCanvas.height = 224
  const ctx = resizeCanvas.getContext('2d')
  ctx.drawImage(canvas, 0, 0, 224, 224)
  const { data } = ctx.getImageData(0, 0, 224, 224)
  const floatData = new Float32Array(224 * 224 * 3)
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
  for (let i = 0; i < data.length / 4; i++) {
    floatData[i] = ((data[i * 4] / 255) - mean[0]) / std[0]
    floatData[i + 224 * 224] = ((data[i * 4 + 1] / 255) - mean[1]) / std[1]
    floatData[i + 224 * 224 * 2] = ((data[i * 4 + 2] / 255) - mean[2]) / std[2]
  }
  return floatData
}

// --- Component ---
export default function BulkCompile() {
  const [status, setStatus] = useState('Idle')
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [isProcessing, setIsProcessing] = useState(false)
  const navigate = useNavigate()

  const startProcessing = async () => {
    setIsProcessing(true)
    setStatus('Fetching metadata...')
    try {
      // 1. Get Model
      const modelSnap = await get(dbRef(db, 'metadata/embedders/latest'))
      if (!modelSnap.exists()) throw new Error('No embedder metadata found')
      const modelMeta = modelSnap.val()
      setStatus(`Downloading model: ${modelMeta.name}...`)
      const res = await fetch(modelMeta.url)
      const arrayBuffer = await res.arrayBuffer()
      const session = await ort.InferenceSession.create(arrayBuffer, { executionProviders: ['wasm'] })
      
      // 2. Get Folders
      setStatus('Fetching folders...')
      const foldersSnap = await get(dbRef(db, 'folders'))
      const allFolders = foldersSnap.exists() ? foldersSnap.val() : {}
      const foldersList = Object.entries(allFolders).filter(([_, d]) => d.download_links?.['-1'] && d.download_links?.['0'] && d.Contour)
      setProgress({ current: 0, total: foldersList.length })

      // 3. Process each folder
      for (let i = 0; i < foldersList.length; i++) {
        const [folderName, data] = foldersList[i]
        setStatus(`Processing folder ${i + 1}/${foldersList.length}: ${folderName}`)
        
        const contour = typeof data.Contour === 'string' ? JSON.parse(data.Contour) : data.Contour
        const angles = data['Chosen Facet PD'] || []
        if (angles.length === 0) continue

        // Check Firestore for existing folder subcollection to see which angles are missing
        const anglesToCalculate = []
        for (const angle of angles) {
          const angleKey = angle.toString().replace('.', '_')
          const angleRef = doc(fs, 'embeddings', modelMeta.name, folderName, angleKey)
          const angleDoc = await getDoc(angleRef)
          if (!angleDoc.exists()) {
            anglesToCalculate.push({ angle, angleRef })
          }
        }

        if (anglesToCalculate.length === 0) {
          setProgress(prev => ({ ...prev, current: i + 1 }))
          continue
        }

        // Load images
        const [img1, img2] = await Promise.all([
          loadImageData(data.download_links['-1']),
          loadImageData(data.download_links['0'])
        ])
        const grey1 = toGreyscale(img1.data), grey2 = toGreyscale(img2.data)
        const unified = new Float32Array(grey1.length)
        for (let j = 0; j < grey1.length; j++) unified[j] = (grey1[j] + grey2[j]) / 2

        for (const { angle, angleRef } of anglesToCalculate) {
          const canvas = createRotatedMaskedImage(unified, img1.width, img1.height, contour, angle)
          const inputData = preprocessForModel(canvas)
          const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 224, 224])
          const outputs = await session.run({ [session.inputNames[0]]: inputTensor })
          const vector = Array.from(outputs[session.outputNames[0]].data)
          
          // Store to Firestore under embeddings/embedder_name/folder_name/angle
          // Using angle as document ID, ensure it's a string
          await setDoc(angleRef, { vector, timestamp: new Date().toISOString() })
        }
        setProgress(prev => ({ ...prev, current: i + 1 }))
      }
      setStatus('Done!')
      toast.success('Bulk processing complete!')
    } catch (err) {
      setStatus(`Error: ${err.message}`)
      toast.error(err.message)
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="train-page">
      <Toaster />
      <header>
        <button onClick={() => navigate('/all_items')}>&larr; Back</button>
        <h1>Bulk Compile Embeddings</h1>
      </header>
      <main className="content">
        <div className="embedder-card">
          <p><strong>Status:</strong> {status}</p>
          {progress.total > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <progress value={progress.current} max={progress.total} style={{ width: '100%' }} />
              <p style={{ textAlign: 'center' }}>{progress.current} / {progress.total} folders</p>
            </div>
          )}
          {!isProcessing && (
            <button className="primary-btn" onClick={startProcessing} style={{ marginTop: '1rem', width: '100%', background: '#27ae60' }}>
              Start Bulk Processing
            </button>
          )}
        </div>
      </main>
    </div>
  )
}
