import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'ignore-onnx-wasm',
      resolveId(id) {
        if (id.includes('onnx-bin')) return { id, external: true }
      }
    }
  ],
  assetsInclude: ['**/*.wasm'],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
