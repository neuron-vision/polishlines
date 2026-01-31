import { initializeApp } from 'firebase/app'
import { getAuth, GoogleAuthProvider } from 'firebase/auth'
import { getDatabase } from 'firebase/database'
import { getFirestore } from 'firebase/firestore'

const firebaseConfig = {
  apiKey: "AIzaSyDNgDBr9sfn-HsyMRCqMdOLwCoxyI-hx10",
  authDomain: "polish-lines.firebaseapp.com",
  databaseURL: "https://polish-lines-default-rtdb.firebaseio.com",
  projectId: "polish-lines",
  storageBucket: "polish-lines.firebasestorage.app",
  messagingSenderId: "18632528423",
  appId: "1:18632528423:web:0903bd0bbd4a2b7f0bf402"
}

const app = initializeApp(firebaseConfig)
export const auth = getAuth(app)
export const googleProvider = new GoogleAuthProvider()
export const db = getDatabase(app)
export const fs = getFirestore(app)
