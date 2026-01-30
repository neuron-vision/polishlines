import { signInWithPopup } from 'firebase/auth'
import { auth, googleProvider } from '../firebase.js'

export default function Login() {
  const handleGoogleLogin = () => signInWithPopup(auth, googleProvider)

  return (
    <div className="login-page">
      <div className="login-container">
        <h1>Polish Lines</h1>
        <button onClick={handleGoogleLogin} className="google-btn">
          Sign in with Google
        </button>
      </div>
    </div>
  )
}
