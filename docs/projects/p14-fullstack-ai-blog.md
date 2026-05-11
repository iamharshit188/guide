# Project 14: Full-Stack AI Blog

**Difficulty:** Advanced  
**Module:** 14 (Frontend + Backend)

## 📌 The Challenge
Develop a production-grade full-stack web application featuring a decoupled, zero-trust architecture. You will build a Vite React UI, a Flask REST API ML Gateway, Firebase Google Authentication, and a PostgreSQL database on Supabase.

This guide bridges the architectural theory with concrete code outlines so a beginner can start building immediately.

---

## 📖 The Architecture & Code Approach

### 1. Initialize the React App (Vite + Tailwind)
Start by scaffolding the frontend environment.
```bash
npm create vite@latest ai-blog -- --template react
cd ai-blog
npm install -D tailwindcss postcss autoprefixer firebase
npx tailwindcss init -p
```
Configure `tailwind.config.js` to scan your React files:
```javascript
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: {} },
  plugins: [],
}
```

### 2. Firebase Client Authentication (React)
Set up Google OAuth. You will extract a JWT (JSON Web Token) containing a cryptographic signature payload.
```javascript
// src/firebase.js
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup } from "firebase/auth";

const app = initializeApp({ /* Your Firebase Config config */ });
export const auth = getAuth(app);
export const provider = new GoogleAuthProvider();

// Trigger this function on a button click
export const loginWithGoogle = async () => {
    const result = await signInWithPopup(auth, provider);
    const token = await result.user.getIdToken(); 
    return { user: result.user, token };
};
```

### 3. API Gateway & Middleware (Flask)
The Flask backend verifies the JWT using the `firebase-admin` SDK, rejecting unauthenticated traffic.
```python
# app.py
from flask import Flask, request, jsonify
from firebase_admin import auth, credentials, initialize_app

app = Flask(__name__)
initialize_app(credentials.Certificate('path/to/firebase-adminsdk.json'))

def require_auth(f):
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization').split('Bearer ')[1]
        try:
            user = auth.verify_id_token(token) # Cryptographically verified
            return f(user, *args, **kwargs)
        except Exception:
            return jsonify({'error': 'Unauthorized'}), 401
    wrapper.__name__ = f.__name__
    return wrapper

@app.route('/api/generate', methods=['POST'])
@require_auth
def generate_blog(user):
    prompt = request.json.get('prompt')
    # Use LLM here (e.g., OpenAI API)
    generated_text = f"Synthetic output for: {prompt}"
    return jsonify({'content': generated_text, 'uid': user['uid']})
```

### 4. Database Operations (Supabase)
Save the generated blog post to a managed Postgres database.
```python
# database.py
from supabase import create_client

supabase = create_client("YOUR_SUPABASE_URL", "YOUR_SUPABASE_KEY")

def save_post(uid, title, content):
    data, count = supabase.table('posts').insert({
        'user_id': uid,
        'title': title,
        'content': content
    }).execute()
    return data
```

### 5. Client Hydration (React UI)
Back on the frontend, execute API calls and render the Tailwind UI components.
```jsx
// src/App.jsx
import { useState, useEffect } from 'react';

export function BlogDashboard({ userToken }) {
  const [posts, setPosts] = useState([]);

  useEffect(() => {
    // Fetch posts using the secure token
    fetch('/api/posts', {
      headers: { 'Authorization': `Bearer ${userToken}` }
    })
    .then(res => res.json())
    .then(data => setPosts(data));
  }, [userToken]);

  return (
    <div className="flex flex-col items-center p-8 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">Your AI Blogs</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-5xl">
        {posts.map(post => (
          <div key={post.id} className="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
            <h2 className="text-xl font-bold text-gray-900">{post.title}</h2>
            <p className="text-gray-600 mt-2">{post.content}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
```

## ✅ Implementation Checkpoints
- [ ] Initialize React + Tailwind.
- [ ] Connect Firebase and extract the OAuth `getIdToken()`.
- [ ] Stand up the Flask `@require_auth` middleware.
- [ ] Execute an LLM API call from Flask strictly after token verification.
- [ ] Insert the result into the Supabase Postgres database.
- [ ] Hydrate a Tailwind dashboard grid in React by fetching the saved datastore mappings.

🚀 Deployment Ready
