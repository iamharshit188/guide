# Project 14: Full-Stack AI Blog

**Difficulty:** Advanced  
**Module:** 14 (Frontend + Backend)

## 📌 The Challenge
Build a production-grade full-stack web application. It should allow users to log in securely (Google Auth), type a short prompt, generate a polished AI blog post via a Flask backend, save that post permanently to Supabase, and display all their posts in a responsive React/Tailwind dashboard.

## 📖 The Approach

This project strictly emphasizes architecture connecting decoupled systems, rather than just the code.

1. **Frontend Bootstrapping:** 
   Initialize a React app (e.g. using Vite: `npm create vite@latest`). Configure Tailwind CSS inside it to handle styling purely via utility classes.
2. **Auth Integration (Firebase/Supabase):** 
   Let the client SDK handle the popup for Google Login. Retrieve the JWT ID token. Pass this token in the headers of all `fetch()` calls to your backend to ensure security. 
3. **Backend Middleware (Flask):** 
   Set up Flask with `flask-cors`. Create a `@require_auth` decorator that uses `firebase-admin` to tear apart the JWT, verify the signature, and attach the `user_uid` to the request context. Reject invalid tokens with `401 Unauthorized`.
4. **Database Operations (Supabase):** 
   When the Flask endpoint generates text from the ML model, initialize the `supabase-py` client and execute an `.insert()` to a `posts` table holding `(user_id, title, content, created_at)`.
5. **State Rendering:** 
   Back in React, use `useEffect` to fetch all posts for the logged-in user upon load, map over the array, and render Tailwind-styled "cards" for each blog post.

## ✅ Checkpoints
- [ ] Connect Firebase SDK to React app with a "Sign In With Google" button.
- [ ] Create a secure Flask endpoint that decodes the Bearer token.
- [ ] Call the selected LLM API inside Flask to generate blog content.
- [ ] Stand up a Supabase Postgres table and insert records using the extracted user ID.
- [ ] Create a Tailwind Grid to map and display the fetched database rows beautifully.
