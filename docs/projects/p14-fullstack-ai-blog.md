# Project 14: Full-Stack AI Blog

**Difficulty:** Advanced  
**Module:** 14 (Frontend + Backend)

## The Challenge
Develop a production-grade full-stack web application featuring a decoupled, zero-trust architecture. The system consists of a Vite/React Single Page Application (SPA), a scalable Flask REST API acting as the ML inference gateway, Firebase OAuth 2.0 authentication, and a PostgreSQL database hosted on Supabase strictly enforcing Row Level Security.

## The Architecture Map

### 1. Client-Side Authentication (Firebase)
The React application integrates the Firebase Client SDK to handle identity safely.
- The user authenticates via Google OAuth Provider popup constraint.
- Firebase securely issues a JSON Web Token (JWT) representing the session.
- The JWT is stored safely in runtime memory.
- The JWT structure explicitly contains three cryptographic parts: Header, Payload (claims including `uid`, `exp`), and Signature (HMAC-SHA256).

### 2. API Gateway & Middleware (Flask)
The Flask backend acts as a stateless broker. 
- Implements CORS (Cross-Origin Resource Sharing) with strict IP origin limits to prevent CSRF abuse.
- A custom `@require_auth` decorator intercepts incoming routes, strips the `Authorization: Bearer <token>` header, and uses the `firebase-admin` SDK. This cryptographically verifies the JWT signature against Google's rotating public keys.
- Invalid or expired tokens immediately reject the request returning a `401 Unauthorized` HTTP status, directly protecting expensive LLM inference compute from unauthorized exploitation.

### 3. Database Schema & Operations (Supabase)
The persistent data layer runs on managed PostgreSQL.
- **Table `users`:** `id` (UUID, Primary Key), `email`, `created_at`.
- **Table `posts`:** `id` (UUID), `user_id` (Foreign Key -> `users.id`), `title`, `content` (Text), `embedding` (Optional: vector type placeholder for semantic search integration), `created_at`.
- The Flask backend utilizes the native `supabase-py` client abstraction to perform synchronous `INSERT` and `SELECT` RPC operations strictly after the LLM generates the blog content, mapping the user's `uid` to the database relation.

## Implementation Checkpoints

- [ ] **Phase 1: React Initialization.** Scaffold a Vite React app. Install Tailwind CSS and configure the PostCSS pipeline `tailwind.config.js`. Build a clean, responsive grid layout for the blog dashboard interface.
- [ ] **Phase 2: Auth Provider Architecture.** Implement a React Context Provider to wrap the top-level application. Expose the `currentUser` state payload and login/logout methods via the initialized Firebase SDK.
- [ ] **Phase 3: Secure API Layer.** Stand up the Flask application. Write the token verification middleware using the `firebase-admin` certificate logic.
- [ ] **Phase 4: LLM Generation.** Expose a secure `POST /api/generate` endpoint. Extract the verified `uid` context and prompt payload. Interface with your selected model engine (OpenAI/Ollama) to synthetically generate markdown blog content.
- [ ] **Phase 5: Persistent Relational Storage.** Connect Flask to the Supabase endpoint. Write the generated generation output strictly to the `posts` table alongside the matched `user_id`.
- [ ] **Phase 6: Client Hydration.** On the React frontend, utilize `useEffect` to trigger a `GET /api/posts` request upon mount. Parse the incoming JSON array and iteratively `.map()` the data into conditionally styled, Tailwind card components.

🚀 Deployment Ready
