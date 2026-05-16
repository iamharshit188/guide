# Module 04 — Backend with Flask & FastAPI

> **Runnable code:** `src/04-backend/`
> ```bash
> python src/04-backend/app.py
> python src/04-backend/ml_serving.py
> python src/04-backend/middleware.py
> python src/04-backend/async_tasks.py
> ```

---

> **Python prerequisite:** This module uses Python, NumPy, and ML libraries throughout. If you need a foundation or refresher, visit the **Languages → Python** guide and read **Section 21 — Python for ML & AI** before starting.

## Prerequisites & Overview

**Prerequisites:** Python functions, HTTP basics (GET/POST), Modules 01–03.
**Estimated time:** 10–15 hours

**Install:**
```bash
pip install flask flask-jwt-extended pydantic requests
```

### Why This Module Matters

AI models alone are not products. A production AI application needs:
- APIs that external systems can call
- Authentication to protect model endpoints
- Request validation to prevent crashes
- Rate limiting to prevent abuse
- Async task queues for long-running inference
- Health/metrics endpoints for production monitoring

Without backend engineering, your model stays a local experiment. With it, you ship a product.

### Module Map

| Section | What You'll Build | Why It Matters |
|---------|------------------|---------------|
| HTTP & Flask basics | REST API with CRUD endpoints | Every API follows this pattern |
| App factory | Scalable Flask structure | Prevents import cycles, enables testing |
| Request validation | Pydantic schemas | Catch bad input before it reaches model |
| Auth (JWT) | Login + protected routes | Secure ML endpoints |
| ML serving | Load model, predict endpoint | Ship your trained model |
| Middleware | Rate limiting, logging, CORS | Production requirements |
| Async tasks | Celery queue | Long inference jobs |

### ML API Request Lifecycle

```
Client (browser / mobile / curl)
  │
  │  POST /predict  {"text": "hello world"}
  │  Authorization: Bearer eyJhbGciOiJIUzI1...
  ▼
┌──────────────────────────────────────────────────────┐
│                    Flask Server                       │
│                                                       │
│  1. Middleware        → check rate limit, log request │
│  2. Auth decorator    → verify JWT token              │
│  3. Request validator → Pydantic schema check         │
│  4. Route handler     → business logic                │
│  5. ML model          → load & run inference          │
│  6. Response builder  → format JSON                   │
└──────────────────────────────────────────────────────┘
  │
  │  200 OK  {"prediction": "positive", "confidence": 0.92}
  ▼
Client

Each step is a separate layer — failures are caught early before reaching the model.
```

---

# 1. How the Web Works

## Intuition

The web is a conversation between **clients** (browsers, apps) and **servers** (your backend).

```
User types URL → Browser sends HTTP Request → Your Flask server
                                               ↓
                                         Processes request
                                         (queries DB, runs ML model)
                                               ↓
                                         Returns HTTP Response
Browser renders HTML/JSON ←────────────────────┘
```

## HTTP Methods — The Vocabulary of APIs

| Method | Meaning | Example |
|--------|---------|---------|
| `GET` | Read data | Get user profile |
| `POST` | Create data | Create new prediction job |
| `PUT` | Replace data | Update entire record |
| `PATCH` | Partial update | Update one field |
| `DELETE` | Remove data | Delete experiment |

## HTTP Status Codes

| Code | Meaning | When to use |
|------|---------|------------|
| 200 OK | Success | GET, PUT, PATCH |
| 201 Created | Resource created | POST |
| 400 Bad Request | Client error | Invalid input |
| 401 Unauthorized | Not authenticated | No/bad token |
| 403 Forbidden | Not authorized | Token valid, no permission |
| 404 Not Found | Resource missing | Wrong ID |
| 422 Unprocessable | Validation error | Wrong data type/shape |
| 429 Too Many Requests | Rate limited | Exceeded quota |
| 500 Internal Server Error | Server bug | Something crashed |

---

# 2. Flask Basics

## Intuition

Flask is a minimal Python web framework. You define **routes** (URL patterns) and **handler functions** (what to do when that URL is hit).

## 2.1 Your First Flask API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Route: GET /health ───────────────────────────────────────
@app.route('/health')
def health():
    """Health check endpoint — tells load balancer if server is alive."""
    return jsonify({"status": "ok", "version": "1.0.0"}), 200

# ── Route: GET /hello/<name> (URL parameter) ─────────────────
@app.route('/hello/<string:name>')
def hello(name):
    return jsonify({"message": f"Hello, {name}!"}), 200

# ── Route: POST /echo (request body) ─────────────────────────
@app.route('/echo', methods=['POST'])
def echo():
    # request.get_json() parses the JSON body
    data = request.get_json()
    if data is None:
        return jsonify({"error": "JSON body required"}), 400
    return jsonify({"you_sent": data}), 200

# ── Route: GET /items with query parameters ───────────────────
@app.route('/items')
def list_items():
    # Query params: /items?category=books&limit=10
    category = request.args.get('category', 'all')
    limit    = int(request.args.get('limit', 10))

    # Fake data
    all_items = [
        {"id": 1, "name": "Python Book",  "category": "books"},
        {"id": 2, "name": "ML Course",    "category": "courses"},
        {"id": 3, "name": "NumPy Book",   "category": "books"},
    ]

    if category != 'all':
        all_items = [i for i in all_items if i['category'] == category]

    return jsonify({"items": all_items[:limit], "total": len(all_items)}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    # debug=True: auto-reload on code changes (NEVER use in production)
```

**Test it:**
```bash
# In terminal 1: start the server
python app.py

# In terminal 2: test with curl
curl http://localhost:5000/health
curl http://localhost:5000/hello/Alice
curl -X POST http://localhost:5000/echo \
     -H "Content-Type: application/json" \
     -d '{"message": "test", "value": 42}'
curl "http://localhost:5000/items?category=books&limit=2"
```

## 2.2 Full CRUD API (Students Example)

```python
from flask import Flask, request, jsonify
import json

app   = Flask(__name__)
store = {}  # in-memory storage (use a DB in production)
next_id = 1

# CREATE — POST /students
@app.route('/students', methods=['POST'])
def create_student():
    global next_id
    data = request.get_json()

    # Validate required fields
    required = ['name', 'age', 'major']
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    if not isinstance(data['age'], int) or data['age'] < 0:
        return jsonify({"error": "age must be a positive integer"}), 422

    student = {
        "id":    next_id,
        "name":  data['name'],
        "age":   data['age'],
        "major": data['major'],
        "gpa":   data.get('gpa', None),
    }
    store[next_id] = student
    next_id += 1

    return jsonify(student), 201  # 201 = Created

# READ ALL — GET /students
@app.route('/students', methods=['GET'])
def list_students():
    students = list(store.values())

    # Optional filter by major
    major = request.args.get('major')
    if major:
        students = [s for s in students if s['major'].lower() == major.lower()]

    return jsonify({"students": students, "count": len(students)}), 200

# READ ONE — GET /students/<id>
@app.route('/students/<int:student_id>', methods=['GET'])
def get_student(student_id):
    student = store.get(student_id)
    if not student:
        return jsonify({"error": f"Student {student_id} not found"}), 404
    return jsonify(student), 200

# UPDATE — PATCH /students/<id>
@app.route('/students/<int:student_id>', methods=['PATCH'])
def update_student(student_id):
    if student_id not in store:
        return jsonify({"error": "Not found"}), 404

    data = request.get_json()
    allowed = {'name', 'age', 'major', 'gpa'}
    updates = {k: v for k, v in data.items() if k in allowed}
    store[student_id].update(updates)

    return jsonify(store[student_id]), 200

# DELETE — DELETE /students/<id>
@app.route('/students/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    if student_id not in store:
        return jsonify({"error": "Not found"}), 404

    deleted = store.pop(student_id)
    return jsonify({"deleted": deleted}), 200

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500
```

---

# 3. App Factory Pattern

## Intuition

As your app grows, putting everything in one file causes problems:
- Circular imports (module A imports module B which imports module A)
- Hard to test (can't swap configurations)
- Hard to extend (can't add blueprints easily)

The **app factory** creates the Flask app inside a function, letting you pass different configs (test, dev, prod).

```python
# ── config.py ───────────────────────────────────────────────
class Config:
    SECRET_KEY      = 'dev-secret-key-change-in-production'
    TESTING         = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB upload limit

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
    DEBUG   = True

class ProductionConfig(Config):
    DEBUG = False
    # SECRET_KEY read from environment variable in real production

config_map = {
    'development': DevelopmentConfig,
    'testing':     TestingConfig,
    'production':  ProductionConfig,
}

# ── blueprints/students.py ───────────────────────────────────
from flask import Blueprint, jsonify, request

students_bp = Blueprint('students', __name__, url_prefix='/api/students')

@students_bp.route('/', methods=['GET'])
def list_students():
    return jsonify({"students": []}), 200

# ── blueprints/models.py ─────────────────────────────────────
from flask import Blueprint, jsonify

models_bp = Blueprint('models', __name__, url_prefix='/api/models')

@models_bp.route('/health', methods=['GET'])
def model_health():
    return jsonify({"status": "loaded", "version": "v2"}), 200

# ── create_app.py (the factory) ──────────────────────────────
from flask import Flask

def create_app(config_name='development'):
    app = Flask(__name__)
    app.config.from_object(config_map[config_name])

    # Register blueprints (modular route groups)
    app.register_blueprint(students_bp)
    app.register_blueprint(models_bp)

    # Register error handlers
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Server error"}), 500

    return app

# ── run.py ──────────────────────────────────────────────────
if __name__ == '__main__':
    app = create_app('development')
    app.run(port=5000)
```

**Benefits:**
- `create_app('testing')` — returns app configured for tests
- Each blueprint is a separate module — team members work independently
- Error handlers registered once, shared across all blueprints

---

# 4. Request Validation with Pydantic

## Intuition

Never trust user input. Before your ML model sees the data, validate it:
- Is it the right type? (int not string)
- Is it within a valid range? (age not -5)
- Are all required fields present?

Pydantic makes this declarative — you define the expected shape and it validates automatically.

```
Without validation:                With Pydantic validation:

  POST /predict                      POST /predict
  {"age": "twenty"}                  {"age": "twenty"}
        │                                   │
        ▼                                   ▼
  model.predict(age="twenty")        Pydantic: "age must be int"
  → ValueError: unsupported type     → 422 {"error": "age: int required"}
  → 500 Internal Server Error        ← clean error, no crash
```

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Request schemas ──────────────────────────────────────────
class PredictionRequest(BaseModel):
    text:         str               = Field(..., min_length=1, max_length=5000)
    model_name:   str               = Field(default="default")
    temperature:  float             = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens:   int               = Field(default=256, ge=1, le=4096)
    labels:       Optional[List[str]] = None

    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('text cannot be empty or whitespace only')
        return v.strip()

    @validator('model_name')
    def valid_model(cls, v):
        allowed = ['default', 'fast', 'accurate']
        if v not in allowed:
            raise ValueError(f"model_name must be one of {allowed}")
        return v

class BatchPredictionRequest(BaseModel):
    texts:      List[str] = Field(..., min_items=1, max_items=100)
    model_name: str       = Field(default="default")

# ── Helper to validate and parse ─────────────────────────────
def validate_request(schema_class):
    """Decorator-style helper for Pydantic validation."""
    def decorator(f):
        from functools import wraps
        @wraps(f)
        def wrapper(*args, **kwargs):
            data = request.get_json()
            if data is None:
                return jsonify({"error": "JSON body required"}), 400
            try:
                validated = schema_class(**data)
                return f(validated, *args, **kwargs)
            except Exception as e:
                return jsonify({"error": "Validation failed", "details": str(e)}), 422
        return wrapper
    return decorator

# ── Endpoints ───────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "JSON body required"}), 400

    try:
        req = PredictionRequest(**data)
    except Exception as e:
        return jsonify({"error": "Validation failed", "details": str(e)}), 422

    # Simulate model prediction
    result = {
        "input_length": len(req.text),
        "model":        req.model_name,
        "temperature":  req.temperature,
        "prediction":   "positive",
        "confidence":   0.87,
    }
    return jsonify(result), 200

# Test validation behavior
def test_validation():
    """Show what Pydantic catches."""
    test_cases = [
        {"text": "Hello world", "temperature": 0.7},             # valid
        {"text": "",            "temperature": 0.7},             # empty text
        {"text": "Hello",       "temperature": 5.0},             # temp out of range
        {"text": "Hello",       "model_name":  "unknown"},       # invalid model
        {"text": "Hello",       "max_tokens": 99999},            # max_tokens too high
    ]

    print("Pydantic validation results:")
    for case in test_cases:
        try:
            req = PredictionRequest(**case)
            print(f"  PASS: {case}")
        except Exception as e:
            print(f"  FAIL: {case}")
            print(f"        → {e}")

test_validation()
```

---

# 5. JWT Authentication

## Intuition

After login, the server issues a **JWT token** — a cryptographically signed JSON blob. On every subsequent request, the client sends this token. The server verifies the signature without hitting the database.

```
JWT Authentication Flow:

Step 1 — Login:
  Client                              Server
    │   POST /auth/login                │
    │   {"user": "alice", "pw": "..."}  │
    │──────────────────────────────────▶│
    │                                   │ verify password
    │                                   │ create JWT token
    │   200 OK {"token": "eyJ..."}      │
    │◀──────────────────────────────────│
    │ (client stores token)             │

Step 2 — Authenticated request:
  Client                              Server
    │   GET /api/predict                │
    │   Authorization: Bearer eyJ...    │
    │──────────────────────────────────▶│
    │                                   │ decode JWT header+payload
    │                                   │ verify signature (no DB hit!)
    │                                   │ check expiry
    │   200 OK {"result": ...}          │ run prediction
    │◀──────────────────────────────────│

JWT Structure:  header.payload.signature
                  │        │        │
              algorithm  user_id  HMAC sign
              (base64)   role     with secret key
                         exp

Key benefit: Stateless — the server stores no session. Any server instance
             can verify the token using just the shared secret key.
```

```
1. POST /auth/login {username, password}
         ↓ server verifies credentials
         ↓ server signs JWT with secret key
         ↓ returns {access_token: "eyJ..."}

2. GET /api/predict
   Header: Authorization: Bearer eyJ...
         ↓ server verifies signature
         ↓ if valid: process request
         ↓ if invalid: return 401
```

```python
import jwt
import hashlib
import datetime
from functools import wraps
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-store-in-env-var'

# ── Fake user store (use a real DB in production) ─────────────
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

USERS = {
    "alice": {
        "password_hash": hash_password("password123"),
        "role":          "user",
        "user_id":       1,
    },
    "admin": {
        "password_hash": hash_password("admin456"),
        "role":          "admin",
        "user_id":       2,
    },
}

# ── Token creation ───────────────────────────────────────────
def create_token(user_id, username, role):
    payload = {
        "user_id":  user_id,
        "username": username,
        "role":     role,
        "exp":      datetime.datetime.utcnow() + datetime.timedelta(hours=24),
        "iat":      datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

# ── Auth decorator ───────────────────────────────────────────
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header required: Bearer <token>"}), 401

        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user = payload  # make user info available in handler
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"error": f"Invalid token: {str(e)}"}), 401

        return f(*args, **kwargs)
    return decorated

def require_role(role):
    def decorator(f):
        @wraps(f)
        @require_auth
        def decorated(*args, **kwargs):
            if request.user.get('role') != role:
                return jsonify({"error": f"Requires {role} role"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator

# ── Auth endpoints ───────────────────────────────────────────
@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"error": "username and password required"}), 400

    user = USERS.get(username)
    if not user or user['password_hash'] != hash_password(password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_token(user['user_id'], username, user['role'])
    return jsonify({
        "access_token": token,
        "token_type":   "bearer",
        "expires_in":   86400,  # 24 hours in seconds
    }), 200

@app.route('/auth/me', methods=['GET'])
@require_auth
def me():
    return jsonify({
        "user_id":  request.user['user_id'],
        "username": request.user['username'],
        "role":     request.user['role'],
    }), 200

@app.route('/api/predict', methods=['POST'])
@require_auth
def predict():
    return jsonify({
        "prediction": "positive",
        "requested_by": request.user['username'],
    }), 200

@app.route('/admin/users', methods=['GET'])
@require_role('admin')
def list_users():
    return jsonify({"users": list(USERS.keys())}), 200

# Demonstrate the auth flow
def demo_auth_flow():
    """Simulate the full auth flow without running the server."""
    import io

    print("=== Auth Flow Demo ===\n")

    # 1. Create token for alice
    token = create_token(1, "alice", "user")
    print(f"1. Generated token for alice:")
    print(f"   {token[:50]}...")

    # 2. Decode token (what the server does)
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        print(f"\n2. Decoded token:")
        for k, v in payload.items():
            if k != 'exp' and k != 'iat':
                print(f"   {k}: {v}")

    except jwt.InvalidTokenError as e:
        print(f"Token invalid: {e}")

    # 3. Tampered token
    tampered = token[:-5] + "xxxxx"
    try:
        jwt.decode(tampered, app.config['SECRET_KEY'], algorithms=['HS256'])
        print("\n3. Tampered token accepted (SECURITY BUG!)")
    except jwt.InvalidTokenError:
        print("\n3. Tampered token rejected ✓")

demo_auth_flow()
```

---

# 6. Serving ML Models

## Intuition

A trained model is a function: input features → prediction. A model serving endpoint wraps this function with an HTTP API so any application (mobile app, web app, another service) can use it.

```python
import numpy as np
from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# ── Model wrapper ────────────────────────────────────────────
class ModelServer:
    def __init__(self):
        self.model      = None
        self.scaler     = None
        self.model_name = None
        self.version    = None

    def load(self, model_path=None):
        """Load a trained sklearn model (or use a dummy for demo)."""
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
            self.model  = saved['model']
            self.scaler = saved.get('scaler')
            print(f"Model loaded from {model_path}")
        else:
            # Dummy model for demo
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            rng = np.random.default_rng(42)
            X   = rng.randn(200, 4)
            y   = (X[:, 0] + X[:, 1] > 0).astype(int)

            self.scaler = StandardScaler()
            X_scaled    = self.scaler.fit_transform(X)
            self.model  = LogisticRegression()
            self.model.fit(X_scaled, y)
            print("Dummy model initialized")

        self.model_name = "logistic-v1"
        self.version    = "1.0.0"
        return self

    def predict(self, features):
        """Run prediction. Returns (class, probability, confidence)."""
        X = np.array(features).reshape(1, -1)
        if self.scaler:
            X = self.scaler.transform(X)

        pred_class = int(self.model.predict(X)[0])
        proba      = self.model.predict_proba(X)[0].tolist()
        confidence = max(proba)

        return {
            "class":      pred_class,
            "probabilities": proba,
            "confidence": confidence,
        }

    def batch_predict(self, features_list):
        """Batch prediction for multiple inputs."""
        X = np.array(features_list)
        if self.scaler:
            X = self.scaler.transform(X)

        classes     = self.model.predict(X).tolist()
        probas      = self.model.predict_proba(X).tolist()
        confidences = [max(p) for p in probas]

        return {
            "classes":       classes,
            "probabilities": probas,
            "confidences":   confidences,
            "batch_size":    len(features_list),
        }

# Initialize on startup
model_server = ModelServer()
model_server.load()

# ── Prediction endpoints ─────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"error": "JSON body with 'features' list required"}), 400

    features = data['features']
    if not isinstance(features, list):
        return jsonify({"error": "'features' must be a list of numbers"}), 422

    try:
        result = model_server.predict(features)
        result['model'] = model_server.model_name
        result['version'] = model_server.version
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict/batch', methods=['POST'])
def batch_predict():
    data = request.get_json()
    if not data or 'features_list' not in data:
        return jsonify({"error": "JSON body with 'features_list' required"}), 400

    features_list = data['features_list']
    if len(features_list) > 1000:
        return jsonify({"error": "Batch size limit is 1000"}), 400

    try:
        result = model_server.batch_predict(features_list)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify({
        "name":    model_server.model_name,
        "version": model_server.version,
        "type":    type(model_server.model).__name__,
        "status":  "loaded" if model_server.model else "not_loaded",
    }), 200

# Demo
def demo_predictions():
    sample_features = [1.2, -0.5, 0.8, 1.5]
    result = model_server.predict(sample_features)
    print(f"Single prediction: class={result['class']}, confidence={result['confidence']:.3f}")

    batch_features = [[rng.uniform(-2, 2) for _ in range(4)] for _ in range(5)]
    batch_result   = model_server.batch_predict(batch_features)
    print(f"Batch prediction: classes={batch_result['classes']}")

rng = np.random.default_rng(42)
demo_predictions()
```

---

# 7. Middleware — Rate Limiting & Logging

## Intuition

Middleware runs **before and after** every request. It handles cross-cutting concerns that apply to every endpoint:
- Log every request (observability)
- Rate limit abusive clients (protection)
- Add CORS headers (browser security)
- Authenticate every request (security)

```python
import time
import uuid
from collections import defaultdict
from flask import Flask, request, jsonify, g

app = Flask(__name__)

# ── Request Logger ────────────────────────────────────────────
@app.before_request
def before_request():
    """Runs before every request."""
    g.start_time  = time.time()
    g.request_id  = str(uuid.uuid4())[:8]  # short unique ID for this request
    g.user_ip     = request.remote_addr

    print(f"[{g.request_id}] → {request.method} {request.path} from {g.user_ip}")

@app.after_request
def after_request(response):
    """Runs after every request — add headers, log duration."""
    duration_ms = int((time.time() - g.start_time) * 1000)

    # Log the response
    print(f"[{g.request_id}] ← {response.status_code} ({duration_ms}ms)")

    # Add useful response headers
    response.headers['X-Request-ID']    = g.request_id
    response.headers['X-Response-Time'] = f"{duration_ms}ms"
    response.headers['X-Content-Type-Options'] = 'nosniff'

    # CORS headers (allow all origins for demo; restrict in production)
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'

    return response

# ── Rate Limiter ──────────────────────────────────────────────
class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_requests=100, window_seconds=60):
        self.max_requests     = max_requests
        self.window_seconds   = window_seconds
        self._requests        = defaultdict(list)  # ip → [timestamps]

    def is_allowed(self, client_id):
        now = time.time()
        window_start = now - self.window_seconds

        # Remove timestamps outside current window
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]

        # Check limit
        if len(self._requests[client_id]) >= self.max_requests:
            return False, self._requests[client_id][0] + self.window_seconds - now

        # Record this request
        self._requests[client_id].append(now)
        return True, 0

limiter = RateLimiter(max_requests=5, window_seconds=60)

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        client_id = request.remote_addr
        allowed, retry_after = limiter.is_allowed(client_id)

        if not allowed:
            return jsonify({
                "error": "Rate limit exceeded",
                "retry_after_seconds": int(retry_after),
            }), 429

        response = f(*args, **kwargs)
        return response
    return decorated

from functools import wraps

# Apply rate limiting to expensive endpoints
@app.route('/api/predict', methods=['POST'])
@rate_limit
def predict():
    return jsonify({"prediction": "result"}), 200

@app.route('/api/free', methods=['GET'])
def free_endpoint():
    return jsonify({"message": "no rate limit here"}), 200

# Demo rate limiting
def demo_rate_limit():
    print("\nRate limiter demo (5 requests per window):")
    for i in range(7):
        allowed, retry = limiter.is_allowed("127.0.0.1")
        status = "✓ ALLOWED" if allowed else f"✗ BLOCKED (retry in {retry:.1f}s)"
        print(f"  Request {i+1}: {status}")

demo_rate_limit()
```

---

# 8. RESTful API Design Principles

## Best Practices

```python
# ── REST design patterns ─────────────────────────────────────

# ✓ Resource-based URLs (nouns, not verbs)
# Good:
#   GET    /experiments        — list all experiments
#   POST   /experiments        — create experiment
#   GET    /experiments/42     — get experiment 42
#   PATCH  /experiments/42     — update experiment 42
#   DELETE /experiments/42     — delete experiment 42

# ✗ Bad (action in URL):
#   GET  /getExperiment?id=42
#   POST /createExperiment
#   POST /deleteExperiment/42

# ── API versioning ───────────────────────────────────────────
# URL prefix (most common):
#   /api/v1/experiments
#   /api/v2/experiments

# ── Consistent error format ──────────────────────────────────
def error_response(message, code, details=None):
    body = {"error": message, "code": code}
    if details:
        body["details"] = details
    return jsonify(body), code

# ── Pagination ───────────────────────────────────────────────
@app.route('/api/v1/experiments', methods=['GET'])
def list_experiments():
    # Always paginate list endpoints — never return unbounded results
    page     = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('per_page', 20)), 100)  # cap at 100
    offset   = (page - 1) * per_page

    # Fake data
    total        = 150
    experiments  = [{"id": i, "name": f"exp_{i}"} for i in range(offset, min(offset+per_page, total))]

    return jsonify({
        "data": experiments,
        "pagination": {
            "page":       page,
            "per_page":   per_page,
            "total":      total,
            "total_pages": -(-total // per_page),  # ceiling division
            "has_next":   page * per_page < total,
            "has_prev":   page > 1,
        }
    }), 200
```

---

# 9. Interview Q&A

## Q1: What is the difference between GET and POST?

`GET` is **idempotent** — calling it multiple times produces the same result. Used for reading data. Parameters go in the URL. `POST` creates or processes data; parameters go in the body. `GET` responses can be cached; `POST` responses generally cannot.

## Q2: What is a JWT and why is it stateless?

JWT (JSON Web Token) = Base64-encoded header + payload + HMAC signature. The server doesn't store any session state — it just verifies the signature cryptographically on each request. This makes it horizontally scalable (any server instance can verify any token). Trade-off: can't revoke tokens before expiry without a blacklist.

## Q3: When should you use blueprints?

Use blueprints when you have distinct feature areas (auth, predictions, admin) that each contain multiple routes. Benefits: team members work in separate files, routes are testable in isolation, URL prefixes are declared once per blueprint.

## Q4: Why validate requests before they reach business logic?

Validation at the boundary (HTTP layer) prevents bad data from propagating into the system, crashing ML models, corrupting databases, or causing security vulnerabilities. Pydantic makes this declarative: define the schema once, get validation + auto-documentation + serialization for free.

## Q5: How does rate limiting prevent abuse?

Rate limiting counts requests per client (by IP or API key) within a time window. If a client exceeds the limit, return `429 Too Many Requests`. This prevents:
- Brute-force auth attacks
- DoS via model inference
- Cost explosion from runaway clients
- Unfair resource consumption

---

# 10. Cheat Sheet

| Concept | Code | Use |
|---------|------|-----|
| Route with URL param | `@app.route('/users/<int:id>')` | Path variable |
| Get JSON body | `request.get_json()` | POST request body |
| Get query params | `request.args.get('key', default)` | URL ?key=value |
| Return JSON | `jsonify(dict)` | Standard JSON response |
| Error response | `jsonify({"error": "msg"}), 400` | Always include HTTP code |
| Before each request | `@app.before_request` | Logging, auth check |
| After each request | `@app.after_request` | Add headers, log timing |
| Blueprint | `Blueprint('name', __name__, url_prefix='/api')` | Modular routes |
| JWT encode | `jwt.encode(payload, secret, algorithm='HS256')` | Create token |
| JWT decode | `jwt.decode(token, secret, algorithms=['HS256'])` | Verify token |

---

# MINI-PROJECT — ML Model Serving API

**What you will build:** A production-ready Flask API that serves a trained sentiment classifier, with authentication, validation, rate limiting, and health monitoring. This is exactly what you'd build to deploy an ML model at a company.

---

## Step 1 — Train a Simple Classifier

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Training data: positive/negative reviews
train_texts = [
    "This product is amazing and works perfectly",
    "Absolutely love it, best purchase ever",
    "Great quality, highly recommend to everyone",
    "Excellent service, very satisfied",
    "Works as advertised, very happy",
    "Terrible product, broke immediately",
    "Worst purchase of my life, complete waste",
    "Very disappointed, nothing works correctly",
    "Horrible quality, returns required immediately",
    "Do not buy this, total garbage",
]
train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# Pipeline: TF-IDF → Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=500, ngram_range=(1,2))),
    ('clf',   LogisticRegression(max_iter=1000)),
])
pipeline.fit(train_texts, train_labels)

# Test
test_cases = ["I love this!", "Terrible quality", "It's okay"]
preds = pipeline.predict(test_cases)
probas = pipeline.predict_proba(test_cases)
print("Model test:")
for text, pred, proba in zip(test_cases, preds, probas):
    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    conf  = max(proba)
    print(f"  '{text}' → {label} ({conf:.2f})")
```

---

## Step 2 — Full Production API

```python
import time
import uuid
import hashlib
import datetime
from collections import defaultdict
from functools import wraps
from flask import Flask, request, jsonify, g

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-in-production'

# ── State ───────────────────────────────────────────────────
MODEL = pipeline
STATS = {"total_requests": 0, "total_predictions": 0, "errors": 0}
USERS = {"alice": hashlib.sha256(b"pass123").hexdigest()}

# ── Middleware ───────────────────────────────────────────────
limiter_store = defaultdict(list)

def rate_limit_check(max_req=10, window=60):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            client = request.remote_addr
            now    = time.time()
            limiter_store[client] = [t for t in limiter_store[client] if t > now - window]
            if len(limiter_store[client]) >= max_req:
                return jsonify({"error": "Rate limit exceeded"}), 429
            limiter_store[client].append(now)
            return f(*args, **kwargs)
        return wrapper
    return decorator

@app.before_request
def track_request():
    g.start_time = time.time()
    STATS['total_requests'] += 1

@app.after_request
def add_headers(response):
    response.headers['X-Response-Time'] = f"{int((time.time()-g.start_time)*1000)}ms"
    return response

# ── Auth ─────────────────────────────────────────────────────
def simple_token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('X-API-Key')
        if token not in USERS.values():
            return jsonify({"error": "Invalid API key"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/auth/token', methods=['POST'])
def get_token():
    data = request.get_json() or {}
    pw_hash = hashlib.sha256(data.get('password', '').encode()).hexdigest()
    username = data.get('username', '')
    if USERS.get(username) != pw_hash:
        return jsonify({"error": "Invalid credentials"}), 401
    return jsonify({"api_key": USERS[username], "expires_in": 86400}), 200

# ── Prediction endpoints ─────────────────────────────────────
@app.route('/predict', methods=['POST'])
@simple_token_required
@rate_limit_check(max_req=50, window=60)
def predict():
    data = request.get_json() or {}
    text = data.get('text', '').strip()

    if not text:
        STATS['errors'] += 1
        return jsonify({"error": "'text' field required"}), 400
    if len(text) > 5000:
        return jsonify({"error": "text must be under 5000 characters"}), 422

    try:
        pred    = int(MODEL.predict([text])[0])
        proba   = MODEL.predict_proba([text])[0].tolist()
        conf    = max(proba)
        label   = "positive" if pred == 1 else "negative"

        STATS['total_predictions'] += 1
        return jsonify({
            "text":        text[:100] + "..." if len(text) > 100 else text,
            "label":       label,
            "confidence":  round(conf, 4),
            "probabilities": {
                "negative": round(proba[0], 4),
                "positive": round(proba[1], 4),
            },
            "model_version": "logistic-tfidf-v1",
        }), 200

    except Exception as e:
        STATS['errors'] += 1
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
@simple_token_required
@rate_limit_check(max_req=10, window=60)
def batch_predict():
    data  = request.get_json() or {}
    texts = data.get('texts', [])

    if not texts or not isinstance(texts, list):
        return jsonify({"error": "'texts' list required"}), 400
    if len(texts) > 100:
        return jsonify({"error": "Max batch size is 100"}), 400

    preds  = MODEL.predict(texts).tolist()
    probas = MODEL.predict_proba(texts).tolist()

    results = [
        {
            "text":       t[:50],
            "label":      "positive" if p == 1 else "negative",
            "confidence": round(max(prob), 4),
        }
        for t, p, prob in zip(texts, preds, probas)
    ]

    STATS['total_predictions'] += len(texts)
    return jsonify({"results": results, "count": len(results)}), 200

# ── Health & Monitoring ──────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        "status":  "healthy",
        "model":   "loaded",
        "uptime":  "ok",
    }), 200

@app.route('/metrics')
@simple_token_required
def metrics():
    return jsonify({
        "total_requests":    STATS['total_requests'],
        "total_predictions": STATS['total_predictions'],
        "error_count":       STATS['errors'],
        "error_rate":        round(STATS['errors'] / max(STATS['total_requests'], 1), 4),
    }), 200

# ── Test the full API ─────────────────────────────────────────
def test_api():
    print("\n=== API Integration Test ===\n")
    api_key = USERS['alice']
    headers = {'X-API-Key': api_key, 'Content-Type': 'application/json'}

    import json

    # Single prediction
    with app.test_client() as client:
        res = client.post('/predict',
                          data=json.dumps({"text": "This is absolutely wonderful!"}),
                          headers=headers)
        data = json.loads(res.data)
        print(f"Single predict: {data['label']} ({data['confidence']})")

        # Batch prediction
        res2 = client.post('/predict/batch',
                           data=json.dumps({"texts": [
                               "Amazing product!",
                               "Terrible experience",
                               "Pretty average quality",
                           ]}),
                           headers=headers)
        data2 = json.loads(res2.data)
        print(f"\nBatch predict ({data2['count']} items):")
        for item in data2['results']:
            print(f"  '{item['text']}' → {item['label']} ({item['confidence']})")

        # Metrics
        res3 = client.get('/metrics', headers=headers)
        data3 = json.loads(res3.data)
        print(f"\nMetrics: {data3['total_predictions']} predictions, "
              f"{data3['error_rate']:.1%} error rate")

        # Auth failure
        res4 = client.post('/predict',
                           data=json.dumps({"text": "test"}),
                           headers={'X-API-Key': 'wrong-key'})
        print(f"\nBad API key: status={res4.status_code} (expected 401)")

if __name__ == '__main__':
    test_api()
    # app.run(debug=True, port=5000)  # uncomment to run server
```

---

## What This Project Demonstrated

| Module Concept | Where it appeared |
|---------------|------------------|
| REST endpoints | `/predict`, `/predict/batch`, `/health`, `/metrics` |
| Request validation | Text length, required fields, batch size limit |
| JWT/API key auth | `X-API-Key` header on every protected route |
| Rate limiting | Token bucket on predict endpoints |
| Middleware | `@before_request`, `@after_request` |
| ML serving | TF-IDF + LogisticRegression pipeline |
| Error handling | 400, 401, 422, 429, 500 responses |
| Metrics | Request counter, error rate |
| Batch endpoint | Up to 100 texts in one call |
| Blueprint structure | Showed in Section 3 (factory pattern) |

This is the exact structure of a production ML API — auth, rate limiting, validation, serving, and observability in one coherent system.

---

*Next: [Module 05 — Deep Learning](05-deep-learning.md)*
