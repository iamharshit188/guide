# Module 04 — Backend with Flask

> **Run:**
> ```bash
> cd src/04-backend
> python app.py          # app factory + blueprints demo
> python ml_serving.py   # ML model serving endpoint
> python middleware.py   # auth, rate limiting, request logging
> python async_tasks.py  # Celery + Redis task queue patterns
> ```

```bash
cd src/04-backend && python app.py
cd src/04-backend && python ml_serving.py
cd src/04-backend && python middleware.py
cd src/04-backend && python async_tasks.py
```

---

## Prerequisites & Overview

**Prerequisites:** Python functions, decorators, and basic HTTP (GET/POST/status codes). No prior Flask or web framework experience needed.
**Estimated time:** 8–12 hours (4 scripts, each self-contained and runnable without a live server)

### Why This Module Matters
ML models are useless without a way to call them. Backend engineering is the bridge between a trained model and a user or downstream system. Interview loops at AI/ML companies routinely include system design questions about model serving, rate limiting, auth, and async tasks — all covered here.

### What You'll Build Understanding Of

| Topic | Production Use Case |
|-------|-------------------|
| Flask app factory + blueprints | Versioned REST APIs (`/api/v1/`, `/api/v2/`) |
| ML model registry | Serving multiple models without restart |
| HMAC auth + token bucket rate limiter | Protecting inference endpoints |
| Celery task queue | Long-running model training / batch inference |

### Before You Start
- Python functions and `*args`/`**kwargs`
- Know what decorators are: `@app.route('/path')` is a decorator
- Know what HTTP verbs (GET, POST) and status codes (200, 404, 500) mean at a high level
- `pip install flask`: that's the only required install for the core demo

### HTTP Primer for ML Engineers

| Method | Semantics | ML Use Case |
|--------|-----------|-------------|
| `GET` | Read, idempotent | Fetch model metadata, health check |
| `POST` | Create / trigger action | Submit inference request, upload data |
| `PUT` | Replace resource | Update model registry entry |
| `DELETE` | Remove resource | Remove a model version |

Status codes you must know cold: **200** (OK), **201** (Created), **400** (Bad Request — your fault), **401** (Unauthenticated), **403** (Forbidden), **404** (Not Found), **422** (Validation Error), **429** (Rate Limited), **500** (Server Error), **503** (Service Unavailable).

---

## 1. Flask Internals — Request Lifecycle

Every incoming HTTP request passes through this pipeline:

```
Client Request
    → WSGI server (Gunicorn/uWSGI)
        → Flask WSGI app (__call__)
            → Request context push (flask.g, flask.request)
                → URL routing (Werkzeug Map.match)
                    → Before-request hooks
                        → View function
                    → After-request hooks
                → Response object construction
            → Request context pop (teardown_appcontext)
        → WSGI response
```

**Request context** contains `flask.request` and `flask.g`. **Application context** contains `flask.current_app` and `flask.g`. Both are thread-local (or greenlet-local with gevent) — safe under concurrent requests.

### WSGI Interface

Flask is a WSGI application — a callable:

```python
def app(environ, start_response):
    ...
```

`environ` is a dict with HTTP headers, body, path, method. `start_response` writes status + headers. Flask wraps this in `werkzeug.serving.run_simple` for development.

---

## 2. App Factory Pattern

**Why:** Single app instance = cannot create multiple configurations for testing/staging. Factory function = call with different `config` objects → isolated instances.

```python
def create_app(config_object="config.ProductionConfig"):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_object)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app)

    # Register blueprints
    from .api.v1 import bp as v1_bp
    app.register_blueprint(v1_bp, url_prefix="/api/v1")

    return app
```

**Key rule:** Never import `app` at module level inside blueprints. Use `current_app` proxy instead.

### Config Object Pattern

```python
class BaseConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-key")
    JSON_SORT_KEYS = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///dev.db"

class ProductionConfig(BaseConfig):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ["DATABASE_URL"]
    PROPAGATE_EXCEPTIONS = True
```

---

## 3. Blueprints

A `Blueprint` is a recipe for URL rules, error handlers, and before/after-request hooks scoped to a namespace.

```python
from flask import Blueprint

bp = Blueprint("api", __name__, url_prefix="/api/v1")

@bp.route("/predict", methods=["POST"])
def predict():
    ...

@bp.errorhandler(404)
def not_found(e):
    return {"error": "not found"}, 404
```

**Blueprint vs App:**

| Feature | App | Blueprint |
|---------|-----|-----------|
| `url_prefix` | Set on register | Set on init or register |
| Error handlers | Global | Scoped to blueprint |
| `before_request` | All routes | Only blueprint routes |
| Static folder | Yes | Yes (separate) |
| Template folder | Yes | Yes (separate) |

Multiple blueprints enable versioned APIs: `/api/v1/...` and `/api/v2/...` from two separate `Blueprint` objects registered on the same app.

---

## 4. RESTful API Design

### Status Codes — Interview Must-Knows

| Code | Meaning | ML API use case |
|------|---------|-----------------|
| 200 | OK | Successful prediction |
| 201 | Created | New model version registered |
| 204 | No Content | Delete succeeded |
| 400 | Bad Request | Input validation failed |
| 401 | Unauthorized | Missing/invalid API key |
| 403 | Forbidden | Valid key, insufficient permissions |
| 404 | Not Found | Model ID doesn't exist |
| 409 | Conflict | Duplicate resource |
| 422 | Unprocessable Entity | Semantically invalid payload |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Unhandled exception |
| 503 | Service Unavailable | Model not loaded / downstream down |

### Error Response Format (consistent across all endpoints)

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Field 'features' must be a list of floats",
    "field": "features",
    "request_id": "req_7f3a9b2c"
  }
}
```

### Versioning Strategies

| Strategy | Example | Tradeoff |
|----------|---------|----------|
| URL path | `/api/v1/predict` | Explicit, cacheable, widely adopted |
| Query param | `/predict?version=1` | Simple but pollutes URL |
| Header | `Accept: application/vnd.api+json;version=1` | Clean URL, harder to test in browser |
| Subdomain | `v1.api.example.com` | DNS overhead, CORS complexity |

URL path versioning is the standard for ML APIs.

---

## 5. ML Model Serving

### Model Loading Strategy

**Never** load a model inside the view function — it runs on every request. Load at startup:

```python
# At module level or in create_app()
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    features = request.json["features"]
    result = model.predict([features])
    return jsonify({"prediction": result[0]})
```

**Thread safety:** scikit-learn models are read-only after training — safe for concurrent reads. PyTorch models with `model.eval()` + `torch.no_grad()` are also thread-safe for inference.

### Predict Endpoint Contract

```
POST /api/v1/predict
Content-Type: application/json
Authorization: Bearer <token>

{
  "model_id": "iris_v2",
  "features": [5.1, 3.5, 1.4, 0.2]
}

→ 200 OK
{
  "prediction": 0,
  "probability": [0.98, 0.01, 0.01],
  "model_id": "iris_v2",
  "model_version": "2.1.0",
  "latency_ms": 1.2
}
```

### Pydantic Validation

```python
from pydantic import BaseModel, validator, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=64)
    features: List[float] = Field(..., min_items=1, max_items=1000)

    @validator("features")
    def features_finite(cls, v):
        import math
        if any(math.isnan(x) or math.isinf(x) for x in v):
            raise ValueError("features must be finite numbers")
        return v

class PredictResponse(BaseModel):
    prediction: int
    probability: List[float]
    model_id: str
    latency_ms: float
```

### Batch Prediction

```python
@bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    payload = request.json
    instances = payload["instances"]        # list of feature lists
    max_batch = current_app.config["MAX_BATCH_SIZE"]
    if len(instances) > max_batch:
        return {"error": f"batch size exceeds {max_batch}"}, 400

    X = np.array(instances, dtype=np.float32)
    preds = model.predict(X).tolist()
    probs = model.predict_proba(X).tolist()
    return jsonify({"predictions": preds, "probabilities": probs})
```

---

## 6. Request Validation

### Marshmallow (schema-level)

```python
from marshmallow import Schema, fields, validate, ValidationError

class PredictSchema(Schema):
    model_id = fields.Str(required=True, validate=validate.Length(min=1, max=64))
    features = fields.List(fields.Float(), required=True,
                           validate=validate.Length(min=1, max=1000))

schema = PredictSchema()

@bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = schema.load(request.json or {})
    except ValidationError as e:
        return {"error": e.messages}, 400
    ...
```

### Decorator-based validation

```python
from functools import wraps

def validate_json(*required_fields):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            data = request.get_json(silent=True)
            if not data:
                return {"error": "JSON body required"}, 400
            missing = [k for k in required_fields if k not in data]
            if missing:
                return {"error": f"missing fields: {missing}"}, 400
            return f(*args, **kwargs)
        return wrapper
    return decorator

@bp.route("/predict", methods=["POST"])
@validate_json("model_id", "features")
def predict():
    data = request.json
    ...
```

---

## 7. Middleware — Before/After Request Hooks

### Execution Order

```
Request arrives
  → @app.before_request (all blueprints)
  → @bp.before_request (this blueprint only)
  → view function
  → @bp.after_request (reverse order of registration)
  → @app.after_request
  → @app.teardown_request (always, even on exception)
```

### Auth Middleware

API key pattern: client sends `Authorization: Bearer <key>`. Server hashes and compares against stored hash (never store raw keys).

```python
import hmac, hashlib

VALID_KEY_HASHES = {
    hashlib.sha256(b"secret-api-key-1").hexdigest(),
}

@bp.before_request
def require_auth():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return {"error": "missing Authorization header"}, 401
    token = auth_header[7:]
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    # hmac.compare_digest prevents timing attacks
    if not any(hmac.compare_digest(token_hash, h) for h in VALID_KEY_HASHES):
        return {"error": "invalid API key"}, 401
```

**Why `hmac.compare_digest`:** Standard `==` short-circuits on first mismatch — timing varies with match length, enabling timing side-channel attacks. `compare_digest` runs in constant time.

### Rate Limiting — Token Bucket Algorithm

$$\text{tokens}(t) = \min\!\left(\text{capacity},\; \text{tokens}(t_{prev}) + \text{rate} \times (t - t_{prev})\right)$$

A request consumes 1 token. Denied if tokens < 1.

```python
import threading, time

class TokenBucket:
    def __init__(self, capacity, rate):
        self.capacity = capacity        # max burst
        self.rate = rate                # tokens/second refill
        self.tokens = capacity
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def consume(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
```

**Alternative algorithms:**

| Algorithm | Properties | Use Case |
|-----------|-----------|----------|
| Token Bucket | Burst-friendly, smooth | Default API rate limiting |
| Leaky Bucket | Strict output rate | Traffic shaping |
| Fixed Window | Simple, boundary burst problem | Low-traffic internal APIs |
| Sliding Window Log | Exact, memory-heavy | High-precision billing |
| Sliding Window Counter | Approximate, memory-efficient | Large-scale APIs |

### Request Logging

Log **before** response (request metadata) and **after** (response status + latency):

```python
import uuid, logging

logger = logging.getLogger("api")

@app.before_request
def log_request():
    flask.g.request_id = str(uuid.uuid4())[:8]
    flask.g.start_time = time.perf_counter()
    logger.info(
        "request",
        extra={
            "request_id": flask.g.request_id,
            "method": request.method,
            "path": request.path,
            "ip": request.remote_addr,
        }
    )

@app.after_request
def log_response(response):
    latency_ms = (time.perf_counter() - flask.g.start_time) * 1000
    logger.info(
        "response",
        extra={
            "request_id": flask.g.request_id,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 2),
        }
    )
    response.headers["X-Request-ID"] = flask.g.request_id
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    return response
```

---

## 8. Error Handling

### Global Error Handlers

```python
from werkzeug.exceptions import HTTPException

@app.errorhandler(HTTPException)
def handle_http(e):
    return {
        "error": {
            "code": e.name.upper().replace(" ", "_"),
            "message": e.description,
            "status": e.code,
        }
    }, e.code

@app.errorhandler(Exception)
def handle_unexpected(e):
    logger.exception("unhandled exception", exc_info=e)
    return {"error": {"code": "INTERNAL_ERROR", "message": "unexpected error"}}, 500
```

**Key:** `@app.errorhandler(Exception)` catches everything not matched by more specific handlers. Register HTTP handlers before the generic one.

### abort() vs return

```python
# abort() raises HTTPException — triggers errorhandler
from flask import abort
abort(404)

# return dict directly — bypasses errorhandler, but cleaner for expected errors
return {"error": "not found"}, 404
```

Use `abort()` inside helper functions (where you can't return a response directly). Use `return` in view functions.

---

## 9. Async Task Queue — Celery + Redis

### Why async tasks for ML

Inference can take 100ms–30s. Holding an HTTP connection that long: blocks server workers, times out load balancers (default 60s), wastes client connections.

**Pattern:** accept request → enqueue task → return `task_id` → client polls `/tasks/<id>`.

### Architecture

```
Client → Flask API → Redis (broker) → Celery worker → Redis (result backend)
                  ↑                                           ↓
                  └──── GET /tasks/<id> ◄── poll ────────────┘
```

### Task States

```
PENDING → STARTED → SUCCESS
                 ↘ FAILURE
                 ↘ RETRY
```

### Celery Configuration

```python
from celery import Celery

def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config["CELERY_BROKER_URL"],       # redis://localhost:6379/0
        backend=app.config["CELERY_RESULT_BACKEND"],   # redis://localhost:6379/0
    )
    celery.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        result_expires=3600,        # results expire after 1 hour
        task_acks_late=True,        # ack after task completes (safer for crashes)
        worker_prefetch_multiplier=1,  # 1 task/worker (fair scheduling for long tasks)
    )
    # Push Flask app context into Celery tasks
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    celery.Task = ContextTask
    return celery
```

### Task Definition

```python
@celery.task(bind=True, max_retries=3, default_retry_delay=5)
def run_inference(self, model_id, features):
    try:
        model = load_model(model_id)
        result = model.predict([features])
        return {"prediction": int(result[0]), "status": "done"}
    except ModelNotFoundError as e:
        raise self.retry(exc=e, countdown=2)
    except Exception as e:
        logger.exception(f"task {self.request.id} failed")
        raise
```

**`bind=True`:** Task instance (`self`) is first argument → access `self.request.id`, `self.retry()`.

### Flask Endpoint for Async Pattern

```python
@bp.route("/predict/async", methods=["POST"])
def predict_async():
    data = request.json
    task = run_inference.apply_async(
        args=[data["model_id"], data["features"]],
        countdown=0,
        expires=300,   # task expires from queue after 5 min
    )
    return {"task_id": task.id, "status": "queued"}, 202

@bp.route("/tasks/<task_id>", methods=["GET"])
def task_status(task_id):
    task = run_inference.AsyncResult(task_id)
    if task.state == "PENDING":
        return {"task_id": task_id, "status": "pending"}, 202
    elif task.state == "STARTED":
        return {"task_id": task_id, "status": "running"}, 202
    elif task.state == "SUCCESS":
        return {"task_id": task_id, "status": "done", "result": task.result}, 200
    elif task.state == "FAILURE":
        return {"task_id": task_id, "status": "failed",
                "error": str(task.info)}, 500
    return {"task_id": task_id, "status": task.state}, 202
```

**202 Accepted:** standard HTTP status for "request accepted, processing asynchronously".

### Worker Concurrency Models

| Mode | Flag | Best for |
|------|------|----------|
| Prefork (multiprocessing) | `--pool=prefork` | CPU-bound tasks (model inference) |
| Gevent | `--pool=gevent` | I/O-bound tasks (API calls, DB queries) |
| Eventlet | `--pool=eventlet` | Similar to gevent |
| Solo | `--pool=solo` | Debugging (single-threaded) |

```bash
celery -A app.celery worker --pool=prefork --concurrency=4 --loglevel=info
```

---

## 10. Production Deployment Patterns

### Gunicorn Configuration

```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4                  # 2 * CPU_cores + 1
worker_class = "sync"        # or "gevent" for async
worker_connections = 1000
timeout = 120                # increase for long inference
keepalive = 5
max_requests = 1000          # restart workers after N requests (memory leak guard)
max_requests_jitter = 100    # randomize restart to avoid thundering herd
accesslog = "-"
errorlog = "-"
loglevel = "info"
```

### Health Checks

```python
@app.route("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}, 200

@app.route("/health/ready")
def readiness():
    checks = {}
    # Check model loaded
    checks["model"] = "ok" if model is not None else "not_loaded"
    # Check Redis (if used)
    try:
        redis_client.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unreachable"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ready" if all_ok else "degraded", "checks": checks}, \
           200 if all_ok else 503
```

**Liveness** (`/health`): is the process alive? Kubernetes restarts if this fails.
**Readiness** (`/health/ready`): is the service ready to receive traffic? Load balancer stops routing if this fails.

### Request ID Propagation

Every request gets a unique ID injected at the API gateway. Pass it downstream:

```python
@app.before_request
def extract_request_id():
    flask.g.request_id = (
        request.headers.get("X-Request-ID")
        or str(uuid.uuid4())[:8]
    )
```

Include in all log lines + error responses + downstream API calls.

---

## 11. Performance Patterns

### Connection Pooling

Never create a new DB connection per request — use a pool:

```python
from sqlalchemy import create_engine

engine = create_engine(
    DATABASE_URL,
    pool_size=10,           # persistent connections
    max_overflow=20,        # temporary connections beyond pool_size
    pool_timeout=30,        # seconds to wait for connection
    pool_recycle=1800,      # recycle connections after 30 min (avoids stale conn)
)
```

### Response Caching

```python
from functools import lru_cache
import hashlib, json

@lru_cache(maxsize=512)
def _cached_predict(model_id, features_tuple):
    model = load_model(model_id)
    return model.predict([list(features_tuple)])[0]

@bp.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Convert list → tuple for lru_cache (must be hashable)
    features_key = tuple(data["features"])
    result = _cached_predict(data["model_id"], features_key)
    return jsonify({"prediction": int(result)})
```

For distributed caching, use Redis with a serialized key:

```python
cache_key = f"pred:{model_id}:{hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()}"
cached = redis_client.get(cache_key)
if cached:
    return json.loads(cached)
```

### Streaming Responses

For large results or server-sent events (e.g., LLM token streaming):

```python
from flask import stream_with_context, Response

@bp.route("/stream")
def stream():
    def generate():
        for token in model.generate_tokens(prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

---

## 12. Interview Reference — Flask Internals

### Q: How does Flask handle concurrent requests?

Flask is WSGI — single-threaded per worker by default. Under `flask run`, it uses `werkzeug`'s threaded development server (not production-safe). In production, Gunicorn spawns N worker processes, each running a Flask WSGI app instance. Workers share no memory — `g`, `current_app`, etc. are process-local.

### Q: What's the difference between `g`, `session`, and `config`?

| Object | Scope | Storage |
|--------|-------|---------|
| `flask.g` | Single request | In-memory, request context |
| `flask.session` | Across requests (same client) | Signed cookie (or server-side) |
| `app.config` | Application lifetime | In-memory, application context |

### Q: How do you prevent SQL injection in Flask?

Use parameterized queries — never string-format SQL:

```python
# WRONG — SQL injection
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# CORRECT — parameterized
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
# SQLAlchemy ORM — always parameterized
User.query.filter_by(id=user_id).first()
```

### Q: How does `@app.teardown_appcontext` differ from `@app.after_request`?

`after_request` runs only if the view function returns normally. `teardown_appcontext` runs always — even on exceptions — making it the right place for cleanup (closing DB connections, releasing locks).

### Q: What is a WSGI middleware?

A callable wrapping the Flask app that intercepts/modifies the WSGI `environ` or response:

```python
class TimingMiddleware:
    def __init__(self, app):
        self.app = app
    def __call__(self, environ, start_response):
        t0 = time.perf_counter()
        result = self.app(environ, start_response)
        print(f"request took {(time.perf_counter()-t0)*1000:.1f}ms")
        return result

app.wsgi_app = TimingMiddleware(app.wsgi_app)
```

---

## 13. gRPC for ML Serving

gRPC (Google Remote Procedure Call) uses Protocol Buffers (protobuf) over HTTP/2. Advantages over REST for ML serving: binary encoding (~5-10× smaller payloads than JSON), multiplexed streams (no head-of-line blocking), bidirectional streaming, strongly typed contracts, generated client/server stubs.

### 13.1 Protocol Buffers

```proto
// prediction.proto
syntax = "proto3";

service Predictor {
  rpc Predict (PredictRequest) returns (PredictResponse);
  rpc PredictStream (stream PredictRequest) returns (stream PredictResponse);
}

message PredictRequest {
  repeated float features = 1;
  string model_version = 2;
}

message PredictResponse {
  float score = 1;
  string label = 2;
  float latency_ms = 3;
}
```

Compile: `python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. prediction.proto`

### 13.2 Server Implementation

```python
import grpc, time, threading
import prediction_pb2, prediction_pb2_grpc

class PredictorServicer(prediction_pb2_grpc.PredictorServicer):
    def __init__(self, model):
        self.model = model

    def Predict(self, request, context):
        t0 = time.perf_counter()
        features = list(request.features)
        score = float(self.model.predict_proba([features])[0, 1])
        label = "positive" if score > 0.5 else "negative"
        return prediction_pb2.PredictResponse(
            score=score, label=label,
            latency_ms=(time.perf_counter()-t0)*1000
        )

    def PredictStream(self, request_iterator, context):
        for request in request_iterator:
            yield self.Predict(request, context)

def serve(model, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_pb2_grpc.add_PredictorServicer_to_server(
        PredictorServicer(model), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()
```

### 13.3 Client

```python
import grpc
import prediction_pb2, prediction_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub    = prediction_pb2_grpc.PredictorStub(channel)

# Unary
response = stub.Predict(prediction_pb2.PredictRequest(
    features=[1.2, 0.5, -0.3], model_version="v1"
))
print(response.score, response.label)

# Client streaming
def feature_generator():
    for row in dataset:
        yield prediction_pb2.PredictRequest(features=row)

for response in stub.PredictStream(feature_generator()):
    print(response.label)
```

### 13.4 REST vs. gRPC for ML Serving

| Dimension | REST (JSON) | gRPC (protobuf) |
|-----------|-------------|-----------------|
| Encoding | Text (JSON) | Binary (protobuf) |
| Payload size | Baseline | ~5-10× smaller |
| Latency | Higher (serialization) | Lower |
| Streaming | SSE / chunked | Native bidirectional |
| Schema | OpenAPI (optional) | Protobuf (required) |
| Browser support | Native | Requires grpc-web |
| Tooling | Universal | Requires code-gen |
| Use case | Public APIs, browser clients | Internal microservices |

gRPC is preferred for high-throughput internal services (model servers, feature stores). REST is preferred for public APIs and browser clients.

---

## Interview Reference — Backend with Flask (Extended)

### Q: What is WSGI and how does it differ from ASGI?

WSGI (PEP 3333): synchronous interface. A WSGI app is a callable `app(environ, start_response)` → returns an iterable of byte strings. One request per call, one thread per concurrent request. Flask, Django (classic) are WSGI. ASGI: async interface used by FastAPI, Starlette. Supports WebSockets, long-polling, and concurrent requests within a single thread via `async/await`. For ML serving: WSGI + Gunicorn is simpler; ASGI needed for streaming inference or WebSocket chat interfaces.

### Q: How would you version a REST API?

URL versioning: `/v1/predict`, `/v2/predict` — simplest, most explicit. Header versioning: `Accept: application/vnd.api+json; version=2` — cleaner URLs, harder to cache. Query param: `?version=2` — least REST-compliant. Strategy: URL versioning is standard for breaking changes; keep v1 alive until all clients migrate. Never remove v1 without deprecation period + monitoring.

### Q: What is a circuit breaker and when do you use it?

Circuit breaker wraps external calls (model server, DB, 3rd-party API). States: CLOSED (normal, passes calls through) → OPEN (after N failures in T seconds, rejects calls immediately, returns fallback) → HALF-OPEN (after reset_timeout, allows one probe call to check recovery). Prevents cascading failures when a downstream service is down. Library: `circuitbreaker`, `tenacity`, or implement with `threading.Lock`.

### Q: How does Celery distribute work and what is a broker?

Celery workers pull tasks from a message broker (Redis queue or RabbitMQ exchange). Client calls `task.delay(args)` → serializes task to broker. Worker dequeues, executes, stores result in backend (another Redis key). Concurrency: each worker runs N concurrent tasks (threaded, greenlet, or prefork depending on `--pool`). For CPU-bound ML inference: use prefork (separate processes, GIL-free).

### Q: How do you prevent race conditions when serving ML models in Flask?

Flask on Gunicorn with multiple workers: each worker loads its own model copy — no shared state, no race conditions. Within a single process: if you reload a model at runtime, use a `threading.Lock` to prevent reads during model swap. For zero-downtime model updates in production: use A/B routing (blue-green) — spin up new worker with new model, shift traffic, then kill old worker. Never `model = new_model` without a lock in a threaded app.

---

## Cheat Sheet — Backend with Flask

| Concept | Key Fact |
|---------|----------|
| Flask request lifecycle | Routing → before_request → view → after_request → teardown |
| `g` object | Per-request storage; dies at end of request |
| `current_app` | Proxy to the active app; usable outside app context via `app_context()` |
| App factory | `create_app(config)` returns configured Flask instance; enables testing |
| Blueprint | Logical grouping of routes; prefix: `Blueprint("api", url_prefix="/api/v1")` |
| `@app.errorhandler` | Catch specific HTTP codes; return JSON error responses |
| Marshmallow | Schema → `load()` validates input; `dump()` serializes output |
| Celery task | `@celery.task` decorator; call with `.delay()` or `.apply_async()` |
| Gunicorn workers | `2 × CPU + 1` sync workers; `--worker-class geventlet` for async |
| REST status codes | 200 OK, 201 Created, 400 Bad Request, 401 Unauth, 404 Not Found, 422 Validation, 500 Server Error |
| WSGI | Synchronous; one thread per request; use with Gunicorn |
| ASGI | Async; use with Uvicorn + FastAPI for streaming / WebSockets |
| gRPC | Protobuf binary; HTTP/2; bidirectional streaming; for internal ML serving |
| Circuit breaker | CLOSED → OPEN (after N failures) → HALF-OPEN (probe) → CLOSED |
| SQL injection prevention | Parameterized queries only; never f-string SQL |

---

## Resources

### Flask & Python Web
- **Flask documentation** (`flask.palletsprojects.com/en/stable/`): The official docs. Start with "Quickstart" and "Application Factories".
- **Flask Mega-Tutorial** — Miguel Grinberg: The most complete free Flask tutorial. Covers blueprints, auth, database, deployment. Search "Flask Mega-Tutorial Grinberg".
- **Werkzeug documentation** (`werkzeug.palletsprojects.com`): Flask is built on Werkzeug; this explains the WSGI internals.

### ML Serving
- **BentoML** (`bentoml.com`): Higher-level ML serving framework built on similar principles. Read their architecture docs after this module.
- **TorchServe** (`pytorch.org/serve`): PyTorch's official model server. Uses model registry and handler patterns identical to what's built here.
- **"Serving Machine Learning Models"** — O'Reilly Report (search title): Covers REST, gRPC, batch, and streaming serving patterns.

### Async Tasks & Queues
- **Celery documentation** (`docs.celeryq.dev`): Task routing, retries, chord/chain primitives.
- **Redis documentation** (`redis.io/docs`): The broker/backend used by Celery in production.

### System Design
- **"Designing Machine Learning Systems"** — Chip Huyen: Chapters on feature pipelines, model deployment, and monitoring directly extend this module.

---

> **Next module:** [Module 05 — Deep Learning & MLOps](05-deep-learning.md)
