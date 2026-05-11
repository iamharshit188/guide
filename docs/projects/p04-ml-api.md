# Project 04 — Production ML Serving API

> **Difficulty:** Intermediate · **Module:** 04 — Backend with Flask
> **Skills:** Flask app factory, blueprints, HMAC auth, token-bucket rate limiting, async tasks, latency benchmarking

---

## What You'll Build

A Flask API that serves a pickled sklearn Pipeline via two versioned blueprints: `/api/v1/` (synchronous single prediction) and `/api/v2/` (async batch with task polling). HMAC authentication, token-bucket rate limiter, structured JSON logging, and a p50/p95/p99 latency report from 50 concurrent requests.

---

## Skills Exercised

- Flask app factory pattern with `create_app(env)`
- Blueprint registration with URL prefixes
- HMAC-SHA256 timing-safe request authentication
- Token bucket rate limiter (per-client, thread-safe with `RLock`)
- Async batch prediction via in-process thread (or Celery stub)
- Structured JSON logging with `X-Request-ID`
- `concurrent.futures.ThreadPoolExecutor` for load testing

---

## Approach

### Phase 1 — Train and pickle a model
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
# use sklearn.datasets.make_classification
pipe = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(42))])
pipe.fit(X_train, y_train)
import pickle; pickle.dump(pipe, open('model.pkl', 'wb'))
```

### Phase 2 — App factory + blueprints
```
create_app(env='development'):
    app = Flask(__name__)
    app.config.from_object(Config[env])     # Base/Dev/Prod
    app.register_blueprint(v1_bp, url_prefix='/api/v1')
    app.register_blueprint(v2_bp, url_prefix='/api/v2')
    return app

v1_bp: GET /health, POST /predict (single sample, sync)
v2_bp: GET /health, POST /predict/batch (list of samples, returns task_id),
       GET /tasks/<task_id> (poll status + result)
```

### Phase 3 — Auth middleware
```
HMAC-SHA256: client sends X-Signature: sha256_hex(secret + request_body)
server recomputes, uses hmac.compare_digest (timing-safe)
reject 401 if mismatch
```

### Phase 4 — Rate limiter
```
TokenBucket(capacity=10, refill_rate=2.0):
    tokens: float = capacity
    last_refill: float = time.time()
    lock: threading.RLock

    def consume(self, n=1) -> bool:
        with lock:
            refill tokens based on elapsed time
            if tokens >= n: tokens -= n; return True
            return False   # → 429 Too Many Requests

per_client_limiter: dict[ip, TokenBucket]
```

### Phase 5 — Async batch
```
POST /api/v2/predict/batch:
    validate input (list of feature dicts)
    task_id = uuid4()
    tasks[task_id] = {"status": "PENDING", "result": None}
    thread = Thread(target=run_batch, args=(task_id, samples))
    thread.start()
    return 202 {"task_id": task_id}

run_batch(task_id, samples):
    tasks[task_id]["status"] = "STARTED"
    results = model.predict(np.array(samples))
    tasks[task_id] = {"status": "SUCCESS", "result": results.tolist()}

GET /api/v2/tasks/<task_id> → {"status": ..., "result": ...}
```

### Phase 6 — Load test + latency table
```python
with ThreadPoolExecutor(max_workers=10) as pool:
    futures = [pool.submit(post_predict, sample) for _ in range(50)]
latencies = [f.result() for f in futures]
print(f"p50={np.percentile(latencies,50):.1f}ms  "
      f"p95={np.percentile(latencies,95):.1f}ms  "
      f"p99={np.percentile(latencies,99):.1f}ms")
```

---

## Checkpoints

| Phase | What correct output looks like |
|-------|-------------------------------|
| 1 | `pickle.load` + `pipe.predict([[...]])` returns class label |
| 2 | `GET /api/v1/health` → `{"status":"ok"}`, `GET /api/v2/health` → same |
| 3 | Wrong HMAC → `{"error":"Unauthorized"}` 401; correct HMAC → 200 |
| 4 | 11th request within 5s from same IP → 429; after 5s refill → 200 again |
| 5 | POST batch → 202 with task_id; GET tasks/{id} → PENDING → SUCCESS |
| 6 | p50 < 20ms on local machine; p95 < 50ms |

---

## Extensions

1. **Model versioning** — store multiple model versions in a dict keyed by semver; add `?version=1.0` query param to `/predict`; track which version is "default".
2. **Health + metrics endpoints** — `GET /metrics` returns JSON: total requests, 4xx count, 5xx count, avg latency (from an in-process ring buffer of last 1000 request durations).
3. **Gunicorn config** — add `gunicorn_config.py`: 4 workers, 30s timeout, access log format. Document how to run: `gunicorn -c gunicorn_config.py "app:create_app()"`.

---

## Hints

<details><summary>Hint 1 — HMAC construction</summary>
<code>hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()</code>. The client must send the raw request body bytes, not JSON-re-serialized (order matters).
</details>

<details><summary>Hint 2 — Token bucket refill</summary>
<code>elapsed = now - last_refill; tokens = min(capacity, tokens + elapsed * refill_rate); last_refill = now</code>. Do this inside the lock every call.
</details>

<details><summary>Hint 3 — Thread-safe task dict</summary>
Use a module-level <code>tasks = {}</code> dict and a <code>tasks_lock = threading.Lock()</code> when reading/writing. Python dict ops are GIL-protected for single reads/writes but compound read-modify-write needs explicit locking.
</details>

<details><summary>Hint 4 — 202 vs 200 for async</summary>
RFC 7231: 202 Accepted means "request received, processing has not completed." Return 202 immediately after spawning the thread. Return 200 from the poll endpoint when status == SUCCESS.
</details>

<details><summary>Hint 5 — Measuring latency in the load test</summary>
<code>start = time.perf_counter(); requests.post(...); return (time.perf_counter()-start)*1000</code>. Use <code>perf_counter</code>, not <code>time.time</code>, for sub-millisecond precision.
</details>

---

*Back to [Module 04 — Backend with Flask](../04-backend.md)*
