# Module 11 — Deployment & Production ML

> **Run the code:**
> ```bash
> cd src/11-deployment
> python3.14 onnx_export.py
> python3.14 quantize.py
> python3.14 ab_serving.py
> python3.14 health_check.py
> ```

---

## Prerequisites & Overview

**Time estimate:** 6–8 hours

| Prerequisite | From |
|-------------|------|
| Flask app factory + blueprints | Module 04 |
| sklearn Pipeline + pickle | Module 02 |
| Model serving fundamentals | Module 04 |

**Before you start:**
- [ ] Know what a Docker image and container are (image = template, container = running instance)
- [ ] Understand WSGI (Web Server Gateway Interface) — how Flask connects to Gunicorn
- [ ] Know the difference between model training and model inference

**Module map:**

| Section | Core concept |
|---------|-------------|
| Docker | Layered image, Dockerfile, docker-compose |
| ONNX | Intermediate representation, opset, runtime |
| TorchScript | trace vs. script, frozen module |
| Quantization | FP32 → FP16 / INT8, dynamic vs. static |
| Gunicorn + nginx | Multi-worker WSGI, reverse proxy, load balancing |
| A/B serving | Traffic splitting, canary deployments |
| Health checks | /health, /ready, /metrics, liveness probe |

---

## Docker — Containerizing ML Services

### Why Docker for ML

A Docker image captures the entire runtime: Python version, package versions, model weights, config. The same image runs identically on a laptop, CI, and production Kubernetes cluster. "Works on my machine" disappears.

### Dockerfile for a Flask ML Service

```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime image (smaller)
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

# Gunicorn: 4 workers, 30s timeout
CMD ["gunicorn", "--workers=4", "--timeout=30",
     "--bind=0.0.0.0:8080", "--access-logfile=-",
     "app:create_app()"]
```

**Key choices:**
- **Multi-stage build:** separates build environment from runtime image — reduces final image size by ~60%.
- **`--no-cache-dir`:** pip won't store wheel cache in the image layer.
- **Gunicorn instead of `flask run`:** production-grade WSGI server; `flask run` is single-threaded dev only.

### docker-compose for ML Stack

```yaml
version: "3.9"
services:
  model-api:
    build: .
    ports: ["8080:8080"]
    environment:
      - MODEL_PATH=/models/model.pkl
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/models:ro   # read-only model mount
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru

  nginx:
    image: nginx:alpine
    ports: ["80:80"]
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on: [model-api]
```

### nginx Reverse Proxy Configuration

```nginx
upstream model_api {
    server model-api:8080;
    keepalive 32;
}

server {
    listen 80;

    location /api/ {
        proxy_pass         http://model_api;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 30s;
        proxy_connect_timeout 5s;
    }

    location /health {
        proxy_pass http://model_api/health;
        access_log off;   # don't log health checks
    }
}
```

**nginx's role:** terminates TLS, compresses responses, rate-limits at the edge, and load-balances across multiple upstream replicas.

---

## ONNX — Open Neural Network Exchange

### What ONNX Is

ONNX is an open intermediate representation for ML models. Export once → run on any ONNX Runtime (CPU, GPU, TensorRT, CoreML). Decouples training framework (sklearn, PyTorch, XGBoost) from serving runtime.

### ONNX Graph Structure

An ONNX model is a computation graph:
- **Nodes:** operations (MatMul, Relu, Softmax, ...)
- **Initializers:** learned weights (stored as tensors)
- **Inputs/Outputs:** typed and shaped

```
Input(float32, [batch, 10])
    ↓ MatMul(W: [10, 64])
    ↓ Add(bias: [64])
    ↓ Relu
    ↓ MatMul(W: [64, 2])
    ↓ Softmax(axis=1)
Output(float32, [batch, 2])
```

### Exporting sklearn to ONNX

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Assumes a trained sklearn Pipeline: pipe = Pipeline([...])
initial_type = [("float_input", FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=17)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### ONNX Runtime Inference

```python
import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession("model.onnx",
                           providers=["CPUExecutionProvider"])
input_name  = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

X_test = np.array([[...]], dtype=np.float32)   # must be float32
predictions = sess.run([output_name], {input_name: X_test})[0]
```

**Key constraint:** ONNX Runtime requires `float32` inputs. Most sklearn pipelines work in `float64`; cast explicitly.

### Opset Versions

The `target_opset` controls which ONNX operations are available. Higher opset = more ops + better optimization support. Use `17` for maximum hardware backend compatibility as of 2025.

---

## TorchScript — Serializable PyTorch

### Trace vs. Script

| Method | How | When |
|--------|-----|------|
| `torch.jit.trace(model, example_input)` | Records tensor operations on example input | Static control flow (no data-dependent branches) |
| `torch.jit.script(model)` | Compiles Python → TorchScript IR statically | Dynamic control flow (if/for on tensors) |

```python
try:
    import torch

    model = MyModel()
    model.eval()

    # Trace (faster, simpler)
    example = torch.randn(1, 10)
    traced = torch.jit.trace(model, example)
    traced.save("model_traced.pt")

    # Script (handles if/else on tensor values)
    scripted = torch.jit.script(model)
    scripted.save("model_scripted.pt")

    # Load and infer (no Python dependencies needed)
    loaded = torch.jit.load("model_traced.pt")
    out = loaded(torch.randn(4, 10))

except ImportError:
    print("PyTorch not available — skipping TorchScript demo")
```

**Deployment advantage:** a `.pt` TorchScript file can be loaded in C++ (`libtorch`) without Python, enabling embedding in mobile apps or low-latency services.

---

## Quantization for Serving

### Why Quantize

| Format | Bits/weight | Inference speed | Memory | Accuracy |
|--------|-------------|----------------|--------|----------|
| FP32 | 32 | 1× (baseline) | 4 bytes/param | Full |
| FP16 | 16 | 2–4× on GPU | 2 bytes/param | Near-full |
| INT8 | 8 | 2–4× on CPU | 1 byte/param | Small degradation |
| INT4 | 4 | 2–3× (with kernels) | 0.5 bytes/param | Moderate degradation |

### FP32 → FP16 (Half Precision)

For GPU inference, FP16 halves memory and doubles throughput (NVIDIA Tensor Cores run FP16 2× faster than FP32):

```python
# NumPy simulation of FP16 quantization
W_fp32 = np.random.randn(256, 256).astype(np.float32)
W_fp16 = W_fp32.astype(np.float16)

rmse_fp16 = np.sqrt(np.mean((W_fp32 - W_fp16.astype(np.float32))**2))
size_reduction = W_fp32.nbytes / W_fp16.nbytes
# rmse ≈ 1e-4 to 1e-3, size_reduction = 2.0
```

### INT8 Dynamic Quantization

Dynamic quantization computes activation scale factors at inference time (no calibration dataset needed):

```python
# INT8 quantization: map float → int8 via scale + zero_point
def quantize_int8(W: np.ndarray):
    W_min, W_max = W.min(), W.max()
    scale      = (W_max - W_min) / 255.0
    zero_point = -round(W_min / scale)
    W_int8     = np.clip(np.round(W / scale + zero_point), 0, 255).astype(np.uint8)
    return W_int8, scale, zero_point

def dequantize_int8(W_int8, scale, zero_point):
    return (W_int8.astype(np.float32) - zero_point) * scale

W = np.random.randn(256, 256).astype(np.float32)
W_q, scale, zp = quantize_int8(W)
W_recovered = dequantize_int8(W_q, scale, zp)
rmse_int8 = np.sqrt(np.mean((W - W_recovered)**2))
```

### Static Quantization (with Calibration)

Static quantization collects activation statistics on a calibration dataset to pre-compute scale factors:

```
1. Run model on calibration set (1K–10K samples)
2. Collect min/max (or percentile) of each activation tensor
3. Compute per-layer scale and zero_point
4. Quantize weights + activations with pre-computed factors
5. Inference: no floating-point activations in the critical path
```

**Trade-off:** Dynamic quantization is faster to apply (no calibration) but less accurate than static, because per-batch scale computation adds overhead.

### Post-Training Quantization vs. QAT

| Method | Accuracy | Effort |
|--------|----------|--------|
| PTQ (post-training quantization) | ≈FP32 -1% | Low: no retraining |
| QAT (quantization-aware training) | ≈FP32 -0.2% | High: retrain with fake-quant ops |

QAT simulates quantization during training (`fake_quant` nodes clamp and round gradients), allowing the model to adapt its weights to quantization noise.

---

## Gunicorn — Production WSGI Server

### Why Not `flask run`

`flask run` uses Werkzeug's single-threaded development server. One blocked request blocks all others. Gunicorn spawns multiple OS-level worker processes, each handling requests independently.

### Worker Models

| Worker type | Config | When to use |
|-------------|--------|-------------|
| `sync` (default) | `--workers=4` | CPU-bound inference |
| `gevent` (async) | `--worker-class=gevent` | I/O-bound (DB calls, external APIs) |
| `gthread` (threaded) | `--worker-class=gthread --threads=4` | Mixed I/O + CPU |

```python
# gunicorn_config.py
import multiprocessing

workers          = multiprocessing.cpu_count() * 2 + 1  # standard formula
worker_class     = "sync"
timeout          = 30        # kill stuck workers after 30s
keepalive        = 5         # keep connections alive 5s
bind             = "0.0.0.0:8080"
accesslog        = "-"       # stdout
errorlog         = "-"       # stderr
loglevel         = "info"

# Preload: load model once in master, fork to workers (saves memory via CoW)
preload_app      = True
```

**Run:** `gunicorn -c gunicorn_config.py "app:create_app()"`

### Worker Count Formula

Rule of thumb: `workers = 2 × CPU_cores + 1`. For ML inference (CPU-heavy), `workers = CPU_cores` is often better to avoid thrashing the CPU cache.

---

## A/B Model Serving

### Traffic Splitting

Route a percentage of traffic to a new model version while keeping the old version live:

```python
import random, threading

class ABRouter:
    def __init__(self):
        self.models    = {}    # {"v1": model, "v2": model}
        self.weights   = {}    # {"v1": 0.9, "v2": 0.1}
        self._lock     = threading.RLock()
        self._counters = {}    # {"v1": 0, "v2": 0}

    def register(self, version: str, model, weight: float):
        with self._lock:
            self.models[version]    = model
            self.weights[version]   = weight
            self._counters[version] = 0
        self._normalize()

    def _normalize(self):
        total = sum(self.weights.values())
        with self._lock:
            self.weights = {k: v/total for k, v in self.weights.items()}

    def route(self) -> str:
        r = random.random()
        cumulative = 0.0
        with self._lock:
            for version, weight in self.weights.items():
                cumulative += weight
                if r < cumulative:
                    self._counters[version] += 1
                    return version
            # fallback to last
            last = list(self.weights.keys())[-1]
            self._counters[last] += 1
            return last

    def predict(self, X):
        version = self.route()
        return self.models[version].predict(X), version

    def stats(self) -> dict:
        with self._lock:
            return {"weights": dict(self.weights), "request_counts": dict(self._counters)}
```

### Canary Deployment

Canary = route a small fraction (1–5%) of traffic to the new model. Monitor metrics. Ramp up if healthy:

```
Phase 1: v1=99%, v2_canary=1%  → monitor error rate, latency, accuracy
Phase 2: v1=90%, v2_canary=10% → check for regressions
Phase 3: v1=50%, v2_canary=50% → parity check
Phase 4: v1=0%,  v2=100%       → full migration
```

**Rollback trigger:** if the canary's error rate exceeds baseline by >5% or p95 latency increases >20%, automatically route all traffic back to v1.

---

## Health Checks & Observability

### The Three Endpoints

| Endpoint | Purpose | Probe type |
|----------|---------|-----------|
| `GET /health` | Is the process alive? | Kubernetes liveness probe |
| `GET /ready` | Is the process ready to serve traffic? | Kubernetes readiness probe |
| `GET /metrics` | Performance counters for monitoring | Prometheus scrape |

```python
import time, threading
from collections import deque

# In-process metrics store
class MetricsStore:
    def __init__(self, window: int = 1000):
        self._lock     = threading.Lock()
        self._latencies = deque(maxlen=window)
        self._counts   = {"2xx": 0, "4xx": 0, "5xx": 0}

    def record(self, status_code: int, latency_ms: float):
        with self._lock:
            self._latencies.append(latency_ms)
            key = f"{status_code // 100}xx"
            self._counts[key] = self._counts.get(key, 0) + 1

    def summary(self) -> dict:
        with self._lock:
            lats = list(self._latencies)
            if not lats:
                return {"error": "no requests yet"}
            lats.sort()
            n = len(lats)
            return {
                "request_counts": dict(self._counts),
                "p50_ms":  lats[n // 2],
                "p95_ms":  lats[int(n * 0.95)],
                "p99_ms":  lats[int(n * 0.99)],
                "mean_ms": sum(lats) / n,
            }

METRICS = MetricsStore()

# Flask endpoints
@app.route("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}, 200

@app.route("/ready")
def ready():
    if model is None:
        return {"status": "not ready", "reason": "model not loaded"}, 503
    return {"status": "ready"}, 200

@app.route("/metrics")
def metrics():
    return METRICS.summary(), 200
```

### Rolling Deploys

Zero-downtime deployment without Kubernetes:

```
1. Start new container (v2) on port 8081
2. Run smoke test: curl -f http://localhost:8081/health
3. Update nginx upstream to include port 8081 (nginx -s reload — zero downtime)
4. Remove port 8080 from nginx upstream
5. Drain old container (wait for in-flight requests to complete)
6. Stop old container
```

In Kubernetes: `RollingUpdate` strategy with `maxUnavailable=0` and `maxSurge=1` achieves this automatically.

---

## Interview Q&A

**Q: Why is multi-stage Docker build important for ML services?**
**A:** Build stage installs compilers and build tools to compile native extensions (like numpy, scipy). Runtime stage discards all build artifacts — only the compiled `.so` files and Python packages are copied. This typically reduces image size from 2–4 GB (full build environment) to 500 MB–1 GB. Smaller images mean faster pulls in CI/CD, reduced attack surface, and lower storage costs in container registries.

**Q: What does ONNX opset version control?**
**A:** Each opset version adds or modifies operator definitions. Higher opset = more operations available (e.g., opset 17 added QLinearMatMul improvements). When exporting, choose the opset that matches your ONNX Runtime version — newer runtimes support higher opsets. Mismatched opsets cause `InvalidGraph` errors at load time. For maximum compatibility across backends (TensorRT, CoreML, Azure ML), target opset 11–13.

**Q: Trace vs. script in TorchScript — when does tracing fail?**
**A:** Tracing records the path through the model for one specific input. If the model has `if tensor.shape[0] > 1: ...` (a data-dependent branch), tracing records only the branch taken for the example input. The other branch is silently omitted. Scripting compiles the Python AST and handles all branches. Use tracing for simple feedforward models; use scripting for models with dynamic shapes or control flow.

**Q: What is the difference between dynamic and static quantization?**
**A:** Dynamic quantization computes activation scale factors per-batch at inference time — easy to apply, no calibration data needed, but adds per-inference overhead. Static quantization pre-computes scale factors on a calibration dataset (1K–10K samples), so inference uses pre-stored scales — faster, but requires representative calibration data and a re-export step. Static is preferred for production latency-critical paths.

**Q: How many Gunicorn workers should you run for a CPU-bound ML inference service?**
**A:** Rule of thumb: `2 × CPU_cores + 1`. But for CPU-heavy inference (e.g., batch sklearn predictions), `workers = CPU_cores` is often better — more workers than cores causes context-switching overhead and cache thrashing. Profile with a load test (`wrk` or `locust`) at different worker counts; the optimal is where throughput plateaus and p99 latency starts rising.

**Q: What is a canary deployment and what triggers a rollback?**
**A:** A canary deploys a new model version to a small traffic fraction (1–5%) while the old version handles the rest. Rollback triggers: (1) error rate of canary > baseline + threshold (e.g., 5%), (2) p95 latency increase > 20%, (3) custom metric regression (e.g., downstream business metric like CTR drops). Automated rollback is implemented via feature flags or a deployment controller that monitors the metrics and adjusts traffic weights.

**Q: What is the /ready endpoint used for that /health cannot cover?**
**A:** `/health` (liveness): "is this process running and not deadlocked?" Used by orchestrators to restart crashed processes. `/ready` (readiness): "is this process ready to serve traffic?" A process can be alive (health=ok) but not ready (model still loading from disk, warming up caches, waiting for DB connection). Kubernetes uses readiness to decide whether to route traffic. During a rolling deploy, new pods are only added to the load balancer once they pass readiness.

**Q: How does nginx handle load balancing across multiple Gunicorn workers?**
**A:** nginx uses `upstream` blocks with a list of backends. By default: round-robin (each request goes to the next server in rotation). Other strategies: `least_conn` (route to the server with fewest active connections — better for variable-length requests like ML inference), `ip_hash` (sticky sessions — same client always goes to same server). For ML inference with stateless models, `least_conn` minimizes queue buildup during slow requests.

---

## Resources

**Papers & Docs:**
- [ONNX Opset Registry](https://onnx.ai/onnx/operators/) — all operator definitions by opset
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html) — trace vs. script guide
- [Quantization in PyTorch](https://pytorch.org/docs/stable/quantization.html) — PTQ and QAT APIs
- [Gunicorn Configuration Docs](https://docs.gunicorn.org/en/stable/settings.html) — all config options

**Tools:**
- `skl2onnx` — sklearn → ONNX conversion
- `onnxruntime` — cross-platform ONNX inference
- `locust` — Python-based load testing for Flask APIs
- `Prometheus + Grafana` — standard metrics stack

**Books:**
- *Designing Machine Learning Systems* — Chip Huyen (2022) — Chapter 7: Model Deployment, Chapter 9: Continual Learning
- *Building Machine Learning Powered Applications* — Emmanuel Ameisen (2020) — Chapter 10: Build Confidence in Your Model

---

*Next: [Module 12 — RLHF & Alignment](12-rlhf.md)*
