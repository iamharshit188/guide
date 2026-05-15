# Module 11 — Deployment & Production ML

> **Prerequisites:** Modules 04–05 (Flask backend, deep learning). Docker basics helpful.
> **Estimated time:** 10–15 hours

## From Notebook to Production

A model that scores 97% in a notebook but crashes in production is worthless. Production ML adds constraints your notebook never had:
- **Latency**: users expect responses in < 200ms, not 5 seconds
- **Throughput**: 1 request/second in testing → 1000/second in prod
- **Reliability**: 99.9% uptime, graceful degradation on failures
- **Observability**: know when model quality degrades before users complain
- **Cost**: inference at scale is expensive — optimize ruthlessly

```
ML Deployment Pipeline:

Data Scientist                        MLOps / Engineer
┌──────────────┐                     ┌──────────────────────────────────┐
│  Notebook    │                     │  Production System               │
│  model.pkl   │──[serialize ONNX]──▶│                                  │
│  accuracy=97%│                     │  [Docker container]              │
└──────────────┘                     │  ┌──────────────────────┐        │
                                     │  │  Flask/FastAPI server │        │
                                     │  │  /predict endpoint    │        │
                                     │  │  validation + auth    │        │
                                     │  └──────────────────────┘        │
                                     │           │                       │
                                     │  [Load balancer]                  │
                                     │  [Monitoring + alerts]            │
                                     │  [A/B testing router]             │
                                     └──────────────────────────────────┘

Key steps: serialize → containerize → serve → monitor → iterate
```

---

## 1. Model Serialization

### Why Serialization Matters

Training saves a model. Deployment needs to load it — possibly in a different process, on a different machine, in a different language. Serialization format determines portability, size, and load speed.

| Format | Use case | Portability | Size |
|---|---|---|---|
| `pickle` | Python-only, quick experiments | Python only | Large |
| `joblib` | Sklearn models, arrays | Python only | Smaller (mmap) |
| `ONNX` | Cross-framework, cross-language | Any ONNX runtime | Medium |
| `TorchScript` | PyTorch production | C++ / Python | Same as model |
| `safetensors` | LLM weights, fast loading | Python/Rust | Same as weights |

```python
import numpy as np
import json
import struct
import os


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ── Minimal model to serialize ─────────────────────────────────
class LogisticRegression:
    """2-class logistic regression — the simplest production model."""
    
    def __init__(self, n_features: int):
        rng = np.random.default_rng(42)
        self.weights = rng.standard_normal(n_features) * 0.01
        self.bias = 0.0
        self.n_features = n_features
        self.metadata = {
            "model_type": "logistic_regression",
            "n_features": n_features,
            "version": "1.0.0",
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid(Xw + b) → probability of class 1."""
        logits = X @ self.weights + self.bias  # (batch,)
        return 1.0 / (1.0 + np.exp(-logits))   # sigmoid
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 100) -> "LogisticRegression":
        for _ in range(epochs):
            proba = self.predict_proba(X)       # (N,)
            error = proba - y                   # (N,)
            
            self.weights -= lr * (X.T @ error) / len(y)
            self.bias -= lr * error.mean()
        
        return self


# ── JSON serialization ─────────────────────────────────────────
def save_json(model: LogisticRegression, path: str) -> None:
    """Human-readable, portable — but slow for large arrays."""
    data = {
        "metadata": model.metadata,
        "weights": model.weights.tolist(),  # numpy → python list
        "bias": float(model.bias),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON: {os.path.getsize(path)} bytes")


def load_json(path: str) -> LogisticRegression:
    with open(path) as f:
        data = json.load(f)
    
    model = LogisticRegression(data["metadata"]["n_features"])
    model.weights = np.array(data["weights"])
    model.bias = data["bias"]
    model.metadata = data["metadata"]
    return model


# ── Binary serialization (compact) ────────────────────────────
def save_binary(model: LogisticRegression, path: str) -> None:
    """
    Custom binary format: header + raw float32 array.
    Format: [n_features: int32][bias: float64][weights: float32 × n]
    """
    with open(path, "wb") as f:
        # Header: 4-byte int for n_features
        f.write(struct.pack("<i", model.n_features))
        
        # Bias: 8-byte float64
        f.write(struct.pack("<d", model.bias))
        
        # Weights: float32 array (4 bytes each — saves 2× vs float64)
        f.write(model.weights.astype(np.float32).tobytes())
    
    print(f"Saved binary: {os.path.getsize(path)} bytes")


def load_binary(path: str) -> LogisticRegression:
    with open(path, "rb") as f:
        n_features = struct.unpack("<i", f.read(4))[0]
        bias = struct.unpack("<d", f.read(8))[0]
        weights = np.frombuffer(f.read(n_features * 4), dtype=np.float32).astype(np.float64)
    
    model = LogisticRegression(n_features)
    model.weights = weights
    model.bias = bias
    return model


section("Serialization Demo")

# Train a model
rng = np.random.default_rng(42)
n, d = 200, 10
X = rng.standard_normal((n, d))
w_true = rng.standard_normal(d)
y = (X @ w_true > 0).astype(float)

model = LogisticRegression(d)
model.fit(X, y, lr=0.1, epochs=200)
acc = np.mean(model.predict(X) == y)
print(f"Training accuracy: {acc:.3f}")

# Serialize and reload
save_json(model, "/tmp/model.json")
save_binary(model, "/tmp/model.bin")

model_json = load_json("/tmp/model.json")
model_bin = load_binary("/tmp/model.bin")

# Verify identical predictions
preds_orig = model.predict(X[:5])
preds_json = model_json.predict(X[:5])
preds_bin = model_bin.predict(X[:5])
print(f"JSON reload matches: {np.array_equal(preds_orig, preds_json)}")
print(f"Binary reload matches: {np.array_equal(preds_orig, preds_bin)}")

json_size = os.path.getsize("/tmp/model.json")
bin_size = os.path.getsize("/tmp/model.bin")
print(f"JSON: {json_size} bytes, Binary: {bin_size} bytes, Ratio: {json_size/bin_size:.1f}×")
```

---

## 2. Inference Optimization

### Batching

The single biggest performance lever. A GPU processes a batch of 64 samples almost as fast as 1 sample — hardware utilization goes from 2% to 90%.

```python
import time

class ModelServer:
    """Demonstrates batching impact on throughput."""
    
    def __init__(self, model: LogisticRegression):
        self.model = model
        self.n_processed = 0
    
    def _simulate_gpu_forward(self, X: np.ndarray) -> np.ndarray:
        """
        Simulate GPU latency: fixed overhead + tiny per-sample cost.
        This models real GPU behavior: fixed kernel launch overhead dominates for small batches.
        """
        fixed_overhead_ms = 2.0    # kernel launch, memory transfer
        per_sample_ms = 0.01       # actual computation per sample
        
        total_ms = fixed_overhead_ms + per_sample_ms * len(X)
        time.sleep(total_ms / 1000)  # simulate latency
        
        self.n_processed += len(X)
        return self.model.predict_proba(X)
    
    def benchmark(self, X: np.ndarray, batch_size: int) -> dict:
        """Measure throughput at a given batch size."""
        n_batches = len(X) // batch_size
        if n_batches == 0:
            n_batches = 1
        
        start = time.perf_counter()
        total_samples = 0
        
        for i in range(n_batches):
            batch = X[i*batch_size:(i+1)*batch_size]
            if len(batch) == 0:
                break
            _ = self._simulate_gpu_forward(batch)
            total_samples += len(batch)
        
        elapsed = time.perf_counter() - start
        throughput = total_samples / elapsed
        latency_ms = elapsed / n_batches * 1000
        
        return {
            "batch_size": batch_size,
            "total_samples": total_samples,
            "throughput_qps": throughput,
            "avg_latency_ms": latency_ms,
        }


section("Batching Impact on Throughput")
X_bench = rng.standard_normal((256, d))
server = ModelServer(model)

print(f"{'Batch size':>12} {'Throughput (QPS)':>18} {'Latency (ms)':>14}")
print("-" * 48)
for bs in [1, 4, 8, 16, 32, 64]:
    stats = server.benchmark(X_bench, batch_size=bs)
    print(f"{bs:>12} {stats['throughput_qps']:>18.1f} {stats['avg_latency_ms']:>14.2f}")
```

### Request Batching Queue

In production, requests arrive one-at-a-time. A batching queue collects them and fires GPU calls in groups:

```python
import threading
import queue
from dataclasses import dataclass

@dataclass
class InferenceRequest:
    """One client request waiting for a prediction."""
    features: np.ndarray          # input to model
    result_event: threading.Event # signals when result is ready
    result: list                   # written by server, read by client


class DynamicBatchingServer:
    """
    Collects incoming requests, fires them as a batch every `max_wait_ms` ms
    or when batch reaches `max_batch_size` — whichever comes first.
    """
    
    def __init__(self, model: LogisticRegression, max_batch_size: int = 8, max_wait_ms: float = 10.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        self.request_queue: queue.Queue = queue.Queue()
        self.running = True
        
        # Worker thread processes batches
        self.worker = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker.start()
    
    def predict(self, features: np.ndarray) -> float:
        """Submit one request and block until result arrives."""
        req = InferenceRequest(
            features=features,
            result_event=threading.Event(),
            result=[]
        )
        self.request_queue.put(req)
        req.result_event.wait(timeout=5.0)  # block until result written
        return req.result[0] if req.result else -1.0
    
    def _batch_worker(self) -> None:
        """Background thread: batch requests and process together."""
        while self.running:
            requests: list[InferenceRequest] = []
            
            # Wait for first request
            try:
                first = self.request_queue.get(timeout=0.1)
                requests.append(first)
            except queue.Empty:
                continue
            
            # Collect more requests up to max_wait_ms
            deadline = time.perf_counter() + self.max_wait_ms / 1000
            while len(requests) < self.max_batch_size and time.perf_counter() < deadline:
                try:
                    req = self.request_queue.get_nowait()
                    requests.append(req)
                except queue.Empty:
                    time.sleep(0.001)
            
            # Process batch
            batch = np.array([r.features for r in requests])
            probas = self.model.predict_proba(batch)
            
            # Return results to waiting clients
            for req, prob in zip(requests, probas):
                req.result.append(float(prob))
                req.result_event.set()  # unblock the waiting client thread
    
    def shutdown(self) -> None:
        self.running = False
        self.worker.join(timeout=2.0)


section("Dynamic Batching Server Demo")
batch_server = DynamicBatchingServer(model, max_batch_size=4, max_wait_ms=5.0)

# Simulate 10 concurrent requests
results = []
def make_request(idx: int) -> None:
    x = rng.standard_normal(d)
    proba = batch_server.predict(x)
    results.append((idx, proba))

threads = [threading.Thread(target=make_request, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

batch_server.shutdown()

results.sort()
print(f"Processed {len(results)} requests via dynamic batching:")
for idx, proba in results[:5]:
    print(f"  Request {idx}: probability = {proba:.4f}")
```

### Model Quantization (Inference)

Post-training quantization reduces model size and speeds up inference without retraining.

```
Quantization — trading precision for speed and memory:

FP32 (float32):  4 bytes/param, full precision
  [0.3142, -0.2718, 0.1414, ...]  ← 32 bits each, ~±3.4×10^38 range

FP16 (float16):  2 bytes/param, half precision
  [0.314, -0.272, 0.141, ...]     ← 16 bits each, still good for inference

INT8:             1 byte/param, 256 levels
  [40, -35, 18, ...]              ← integer 0-255 mapped via scale factor
  Dequantize: actual_value = int_value × scale

INT4 (NF4):      0.5 bytes/param, 16 levels
  [0100, 1011, 0010, ...]         ← 4 bits, used in QLoRA

Memory comparison for a 7B param model:
  FP32:  28 GB  (7B × 4 bytes)
  FP16:  14 GB  (7B × 2 bytes)
  INT8:   7 GB  (7B × 1 byte)
  INT4: 3.5 GB  (7B × 0.5 bytes)  ← fits on a consumer GPU!

Speed:  INT8 matrix multiply is 2-4× faster than FP32 on modern hardware.
Accuracy drop:  FP16 ≈ 0%, INT8 ≈ 0.5%, INT4 ≈ 1-2% on typical tasks.
```

```python
class QuantizedLinear:
    """
    Simulates INT8 quantization of a weight matrix.
    
    Quantization: map float32 range [min, max] to int8 range [-128, 127].
    scale = (max - min) / 255
    zero_point = round(-min / scale) - 128
    quantized = round(x / scale + zero_point)
    """
    
    def __init__(self, weight: np.ndarray):
        self.original_shape = weight.shape
        
        # Compute quantization parameters
        w_min, w_max = weight.min(), weight.max()
        self.scale = (w_max - w_min) / 255.0 + 1e-8
        self.zero_point = int(round(-w_min / self.scale)) - 128
        self.zero_point = max(-128, min(127, self.zero_point))
        
        # Quantize to int8
        quantized = np.round(weight / self.scale + self.zero_point)
        self.W_int8 = np.clip(quantized, -128, 127).astype(np.int8)
    
    def dequantize(self) -> np.ndarray:
        """Recover approximate float32 weights."""
        return (self.W_int8.astype(np.float32) - self.zero_point) * self.scale
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: dequantize on the fly."""
        W = self.dequantize()
        return x @ W.T
    
    @property
    def compression_ratio(self) -> float:
        """float32 size / int8 size = 4× for pure weight storage."""
        return 4.0
    
    def quantization_error(self, original_weight: np.ndarray) -> float:
        return float(np.mean(np.abs(original_weight - self.dequantize())))


section("INT8 Quantization")
W = rng.standard_normal((32, 16)) * 0.1  # simulate weight matrix
quant = QuantizedLinear(W)

x = rng.standard_normal((8, 16))
out_float = x @ W.T
out_quant = quant.forward(x)

mse = np.mean((out_float - out_quant) ** 2)
error_pct = quant.quantization_error(W) / (np.abs(W).mean() + 1e-9) * 100

print(f"Weight shape: {W.shape}")
print(f"INT8 compression: {quant.compression_ratio}×")
print(f"Quantization error: {error_pct:.2f}%")
print(f"Output MSE (float vs quant): {mse:.6f}")
print(f"W_int8 dtype: {quant.W_int8.dtype}, W original dtype: {W.dtype}")
```

---

## 3. Serving Architecture

### WSGI vs ASGI

```
WSGI (Flask): synchronous — one request at a time per worker
ASGI (FastAPI): asynchronous — many requests concurrently per worker

For ML: use async for I/O (database, external APIs)
        use sync for CPU/GPU compute (model forward pass)
```

### Production Flask Server with Middleware

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time
import threading
from collections import deque


class MetricsCollector:
    """Thread-safe latency and error tracking."""
    
    def __init__(self, window: int = 1000):
        self.lock = threading.Lock()
        self.latencies: deque = deque(maxlen=window)  # ms
        self.error_count = 0
        self.request_count = 0
    
    def record(self, latency_ms: float, is_error: bool = False) -> None:
        with self.lock:
            self.latencies.append(latency_ms)
            self.request_count += 1
            if is_error:
                self.error_count += 1
    
    def summary(self) -> dict:
        with self.lock:
            if not self.latencies:
                return {}
            lats = sorted(self.latencies)
            n = len(lats)
            return {
                "p50_ms": lats[n // 2],
                "p95_ms": lats[int(n * 0.95)],
                "p99_ms": lats[int(n * 0.99)],
                "mean_ms": sum(lats) / n,
                "total_requests": self.request_count,
                "error_rate": self.error_count / max(self.request_count, 1),
            }


class InputValidator:
    """Validates and cleans incoming prediction requests."""
    
    def __init__(self, n_features: int, feature_names: list[str] | None = None):
        self.n_features = n_features
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
    
    def validate(self, payload: dict) -> tuple[np.ndarray, str | None]:
        """
        Returns (features_array, error_message).
        error_message is None if valid.
        """
        if "features" not in payload:
            return None, "Missing 'features' field in request body"
        
        features = payload["features"]
        
        if not isinstance(features, list):
            return None, f"'features' must be a list, got {type(features).__name__}"
        
        if len(features) != self.n_features:
            return None, f"Expected {self.n_features} features, got {len(features)}"
        
        try:
            arr = np.array(features, dtype=np.float32)
        except (ValueError, TypeError) as e:
            return None, f"Features must be numeric: {e}"
        
        # Check for NaN/Inf
        if not np.isfinite(arr).all():
            return None, "Features contain NaN or Inf values"
        
        return arr.reshape(1, -1), None


class MLServer:
    """
    Production ML serving class (without HTTP dependencies for demonstration).
    Shows the full request handling pipeline.
    """
    
    def __init__(self, model: LogisticRegression):
        self.model = model
        self.validator = InputValidator(model.n_features)
        self.metrics = MetricsCollector()
        self.request_id = 0
        self.lock = threading.Lock()
    
    def _next_request_id(self) -> int:
        with self.lock:
            self.request_id += 1
            return self.request_id
    
    def predict(self, payload: dict) -> dict:
        """Full prediction pipeline: validate → infer → respond."""
        start = time.perf_counter()
        req_id = self._next_request_id()
        is_error = False
        
        try:
            # 1. Validate input
            features, error = self.validator.validate(payload)
            if error:
                is_error = True
                return {
                    "request_id": req_id,
                    "status": "error",
                    "message": error,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                }
            
            # 2. Run inference
            proba = float(self.model.predict_proba(features)[0])
            predicted_class = int(proba >= 0.5)
            
            # 3. Build response
            response = {
                "request_id": req_id,
                "status": "ok",
                "prediction": predicted_class,
                "probability": round(proba, 4),
                "confidence": round(max(proba, 1 - proba), 4),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
            return response
        
        except Exception as e:
            is_error = True
            return {
                "request_id": req_id,
                "status": "error",
                "message": f"Internal error: {str(e)}",
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        finally:
            latency = (time.perf_counter() - start) * 1000
            self.metrics.record(latency, is_error)
    
    def health(self) -> dict:
        """Health check endpoint — used by load balancers."""
        return {
            "status": "healthy",
            "model_version": self.model.metadata.get("version", "unknown"),
            "metrics": self.metrics.summary(),
        }


section("ML Server Demo")
ml_server = MLServer(model)

# Valid request
valid_payload = {"features": rng.standard_normal(d).tolist()}
response = ml_server.predict(valid_payload)
print(f"Valid request: {json.dumps(response, indent=2)}")

# Invalid requests
invalid_cases = [
    {},                                              # missing features
    {"features": [1.0] * (d - 1)},                  # wrong length
    {"features": ["a", "b"] + [0.0] * (d - 2)},     # non-numeric
    {"features": [float('nan')] + [0.0] * (d - 1)}, # NaN
]
for case in invalid_cases:
    resp = ml_server.predict(case)
    print(f"\n[{resp['status']}] {resp.get('message', '')}")

# Simulate load and check metrics
for _ in range(50):
    x = rng.standard_normal(d).tolist()
    ml_server.predict({"features": x})

print(f"\nMetrics summary:")
print(json.dumps(ml_server.health(), indent=2))
```

---

## 4. Monitoring and Observability

### Data Drift Detection

Model accuracy silently degrades when the input distribution shifts. Detect this before users notice.

```python
from collections import Counter

class DriftDetector:
    """
    Detects statistical shift between training distribution and live data.
    Uses Population Stability Index (PSI) for numerical features.
    
    PSI < 0.1:  no drift
    PSI 0.1-0.2: minor drift — monitor
    PSI > 0.2:  significant drift — retrain
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.train_stats: dict | None = None
    
    def fit(self, X_train: np.ndarray) -> "DriftDetector":
        """Compute reference distribution from training data."""
        self.train_stats = {
            "mean": X_train.mean(axis=0),    # shape (n_features,)
            "std": X_train.std(axis=0) + 1e-8,
            "min": X_train.min(axis=0),
            "max": X_train.max(axis=0),
            "bin_edges": [],
            "bin_freqs": [],
        }
        
        # Build per-feature histograms
        for j in range(X_train.shape[1]):
            counts, edges = np.histogram(X_train[:, j], bins=self.n_bins)
            freqs = counts / max(counts.sum(), 1)
            freqs = np.clip(freqs, 1e-4, None)  # avoid log(0)
            self.train_stats["bin_edges"].append(edges)
            self.train_stats["bin_freqs"].append(freqs)
        
        return self
    
    def compute_psi(self, X_live: np.ndarray) -> np.ndarray:
        """
        PSI = Σ (actual% - expected%) × ln(actual% / expected%)
        Returns PSI per feature.
        """
        if self.train_stats is None:
            raise RuntimeError("Call fit() first")
        
        psi_scores = np.zeros(X_live.shape[1])
        
        for j in range(X_live.shape[1]):
            edges = self.train_stats["bin_edges"][j]
            expected = self.train_stats["bin_freqs"][j]
            
            # Compute live distribution using same bin edges
            live_counts, _ = np.histogram(X_live[:, j], bins=edges)
            live_freqs = live_counts / max(live_counts.sum(), 1)
            live_freqs = np.clip(live_freqs, 1e-4, None)
            
            # PSI formula
            psi = np.sum((live_freqs - expected) * np.log(live_freqs / expected))
            psi_scores[j] = psi
        
        return psi_scores
    
    def report(self, X_live: np.ndarray) -> dict:
        psi = self.compute_psi(X_live)
        
        def severity(v):
            if v < 0.1: return "stable"
            if v < 0.2: return "warning"
            return "DRIFT DETECTED"
        
        results = {}
        for j, p in enumerate(psi):
            results[f"feature_{j}"] = {
                "psi": round(float(p), 4),
                "status": severity(p)
            }
        
        overall = float(psi.mean())
        results["overall"] = {"psi": round(overall, 4), "status": severity(overall)}
        return results


section("Data Drift Detection")

# Training distribution: N(0,1)
X_train = rng.standard_normal((1000, d))

detector = DriftDetector(n_bins=10)
detector.fit(X_train)

# Scenario 1: same distribution (should show no drift)
X_same = rng.standard_normal((500, d))
report_same = detector.report(X_same)
print("Same distribution (expected: stable):")
print(f"  Overall PSI: {report_same['overall']['psi']:.4f} — {report_same['overall']['status']}")

# Scenario 2: shifted distribution (mean shift of 1.5)
X_shifted = rng.standard_normal((500, d)) + 1.5
report_shifted = detector.report(X_shifted)
print("\nShifted distribution (expected: DRIFT):")
print(f"  Overall PSI: {report_shifted['overall']['psi']:.4f} — {report_shifted['overall']['status']}")
for feat, stats in list(report_shifted.items())[:4]:
    print(f"  {feat}: PSI={stats['psi']:.4f} — {stats['status']}")
```

### Prediction Confidence Monitoring

Log when the model is uncertain — low-confidence predictions are worth flagging:

```python
class PredictionMonitor:
    """
    Tracks prediction confidence distribution.
    Low confidence predictions → model may be seeing OOD data.
    """
    
    def __init__(self, low_conf_threshold: float = 0.6):
        self.threshold = low_conf_threshold
        self.all_probas: list[float] = []
        self.low_conf_count = 0
        self.total = 0
    
    def record(self, proba: float) -> None:
        confidence = max(proba, 1 - proba)  # distance from 0.5
        self.all_probas.append(confidence)
        self.total += 1
        if confidence < self.threshold:
            self.low_conf_count += 1
    
    def report(self) -> dict:
        if not self.all_probas:
            return {}
        
        arr = np.array(self.all_probas)
        return {
            "total_predictions": self.total,
            "low_confidence_rate": self.low_conf_count / self.total,
            "mean_confidence": float(arr.mean()),
            "p10_confidence": float(np.percentile(arr, 10)),
            "alert": self.low_conf_count / self.total > 0.3,  # > 30% low conf = alert
        }


section("Prediction Confidence Monitoring")
monitor = PredictionMonitor(low_conf_threshold=0.65)

# Simulate predictions on in-distribution data
for _ in range(200):
    x = rng.standard_normal((1, d))
    proba = float(model.predict_proba(x)[0])
    monitor.record(proba)

report = monitor.report()
print("In-distribution predictions:")
print(json.dumps(report, indent=2))
```

---

## 5. A/B Testing for Models

Deploy two model versions, route traffic, measure which performs better:

```python
class ABTest:
    """
    Routes requests between two model versions.
    Tracks outcomes to detect which model performs better.
    """
    
    def __init__(self, model_a, model_b, traffic_split: float = 0.5, seed: int = 42):
        """
        traffic_split: fraction sent to model_a (0.5 = 50/50 split)
        """
        self.models = {"a": model_a, "b": model_b}
        self.split = traffic_split
        self.rng = np.random.default_rng(seed)
        
        self.results: dict[str, list] = {"a": [], "b": []}
        self.assignments: dict[str, int] = {"a": 0, "b": 0}
    
    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Route to model A or B based on split. Returns (variant, probability)."""
        variant = "a" if self.rng.random() < self.split else "b"
        self.assignments[variant] += 1
        
        proba = float(self.models[variant].predict_proba(features.reshape(1, -1))[0])
        return variant, proba
    
    def record_outcome(self, variant: str, correct: bool) -> None:
        """Record whether this prediction was correct (in A/B test, truth comes later)."""
        self.results[variant].append(int(correct))
    
    def analyze(self) -> dict:
        """Compare model performance with statistical significance."""
        results = {}
        for variant in ["a", "b"]:
            outcomes = self.results[variant]
            if not outcomes:
                continue
            accuracy = sum(outcomes) / len(outcomes)
            results[variant] = {
                "n": len(outcomes),
                "accuracy": round(accuracy, 4),
                "requests": self.assignments[variant],
            }
        
        # Simple z-test for difference in proportions
        if "a" in results and "b" in results:
            p_a = results["a"]["accuracy"]
            p_b = results["b"]["accuracy"]
            n_a = results["a"]["n"]
            n_b = results["b"]["n"]
            
            if n_a > 0 and n_b > 0:
                # Pooled proportion
                p_pool = (p_a * n_a + p_b * n_b) / (n_a + n_b)
                se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
                z = (p_a - p_b) / (se + 1e-9)
                
                results["statistical_test"] = {
                    "z_score": round(float(z), 3),
                    "significant_at_95pct": abs(z) > 1.96,
                    "winner": "a" if z > 1.96 else ("b" if z < -1.96 else "inconclusive"),
                }
        
        return results


section("A/B Testing")

# Model A: trained normally
model_a = LogisticRegression(d)
model_a.fit(X, y, lr=0.1, epochs=200)

# Model B: slightly different hyperparams (simulated by adding noise to weights)
model_b = LogisticRegression(d)
model_b.fit(X, y, lr=0.05, epochs=400)

ab_test = ABTest(model_a, model_b, traffic_split=0.5)

# Simulate 500 requests with known ground truth
X_test = rng.standard_normal((500, d))
y_test = (X_test @ w_true > 0).astype(int)

for i in range(500):
    variant, proba = ab_test.predict(X_test[i])
    predicted = int(proba >= 0.5)
    ab_test.record_outcome(variant, predicted == y_test[i])

analysis = ab_test.analyze()
print(json.dumps(analysis, indent=2))
```

---

## 6. CI/CD for ML

### Model Validation Before Deployment

```python
class ModelValidator:
    """
    Pre-deployment validation checks.
    A model fails validation → it doesn't ship.
    """
    
    def __init__(self):
        self.checks: list[tuple[str, callable]] = []
    
    def add_check(self, name: str, fn: callable) -> "ModelValidator":
        self.checks.append((name, fn))
        return self
    
    def run(self, model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        results = {"passed": [], "failed": []}
        
        for name, check_fn in self.checks:
            try:
                passed, message = check_fn(model, X_val, y_val)
                if passed:
                    results["passed"].append({"check": name, "message": message})
                else:
                    results["failed"].append({"check": name, "message": message})
            except Exception as e:
                results["failed"].append({"check": name, "message": f"Error: {e}"})
        
        results["all_passed"] = len(results["failed"]) == 0
        return results


def check_accuracy(min_accuracy: float):
    def _check(model, X, y):
        acc = np.mean(model.predict(X) == y)
        return acc >= min_accuracy, f"Accuracy={acc:.4f} (required >= {min_accuracy})"
    return _check


def check_latency(max_ms: float, n_samples: int = 100):
    def _check(model, X, y):
        start = time.perf_counter()
        _ = model.predict(X[:n_samples])
        latency = (time.perf_counter() - start) * 1000 / n_samples
        return latency <= max_ms, f"Latency={latency:.3f}ms/sample (max {max_ms}ms)"
    return _check


def check_calibration(max_ece: float):
    """Expected Calibration Error: predicted probabilities should match actual frequencies."""
    def _check(model, X, y):
        proba = model.predict_proba(X)
        
        # Bin predictions into 10 buckets
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (proba >= bins[i]) & (proba < bins[i+1])
            if mask.sum() == 0:
                continue
            bin_confidence = proba[mask].mean()
            bin_accuracy = y[mask].mean()
            ece += mask.sum() / len(y) * abs(bin_confidence - bin_accuracy)
        
        return ece <= max_ece, f"ECE={ece:.4f} (max {max_ece})"
    return _check


section("Pre-Deployment Validation")
X_val = rng.standard_normal((200, d))
y_val = (X_val @ w_true > 0).astype(int)

validator = (ModelValidator()
    .add_check("accuracy", check_accuracy(min_accuracy=0.80))
    .add_check("latency", check_latency(max_ms=10.0))
    .add_check("calibration", check_calibration(max_ece=0.15))
)

results = validator.run(model, X_val, y_val)

print(f"Validation {'PASSED' if results['all_passed'] else 'FAILED'}:")
for check in results["passed"]:
    print(f"  ✓ {check['check']}: {check['message']}")
for check in results["failed"]:
    print(f"  ✗ {check['check']}: {check['message']}")
```

---

## 7. Interview Q&A

**Q: What's the difference between model latency and throughput?**
Latency is the time for one request from arrival to response (milliseconds). Throughput is requests processed per second across all requests. Batching increases throughput but increases per-request latency. You optimize for one based on SLA requirements.

**Q: Explain p50/p95/p99 latency. Which matters most?**
p50: median — half of requests are faster. p95: 95% are faster — the typical "slow" user experience. p99: 99% are faster — catches rare but severe spikes. For user-facing: p95 or p99. For batch: p50 suffices. p99 matters most for SLA commitments because outliers cause timeouts.

**Q: What causes model serving latency?**
1. Input preprocessing (tokenization, normalization)
2. Model forward pass (compute)
3. Output postprocessing
4. Network round-trip (client ↔ server)
5. Queue wait time (under high load)
6. Memory transfer (CPU↔GPU)

**Q: What is data drift and how do you detect it?**
Covariate drift: input distribution $P(X)$ changes. Label drift: output distribution $P(Y)$ changes. Concept drift: relationship $P(Y|X)$ changes. Detect with: PSI (Population Stability Index), KL divergence, K-S test (Kolmogorov-Smirnov), or directly monitoring accuracy on labeled samples.

**Q: How would you design a rollback strategy for model deployments?**
Blue-green: run old model (blue) and new model (green) simultaneously. Route 100% to green after validation; if failure, flip back to blue instantly. Canary: route 1% → 10% → 50% → 100% traffic to new model incrementally, monitoring error rate at each stage. Shadow: new model runs on all traffic but responses are not sent to users — compare outputs with production model.

**Q: What metrics would you monitor for an ML API in production?**
Infrastructure: CPU/GPU utilization, memory, request queue depth. Model: prediction latency (p50/p95/p99), throughput (QPS), error rate. Data quality: feature drift (PSI), null rates, out-of-range values. Business: model accuracy (if labels available), prediction distribution shift.

---

## 8. Cheat Sheet

```
SERIALIZATION
  JSON:       human-readable; slow for large arrays
  binary:     compact; fast; not human-readable
  ONNX:       cross-framework; hardware-optimized runtimes
  safetensors: LLM-scale; memory-mapped; safe (no pickle)

INFERENCE OPTIMIZATION
  batching:      fixed overhead amortized across N samples
  quantization:  INT8=2× smaller; INT4=4× smaller; small accuracy loss
  caching:       cache repeated inputs (KV cache, request dedup)
  async:         non-blocking I/O; GPU runs while CPU preprocesses

MONITORING
  PSI < 0.10:  stable
  PSI 0.10-0.20: minor drift
  PSI > 0.20:  retrain needed
  
  ECE:         calibration — predicted proba ≈ actual frequency
  p99 latency: SLA-critical metric
  
DEPLOYMENT PATTERNS
  blue-green:  instant rollback; requires 2× resources
  canary:      gradual traffic shift; early failure detection
  shadow:      parallel inference; no user impact; compare outputs
  A/B test:    statistical comparison; needs outcome labels

VALIDATION CHECKLIST
  □ Accuracy >= baseline
  □ Latency within SLA
  □ No data leakage from future data
  □ Calibrated probabilities (ECE)
  □ Handles edge cases (nulls, OOV, extreme values)
  □ Memory footprint acceptable
```

---

## Mini-Project: Production ML Service

Build a complete ML service with model training, validation, serving, drift monitoring, and A/B testing.

```python
# production_service.py
import numpy as np
import json
import time
import threading
from collections import deque


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


class LogisticRegression:
    def __init__(self, n_features: int, version: str = "1.0"):
        rng = np.random.default_rng(42)
        self.weights = rng.standard_normal(n_features) * 0.01
        self.bias = 0.0
        self.n_features = n_features
        self.version = version
        self.trained = False
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(X @ self.weights + self.bias)))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 200) -> "LogisticRegression":
        for _ in range(epochs):
            p = self.predict_proba(X)
            e = p - y
            self.weights -= lr * (X.T @ e) / len(y)
            self.bias -= lr * e.mean()
        self.trained = True
        return self
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


class ProductionService:
    """
    Full ML service with monitoring, drift detection, and A/B testing.
    """
    
    def __init__(self):
        self.models: dict[str, LogisticRegression] = {}
        self.active_model: str | None = None
        self.request_log: deque = deque(maxlen=1000)
        self.drift_reference: np.ndarray | None = None
        self._req_id = 0
        self._lock = threading.Lock()
    
    def train_and_register(self, name: str, X: np.ndarray, y: np.ndarray,
                           lr: float = 0.1, epochs: int = 200) -> float:
        """Train a model and register it (not yet active)."""
        model = LogisticRegression(X.shape[1], version=name)
        model.fit(X, y, lr=lr, epochs=epochs)
        self.models[name] = model
        
        train_acc = model.score(X, y)
        print(f"  Registered model '{name}': train accuracy = {train_acc:.4f}")
        return train_acc
    
    def set_drift_reference(self, X: np.ndarray) -> None:
        """Set baseline distribution for drift detection."""
        self.drift_reference = X.copy()
    
    def validate_and_deploy(self, name: str, X_val: np.ndarray, y_val: np.ndarray,
                            min_accuracy: float = 0.80) -> bool:
        """Run pre-deployment checks; deploy if all pass."""
        if name not in self.models:
            print(f"  Model '{name}' not found")
            return False
        
        model = self.models[name]
        val_acc = model.score(X_val, y_val)
        
        # Check 1: accuracy
        if val_acc < min_accuracy:
            print(f"  BLOCKED: '{name}' accuracy {val_acc:.4f} < {min_accuracy}")
            return False
        
        # Check 2: latency
        start = time.perf_counter()
        _ = model.predict(X_val[:100])
        latency = (time.perf_counter() - start) * 10  # per-sample ms
        
        if latency > 1.0:  # 1ms threshold
            print(f"  BLOCKED: '{name}' latency {latency:.3f}ms > 1ms")
            return False
        
        # All checks passed
        self.active_model = name
        print(f"  DEPLOYED: '{name}' (val_acc={val_acc:.4f}, latency={latency:.4f}ms)")
        return True
    
    def predict(self, features: list) -> dict:
        """Handle one prediction request."""
        with self._lock:
            self._req_id += 1
            req_id = self._req_id
        
        if self.active_model is None:
            return {"error": "No active model", "request_id": req_id}
        
        start = time.perf_counter()
        
        try:
            x = np.array(features, dtype=np.float32).reshape(1, -1)
            model = self.models[self.active_model]
            proba = float(model.predict_proba(x)[0])
            latency_ms = (time.perf_counter() - start) * 1000
            
            record = {
                "request_id": req_id,
                "model": self.active_model,
                "features": features,
                "probability": round(proba, 4),
                "prediction": int(proba >= 0.5),
                "latency_ms": round(latency_ms, 3),
            }
            self.request_log.append(record)
            return record
        
        except Exception as e:
            return {"error": str(e), "request_id": req_id}
    
    def detect_drift(self, X_live: np.ndarray, n_bins: int = 10) -> dict:
        """Compare live vs reference distribution using PSI."""
        if self.drift_reference is None:
            return {"error": "No reference distribution set"}
        
        psi_scores = []
        for j in range(X_live.shape[1]):
            ref_col = self.drift_reference[:, j]
            live_col = X_live[:, j]
            
            edges = np.histogram_bin_edges(ref_col, bins=n_bins)
            ref_freq, _ = np.histogram(ref_col, bins=edges)
            live_freq, _ = np.histogram(live_col, bins=edges)
            
            ref_freq = np.clip(ref_freq / max(ref_freq.sum(), 1), 1e-4, None)
            live_freq = np.clip(live_freq / max(live_freq.sum(), 1), 1e-4, None)
            
            psi = float(np.sum((live_freq - ref_freq) * np.log(live_freq / ref_freq)))
            psi_scores.append(psi)
        
        overall = float(np.mean(psi_scores))
        return {
            "overall_psi": round(overall, 4),
            "status": "stable" if overall < 0.1 else ("warning" if overall < 0.2 else "DRIFT"),
            "per_feature": [round(p, 4) for p in psi_scores],
        }
    
    def metrics(self) -> dict:
        """Aggregate metrics from request log."""
        if not self.request_log:
            return {"total_requests": 0}
        
        latencies = [r["latency_ms"] for r in self.request_log if "latency_ms" in r]
        predictions = [r["prediction"] for r in self.request_log if "prediction" in r]
        
        lats = sorted(latencies)
        n = len(lats)
        return {
            "total_requests": len(self.request_log),
            "active_model": self.active_model,
            "latency_p50": lats[n//2] if lats else 0,
            "latency_p95": lats[int(n*0.95)] if lats else 0,
            "prediction_rate_class1": sum(predictions) / max(len(predictions), 1),
        }


def main():
    rng = np.random.default_rng(42)
    d = 12  # features
    
    section("1. Generate Data")
    w_true = rng.standard_normal(d)
    X_all = rng.standard_normal((1500, d))
    y_all = (X_all @ w_true > 0).astype(float)
    
    X_train, y_train = X_all[:800], y_all[:800]
    X_val, y_val = X_all[800:1000], y_all[800:1000]
    X_test = X_all[1000:]
    y_test = y_all[1000:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    section("2. Train and Register Models")
    service = ProductionService()
    service.set_drift_reference(X_train)
    
    service.train_and_register("v1", X_train, y_train, lr=0.1, epochs=200)
    service.train_and_register("v2", X_train, y_train, lr=0.05, epochs=400)
    
    section("3. Validate and Deploy")
    deployed = service.validate_and_deploy("v1", X_val, y_val, min_accuracy=0.80)
    if not deployed:
        print("Falling back to v2...")
        service.validate_and_deploy("v2", X_val, y_val, min_accuracy=0.75)
    
    section("4. Serve Predictions")
    for i in range(10):
        x = X_test[i].tolist()
        resp = service.predict(x)
        print(f"  Req {resp['request_id']}: pred={resp.get('prediction')}, "
              f"proba={resp.get('probability'):.3f}, lat={resp.get('latency_ms'):.3f}ms")
    
    section("5. Drift Detection")
    
    # No drift: same distribution
    drift_no = service.detect_drift(X_test[:200])
    print(f"Same distribution: PSI={drift_no['overall_psi']:.4f} — {drift_no['status']}")
    
    # Drift: shifted distribution
    X_drift = X_test[:200] + 2.0  # mean shift
    drift_yes = service.detect_drift(X_drift)
    print(f"Shifted +2.0:       PSI={drift_yes['overall_psi']:.4f} — {drift_yes['status']}")
    
    section("6. Service Metrics")
    # Warm up with 100 requests
    for i in range(100):
        service.predict(X_test[i % len(X_test)].tolist())
    
    m = service.metrics()
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
```

### Expected Output

```
============================================================
1. Generate Data
============================================================
Train: 800, Val: 200, Test: 500

============================================================
2. Train and Register Models
============================================================
  Registered model 'v1': train accuracy = 0.9700
  Registered model 'v2': train accuracy = 0.9750

============================================================
3. Validate and Deploy
============================================================
  DEPLOYED: 'v1' (val_acc=0.9650, latency=0.0002ms)

============================================================
5. Drift Detection
============================================================
Same distribution: PSI=0.0034 — stable
Shifted +2.0:       PSI=1.2847 — DRIFT
```
