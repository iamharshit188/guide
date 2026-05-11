"""
Production health checks, readiness probes, and metrics endpoints.
Covers: /health (liveness), /ready (readiness), /metrics (Prometheus-style),
        rolling latency percentiles, circuit breaker pattern,
        graceful shutdown, and rolling deploy simulation.
pip install flask  (optional — core logic runs without it)
"""

import time
import threading
import json
from collections import deque

rng_seed = 42


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Ring Buffer Latency Store ─────────────────────────────────────
class LatencyTracker:
    """Thread-safe rolling window of request latencies and error counts."""

    def __init__(self, window: int = 1000):
        self._lock      = threading.Lock()
        self._latencies = deque(maxlen=window)
        self._statuses  = deque(maxlen=window)   # HTTP status codes
        self._start_time = time.time()

    def record(self, latency_ms: float, status_code: int):
        with self._lock:
            self._latencies.append(latency_ms)
            self._statuses.append(status_code)

    def percentile(self, p: float) -> float:
        with self._lock:
            lats = sorted(self._latencies)
        if not lats:
            return 0.0
        idx = int(len(lats) * p / 100)
        return lats[min(idx, len(lats) - 1)]

    def error_rate(self) -> float:
        with self._lock:
            codes = list(self._statuses)
        if not codes:
            return 0.0
        return sum(1 for c in codes if c >= 500) / len(codes)

    def request_count(self) -> int:
        with self._lock:
            return len(self._statuses)

    def throughput_rps(self) -> float:
        elapsed = time.time() - self._start_time
        return self.request_count() / elapsed if elapsed > 0 else 0.0

    def summary(self) -> dict:
        return {
            "p50_ms":     round(self.percentile(50), 3),
            "p95_ms":     round(self.percentile(95), 3),
            "p99_ms":     round(self.percentile(99), 3),
            "error_rate": round(self.error_rate(), 4),
            "requests":   self.request_count(),
            "rps":        round(self.throughput_rps(), 2),
        }


TRACKER = LatencyTracker()


# ── Model Registry ─────────────────────────────────────────────────
class ModelRegistry:
    """Thread-safe model store. Supports versioned models + warmup."""

    def __init__(self):
        self._lock    = threading.RLock()
        self._models  = {}
        self._default = None
        self._healthy = False
        self._loaded_at: float = 0

    def register(self, name: str, model, make_default: bool = False):
        with self._lock:
            self._models[name] = model
            if make_default:
                self._default = name
            self._healthy = True
            self._loaded_at = time.time()

    def get(self, name: str = None):
        with self._lock:
            key = name or self._default
            if key not in self._models:
                raise KeyError(f"Model '{key}' not found")
            return self._models[key]

    def is_healthy(self) -> bool:
        return self._healthy

    def list_models(self) -> list:
        with self._lock:
            return list(self._models.keys())

    def info(self) -> dict:
        with self._lock:
            return {
                "default":   self._default,
                "models":    list(self._models.keys()),
                "loaded_at": self._loaded_at,
                "healthy":   self._healthy,
            }


REGISTRY = ModelRegistry()


# ── Circuit Breaker ────────────────────────────────────────────────
class CircuitBreaker:
    """
    States: CLOSED (normal) → OPEN (blocking calls) → HALF_OPEN (testing)
    Opens when error_rate > threshold for the last N calls.
    Resets to HALF_OPEN after reset_timeout seconds.
    """

    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold: float = 0.5, min_calls: int = 10,
                 reset_timeout: float = 30.0):
        self._lock              = threading.Lock()
        self._state             = self.CLOSED
        self._failure_threshold = failure_threshold
        self._min_calls         = min_calls
        self._reset_timeout     = reset_timeout
        self._calls             = deque(maxlen=min_calls * 2)
        self._opened_at         = 0.0

    def call(self, fn, *args, **kwargs):
        with self._lock:
            state = self._state
            if state == self.OPEN:
                if time.time() - self._opened_at >= self._reset_timeout:
                    self._state = self.HALF_OPEN
                    state = self.HALF_OPEN
                else:
                    raise RuntimeError("Circuit breaker OPEN — call blocked")

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            self._calls.append(0)
            if self._state == self.HALF_OPEN:
                self._state = self.CLOSED

    def _on_failure(self):
        with self._lock:
            self._calls.append(1)
            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
                self._opened_at = time.time()
                return
            if len(self._calls) >= self._min_calls:
                error_rate = sum(self._calls) / len(self._calls)
                if error_rate >= self._failure_threshold:
                    self._state = self.OPEN
                    self._opened_at = time.time()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state


# ── Flask Health Endpoints (if available) ─────────────────────────
def build_flask_app():
    try:
        from flask import Flask, jsonify
    except ImportError:
        return None

    app = Flask(__name__)

    @app.route("/health")
    def health():
        """Liveness probe: always 200 if process is running."""
        return jsonify({"status": "ok", "timestamp": time.time()}), 200

    @app.route("/ready")
    def ready():
        """Readiness probe: 503 until model is loaded."""
        if not REGISTRY.is_healthy():
            return jsonify({"status": "not_ready", "reason": "model not loaded"}), 503
        return jsonify({"status": "ready", **REGISTRY.info()}), 200

    @app.route("/metrics")
    def metrics():
        """Prometheus-compatible text metrics + JSON summary."""
        summary = TRACKER.summary()
        prom_text = "\n".join([
            "# HELP request_latency_p50 P50 request latency in ms",
            f"request_latency_p50 {summary['p50_ms']}",
            f"request_latency_p95 {summary['p95_ms']}",
            f"request_latency_p99 {summary['p99_ms']}",
            f"request_error_rate {summary['error_rate']}",
            f"request_total {summary['requests']}",
            f"request_rps {summary['rps']}",
        ])
        return prom_text, 200, {"Content-Type": "text/plain"}

    @app.route("/models")
    def list_models():
        return jsonify(REGISTRY.info()), 200

    @app.route("/predict", methods=["POST"])
    def predict():
        from flask import request as req
        start = time.perf_counter()
        try:
            data  = req.get_json()
            model = REGISTRY.get(data.get("model"))
            result = {"prediction": model(data["features"])}
            latency_ms = (time.perf_counter() - start) * 1000
            TRACKER.record(latency_ms, 200)
            return jsonify(result), 200
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            TRACKER.record(latency_ms, 500)
            return jsonify({"error": str(e)}), 500

    return app


# ── Graceful Shutdown ─────────────────────────────────────────────
class GracefulShutdown:
    """
    Tracks in-flight requests. Shutdown waits until all complete.
    Register with SIGTERM/SIGINT handlers.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._count   = 0
        self._shutdown = False
        self._event   = threading.Event()

    def __enter__(self):
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Server is shutting down")
            self._count += 1
        return self

    def __exit__(self, *args):
        with self._lock:
            self._count -= 1
            if self._count == 0 and self._shutdown:
                self._event.set()

    def initiate(self, timeout: float = 30.0):
        self._shutdown = True
        print(f"  Shutdown initiated. Waiting for {self._count} in-flight requests...")
        self._event.wait(timeout=timeout)
        if self._count > 0:
            print(f"  Timeout: force-stopping with {self._count} requests still in-flight")
        else:
            print("  All in-flight requests completed. Clean shutdown.")


def simulate_requests(tracker: LatencyTracker, n: int = 200, error_rate: float = 0.05):
    """Simulate a load pattern with realistic latency distribution."""
    import random
    rng = random.Random(42)
    for i in range(n):
        # Bimodal latency: most fast (5–20ms), occasional slow (50–200ms)
        if rng.random() < 0.05:
            latency = rng.uniform(50, 200)   # slow request
        else:
            latency = rng.uniform(5, 20)     # fast request
        status = 500 if rng.random() < error_rate else 200
        tracker.record(latency, status)


def main():
    section("1. LATENCY TRACKER")
    simulate_requests(TRACKER, n=500, error_rate=0.04)
    summary = TRACKER.summary()
    print(f"  Simulated 500 requests (4% error rate)")
    print(f"  p50={summary['p50_ms']}ms  p95={summary['p95_ms']}ms  "
          f"p99={summary['p99_ms']}ms")
    print(f"  error_rate={summary['error_rate']}  "
          f"requests={summary['requests']}  rps≈{summary['rps']:.1f}")

    section("2. MODEL REGISTRY")
    # Simulate loading a model
    dummy_model = lambda x: sum(x)   # trivial model for demo
    REGISTRY.register("logistic_v1", dummy_model, make_default=True)
    REGISTRY.register("rf_v2",       dummy_model)

    print(f"  {REGISTRY.info()}")
    print(f"  healthy={REGISTRY.is_healthy()}")
    print(f"  models={REGISTRY.list_models()}")

    section("3. HEALTH / READY ENDPOINT SIMULATION")
    def make_health_response(registry):
        return {"status": "ok", "timestamp": round(time.time(), 2)}

    def make_ready_response(registry):
        if not registry.is_healthy():
            return 503, {"status": "not_ready", "reason": "model not loaded"}
        return 200, {"status": "ready", **registry.info()}

    health_status = make_health_response(REGISTRY)
    ready_code, ready_body = make_ready_response(REGISTRY)
    print(f"  GET /health → 200 {json.dumps(health_status)}")
    print(f"  GET /ready  → {ready_code} {json.dumps(ready_body)[:80]}...")

    # Simulate not-ready state
    empty_registry = ModelRegistry()
    nr_code, nr_body = make_ready_response(empty_registry)
    print(f"  GET /ready (no model) → {nr_code} {json.dumps(nr_body)}")

    section("4. PROMETHEUS METRICS FORMAT")
    prom_lines = [
        "# HELP request_latency_p50 P50 request latency in milliseconds",
        "# TYPE request_latency_p50 gauge",
        f"request_latency_p50 {summary['p50_ms']}",
        f"request_latency_p95 {summary['p95_ms']}",
        f"request_latency_p99 {summary['p99_ms']}",
        "# HELP request_error_rate Fraction of 5xx responses",
        "# TYPE request_error_rate gauge",
        f"request_error_rate {summary['error_rate']}",
        "# HELP request_total Total requests served",
        "# TYPE request_total counter",
        f"request_total {summary['requests']}",
    ]
    for line in prom_lines:
        print(f"  {line}")

    section("5. CIRCUIT BREAKER")
    cb = CircuitBreaker(failure_threshold=0.5, min_calls=10, reset_timeout=1.0)

    def unstable_call(fail: bool):
        if fail:
            raise RuntimeError("downstream service error")
        return "ok"

    print("  Sending 5 successes + 8 failures to trip the circuit breaker:")
    for i in range(5):
        try:
            result = cb.call(unstable_call, fail=False)
            print(f"    call {i+1}: success  state={cb.state}")
        except Exception as e:
            print(f"    call {i+1}: ERROR={e}  state={cb.state}")

    for i in range(8):
        try:
            result = cb.call(unstable_call, fail=True)
        except RuntimeError as e:
            print(f"    call {5+i+1}: ERROR={str(e)[:40]}  state={cb.state}")

    print(f"\n  Circuit is now {cb.state}")
    # Next call should be blocked
    try:
        cb.call(unstable_call, fail=False)
    except RuntimeError as e:
        print(f"  Call after OPEN: blocked → '{e}'")

    # Wait for reset timeout and transition to HALF_OPEN
    print(f"  Waiting 1.1s for reset timeout...")
    time.sleep(1.1)
    try:
        cb.call(unstable_call, fail=False)
        print(f"  Test call succeeded → circuit is now {cb.state}")
    except Exception as e:
        print(f"  Test call failed: {e}  state={cb.state}")

    section("6. GRACEFUL SHUTDOWN DEMO")
    gs = GracefulShutdown()
    threads = []

    def slow_request(i):
        with gs:
            time.sleep(0.05)
            print(f"    Request {i} completed")

    print("  Starting 3 concurrent requests then initiating shutdown...")
    for i in range(3):
        t = threading.Thread(target=slow_request, args=(i,))
        threads.append(t)
        t.start()

    time.sleep(0.02)   # let threads start
    gs.initiate(timeout=5.0)
    for t in threads:
        t.join()

    section("7. FLASK APP (if available)")
    app = build_flask_app()
    if app:
        print("  Flask app built. Endpoints:")
        for rule in app.url_map.iter_rules():
            print(f"    {rule.methods - {'HEAD', 'OPTIONS'}} {rule.rule}")
        print("\n  Run with: gunicorn --workers=4 'health_check:build_flask_app()'")
        print("  Or in dev: app.run(port=3001)")
    else:
        print("  Flask not available — install with: pip install flask")

    section("8. ROLLING DEPLOY CHECKLIST")
    print("""
  Zero-downtime rolling deploy steps:
  1. Build + push new Docker image to registry
  2. Start new container on port 8081
     docker run -d -p 8081:8080 ml-api:v2
  3. Smoke test new container
     curl -f http://localhost:8081/health
     curl -f http://localhost:8081/ready
  4. Add new backend to nginx upstream
     nginx -s reload   ← zero-downtime config reload
  5. Remove old backend from nginx upstream
     nginx -s reload
  6. Wait for old container to drain (all in-flight requests complete)
     docker stop --time=30 old_container
  7. Monitor metrics for 10 min: error_rate, p95 latency
  8. If regression: rollback nginx to old backend, restart old container
""")


if __name__ == "__main__":
    main()
