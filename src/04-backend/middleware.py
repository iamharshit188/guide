"""
Flask middleware — auth token check, token-bucket rate limiting, request logging.
pip install flask
"""

import os
import time
import uuid
import hmac
import hashlib
import logging
import threading
import json
import collections
from functools import wraps

try:
    from flask import Flask, Blueprint, request, jsonify, g
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("flask not installed. Run: pip install flask")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ═══════════════════════════════════════════════════════════════
# 1. AUTH MIDDLEWARE
# ═══════════════════════════════════════════════════════════════

# Store only SHA-256 hashes of API keys — never raw keys.
# Generate: hashlib.sha256(b"my-key").hexdigest()
_VALID_KEY_HASHES: set = set()


def _register_key(raw_key: str):
    _VALID_KEY_HASHES.add(hashlib.sha256(raw_key.encode()).hexdigest())


_register_key("test-api-key-alpha")
_register_key("test-api-key-beta")


def check_auth(skip_paths=("/health",)):
    """
    Returns an error response or None (pass through).
    Timing-safe comparison via hmac.compare_digest.
    """
    if request.path in skip_paths:
        return None

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": {
            "code": "UNAUTHORIZED",
            "message": "Authorization header required (Bearer <token>)",
        }}), 401

    token = auth_header[7:].strip()
    if not token:
        return jsonify({"error": {"code": "UNAUTHORIZED", "message": "empty token"}}), 401

    token_hash = hashlib.sha256(token.encode()).hexdigest()

    # hmac.compare_digest: constant-time comparison regardless of match position
    authenticated = any(
        hmac.compare_digest(token_hash, valid_hash)
        for valid_hash in _VALID_KEY_HASHES
    )
    if not authenticated:
        return jsonify({"error": {
            "code": "FORBIDDEN",
            "message": "invalid API key",
        }}), 403
    return None


# ═══════════════════════════════════════════════════════════════
# 2. TOKEN BUCKET RATE LIMITER
# ═══════════════════════════════════════════════════════════════

class TokenBucket:
    """
    Token Bucket algorithm.

    Tokens refill continuously at `rate` tokens/second up to `capacity`.
    Each request consumes 1 token. Denied if tokens < 1.

    Thread-safe via threading.Lock.
    """

    __slots__ = ("capacity", "rate", "tokens", "_last", "_lock")

    def __init__(self, capacity: float, rate: float):
        self.capacity = float(capacity)
        self.rate = float(rate)          # tokens per second
        self.tokens = float(capacity)    # start full
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, n: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            # Refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self._last = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

    def wait_time(self) -> float:
        """Seconds until next token is available (approximate)."""
        with self._lock:
            if self.tokens >= 1:
                return 0.0
            return (1.0 - self.tokens) / self.rate

    def state(self) -> dict:
        with self._lock:
            return {
                "tokens": round(self.tokens, 3),
                "capacity": self.capacity,
                "rate_per_sec": self.rate,
            }


class PerClientRateLimiter:
    """Maintains one TokenBucket per client IP."""

    def __init__(self, capacity=20, rate=5, max_clients=10_000):
        self._buckets: dict = {}
        self._lock = threading.Lock()
        self._capacity = capacity
        self._rate = rate
        self._max_clients = max_clients

    def _get_bucket(self, client_id: str) -> TokenBucket:
        with self._lock:
            if client_id not in self._buckets:
                if len(self._buckets) >= self._max_clients:
                    # Evict oldest — in production use an LRU cache
                    oldest = next(iter(self._buckets))
                    del self._buckets[oldest]
                self._buckets[client_id] = TokenBucket(self._capacity, self._rate)
            return self._buckets[client_id]

    def check(self, client_id: str) -> tuple:
        """Returns (allowed: bool, state: dict)."""
        bucket = self._get_bucket(client_id)
        allowed = bucket.consume()
        return allowed, bucket.state()

    def client_count(self) -> int:
        with self._lock:
            return len(self._buckets)


# ═══════════════════════════════════════════════════════════════
# 3. STRUCTURED REQUEST LOGGER
# ═══════════════════════════════════════════════════════════════

class StructuredLogger:
    """JSON-line logger suitable for log aggregators (Datadog, CloudWatch, ELK)."""

    def __init__(self, name="api", level=logging.DEBUG):
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(level)
        self._logger.propagate = False

    def _log(self, level, event, **fields):
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "level": level, "event": event, **fields}
        line = json.dumps(record, default=str)
        getattr(self._logger, level)(line)

    def info(self, event, **kw):   self._log("info", event, **kw)
    def warn(self, event, **kw):   self._log("warning", event, **kw)
    def error(self, event, **kw):  self._log("error", event, **kw)
    def debug(self, event, **kw):  self._log("debug", event, **kw)


# ═══════════════════════════════════════════════════════════════
# 4. FLASK APP WITH ALL MIDDLEWARE WIRED UP
# ═══════════════════════════════════════════════════════════════

def create_app_with_middleware(
    rate_capacity=20,
    rate_per_sec=5,
    require_auth=True,
):
    app = Flask(__name__)
    api_bp = Blueprint("api", __name__)
    logger = StructuredLogger("api_demo")
    limiter = PerClientRateLimiter(capacity=rate_capacity, rate=rate_per_sec)

    # ── Before request ────────────────────────────────────────

    @app.before_request
    def attach_request_id():
        g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        g.start_time = time.perf_counter()

    @app.before_request
    def auth_check():
        if not require_auth:
            return
        err = check_auth(skip_paths=("/health", "/metrics"))
        if err:
            logger.warn("auth_failed",
                        request_id=g.request_id,
                        path=request.path,
                        ip=request.remote_addr)
            return err

    @app.before_request
    def rate_limit_check():
        client_id = request.headers.get("X-API-Key") or request.remote_addr or "unknown"
        allowed, state = limiter.check(client_id)
        g.rate_state = state
        if not allowed:
            wait = (1.0 - state["tokens"]) / state["rate_per_sec"]
            logger.warn("rate_limited",
                        request_id=g.request_id,
                        client=client_id,
                        tokens=state["tokens"],
                        wait_sec=round(wait, 2))
            resp = jsonify({"error": {
                "code": "RATE_LIMITED",
                "message": "too many requests",
                "retry_after_sec": round(wait, 2),
            }})
            resp.status_code = 429
            resp.headers["Retry-After"] = str(int(wait) + 1)
            resp.headers["X-RateLimit-Limit"] = str(state["capacity"])
            resp.headers["X-RateLimit-Remaining"] = "0"
            return resp

    @app.before_request
    def log_incoming():
        body_size = request.content_length or 0
        logger.debug("request_in",
                     request_id=g.request_id,
                     method=request.method,
                     path=request.path,
                     query=request.query_string.decode(),
                     ip=request.remote_addr,
                     user_agent=request.user_agent.string[:80],
                     body_bytes=body_size)

    # ── After request ─────────────────────────────────────────

    @app.after_request
    def log_response(response):
        latency_ms = (time.perf_counter() - g.start_time) * 1000
        rate_state = getattr(g, "rate_state", {})

        logger.info("request_out",
                    request_id=g.request_id,
                    method=request.method,
                    path=request.path,
                    status=response.status_code,
                    latency_ms=round(latency_ms, 2),
                    tokens_remaining=rate_state.get("tokens", "?"))

        response.headers["X-Request-ID"] = g.request_id
        response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
        if rate_state:
            response.headers["X-RateLimit-Limit"] = str(rate_state["capacity"])
            response.headers["X-RateLimit-Remaining"] = str(int(rate_state["tokens"]))
        return response

    # ── Teardown (always runs) ────────────────────────────────

    @app.teardown_request
    def on_teardown(exc):
        if exc is not None:
            latency_ms = (time.perf_counter() - g.start_time) * 1000
            logger.error("request_exception",
                         request_id=g.request_id,
                         exception=str(exc),
                         latency_ms=round(latency_ms, 2))

    # ── Routes ────────────────────────────────────────────────

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @api_bp.route("/protected")
    def protected():
        return jsonify({
            "message": "authenticated request succeeded",
            "request_id": g.request_id,
            "rate_limit": g.rate_state,
        }), 200

    @api_bp.route("/public")
    def public_endpoint():
        return jsonify({"message": "public endpoint", "request_id": g.request_id}), 200

    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.error("unhandled_exception", exception=str(e), path=request.path)
        return jsonify({"error": {"code": "INTERNAL_ERROR", "message": str(e)}}), 500

    app.register_blueprint(api_bp, url_prefix="/api")
    return app


# ═══════════════════════════════════════════════════════════════
# 5. STANDALONE DEMOS
# ═══════════════════════════════════════════════════════════════

def demo_token_bucket():
    section("TOKEN BUCKET ALGORITHM")
    print("  capacity=5, rate=2 tokens/sec\n")

    bucket = TokenBucket(capacity=5, rate=2.0)
    events = []

    for i in range(8):
        allowed = bucket.consume()
        state = bucket.state()
        events.append((i+1, allowed, state["tokens"]))
        print(f"  request {i+1:2d}: {'ALLOW' if allowed else 'DENY ':5s}  "
              f"tokens_remaining={state['tokens']:.2f}")

    print(f"\n  After 8 rapid requests (started with 5 tokens, rate=2/s, ~instant):")
    print(f"  First 5 allowed, next 3 denied (bucket empty)")

    print(f"\n  Simulating 1-second wait + 3 more requests:")
    # Simulate time passing by directly manipulating last
    bucket._last -= 1.5  # pretend 1.5 seconds passed
    for i in range(3):
        allowed = bucket.consume()
        state = bucket.state()
        print(f"  request {i+1:2d}: {'ALLOW' if allowed else 'DENY ':5s}  "
              f"tokens_remaining={state['tokens']:.2f}")


def demo_timing_attack():
    section("TIMING ATTACK PREVENTION")
    print("""
  Standard string comparison:
    "abc" == "xyz"  → short-circuits at 'a' vs 'x' (fast, leaks info)
    "abc" == "abz"  → checks 3 chars before failing (slower)

  An attacker can measure response time to guess tokens character by character.
  For a 32-char token, only 32 * 256 = 8192 requests needed.

  hmac.compare_digest(a, b):
    - Compares all bytes regardless of where mismatch occurs.
    - Timing is constant relative to input length (not match position).
    - Prevents timing side-channel attacks on auth tokens.
    """)

    import timeit

    correct_hash = hashlib.sha256(b"secret-key").hexdigest()
    wrong_hash   = hashlib.sha256(b"wrong-key").hexdigest()
    # Same-prefix wrong hash: would leak info with ==
    wrong_end    = correct_hash[:-4] + "ffff"

    n = 100_000
    t_eq_correct = timeit.timeit(lambda: correct_hash == correct_hash, number=n) / n * 1e6
    t_eq_wrong   = timeit.timeit(lambda: correct_hash == wrong_hash,   number=n) / n * 1e6
    t_hmac_correct = timeit.timeit(
        lambda: hmac.compare_digest(correct_hash, correct_hash), number=n) / n * 1e6
    t_hmac_wrong   = timeit.timeit(
        lambda: hmac.compare_digest(correct_hash, wrong_hash),   number=n) / n * 1e6

    print(f"  == correct:          {t_eq_correct:.4f} µs")
    print(f"  == wrong:            {t_eq_wrong:.4f} µs  (may differ → leaks info)")
    print(f"  compare_digest correct: {t_hmac_correct:.4f} µs")
    print(f"  compare_digest wrong:   {t_hmac_wrong:.4f} µs  (consistent → safe)")


def demo_flask_middleware():
    if not FLASK_AVAILABLE:
        return

    section("FLASK MIDDLEWARE DEMO — AUTH + RATE LIMIT + LOGGING")
    app = create_app_with_middleware(rate_capacity=5, rate_per_sec=10,
                                     require_auth=True)
    client = app.test_client()

    good_key  = "test-api-key-alpha"
    bad_key   = "not-a-real-key"

    print("\n  -- Auth tests --")
    cases_auth = [
        ("no auth",    {},                         "/api/protected"),
        ("bad key",    {"Authorization": f"Bearer {bad_key}"},  "/api/protected"),
        ("good key",   {"Authorization": f"Bearer {good_key}"}, "/api/protected"),
        ("health (no auth needed)", {},            "/health"),
    ]
    for desc, headers, path in cases_auth:
        resp = client.get(path, headers=headers)
        d = resp.get_json()
        print(f"  [{resp.status_code}] {desc:<30s}  "
              f"{json.dumps(d)[:70]}")

    print("\n  -- Rate limit tests (capacity=5, 8 rapid requests) --")
    for i in range(8):
        resp = client.get("/api/protected",
                          headers={"Authorization": f"Bearer {good_key}"})
        remaining = resp.headers.get("X-RateLimit-Remaining", "?")
        status = resp.status_code
        label = "OK   " if status == 200 else "DENY "
        print(f"  request {i+1:2d}: {label} status={status}  "
              f"tokens_remaining={remaining}")


def demo_logging_patterns():
    section("STRUCTURED LOGGING FORMAT")
    logger = StructuredLogger("demo", level=logging.DEBUG)
    logger.info("request_out",
                request_id="abc12",
                method="POST",
                path="/api/v1/predict",
                status=200,
                latency_ms=3.21,
                tokens_remaining=14)
    logger.warn("rate_limited",
                request_id="xyz99",
                client="192.168.1.1",
                tokens=0.0,
                wait_sec=0.4)
    logger.error("auth_failed",
                 request_id="def45",
                 path="/api/v1/admin",
                 ip="10.0.0.5")
    print("""
  Log line format: JSON on a single line — parseable by log aggregators.
  Fields: ts (ISO8601), level, event, + arbitrary kwargs.

  In production ship to:
    Datadog:     DD_LOGS_ENABLED=true  (agent auto-collects stdout)
    CloudWatch:  awslogs driver in docker-compose / ECS task def
    ELK Stack:   Filebeat → Logstash → Elasticsearch → Kibana
    """)


def demo_middleware_order():
    section("MIDDLEWARE EXECUTION ORDER")
    print("""
  Flask request lifecycle:

  before_request hooks (in registration order):
    1. attach_request_id      → sets g.request_id, g.start_time
    2. auth_check             → validates Bearer token
    3. rate_limit_check       → checks / consumes token bucket
    4. log_incoming           → logs method, path, body size

  view function executes

  after_request hooks (in REVERSE registration order):
    4. log_response           → logs status, latency, rate state

  teardown_request (ALWAYS runs, even on exception):
    → logs exception if one occurred

  Key: before_request returning a value (non-None) SHORT-CIRCUITS
  the rest of the chain and the view function. Used by auth (401/403)
  and rate limit (429) to abort early without reaching business logic.
    """)


def main():
    demo_token_bucket()
    demo_timing_attack()
    demo_logging_patterns()
    demo_middleware_order()

    if FLASK_AVAILABLE:
        demo_flask_middleware()
    else:
        section("FLASK NOT AVAILABLE")
        print("  Install flask to run the full middleware demo:")
        print("  pip install flask")


if __name__ == "__main__":
    main()
