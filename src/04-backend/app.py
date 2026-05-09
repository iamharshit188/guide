"""
Flask app factory pattern — blueprints, error handlers, config objects, CORS.
pip install flask flask-cors
"""

import os
import time
import uuid
import logging
import json
from functools import wraps

try:
    from flask import Flask, Blueprint, request, jsonify, g, current_app
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("flask / flask-cors not installed. Run: pip install flask flask-cors")
    print("Showing architecture and expected behaviour.\n")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Config objects ─────────────────────────────────────────────

class BaseConfig:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-insecure-key")
    JSON_SORT_KEYS = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024   # 16 MB
    RATELIMIT_CAPACITY = 100                 # token bucket capacity
    RATELIMIT_RATE = 10                      # tokens/second refill
    MAX_BATCH_SIZE = 256


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = False
    LOG_LEVEL = "DEBUG"


class TestingConfig(BaseConfig):
    TESTING = True
    DEBUG = True
    LOG_LEVEL = "WARNING"


class ProductionConfig(BaseConfig):
    DEBUG = False
    TESTING = False
    LOG_LEVEL = "INFO"


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing":     TestingConfig,
    "production":  ProductionConfig,
}


# ── Logging setup ──────────────────────────────────────────────

def configure_logging(app):
    level = getattr(logging, app.config.get("LOG_LEVEL", "INFO"))
    handler = logging.StreamHandler()
    handler.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(level)


# ── Blueprint: health ──────────────────────────────────────────

health_bp = Blueprint("health", __name__)


@health_bp.route("/health")
def liveness():
    return jsonify({"status": "ok", "timestamp": time.time()}), 200


@health_bp.route("/health/ready")
def readiness():
    checks = {"api": "ok"}
    all_ok = all(v == "ok" for v in checks.values())
    return jsonify({
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
    }), (200 if all_ok else 503)


# ── Blueprint: api v1 ──────────────────────────────────────────

api_v1_bp = Blueprint("api_v1", __name__)


def require_json(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return jsonify({"error": {
                "code": "UNSUPPORTED_MEDIA_TYPE",
                "message": "Content-Type must be application/json",
            }}), 415
        return f(*args, **kwargs)
    return wrapper


@api_v1_bp.before_request
def tag_request():
    """Attach a per-request ID and start-time to flask.g."""
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    g.start_time = time.perf_counter()


@api_v1_bp.after_request
def add_headers(response):
    """Inject request ID and latency into response headers."""
    latency_ms = (time.perf_counter() - g.start_time) * 1000
    response.headers["X-Request-ID"] = g.request_id
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    current_app.logger.info(
        f"[{g.request_id}] {request.method} {request.path} "
        f"→ {response.status_code} ({latency_ms:.1f}ms)"
    )
    return response


@api_v1_bp.route("/echo", methods=["POST"])
@require_json
def echo():
    """Simple echo endpoint demonstrating request/response cycle."""
    data = request.get_json()
    return jsonify({
        "echo": data,
        "request_id": g.request_id,
        "method": request.method,
        "path": request.path,
    }), 200


@api_v1_bp.route("/info")
def info():
    """Return API metadata."""
    return jsonify({
        "version": "1.0.0",
        "name": "ML Platform API",
        "config": current_app.config["ENV"] if hasattr(current_app.config, "ENV") else "custom",
        "debug": current_app.debug,
    }), 200


@api_v1_bp.route("/error/handled")
def handled_error():
    """Demonstrate structured error response."""
    return jsonify({"error": {
        "code": "DEMO_ERROR",
        "message": "This is a handled error example",
        "request_id": g.request_id,
    }}), 400


@api_v1_bp.route("/error/unhandled")
def unhandled_error():
    """Trigger an unhandled exception to show global error handler."""
    raise RuntimeError("intentional unhandled exception for demo")


# ── Blueprint: api v2 (versioning demo) ───────────────────────

api_v2_bp = Blueprint("api_v2", __name__)


@api_v2_bp.before_request
def tag_request_v2():
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    g.start_time = time.perf_counter()


@api_v2_bp.after_request
def add_headers_v2(response):
    latency_ms = (time.perf_counter() - g.start_time) * 1000
    response.headers["X-Request-ID"] = g.request_id
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    response.headers["X-API-Version"] = "2"
    return response


@api_v2_bp.route("/info")
def info_v2():
    """v2 adds extra fields — backwards-compatible versioning demo."""
    return jsonify({
        "version": "2.0.0",
        "name": "ML Platform API",
        "features": ["batch_predict", "streaming", "async_tasks"],
        "deprecated_in": None,
    }), 200


# ── App factory ────────────────────────────────────────────────

def create_app(env="development"):
    app = Flask(__name__)
    app.config.from_object(CONFIG_MAP.get(env, DevelopmentConfig))

    configure_logging(app)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints with URL prefixes
    app.register_blueprint(health_bp)
    app.register_blueprint(api_v1_bp, url_prefix="/api/v1")
    app.register_blueprint(api_v2_bp, url_prefix="/api/v2")

    # ── Global error handlers ──────────────────────────────────

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": {
            "code": "NOT_FOUND",
            "message": str(e),
        }}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": {
            "code": "METHOD_NOT_ALLOWED",
            "message": f"Method {request.method} not allowed on {request.path}",
            "allowed": e.valid_methods,
        }}), 405

    @app.errorhandler(413)
    def request_too_large(e):
        return jsonify({"error": {
            "code": "REQUEST_TOO_LARGE",
            "message": f"Request body exceeds {app.config['MAX_CONTENT_LENGTH']} bytes",
        }}), 413

    @app.errorhandler(Exception)
    def unhandled(e):
        app.logger.exception(f"Unhandled exception: {e}")
        return jsonify({"error": {
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
        }}), 500

    return app


# ── Demo: print blueprint route table ─────────────────────────

def print_route_table(app):
    section("REGISTERED ROUTES")
    rows = []
    for rule in app.url_map.iter_rules():
        methods = ", ".join(sorted(m for m in rule.methods if m not in ("HEAD", "OPTIONS")))
        rows.append((rule.endpoint, methods, str(rule)))
    rows.sort(key=lambda r: r[2])
    print(f"  {'Endpoint':35s}  {'Methods':20s}  {'URL'}")
    print("  " + "-" * 80)
    for endpoint, methods, url in rows:
        print(f"  {endpoint:35s}  {methods:20s}  {url}")


def demo_config(app):
    section("CONFIGURATION OBJECTS")
    print(f"  DEBUG:            {app.debug}")
    print(f"  TESTING:          {app.config['TESTING']}")
    print(f"  SECRET_KEY:       {'*' * len(app.config['SECRET_KEY'])}")
    print(f"  MAX_CONTENT:      {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)} MB")
    print(f"  RATELIMIT:        {app.config['RATELIMIT_CAPACITY']} cap / "
          f"{app.config['RATELIMIT_RATE']} tokens/s")

    section("CONFIG CLASS HIERARCHY")
    for name, cls in CONFIG_MAP.items():
        base = "BaseConfig"
        attrs = {k: v for k, v in vars(cls).items()
                 if not k.startswith("_") and k.isupper()}
        overrides = list(attrs.keys())
        print(f"  {name:15s} → {cls.__name__:20s} overrides: {overrides}")


def demo_requests(app):
    section("SIMULATED REQUESTS (test client)")
    client = app.test_client()

    cases = [
        ("GET",  "/health",              None),
        ("GET",  "/health/ready",        None),
        ("GET",  "/api/v1/info",         None),
        ("POST", "/api/v1/echo",         {"message": "hello", "value": 42}),
        ("GET",  "/api/v2/info",         None),
        ("GET",  "/api/v1/error/handled",None),
        ("GET",  "/api/v1/nonexistent",  None),
        ("POST", "/api/v1/echo",         None),      # missing Content-Type
    ]

    for method, path, body in cases:
        if body is not None:
            resp = client.open(path, method=method,
                               json=body,
                               headers={"X-Request-ID": "demo-001"})
        else:
            resp = client.open(path, method=method)

        data = resp.get_json()
        rid = resp.headers.get("X-Request-ID", "-")
        lat = resp.headers.get("X-Latency-Ms", "-")
        print(f"\n  {method} {path}")
        print(f"    Status:      {resp.status_code}")
        print(f"    Request-ID:  {rid}")
        print(f"    Latency:     {lat} ms")
        print(f"    Body:        {json.dumps(data)[:100]}")


def demo_blueprint_isolation(app):
    section("BLUEPRINT ISOLATION — before_request SCOPING")
    print("""
  @api_v1_bp.before_request hooks only run for routes on api_v1_bp.
  @api_v2_bp.before_request hooks only run for routes on api_v2_bp.
  @app.before_request hooks run for ALL routes regardless of blueprint.

  Execution order per request:
    app.before_request (all blueprints)
    → blueprint.before_request (current blueprint only)
    → view function
    → blueprint.after_request (reverse registration order)
    → app.after_request
    → app.teardown_request (always, even on exception)

  Use case: per-blueprint auth (e.g., admin blueprint has stricter auth,
  public blueprint has no auth) without duplicating decorators on every view.
    """)


def main():
    if not FLASK_AVAILABLE:
        print("Flask not installed — showing structure only.")
        section("APP FACTORY PATTERN SUMMARY")
        print("""
  def create_app(env='development'):
      app = Flask(__name__)
      app.config.from_object(CONFIG_MAP[env])
      CORS(app)
      app.register_blueprint(health_bp)
      app.register_blueprint(api_v1_bp, url_prefix='/api/v1')
      app.register_blueprint(api_v2_bp, url_prefix='/api/v2')
      # global error handlers
      return app
        """)
        return

    app = create_app("development")

    demo_config(app)
    print_route_table(app)
    demo_requests(app)
    demo_blueprint_isolation(app)

    section("RUNNING THE SERVER")
    print("""
  To run the Flask dev server:
    export FLASK_APP=app:create_app
    export FLASK_ENV=development
    flask run --port 5001

  Production (Gunicorn):
    gunicorn "app:create_app('production')" --workers 4 --bind 0.0.0.0:8000

  The server will be accessible at http://localhost:5001
    """)


if __name__ == "__main__":
    main()
