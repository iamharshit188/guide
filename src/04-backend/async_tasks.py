"""
Async task queue — Celery + Redis patterns, pure-Python simulation for zero-install demo.
pip install celery redis flask  (for real mode)
No install needed for simulation mode.
"""

import time
import uuid
import json
import queue
import random
import threading
import collections
from enum import Enum
from functools import wraps

try:
    from celery import Celery
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

try:
    from flask import Flask, request, jsonify, g
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ═══════════════════════════════════════════════════════════════
# 1. TASK STATE MACHINE SIMULATION
# ═══════════════════════════════════════════════════════════════

class TaskState(str, Enum):
    PENDING  = "PENDING"
    STARTED  = "STARTED"
    SUCCESS  = "SUCCESS"
    FAILURE  = "FAILURE"
    RETRY    = "RETRY"
    REVOKED  = "REVOKED"


class TaskRecord:
    __slots__ = ("task_id", "state", "result", "error",
                 "submitted_at", "started_at", "completed_at",
                 "retries", "name", "args", "kwargs")

    def __init__(self, name, args=(), kwargs=None):
        self.task_id = str(uuid.uuid4())
        self.state = TaskState.PENDING
        self.result = None
        self.error = None
        self.submitted_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.retries = 0
        self.name = name
        self.args = args
        self.kwargs = kwargs or {}

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "name": self.name,
            "state": self.state.value,
            "result": self.result,
            "error": self.error,
            "retries": self.retries,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "queue_time_ms": (
                (self.started_at - self.submitted_at) * 1000
                if self.started_at else None
            ),
            "exec_time_ms": (
                (self.completed_at - self.started_at) * 1000
                if self.completed_at and self.started_at else None
            ),
        }


# ═══════════════════════════════════════════════════════════════
# 2. IN-PROCESS TASK QUEUE SIMULATION
# ═══════════════════════════════════════════════════════════════

class SimulatedBroker:
    """
    Mimics a Redis/RabbitMQ broker: a FIFO queue of task records.
    Workers pull from this queue and process tasks.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._results: dict = {}
        self._lock = threading.Lock()

    def enqueue(self, record: TaskRecord):
        with self._lock:
            self._results[record.task_id] = record
        self._queue.put(record.task_id)
        return record.task_id

    def get_result(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._results.get(task_id)

    def _update(self, task_id: str, **kwargs):
        with self._lock:
            rec = self._results.get(task_id)
            if rec:
                for k, v in kwargs.items():
                    setattr(rec, k, v)

    def worker_loop(self, task_registry: dict, worker_id: int, stats: dict):
        """Run in a thread — pulls tasks and executes them."""
        while True:
            try:
                task_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                return  # broker signals done

            rec = self.get_result(task_id)
            if rec is None:
                continue

            self._update(task_id, state=TaskState.STARTED,
                         started_at=time.time())

            fn = task_registry.get(rec.name)
            if fn is None:
                self._update(task_id, state=TaskState.FAILURE,
                             error=f"unknown task '{rec.name}'",
                             completed_at=time.time())
                stats["failed"] += 1
                continue

            try:
                result = fn(*rec.args, **rec.kwargs)
                self._update(task_id, state=TaskState.SUCCESS,
                             result=result, completed_at=time.time())
                stats["succeeded"] += 1
            except RetryException as e:
                rec.retries += 1
                if rec.retries <= e.max_retries:
                    self._update(task_id, state=TaskState.RETRY)
                    time.sleep(e.countdown)
                    self._queue.put(task_id)
                    stats["retried"] += 1
                else:
                    self._update(task_id, state=TaskState.FAILURE,
                                 error=str(e.exc), completed_at=time.time())
                    stats["failed"] += 1
            except Exception as e:
                self._update(task_id, state=TaskState.FAILURE,
                             error=str(e), completed_at=time.time())
                stats["failed"] += 1
            finally:
                self._queue.task_done()

    def start_workers(self, task_registry, n_workers=2):
        stats = collections.Counter()
        threads = []
        for i in range(n_workers):
            t = threading.Thread(
                target=self.worker_loop,
                args=(task_registry, i, stats),
                daemon=True,
            )
            t.start()
            threads.append(t)
        return threads, stats

    def drain(self):
        """Wait until queue is empty."""
        self._queue.join()


class RetryException(Exception):
    def __init__(self, exc, countdown=5, max_retries=3):
        self.exc = exc
        self.countdown = countdown
        self.max_retries = max_retries
        super().__init__(str(exc))


# ═══════════════════════════════════════════════════════════════
# 3. TASK DEFINITIONS (simulated and real Celery variants)
# ═══════════════════════════════════════════════════════════════

_rng = random.Random(42)

TASK_REGISTRY = {}


def task(name):
    """Decorator to register a function as a named task."""
    def decorator(fn):
        TASK_REGISTRY[name] = fn
        fn.task_name = name
        return fn
    return decorator


@task("ml.inference")
def inference_task(model_id: str, features: list) -> dict:
    """Simulate ML model inference (variable latency)."""
    # Simulate: small models fast, large models slow
    latency = {"iris_rf": 0.01, "bert": 0.3, "llm": 2.5}.get(model_id, 0.05)
    time.sleep(latency + _rng.uniform(0, 0.005))

    # Simulate occasional failure (5% chance)
    if _rng.random() < 0.05:
        raise ValueError(f"model {model_id} returned NaN output")

    n_classes = 3
    probs = [_rng.random() for _ in range(n_classes)]
    total = sum(probs)
    probs = [p / total for p in probs]
    pred = probs.index(max(probs))
    return {
        "model_id": model_id,
        "prediction": pred,
        "probabilities": [round(p, 4) for p in probs],
    }


@task("ml.batch_inference")
def batch_inference_task(model_id: str, batch: list) -> dict:
    """Simulate batch inference — O(n) with per-item overhead."""
    results = []
    for features in batch:
        latency = 0.005 + _rng.uniform(0, 0.002)
        time.sleep(latency)
        n_classes = 3
        probs = [_rng.random() for _ in range(n_classes)]
        total = sum(probs)
        probs = [p / total for p in probs]
        pred = probs.index(max(probs))
        results.append({"prediction": pred,
                         "probabilities": [round(p, 4) for p in probs]})
    return {"model_id": model_id, "n_instances": len(batch), "results": results}


@task("data.embed_document")
def embed_document_task(doc_id: str, text: str, model: str = "minilm") -> dict:
    """Simulate document embedding (IO-heavy — network call to embedding service)."""
    time.sleep(_rng.uniform(0.05, 0.15))  # network latency
    # Fake embedding: hash text to fixed-dim vector
    dim = 384
    seed = sum(ord(c) for c in text) % (2**31)
    rng = random.Random(seed)
    embedding = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x**2 for x in embedding) ** 0.5
    embedding = [round(x / norm, 6) for x in embedding]
    return {"doc_id": doc_id, "model": model, "dim": dim, "embedding_head": embedding[:4]}


@task("report.generate")
def generate_report_task(report_type: str, start_date: str, end_date: str) -> dict:
    """Simulate long-running report generation (10+ seconds in production)."""
    time.sleep(_rng.uniform(0.1, 0.3))
    return {
        "report_type": report_type,
        "start_date": start_date,
        "end_date": end_date,
        "rows": _rng.randint(100, 50000),
        "status": "generated",
    }


@task("notify.email")
def send_email_task(to: str, subject: str, body: str) -> dict:
    """Simulate email sending via SMTP/SES."""
    time.sleep(_rng.uniform(0.02, 0.08))
    # Retry on simulated network error
    if _rng.random() < 0.15:
        raise RetryException(
            exc=ConnectionError("SMTP connection timeout"),
            countdown=2, max_retries=3,
        )
    return {"to": to, "subject": subject, "delivered": True,
            "message_id": f"msg_{uuid.uuid4().hex[:8]}"}


# ═══════════════════════════════════════════════════════════════
# 4. FLASK API — ASYNC TASK ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def create_async_app(broker: SimulatedBroker):
    app = Flask(__name__)

    @app.before_request
    def tag():
        g.t0 = time.perf_counter()

    @app.after_request
    def add_timing(resp):
        resp.headers["X-Latency-Ms"] = f"{(time.perf_counter()-g.t0)*1000:.2f}"
        return resp

    @app.route("/api/v1/predict/async", methods=["POST"])
    def predict_async():
        data = request.get_json(silent=True) or {}
        model_id = data.get("model_id", "iris_rf")
        features = data.get("features", [5.1, 3.5, 1.4, 0.2])

        rec = TaskRecord("ml.inference", args=(model_id, features))
        task_id = broker.enqueue(rec)

        return jsonify({
            "task_id": task_id,
            "status": "queued",
            "poll_url": f"/api/v1/tasks/{task_id}",
        }), 202

    @app.route("/api/v1/embed/async", methods=["POST"])
    def embed_async():
        data = request.get_json(silent=True) or {}
        rec = TaskRecord("data.embed_document", kwargs={
            "doc_id": data.get("doc_id", "doc_001"),
            "text":   data.get("text", "hello world"),
            "model":  data.get("model", "minilm"),
        })
        task_id = broker.enqueue(rec)
        return jsonify({"task_id": task_id, "status": "queued"}), 202

    @app.route("/api/v1/tasks/<task_id>")
    def task_status(task_id):
        rec = broker.get_result(task_id)
        if rec is None:
            return jsonify({"error": "task not found"}), 404
        d = rec.to_dict()
        status_code = {
            TaskState.PENDING:  202,
            TaskState.STARTED:  202,
            TaskState.RETRY:    202,
            TaskState.SUCCESS:  200,
            TaskState.FAILURE:  500,
            TaskState.REVOKED:  410,
        }.get(rec.state, 202)
        return jsonify(d), status_code

    @app.route("/api/v1/tasks")
    def list_tasks():
        with broker._lock:
            tasks = [rec.to_dict() for rec in broker._results.values()]
        tasks.sort(key=lambda t: t["submitted_at"], reverse=True)
        return jsonify({"tasks": tasks, "total": len(tasks)}), 200

    return app


# ═══════════════════════════════════════════════════════════════
# 5. REAL CELERY CONFIGURATION (shown but not executed)
# ═══════════════════════════════════════════════════════════════

CELERY_CONFIG_SNIPPET = '''
# celery_app.py — production Celery configuration

from celery import Celery
from flask import Flask

def make_celery(flask_app):
    celery = Celery(
        flask_app.import_name,
        broker=flask_app.config["CELERY_BROKER_URL"],     # redis://localhost:6379/0
        backend=flask_app.config["CELERY_RESULT_BACKEND"], # redis://localhost:6379/0
    )
    celery.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        result_expires=3600,           # results expire after 1 hour
        task_acks_late=True,           # ack after task completes (safer for crashes)
        worker_prefetch_multiplier=1,  # 1 task/worker (fair for long tasks)
        task_track_started=True,       # enable STARTED state
        task_routes={
            "ml.*":      {"queue": "gpu"},       # route ML tasks to GPU queue
            "data.*":    {"queue": "cpu"},
            "notify.*":  {"queue": "io"},
        },
    )

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with flask_app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


# Start worker:
#   celery -A celery_app.celery worker --pool=prefork --concurrency=4 -Q gpu,cpu,io
#
# Monitor:
#   celery -A celery_app.celery flower   (web UI at localhost:5555)
#   celery -A celery_app.celery inspect active

@celery.task(bind=True, max_retries=3, default_retry_delay=5,
             name="ml.inference", queue="gpu")
def run_inference(self, model_id, features):
    try:
        model = load_model(model_id)
        return model.predict([features]).tolist()
    except TransientError as e:
        raise self.retry(exc=e, countdown=2 ** self.request.retries)
    except Exception as e:
        raise  # propagates as FAILURE state
'''


# ═══════════════════════════════════════════════════════════════
# 6. DEMOS
# ═══════════════════════════════════════════════════════════════

def demo_task_states():
    section("TASK STATE MACHINE")
    print("""
  PENDING  → task submitted to broker, not yet picked up by worker
  STARTED  → worker picked up task, execution in progress
  SUCCESS  → task completed normally, result stored in backend
  FAILURE  → task raised an unhandled exception
  RETRY    → task raised RetryException, re-queued with countdown delay
  REVOKED  → task was manually cancelled (celery.control.revoke)

  State transitions:
    PENDING → STARTED → SUCCESS
                      ↘ FAILURE
    PENDING → STARTED → RETRY → STARTED → SUCCESS
                                        ↘ FAILURE (max retries exceeded)

  Polling pattern (client side):
    task_id = POST /predict/async → 202
    while True:
        resp = GET /tasks/{task_id}
        if resp.status == 200: break   # SUCCESS
        if resp.status == 500: break   # FAILURE
        sleep(1)                       # still PENDING/STARTED/RETRY
    """)


def demo_simulated_broker():
    section("SIMULATED BROKER — ENQUEUE + PROCESS")
    broker = SimulatedBroker()
    threads, stats = broker.start_workers(TASK_REGISTRY, n_workers=3)

    # Enqueue a variety of tasks
    task_ids = []
    tasks_to_submit = [
        TaskRecord("ml.inference",      args=("iris_rf",), kwargs={}),
        TaskRecord("ml.inference",      args=("bert",),    kwargs={}),
        TaskRecord("data.embed_document", kwargs={
            "doc_id": "doc_001", "text": "attention is all you need", "model": "minilm"
        }),
        TaskRecord("report.generate",   kwargs={
            "report_type": "monthly_metrics",
            "start_date": "2024-01-01", "end_date": "2024-01-31"
        }),
        TaskRecord("notify.email",      kwargs={
            "to": "user@example.com",
            "subject": "Model training complete",
            "body": "Your model iris_rf_v2 is ready.",
        }),
        TaskRecord("ml.batch_inference", args=("iris_rf",), kwargs={
            "batch": [[5.1,3.5,1.4,0.2],[6.3,3.3,6.0,2.5],[5.5,2.4,3.8,1.1]]
        }),
    ]

    t_submit = time.perf_counter()
    for rec in tasks_to_submit:
        tid = broker.enqueue(rec)
        task_ids.append((rec.name, tid))
        print(f"  enqueued {rec.name:<25s} → {tid[:8]}...")

    broker.drain()
    total_ms = (time.perf_counter() - t_submit) * 1000
    print(f"\n  All tasks completed in {total_ms:.0f}ms")

    section("TASK RESULTS")
    print(f"  {'Task':<25s}  {'State':<8s}  {'Queue ms':>9s}  {'Exec ms':>8s}  {'Retries':>7s}")
    print("  " + "-" * 70)
    for name, tid in task_ids:
        rec = broker.get_result(tid)
        d = rec.to_dict()
        q_ms = f"{d['queue_time_ms']:.1f}" if d['queue_time_ms'] else "  N/A"
        e_ms = f"{d['exec_time_ms']:.1f}" if d['exec_time_ms'] else "  N/A"
        print(f"  {name:<25s}  {d['state']:<8s}  {q_ms:>9s}  {e_ms:>8s}  {d['retries']:>7d}")

    print(f"\n  Stats: {dict(stats)}")


def demo_flask_async_api():
    if not FLASK_AVAILABLE:
        return

    section("FLASK ASYNC API — ACCEPT + POLL PATTERN")
    broker = SimulatedBroker()
    threads, stats = broker.start_workers(TASK_REGISTRY, n_workers=2)
    app = create_async_app(broker)
    client = app.test_client()

    # Submit async prediction
    resp = client.post("/api/v1/predict/async",
                       json={"model_id": "iris_rf", "features": [5.1, 3.5, 1.4, 0.2]})
    data = resp.get_json()
    task_id = data["task_id"]
    print(f"  POST /predict/async → {resp.status_code} {json.dumps(data)}")

    # Submit embed
    resp2 = client.post("/api/v1/embed/async",
                        json={"doc_id": "p001", "text": "transformers attention mechanism"})
    task_id2 = resp2.get_json()["task_id"]
    print(f"  POST /embed/async   → {resp2.status_code}  task_id={task_id2[:8]}...")

    # Poll until done
    print(f"\n  Polling task {task_id[:8]}...")
    for attempt in range(30):
        time.sleep(0.05)
        resp = client.get(f"/api/v1/tasks/{task_id}")
        d = resp.get_json()
        state = d["state"]
        print(f"  poll {attempt+1:2d}: status={resp.status_code}  state={state}")
        if state in ("SUCCESS", "FAILURE"):
            break

    print(f"\n  Final result: {json.dumps(d.get('result', d.get('error')))[:120]}")

    # List all tasks
    broker.drain()
    resp = client.get("/api/v1/tasks")
    all_tasks = resp.get_json()
    print(f"\n  Total tasks in broker: {all_tasks['total']}")
    for t in all_tasks["tasks"][:4]:
        print(f"  {t['name']:<25s} {t['state']:<8s} exec={t['exec_time_ms']:.1f}ms" if t['exec_time_ms'] else
              f"  {t['name']:<25s} {t['state']:<8s}")


def demo_celery_config():
    section("REAL CELERY CONFIGURATION (reference — not executed)")
    print(CELERY_CONFIG_SNIPPET)

    section("WORKER CONCURRENCY MODELS")
    print(f"""
  {'Pool':12s}  {'Flag':25s}  {'Best for'}
  {'-'*65}
  {'Prefork':12s}  {'--pool=prefork':25s}  CPU-bound (model inference, data processing)
  {'Gevent':12s}  {'--pool=gevent':25s}  I/O-bound (API calls, DB queries, webhooks)
  {'Eventlet':12s}  {'--pool=eventlet':25s}  Same as gevent (different async library)
  {'Threads':12s}  {'--pool=threads':25s}  Light I/O, limited by Python GIL
  {'Solo':12s}  {'--pool=solo':25s}  Debugging (single-threaded, synchronous)

  Production command:
    celery -A app.celery worker \\
           --pool=prefork \\
           --concurrency=4 \\
           -Q gpu,cpu,io \\
           --loglevel=info \\
           --max-memory-per-child=512000  # restart after 512MB (memory leak guard)
    """)


def main():
    demo_task_states()
    demo_simulated_broker()
    demo_flask_async_api()
    demo_celery_config()


if __name__ == "__main__":
    main()
