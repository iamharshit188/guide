"""
ML model serving with Flask — sklearn pickle, torch scripted, /predict endpoint.
pip install flask scikit-learn numpy
Optional: pip install torch  (for PyTorch section)
"""

import os
import time
import json
import math
import pickle
import tempfile
import threading
import numpy as np
from functools import lru_cache

try:
    from flask import Flask, Blueprint, request, jsonify, g
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("flask not installed. Run: pip install flask")

try:
    from sklearn.datasets import load_iris, load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not installed. Run: pip install scikit-learn")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Toy model registry ─────────────────────────────────────────
# In production: load from S3/GCS/MLflow artifact store.

_MODEL_REGISTRY: dict = {}
_REGISTRY_LOCK = threading.RLock()


def register_model(model_id, model_obj, metadata: dict):
    with _REGISTRY_LOCK:
        _MODEL_REGISTRY[model_id] = {
            "model": model_obj,
            "metadata": metadata,
            "loaded_at": time.time(),
        }


def get_model(model_id):
    with _REGISTRY_LOCK:
        entry = _MODEL_REGISTRY.get(model_id)
    if entry is None:
        raise KeyError(f"model '{model_id}' not found in registry")
    return entry["model"], entry["metadata"]


def list_models():
    with _REGISTRY_LOCK:
        return {
            mid: {k: v for k, v in entry.items() if k != "model"}
            for mid, entry in _MODEL_REGISTRY.items()
        }


# ── Model training (run once at startup) ──────────────────────

def train_sklearn_models():
    if not SKLEARN_AVAILABLE:
        return

    iris = load_iris()
    X, y = iris.data, iris.target

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=42)),
    ])
    rf_pipeline.fit(X, y)
    register_model("iris_rf_v1", rf_pipeline, {
        "framework": "sklearn",
        "type": "RandomForestClassifier",
        "version": "1.0.0",
        "n_features": 4,
        "n_classes": 3,
        "class_names": iris.target_names.tolist(),
    })

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=42)),
    ])
    lr_pipeline.fit(X, y)
    register_model("iris_lr_v1", lr_pipeline, {
        "framework": "sklearn",
        "type": "LogisticRegression",
        "version": "1.0.0",
        "n_features": 4,
        "n_classes": 3,
        "class_names": iris.target_names.tolist(),
    })


# ── Optional: tiny PyTorch model ──────────────────────────────

class TinyMLP(nn.Module):
    def __init__(self, in_dim=4, hidden=16, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_torch_model():
    if not (TORCH_AVAILABLE and SKLEARN_AVAILABLE):
        return

    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler as SS
    iris = load_iris()
    scaler = SS()
    X = scaler.fit_transform(iris.data).astype(np.float32)
    y = iris.target

    model = TinyMLP(in_dim=4, hidden=16, out_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.tensor(X)
    y_t = torch.tensor(y, dtype=torch.long)

    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    # Wrap with scaler for consistent API
    register_model("iris_torch_v1", (model, scaler), {
        "framework": "pytorch",
        "type": "TinyMLP",
        "version": "1.0.0",
        "n_features": 4,
        "n_classes": 3,
        "class_names": ["setosa", "versicolor", "virginica"],
    })


# ── Serialization helpers ──────────────────────────────────────

def demo_serialization():
    if not SKLEARN_AVAILABLE:
        return

    section("MODEL SERIALIZATION")
    model, meta = get_model("iris_rf_v1")

    with tempfile.TemporaryDirectory() as tmp:
        pkl_path = os.path.join(tmp, "iris_rf_v1.pkl")

        # pickle.dump — standard sklearn serialization
        t0 = time.perf_counter()
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        save_ms = (time.perf_counter() - t0) * 1000
        size_kb = os.path.getsize(pkl_path) / 1024

        t0 = time.perf_counter()
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        load_ms = (time.perf_counter() - t0) * 1000

        print(f"  pickle save:  {save_ms:.1f}ms   size: {size_kb:.1f}KB")
        print(f"  pickle load:  {load_ms:.1f}ms")

        # Verify identical predictions
        from sklearn.datasets import load_iris as li
        X_test = li().data[:5]
        orig_preds = model.predict(X_test)
        load_preds = loaded.predict(X_test)
        match = np.array_equal(orig_preds, load_preds)
        print(f"  predictions match after reload: {match}")
        print(f"  predictions: {orig_preds.tolist()}")

        if TORCH_AVAILABLE:
            torch_path = os.path.join(tmp, "iris_torch_v1.pt")
            torch_model, scaler_obj = get_model("iris_torch_v1")

            t0 = time.perf_counter()
            torch.save({
                "state_dict": torch_model.state_dict(),
                "scaler_mean": scaler_obj.mean_.tolist(),
                "scaler_scale": scaler_obj.scale_.tolist(),
            }, torch_path)
            torch_save_ms = (time.perf_counter() - t0) * 1000
            torch_size_kb = os.path.getsize(torch_path) / 1024
            print(f"\n  torch.save:   {torch_save_ms:.1f}ms   size: {torch_size_kb:.1f}KB")

            # TorchScript — portable serialization (no Python needed at inference)
            scripted = torch.jit.script(torch_model)
            script_path = os.path.join(tmp, "iris_scripted.pt")
            scripted.save(script_path)
            script_size_kb = os.path.getsize(script_path) / 1024
            print(f"  TorchScript:  size: {script_size_kb:.1f}KB  (no Python required)")


# ── Input validation (no Pydantic dependency) ──────────────────

def validate_predict_request(data, n_features):
    """Returns (features: list, error_msg: str | None)."""
    if not isinstance(data, dict):
        return None, "request body must be a JSON object"
    features = data.get("features")
    if features is None:
        return None, "missing required field 'features'"
    if not isinstance(features, list):
        return None, "'features' must be a JSON array"
    if len(features) != n_features:
        return None, f"expected {n_features} features, got {len(features)}"
    for i, v in enumerate(features):
        if not isinstance(v, (int, float)):
            return None, f"features[{i}] must be a number, got {type(v).__name__}"
        if math.isnan(v) or math.isinf(v):
            return None, f"features[{i}] must be finite (got {v})"
    return [float(v) for v in features], None


# ── Flask app with predict endpoints ──────────────────────────

def create_serving_app():
    app = Flask(__name__)
    bp = Blueprint("serve", __name__)

    @bp.before_request
    def start_timer():
        g.t0 = time.perf_counter()

    @bp.after_request
    def add_timing(response):
        latency_ms = (time.perf_counter() - g.t0) * 1000
        response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
        return response

    @bp.route("/models")
    def models_list():
        return jsonify(list_models()), 200

    @bp.route("/predict/<model_id>", methods=["POST"])
    def predict_single(model_id):
        try:
            model_obj, meta = get_model(model_id)
        except KeyError:
            return jsonify({"error": f"model '{model_id}' not found"}), 404

        data = request.get_json(silent=True) or {}
        n_features = meta["n_features"]
        features, err = validate_predict_request(data, n_features)
        if err:
            return jsonify({"error": err, "model_id": model_id}), 400

        t0 = time.perf_counter()
        if meta["framework"] == "sklearn":
            X = np.array([features])
            pred = int(model_obj.predict(X)[0])
            proba = model_obj.predict_proba(X)[0].tolist()
        elif meta["framework"] == "pytorch":
            torch_model, scaler = model_obj
            X_arr = np.array([features], dtype=np.float32)
            X_scaled = scaler.transform(X_arr).astype(np.float32)
            with torch.no_grad():
                logits = torch_model(torch.tensor(X_scaled))
                proba_t = torch.softmax(logits, dim=1)[0]
                proba = proba_t.tolist()
                pred = int(torch.argmax(proba_t).item())
        else:
            return jsonify({"error": "unsupported framework"}), 500

        inference_ms = (time.perf_counter() - t0) * 1000
        class_names = meta.get("class_names", [])
        return jsonify({
            "model_id": model_id,
            "model_version": meta["version"],
            "prediction": pred,
            "class_name": class_names[pred] if class_names else None,
            "probabilities": {
                name: round(p, 6)
                for name, p in zip(class_names, proba)
            } if class_names else proba,
            "inference_ms": round(inference_ms, 3),
        }), 200

    @bp.route("/predict/<model_id>/batch", methods=["POST"])
    def predict_batch(model_id):
        try:
            model_obj, meta = get_model(model_id)
        except KeyError:
            return jsonify({"error": f"model '{model_id}' not found"}), 404

        data = request.get_json(silent=True) or {}
        instances = data.get("instances")
        if not isinstance(instances, list):
            return jsonify({"error": "'instances' must be a list"}), 400
        if len(instances) > 256:
            return jsonify({"error": "batch size exceeds 256"}), 400

        n_features = meta["n_features"]
        validated = []
        for i, inst in enumerate(instances):
            feats, err = validate_predict_request({"features": inst}, n_features)
            if err:
                return jsonify({"error": f"instance[{i}]: {err}"}), 400
            validated.append(feats)

        t0 = time.perf_counter()
        X = np.array(validated)
        if meta["framework"] == "sklearn":
            preds = model_obj.predict(X).tolist()
            probas = model_obj.predict_proba(X).tolist()
        elif meta["framework"] == "pytorch":
            torch_model, scaler = model_obj
            X_scaled = scaler.transform(X).astype(np.float32)
            with torch.no_grad():
                logits = torch_model(torch.tensor(X_scaled))
                proba_t = torch.softmax(logits, dim=1)
                preds = torch.argmax(proba_t, dim=1).tolist()
                probas = proba_t.tolist()
        else:
            return jsonify({"error": "unsupported framework"}), 500

        inference_ms = (time.perf_counter() - t0) * 1000
        class_names = meta.get("class_names", [])
        return jsonify({
            "model_id": model_id,
            "n_instances": len(validated),
            "predictions": [
                {
                    "prediction": int(p),
                    "class_name": class_names[int(p)] if class_names else None,
                    "probabilities": [round(v, 6) for v in pb],
                }
                for p, pb in zip(preds, probas)
            ],
            "inference_ms": round(inference_ms, 3),
        }), 200

    app.register_blueprint(bp, url_prefix="/api/v1")
    return app


# ── Demo ───────────────────────────────────────────────────────

def demo_endpoints(app):
    client = app.test_client()
    iris_data = [5.1, 3.5, 1.4, 0.2]  # setosa

    section("LIST REGISTERED MODELS")
    resp = client.get("/api/v1/models")
    data = resp.get_json()
    for model_id, info in data.items():
        meta = info["metadata"]
        print(f"  {model_id}: {meta['type']} ({meta['framework']}) v{meta['version']}")

    section("SINGLE PREDICTION — sklearn RF")
    resp = client.post("/api/v1/predict/iris_rf_v1",
                       json={"features": iris_data})
    d = resp.get_json()
    print(f"  Input:       {iris_data}")
    print(f"  Prediction:  {d['prediction']} ({d['class_name']})")
    print(f"  Probabilities: {d['probabilities']}")
    print(f"  Latency:     {d['inference_ms']} ms")

    section("SINGLE PREDICTION — sklearn LR")
    resp = client.post("/api/v1/predict/iris_lr_v1",
                       json={"features": iris_data})
    d = resp.get_json()
    print(f"  Prediction:  {d['prediction']} ({d['class_name']})")
    print(f"  Probabilities: {d['probabilities']}")

    if TORCH_AVAILABLE:
        section("SINGLE PREDICTION — PyTorch MLP")
        resp = client.post("/api/v1/predict/iris_torch_v1",
                           json={"features": iris_data})
        d = resp.get_json()
        print(f"  Prediction:  {d['prediction']} ({d['class_name']})")
        print(f"  Probabilities: {d['probabilities']}")

    section("BATCH PREDICTION")
    batch_instances = [
        [5.1, 3.5, 1.4, 0.2],   # setosa
        [6.3, 3.3, 6.0, 2.5],   # virginica
        [5.5, 2.4, 3.8, 1.1],   # versicolor
    ]
    resp = client.post("/api/v1/predict/iris_rf_v1/batch",
                       json={"instances": batch_instances})
    d = resp.get_json()
    print(f"  Batch size: {d['n_instances']}, inference: {d['inference_ms']}ms")
    for i, pred in enumerate(d["predictions"]):
        print(f"  [{i}] → {pred['prediction']} ({pred['class_name']})")

    section("VALIDATION ERRORS")
    cases = [
        ("model not found",    "iris_missing",   {"features": [1,2,3,4]}),
        ("wrong feature count","iris_rf_v1",      {"features": [1,2,3]}),
        ("non-numeric feature","iris_rf_v1",      {"features": [1,2,"x",4]}),
        ("NaN feature",        "iris_rf_v1",      {"features": [1,2,float("nan"),4]}),
        ("missing field",      "iris_rf_v1",      {}),
    ]
    for desc, model_id, body in cases:
        resp = client.post(f"/api/v1/predict/{model_id}", json=body)
        d = resp.get_json()
        print(f"  [{resp.status_code}] {desc}: {d.get('error', '')[:60]}")

    section("THROUGHPUT BENCHMARK")
    n_requests = 100
    t0 = time.perf_counter()
    for _ in range(n_requests):
        client.post("/api/v1/predict/iris_rf_v1", json={"features": iris_data})
    elapsed = time.perf_counter() - t0
    print(f"  {n_requests} sequential requests in {elapsed*1000:.1f}ms")
    print(f"  Throughput: {n_requests/elapsed:.0f} req/s (test client, no network)")
    print(f"  Mean latency: {elapsed*1000/n_requests:.2f}ms/req")


def main():
    if not SKLEARN_AVAILABLE:
        print("scikit-learn required. Run: pip install scikit-learn flask")
        return

    section("TRAINING MODELS")
    t0 = time.perf_counter()
    train_sklearn_models()
    print(f"  sklearn models trained in {(time.perf_counter()-t0)*1000:.1f}ms")

    if TORCH_AVAILABLE:
        t0 = time.perf_counter()
        train_torch_model()
        print(f"  PyTorch model trained in {(time.perf_counter()-t0)*1000:.1f}ms")
    else:
        print("  PyTorch not installed — skipping torch model. Run: pip install torch")

    demo_serialization()

    if not FLASK_AVAILABLE:
        print("\nFlask not installed — skipping endpoint demo.")
        return

    app = create_serving_app()
    demo_endpoints(app)

    section("THREAD SAFETY ANALYSIS")
    print("""
  sklearn models after .fit() are READ-ONLY — safe for concurrent reads.
  Multiple Gunicorn worker threads can share the same model object.

  PyTorch models:
    - Set model.eval() after training — disables dropout/batchnorm randomness.
    - Use torch.no_grad() context — disables gradient tape (saves memory).
    - nn.Module forward() is thread-safe for inference (no mutable state).
    - Exception: models with custom mutable state in forward() — use threading.Lock().

  Model registry (_MODEL_REGISTRY) uses threading.RLock():
    - RLock (reentrant) allows same thread to acquire multiple times.
    - Protects concurrent model registration from multiple workers.
    - Read path (get_model) holds lock only during dict lookup — minimal contention.
    """)


if __name__ == "__main__":
    main()
