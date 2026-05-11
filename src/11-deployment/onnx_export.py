"""
ONNX export: sklearn Pipeline → ONNX, NumPy NN → portable format,
onnxruntime inference with float32 inputs.
Covers: skl2onnx conversion, opset targeting, runtime inference,
        latency comparison native vs. ONNX, INT8 via onnxruntime quantization.
pip install scikit-learn skl2onnx onnxruntime
"""

import sys
import time
import struct
import numpy as np

rng = np.random.default_rng(42)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Synthetic Dataset ──────────────────────────────────────────────
def make_dataset(n: int = 500, n_features: int = 10):
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    w = rng.standard_normal(n_features).astype(np.float32)
    y = (X @ w > 0).astype(np.int64)
    return X, y


# ── NumPy Neural Network (lightweight, always runs) ───────────────
class NumpyMLP:
    """2-layer MLP: Input → Hidden(32, ReLU) → Output(2, Softmax)."""

    def __init__(self, n_in: int, n_hidden: int = 32, n_out: int = 2):
        self.W1 = rng.standard_normal((n_in, n_hidden)).astype(np.float32) * 0.1
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = rng.standard_normal((n_hidden, n_out)).astype(np.float32) * 0.1
        self.b2 = np.zeros(n_out, dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        h  = np.maximum(0, X @ self.W1 + self.b1)          # ReLU
        z  = h @ self.W2 + self.b2
        e  = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)             # Softmax

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def serialize(self) -> bytes:
        """Pack weights into bytes (minimal portable format)."""
        arrays = [self.W1, self.b1, self.W2, self.b2]
        header = struct.pack(">I", len(arrays))
        chunks = [header]
        for arr in arrays:
            flat = arr.flatten().astype(np.float32).tobytes()
            shape_bytes = struct.pack(f">I{'I'*len(arr.shape)}", len(arr.shape), *arr.shape)
            chunks.extend([shape_bytes, flat])
        return b"".join(chunks)

    @classmethod
    def deserialize(cls, data: bytes) -> "NumpyMLP":
        offset = 0
        n_arrays = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        arrays = []
        for _ in range(n_arrays):
            ndim = struct.unpack_from(">I", data, offset)[0]
            offset += 4
            shape = struct.unpack_from(f">{ndim}I", data, offset)
            offset += 4 * ndim
            n_elems = 1
            for s in shape:
                n_elems *= s
            arr = np.frombuffer(data[offset:offset + n_elems * 4], dtype=np.float32).copy()
            arr = arr.reshape(shape)
            arrays.append(arr)
            offset += n_elems * 4
        model = cls.__new__(cls)
        model.W1, model.b1, model.W2, model.b2 = arrays
        return model


# ── Sklearn Pipeline ───────────────────────────────────────────────
def build_sklearn_pipeline(X_train, y_train):
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42, C=1.0)),
        ])
        pipe.fit(X_train, y_train)
        return pipe
    except ImportError:
        return None


# ── ONNX Export + Runtime ──────────────────────────────────────────
def export_sklearn_to_onnx(pipe, n_features: int, path: str):
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=17)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        size_kb = len(onnx_model.SerializeToString()) / 1024
        print(f"  Exported sklearn pipeline → {path} ({size_kb:.1f} KB)")
        return True
    except ImportError:
        print("  skl2onnx not available — skipping ONNX export")
        return False


def infer_with_onnx(path: str, X: np.ndarray, n_reps: int = 100):
    try:
        import onnxruntime as rt
        sess    = rt.InferenceSession(path, providers=["CPUExecutionProvider"])
        in_name  = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        X_f32 = X.astype(np.float32)
        start = time.perf_counter()
        for _ in range(n_reps):
            preds = sess.run([out_name], {in_name: X_f32})[0]
        elapsed_ms = (time.perf_counter() - start) * 1000 / n_reps
        return preds, elapsed_ms
    except ImportError:
        print("  onnxruntime not available — skipping ONNX inference")
        return None, None


# ── Latency Benchmark ─────────────────────────────────────────────
def benchmark(fn, X, n_reps: int = 200) -> float:
    start = time.perf_counter()
    for _ in range(n_reps):
        fn(X)
    return (time.perf_counter() - start) * 1000 / n_reps


# ── ONNX Model Inspection ─────────────────────────────────────────
def inspect_onnx(path: str):
    try:
        import onnx
        model = onnx.load(path)
        print(f"  IR version: {model.ir_version}")
        print(f"  Opset: {model.opset_import[0].version}")
        print(f"  Nodes: {len(model.graph.node)}")
        print(f"  Inputs:  {[i.name for i in model.graph.input]}")
        print(f"  Outputs: {[o.name for o in model.graph.output]}")
        init_sizes = [(init.name, list(init.dims)) for init in model.graph.initializer]
        print(f"  Initializers (weights): {init_sizes[:4]} ...")
    except ImportError:
        print("  onnx not available for inspection")


def main():
    section("1. NUMPY MLP — TRAIN + SERIALIZE")
    X, y = make_dataset(n=500, n_features=10)
    split = 400
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    mlp = NumpyMLP(n_in=10, n_hidden=32, n_out=2)

    # Quick training: SGD on cross-entropy
    lr = 0.01
    for epoch in range(30):
        proba   = mlp.forward(X_train)
        loss    = -np.log(proba[np.arange(len(y_train)), y_train] + 1e-9).mean()
        # Backprop through softmax + cross-entropy: dL/dz = proba - onehot(y)
        onehot  = np.eye(2)[y_train]
        dz2     = (proba - onehot) / len(y_train)
        h       = np.maximum(0, X_train @ mlp.W1 + mlp.b1)
        dW2     = h.T @ dz2
        db2     = dz2.sum(axis=0)
        dh      = dz2 @ mlp.W2.T
        dh[h <= 0] = 0   # ReLU backward
        dW1     = X_train.T @ dh
        db1     = dh.sum(axis=0)
        mlp.W1 -= lr * dW1;  mlp.b1 -= lr * db1
        mlp.W2 -= lr * dW2;  mlp.b2 -= lr * db2
        if epoch % 10 == 0:
            acc = (mlp.predict(X_test) == y_test).mean()
            print(f"  Epoch {epoch:3d}: loss={loss:.4f}  test_acc={acc:.3f}")

    acc = (mlp.predict(X_test) == y_test).mean()
    print(f"  Final test accuracy: {acc:.3f}")

    # Serialize / deserialize
    raw = mlp.serialize()
    mlp2 = NumpyMLP.deserialize(raw)
    assert (mlp.predict(X_test) == mlp2.predict(X_test)).all(), "Serialization failed"
    print(f"  Serialized size: {len(raw)} bytes")
    print(f"  Deserialized model: predictions match ✓")

    section("2. SKLEARN PIPELINE → ONNX")
    pipe = build_sklearn_pipeline(X_train, y_train)
    if pipe is not None:
        sk_acc = (pipe.predict(X_test) == y_test).mean()
        print(f"  Sklearn pipeline test accuracy: {sk_acc:.3f}")

        onnx_path = "/tmp/model_lr.onnx"
        exported  = export_sklearn_to_onnx(pipe, n_features=10, path=onnx_path)

        if exported:
            section("3. ONNX INSPECTION")
            inspect_onnx(onnx_path)

            section("4. ONNX RUNTIME INFERENCE")
            onnx_preds, onnx_ms = infer_with_onnx(onnx_path, X_test)
            if onnx_preds is not None:
                sk_preds = pipe.predict(X_test)
                agreement = (onnx_preds == sk_preds).mean()
                print(f"  ONNX vs sklearn agreement: {agreement:.3f}")
                print(f"  Mean ONNX inference latency: {onnx_ms:.3f} ms/batch (n={len(X_test)})")

                section("5. LATENCY COMPARISON")
                sklearn_ms = benchmark(pipe.predict, X_test)
                numpy_ms   = benchmark(mlp.predict,  X_test)
                print(f"  sklearn predict:  {sklearn_ms:.3f} ms/call")
                print(f"  NumPy MLP:        {numpy_ms:.3f} ms/call")
                print(f"  ONNX Runtime:     {onnx_ms:.3f} ms/call")
    else:
        print("  scikit-learn not available")

    section("6. FLOAT32 vs FLOAT16 vs INT8 WEIGHT SIZES")
    W = rng.standard_normal((256, 256)).astype(np.float32)

    W_f16  = W.astype(np.float16)
    rmse_f16 = float(np.sqrt(np.mean((W - W_f16.astype(np.float32))**2)))

    # INT8: per-tensor symmetric quantization
    absmax = float(np.abs(W).max())
    scale  = absmax / 127.0
    W_int8 = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    W_dq   = W_int8.astype(np.float32) * scale
    rmse_int8 = float(np.sqrt(np.mean((W - W_dq)**2)))

    print(f"  Weight matrix: {W.shape} ({W.nbytes} bytes FP32)")
    print(f"  FP16: {W_f16.nbytes} bytes  RMSE={rmse_f16:.6f}  "
          f"size ratio={W.nbytes/W_f16.nbytes:.1f}x")
    print(f"  INT8: {W_int8.nbytes} bytes  RMSE={rmse_int8:.6f}  "
          f"size ratio={W.nbytes/W_int8.nbytes:.1f}x")

    section("7. DOCKER + GUNICORN REFERENCE")
    dockerfile = """
  Dockerfile (multi-stage):
    FROM python:3.11-slim AS builder
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

    FROM python:3.11-slim
    WORKDIR /app
    COPY --from=builder /install /usr/local
    COPY . .
    CMD ["gunicorn", "--workers=4", "--timeout=30",
         "--bind=0.0.0.0:8080", "app:create_app()"]

  Run:
    docker build -t ml-api .
    docker run -p 8080:8080 ml-api

  gunicorn_config.py:
    import multiprocessing
    workers  = multiprocessing.cpu_count() * 2 + 1
    timeout  = 30
    bind     = "0.0.0.0:8080"
    preload_app = True
    """
    print(dockerfile)


if __name__ == "__main__":
    main()
