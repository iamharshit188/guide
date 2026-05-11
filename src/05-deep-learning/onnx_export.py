"""
ONNX export walkthrough — Deep Learning context.
Covers: ONNX graph structure (nodes, inputs, outputs, initializers),
        exporting sklearn Pipeline to ONNX, exporting a NumPy MLP,
        onnxruntime inference, latency benchmarking.
All with graceful fallback when onnx/onnxruntime/sklearn not installed.
"""

import struct
import time
import numpy as np

rng = np.random.default_rng(42)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── ONNX Conceptual Overview ──────────────────────────────────────
def explain_onnx_graph():
    """
    An ONNX model is a computation graph:
      - Nodes:        operation (MatMul, Relu, Sigmoid, ...)
      - Inputs:       named tensors (data flowing in)
      - Outputs:      named tensors (data flowing out)
      - Initializers: named constant tensors (weights, biases)

    Example: Linear layer y = x @ W + b
      Initializer "W" (shape d_in × d_out)
      Initializer "b" (shape d_out)
      Node: MatMul(input, W) → "matmul_out"
      Node: Add("matmul_out", b) → "output"
    """
    print("""
  ONNX graph anatomy:
  ┌─────────────────────────────────────────────────────────┐
  │ ModelProto                                              │
  │  └─ GraphProto                                         │
  │       ├─ input:       [name, type, shape]              │
  │       ├─ output:      [name, type, shape]              │
  │       ├─ initializer: [name, dtype, shape, raw_data]   │
  │       └─ node:        [op_type, inputs, outputs, attrs]│
  └─────────────────────────────────────────────────────────┘

  Opset version: defines the operator set (opset 17 = current stable).
  Key ops: MatMul, Gemm, Relu, Sigmoid, Softmax, Reshape, Transpose,
           Conv, BatchNormalization, LSTM, Gather, LayerNormalization.
""")


# ── Pure NumPy MLP → manual ONNX ─────────────────────────────────
class NumpyMLP:
    """
    Two-layer MLP: x → Linear(d_in, 64) → ReLU → Linear(64, d_out) → Sigmoid
    Trained toy demo; exports weights for ONNX.
    """

    def __init__(self, d_in: int = 10, d_hidden: int = 64, d_out: int = 1):
        scale = d_in ** -0.5
        self.W1 = rng.standard_normal((d_in, d_hidden)) * scale
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.standard_normal((d_hidden, d_out)) * (d_hidden ** -0.5)
        self.b2 = np.zeros(d_out)

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ self.W1 + self.b1)   # ReLU
        return 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))  # Sigmoid

    def to_bytes(self) -> bytes:
        """Minimal serialization: header + raw numpy arrays."""
        arrays = [self.W1, self.b1, self.W2, self.b2]
        parts  = []
        for arr in arrays:
            flat = arr.astype(np.float32).flatten()
            parts.append(struct.pack("i", len(flat)))
            parts.append(flat.tobytes())
        return b"NMLP" + b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "NumpyMLP":
        assert data[:4] == b"NMLP"
        shapes = [(10, 64), (64,), (64, 1), (1,)]
        obj = cls.__new__(cls)
        offset = 4
        for attr, shape in zip(["W1", "b1", "W2", "b2"], shapes):
            n = struct.unpack_from("i", data, offset)[0]
            offset += 4
            arr = np.frombuffer(data[offset:offset + n*4], dtype=np.float32)
            setattr(obj, attr, arr.reshape(shape).copy())
            offset += n * 4
        return obj


def build_onnx_mlp(mlp: NumpyMLP) -> bytes:
    """
    Manually construct an ONNX protobuf for the NumpyMLP.
    Returns raw bytes of a valid ONNX model (opset 17).

    This demonstrates the ONNX graph structure without relying on
    onnx library — uses raw protobuf encoding via struct.
    Falls back to a description if onnx is not installed.
    """
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper

        W1 = numpy_helper.from_array(mlp.W1.astype(np.float32), name="W1")
        b1 = numpy_helper.from_array(mlp.b1.astype(np.float32), name="b1")
        W2 = numpy_helper.from_array(mlp.W2.astype(np.float32), name="W2")
        b2 = numpy_helper.from_array(mlp.b2.astype(np.float32), name="b2")

        # Layer 1: Gemm(x, W1, b1) = x @ W1 + b1   (transB=0)
        gemm1 = helper.make_node("Gemm", ["input", "W1", "b1"],
                                 ["gemm1_out"], transA=0, transB=0, alpha=1.0, beta=1.0)
        relu  = helper.make_node("Relu", ["gemm1_out"], ["relu_out"])
        gemm2 = helper.make_node("Gemm", ["relu_out", "W2", "b2"],
                                 ["gemm2_out"], transA=0, transB=0, alpha=1.0, beta=1.0)
        sigmoid = helper.make_node("Sigmoid", ["gemm2_out"], ["output"])

        graph = helper.make_graph(
            [gemm1, relu, gemm2, sigmoid],
            "numpy_mlp",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, mlp.W1.shape[0]])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])],
            initializer=[W1, b1, W2, b2],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        return model.SerializeToString()

    except ImportError:
        return b""   # onnx not installed


def run_onnx_inference(model_bytes: bytes, X: np.ndarray,
                       n_reps: int = 100) -> dict:
    """Run inference using onnxruntime and benchmark latency."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_bytes)
        input_name  = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Warmup
        _ = sess.run([output_name], {input_name: X.astype(np.float32)})[0]

        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            out = sess.run([output_name], {input_name: X.astype(np.float32)})[0]
            times.append((time.perf_counter() - t0) * 1000)

        return {
            "output": out,
            "mean_ms": np.mean(times),
            "p50_ms":  np.percentile(times, 50),
            "p99_ms":  np.percentile(times, 99),
        }

    except ImportError:
        return {}


def export_sklearn_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray) -> dict:
    """
    Export sklearn StandardScaler + LogisticRegression pipeline to ONNX.
    Returns inference results if sklearn + skl2onnx are available.
    """
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=200)),
        ])
        pipe.fit(X_train, y_train)
        sklearn_preds = pipe.predict_proba(X_test)[:, 1]

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            onnx_model = convert_sklearn(
                pipe,
                name="sklearn_pipeline",
                initial_types=[("input", FloatTensorType([None, X_train.shape[1]]))],
                target_opset=17,
            )
            model_bytes = onnx_model.SerializeToString()

            import onnxruntime as ort
            sess = ort.InferenceSession(model_bytes)
            ort_preds = sess.run(
                ["probabilities"],
                {"input": X_test.astype(np.float32)}
            )[0][:, 1]

            max_diff = float(np.abs(sklearn_preds - ort_preds).max())
            return {"sklearn_preds": sklearn_preds, "ort_preds": ort_preds,
                    "max_diff": max_diff, "model_bytes": len(model_bytes)}

        except ImportError:
            return {"sklearn_preds": sklearn_preds, "skl2onnx": "not installed"}

    except ImportError:
        return {"sklearn": "not installed"}


def main():
    section("1. ONNX GRAPH STRUCTURE")
    explain_onnx_graph()

    section("2. NUMPY MLP EXPORT")
    print("  Building NumpyMLP and exporting to ONNX format...")
    mlp = NumpyMLP(d_in=10, d_hidden=64, d_out=1)

    # Verify numpy predictions first
    X_demo = rng.standard_normal((8, 10)).astype(np.float32)
    np_preds = mlp.predict(X_demo)
    print(f"  NumPy MLP predictions (first 5): {np_preds.flatten()[:5]}")

    # Export to ONNX
    model_bytes = build_onnx_mlp(mlp)
    if model_bytes:
        print(f"  ONNX model size: {len(model_bytes)} bytes")
        result = run_onnx_inference(model_bytes, X_demo, n_reps=200)
        if result:
            ort_preds = result["output"].flatten()
            max_diff = float(np.abs(np_preds.flatten() - ort_preds).max())
            print(f"  onnxruntime predictions (first 5): {ort_preds[:5]}")
            print(f"  Max prediction difference (numpy vs ort): {max_diff:.2e}")
            print(f"\n  Latency over {200} runs (batch_size=8):")
            print(f"    mean={result['mean_ms']:.3f}ms  "
                  f"p50={result['p50_ms']:.3f}ms  p99={result['p99_ms']:.3f}ms")
        else:
            print("  onnxruntime not installed — skipping inference benchmark")
    else:
        print("  onnx not installed — skipping ONNX export")
        print("  Install: pip install onnx onnxruntime")

    section("3. SKLEARN PIPELINE EXPORT")
    print("  Training sklearn StandardScaler + LogisticRegression pipeline...")
    n = 500
    X_all = rng.standard_normal((n, 10))
    y_all = (X_all[:, 0] + X_all[:, 1] > 0).astype(int)
    X_tr, y_tr = X_all[:400], y_all[:400]
    X_te = X_all[400:]

    result = export_sklearn_pipeline(X_tr, y_tr, X_te)

    if "sklearn" in result and result["sklearn"] == "not installed":
        print("  sklearn not installed — skipping")
    elif "skl2onnx" in result:
        preds = result["sklearn_preds"]
        print(f"  sklearn predictions (first 5): {preds[:5]}")
        print(f"  skl2onnx not installed — ONNX export skipped")
        print("  Install: pip install skl2onnx")
    elif "max_diff" in result:
        print(f"  sklearn predictions (first 5): {result['sklearn_preds'][:5]}")
        print(f"  onnxruntime preds (first 5):   {result['ort_preds'][:5]}")
        print(f"  Max numerical diff: {result['max_diff']:.2e}")
        print(f"  ONNX model size: {result['model_bytes']} bytes")

    section("4. ONNX vs. NATIVE NUMPY LATENCY")
    print("  Benchmarking numpy vs onnxruntime for MLP inference...")
    X_bench = rng.standard_normal((100, 10)).astype(np.float32)
    n_reps = 500

    # NumPy baseline
    times_np = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = mlp.predict(X_bench)
        times_np.append((time.perf_counter() - t0) * 1000)

    print(f"\n  NumPy MLP (batch=100, n={n_reps}):")
    print(f"    mean={np.mean(times_np):.3f}ms  "
          f"p50={np.percentile(times_np,50):.3f}ms  "
          f"p99={np.percentile(times_np,99):.3f}ms")

    if model_bytes:
        result = run_onnx_inference(model_bytes[:0] or model_bytes,
                                    X_bench, n_reps=n_reps)
        if result:
            print(f"\n  onnxruntime (batch=100, n={n_reps}):")
            print(f"    mean={result['mean_ms']:.3f}ms  "
                  f"p50={result['p50_ms']:.3f}ms  "
                  f"p99={result['p99_ms']:.3f}ms")
            speedup = np.mean(times_np) / (result['mean_ms'] + 1e-9)
            print(f"\n  onnxruntime speedup: {speedup:.2f}×")

    section("5. ONNX RUNTIME OPTIMIZATIONS")
    print("""
  onnxruntime applies graph optimizations automatically:

  Level 1 — Basic:
    - Constant folding: compute static subgraphs at load time
    - Redundant node elimination: remove identity ops
    - Semantics-preserving node fusions

  Level 2 — Extended:
    - Node fusion: Gemm+Relu → FusedGemm (single BLAS call)
    - MatMul+Add → GEMM (uses optimized BLAS routines)

  Level 3 — Layout optimization:
    - Reorder memory layout for better cache behavior (NCHW/NHWC)

  Execution providers:
    ort.InferenceSession(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

  Quantization (INT8):
    from onnxruntime.quantization import quantize_dynamic
    quantize_dynamic("model.onnx", "model_int8.onnx", weight_type=QuantType.QInt8)
    → 4× smaller weights, ~2× faster CPU inference, < 1% accuracy loss
""")

    section("6. ONNX OPSET TABLE")
    print("""
  ┌────────┬──────────────────────────────────────────────────┐
  │ Opset  │ Notable additions                                │
  ├────────┼──────────────────────────────────────────────────┤
  │  9     │ MeanVarianceNormalization, Shrink               │
  │  11    │ Cumsum, Range, Round, Resize (upsample)         │
  │  13    │ ScatterElements with reduction; Softmax axis fix │
  │  14    │ Trilu, BitShift; GRU/LSTM updates               │
  │  17    │ BlackmanWindow, DFT (FFT), LayerNormalization   │
  │  18    │ CenterCropPad, Col2Im, Mish                     │
  └────────┴──────────────────────────────────────────────────┘

  Specify opset when exporting: target_opset=17
  Check model opset: model.opset_import[0].version
  Always test inference after export — some ops have subtle semantics changes.
""")


if __name__ == "__main__":
    main()
