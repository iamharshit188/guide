"""
Model quantization: FP32 → FP16 / INT8 / INT4 from scratch.
Covers: per-tensor + per-channel quantization, dynamic vs. static,
        calibration, quantization error analysis, speedup simulation.
No external deps required.
pip install numpy (already present)
"""

import time
import numpy as np

rng = np.random.default_rng(42)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Precision helpers ─────────────────────────────────────────────
def rmse(a, b):
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64))**2)))


def snr_db(original, reconstructed):
    signal_power = float(np.mean(original.astype(np.float64)**2))
    noise_power  = float(np.mean((original.astype(np.float64)
                                  - reconstructed.astype(np.float64))**2))
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


# ── FP16 ──────────────────────────────────────────────────────────
def quantize_fp16(W: np.ndarray):
    W_f16 = W.astype(np.float16)
    W_rec = W_f16.astype(np.float32)
    return W_f16, W_rec


# ── INT8 Per-Tensor (Symmetric) ────────────────────────────────────
def quantize_int8_symmetric(W: np.ndarray):
    """
    Scale: s = max(|W|) / 127
    Quantize: q = round(W / s)  clipped to [-127, 127]
    Dequantize: W_rec = q * s
    """
    absmax = float(np.abs(W).max())
    if absmax == 0:
        return np.zeros_like(W, dtype=np.int8), 0.0, W.copy()
    scale  = absmax / 127.0
    W_int8 = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    W_rec  = W_int8.astype(np.float32) * scale
    return W_int8, scale, W_rec


# ── INT8 Per-Tensor (Asymmetric / Affine) ─────────────────────────
def quantize_int8_asymmetric(W: np.ndarray):
    """
    Zero-point affine: q = round(W / scale) + zero_point
    Maps [W_min, W_max] → [0, 255]  (uint8)
    Dequantize: W_rec = (q - zero_point) * scale
    """
    W_min = float(W.min())
    W_max = float(W.max())
    scale = (W_max - W_min) / 255.0
    if scale == 0:
        return np.zeros_like(W, dtype=np.uint8), scale, 0, W.copy()
    zero_point = int(round(-W_min / scale))
    zero_point = np.clip(zero_point, 0, 255)
    W_uint8 = np.clip(np.round(W / scale) + zero_point, 0, 255).astype(np.uint8)
    W_rec   = (W_uint8.astype(np.float32) - zero_point) * scale
    return W_uint8, scale, zero_point, W_rec


# ── INT8 Per-Channel ──────────────────────────────────────────────
def quantize_int8_per_channel(W: np.ndarray, axis: int = 0):
    """
    Per-output-channel quantization (reduces quantization error
    for weight tensors with large inter-channel magnitude variance).
    """
    n_channels = W.shape[axis]
    scales  = []
    W_int8  = np.empty_like(W, dtype=np.int8)
    W_rec   = np.empty_like(W, dtype=np.float32)

    for c in range(n_channels):
        slc = [slice(None)] * W.ndim
        slc[axis] = c
        w_c    = W[tuple(slc)]
        absmax = float(np.abs(w_c).max())
        s      = absmax / 127.0 if absmax > 0 else 1.0
        q_c    = np.clip(np.round(w_c / s), -127, 127).astype(np.int8)
        W_int8[tuple(slc)] = q_c
        W_rec[tuple(slc)]  = q_c.astype(np.float32) * s
        scales.append(s)

    return W_int8, np.array(scales, dtype=np.float32), W_rec


# ── INT4 (Uniform) ────────────────────────────────────────────────
def quantize_int4(W: np.ndarray):
    """
    4-bit uniform: 16 levels in [-1, 1] after scaling by absmax.
    Each value stored as 4 bits (2 values per byte).
    """
    absmax = float(np.abs(W).max())
    if absmax == 0:
        return W.copy()
    W_norm  = W / absmax
    W_q4    = np.round(W_norm * 7).clip(-7, 7).astype(np.int8) / 7.0
    W_rec   = (W_q4 * absmax).astype(np.float32)
    return W_rec


# ── NF4 (Normal Float 4) ──────────────────────────────────────────
def quantize_nf4(W: np.ndarray):
    """
    NF4 from QLoRA: 16 quantization levels whose positions are the
    quantiles of N(0,1), then renormalized to [-1,1].
    Optimal for normally-distributed weights.
    """
    try:
        from scipy.stats import norm
        quantiles = [norm.ppf(i / 16) for i in range(1, 16)]
    except ImportError:
        # Hardcoded NF4 quantile values (approximate)
        quantiles = [
            -1.0000, -0.6962, -0.5251, -0.3949, -0.2844,
            -0.1848, -0.0912,  0.0000,  0.0912,  0.1848,
             0.2844,  0.3949,  0.5251,  0.6962,  1.0000,
        ]

    # 16 codes including -1 and +1
    codes = np.array([-1.0] + quantiles + [1.0], dtype=np.float32)
    # Normalize codes to [-1, 1]
    codes = codes / max(abs(codes.min()), abs(codes.max()))

    absmax = float(np.abs(W).max())
    if absmax == 0:
        return W.copy()
    W_norm = (W / absmax).astype(np.float32)

    # Vectorized nearest-code lookup
    diffs   = np.abs(W_norm[:, :, None] - codes[None, None, :])
    indices = diffs.argmin(axis=2)
    W_nf4   = codes[indices] * absmax
    return W_nf4.astype(np.float32)


# ── Calibration-based Static Quantization ─────────────────────────
class StaticQuantizer:
    """
    Collects activation statistics on a calibration dataset,
    then computes per-layer scale and zero_point.
    """

    def __init__(self, method: str = "minmax"):
        self.method  = method   # "minmax" or "percentile"
        self._stats: dict = {}  # layer_name → {"min": ..., "max": ...}

    def observe(self, layer_name: str, activation: np.ndarray,
                percentile: float = 99.9):
        if layer_name not in self._stats:
            self._stats[layer_name] = {"min": float("inf"), "max": float("-inf")}

        if self.method == "minmax":
            self._stats[layer_name]["min"] = min(self._stats[layer_name]["min"],
                                                  float(activation.min()))
            self._stats[layer_name]["max"] = max(self._stats[layer_name]["max"],
                                                  float(activation.max()))
        else:  # percentile
            lo = float(np.percentile(activation, 100 - percentile))
            hi = float(np.percentile(activation, percentile))
            self._stats[layer_name]["min"] = min(self._stats[layer_name]["min"], lo)
            self._stats[layer_name]["max"] = max(self._stats[layer_name]["max"], hi)

    def get_scale_zp(self, layer_name: str):
        stats = self._stats.get(layer_name, {"min": -1.0, "max": 1.0})
        scale = (stats["max"] - stats["min"]) / 255.0
        if scale == 0:
            scale = 1e-8
        zp = int(round(-stats["min"] / scale))
        return scale, np.clip(zp, 0, 255)


def main():
    section("1. WEIGHT MATRIX QUANTIZATION COMPARISON")
    shapes = [(64, 64), (256, 256), (512, 512)]

    for shape in shapes:
        W = rng.standard_normal(shape).astype(np.float32)

        _, W_rec_f16  = quantize_fp16(W)
        _, _, W_rec_s = quantize_int8_symmetric(W)
        _, _, _, W_rec_a = quantize_int8_asymmetric(W)
        _, _, W_rec_c = quantize_int8_per_channel(W, axis=0)
        W_rec_i4  = quantize_int4(W)
        W_rec_nf4 = quantize_nf4(W)

        print(f"\n  W{shape} (FP32 = {W.nbytes} bytes):")
        print(f"    {'Method':<18} {'RMSE':>10} {'SNR (dB)':>10} {'Bytes':>8} {'Ratio':>6}")
        print(f"    {'-'*58}")
        for name, rec, n_bytes in [
            ("FP16",          W_rec_f16,  W.nbytes // 2),
            ("INT8 symmetric", W_rec_s,   W.nbytes // 4),
            ("INT8 asymmetric", W_rec_a,  W.nbytes // 4),
            ("INT8 per-channel", W_rec_c, W.nbytes // 4),
            ("INT4 uniform",  W_rec_i4,   W.nbytes // 8),
            ("NF4",           W_rec_nf4,  W.nbytes // 8),
        ]:
            r  = rmse(W, rec)
            s  = snr_db(W, rec)
            ratio = W.nbytes / n_bytes
            print(f"    {name:<18} {r:>10.6f} {s:>10.2f} {n_bytes:>8} {ratio:>5.1f}x")

    section("2. PER-TENSOR vs PER-CHANNEL INT8 ON SKEWED WEIGHTS")
    # Create weight matrix with large inter-channel magnitude variance
    W_skewed = np.vstack([
        rng.standard_normal((32, 64)).astype(np.float32) * 0.01,    # small channels
        rng.standard_normal((32, 64)).astype(np.float32) * 10.0,    # large channels
    ])

    _, _, W_pt = quantize_int8_symmetric(W_skewed)    # per-tensor
    _, _, W_pc = quantize_int8_per_channel(W_skewed, axis=0)  # per-channel

    print(f"\n  Skewed weight matrix: channels 0-31 have small magnitude,")
    print(f"  channels 32-63 have 1000× larger magnitude.")
    print(f"  Per-tensor RMSE:   {rmse(W_skewed, W_pt):.6f}")
    print(f"  Per-channel RMSE:  {rmse(W_skewed, W_pc):.6f}")
    print(f"  → Per-channel dramatically reduces error for skewed distributions")

    section("3. CALIBRATION-BASED STATIC QUANTIZATION")
    calibrator = StaticQuantizer(method="minmax")

    # Simulate calibration pass on 1000 samples
    for batch in range(50):
        # Simulate activations at two layers
        act1 = rng.standard_normal((20, 64)).astype(np.float32) * 2.0
        act2 = np.maximum(0, act1 @ rng.standard_normal((64, 32)).astype(np.float32))
        calibrator.observe("layer1", act1)
        calibrator.observe("layer2", act2)

    for layer in ["layer1", "layer2"]:
        s, zp = calibrator.get_scale_zp(layer)
        stats = calibrator._stats[layer]
        print(f"  {layer}: min={stats['min']:.3f}  max={stats['max']:.3f}  "
              f"scale={s:.6f}  zero_point={zp}")

    section("4. DYNAMIC vs STATIC QUANTIZATION LATENCY")
    W = rng.standard_normal((512, 512)).astype(np.float32)
    X = rng.standard_normal((32, 512)).astype(np.float32)

    n_reps = 500

    # FP32 baseline
    start = time.perf_counter()
    for _ in range(n_reps):
        _ = X @ W
    fp32_ms = (time.perf_counter() - start) * 1000 / n_reps

    # Dynamic INT8: quantize W per-inference, compute in float (simulated)
    start = time.perf_counter()
    for _ in range(n_reps):
        _, scale, W_q = quantize_int8_symmetric(W)
        _ = X @ W_q.astype(np.float32) * scale
    dynamic_ms = (time.perf_counter() - start) * 1000 / n_reps

    # Static INT8: W pre-quantized, only dequantize during inference
    _, scale_pre, W_int8_pre = quantize_int8_symmetric(W)
    start = time.perf_counter()
    for _ in range(n_reps):
        _ = X @ (W_int8_pre.astype(np.float32) * scale_pre)
    static_ms = (time.perf_counter() - start) * 1000 / n_reps

    print(f"  FP32 MatMul ({W.shape}):        {fp32_ms:.4f} ms/call")
    print(f"  Dynamic INT8 (quant each call): {dynamic_ms:.4f} ms/call  "
          f"(overhead={dynamic_ms/fp32_ms:.2f}x)")
    print(f"  Static INT8 (pre-quantized W):  {static_ms:.4f} ms/call  "
          f"(overhead={static_ms/fp32_ms:.2f}x)")
    print("\n  Note: real INT8 speedup requires hardware INT8 kernels.")
    print("  NumPy always uses FP32 internally — these numbers measure")
    print("  the quantization overhead, not inference acceleration.")

    section("5. NF4 vs INT4 — WHY NF4 IS BETTER FOR LLM WEIGHTS")
    W_normal = rng.standard_normal((256, 256)).astype(np.float32)  # N(0,1)
    W_uniform = rng.uniform(-1, 1, (256, 256)).astype(np.float32)  # Uniform

    print(f"\n  Weight distribution comparison (same 4-bit budget):")
    print(f"  {'Distribution':<16} {'Method':<12} {'RMSE':>10} {'SNR(dB)':>10}")
    print(f"  {'-'*52}")

    for W_desc, W_data in [("Normal N(0,1)", W_normal), ("Uniform(-1,1)", W_uniform)]:
        for method, rec in [("INT4", quantize_int4(W_data)), ("NF4", quantize_nf4(W_data))]:
            r = rmse(W_data, rec)
            s = snr_db(W_data, rec)
            print(f"  {W_desc:<16} {method:<12} {r:>10.6f} {s:>10.2f}")

    print("\n  → NF4 outperforms INT4 for normally distributed weights")
    print("    (neural network weights are approximately Gaussian)")
    print("    NF4 codes are optimally placed at Gaussian quantiles")

    section("6. QUANTIZATION SUMMARY TABLE")
    print(f"""
  Format  Bits  Bytes/param  Notes
  ------  ----  -----------  -----
  FP32      32  4.0          Baseline, full precision
  FP16      16  2.0          2× smaller; GPU tensor cores accelerate
  BF16      16  2.0          Better dynamic range than FP16 (same mantissa as INT8)
  INT8       8  1.0          4× smaller; CPU-friendly; needs scale+zero_point
  INT4       4  0.5          8× smaller; uniform; higher error
  NF4        4  0.5          8× smaller; Gaussian-optimal; used in QLoRA
  FP8        8  1.0          Emerging standard; H100 native support

  Rule of thumb:
    FP16/BF16 → GPU serving, minimal accuracy loss
    INT8       → CPU serving, fast, minimal calibration
    NF4        → LLM weight storage (QLoRA), not for activations
""")


if __name__ == "__main__":
    main()
