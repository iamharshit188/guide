"""
QLoRA: NF4 quantisation from scratch (quantile placement), double quantisation,
memory comparison table, BitsAndBytes pipeline (graceful skip).
pip install numpy  (bitsandbytes, peft, transformers optional)
"""

import numpy as np
import math
from typing import List, Tuple, Dict

RNG = np.random.default_rng(42)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# NF4 quantisation — from scratch
# ---------------------------------------------------------------------------

def normal_quantiles(n_levels: int = 16) -> np.ndarray:
    """
    Compute NF4 quantile levels: place n_levels points at equispaced
    quantiles of N(0,1). Renormalise to [-1, 1].

    q_i = Phi^{-1}(i / (n_levels - 1))  for i = 1..n_levels-1
    Then renorm so max(|q|) = 1.
    """
    # Probit via erfinv approximation (no scipy needed)
    from math import erf, log, sqrt

    def erfinv_approx(x: float) -> float:
        # Numerical approximation accurate to ~1e-5
        # Halley's method seed from series expansion
        a = 0.147
        ln = math.log(1 - x * x)
        t1 = 2 / (math.pi * a) + ln / 2
        t2 = ln / a
        return math.copysign(math.sqrt(math.sqrt(t1 * t1 - t2) - t1), x)

    def probit(p: float) -> float:
        # Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1)
        return math.sqrt(2) * erfinv_approx(2 * p - 1)

    # Place n_levels quantile midpoints: p_i = (i + 0.5) / n_levels
    levels = []
    for i in range(n_levels):
        p = (i + 0.5) / n_levels  # equispaced probability mass per bin
        levels.append(probit(p))

    levels = np.array(sorted(levels))

    # Renormalise to [-1, 1]
    max_abs = max(abs(levels[0]), abs(levels[-1]))
    levels = levels / max_abs

    # NF4 is symmetric — force exact [-1, 1] endpoints
    levels[0] = -1.0
    levels[-1] = 1.0

    return levels


NF4_LEVELS = normal_quantiles(n_levels=16)


def quantise_nf4(weights: np.ndarray,
                 block_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    NF4 quantise a 1D weight tensor.
    Returns:
        quantised: int array with values 0..15
        absmax: float32 array, one per block (dequantisation constant)
    """
    n = weights.shape[0]
    n_blocks = math.ceil(n / block_size)

    quantised = np.zeros(n, dtype=np.uint8)
    absmax = np.zeros(n_blocks, dtype=np.float32)

    levels = NF4_LEVELS

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = weights[start:end]

        amax = float(np.max(np.abs(block)))
        absmax[b] = amax

        if amax == 0:
            quantised[start:end] = 7  # nearest to 0 in NF4
            continue

        norm_block = block / amax  # normalise to [-1, 1]

        # Nearest neighbour lookup into NF4 levels
        for i, w in enumerate(norm_block):
            distances = np.abs(levels - w)
            quantised[start + i] = int(np.argmin(distances))

    return quantised, absmax


def dequantise_nf4(quantised: np.ndarray, absmax: np.ndarray,
                   block_size: int = 64) -> np.ndarray:
    """
    Reconstruct fp32 weights from NF4 quantised indices and absmax constants.
    """
    levels = NF4_LEVELS
    n = quantised.shape[0]
    weights = np.zeros(n, dtype=np.float32)

    for b in range(len(absmax)):
        start = b * block_size
        end = min(start + block_size, n)
        block_q = quantised[start:end]
        weights[start:end] = levels[block_q] * absmax[b]

    return weights


def quantisation_error(original: np.ndarray, reconstructed: np.ndarray) -> Dict:
    diff = original - reconstructed
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "relative_error": float(np.mean(np.abs(diff)) / (np.mean(np.abs(original)) + 1e-10)),
    }


# ---------------------------------------------------------------------------
# INT4 uniform quantisation (for comparison)
# ---------------------------------------------------------------------------

def quantise_int4(weights: np.ndarray,
                  block_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform 4-bit quantisation: 16 equispaced levels in [-1, 1]."""
    n = weights.shape[0]
    n_blocks = math.ceil(n / block_size)
    quantised = np.zeros(n, dtype=np.uint8)
    absmax = np.zeros(n_blocks, dtype=np.float32)
    levels = np.linspace(-1.0, 1.0, 16)

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = weights[start:end]
        amax = float(np.max(np.abs(block)))
        absmax[b] = amax
        if amax == 0:
            continue
        norm_block = block / amax
        for i, w in enumerate(norm_block):
            quantised[start + i] = int(np.argmin(np.abs(levels - w)))

    return quantised, absmax


def dequantise_int4(quantised: np.ndarray, absmax: np.ndarray,
                    block_size: int = 64) -> np.ndarray:
    levels = np.linspace(-1.0, 1.0, 16)
    n = quantised.shape[0]
    weights = np.zeros(n, dtype=np.float32)
    for b in range(len(absmax)):
        start = b * block_size
        end = min(start + block_size, n)
        weights[start:end] = levels[quantised[start:end]] * absmax[b]
    return weights


# ---------------------------------------------------------------------------
# Double quantisation
# ---------------------------------------------------------------------------

def double_quantise(absmax: np.ndarray,
                    inner_block_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantise the absmax constants themselves with 8-bit uniform quantisation.
    Returns (quantised_absmax, super_absmax).
    """
    n = absmax.shape[0]
    n_inner = math.ceil(n / inner_block_size)
    q_absmax = np.zeros(n, dtype=np.uint8)
    super_absmax = np.zeros(n_inner, dtype=np.float32)

    for b in range(n_inner):
        start = b * inner_block_size
        end = min(start + inner_block_size, n)
        block = absmax[start:end]
        amax = float(np.max(np.abs(block)))
        super_absmax[b] = amax
        if amax == 0:
            continue
        norm = block / amax
        q_absmax[start:end] = np.round(norm * 127).astype(np.uint8)

    return q_absmax, super_absmax


def reconstruct_absmax(q_absmax: np.ndarray, super_absmax: np.ndarray,
                       inner_block_size: int = 256) -> np.ndarray:
    n = q_absmax.shape[0]
    n_inner = len(super_absmax)
    absmax = np.zeros(n, dtype=np.float32)
    for b in range(n_inner):
        start = b * inner_block_size
        end = min(start + inner_block_size, n)
        absmax[start:end] = (q_absmax[start:end].astype(float) / 127.0) * super_absmax[b]
    return absmax


# ---------------------------------------------------------------------------
# Memory footprint calculation
# ---------------------------------------------------------------------------

def memory_bytes(n_params: int, dtype_bits: int) -> float:
    """Return memory in GB."""
    return n_params * dtype_bits / 8 / 1e9


def memory_comparison_table(n_params: int = 7_000_000_000):
    model_name = f"LLaMA-2-{n_params//1_000_000_000}B"
    print(f"\n  Model: {model_name} ({n_params:,} parameters)")

    # Adam stores: model (fp32), gradients (fp32), m (fp32), v (fp32) = 4× model size
    # LoRA trainable params ~ 0.06% of 7B ≈ 4.2M
    lora_params = int(n_params * 0.0006)
    r = 8
    # QLoRA: base in NF4 (4-bit), adapters in BF16 (16-bit)
    # Double quant saves ~0.37 GB on constants

    print(f"\n  {'Setup':>30} {'Model':>10} {'Grads':>10} "
          f"{'Optimizer':>12} {'Total':>10}")
    print(f"  {'-'*78}")

    setups = [
        ("Full FT (FP32)",       n_params, 32, 32, 2),
        ("Full FT (BF16/FP32)",  n_params, 16, 32, 2),
        ("LoRA (BF16)",          n_params, 16, 16, 2, lora_params),
        ("QLoRA (NF4+BF16)",     n_params, 4,  16, 2, lora_params),
    ]

    for row in setups:
        name = row[0]
        n = row[1]
        model_bits = row[2]
        grad_bits = row[3]
        opt_mult = row[4]
        trainable = row[5] if len(row) > 5 else n

        model_gb = memory_bytes(n, model_bits)
        grad_gb = memory_bytes(trainable, grad_bits)
        opt_gb = memory_bytes(trainable, 32) * opt_mult  # Adam: 2× fp32 moments
        total_gb = model_gb + grad_gb + opt_gb

        print(f"  {name:>30} {model_gb:>8.2f}G {grad_gb:>8.2f}G "
              f"{opt_gb:>10.2f}G {total_gb:>8.2f}G")

    print(f"\n  Note: QLoRA also saves ~0.37 GB via double quantisation on absmax constants")
    print(f"  Paged Adam offloads optimizer states to CPU RAM during memory spikes")


# ---------------------------------------------------------------------------
# NF4 level visualisation (terminal)
# ---------------------------------------------------------------------------

def visualise_nf4_vs_int4():
    nf4 = NF4_LEVELS
    int4 = np.linspace(-1.0, 1.0, 16)

    print(f"\n  NF4 levels (quantiles of N(0,1), renorm to [-1,1]):")
    print(f"  idx  {'NF4':>8}  {'INT4':>8}  {'Spacing (NF4)':>15}  {'Spacing (INT4)':>15}")
    print(f"  {'-'*60}")
    for i in range(16):
        spacing_nf4 = f"{nf4[i]-nf4[i-1]:+.4f}" if i > 0 else "   —   "
        spacing_int4 = f"{int4[i]-int4[i-1]:+.4f}" if i > 0 else "   —   "
        print(f"  {i:>3}  {nf4[i]:>8.4f}  {int4[i]:>8.4f}  "
              f"{spacing_nf4:>15}  {spacing_int4:>15}")

    print(f"\n  Observation: NF4 has denser levels near 0 (mode of N(0,1))")
    print(f"  → smaller quantisation error for typical neural network weights")


# ---------------------------------------------------------------------------
# Double quantisation memory savings
# ---------------------------------------------------------------------------

def double_quant_savings(n_params: int = 7_000_000_000, block_size: int = 64):
    n_blocks = n_params / block_size
    # Without double quant: one fp32 absmax per block
    size_without = n_blocks * 32 / 8 / 1e9   # GB
    bits_per_param = 32 / block_size

    # With double quant: 8-bit absmax + one fp32 per super-block of 256
    size_with = n_blocks * 8 / 8 / 1e9 + (n_blocks / 256) * 32 / 8 / 1e9
    bits_per_param_dq = (8 / block_size) + (32 / (block_size * 256))

    print(f"\n  For {n_params:,} parameters, block_size={block_size}:")
    print(f"  Absmax constants (fp32, 1 per block): {size_without:.3f} GB "
          f"= {bits_per_param:.4f} bits/param")
    print(f"  After double quantisation (8-bit):    {size_with:.3f} GB "
          f"= {bits_per_param_dq:.4f} bits/param")
    print(f"  Savings: {size_without - size_with:.3f} GB")


# ---------------------------------------------------------------------------
# BitsAndBytes graceful skip
# ---------------------------------------------------------------------------

def show_qlora_config():
    print("\n  Real QLoRA config (requires: pip install bitsandbytes peft transformers):")
    print("""
  import torch
  from transformers import BitsAndBytesConfig, AutoModelForCausalLM
  from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",          # NF4 data type
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,     # double quantisation
  )

  model = AutoModelForCausalLM.from_pretrained(
      "meta-llama/Llama-2-7b-hf",
      quantization_config=bnb_config,
      device_map="auto",
  )

  model = prepare_model_for_kbit_training(model)

  lora_config = LoraConfig(
      r=8,
      lora_alpha=16,
      target_modules=["q_proj", "v_proj"],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
  )

  model = get_peft_model(model, lora_config)

  # Use paged Adam to avoid OOM during long sequences
  from transformers import TrainingArguments
  training_args = TrainingArguments(
      optim="paged_adamw_32bit",
      ...
  )
    """)

    try:
        import bitsandbytes  # noqa: F401
        print("  bitsandbytes installed — real QLoRA would run above.")
    except ImportError:
        print("  bitsandbytes not installed — numpy simulation is equivalent for understanding.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    section("NF4 QUANTISATION LEVELS — VISUALISATION")
    visualise_nf4_vs_int4()

    section("NF4 vs INT4 QUANTISATION ERROR")
    # Simulate normally distributed weights (realistic for LLMs)
    weights = RNG.standard_normal(4096).astype(np.float32) * 0.02

    q_nf4, absmax_nf4 = quantise_nf4(weights, block_size=64)
    r_nf4 = dequantise_nf4(q_nf4, absmax_nf4, block_size=64)
    err_nf4 = quantisation_error(weights, r_nf4)

    q_int4, absmax_int4 = quantise_int4(weights, block_size=64)
    r_int4 = dequantise_int4(q_int4, absmax_int4, block_size=64)
    err_int4 = quantisation_error(weights, r_int4)

    print(f"\n  Weight distribution: N(0, 0.02^2),  n={len(weights):,},  block_size=64")
    print(f"\n  {'Metric':>20}  {'NF4':>12}  {'INT4':>12}  {'NF4 better':>12}")
    print(f"  {'-'*60}")
    for key in ("max_abs_error", "rmse", "relative_error"):
        nf4_v = err_nf4[key]
        int4_v = err_int4[key]
        better = "✓" if nf4_v < int4_v else "✗"
        print(f"  {key:>20}  {nf4_v:>12.6f}  {int4_v:>12.6f}  {better:>12}")

    section("DOUBLE QUANTISATION — MEMORY SAVINGS")
    double_quant_savings(n_params=7_000_000_000, block_size=64)

    section("DOUBLE QUANTISATION — ROUND TRIP ERROR")
    absmax = absmax_nf4.astype(np.float32)
    q_abs, super_abs = double_quantise(absmax, inner_block_size=256)
    absmax_recon = reconstruct_absmax(q_abs, super_abs, inner_block_size=256)
    abs_err = quantisation_error(absmax, absmax_recon)
    print(f"\n  absmax constants: {len(absmax)} values")
    print(f"  Double quant error on absmax: rmse={abs_err['rmse']:.6f}")
    print(f"  Relative error: {abs_err['relative_error']:.4f}")

    # Full round-trip with double quantisation
    r_dq = dequantise_nf4(q_nf4, absmax_recon, block_size=64)
    err_dq = quantisation_error(weights, r_dq)
    print(f"\n  Full round-trip (NF4 + double quant):")
    print(f"    rmse: {err_dq['rmse']:.6f}  (vs NF4-only: {err_nf4['rmse']:.6f})")
    print(f"    overhead: {err_dq['rmse'] - err_nf4['rmse']:.2e} additional error")

    section("MEMORY COMPARISON TABLE")
    memory_comparison_table(n_params=7_000_000_000)

    section("NF4 LEVELS — COMPACT VIEW")
    print(f"\n  16 NF4 quantisation levels (symmetric):")
    print(f"  {np.round(NF4_LEVELS, 4).tolist()}")
    print(f"\n  Property: equal probability mass per bin under N(0,1)")
    print(f"  → minimises expected quantisation error for Gaussian weights")

    section("BITS PER WEIGHT BREAKDOWN — QLORA")
    block_size = 64
    n = 7_000_000_000
    n_blocks = n / block_size
    bits_nf4 = 4.0
    bits_absmax_fp32 = 32 / block_size
    bits_absmax_dq = 8 / block_size + 32 / (block_size * 256)
    bits_lora_adapters = 16 * int(n * 0.0006) / n
    print(f"\n  Base weights (NF4):           {bits_nf4:.4f} bits/param")
    print(f"  Absmax constants (fp32):      {bits_absmax_fp32:.4f} bits/param")
    print(f"  Absmax constants (dq 8-bit):  {bits_absmax_dq:.4f} bits/param")
    print(f"  LoRA adapters (BF16, 0.06%):  {bits_lora_adapters:.4f} bits/param")
    total = bits_nf4 + bits_absmax_dq + bits_lora_adapters
    print(f"  ─────────────────────────────")
    print(f"  Total effective:              {total:.4f} bits/param")
    print(f"  vs FP16 (16 bits/param):      {16.0:.4f} bits/param")
    print(f"  Compression ratio: {16/total:.1f}×")

    section("QLORA PIPELINE — REAL LIBRARY (GRACEFUL SKIP)")
    show_qlora_config()

    section("PAGED OPTIMISER — CONCEPT")
    print("""
  Adam stores 2 momentum tensors (m, v) for each trainable parameter.
  For 7B × 0.06% trainable = 4.2M LoRA params:
    Adam state (fp32): 4.2M × 2 × 4 bytes = 33.6 MB  (manageable)

  For full fine-tuning of 7B:
    Adam state (fp32): 7B × 2 × 4 bytes = 56 GB  (exceeds VRAM)

  Paged Adam uses NVIDIA unified memory (UVM):
    - Optimizer state lives in CPU RAM (pinned memory)
    - Pages are DMA'd to GPU only when needed for update
    - Prevents OOM crashes on long sequences that cause memory spikes
    - Small throughput penalty (~5-10%) vs pure GPU Adam
    """)


if __name__ == "__main__":
    main()
