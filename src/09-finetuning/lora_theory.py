"""
LoRA rank decomposition from scratch: W' = W0 + BA, parameter count math,
forward pass with alpha/r scaling, initialisation verification, rank sweep.
pip install numpy
"""

import numpy as np
import math

RNG = np.random.default_rng(42)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# LoRA layer — numpy implementation
# ---------------------------------------------------------------------------

class LoRALayer:
    """
    Simulates a single linear layer with a LoRA adapter.
    W_frozen: (m, n) — pre-trained, frozen
    B: (m, r) — initialised to zero
    A: (r, n) — initialised from N(0, sigma^2)
    Forward: h = W0 @ x + (alpha/r) * B @ A @ x
    """

    def __init__(self, m: int, n: int, r: int, alpha: float = 16.0,
                 sigma: float = 0.02):
        self.m = m
        self.n = n
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        # Frozen pre-trained weights (simulated as random)
        self.W0 = RNG.standard_normal((m, n)) * 0.02

        # LoRA adapter: B=0, A~N(0,sigma^2)
        self.A = RNG.standard_normal((r, n)) * sigma
        self.B = np.zeros((m, r))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (n,) or (batch, n)"""
        base = x @ self.W0.T         # (..., m)
        delta = x @ self.A.T @ self.B.T * self.scale  # (..., m)
        return base + delta

    def lora_params(self) -> int:
        return self.r * (self.m + self.n)

    def full_params(self) -> int:
        return self.m * self.n

    def reduction_factor(self) -> float:
        return self.full_params() / self.lora_params()

    def merged_weight(self) -> np.ndarray:
        """W_merged = W0 + (alpha/r) * B @ A"""
        return self.W0 + self.scale * (self.B @ self.A)

    def verify_init_delta_zero(self) -> bool:
        """At init: B=0 so delta=0, merged weight == W0"""
        diff = np.max(np.abs(self.merged_weight() - self.W0))
        return diff < 1e-12


# ---------------------------------------------------------------------------
# Gradient computation (manual backprop through LoRA)
# ---------------------------------------------------------------------------

def lora_gradients(layer: LoRALayer, x: np.ndarray,
                   upstream_grad: np.ndarray):
    """
    Compute dL/dA and dL/dB given upstream gradient dL/dh.
    h = W0 x + scale * B A x
    dL/dA = scale * B^T dL/dh * x^T
    dL/dB = scale * dL/dh * (A x)^T
    """
    Ax = layer.A @ x          # (r,)
    scale = layer.scale
    # upstream_grad: (m,)
    dB = scale * np.outer(upstream_grad, Ax)   # (m, r)
    dA = scale * layer.B.T @ np.outer(upstream_grad, x)  # (r, n)
    return dA, dB


# ---------------------------------------------------------------------------
# Rank sensitivity sweep
# ---------------------------------------------------------------------------

def rank_sweep(m: int = 4096, n: int = 4096, ranks=None):
    if ranks is None:
        ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    full = m * n
    print(f"\n  Layer: ({m} × {n})   full params = {full:,}")
    print(f"\n  {'r':>6} {'LoRA params':>14} {'Reduction':>12} {'% of full':>12}")
    print(f"  {'-'*50}")
    for r in ranks:
        lora = r * (m + n)
        red = full / lora
        pct = 100.0 * lora / full
        print(f"  {r:>6} {lora:>14,} {red:>11.1f}× {pct:>11.4f}%")


# ---------------------------------------------------------------------------
# Layer-wise parameter savings for LLaMA-2-7B attention projections
# ---------------------------------------------------------------------------

LLAMA2_7B_LAYERS = [
    # (name, m, n)
    ("q_proj", 4096, 4096),
    ("k_proj", 4096, 4096),
    ("v_proj", 4096, 4096),
    ("o_proj", 4096, 4096),
    ("gate_proj", 4096, 11008),
    ("up_proj", 4096, 11008),
    ("down_proj", 11008, 4096),
]
N_TRANSFORMER_LAYERS = 32


def param_savings_table(r: int = 8, alpha: float = 16.0,
                        target_modules=("q_proj", "v_proj")):
    print(f"\n  LoRA config: r={r}, alpha={alpha}, "
          f"target_modules={list(target_modules)}")
    print(f"  Model: LLaMA-2-7B ({N_TRANSFORMER_LAYERS} transformer layers)")
    print(f"\n  {'Layer':>12} {'Full (per)':>12} {'LoRA (per)':>12} "
          f"{'Reduction':>12} {'Total LoRA':>12}")
    print(f"  {'-'*64}")

    total_lora = 0
    total_full = 0
    for name, m, n in LLAMA2_7B_LAYERS:
        full_per = m * n
        if name in target_modules:
            lora_per = r * (m + n)
            red = full_per / lora_per
            total_lora_layer = lora_per * N_TRANSFORMER_LAYERS
            tag = "✓"
        else:
            lora_per = 0
            red = float('inf')
            total_lora_layer = 0
            tag = "—"
        total_lora += total_lora_layer
        total_full += full_per * N_TRANSFORMER_LAYERS
        print(f"  {name:>12} {full_per:>12,} {lora_per:>12,} "
              f"{'∞' if lora_per == 0 else f'{red:.0f}×':>12} "
              f"{total_lora_layer:>12,} {tag}")

    print(f"  {'-'*64}")
    print(f"  {'TOTAL':>12} {total_full:>12,} {total_lora:>12,} "
          f"{total_full/max(total_lora,1):>11.1f}×")
    print(f"\n  Trainable%: {100*total_lora/total_full:.4f}%")


# ---------------------------------------------------------------------------
# Scaling factor analysis: alpha/r
# ---------------------------------------------------------------------------

def scaling_analysis():
    r_vals = [4, 8, 16, 32]
    alpha_vals = [8, 16, 32]

    print(f"\n  alpha/r scaling table (affects effective LoRA LR):")
    print(f"\n  {'':>8}", end="")
    for a in alpha_vals:
        print(f"  alpha={a:>2}", end="")
    print()
    print(f"  {'-'*40}")
    for r in r_vals:
        print(f"  r={r:>4} |", end="")
        for a in alpha_vals:
            print(f"   {a/r:>6.3f}  ", end="")
        print()
    print("\n  Convention: keep alpha = 2r so scale = 2.0 always.")


# ---------------------------------------------------------------------------
# Simulate one training step through LoRA
# ---------------------------------------------------------------------------

def simulate_training_step(m=64, n=64, r=4, alpha=8.0, lr=1e-3):
    layer = LoRALayer(m, n, r, alpha)
    x = RNG.standard_normal(n)
    target = RNG.standard_normal(m)

    # Forward
    h = layer.forward(x)
    loss = 0.5 * np.sum((h - target) ** 2)
    upstream = h - target  # dL/dh for MSE

    # Backward
    dA, dB = lora_gradients(layer, x, upstream)

    # Before update
    delta_before = np.max(np.abs(layer.B @ layer.A))

    # Gradient step (SGD)
    layer.A -= lr * dA
    layer.B -= lr * dB

    # After update
    delta_after = np.max(np.abs(layer.B @ layer.A))
    new_h = layer.forward(x)
    new_loss = 0.5 * np.sum((new_h - target) ** 2)

    return {
        "loss_before": loss,
        "loss_after": new_loss,
        "delta_W_before": delta_before,
        "delta_W_after": delta_after,
        "loss_reduced": new_loss < loss,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    section("LORA INITIALISATION VERIFICATION")
    layer = LoRALayer(m=256, n=512, r=8, alpha=16.0)
    ok = layer.verify_init_delta_zero()
    print(f"\n  Layer: (256×512), r=8, alpha=16")
    print(f"  B initialised to zero: {np.allclose(layer.B, 0)}")
    print(f"  delta_W = B@A at init: max_abs = {np.max(np.abs(layer.B @ layer.A)):.2e}")
    print(f"  merged_weight == W0 at init: {ok}")
    print(f"  LoRA params: {layer.lora_params():,}")
    print(f"  Full params: {layer.full_params():,}")
    print(f"  Reduction: {layer.reduction_factor():.1f}×")

    section("RANK SENSITIVITY SWEEP — (4096 × 4096) LAYER")
    rank_sweep(m=4096, n=4096)

    section("LAYER-WISE PARAMETER SAVINGS — LLaMA-2-7B")
    param_savings_table(r=8, alpha=16, target_modules=("q_proj", "v_proj"))
    print()
    param_savings_table(r=8, alpha=16,
                        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))

    section("ALPHA/R SCALING FACTOR ANALYSIS")
    scaling_analysis()

    section("FORWARD PASS — BASE vs LORA")
    small = LoRALayer(m=4, n=6, r=2, alpha=4.0)
    x = RNG.standard_normal(6)
    base_out = x @ small.W0.T
    lora_out = small.forward(x)
    print(f"\n  x      = {x.round(3)}")
    print(f"  W0 x   = {base_out.round(4)}")
    print(f"  LoRA x = {lora_out.round(4)} (same at init since B=0)")
    print(f"  delta  = {(lora_out - base_out).round(4)}")

    section("ONE TRAINING STEP SIMULATION")
    result = simulate_training_step(m=64, n=64, r=4, alpha=8.0, lr=1e-3)
    print(f"\n  Loss before: {result['loss_before']:.6f}")
    print(f"  Loss after:  {result['loss_after']:.6f}")
    print(f"  Loss reduced: {result['loss_reduced']}")
    print(f"  |delta_W| before: {result['delta_W_before']:.2e}")
    print(f"  |delta_W| after:  {result['delta_W_after']:.2e}")

    section("ADAPTER MERGING — WEIGHT EQUIVALENCE")
    layer2 = LoRALayer(m=32, n=32, r=4, alpha=8.0)
    # Simulate a few gradient updates
    for _ in range(10):
        x_t = RNG.standard_normal(32)
        target_t = RNG.standard_normal(32)
        h_t = layer2.forward(x_t)
        upstream_t = h_t - target_t
        dA, dB = lora_gradients(layer2, x_t, upstream_t)
        layer2.A -= 1e-3 * dA
        layer2.B -= 1e-3 * dB

    W_merged = layer2.merged_weight()

    x_test = RNG.standard_normal(32)
    out_peft = layer2.forward(x_test)
    out_merged = x_test @ W_merged.T
    max_diff = np.max(np.abs(out_peft - out_merged))
    print(f"\n  After 10 gradient steps:")
    print(f"  PEFT output == Merged output: max_diff = {max_diff:.2e}")
    print(f"  Merge is lossless: {max_diff < 1e-10}")
    print(f"\n  Merged weight shape: {W_merged.shape}")
    print(f"  W0 max_abs: {np.max(np.abs(layer2.W0)):.4f}")
    print(f"  delta max_abs: {np.max(np.abs(layer2.scale * layer2.B @ layer2.A)):.4f}")

    section("MULTIPLE RANK COMPARISON — SAME LAYER")
    m, n = 512, 512
    print(f"\n  Layer: ({m}×{n}),  full params = {m*n:,}")
    print(f"\n  {'r':>4}  {'params':>10}  {'reduction':>12}  {'trainable%':>12}")
    print(f"  {'-'*45}")
    for r in [2, 4, 8, 16, 32, 64]:
        lp = r * (m + n)
        fp = m * n
        print(f"  {r:>4}  {lp:>10,}  {fp/lp:>11.1f}×  {100*lp/fp:>11.4f}%")


if __name__ == "__main__":
    main()
