"""
LoRA adapter merging: W_merged = W0 + (alpha/r)*BA, PEFT merge_and_unload
simulation, adapter inspection, before/after weight comparison.
pip install numpy  (peft, transformers optional)
"""

import numpy as np
import math
import json
from typing import Dict, List, Optional, Any

RNG = np.random.default_rng(42)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# LoRA adapter state (simulates what PEFT saves to disk)
# ---------------------------------------------------------------------------

class LoRAAdapterState:
    """
    Represents the saved state of a LoRA adapter:
    - config (r, alpha, target_modules, dropout, bias, task_type)
    - weight matrices {layer_name: {"A": ..., "B": ...}}
    """

    def __init__(self, r: int = 8, alpha: float = 16.0,
                 target_modules: Optional[List[str]] = None,
                 dropout: float = 0.05, bias: str = "none",
                 task_type: str = "CAUSAL_LM",
                 base_model: str = "meta-llama/Llama-2-7b-hf"):
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.dropout = dropout
        self.bias = bias
        self.task_type = task_type
        self.base_model = base_model
        self.weights: Dict[str, Dict[str, np.ndarray]] = {}

    def add_layer(self, name: str, m: int, n: int):
        """Add a LoRA-adapted layer with random weights (simulates trained adapter)."""
        A = RNG.standard_normal((self.r, n)).astype(np.float32) * 0.01
        B = RNG.standard_normal((m, self.r)).astype(np.float32) * 0.01
        self.weights[name] = {"A": A, "B": B, "m": m, "n": n}

    def n_adapter_params(self) -> int:
        total = 0
        for v in self.weights.values():
            total += v["A"].size + v["B"].size
        return total

    def config_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "base_model_name_or_path": self.base_model,
        }


# ---------------------------------------------------------------------------
# Base model (simulates frozen pre-trained weights)
# ---------------------------------------------------------------------------

class BaseModel:
    """Simulates the frozen weight matrices of a base LLM."""

    def __init__(self, layer_dims: Dict[str, tuple]):
        self.weights: Dict[str, np.ndarray] = {}
        for name, (m, n) in layer_dims.items():
            # Simulate pre-trained weights: small Gaussian
            self.weights[name] = RNG.standard_normal((m, n)).astype(np.float32) * 0.02

    def forward(self, x: np.ndarray, layer: str) -> np.ndarray:
        return x @ self.weights[layer].T


# ---------------------------------------------------------------------------
# Merger
# ---------------------------------------------------------------------------

def merge_adapter(base_weight: np.ndarray,
                  A: np.ndarray, B: np.ndarray,
                  scale: float) -> np.ndarray:
    """
    W_merged = W0 + scale * B @ A
    scale = alpha / r
    """
    delta = scale * (B @ A)
    return base_weight + delta


def merge_all_layers(base: BaseModel,
                     adapter: LoRAAdapterState) -> Dict[str, np.ndarray]:
    """
    Merge all LoRA adapters into base weights.
    Returns {layer_name: merged_weight_matrix}.
    """
    merged = {}
    for name, W0 in base.weights.items():
        if name in adapter.weights:
            A = adapter.weights[name]["A"]
            B = adapter.weights[name]["B"]
            merged[name] = merge_adapter(W0, A, B, adapter.scale)
        else:
            merged[name] = W0.copy()
    return merged


def verify_merge_equivalence(base: BaseModel, adapter: LoRAAdapterState,
                              merged: Dict[str, np.ndarray],
                              n_test: int = 16) -> Dict[str, float]:
    """
    Verify that:
    merged_weight @ x == W0 @ x + scale * B @ A @ x
    for all adapted layers.
    """
    max_diffs = {}
    for name, W0 in base.weights.items():
        if name not in adapter.weights:
            continue
        A = adapter.weights[name]["A"]
        B = adapter.weights[name]["B"]
        _, n = A.shape

        X = RNG.standard_normal((n_test, n)).astype(np.float32)

        # PEFT path
        out_peft = (X @ W0.T) + adapter.scale * (X @ A.T @ B.T)
        # Merged path
        out_merged = X @ merged[name].T

        max_diff = float(np.max(np.abs(out_peft - out_merged)))
        max_diffs[name] = max_diff

    return max_diffs


# ---------------------------------------------------------------------------
# Adapter inspection
# ---------------------------------------------------------------------------

def inspect_adapter(adapter: LoRAAdapterState) -> None:
    print(f"\n  Adapter config:")
    config = adapter.config_dict()
    for k, v in config.items():
        print(f"    {k}: {v}")

    print(f"\n  Adapter layers:")
    total_params = 0
    print(f"  {'Layer':>15} {'Shape A':>15} {'Shape B':>15} {'Params':>10}")
    print(f"  {'-'*60}")
    for name, w in adapter.weights.items():
        A_shape = w["A"].shape
        B_shape = w["B"].shape
        n_params = w["A"].size + w["B"].size
        total_params += n_params
        print(f"  {name:>15} {str(A_shape):>15} {str(B_shape):>15} {n_params:>10,}")
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':>15} {'':>15} {'':>15} {total_params:>10,}")


# ---------------------------------------------------------------------------
# Delta W statistics
# ---------------------------------------------------------------------------

def delta_statistics(base: BaseModel, adapter: LoRAAdapterState,
                     merged: Dict[str, np.ndarray]) -> None:
    print(f"\n  {'Layer':>15} {'||W0||_F':>12} {'||delta||_F':>14} "
          f"{'delta/W0':>10} {'rank':>6}")
    print(f"  {'-'*65}")
    for name in adapter.weights:
        W0 = base.weights[name]
        W_m = merged[name]
        delta = W_m - W0
        norm_W0 = float(np.linalg.norm(W0, "fro"))
        norm_delta = float(np.linalg.norm(delta, "fro"))
        ratio = norm_delta / norm_W0 if norm_W0 > 0 else float("inf")
        # Effective rank of delta (should equal adapter.r)
        svd_s = np.linalg.svd(delta, compute_uv=False)
        eff_rank = int(np.sum(svd_s > 1e-6 * svd_s[0]))
        print(f"  {name:>15} {norm_W0:>12.4f} {norm_delta:>14.6f} "
              f"{ratio:>10.6f} {eff_rank:>6}")


# ---------------------------------------------------------------------------
# Simulated Hub push
# ---------------------------------------------------------------------------

def simulate_hub_push(adapter: LoRAAdapterState, repo_id: str):
    config_json = json.dumps(adapter.config_dict(), indent=2)
    adapter_size_mb = sum(
        (w["A"].nbytes + w["B"].nbytes) for w in adapter.weights.values()
    ) / 1e6

    print(f"\n  Pushing to Hub: {repo_id}")
    print(f"\n  Files to upload:")
    print(f"    adapter_config.json  ({len(config_json)} bytes)")
    print(f"    adapter_model.bin    ({adapter_size_mb:.2f} MB)")
    print(f"\n  adapter_config.json:")
    for line in config_json.split("\n"):
        print(f"    {line}")

    print(f"\n  Loading from Hub:")
    print(f"    from peft import PeftModel, PeftConfig")
    print(f"    config = PeftConfig.from_pretrained('{repo_id}')")
    print(f"    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)")
    print(f"    model = PeftModel.from_pretrained(model, '{repo_id}')")


# ---------------------------------------------------------------------------
# PEFT graceful skip
# ---------------------------------------------------------------------------

def show_peft_merge():
    print("""
  Real PEFT merge (requires: pip install peft transformers):

  from peft import PeftModel

  # Load trained PEFT model
  model = PeftModel.from_pretrained(base_model, "./lora-adapter")

  # Merge adapters into base weights (in-place)
  # Returns a plain nn.Module — no PEFT overhead at inference
  merged = model.merge_and_unload()

  # Save merged model (full weights + config + tokenizer)
  merged.save_pretrained("./merged-model")
  tokenizer.save_pretrained("./merged-model")

  # Push to Hub (merged — no PEFT needed to load)
  merged.push_to_hub("username/my-finetuned-model")

  # OR push adapter only (base model loaded separately at inference)
  model.push_to_hub("username/my-lora-adapter")
  # Load later:
  # PeftModel.from_pretrained(base, "username/my-lora-adapter")
    """)

    try:
        import peft  # noqa: F401
        print("  peft installed — real merge_and_unload would run above.")
    except ImportError:
        print("  peft not installed — numpy simulation above is equivalent.")


# ---------------------------------------------------------------------------
# Multi-adapter (multiple LoRA checkpoints) composition
# ---------------------------------------------------------------------------

def multi_adapter_composition(base: BaseModel,
                               adapters: List[LoRAAdapterState],
                               weights_list: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
    """
    Compose multiple LoRA adapters via weighted sum:
    W_final = W0 + sum_i (w_i * scale_i * B_i @ A_i)
    """
    if weights_list is None:
        weights_list = [1.0 / len(adapters)] * len(adapters)

    merged = {name: W0.copy() for name, W0 in base.weights.items()}

    for adapter, w in zip(adapters, weights_list):
        for name, lora_w in adapter.weights.items():
            if name in merged:
                delta = adapter.scale * (lora_w["B"] @ lora_w["A"])
                merged[name] += w * delta

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LAYER_DIMS = {
    "q_proj": (64, 64),
    "k_proj": (64, 64),
    "v_proj": (64, 64),
    "o_proj": (64, 64),
}


def main():
    # Build base model and trained adapter
    base = BaseModel(LAYER_DIMS)
    adapter = LoRAAdapterState(r=8, alpha=16.0,
                               target_modules=["q_proj", "v_proj"])
    for name, (m, n) in LAYER_DIMS.items():
        if name in adapter.target_modules:
            adapter.add_layer(name, m, n)

    section("ADAPTER INSPECTION")
    inspect_adapter(adapter)

    section("MERGE MATH: W_merged = W0 + (alpha/r) * B @ A")
    merged = merge_all_layers(base, adapter)
    print(f"\n  scale = alpha/r = {adapter.alpha}/{adapter.r} = {adapter.scale:.2f}")
    for name in adapter.weights:
        A = adapter.weights[name]["A"]
        B = adapter.weights[name]["B"]
        delta = adapter.scale * (B @ A)
        print(f"\n  Layer: {name}")
        print(f"    A shape: {A.shape},  B shape: {B.shape}")
        print(f"    delta shape: {delta.shape}")
        print(f"    ||delta||_F: {np.linalg.norm(delta, 'fro'):.6f}")
        print(f"    ||W0||_F:    {np.linalg.norm(base.weights[name], 'fro'):.6f}")

    section("MERGE EQUIVALENCE VERIFICATION")
    diffs = verify_merge_equivalence(base, adapter, merged, n_test=32)
    print(f"\n  Testing: (W0 @ x + scale*B@A@x) == (W_merged @ x)")
    for name, diff in diffs.items():
        ok = "✓" if diff < 1e-5 else "✗"
        print(f"  {name}: max_diff={diff:.2e}  {ok}")

    section("DELTA W STATISTICS")
    delta_statistics(base, adapter, merged)

    section("SINGULAR VALUE DECOMPOSITION OF DELTA W")
    print(f"\n  Verifying rank(delta) = r = {adapter.r}")
    print(f"\n  {'Layer':>12}  Top {adapter.r + 2} singular values of delta_W:")
    print(f"  {'-'*55}")
    for name in adapter.weights:
        delta = merged[name] - base.weights[name]
        svd_s = np.linalg.svd(delta, compute_uv=False)
        top = svd_s[:adapter.r + 2]
        vals = "  ".join(f"{s:.4f}" for s in top)
        print(f"  {name:>12}  [{vals}]")
    print(f"\n  Observation: only first {adapter.r} singular values are non-negligible")
    print(f"  → delta_W lies exactly in a rank-{adapter.r} subspace")

    section("MULTI-ADAPTER COMPOSITION")
    # Two adapters trained on different tasks
    adapter2 = LoRAAdapterState(r=4, alpha=8.0,
                                target_modules=["q_proj", "v_proj"])
    for name, (m, n) in LAYER_DIMS.items():
        if name in adapter2.target_modules:
            adapter2.add_layer(name, m, n)

    merged_equal = multi_adapter_composition(base, [adapter, adapter2], [0.5, 0.5])
    merged_skewed = multi_adapter_composition(base, [adapter, adapter2], [0.8, 0.2])

    x_test = RNG.standard_normal(64).astype(np.float32)
    out_equal = x_test @ merged_equal["q_proj"].T
    out_skewed = x_test @ merged_skewed["q_proj"].T
    print(f"\n  Equal blend (0.5/0.5) vs skewed (0.8/0.2) q_proj output:")
    print(f"  Max diff: {np.max(np.abs(out_equal - out_skewed)):.4f}")
    print(f"  (Demonstrates task interpolation via adapter weights)")

    section("SIMULATED HUB PUSH — ADAPTER ONLY")
    simulate_hub_push(adapter, "harshit/llama2-7b-lora-qa")

    section("PEFT MERGE_AND_UNLOAD — REAL LIBRARY")
    show_peft_merge()

    section("INFERENCE OVERHEAD COMPARISON")
    print(f"""
  +-------------------------+--------------------+--------------------+
  | Setup                   | Inference cost     | Notes              |
  +-------------------------+--------------------+--------------------+
  | Base model              | W0 @ x             | Baseline           |
  | PEFT (adapter loaded)   | W0 @ x + scale*BAx | +r(m+n) FLOPs/layer|
  | Merged (merge_unload)   | W_merged @ x       | Identical to base  |
  | Quantised (NF4)         | dequant → W0 @ x  | ~2× slower forward |
  +-------------------------+--------------------+--------------------+

  Merged inference:  0 overhead — adapter absorbed into W0
  PEFT inference:    2× matrix multiplies per adapted layer (small r → fast)
    """)

    section("SUMMARY — MERGE PIPELINE")
    print(f"""
  1. Load base model (fp16 or nf4)
  2. Load LoRA adapter (adapter_config.json + adapter_model.bin)
  3. Merge: W_merged = W0 + (alpha/r) * B @ A   [for each adapted layer]
  4. Save merged weights (no PEFT dependency needed at inference)
  5. Optionally quantise merged weights for deployment

  Merge is:
    - Lossless: merged_output == PEFT_output to float precision
    - Irreversible: cannot recover W0 and B,A from W_merged alone
    - O(r(m+n)) memory for adapter, O(mn) for merge operation
    """)


if __name__ == "__main__":
    main()
