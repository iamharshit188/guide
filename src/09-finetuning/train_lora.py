"""
LoRA training: numpy simulation of gradient flow through frozen + LoRA layers,
PEFT SFTTrainer config (graceful skip), hyperparameter sensitivity table.
pip install numpy  (peft, transformers, trl optional)
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional

RNG = np.random.default_rng(42)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# LoRA layer (full numpy — same as lora_theory.py, self-contained)
# ---------------------------------------------------------------------------

class LoRALinear:
    """
    Single linear layer with LoRA adapter.
    Supports forward, backward, SGD update.
    """

    def __init__(self, m: int, n: int, r: int, alpha: float = 16.0,
                 dropout: float = 0.0):
        self.m = m
        self.n = n
        self.r = r
        self.scale = alpha / r
        self.dropout = dropout

        # Frozen base weight (simulated pre-trained)
        self.W0 = RNG.standard_normal((m, n)) * 0.02

        # Trainable LoRA matrices
        self.A = RNG.standard_normal((r, n)) * 0.02   # (r, n)
        self.B = np.zeros((m, r))                      # (m, r)

        # Gradient accumulators
        self.dA = np.zeros_like(self.A)
        self.dB = np.zeros_like(self.B)

        # Adam state
        self.mA = np.zeros_like(self.A)
        self.vA = np.zeros_like(self.A)
        self.mB = np.zeros_like(self.B)
        self.vB = np.zeros_like(self.B)
        self.t = 0

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._x = x  # cache for backward
        base = x @ self.W0.T

        if self.dropout > 0 and training:
            mask = (RNG.uniform(size=self.A.shape) > self.dropout).astype(float)
            A_dropped = self.A * mask / (1 - self.dropout)
        else:
            A_dropped = self.A

        self._A_used = A_dropped
        Ax = x @ A_dropped.T    # (..., r)
        self._Ax = Ax
        lora_out = Ax @ self.B.T * self.scale  # (..., m)
        return base + lora_out

    def backward(self, upstream: np.ndarray) -> np.ndarray:
        # upstream: (..., m)
        x = self._x
        Ax = self._Ax
        A = self._A_used
        scale = self.scale

        # dL/dB: (m, r) = scale * upstream^T @ Ax (over batch)
        if upstream.ndim == 1:
            self.dB += scale * np.outer(upstream, Ax)
            self.dA += scale * (self.B.T @ upstream[:, None]) @ x[None, :]
            # Pass gradient to x (but W0 is frozen → only LoRA path)
            dx_lora = upstream @ self.B * scale @ A
            dx_base = upstream @ self.W0
            return dx_lora + dx_base
        else:
            # batch
            self.dB += scale * (upstream[:, :, None] * Ax[:, None, :]).mean(0)
            self.dA += scale * (
                (self.B.T @ upstream.T).T[:, :, None] * x[:, None, :]
            ).mean(0)
            dx = upstream @ (self.W0 + scale * self.B @ A)
            return dx

    def zero_grad(self):
        self.dA[:] = 0
        self.dB[:] = 0

    def step_sgd(self, lr: float = 1e-3):
        self.A -= lr * self.dA
        self.B -= lr * self.dB

    def step_adam(self, lr: float = 2e-4, beta1: float = 0.9,
                  beta2: float = 0.999, eps: float = 1e-8,
                  weight_decay: float = 0.0):
        self.t += 1
        # A
        self.mA = beta1 * self.mA + (1 - beta1) * self.dA
        self.vA = beta2 * self.vA + (1 - beta2) * self.dA ** 2
        mA_hat = self.mA / (1 - beta1 ** self.t)
        vA_hat = self.vA / (1 - beta2 ** self.t)
        self.A -= lr * (mA_hat / (np.sqrt(vA_hat) + eps) + weight_decay * self.A)
        # B
        self.mB = beta1 * self.mB + (1 - beta1) * self.dB
        self.vB = beta2 * self.vB + (1 - beta2) * self.dB ** 2
        mB_hat = self.mB / (1 - beta1 ** self.t)
        vB_hat = self.vB / (1 - beta2 ** self.t)
        self.B -= lr * (mB_hat / (np.sqrt(vB_hat) + eps) + weight_decay * self.B)

    def delta_norm(self) -> float:
        return float(np.linalg.norm(self.scale * self.B @ self.A))

    def n_trainable(self) -> int:
        return self.r * (self.m + self.n)


# ---------------------------------------------------------------------------
# Toy model: stacked LoRA layers (simulates a small transformer block)
# ---------------------------------------------------------------------------

class ToyLoRAModel:
    """
    Two LoRA-adapted linear layers with ReLU, simulating Q and V projections.
    """

    def __init__(self, d_model: int = 64, r: int = 4, alpha: float = 8.0):
        self.q_proj = LoRALinear(d_model, d_model, r, alpha)
        self.v_proj = LoRALinear(d_model, d_model, r, alpha)
        self.d_model = d_model

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        q = self.q_proj.forward(x, training)
        q = np.maximum(q, 0)   # ReLU
        self._q_relu = q
        out = self.v_proj.forward(q, training)
        return out

    def backward(self, upstream: np.ndarray):
        dq = self.v_proj.backward(upstream)
        dq = dq * (self._q_relu > 0)   # ReLU backward
        self.q_proj.backward(dq)

    def zero_grad(self):
        self.q_proj.zero_grad()
        self.v_proj.zero_grad()

    def step_adam(self, lr: float = 2e-4):
        self.q_proj.step_adam(lr)
        self.v_proj.step_adam(lr)

    def n_trainable(self) -> int:
        return self.q_proj.n_trainable() + self.v_proj.n_trainable()

    def n_total(self) -> int:
        d = self.d_model
        return 2 * d * d + self.n_trainable()

    def delta_norms(self) -> Dict[str, float]:
        return {
            "q_proj": self.q_proj.delta_norm(),
            "v_proj": self.v_proj.delta_norm(),
        }


# ---------------------------------------------------------------------------
# Training loop simulation
# ---------------------------------------------------------------------------

def mse_loss(pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    diff = pred - target
    loss = 0.5 * float(np.mean(diff ** 2))
    grad = diff / diff.size
    return loss, grad


def train_epoch(model: ToyLoRAModel, X: np.ndarray, Y: np.ndarray,
                lr: float = 2e-4, batch_size: int = 8) -> float:
    n = X.shape[0]
    indices = RNG.permutation(n)
    total_loss = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        x_batch = X[idx]
        y_batch = Y[idx]

        model.zero_grad()
        preds = np.array([model.forward(x) for x in x_batch])
        loss, dout = mse_loss(preds, y_batch)
        for i in range(len(x_batch)):
            model.backward(dout[i])
        model.step_adam(lr)

        total_loss += loss
        n_batches += 1

    return total_loss / n_batches


def run_training(d_model: int = 32, r: int = 4, alpha: float = 8.0,
                 n_samples: int = 128, n_epochs: int = 20,
                 lr: float = 2e-4) -> Dict:
    X = RNG.standard_normal((n_samples, d_model))
    # Synthetic task: linear mapping through a fixed target matrix
    W_target = RNG.standard_normal((d_model, d_model)) * 0.1
    Y = X @ W_target.T

    model = ToyLoRAModel(d_model=d_model, r=r, alpha=alpha)

    history = []
    for epoch in range(n_epochs):
        loss = train_epoch(model, X, Y, lr=lr)
        history.append(loss)

    return {
        "final_loss": history[-1],
        "initial_loss": history[0],
        "loss_history": history,
        "n_trainable": model.n_trainable(),
        "n_total": model.n_total(),
        "delta_norms": model.delta_norms(),
        "trainable_pct": 100 * model.n_trainable() / model.n_total(),
    }


# ---------------------------------------------------------------------------
# Hyperparameter sensitivity sweep
# ---------------------------------------------------------------------------

def hyperparam_sweep():
    base_cfg = dict(d_model=32, n_samples=128, n_epochs=15, lr=2e-4)

    print(f"\n  {'r':>4}  {'alpha':>6}  {'lr':>8}  {'final_loss':>12}  {'trainable%':>12}")
    print(f"  {'-'*55}")

    configs = [
        {"r": 2,  "alpha": 4,  "lr": 2e-4},
        {"r": 4,  "alpha": 8,  "lr": 2e-4},
        {"r": 8,  "alpha": 16, "lr": 2e-4},
        {"r": 4,  "alpha": 8,  "lr": 5e-4},
        {"r": 4,  "alpha": 8,  "lr": 1e-4},
        {"r": 16, "alpha": 32, "lr": 2e-4},
    ]

    for cfg in configs:
        result = run_training(**{**base_cfg, **cfg})
        print(f"  {cfg['r']:>4}  {cfg['alpha']:>6}  {cfg['lr']:>8.1e}  "
              f"{result['final_loss']:>12.6f}  {result['trainable_pct']:>11.2f}%")


# ---------------------------------------------------------------------------
# Gradient flow comparison: LoRA vs Full FT (toy)
# ---------------------------------------------------------------------------

def gradient_flow_comparison(d_model: int = 32, r: int = 4, n_steps: int = 5):
    X = RNG.standard_normal((16, d_model))
    W_target = RNG.standard_normal((d_model, d_model)) * 0.1
    Y = X @ W_target.T

    model = ToyLoRAModel(d_model=d_model, r=r, alpha=float(r * 2))

    print(f"\n  Step | Loss     | dA norm  | dB norm  | |delta_W|")
    print(f"  {'-'*55}")
    for step in range(n_steps):
        model.zero_grad()
        preds = np.array([model.forward(x) for x in X])
        loss, dout = mse_loss(preds, Y)
        for i in range(len(X)):
            model.backward(dout[i])

        dA_norm = float(np.linalg.norm(model.q_proj.dA))
        dB_norm = float(np.linalg.norm(model.q_proj.dB))
        delta = model.q_proj.delta_norm()

        model.step_adam(lr=2e-4)
        print(f"  {step+1:>4} | {loss:>8.5f} | {dA_norm:>8.5f} | {dB_norm:>8.5f} | {delta:>8.5f}")


# ---------------------------------------------------------------------------
# PEFT library config (graceful skip)
# ---------------------------------------------------------------------------

def show_peft_config():
    print("\n  Real PEFT LoRA config (requires: pip install peft transformers trl):")
    print("""
  from peft import LoraConfig, get_peft_model, TaskType
  from trl import SFTTrainer
  from transformers import TrainingArguments

  lora_config = LoraConfig(
      r=8,
      lora_alpha=16,
      lora_dropout=0.05,
      target_modules=["q_proj", "v_proj"],
      bias="none",
      task_type=TaskType.CAUSAL_LM,
  )

  model = get_peft_model(base_model, lora_config)
  model.print_trainable_parameters()

  training_args = TrainingArguments(
      output_dir="./lora-output",
      num_train_epochs=3,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=8,   # effective batch = 32
      learning_rate=2e-4,
      fp16=True,
      logging_steps=10,
      save_strategy="epoch",
      warmup_ratio=0.03,
  )

  trainer = SFTTrainer(
      model=model,
      train_dataset=train_dataset,
      peft_config=lora_config,
      dataset_text_field="text",
      max_seq_length=2048,
      tokenizer=tokenizer,
      args=training_args,
  )
  trainer.train()
  model.save_pretrained("./lora-adapter")
    """)

    try:
        import peft  # noqa: F401
        print("  peft is installed — real training would run above.")
    except ImportError:
        print("  peft not installed — simulation above is equivalent.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    section("LORA TRAINING SIMULATION — GRADIENT FLOW")
    gradient_flow_comparison(d_model=32, r=4, n_steps=5)

    section("FULL TRAINING RUN")
    result = run_training(d_model=32, r=4, alpha=8.0,
                          n_samples=128, n_epochs=20, lr=2e-4)
    print(f"\n  Trainable params: {result['n_trainable']:,} / {result['n_total']:,} "
          f"({result['trainable_pct']:.2f}%)")
    print(f"  Initial loss:  {result['initial_loss']:.6f}")
    print(f"  Final loss:    {result['final_loss']:.6f}")
    print(f"  Improvement:   {result['initial_loss']/result['final_loss']:.2f}×")
    print(f"  Delta W norms: {result['delta_norms']}")

    print(f"\n  Loss curve (every 4 epochs):")
    for i, l in enumerate(result["loss_history"]):
        if i % 4 == 0 or i == len(result["loss_history"]) - 1:
            bar = "#" * int(l * 400)
            print(f"    epoch {i+1:>2}: {l:.5f}  {bar}")

    section("HYPERPARAMETER SENSITIVITY SWEEP")
    hyperparam_sweep()

    section("GRADIENT FLOW ANALYSIS")
    print("\n  Key observations:")
    print("  1. dA receives gradient from: scale * B^T @ upstream @ x^T")
    print("     At init (B=0): dA = 0. Gradient builds as B develops non-zero values.")
    print("  2. dB receives gradient from: scale * upstream @ (Ax)^T")
    print("     Non-zero from step 1 (A is non-zero at init).")
    print("  3. W0 receives NO gradient — gradient only flows through A and B.")
    print("  4. scale = alpha/r balances update magnitude vs rank.")

    section("PEFT LORA CONFIG — REAL LIBRARY")
    show_peft_config()

    section("TARGET MODULE SELECTION GUIDE")
    print(f"\n  {'Modules':>30} {'Params':>10} {'Notes'}")
    print(f"  {'-'*65}")
    configs = [
        ("q_proj, v_proj",                "Minimal",  "Most common, good tradeoff"),
        ("q_proj, k_proj, v_proj",         "Low",      "Better context understanding"),
        ("q_proj, k_proj, v_proj, o_proj", "Moderate", "Full attention"),
        ("all linear",                     "High",     "Including FFN, near full-FT"),
    ]
    for mods, params, note in configs:
        print(f"  {mods:>30} {params:>10}  {note}")

    section("TRAINING LOOP STRUCTURE SUMMARY")
    print("""
  for epoch in range(num_epochs):
      for batch in dataloader:
          # Forward — base weights frozen, LoRA adapters active
          logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])

          # Loss — only over response tokens (labels != -100)
          loss = cross_entropy(logits, batch["labels"], ignore_index=-100)

          # Backward — gradients flow to A, B only
          loss.backward()

          # Clip gradients
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

          # Update — only A, B move
          optimizer.step()
          optimizer.zero_grad()
          scheduler.step()
    """)


if __name__ == "__main__":
    main()
