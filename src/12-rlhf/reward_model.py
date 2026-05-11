"""
Reward model training from human preference data (Bradley-Terry model).
Covers: preference loss, reward model accuracy, margin loss, length bias,
        training loop, and pairwise ranking evaluation.
All from scratch with NumPy — no PyTorch required.
"""

import numpy as np
from collections import defaultdict

rng = np.random.default_rng(42)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Sigmoid ────────────────────────────────────────────────────────
def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.log1p(np.exp(-np.abs(x))) + np.minimum(x, 0)


# ── Synthetic Preference Dataset ──────────────────────────────────
def make_preference_data(n_pairs: int = 1000, n_features: int = 16) -> dict:
    """
    Simulates a human preference dataset.
    Each example: (prompt_features, chosen_features, rejected_features)
    Ground truth reward: linear function of features.
    """
    true_w = rng.standard_normal(n_features)   # true reward weights (hidden)

    X_prompts   = rng.standard_normal((n_pairs, n_features))
    X_chosen    = rng.standard_normal((n_pairs, n_features))
    X_rejected  = rng.standard_normal((n_pairs, n_features))

    r_chosen    = X_chosen  @ true_w     # true reward scores
    r_rejected  = X_rejected @ true_w

    # Simulate noisy human labels: human prefers "chosen" with prob sigmoid(r_w - r_l)
    prob_correct = sigmoid(r_chosen - r_rejected)
    labels = (rng.uniform(0, 1, n_pairs) < prob_correct).astype(float)

    # Swap chosen/rejected for mislabeled examples (labels=0 means human preferred rejected)
    X_c = X_chosen.copy();  X_r = X_rejected.copy()
    swap = labels == 0
    X_c[swap], X_r[swap] = X_rejected[swap].copy(), X_chosen[swap].copy()

    return {
        "X_prompt":   X_prompts,
        "X_chosen":   X_c,
        "X_rejected": X_r,
        "r_chosen":   np.maximum(r_chosen, r_rejected),      # after swap: always higher
        "r_rejected": np.minimum(r_chosen, r_rejected),
        "true_w":     true_w,
    }


# ── Reward Model ──────────────────────────────────────────────────
class LinearRewardModel:
    """
    Simplest reward model: linear projection from feature space to scalar.
    r(x, y) = (x || y) · w  where || denotes concatenation.
    """

    def __init__(self, n_features: int):
        self.w = rng.standard_normal(n_features * 2) * 0.01   # prompt + response features

    def predict(self, X_prompt: np.ndarray, X_response: np.ndarray) -> np.ndarray:
        X = np.concatenate([X_prompt, X_response], axis=1)
        return X @ self.w   # scalar per example

    def loss(self, r_chosen: np.ndarray, r_rejected: np.ndarray,
             margin: float = 0.0) -> float:
        """Bradley-Terry loss with optional margin."""
        diff = r_chosen - r_rejected - margin
        return -log_sigmoid(diff).mean()

    def accuracy(self, r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
        return float((r_chosen > r_rejected).mean())

    def grad(self, X_prompt: np.ndarray, X_chosen: np.ndarray,
             X_rejected: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Gradient of Bradley-Terry loss w.r.t. w."""
        r_w = self.predict(X_prompt, X_chosen)
        r_l = self.predict(X_prompt, X_rejected)
        diff = r_w - r_l - margin

        # d(loss)/d(r_w - r_l) = sigmoid(diff) - 1 = -sigmoid(-diff)
        delta = sigmoid(diff) - 1.0   # (B,)

        X_c = np.concatenate([X_prompt, X_chosen],   axis=1)
        X_r = np.concatenate([X_prompt, X_rejected], axis=1)
        # dL/dw = delta · (X_c - X_r)  (gradient for reward difference)
        return (delta[:, None] * (X_c - X_r)).mean(axis=0)


def train_reward_model(model: LinearRewardModel, data: dict,
                       n_epochs: int = 50, lr: float = 0.01,
                       batch_size: int = 64, margin: float = 0.0) -> list:
    n = len(data["X_prompt"])
    history = []

    for epoch in range(n_epochs):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, batch_size):
            b = idx[start:start+batch_size]
            r_w  = model.predict(data["X_prompt"][b], data["X_chosen"][b])
            r_l  = model.predict(data["X_prompt"][b], data["X_rejected"][b])
            loss = model.loss(r_w, r_l, margin)
            grad = model.grad(data["X_prompt"][b], data["X_chosen"][b],
                              data["X_rejected"][b], margin)
            model.w -= lr * grad
            epoch_loss += loss
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        r_w_all  = model.predict(data["X_prompt"], data["X_chosen"])
        r_l_all  = model.predict(data["X_prompt"], data["X_rejected"])
        acc      = model.accuracy(r_w_all, r_l_all)
        history.append({"epoch": epoch, "loss": avg_loss, "acc": acc})

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}  acc={acc:.3f}")

    return history


# ── Length Bias Analysis ──────────────────────────────────────────
def length_bias_demo(n: int = 500, n_features: int = 8):
    """
    Demonstrates length bias: longer responses tend to score higher
    even if quality is controlled.
    """
    true_w = rng.standard_normal(n_features)

    lengths_chosen    = rng.integers(10, 200, n)   # random lengths
    lengths_rejected  = rng.integers(10, 200, n)

    X_c = rng.standard_normal((n, n_features))
    X_r = rng.standard_normal((n, n_features))

    # Add length signal to features (simulates a length-biased reward)
    X_c_biased = np.column_stack([X_c, lengths_chosen.reshape(-1, 1)])
    X_r_biased = np.column_stack([X_r, lengths_rejected.reshape(-1, 1)])
    biased_w   = np.append(true_w, 0.1)   # length weight = 0.1

    r_c_biased = X_c_biased @ biased_w
    r_r_biased = X_r_biased @ biased_w

    # Fraction of preferences explained by length alone
    length_correct = ((lengths_chosen > lengths_rejected) == (r_c_biased > r_r_biased)).mean()
    return length_correct


# ── Reward Normalization ──────────────────────────────────────────
def normalize_rewards(rewards: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize rewards to prevent reward hacking via scale."""
    if method == "zscore":
        return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    elif method == "minmax":
        lo, hi = rewards.min(), rewards.max()
        return (rewards - lo) / (hi - lo + 1e-8)
    elif method == "clip":
        return np.clip(rewards, -5.0, 5.0)
    return rewards


def main():
    section("1. BRADLEY-TERRY PREFERENCE MODEL")
    print("  P(chosen ≻ rejected | x) = σ(r(x, chosen) - r(x, rejected))")
    print("  Loss = -E[log σ(r_chosen - r_rejected)]")
    print()

    # Demonstrate loss shape
    diffs = np.linspace(-5, 5, 11)
    print(f"  {'r_w - r_l':>10} {'Loss':>10} {'Accuracy signal':>16}")
    print(f"  {'-'*40}")
    for d in diffs:
        loss = -log_sigmoid(np.array([d]))[0]
        acc  = sigmoid(np.array([d]))[0]
        print(f"  {d:>10.1f} {loss:>10.4f} {acc:>16.4f}")

    section("2. REWARD MODEL TRAINING")
    data = make_preference_data(n_pairs=2000, n_features=16)
    n_train = 1600
    train_data = {k: v[:n_train] for k, v in data.items()}
    val_data   = {k: v[n_train:] for k, v in data.items()}

    model = LinearRewardModel(n_features=16)
    print("  Training on 1600 preference pairs, validating on 400...")
    history = train_reward_model(model, train_data, n_epochs=50, lr=0.01, batch_size=64)

    r_w_val = model.predict(val_data["X_prompt"], val_data["X_chosen"])
    r_l_val = model.predict(val_data["X_prompt"], val_data["X_rejected"])
    val_acc = model.accuracy(r_w_val, r_l_val)
    print(f"\n  Validation accuracy: {val_acc:.3f}")
    print(f"  (Random baseline: 0.500, well-trained RM: ~0.75+)")

    section("3. MARGIN LOSS COMPARISON")
    for margin in [0.0, 0.5, 1.0, 2.0]:
        model_m = LinearRewardModel(n_features=16)
        hist = train_reward_model(model_m, train_data, n_epochs=30, lr=0.01,
                                  batch_size=64, margin=margin)
        r_w = model_m.predict(val_data["X_prompt"], val_data["X_chosen"])
        r_l = model_m.predict(val_data["X_prompt"], val_data["X_rejected"])
        acc  = model_m.accuracy(r_w, r_l)
        mean_gap = float((r_w - r_l).mean())
        print(f"  margin={margin:.1f}: val_acc={acc:.3f}  mean_r_gap={mean_gap:.3f}")

    section("4. LENGTH BIAS ANALYSIS")
    bias_frac = length_bias_demo()
    print(f"  Fraction of preferences where 'longer = preferred': {bias_frac:.3f}")
    print("  → If > 0.6, reward model has significant length bias")
    print("  Mitigation: normalize by sqrt(seq_len) or train on length-balanced data")

    section("5. REWARD NORMALIZATION")
    raw_rewards = rng.standard_normal(1000) * 3.0 + 1.5
    print(f"  Raw rewards: mean={raw_rewards.mean():.2f}  std={raw_rewards.std():.2f}  "
          f"range=[{raw_rewards.min():.2f}, {raw_rewards.max():.2f}]")

    for method in ["zscore", "minmax", "clip"]:
        normed = normalize_rewards(raw_rewards, method)
        print(f"  {method:8}: mean={normed.mean():.2f}  std={normed.std():.2f}  "
              f"range=[{normed.min():.2f}, {normed.max():.2f}]")

    section("6. PAIRWISE RANKING EVALUATION")
    model_final = LinearRewardModel(n_features=16)
    train_reward_model(model_final, train_data, n_epochs=50, lr=0.01, batch_size=64)

    # Evaluate on 3-way ranking: does RM correctly rank A > B > C?
    n_triples = 200
    X_a = rng.standard_normal((n_triples, 16))
    X_b = rng.standard_normal((n_triples, 16))
    X_c_data = rng.standard_normal((n_triples, 16))
    X_p = rng.standard_normal((n_triples, 16))

    true_w = data["true_w"]
    r_a = X_a @ true_w
    r_b = X_b @ true_w
    r_c = X_c_data @ true_w

    # Sort by true reward: A=best, B=mid, C=worst
    best_idx = np.argmax(np.stack([r_a, r_b, r_c], axis=1), axis=1)

    rm_a = model_final.predict(X_p, X_a)
    rm_b = model_final.predict(X_p, X_b)
    rm_c = model_final.predict(X_p, X_c_data)

    rm_ranks = np.stack([rm_a, rm_b, rm_c], axis=1)
    rm_best  = rm_ranks.argmax(axis=1)

    top1_acc = (rm_best == best_idx).mean()
    print(f"  Top-1 accuracy on 3-way ranking: {top1_acc:.3f}")
    print(f"  (Random baseline: 0.333)")

    section("7. REWARD MODEL SUMMARY")
    print(f"""
  Architecture: LM backbone → mean pool → Linear(d, 1)
  Loss:         -E[log σ(r_chosen - r_rejected)]
  With margin:  -E[log σ(r_chosen - r_rejected - m)]  (m > 0)
  Evaluation:   pairwise accuracy = P(r_chosen > r_rejected)
  Typical:      70-80% on human preference test set

  Common failure modes:
    - Length bias: longer → higher reward (add length normalization)
    - Sycophancy: agreeable but wrong answers score high
    - Distribution shift: RM trained on SFT outputs, tested on RLHF outputs
    - Reward hacking: policy finds inputs that exploit RM blind spots
""")


if __name__ == "__main__":
    main()
