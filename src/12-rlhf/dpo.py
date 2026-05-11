"""
Direct Preference Optimization (DPO) from scratch.
Covers: DPO loss derivation, implicit reward, training simulation,
        IPO variant, KTO (binary feedback), SimPO (no reference model),
        comparison table with PPO.
All from scratch with NumPy — no PyTorch required.
"""

import numpy as np

rng = np.random.default_rng(42)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Sigmoid + log-sigmoid ─────────────────────────────────────────
def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.log1p(np.exp(-np.abs(x))) + np.minimum(x, 0)


# ── Synthetic Language Model ──────────────────────────────────────
class TokenModel:
    """
    Minimal language model over a 10-token vocabulary, 8-dimensional embedding.
    Produces per-token log-probabilities for a fixed-length sequence.
    """

    VOCAB_SIZE = 10
    SEQ_LEN    = 8
    DIM        = 16

    def __init__(self):
        self.W = rng.standard_normal((self.VOCAB_SIZE, self.DIM)) * 0.1
        self.U = rng.standard_normal((self.DIM, self.VOCAB_SIZE)) * 0.1

    def log_probs(self, token_ids: np.ndarray) -> np.ndarray:
        """
        token_ids: (B, T) integer token ids
        Returns: (B, T) log-probabilities of each token given position
        """
        B, T = token_ids.shape
        embeds = self.W[token_ids]              # (B, T, D)
        logits = embeds @ self.U                # (B, T, V)
        # log-softmax
        m      = logits.max(axis=-1, keepdims=True)
        log_p  = logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
        # Select log-prob of actual tokens
        row_idx = np.arange(B)[:, None].repeat(T, axis=1)   # (B, T)
        col_idx = np.arange(T)[None, :].repeat(B, axis=0)   # (B, T)
        return log_p[row_idx, col_idx, token_ids]            # (B, T)

    def sequence_log_prob(self, token_ids: np.ndarray) -> np.ndarray:
        """Sum log-probs over sequence → (B,) sequence log-prob."""
        return self.log_probs(token_ids).sum(axis=-1)

    def implicit_reward(self, token_ids: np.ndarray,
                        ref_model: "TokenModel", beta: float) -> np.ndarray:
        """
        r_θ(x, y) = β [log π_θ(y|x) - log π_ref(y|x)]
        In our simulated setting, x is implicit (model position-dependent).
        """
        return beta * (self.sequence_log_prob(token_ids)
                       - ref_model.sequence_log_prob(token_ids))


# ── DPO Loss ──────────────────────────────────────────────────────
def dpo_loss(policy: TokenModel, ref_model: TokenModel,
             chosen_ids: np.ndarray, rejected_ids: np.ndarray,
             beta: float = 0.1) -> tuple:
    """
    L_DPO = -E[log σ(β(log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))]

    Returns (loss, diagnostics_dict)
    """
    chosen_lp  = policy.sequence_log_prob(chosen_ids)    # (B,)
    rejected_lp = policy.sequence_log_prob(rejected_ids)  # (B,)

    with_no_grad = True  # reference model: no gradient
    ref_chosen_lp  = ref_model.sequence_log_prob(chosen_ids)
    ref_rejected_lp = ref_model.sequence_log_prob(rejected_ids)

    chosen_rewards  = beta * (chosen_lp   - ref_chosen_lp)    # (B,)
    rejected_rewards = beta * (rejected_lp - ref_rejected_lp)  # (B,)

    reward_margin = chosen_rewards - rejected_rewards          # (B,)
    loss = -log_sigmoid(reward_margin).mean()

    diag = {
        "loss":              float(loss),
        "reward_margin":     float(reward_margin.mean()),
        "implicit_acc":      float((chosen_rewards > rejected_rewards).mean()),
        "chosen_reward":     float(chosen_rewards.mean()),
        "rejected_reward":   float(rejected_rewards.mean()),
        "chosen_kl":         float((chosen_lp - ref_chosen_lp).mean()),
        "rejected_kl":       float((rejected_lp - ref_rejected_lp).mean()),
    }
    return loss, diag


# ── IPO Loss (Identity Preference Optimization) ───────────────────
def ipo_loss(policy: TokenModel, ref_model: TokenModel,
             chosen_ids: np.ndarray, rejected_ids: np.ndarray,
             beta: float = 0.1) -> tuple:
    """
    IPO: uses MSE instead of log-sigmoid.
    L_IPO = E[(β(log π(c)/π_ref(c) - log π(r)/π_ref(r)) - 1/(2β))^2]
    More robust when one response is clearly better (avoids saturation).
    """
    chosen_lp    = policy.sequence_log_prob(chosen_ids)
    rejected_lp  = policy.sequence_log_prob(rejected_ids)
    ref_c_lp     = ref_model.sequence_log_prob(chosen_ids)
    ref_r_lp     = ref_model.sequence_log_prob(rejected_ids)

    margin = beta * ((chosen_lp - ref_c_lp) - (rejected_lp - ref_r_lp))
    target = 1.0 / (2 * beta)
    loss   = ((margin - target)**2).mean()
    return float(loss), {"margin_mean": float(margin.mean())}


# ── KTO Loss (Kahneman-Tversky Optimization) ──────────────────────
def kto_loss(policy: TokenModel, ref_model: TokenModel,
             token_ids: np.ndarray, labels: np.ndarray,
             beta: float = 0.1) -> float:
    """
    KTO: per-response binary labels (1=good, 0=bad).
    No need for paired (chosen, rejected) data.
    L_KTO = E_good[-log σ(β(log π(y)/π_ref(y) - KL))]
           + E_bad[-log σ(-(β(log π(y)/π_ref(y) - KL)))]
    """
    log_p     = policy.sequence_log_prob(token_ids)
    ref_log_p = ref_model.sequence_log_prob(token_ids)
    log_ratio = log_p - ref_log_p

    # KL from reference (estimated as mean log-ratio)
    kl = float(log_ratio.mean())

    scaled = beta * (log_ratio - kl)

    good_mask = labels == 1
    bad_mask  = labels == 0

    loss = 0.0
    if good_mask.any():
        loss += -log_sigmoid(scaled[good_mask]).mean()
    if bad_mask.any():
        loss += -log_sigmoid(-scaled[bad_mask]).mean()

    return float(loss / 2)


# ── SimPO Loss (Simple Preference Optimization) ───────────────────
def simpo_loss(policy: TokenModel,
               chosen_ids: np.ndarray, rejected_ids: np.ndarray,
               beta: float = 2.0, gamma: float = 0.5) -> float:
    """
    SimPO: no reference model. Uses length-normalized log-probs.
    L_SimPO = -log σ(β/|y_w| Σ log π(y_w) - β/|y_l| Σ log π(y_l) - γ)
    """
    T = chosen_ids.shape[1]
    chosen_lps   = policy.log_probs(chosen_ids).sum(axis=-1) / T    # (B,)
    rejected_lps = policy.log_probs(rejected_ids).sum(axis=-1) / T  # (B,)
    margin = beta * (chosen_lps - rejected_lps) - gamma
    return float(-log_sigmoid(margin).mean())


# ── DPO Gradient (analytical) ─────────────────────────────────────
def dpo_gradient_U(policy: TokenModel, ref_model: TokenModel,
                   chosen_ids: np.ndarray, rejected_ids: np.ndarray,
                   beta: float = 0.1) -> np.ndarray:
    """
    Compute gradient of DPO loss w.r.t. policy.U (output projection).
    Uses finite differences for correctness check.
    """
    eps  = 1e-5
    base_loss, _ = dpo_loss(policy, ref_model, chosen_ids, rejected_ids, beta)
    grad = np.zeros_like(policy.U)

    for i in range(min(3, policy.U.shape[0])):       # check first 3 rows only
        for j in range(min(3, policy.U.shape[1])):
            policy.U[i, j] += eps
            l_plus, _ = dpo_loss(policy, ref_model, chosen_ids, rejected_ids, beta)
            policy.U[i, j] -= 2*eps
            l_minus, _ = dpo_loss(policy, ref_model, chosen_ids, rejected_ids, beta)
            policy.U[i, j] += eps   # restore
            grad[i, j] = (l_plus - l_minus) / (2*eps)

    return grad


# ── DPO Training Simulation ────────────────────────────────────────
def train_dpo(policy: TokenModel, ref_model: TokenModel,
              chosen_ids: np.ndarray, rejected_ids: np.ndarray,
              n_steps: int = 100, lr: float = 0.01, beta: float = 0.1,
              batch_size: int = 32) -> list:
    """Simulated DPO training with finite-difference gradient updates."""
    n = len(chosen_ids)
    history = []
    eps = 1e-4

    for step in range(n_steps):
        # Mini-batch
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        c_batch = chosen_ids[idx]
        r_batch = rejected_ids[idx]

        loss, diag = dpo_loss(policy, ref_model, c_batch, r_batch, beta)

        # Approximate gradient on a subset of U parameters
        for i in range(policy.U.shape[0]):
            policy.U[i, 0] -= eps
            l_minus, _ = dpo_loss(policy, ref_model, c_batch, r_batch, beta)
            policy.U[i, 0] += 2*eps
            l_plus, _ = dpo_loss(policy, ref_model, c_batch, r_batch, beta)
            policy.U[i, 0] -= eps
            grad_i = (l_plus - l_minus) / (2*eps)
            policy.U[i, 0] -= lr * grad_i

        history.append(diag)

        if step % 20 == 0 or step == n_steps - 1:
            print(f"  Step {step:4d}: loss={diag['loss']:.4f}  "
                  f"margin={diag['reward_margin']:.4f}  "
                  f"acc={diag['implicit_acc']:.3f}")

    return history


def main():
    section("1. DPO LOSS DERIVATION RECAP")
    print("""
  Optimal RLHF solution: π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)
  Rearranging: r(x,y) = β log[π*(y|x)/π_ref(y|x)] + β log Z(x)

  Substitute into Bradley-Terry loss:
    L = -log σ(r(x,y_w) - r(x,y_l))
      = -log σ(β log[π_θ(y_w|x)/π_ref(y_w|x)]
               - β log[π_θ(y_l|x)/π_ref(y_l|x)])

  Z(x) cancels! No explicit reward model needed.
  Requires only: two forward passes (policy + reference).
""")

    section("2. IMPLICIT REWARD VISUALIZATION")
    model = TokenModel()
    ref   = TokenModel()
    ref.W = model.W.copy(); ref.U = model.U.copy()   # start identical

    # Slightly perturb policy to simulate post-training
    model.W += rng.standard_normal(model.W.shape) * 0.05
    model.U += rng.standard_normal(model.U.shape) * 0.05

    n_seq = 20
    seq_ids = rng.integers(0, TokenModel.VOCAB_SIZE, (n_seq, TokenModel.SEQ_LEN))

    imp_rewards = model.implicit_reward(seq_ids, ref, beta=0.1)
    print(f"  Implicit rewards β(log π - log π_ref) for {n_seq} sequences:")
    print(f"  min={imp_rewards.min():.4f}  mean={imp_rewards.mean():.4f}  "
          f"max={imp_rewards.max():.4f}  std={imp_rewards.std():.4f}")

    section("3. DPO LOSS COMPUTATION")
    B = 50
    chosen_ids  = rng.integers(0, TokenModel.VOCAB_SIZE, (B, TokenModel.SEQ_LEN))
    rejected_ids = rng.integers(0, TokenModel.VOCAB_SIZE, (B, TokenModel.SEQ_LEN))

    model2 = TokenModel()
    ref2   = TokenModel()

    loss, diag = dpo_loss(model2, ref2, chosen_ids, rejected_ids, beta=0.1)
    print(f"  DPO loss: {loss:.4f}")
    for k, v in diag.items():
        print(f"    {k}: {v:.4f}")

    section("4. DPO TRAINING SIMULATION")
    print("  Training DPO policy on 200 synthetic preference pairs...")
    policy3   = TokenModel()
    ref3      = TokenModel()
    ref3.W    = policy3.W.copy(); ref3.U = policy3.U.copy()

    N = 200
    c_ids = rng.integers(0, TokenModel.VOCAB_SIZE, (N, TokenModel.SEQ_LEN))
    r_ids = rng.integers(0, TokenModel.VOCAB_SIZE, (N, TokenModel.SEQ_LEN))

    history = train_dpo(policy3, ref3, c_ids, r_ids, n_steps=100, lr=0.005, beta=0.1)

    print(f"\n  Training summary:")
    print(f"    Initial acc:  {history[0]['implicit_acc']:.3f}")
    print(f"    Final acc:    {history[-1]['implicit_acc']:.3f}")
    print(f"    Reward margin gain: {history[-1]['reward_margin'] - history[0]['reward_margin']:.4f}")

    section("5. DPO VARIANTS COMPARISON")
    model_v = TokenModel()
    ref_v   = TokenModel()

    print(f"  {'Method':<12} {'Loss':>8} {'Notes'}")
    print(f"  {'-'*50}")

    dpo_l, dpo_d = dpo_loss(model_v, ref_v, chosen_ids, rejected_ids, beta=0.1)
    print(f"  {'DPO':<12} {dpo_l:8.4f}  Standard; requires paired (chosen, rejected)")

    ipo_l, _ = ipo_loss(model_v, ref_v, chosen_ids, rejected_ids, beta=0.1)
    print(f"  {'IPO':<12} {ipo_l:8.4f}  MSE; robust to overconfident labels")

    labels = rng.integers(0, 2, N)   # 0=bad, 1=good
    kto_l = kto_loss(model_v, ref_v, c_ids, labels, beta=0.1)
    print(f"  {'KTO':<12} {kto_l:8.4f}  Binary labels; no pairing needed")

    simpo_l = simpo_loss(model_v, c_ids, r_ids, beta=2.0, gamma=0.5)
    print(f"  {'SimPO':<12} {simpo_l:8.4f}  No reference model; length-normalized")

    section("6. DPO vs PPO COMPARISON TABLE")
    print("""
  Dimension         PPO-RLHF                  DPO
  ----------------  --------                  ---
  Models needed     4 (actor/critic/ref/RM)   2 (policy + ref)
  RL loop           Yes (rollouts + updates)  No (supervised)
  Stability         Sensitive to ε, β, KL     More stable
  Sample efficiency Generates new responses   Uses existing pairs
  Memory            4× base model params      2× base model params
  Flexibility       Any reward signal          Preference pairs only
  Typical LR        1e-6 to 1e-5              5e-7 to 5e-6
  Used by           InstructGPT (initial)     LLaMA-3, Mistral, Claude

  Key insight: DPO is equivalent to PPO when:
    - Reference policy is SFT model
    - Reward model satisfies Bradley-Terry assumptions
    - Optimal policy is reachable

  DPO limitations:
    - Cannot incorporate external reward signals (e.g., execution success for code)
    - Requires good preference data quality (mislabeled pairs hurt more)
    - Offline: no exploration of responses outside the preference dataset
""")

    section("7. TRAINING DIAGNOSTICS TO MONITOR")
    print("""
  During DPO training, monitor:

  1. implicit_accuracy = P(r_θ(chosen) > r_θ(rejected))
     → Should increase toward ~0.85-0.95 (don't push to 1.0)

  2. reward_margin = mean(r_θ(chosen) - r_θ(rejected))
     → Should increase; if it grows too fast → possible overfitting

  3. chosen_kl = mean(log π_θ(chosen) - log π_ref(chosen))
     → Should remain small (< 2 nats); large KL = drifting from SFT

  4. rejected_kl = mean(log π_θ(rejected) - log π_ref(rejected))
     → Should become more negative (model discourages rejected responses)

  5. Win-rate vs. SFT baseline (eval on held-out prompts)
     → Ultimate metric; target > 60% for meaningful improvement
""")


if __name__ == "__main__":
    main()
