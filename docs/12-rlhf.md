# Module 12 — RLHF & Alignment

## Why Alignment Matters

A language model trained on next-token prediction learns to imitate the internet. The internet contains misinformation, harmful content, and text that confidently states falsehoods. A model that perfectly imitates this data is not helpful — it's dangerous.

**The alignment problem**: how do we train models to be helpful, harmless, and honest when:
1. We can't enumerate all bad behaviors in advance
2. Reward signals are expensive (human annotation)
3. Models can find unexpected ways to optimize a reward

RLHF (Reinforcement Learning from Human Feedback) is the current practical answer. DPO (Direct Preference Optimization) is its simpler successor.

---

> **Python prerequisite:** This module uses Python, NumPy, and ML libraries throughout. If you need a foundation or refresher, visit the **Languages → Python** guide and read **Section 21 — Python for ML & AI** before starting.

## 1. The RLHF Pipeline

RLHF has three phases:

```
Phase 1: Supervised Fine-Tuning (SFT)
  Input: demonstrations from human experts
  Output: SFT model that knows the format

Phase 2: Reward Model Training
  Input: pairs of responses, human ranks one as better
  Output: reward model R(prompt, response) → scalar

Phase 3: RL Optimization (PPO)
  Input: SFT model + reward model
  Output: policy that maximizes R while staying close to SFT
```

```
Full RLHF pipeline — data flow:

┌──────────────────────────────────────────────────────────────┐
│ PHASE 1: Supervised Fine-Tuning                              │
│                                                              │
│  Prompt: "Explain gravity"                                   │
│  Human: writes a good response                               │
│  SFT model: trained to imitate this format                   │
└─────────────────────────┬────────────────────────────────────┘
                          ▼ SFT model
┌──────────────────────────────────────────────────────────────┐
│ PHASE 2: Reward Model Training                               │
│                                                              │
│  Prompt: "Explain gravity"                                   │
│  Response A: "Gravity is a force that..."   ← human picks A │
│  Response B: "Gravity makes things fall"    ← rejected       │
│                                                              │
│  Reward Model learns: r(A) > r(B)                            │
│  Loss: -log σ(r(A) - r(B))                                   │
└─────────────────────────┬────────────────────────────────────┘
                          ▼ reward model
┌──────────────────────────────────────────────────────────────┐
│ PHASE 3: PPO (Proximal Policy Optimization)                  │
│                                                              │
│  Policy (SFT model) generates response                       │
│  Reward model scores it: r = 0.87                            │
│  KL penalty: -β × KL(policy || SFT)  ← prevents "gaming"    │
│  Objective: maximize r - β × KL                              │
│  PPO update: clip(policy ratio) to prevent big jumps         │
└──────────────────────────────────────────────────────────────┘
```

### Phase 1: Supervised Fine-Tuning

Covered in Module 09. Train on `(instruction, high_quality_response)` pairs. The SFT model is the starting point for phases 2 and 3.

---

## 2. Reward Model

The reward model learns human preferences from pairwise comparisons. Given $(x, y_w, y_l)$ where $y_w$ is the preferred response ("winner") and $y_l$ is the rejected response ("loser"):

$$\mathcal{L}_\text{RM} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

This is the Bradley-Terry model for pairwise preferences. It pushes $r(y_w) > r(y_l)$:
- If $r(y_w) \gg r(y_l)$: $\sigma(\text{large}) \approx 1$ → $\log(1) \approx 0$ → low loss ✓
- If $r(y_w) < r(y_l)$: $\sigma(\text{negative}) < 0.5$ → $\log(\text{small}) \ll 0$ → high loss ✗

```python
import numpy as np
import math
from collections import defaultdict


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid: 1/(1 + e^-x)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class RewardModel:
    """
    A simple scalar reward model.
    
    Architecture: linear function over hand-crafted features.
    In practice: a language model with a linear head replacing the LM head.
    
    Learns: which responses humans prefer for a given prompt.
    """
    
    def __init__(self, n_features: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Random initialization — will be trained to match human preferences
        self.weights = rng.standard_normal(n_features) * 0.01
        self.bias = 0.0
        self.n_features = n_features
    
    def score(self, features: np.ndarray) -> float:
        """r(x, y) → scalar reward. Higher = more preferred."""
        return float(features @ self.weights + self.bias)
    
    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Score multiple (prompt, response) feature vectors. Shape: (N,)."""
        return X @ self.weights + self.bias
    
    def bradley_terry_loss(
        self,
        X_win: np.ndarray,
        X_lose: np.ndarray
    ) -> tuple[float, np.ndarray, float]:
        """
        Bradley-Terry pairwise loss for a batch of comparisons.
        
        X_win:  features for preferred responses, shape (batch, n_features)
        X_lose: features for rejected responses, shape (batch, n_features)
        
        Returns: (loss, gradient_weights, gradient_bias)
        
        Loss: -mean(log σ(r_win - r_lose))
        """
        r_win = self.score_batch(X_win)   # (batch,)
        r_lose = self.score_batch(X_lose)  # (batch,)
        
        margin = r_win - r_lose  # positive = model correctly ranks winner higher
        
        # Log-sigmoid loss: -log(σ(margin))
        # Numerically stable: log(σ(x)) = -log(1 + e^-x) = -softplus(-x)
        loss = -np.mean(np.log(sigmoid(margin) + 1e-9))
        
        # Gradient of -log(σ(margin)) w.r.t. margin:
        # d/d(margin) [-log(σ(m))] = -(1 - σ(m)) = σ(m) - 1
        grad_margin = sigmoid(margin) - 1.0  # (batch,) — negative = push margin up
        
        # Gradient w.r.t. weights:
        # margin = (X_win - X_lose) @ w, so d(margin)/d(w) = (X_win - X_lose)
        X_diff = X_win - X_lose  # (batch, n_features)
        grad_w = (grad_margin[:, None] * X_diff).mean(axis=0)  # (n_features,)
        grad_b = float(grad_margin.mean())
        
        return loss, grad_w, grad_b
    
    def accuracy(self, X_win: np.ndarray, X_lose: np.ndarray) -> float:
        """Fraction of pairs where reward model correctly prefers winner."""
        r_win = self.score_batch(X_win)
        r_lose = self.score_batch(X_lose)
        return float(np.mean(r_win > r_lose))
    
    def train(self, X_win: np.ndarray, X_lose: np.ndarray, lr: float = 0.01, epochs: int = 100) -> list[float]:
        """Train reward model on preference pairs."""
        losses = []
        for epoch in range(epochs):
            loss, grad_w, grad_b = self.bradley_terry_loss(X_win, X_lose)
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b
            losses.append(loss)
            
            if epoch % 20 == 0:
                acc = self.accuracy(X_win, X_lose)
                print(f"  Epoch {epoch:3d}: loss={loss:.4f}, acc={acc:.4f}")
        
        return losses


section("Reward Model Training")
rng = np.random.default_rng(42)
n_features = 8

# Simulate: response quality correlated with first 3 features
# Feature 0: response length (longer = better for this task)
# Feature 1: answer relevance
# Feature 2: factual accuracy
# Features 3-7: noise

n_pairs = 500
X_win = rng.standard_normal((n_pairs, n_features))
X_lose = rng.standard_normal((n_pairs, n_features))

# Make winners genuinely better: higher values in quality features
X_win[:, :3] += 1.0   # winners have higher quality features
X_lose[:, :3] -= 0.5  # losers have lower quality features

rm = RewardModel(n_features=n_features)
losses = rm.train(X_win, X_lose, lr=0.05, epochs=100)

print(f"\nFinal train accuracy: {rm.accuracy(X_win, X_lose):.4f}")
print(f"Learned weights (first 3 should dominate): {rm.weights.round(3)}")
```

---

## 3. Proximal Policy Optimization (PPO)

PPO optimizes the language model policy $\pi_\theta$ to maximize the reward model's score, while staying close to the SFT reference policy $\pi_\text{ref}$ (to prevent reward hacking).

The RLHF objective:

$$\mathcal{L}_\text{PPO} = \mathbb{E}_{y \sim \pi_\theta} \left[ r(x, y) - \beta \underbrace{\log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}_{\text{KL penalty}} \right]$$

- $r(x, y)$: reward model score
- $\beta$: KL penalty coefficient (typically 0.01–0.1) — prevents the policy from drifting too far from the reference

The KL term ensures the model doesn't find degenerate high-reward outputs that exploit the reward model's blind spots.

### KL Divergence Between Policies

For two policies with token probability distributions $\pi_\theta$ and $\pi_\text{ref}$:

$$D_\text{KL}(\pi_\theta \| \pi_\text{ref}) = \sum_y \pi_\theta(y) \log \frac{\pi_\theta(y)}{\pi_\text{ref}(y)}$$

```python
class PolicyModel:
    """
    Simplified policy model (language model).
    
    In RLHF:
    - policy_model: being trained (starts as copy of SFT)
    - reference_model: frozen SFT model (used for KL penalty)
    
    Here: a softmax over a small vocabulary for demonstration.
    """
    
    def __init__(self, vocab_size: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.logits = rng.standard_normal(vocab_size) * 0.1  # unnormalized scores
        self.vocab_size = vocab_size
    
    def probabilities(self) -> np.ndarray:
        """Softmax over logits → probability distribution over vocabulary."""
        logits = self.logits - self.logits.max()  # numerical stability
        exp = np.exp(logits)
        return exp / exp.sum()
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample token indices according to current policy."""
        probs = self.probabilities()
        return np.random.choice(self.vocab_size, size=n, p=probs)
    
    def log_prob(self, token_idx: int) -> float:
        """Log probability of a specific token."""
        return float(np.log(self.probabilities()[token_idx] + 1e-9))
    
    def clone(self) -> "PolicyModel":
        """Create a frozen copy (reference policy)."""
        ref = PolicyModel(self.vocab_size)
        ref.logits = self.logits.copy()
        return ref


def kl_divergence(pi: PolicyModel, pi_ref: PolicyModel) -> float:
    """
    KL(π || π_ref) = Σ π(y) log(π(y) / π_ref(y))
    
    Measures how far the current policy has drifted from reference.
    KL = 0 when policies are identical; increases as they diverge.
    """
    p = pi.probabilities()       # current policy
    q = pi_ref.probabilities()   # reference policy
    
    # Add epsilon to avoid log(0)
    return float(np.sum(p * np.log((p + 1e-9) / (q + 1e-9))))


class PPOTrainer:
    """
    Simplified PPO for language model alignment.
    
    The PPO objective: maximize reward while penalizing KL divergence from reference.
    PPO adds a clipping trick on top of this to prevent large updates.
    """
    
    def __init__(
        self,
        policy: PolicyModel,
        reference: PolicyModel,
        reward_model: RewardModel,
        beta: float = 0.1,     # KL penalty coefficient
        lr: float = 0.01,
        clip_eps: float = 0.2,  # PPO clipping range
    ):
        self.policy = policy
        self.reference = reference
        self.reward_fn = reward_model
        self.beta = beta
        self.lr = lr
        self.clip_eps = clip_eps
        
        self.rewards_history: list[float] = []
        self.kl_history: list[float] = []
    
    def compute_reward(self, action: int) -> float:
        """
        Get reward for a generated response (token).
        In real RLHF: generate full response, pass to reward model.
        Here: use token index as a proxy feature.
        """
        # Create feature vector from token action
        feature = np.zeros(self.reward_fn.n_features)
        feature[action % self.reward_fn.n_features] = 1.0
        feature[0] = action / self.policy.vocab_size  # normalize
        return self.reward_fn.score(feature)
    
    def ppo_step(self) -> dict:
        """
        One PPO update step.
        
        1. Sample action from policy
        2. Get reward
        3. Compute PPO loss (clipped surrogate + KL penalty)
        4. Update policy logits
        """
        # 1. Sample an action (which token/response to generate)
        action = int(self.policy.sample(1)[0])
        
        # 2. Get reward from reward model
        reward = self.compute_reward(action)
        
        # 3. Compute KL penalty
        kl = kl_divergence(self.policy, self.reference)
        
        # 4. RLHF objective: reward - β·KL
        rlhf_reward = reward - self.beta * kl
        
        # 5. PPO policy gradient
        # Log probability of the sampled action under current policy
        pi_log_prob = self.policy.log_prob(action)
        pi_ref_log_prob = self.reference.log_prob(action)
        
        # Probability ratio: π_θ / π_ref
        ratio = math.exp(pi_log_prob - pi_ref_log_prob)
        
        # Clipped PPO objective: min(r·A, clip(r, 1-ε, 1+ε)·A)
        # Advantage A = rlhf_reward (simplified — no value function baseline)
        advantage = rlhf_reward
        
        clipped_ratio = max(1 - self.clip_eps, min(1 + self.clip_eps, ratio))
        ppo_loss = -min(ratio * advantage, clipped_ratio * advantage)
        
        # Gradient of loss w.r.t. logits
        # d(log π(a))/d(logit_a) = 1 - π(a) (for sampled action)
        # d(log π(a))/d(logit_j) = -π(j)     (for all other actions)
        probs = self.policy.probabilities()
        grad = probs.copy()
        grad[action] -= 1.0  # ∇ log π(a) = one_hot(a) - π
        
        # gradient of loss = -(advantage) × ∇ log π(a)
        self.policy.logits -= self.lr * (-advantage * grad)
        
        self.rewards_history.append(reward)
        self.kl_history.append(kl)
        
        return {"reward": reward, "kl": kl, "rlhf_reward": rlhf_reward}
    
    def train(self, n_steps: int = 200) -> None:
        for step in range(n_steps):
            stats = self.ppo_step()
            if step % 50 == 0:
                avg_reward = sum(self.rewards_history[-50:]) / max(len(self.rewards_history[-50:]), 1)
                avg_kl = sum(self.kl_history[-50:]) / max(len(self.kl_history[-50:]), 1)
                print(f"  Step {step:4d}: reward={avg_reward:.4f}, KL={avg_kl:.4f}")


section("PPO Training Loop")
policy = PolicyModel(vocab_size=10, seed=42)
reference = policy.clone()  # frozen copy

ppo = PPOTrainer(policy, reference, rm, beta=0.05, lr=0.02)
print("Before training:")
print(f"  Policy probs: {policy.probabilities().round(3)}")
print(f"  KL from reference: {kl_divergence(policy, reference):.6f}")

ppo.train(n_steps=200)

print("\nAfter training:")
print(f"  Policy probs: {policy.probabilities().round(3)}")
print(f"  KL from reference: {kl_divergence(policy, reference):.4f}")
print(f"  Most favored token: {policy.probabilities().argmax()} (highest reward probability)")
```

---

## 4. DPO: Direct Preference Optimization

PPO requires training and maintaining 4 models simultaneously: policy, reference, reward model, and value function. This is complex, memory-intensive, and training is unstable.

DPO (Rafailov et al. 2023) eliminates the separate reward model entirely. The key insight: the optimal policy under RLHF can be expressed directly in terms of preference data.

```
RLHF/PPO (complex):                   DPO (simple):

Preference data                        Preference data
     │                                      │
     ▼                                      ▼
Reward model training (separate)       Directly compute DPO loss
     │
     ▼
PPO loop:                              ┌─────────────────────────┐
  policy model      ◀── gradients ──── │ π_θ (policy, trainable) │
  reference model                      │ π_ref (frozen baseline)  │
  reward model                         └─────────────────────────┘
  value function                            │
(4 models!)                            loss = -log σ(β·(log π_θ(y_w)/π_ref(y_w)
                                               - log π_θ(y_l)/π_ref(y_l)))

DPO intuition:
  "Make the policy assign relatively higher probability to y_w than y_l,
   compared to where it started (the reference model)."
```

**DPO reparameterization:**

The reward model can be implicitly defined as:
$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

where $Z(x)$ is a partition function that cancels in the loss. Substituting into the Bradley-Terry objective:

$$\mathcal{L}_\text{DPO}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]$$

**What this means**: push up $\log \pi_\theta(y_w)$ (preferred) and push down $\log \pi_\theta(y_l)$ (rejected), relative to the reference.

```python
class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    
    Key advantage over PPO:
    - No separate reward model needed
    - No RL loop — standard supervised loss
    - More stable training
    - Only 2 models needed: policy + frozen reference
    """
    
    def __init__(
        self,
        policy: PolicyModel,
        reference: PolicyModel,
        beta: float = 0.1,
        lr: float = 0.01,
    ):
        self.policy = policy
        self.reference = reference
        self.beta = beta
        self.lr = lr
    
    def dpo_loss(self, preferred_token: int, rejected_token: int) -> tuple[float, np.ndarray]:
        """
        DPO loss for one (preferred, rejected) pair.
        
        Loss = -log σ( β·(log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)) )
        
        preferred_token: y_w — the response humans prefer
        rejected_token:  y_l — the response humans reject
        """
        # Log ratios: log π_θ(y) - log π_ref(y) for each response
        log_ratio_w = (
            self.policy.log_prob(preferred_token) -
            self.reference.log_prob(preferred_token)
        )
        log_ratio_l = (
            self.policy.log_prob(rejected_token) -
            self.reference.log_prob(rejected_token)
        )
        
        # DPO margin: how much more the policy prefers y_w over y_l (vs reference)
        margin = self.beta * (log_ratio_w - log_ratio_l)
        
        # Loss: -log(σ(margin))
        loss = -math.log(sigmoid(margin) + 1e-9)
        
        # Gradient w.r.t. policy logits
        # d(-log σ(m))/dm = -(1 - σ(m)) = σ(m) - 1
        d_margin = sigmoid(margin) - 1.0  # negative → push margin up
        
        probs = self.policy.probabilities()
        
        # Gradient from preferred token: β · d_margin · d(log π(y_w))/d(logits)
        # d(log π(y_k))/d(logit_j) = 1[j=k] - π(j)
        grad = np.zeros(self.policy.vocab_size)
        
        # Preferred: push up log_ratio_w → push up log π(y_w)
        grad_preferred = probs.copy()
        grad_preferred[preferred_token] -= 1.0  # = -(one_hot(y_w) - probs)
        
        # Rejected: push down log_ratio_l → push down log π(y_l)
        grad_rejected = probs.copy()
        grad_rejected[rejected_token] -= 1.0
        
        # Combined: β · d_margin · (grad_preferred - grad_rejected)
        grad = self.beta * d_margin * (grad_preferred - grad_rejected)
        
        return loss, grad
    
    def train(self, preference_pairs: list[tuple[int, int]], epochs: int = 100) -> list[float]:
        """
        preference_pairs: list of (preferred_token, rejected_token)
        """
        losses = []
        n = len(preference_pairs)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_grad = np.zeros(self.policy.vocab_size)
            
            for y_w, y_l in preference_pairs:
                loss, grad = self.dpo_loss(y_w, y_l)
                epoch_loss += loss
                epoch_grad += grad
            
            # Average gradient over batch
            avg_loss = epoch_loss / n
            avg_grad = epoch_grad / n
            
            # SGD update
            self.policy.logits -= self.lr * avg_grad
            
            losses.append(avg_loss)
            
            if epoch % 20 == 0:
                acc = self.compute_accuracy(preference_pairs)
                print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={acc:.4f}")
        
        return losses
    
    def compute_accuracy(self, preference_pairs: list[tuple[int, int]]) -> float:
        """Fraction of pairs where policy assigns higher prob to preferred response."""
        correct = 0
        for y_w, y_l in preference_pairs:
            if self.policy.log_prob(y_w) > self.policy.log_prob(y_l):
                correct += 1
        return correct / len(preference_pairs)
    
    def implicit_reward(self, token: int) -> float:
        """
        The reward implicitly learned by DPO:
        r(y) = β · log(π_θ(y) / π_ref(y))
        """
        return self.beta * (self.policy.log_prob(token) - self.reference.log_prob(token))


section("DPO Training")

# Fresh policies for DPO comparison
dpo_policy = PolicyModel(vocab_size=10, seed=99)
dpo_reference = dpo_policy.clone()

# Create preference dataset: tokens 7, 8, 9 are "good" responses
# tokens 0, 1, 2 are "bad" responses
rng_pairs = np.random.default_rng(0)
preference_pairs = []
for _ in range(50):
    y_w = int(rng_pairs.choice([7, 8, 9]))   # preferred: tokens 7-9
    y_l = int(rng_pairs.choice([0, 1, 2]))   # rejected: tokens 0-2
    preference_pairs.append((y_w, y_l))

print("Before DPO training:")
probs_before = dpo_policy.probabilities()
print(f"  Probs (tokens 7-9): {probs_before[7:]:.3f}")
print(f"  Probs (tokens 0-2): {probs_before[:3]:.3f}")

dpo_trainer = DPOTrainer(dpo_policy, dpo_reference, beta=0.1, lr=0.05)
dpo_trainer.train(preference_pairs, epochs=100)

print("\nAfter DPO training:")
probs_after = dpo_policy.probabilities()
print(f"  Probs (tokens 7-9): {probs_after[7:]:.3f}  (should increase)")
print(f"  Probs (tokens 0-2): {probs_after[:3]:.3f}  (should decrease)")

print("\nImplicit rewards:")
for tok in [0, 1, 7, 8, 9]:
    r = dpo_trainer.implicit_reward(tok)
    print(f"  Token {tok}: implicit reward = {r:.4f}")
```

---

## 5. RLHF vs DPO Comparison

```python
section("RLHF vs DPO: Side-by-Side")

comparison = """
┌─────────────────────────┬────────────────────────────┬────────────────────────────┐
│ Aspect                  │ RLHF (PPO)                 │ DPO                        │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Models needed           │ Policy + Reference +        │ Policy + Reference (frozen)│
│                         │ Reward Model + Value Fn     │                            │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Data format             │ (prompt, response) pairs   │ (prompt, win, lose) pairs  │
│                         │ + scalar rewards            │                            │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Training stability      │ Unstable (RL variance)     │ Stable (supervised loss)   │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Memory                  │ ~4× base model             │ ~2× base model             │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Online vs offline       │ Online (sample during FT)  │ Offline (static dataset)   │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Reward hacking risk     │ High (explicit RM)         │ Lower (no separate RM)     │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Hyperparameter sens.    │ High (β, clip ε, LR, ...)  │ Low (mainly β, LR)         │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Quality ceiling         │ Higher (can improve online)│ Limited by dataset quality │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Who uses it             │ OpenAI GPT-3.5/4           │ Mistral, Zephyr, LLaMA-3   │
└─────────────────────────┴────────────────────────────┴────────────────────────────┘
"""
print(comparison)
```

---

## 6. Constitutional AI (CAI) and RLAIF

**CAI** (Anthropic, 2022): instead of human preference labels, use the AI itself to critique and revise responses according to a constitution (a list of principles).

```
Phase 1 (Supervised): Generate harmful response → have AI critique it → revise → fine-tune on revisions
Phase 2 (RL): Generate response pairs → have AI rank them → use these as reward signal (RLAIF)
```

RLAIF (RL from AI Feedback) uses an LLM as the rater instead of humans. Much cheaper to scale; quality depends on rater model strength.

```python
class ConstitutionalAI:
    """
    Simulates CAI's critique-and-revise loop.
    
    Given a draft response, checks it against a set of principles
    and generates a revised, more aligned response.
    """
    
    PRINCIPLES = [
        "Be helpful and directly address the question",
        "Do not provide information that could cause harm",
        "Be honest and acknowledge uncertainty",
        "Respect user autonomy and avoid being patronizing",
        "Be concise — do not pad responses with unnecessary content",
    ]
    
    @classmethod
    def critique(cls, response: str) -> list[str]:
        """
        Identify which principles the response may violate.
        In production: use a strong LLM to do this.
        Here: rule-based simulation.
        """
        issues = []
        
        if len(response) > 500:
            issues.append("Response is too long — violates principle: Be concise")
        
        harm_words = ["kill", "destroy", "harm", "illegal", "hack"]
        if any(w in response.lower() for w in harm_words):
            issues.append("Response contains potentially harmful content")
        
        if "?" not in response and len(response.split()) > 50:
            issues.append("Response may be padding — check if it directly addresses the question")
        
        return issues
    
    @classmethod
    def revise(cls, original: str, issues: list[str]) -> str:
        """
        Revise response to address identified issues.
        In production: prompt a strong LLM with issues + original.
        """
        if not issues:
            return original
        
        revised = original
        
        if any("long" in issue for issue in issues):
            # Truncate to first 2 sentences
            sentences = revised.split(". ")
            revised = ". ".join(sentences[:2]) + ("." if len(sentences) > 2 else "")
        
        # In production: LLM rewrites to address each issue
        return f"[Revised after {len(issues)} critique(s)]: {revised}"
    
    @classmethod
    def pipeline(cls, prompt: str, response: str) -> dict:
        """Full CAI pipeline: critique → revise → return both."""
        issues = cls.critique(response)
        revised = cls.revise(response, issues)
        
        return {
            "original": response[:100] + "..." if len(response) > 100 else response,
            "issues_found": issues,
            "revised": revised[:100] + "..." if len(revised) > 100 else revised,
            "was_revised": bool(issues),
        }


section("Constitutional AI Demo")

test_cases = [
    (
        "How do I write a function in Python?",
        "To write a function in Python, use the def keyword followed by the function name and parameters. "
        "Here's an example: def greet(name): return f'Hello, {name}'"
    ),
    (
        "Tell me something.",
        "Well, I could tell you many things. The universe is vast and incomprehensible. " * 20  # too long
    ),
]

for prompt, response in test_cases:
    result = ConstitutionalAI.pipeline(prompt, response)
    print(f"\nPrompt: {prompt[:60]}")
    print(f"Issues: {result['issues_found']}")
    print(f"Revised: {result['was_revised']}")
    if result['was_revised']:
        print(f"Output: {result['revised'][:100]}")
```

---

## 7. Reward Hacking and Overoptimization

**Reward hacking**: the model finds unexpected high-reward behaviors that don't reflect true human preferences.

Examples:
- Reward model scores very long responses highly → model generates padded, repetitive text
- Reward model scores confident-sounding responses → model becomes overconfident regardless of factual accuracy
- Reward model prefers bullet points → every response becomes a bullet list even when inappropriate

The KL penalty in PPO/DPO partially prevents this, but doesn't eliminate it.

```python
def demonstrate_overoptimization():
    """
    Show the reward-KL tradeoff at different β values.
    High β: stays close to reference, lower reward
    Low β: achieves higher reward, but drifts far from reference
    """
    section("Reward-KL Tradeoff (Overoptimization Demo)")
    
    # Simple proxy: policy logits shift toward high-reward token
    print(f"{'β (KL penalty)':>16} {'Achieved Reward':>16} {'KL from ref':>14} {'Verdict':>12}")
    print("-" * 62)
    
    # Simulate: higher β → less reward but stays close to reference
    for beta in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        # Optimal policy logit shift under KL-penalized RLHF
        # (simplified analytical approximation)
        reward_achieved = 2.0 / (1.0 + beta * 5)    # diminishing reward with high β
        kl_from_ref = 0.5 / (beta + 0.01)           # lower β = more drift
        
        if kl_from_ref > 2.0:
            verdict = "HACKING"
        elif kl_from_ref > 0.5:
            verdict = "risk"
        else:
            verdict = "safe"
        
        print(f"{beta:>16.3f} {reward_achieved:>16.4f} {kl_from_ref:>14.4f} {verdict:>12}")
    
    print("\nConclusion: β ∈ [0.05, 0.1] typically balances reward and safety.")


demonstrate_overoptimization()
```

---

## 8. Interview Q&A

**Q: What problem does RLHF solve?**
SFT trains on demonstrations — what an expert would say. But what users want is harder to demonstrate than to evaluate: it's easier to say "response A is better than B" than to write a perfect response from scratch. RLHF uses pairwise preferences to learn what "good" looks like, then optimizes toward it.

**Q: Why is the KL penalty necessary in PPO?**
Without it, the policy will exploit the reward model's blind spots — finding inputs that score high but aren't actually preferred. The reward model is an imperfect proxy; the KL penalty keeps the policy anchored to the SFT model's sensible behavior distribution.

**Q: How does DPO differ from PPO?**
PPO: train reward model first, then use RL loop with the policy sampling from itself. DPO: reparameterize the problem so the optimal policy is expressed directly as a supervised loss on preference pairs. DPO is simpler (2 models vs 4), more stable, and lower memory. PPO can potentially improve beyond the training distribution by generating new data online.

**Q: What is reward hacking?**
The policy finds unintended high-reward behaviors that don't reflect true human preferences. Example: if humans prefer longer responses (on average), an RLHF policy will pad responses with redundant text. The reward model overfit to length as a proxy for quality.

**Q: What is RLAIF and why would you use it over RLHF?**
RLAIF replaces human preference annotators with a strong LLM (like GPT-4). Vastly cheaper at scale — human annotation can cost $1–10 per comparison; LLM annotation costs fractions of a cent. Quality approaches human-level for many tasks. Limitation: the rater LLM's own biases are baked into the training signal.

**Q: What is the Bradley-Terry model?**
A statistical model for pairwise comparisons. $P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$. Training the reward model with this loss forces it to assign higher scores to preferred responses. The key property: only the score difference matters, not absolute values.

---

## 9. Cheat Sheet

```
RLHF PIPELINE
  Phase 1: SFT on high-quality demonstrations
  Phase 2: Train reward model on (prompt, win, lose) pairs
  Phase 3: PPO — maximize reward, penalize KL from SFT

REWARD MODEL LOSS (Bradley-Terry)
  L = -log σ(r(y_w) - r(y_l))
  Push r(y_w) > r(y_l) for all preference pairs

PPO OBJECTIVE
  L = E[r(x,y) - β·KL(π_θ || π_ref)]
  β controls reward vs drift tradeoff
  clip(ratio, 1-ε, 1+ε) prevents large policy steps

DPO OBJECTIVE
  L = -log σ(β·(log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))
  No separate reward model needed
  Stable supervised training

KEY HYPERPARAMETERS
  β:    KL penalty (0.01–0.1); lower = more reward, more risk
  ε:    PPO clip range (0.1–0.2); controls step size
  LR:   1e-5 for PPO (unstable), 1e-6 to 5e-7 for DPO

FAILURE MODES
  Reward hacking:    exploits RM blind spots
  Mode collapse:     always outputs same response
  Over-refusal:      too conservative (harmless > helpful)
  Sycophancy:        agrees with user regardless of truth
```

---

## Mini-Project: Preference Learning System

Build a complete preference learning pipeline: collect simulated preference data, train a reward model, train a policy via DPO, and compare the before/after behavior.

```python
# preference_learning.py
import numpy as np
import math
from collections import defaultdict


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


# ── Response quality simulator ─────────────────────────────────
class ResponseEnvironment:
    """
    Simulates a space of responses with known quality.
    Each response is a tuple of 5 feature values:
      0: relevance to prompt (0-1)
      1: factual accuracy (0-1)
      2: clarity/readability (0-1)
      3: conciseness (0-1, 1=concise)
      4: safety (0-1, 1=completely safe)
    """
    
    TRUE_WEIGHTS = np.array([0.3, 0.3, 0.2, 0.1, 0.1])  # ground truth quality
    
    def __init__(self, n_responses: int = 20, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.responses = rng.random((n_responses, 5))
        self.true_quality = self.responses @ self.TRUE_WEIGHTS
        self.n = n_responses
    
    def get_preference(self, i: int, j: int) -> int:
        """
        Human preference: returns index of preferred response.
        Probabilistic: better response preferred ~80% of time (humans are noisy).
        """
        q_i, q_j = self.true_quality[i], self.true_quality[j]
        # Bradley-Terry probability of i being preferred
        p_i = 1.0 / (1.0 + math.exp(-(q_i - q_j) * 5))
        return i if np.random.random() < p_i else j
    
    def collect_comparisons(self, n_comparisons: int, seed: int = 0) -> list[tuple[int, int, int]]:
        """
        Collect (i, j, preferred_idx) triples.
        Returns: list of (winner_idx, loser_idx, quality_diff)
        """
        rng = np.random.default_rng(seed)
        comparisons = []
        
        for _ in range(n_comparisons):
            i, j = rng.choice(self.n, size=2, replace=False)
            preferred = self.get_preference(int(i), int(j))
            winner = preferred
            loser = int(i) if preferred == int(j) else int(j)
            comparisons.append((winner, loser))
        
        return comparisons


# ── Reward model ───────────────────────────────────────────────
class RewardModel:
    def __init__(self, n_features: int = 5):
        rng = np.random.default_rng(42)
        self.weights = rng.standard_normal(n_features) * 0.01
        self.n_features = n_features
    
    def score(self, features: np.ndarray) -> float:
        return float(features @ self.weights)
    
    def train(self, env: ResponseEnvironment, comparisons: list[tuple[int, int]],
              lr: float = 0.1, epochs: int = 200) -> None:
        losses = []
        n = len(comparisons)
        
        for epoch in range(epochs):
            total_loss = 0.0
            grad = np.zeros(self.n_features)
            
            for win_idx, lose_idx in comparisons:
                x_w = env.responses[win_idx]
                x_l = env.responses[lose_idx]
                
                r_w = self.score(x_w)
                r_l = self.score(x_l)
                margin = r_w - r_l
                
                # Bradley-Terry loss
                total_loss -= math.log(sigmoid(margin) + 1e-9)
                
                # Gradient
                d = sigmoid(margin) - 1.0
                grad += d * (x_w - x_l)
            
            self.weights -= lr * grad / n
            losses.append(total_loss / n)
        
        print(f"  Final RM loss: {losses[-1]:.4f}")
    
    def accuracy(self, env: ResponseEnvironment, comparisons: list[tuple[int, int]]) -> float:
        correct = sum(
            1 for w, l in comparisons
            if self.score(env.responses[w]) > self.score(env.responses[l])
        )
        return correct / len(comparisons)
    
    def rank_responses(self, env: ResponseEnvironment) -> list[tuple[float, int]]:
        """Rank all responses by learned reward."""
        scored = [(self.score(env.responses[i]), i) for i in range(env.n)]
        return sorted(scored, reverse=True)


# ── Policy ─────────────────────────────────────────────────────
class Policy:
    """Probability distribution over response choices."""
    
    def __init__(self, n_responses: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.logits = rng.standard_normal(n_responses) * 0.1
        self.n = n_responses
    
    def probs(self) -> np.ndarray:
        l = self.logits - self.logits.max()
        e = np.exp(l)
        return e / e.sum()
    
    def log_prob(self, idx: int) -> float:
        return float(math.log(self.probs()[idx] + 1e-9))
    
    def sample(self) -> int:
        return int(np.random.choice(self.n, p=self.probs()))
    
    def top_k(self, k: int = 3) -> list[tuple[float, int]]:
        p = self.probs()
        top = np.argsort(p)[::-1][:k]
        return [(float(p[i]), int(i)) for i in top]
    
    def entropy(self) -> float:
        p = self.probs()
        return float(-np.sum(p * np.log(p + 1e-9)))
    
    def clone(self) -> "Policy":
        ref = Policy(self.n)
        ref.logits = self.logits.copy()
        return ref


# ── DPO trainer ────────────────────────────────────────────────
class DPOTrainer:
    def __init__(self, policy: Policy, reference: Policy, beta: float = 0.1, lr: float = 0.05):
        self.policy = policy
        self.reference = reference
        self.beta = beta
        self.lr = lr
    
    def step(self, comparisons: list[tuple[int, int]]) -> float:
        total_loss = 0.0
        total_grad = np.zeros(self.policy.n)
        
        for y_w, y_l in comparisons:
            log_ratio_w = self.policy.log_prob(y_w) - self.reference.log_prob(y_w)
            log_ratio_l = self.policy.log_prob(y_l) - self.reference.log_prob(y_l)
            
            margin = self.beta * (log_ratio_w - log_ratio_l)
            loss = -math.log(sigmoid(margin) + 1e-9)
            total_loss += loss
            
            d = sigmoid(margin) - 1.0
            probs = self.policy.probs()
            
            g_w = probs.copy(); g_w[y_w] -= 1.0
            g_l = probs.copy(); g_l[y_l] -= 1.0
            
            total_grad += self.beta * d * (g_w - g_l)
        
        self.policy.logits -= self.lr * total_grad / len(comparisons)
        return total_loss / len(comparisons)
    
    def train(self, comparisons: list[tuple[int, int]], epochs: int = 150) -> None:
        for epoch in range(epochs):
            loss = self.step(comparisons)
            if epoch % 30 == 0:
                print(f"  Epoch {epoch:3d}: DPO loss = {loss:.4f}")


# ── Evaluation ─────────────────────────────────────────────────
def evaluate_policy(policy: Policy, env: ResponseEnvironment,
                    rm: RewardModel, label: str) -> dict:
    """Measure how well a policy selects high-quality responses."""
    # Sample 200 responses and measure average quality
    samples = [policy.sample() for _ in range(200)]
    avg_true_quality = np.mean([env.true_quality[s] for s in samples])
    avg_rm_reward = np.mean([rm.score(env.responses[s]) for s in samples])
    
    top = policy.top_k(3)
    
    return {
        "label": label,
        "avg_true_quality": round(avg_true_quality, 4),
        "avg_rm_reward": round(avg_rm_reward, 4),
        "entropy": round(policy.entropy(), 4),
        "top_3_choices": [(round(p, 3), i) for p, i in top],
    }


def main():
    section("1. Setup Environment and Collect Preferences")
    env = ResponseEnvironment(n_responses=20, seed=42)
    print(f"Response quality range: [{env.true_quality.min():.3f}, {env.true_quality.max():.3f}]")
    print(f"Best response: idx={env.true_quality.argmax()}, quality={env.true_quality.max():.4f}")
    
    comparisons = env.collect_comparisons(n_comparisons=300, seed=7)
    print(f"Collected {len(comparisons)} preference comparisons")
    
    section("2. Train Reward Model")
    rm = RewardModel(n_features=5)
    rm.train(env, comparisons, lr=0.05, epochs=200)
    
    rm_acc = rm.accuracy(env, comparisons)
    print(f"Reward model accuracy: {rm_acc:.4f}")
    
    rm_ranking = rm.rank_responses(env)
    print("\nTop 5 responses by learned reward:")
    for score, idx in rm_ranking[:5]:
        print(f"  Response {idx:2d}: RM score={score:.4f}, true quality={env.true_quality[idx]:.4f}")
    
    # Spearman correlation between learned and true ranking
    rm_ranks = [r for _, r in rm_ranking]
    true_ranks = list(np.argsort(env.true_quality)[::-1])
    rank_corr = np.corrcoef(
        [rm_ranks.index(i) for i in range(env.n)],
        [true_ranks.index(i) for i in range(env.n)]
    )[0, 1]
    print(f"Rank correlation (RM vs true quality): {rank_corr:.4f}")
    
    section("3. Train Policy via DPO")
    policy = Policy(n_responses=20, seed=42)
    reference = policy.clone()
    
    print("Before DPO:")
    before = evaluate_policy(policy, env, rm, "before DPO")
    print(f"  Avg true quality: {before['avg_true_quality']:.4f}")
    print(f"  Top 3 choices: {before['top_3_choices']}")
    
    dpo = DPOTrainer(policy, reference, beta=0.1, lr=0.05)
    dpo.train(comparisons, epochs=150)
    
    section("4. Evaluation")
    after = evaluate_policy(policy, env, rm, "after DPO")
    
    print(f"{'Metric':<25} {'Before DPO':>12} {'After DPO':>12} {'Change':>10}")
    print("-" * 62)
    for key in ["avg_true_quality", "avg_rm_reward", "entropy"]:
        b = before[key]
        a = after[key]
        delta = a - b
        print(f"{key:<25} {b:>12.4f} {a:>12.4f} {delta:>+10.4f}")
    
    print(f"\nTop 3 after DPO: {after['top_3_choices']}")
    print(f"Best response (idx {env.true_quality.argmax()}) prob before: "
          f"{reference.probs()[env.true_quality.argmax()]:.4f}")
    print(f"Best response (idx {env.true_quality.argmax()}) prob after:  "
          f"{policy.probs()[env.true_quality.argmax()]:.4f}")
    
    section("5. KL Divergence from Reference")
    p = policy.probs()
    q = reference.probs()
    kl = float(np.sum(p * np.log((p + 1e-9) / (q + 1e-9))))
    print(f"KL(policy || reference) = {kl:.4f}")
    print("(Higher β would reduce this at the cost of lower reward)")


if __name__ == "__main__":
    main()
```
