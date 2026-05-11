"""
PPO (Proximal Policy Optimization) from scratch for RLHF.
Covers: clipped surrogate objective, value function, GAE advantage estimation,
        entropy bonus, KL penalty, full PPO update loop.
Uses a tabular MDP (not an LM) to demonstrate all mechanics without GPU deps.
"""

import numpy as np

rng = np.random.default_rng(42)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Sigmoid / Softmax ──────────────────────────────────────────────
def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x-m).sum(axis=-1, keepdims=True))


# ── Simple MDP Environment ────────────────────────────────────────
class GridMDP:
    """
    5-state linear grid: [0] [1] [2] [3] [4(GOAL)]
    Actions: 0=left, 1=right
    Reward: +1 for reaching state 4, else 0.
    Simulates the "response generation" step at the token level.
    """

    N_STATES  = 5
    N_ACTIONS = 2

    def reset(self):
        self.state = rng.integers(0, 4)   # start anywhere except goal
        return self.state

    def step(self, action):
        if action == 0:   # left
            self.state = max(0, self.state - 1)
        else:             # right
            self.state = min(4, self.state + 1)
        done   = self.state == 4
        reward = 1.0 if done else 0.0
        return self.state, reward, done


# ── Policy Network (tabular softmax) ─────────────────────────────
class Policy:
    """
    Tabular softmax policy: logits[state, action].
    Represents π_θ(action | state).
    """

    def __init__(self, n_states: int, n_actions: int):
        self.logits = rng.standard_normal((n_states, n_actions)) * 0.1

    def action_probs(self, state: int) -> np.ndarray:
        return softmax(self.logits[state])

    def log_prob(self, state: int, action: int) -> float:
        return float(log_softmax(self.logits[state])[action])

    def sample(self, state: int) -> int:
        probs = self.action_probs(state)
        return int(rng.choice(len(probs), p=probs))

    def entropy(self, state: int) -> float:
        probs = self.action_probs(state)
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    def kl_from(self, ref_policy: "Policy", state: int) -> float:
        p = self.action_probs(state)
        q = ref_policy.action_probs(state)
        return float(np.sum(p * np.log((p + 1e-10) / (q + 1e-10))))


# ── Value Function (tabular) ──────────────────────────────────────
class ValueFunction:
    def __init__(self, n_states: int):
        self.v = np.zeros(n_states)

    def predict(self, state: int) -> float:
        return float(self.v[state])

    def update(self, state: int, target: float, lr: float = 0.1):
        self.v[state] += lr * (target - self.v[state])


# ── Trajectory Collection ─────────────────────────────────────────
def collect_rollout(policy: Policy, env: GridMDP, n_steps: int = 50):
    """Collect a trajectory under the current policy."""
    trajectory = []
    state = env.reset()

    for _ in range(n_steps):
        action  = policy.sample(state)
        log_p   = policy.log_prob(state, action)
        next_s, reward, done = env.step(action)
        trajectory.append({
            "state": state, "action": action,
            "log_prob": log_p, "reward": reward,
            "done": done,
        })
        if done:
            state = env.reset()
        else:
            state = next_s

    return trajectory


# ── GAE Advantage Estimation ──────────────────────────────────────
def compute_gae(trajectory: list, value_fn: ValueFunction,
                gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
    """
    GAE: A_t = Σ_{k=0}^∞ (γλ)^k δ_{t+k}
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    """
    n = len(trajectory)
    advantages = np.zeros(n)
    gae = 0.0

    for t in reversed(range(n)):
        s  = trajectory[t]["state"]
        r  = trajectory[t]["reward"]
        d  = trajectory[t]["done"]
        Vs = value_fn.predict(s)

        if t + 1 < n and not d:
            Vs_next = value_fn.predict(trajectory[t+1]["state"])
        else:
            Vs_next = 0.0

        delta = r + gamma * Vs_next - Vs
        gae   = delta + gamma * lam * (0.0 if d else gae)
        advantages[t] = gae

    return advantages


# ── PPO Clipped Surrogate Loss ────────────────────────────────────
def ppo_loss(new_log_probs: np.ndarray, old_log_probs: np.ndarray,
             advantages: np.ndarray, epsilon: float = 0.2):
    """
    L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    r_t = exp(log π_new - log π_old)
    """
    ratios     = np.exp(new_log_probs - old_log_probs)
    surr1      = ratios * advantages
    surr2      = np.clip(ratios, 1 - epsilon, 1 + epsilon) * advantages
    clip_loss  = np.minimum(surr1, surr2)
    return float(clip_loss.mean()), ratios


# ── PPO Update Step ───────────────────────────────────────────────
def ppo_update(policy: Policy, ref_policy: Policy, value_fn: ValueFunction,
               trajectory: list, advantages: np.ndarray,
               lr_policy: float = 0.05, lr_value: float = 0.1,
               epsilon: float = 0.2, c1: float = 0.1, c2: float = 0.01,
               beta_kl: float = 0.05, n_epochs: int = 4) -> dict:
    """
    Full PPO update (multiple epochs over same batch).
    Returns metrics dict.
    """
    n = len(trajectory)
    old_log_probs = np.array([t["log_prob"] for t in trajectory])
    returns = advantages + np.array([value_fn.predict(t["state"]) for t in trajectory])

    metrics = {"pg_loss": [], "vf_loss": [], "kl": [], "entropy": []}

    for _ in range(n_epochs):
        new_log_probs = np.array([
            policy.log_prob(t["state"], t["action"]) for t in trajectory
        ])

        # Policy gradient loss (clipped)
        pg_l, ratios = ppo_loss(new_log_probs, old_log_probs, advantages, epsilon)

        # Value function loss (MSE)
        values = np.array([value_fn.predict(t["state"]) for t in trajectory])
        vf_l   = float(((values - returns)**2).mean())

        # Entropy bonus
        entropies = np.array([policy.entropy(t["state"]) for t in trajectory])
        ent = float(entropies.mean())

        # KL from reference policy
        kls = np.array([policy.kl_from(ref_policy, t["state"]) for t in trajectory])
        kl  = float(kls.mean())

        # Total loss (we maximize pg_l + c2*ent, minimize c1*vf_l + beta_kl*kl)
        # Gradient step: increase log-prob of high-advantage actions
        for i, t in enumerate(trajectory):
            s, a = t["state"], t["action"]
            adv   = advantages[i]
            r_t   = ratios[i]

            # Only update if ratio is not clipped (approximate gradient step)
            if (1 - epsilon) <= r_t <= (1 + epsilon):
                # Gradient of clipped objective ≈ advantage * indicator(in range)
                grad = adv * np.eye(policy.logits.shape[1])[a] - \
                       adv * policy.action_probs(s)
                # KL penalty gradient: push toward ref policy
                kl_grad = policy.action_probs(s) - ref_policy.action_probs(s)
                policy.logits[s] += lr_policy * (grad - beta_kl * kl_grad)

        # Value function update
        for i, t in enumerate(trajectory):
            value_fn.update(t["state"], returns[i], lr=lr_value)

        metrics["pg_loss"].append(pg_l)
        metrics["vf_loss"].append(vf_l)
        metrics["kl"].append(kl)
        metrics["entropy"].append(ent)

    return {k: np.mean(v) for k, v in metrics.items()}


# ── Win Rate Evaluation ────────────────────────────────────────────
def evaluate_policy(policy: Policy, env: GridMDP, n_episodes: int = 100) -> float:
    """Fraction of episodes that reach the goal within 10 steps."""
    successes = 0
    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(10):
            a = policy.sample(s)
            s, r, done = env.step(a)
            if done:
                successes += 1
                break
    return successes / n_episodes


def main():
    section("1. PPO CLIP OBJECTIVE VISUALIZATION")
    print("  L_CLIP = E[min(r·A, clip(r, 1-ε, 1+ε)·A)]")
    print("  where r = π_new/π_old (probability ratio)")
    print()
    eps = 0.2
    print(f"  {'r':>6} {'A':>6} {'L_CLIP':>10} {'Clipped?':>10}")
    print(f"  {'-'*40}")
    for r in [0.5, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0]:
        for A in [1.0, -1.0]:
            surr1 = r * A
            surr2 = np.clip(r, 1-eps, 1+eps) * A
            clip_l = min(surr1, surr2)
            clipped = abs(r - np.clip(r, 1-eps, 1+eps)) > 1e-6
            print(f"  {r:6.2f} {A:6.1f} {clip_l:10.4f} {str(clipped):>10}")

    section("2. ENVIRONMENT + POLICY SETUP")
    env = GridMDP()
    policy   = Policy(n_states=5, n_actions=2)
    ref_policy = Policy(n_states=5, n_actions=2)   # frozen SFT reference
    value_fn = ValueFunction(n_states=5)

    # Copy policy to reference (ref = initial policy)
    ref_policy.logits = policy.logits.copy()

    initial_wr = evaluate_policy(policy, env)
    print(f"  Initial win-rate (reach goal in 10 steps): {initial_wr:.3f}")

    section("3. PPO TRAINING LOOP")
    n_iterations = 40
    rollout_steps = 100
    history = []

    for it in range(n_iterations):
        trajectory = collect_rollout(policy, env, n_steps=rollout_steps)
        advantages  = compute_gae(trajectory, value_fn, gamma=0.99, lam=0.95)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = ppo_update(
            policy, ref_policy, value_fn, trajectory, advantages,
            lr_policy=0.05, lr_value=0.1, epsilon=0.2,
            c1=0.1, c2=0.01, beta_kl=0.05, n_epochs=4
        )

        win_rate = evaluate_policy(policy, env, n_episodes=50)
        metrics["win_rate"] = win_rate
        history.append(metrics)

        if it % 5 == 0 or it == n_iterations - 1:
            print(f"  Iter {it:3d}: win_rate={win_rate:.3f}  "
                  f"pg_loss={metrics['pg_loss']:.4f}  "
                  f"kl={metrics['kl']:.4f}  "
                  f"entropy={metrics['entropy']:.4f}")

    final_wr = evaluate_policy(policy, env, n_episodes=200)
    print(f"\n  Final win-rate: {final_wr:.3f}  (initial: {initial_wr:.3f})")

    section("4. KL DIVERGENCE ANALYSIS")
    all_kls = [h["kl"] for h in history]
    print(f"  KL divergence from reference policy over training:")
    print(f"    Initial: {all_kls[0]:.4f}")
    print(f"    Final:   {all_kls[-1]:.4f}")
    print(f"    Max:     {max(all_kls):.4f}")
    print(f"  → KL should remain bounded (no reward hacking if KL < 0.5)")

    # Check: does increasing beta_kl keep KL smaller?
    print(f"\n  Effect of beta_kl on KL:")
    for beta in [0.0, 0.05, 0.2, 1.0]:
        p = Policy(n_states=5, n_actions=2)
        vf = ValueFunction(n_states=5)
        ref = Policy(n_states=5, n_actions=2)
        ref.logits = p.logits.copy()
        kl_vals = []
        for _ in range(10):
            traj = collect_rollout(p, env, n_steps=50)
            adv  = compute_gae(traj, vf)
            adv  = (adv - adv.mean()) / (adv.std() + 1e-8)
            m    = ppo_update(p, ref, vf, traj, adv, beta_kl=beta, n_epochs=2)
            kl_vals.append(m["kl"])
        wr = evaluate_policy(p, env, n_episodes=100)
        print(f"    beta_kl={beta:.2f}: final_kl={kl_vals[-1]:.4f}  win_rate={wr:.3f}")

    section("5. ADVANTAGE ESTIMATION (GAE)")
    traj = collect_rollout(policy, env, n_steps=20)
    advantages = compute_gae(traj, value_fn, gamma=0.99, lam=0.95)
    print(f"  GAE advantages for 20-step rollout:")
    print(f"  {'t':>3} {'state':>6} {'action':>7} {'reward':>7} {'advantage':>10}")
    print(f"  {'-'*40}")
    for i, (t, a) in enumerate(zip(traj[:10], advantages[:10])):
        print(f"  {i:>3} {t['state']:>6} {t['action']:>7} {t['reward']:>7.1f} {a:>10.4f}")

    print(f"\n  GAE properties:")
    print(f"    lam=0: pure TD(0) estimates (low variance, high bias)")
    print(f"    lam=1: Monte Carlo returns (low bias, high variance)")
    print(f"    lam=0.95: standard RLHF setting (balances both)")

    section("6. PPO vs. KL-PENALTY COMPARISON")
    print("""
  Two variants of the RLHF RL objective:

  PPO-CLIP:     L = E[min(r·A, clip(r, 1-ε, 1+ε)·A)]
                Implicit trust region via clipping
                ε=0.2 standard; clip prevents large updates

  KL-Penalty:   L = E[r·A] - β·KL(π_new || π_old)
                Explicit KL penalty in the loss
                Adaptive β: increase if KL > target, decrease if KL < target/1.5

  PPO-CLIP is preferred in practice:
  - More stable training (KL-penalty β is hard to tune)
  - Better sample efficiency
  - Used in InstructGPT, Llama-2-chat, Gemini

  DPO replaces both:
  - No RL loop, no value function, no 4-model setup
  - Equivalent asymptotically under some conditions
  - Used in LLaMA-3, Mistral-7B-Instruct, Claude (partially)
""")


if __name__ == "__main__":
    main()
