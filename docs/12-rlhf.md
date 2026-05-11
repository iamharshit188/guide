# Module 12 — RLHF & Alignment

> **Run the code:**
> ```bash
> cd src/12-rlhf
> python3.14 reward_model.py
> python3.14 ppo_scratch.py
> python3.14 dpo.py
> python3.14 evaluate_alignment.py
> ```

---

## Prerequisites & Overview

**Time estimate:** 8–10 hours

| Prerequisite | From |
|-------------|------|
| Neural network backprop | Module 05 |
| Language model basics | Module 07 |
| Fine-tuning (LoRA, loss functions) | Module 09 |

**Before you start:**
- [ ] Understand cross-entropy loss and how to compute gradients through it
- [ ] Know what SFT (Supervised Fine-Tuning) means — training on (prompt, response) pairs
- [ ] Understand KL divergence: $D_{\text{KL}}(P \| Q) = \mathbb{E}_P[\log P/Q]$

**Module map:**

| Section | Core formula |
|---------|-------------|
| Reward modeling | Bradley-Terry: $P(y_w \succ y_l) = \sigma(r(x,y_w) - r(x,y_l))$ |
| PPO | Clip objective: $\mathcal{L}_{\text{CLIP}} = \mathbb{E}[\min(r_t A_t,\;\text{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]$ |
| KL penalty | $\mathcal{L}_{\text{total}} = -r(x,y) + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ |
| DPO | $\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta(r_\theta(x,y_w) - r_\theta(x,y_l)))]$ |
| Constitutional AI | RLAIF: model critiques → revision → preference labels → DPO |
| Win-rate evaluation | Human or LLM judge: fraction of responses preferred over baseline |

---

## RLHF Pipeline Overview

RLHF (Reinforcement Learning from Human Feedback) transforms a supervised language model into one that humans prefer. It has three stages:

```
Stage 1: SFT (Supervised Fine-Tuning)
    Train on high-quality (prompt, response) pairs
    → π_SFT: the starting policy

Stage 2: Reward Model Training
    Collect human preferences: (prompt, chosen, rejected)
    Train r_φ(x, y) to score preferred responses higher
    → r_φ: the reward signal

Stage 3: RL Optimization (PPO or DPO)
    Optimize π_θ to maximize E[r_φ(x, y)] - β·KL(π_θ || π_SFT)
    → π_RLHF: the aligned model
```

**Why KL penalty?** Without it, the policy would collapse to repeating the single highest-scoring response — reward hacking. The KL term keeps the model close to the SFT distribution.

---

## Reward Modeling

### Bradley-Terry Preference Model

Given a pair (chosen $y_w$, rejected $y_l$) for prompt $x$, the probability that $y_w$ is preferred:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) = \frac{e^{r_w}}{e^{r_w} + e^{r_l}}$$

where $\sigma$ is the sigmoid function. This is equivalent to logistic regression on the reward difference.

### Loss Function

Maximize log-likelihood of observed preferences:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\!\left[\log\sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

In code: `loss = -log_sigmoid(r_chosen - r_rejected)` = `BCELoss` on `(r_chosen - r_rejected)` with target 1.

### Architecture

Take a pre-trained LM, replace the language model head with a scalar regression head:

```
LM backbone (frozen or LoRA) → hidden_state(sequence) → mean_pool → Linear(d, 1) → scalar reward
```

```python
class RewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone                   # pre-trained LM
        self.head     = nn.Linear(backbone.d_model, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask).last_hidden_state
        pooled = hidden.mean(dim=1)               # mean pool over sequence
        return self.head(pooled).squeeze(-1)      # scalar per example

    def loss(self, r_chosen, r_rejected):
        diff = r_chosen - r_rejected
        return -torch.nn.functional.logsigmoid(diff).mean()
```

### Training Notes

- **Margin loss** — some implementations add a margin: $\text{loss} = -\log\sigma(r_w - r_l - m)$ where $m>0$ forces a minimum gap.
- **Length bias** — longer responses tend to get higher rewards (sycophancy). Normalize by dividing reward by $\sqrt{\text{seq\_len}}$ or fine-tune on length-balanced data.
- **Evaluation** — Reward model accuracy: fraction of preference pairs where $r_w > r_l$.

---

## PPO for RLHF

### The Policy Optimization Problem

Maximize expected reward subject to staying close to the reference policy:

$$J(\theta) = \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}\!\left[r_\phi(x,y)\right] - \beta\,D_{\text{KL}}(\pi_\theta(\cdot|x)\|\pi_{\text{ref}}(\cdot|x))$$

The KL term is computed token-by-token:

$$D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}}) = \sum_{t=1}^{T} \pi_\theta(y_t|x,y_{<t})\log\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\text{ref}}(y_t|x,y_{<t})}$$

### PPO Components for Language Models

PPO-RLHF uses four models simultaneously:

| Model | Role | Gradient? |
|-------|------|-----------|
| Actor $\pi_\theta$ | Generates responses; being trained | Yes |
| Critic $V_\psi$ | Estimates value of current state | Yes |
| Reference $\pi_{\text{ref}}$ | SFT model; provides KL baseline | Frozen |
| Reward $r_\phi$ | Scores complete responses | Frozen |

### Advantage Estimation

PPO uses Generalized Advantage Estimation (GAE) to reduce variance:

$$A_t^{\text{GAE}} = \sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}, \qquad \delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$$

For RLHF with LMs: $r_t = 0$ for all tokens except the last, where $r_T = r_\phi(x, y) - \beta\log\frac{\pi_\theta(y_T|x)}{\pi_{\text{ref}}(y_T|x)}$.

### Clipped Surrogate Objective

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)A_t,\;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x,y_{<t})}$ is the probability ratio.

**Intuition:** If the advantage $A_t > 0$ (action was better than baseline), we want to increase $\pi_\theta$. But clip at $1+\epsilon$ prevents overshooting. If $A_t < 0$, clip at $1-\epsilon$ prevents excessive probability decrease.

### Full PPO RLHF Loss

$$\mathcal{L}_{\text{total}} = -\mathcal{L}^{\text{CLIP}} + c_1\mathcal{L}^{\text{VF}} - c_2\mathcal{H}(\pi_\theta)$$

- $\mathcal{L}^{\text{VF}} = \mathbb{E}[(V_\psi(s_t) - V_t^{\text{target}})^2]$ — critic MSE loss
- $\mathcal{H}(\pi_\theta) = -\mathbb{E}[\log\pi_\theta]$ — entropy bonus (prevents collapse)
- $c_1 \approx 0.1$, $c_2 \approx 0.01$, $\epsilon \approx 0.2$ (standard hyperparameters)

### PPO Implementation Sketch

```python
def ppo_step(actor, critic, ref_model, reward_model, batch, beta=0.05, eps=0.2):
    prompts, responses = batch["prompts"], batch["responses"]

    with torch.no_grad():
        rewards    = reward_model(prompts, responses)          # scalar per response
        ref_logps  = ref_model.log_probs(prompts, responses)  # (B, T)
        old_logps  = actor.log_probs(prompts, responses)      # (B, T)
        values     = critic(prompts, responses)                # (B, T)

    # KL-penalized token-level rewards
    kl = old_logps - ref_logps    # (B, T); approximate KL
    token_rewards = -beta * kl
    token_rewards[:, -1] += rewards   # add scalar reward at last token

    # Compute advantages via GAE
    advantages = compute_gae(token_rewards, values, gamma=1.0, lam=0.95)
    returns    = advantages + values

    # PPO update (multiple epochs over same batch)
    for _ in range(4):
        new_logps  = actor.log_probs(prompts, responses)
        new_values = critic(prompts, responses)

        ratio  = (new_logps - old_logps).exp()   # (B, T)
        clip_ratio = ratio.clamp(1-eps, 1+eps)
        pg_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
        vf_loss = ((new_values - returns)**2).mean()
        entropy = -new_logps.mean()

        loss = pg_loss + 0.1*vf_loss - 0.01*entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        optimizer.step(); optimizer.zero_grad()

    return {"pg_loss": pg_loss.item(), "kl": kl.mean().item(), "reward": rewards.mean().item()}
```

---

## DPO — Direct Preference Optimization

### Why DPO Replaced PPO

PPO-RLHF requires 4 models simultaneously (actor, critic, reference, reward), making it memory-intensive and unstable. DPO (Rafailov et al., 2023) shows that the reward model can be implicitly parameterized by the policy itself, eliminating RL entirely.

### Derivation

Start from the RLHF optimization objective with KL constraint. The optimal policy has a closed form:

$$\pi^*(y|x) = \frac{\pi_{\text{ref}}(y|x)\exp\!\left(\frac{r(x,y)}{\beta}\right)}{Z(x)}$$

Rearrange to express the reward in terms of the policy:

$$r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta\log Z(x)$$

Substitute into the Bradley-Terry loss. $Z(x)$ cancels between $r(x,y_w)$ and $r(x,y_l)$:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

The term $\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} = \sum_t \log\pi_\theta(y_t|x,y_{<t}) - \log\pi_{\text{ref}}(y_t|x,y_{<t})$ is the token-level log-ratio, computable from both models in one forward pass each.

### DPO Implementation

```python
def dpo_loss(model, ref_model, chosen_ids, rejected_ids, beta=0.1):
    """
    chosen_ids, rejected_ids: (B, T) token id tensors
    """
    # Forward pass on policy
    chosen_logps  = model.token_log_probs(chosen_ids).sum(dim=-1)   # (B,)
    rejected_logps = model.token_log_probs(rejected_ids).sum(dim=-1)

    # Forward pass on frozen reference
    with torch.no_grad():
        ref_chosen_logps  = ref_model.token_log_probs(chosen_ids).sum(dim=-1)
        ref_rejected_logps = ref_model.token_log_probs(rejected_ids).sum(dim=-1)

    # Log-ratio differences
    chosen_rewards  = beta * (chosen_logps  - ref_chosen_logps)
    rejected_rewards = beta * (rejected_logps - ref_rejected_logps)

    # DPO loss (negative log-sigmoid of reward margin)
    loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # Diagnostics
    reward_margin = (chosen_rewards - rejected_rewards).mean().item()
    acc = (chosen_rewards > rejected_rewards).float().mean().item()  # implicit preference acc

    return loss, {"reward_margin": reward_margin, "accuracy": acc}
```

### DPO vs. PPO Comparison

| Dimension | PPO | DPO |
|-----------|-----|-----|
| Models needed | 4 (actor, critic, ref, reward) | 2 (policy, ref) |
| Stability | Sensitive to hyperparameters (clip ε, KL β) | More stable |
| Sample efficiency | Generates new responses per step | Uses existing preference data |
| Flexibility | Can incorporate any reward signal | Limited to preference pairs |
| Production use | GPT-4 initial, older systems | Claude, LLaMA-3, modern systems |
| Memory | 4× base model | 2× base model |

### DPO Variants

| Variant | Change | When to use |
|---------|--------|-------------|
| **IPO** (Identity Preference Optimization) | Replaces log-sigmoid with MSE | Avoids degenerate solutions when $P(y_w\succ y_l)=1$ |
| **KTO** (Kahneman-Tversky Opt.) | Per-response labels (good/bad), not pairs | When you have binary feedback, not pairwise |
| **SimPO** | No reference model; uses length-normalized log-prob | Simpler, less memory |

---

## KL Penalty

The KL term prevents reward hacking. Without it, the policy learns to exploit reward model blind spots:

$$\text{KL budget} = \mathbb{E}_{x,y\sim\pi_\theta}\!\left[\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right]$$

In practice, $\beta$ is tuned to keep KL in a target range (e.g., 0.1–2 nats). Too small $\beta$ → reward hacking. Too large $\beta$ → no improvement over SFT.

**Monitoring:** Log KL divergence per training step. If KL spikes and reward also spikes, you have reward hacking. Reduce $\beta$ or add reward normalization.

---

## Constitutional AI (RLAIF)

Constitutional AI (Anthropic, 2022) replaces human preference labelers with a model-generated critique loop:

```
1. Generate responses to potentially harmful prompts
2. Model critiques its own responses using a set of principles (the "constitution")
3. Model revises its responses based on the critique
4. Use (original, revised) pairs or model-judged preference labels for DPO
```

This is RLAIF (RL from AI Feedback): a stronger model (or the model itself) generates preference labels instead of humans. Scales better, cheaper, but depends on labeler model quality.

**Constitution example principles:**
- "Choose the response that is most helpful, harmless, and honest."
- "Choose the response that is less likely to be used to commit harm."
- "Choose the response that better explains the reasoning step by step."

---

## Evaluation

### Win-Rate

Sample N responses from the aligned model $\pi_\theta$ and baseline $\pi_{\text{SFT}}$ for the same prompts. An LLM judge (or human) decides which is preferred:

$$\text{Win-rate} = \frac{\#(\pi_\theta \text{ preferred over } \pi_{\text{SFT}})}{N}$$

Win-rate > 50% means the aligned model is better than baseline. Typical good alignment: 65–75% win-rate.

### MT-Bench

MT-Bench (Zheng et al., 2023) uses GPT-4 as a judge to score responses 1–10 across 8 categories (coding, math, reasoning, roleplay, writing, extraction, STEM, humanities). Each category has 10 multi-turn questions.

```python
MT_BENCH_CATEGORIES = [
    "writing", "roleplay", "extraction", "reasoning",
    "math", "coding", "stem", "humanities",
]

def mt_bench_score(judge_llm, model, questions_by_category):
    scores = {}
    for category, questions in questions_by_category.items():
        cat_scores = []
        for q1, q2 in questions:   # 2-turn: q1 = first turn, q2 = follow-up
            r1 = model(q1)
            r2 = model(q2, context=r1)
            score = judge_llm(
                f"Rate this 2-turn response 1-10:\n{r1}\n{r2}\nScore:",
                max_tokens=5
            )
            cat_scores.append(float(score))
        scores[category] = sum(cat_scores) / len(cat_scores)
    return scores, sum(scores.values()) / len(scores)
```

### Reward Model Accuracy

On a held-out preference test set: fraction of pairs where $r_\phi(x, y_w) > r_\phi(x, y_l)$.

$$\text{RM Accuracy} = \frac{1}{N}\sum_i \mathbf{1}[r_\phi(x_i, y_w^i) > r_\phi(x_i, y_l^i)]$$

Well-trained reward models achieve 70–80% accuracy on human preference test sets.

---

## Interview Q&A

**Q: Why does RLHF need a KL penalty between the policy and reference model?**
**A:** Without the KL term, the policy optimizes purely for the reward model signal and quickly finds reward model blind spots — sequences that score high but are pathological (e.g., repetitive gibberish, sycophantic statements). The KL penalty $\beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ keeps the policy from drifting too far from the SFT distribution. $\beta$ is the only critical hyperparameter: too small → reward hacking; too large → no improvement over SFT. Typical values: $\beta = 0.01$–$0.2$.

**Q: Derive why DPO eliminates the need for an explicit reward model.**
**A:** Start from the optimal RLHF solution: $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$. Solving for $r$: $r(x,y) = \beta\log(\pi^*(y|x)/\pi_{\text{ref}}(y|x)) + \beta\log Z(x)$. Plugging into the Bradley-Terry loss, the partition $Z(x)$ cancels between chosen and rejected. What remains is the DPO loss, which only requires $\pi_\theta$ and $\pi_{\text{ref}}$ forward passes — no separate reward model.

**Q: What is the clip mechanism in PPO and why is it necessary?**
**A:** The clip in PPO limits how much the probability ratio $r_t = \pi_\theta / \pi_{\text{old}}$ can deviate from 1. Without clipping, a large gradient update could move the policy far from the data distribution used to compute advantages, making the advantage estimates invalid (they assumed the old policy). $\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$ ensures that even with a large gradient signal, the actual policy update is bounded. This is what makes PPO "proximal" — it's a trust-region method without the expensive KL constraint projection.

**Q: What is reward hacking and how do you detect it?**
**A:** Reward hacking occurs when the policy finds inputs that maximize the reward model score without actually being helpful or harmless. Example: a reward model trained to prefer longer responses → policy generates verbose, padded text. Symptoms: (1) reward increases but win-rate (human or LLM judge) does not, (2) KL divergence from reference spikes, (3) perplexity under the reference model increases dramatically. Prevention: regularize with KL penalty, train reward model on diverse data, use ensemble reward models.

**Q: How does DPO handle the case where the preference data has ties or ambiguous labels?**
**A:** Standard DPO assumes all labels are correct ($y_w$ is always preferred). With noisy or ambiguous labels, DPO loss still backpropagates in the "wrong" direction for mislabeled pairs. Mitigations: (1) confidence weighting — scale each loss term by the human rater's confidence score, (2) IPO (Identity Preference Optimization) — uses MSE instead of log-sigmoid, which is less sensitive to overconfident labels near 0/1, (3) data filtering — remove pairs with low inter-annotator agreement.

**Q: How is win-rate different from reward model accuracy?**
**A:** Reward model accuracy measures how well $r_\phi$ predicts human preferences on a held-out pair dataset — it's a property of the reward model, not the final policy. Win-rate measures whether the final aligned policy $\pi_\theta$ generates responses humans prefer over the SFT baseline — it evaluates the complete RLHF pipeline end-to-end. A model can have a high-accuracy reward model but poor win-rate (if PPO/DPO overfit to reward hacking), or a lower-accuracy RM but good win-rate (if the RM captures the most important preference dimensions).

**Q: What is Constitutional AI and how does it differ from standard RLHF?**
**A:** Standard RLHF uses human annotators to produce preference labels, which is expensive and slow. Constitutional AI replaces human labelers with an LLM that applies a set of written principles (the "constitution") to generate (critique, revision) pairs or preference labels. This is RLAIF — RL from AI Feedback. The LLM judge first critiques a response based on the constitution, then generates an improved version, and either uses this as the "chosen" response (for SFT) or has the critic LLM pick between two responses (for DPO). It scales better than human annotation but depends heavily on the labeler LLM's quality.

**Q: Why is $\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ computed as a sum of token log-probabilities?**
**A:** Both $\log\pi_\theta(y|x)$ and $\log\pi_{\text{ref}}(y|x)$ factor over the token sequence by the chain rule: $\log\pi(y|x) = \sum_{t=1}^{T}\log\pi(y_t|x, y_{<t})$. The log-ratio therefore decomposes as $\sum_t [\log\pi_\theta(y_t|x,y_{<t}) - \log\pi_{\text{ref}}(y_t|x,y_{<t})]$ — a sum of per-token differences. This is exactly what autoregressive LMs output (per-token logits → log-softmax → selected token log-prob), making the computation efficient: one forward pass through each model produces all token log-probs.

---

## Resources

**Papers:**
- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) — Stiennon et al., 2020 (original RLHF for LMs)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022 (InstructGPT / ChatGPT)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al., 2022
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023
- [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) — Askell et al., 2021

**Books:**
- *AI Alignment* — Victoria Krakovna — distill.pub series on reward hacking and specification gaming

---

*Next: [Module 13 — Multimodal Models](13-multimodal.md)*
