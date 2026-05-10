# Module 09 — Fine-Tuning (LoRA / QLoRA)

> **Run:**
> ```bash
> cd src/09-finetuning
> python3.14 lora_theory.py
> python3.14 prepare_dataset.py
> python3.14 train_lora.py
> python3.14 train_qlora.py
> python3.14 evaluate.py
> python3.14 merge_push.py
> ```

---

## 1. Why Fine-Tune?

Pre-trained LLMs encode general world knowledge but underperform on:
- **Domain-specific vocabulary** (medical, legal, code dialects)
- **Task-specific output format** (JSON, structured extraction)
- **Alignment** (instruction following, tone, refusal behaviour)

| Approach | Trainable Params | Memory | Speed | When to Use |
|----------|-----------------|--------|-------|-------------|
| Full fine-tuning | 100% of $\theta$ | Very high | Slow | Small models, abundant GPU |
| Adapter layers | ~1–3% | Moderate | Fast | Moderate resources |
| LoRA | ~0.1–1% | Low | Fast | Most practical use |
| QLoRA | ~0.1–1% | Very low | Moderate | Consumer GPU (24 GB) |
| Prompt tuning | <0.01% | Minimal | Very fast | Frozen model |

Full fine-tuning: every weight matrix $W \in \mathbb{R}^{m \times n}$ is updated → total trainable parameters $= \sum_{l} m_l n_l$.

For LLaMA-2-7B: $\approx 7 \times 10^9$ parameters, requiring $>$112 GB GPU RAM for Adam (weights + gradients + 2 moment tensors, all in fp32).

---

## 2. LoRA — Low-Rank Adaptation

### 2.1 Motivation

**Hypothesis (Aghajanyan et al., 2020):** Pre-trained weight matrices have low intrinsic rank. The weight updates during fine-tuning also lie in a low-dimensional subspace.

### 2.2 Formulation

For a pre-trained weight matrix $W_0 \in \mathbb{R}^{m \times n}$, LoRA parameterises the update as:

$$W' = W_0 + \Delta W = W_0 + BA$$

where:
- $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$
- $r \ll \min(m, n)$ is the **rank** hyperparameter

**Forward pass:**

$$h = W' x = W_0 x + \frac{\alpha}{r} B A x$$

The scaling factor $\alpha/r$ (where $\alpha$ = `lora_alpha`) controls update magnitude independently of $r$, allowing rank changes without LR retuning.

### 2.3 Initialisation

$$A \sim \mathcal{N}(0, \sigma^2), \quad B = \mathbf{0}$$

At step 0: $\Delta W = BA = B \cdot 0 = 0$. The model starts identical to the pre-trained checkpoint.

### 2.4 Parameter Count

| Config | Parameters | Formula |
|--------|-----------|---------|
| Full FT | $mn$ | All weights updated |
| LoRA | $r(m+n)$ | Only $A, B$ trained |
| Reduction | $mn / r(m+n)$ | E.g., $4096 \times 4096, r=8$: $\approx 256\times$ |

For a single attention projection $W_Q \in \mathbb{R}^{4096 \times 4096}$, $r=8$:

$$\text{LoRA params} = 8(4096 + 4096) = 65{,}536 \quad \text{vs} \quad 4096^2 = 16{,}777{,}216$$

**Reduction: 256×**

### 2.5 Which Layers to Adapt

LoRA is typically applied to attention projection matrices: $W_Q, W_K, W_V, W_O$ (and sometimes $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$ in FFN).

**Why not all layers?** Embedding and LM-head matrices are large and task-specific; adapting them costs more without proportional benefit.

### 2.6 Rank Selection

| Rank $r$ | Params (per 4096×4096 layer) | Typical Use |
|----------|------------------------------|-------------|
| 4 | 32,768 | Minimal adaptation, low data |
| 8 | 65,536 | General-purpose (**default**) |
| 16 | 131,072 | Complex tasks |
| 64 | 524,288 | Near full-FT quality |

Increasing $r$ beyond 64 rarely helps; the model's update subspace is low-rank by hypothesis.

### 2.7 Adapter Merging

After training, adapters can be merged into the base weights with **zero inference overhead**:

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$$

No extra computation at inference time — unlike adapter architectures that add serial bottleneck layers.

---

## 3. QLoRA — Quantised LoRA

### 3.1 Stack Overview

```
Base model weights  →  4-bit NF4 quantisation  (frozen)
                    +  LoRA adapters in BF16   (trainable)
                    +  Paged Adam optimiser     (CPU offload)
```

QLoRA (Dettmers et al., 2023) makes 65B model fine-tuning feasible on a single 48 GB GPU.

### 3.2 NF4 Data Type

Normal Float 4-bit (NF4) quantisation exploits the empirical observation that pre-trained neural network weights are approximately Gaussian.

**Construction:** Place $2^4 = 16$ quantisation levels at the **quantiles** of $\mathcal{N}(0,1)$:

$$q_i = \Phi^{-1}\left(\frac{i}{15}\right), \quad i = 0, 1, \ldots, 15$$

where $\Phi^{-1}$ is the standard normal inverse CDF (probit function).

The quantisation levels are then renormalised to $[-1, 1]$ and stored as a lookup table.

**Why quantiles?** Uniform quantisation wastes precision near the tails (rare values) and undershoots near the mode (frequent values). NF4 allocates equal probability mass per bin → minimises expected quantisation error for normally distributed weights.

**Quantisation of weight tensor $W$:**

$$W_{NF4} = \text{round}\left(\frac{W}{\max(|W|)} \cdot 7\right) \quad \text{(conceptually)}$$

Then dequantise to BF16 before the matrix multiply:

$$W_{BF16} = \text{lookup}(W_{NF4}) \cdot \text{absmax}$$

### 3.3 Double Quantisation

The quantisation constants (one `absmax` per 64-weight block) themselves consume memory:

$$\text{absmax memory} = \frac{N}{64} \times 32 \text{ bits} = 0.5 \text{ bits/weight}$$

Double quantisation quantises these constants further with 8-bit quantisation, reducing to $\approx 0.127$ bits/weight for constants — saving $\approx 0.37$ GB on a 7B model.

### 3.4 Paged Optimisers

NVIDIA unified memory allows CPU RAM to act as overflow for GPU optimizer states (Adam's $m_t$ and $v_t$ tensors). Paged optimisers swap state pages to CPU during memory spikes (e.g., long sequences), preventing OOM crashes.

### 3.5 Memory Comparison

| Setup | Precision | GPU RAM (7B model) |
|-------|-----------|-------------------|
| Full fine-tuning | FP32 | ~112 GB |
| Full fine-tuning | BF16 | ~56 GB |
| LoRA | BF16 base + BF16 adapters | ~28 GB |
| QLoRA | NF4 base + BF16 adapters | ~8 GB |

---

## 4. Dataset Preparation

### 4.1 Instruction Formats

**Alpaca format:**
```
Below is an instruction that describes a task.

### Instruction:
{instruction}

### Response:
{output}
```

**ChatML format (used by Mistral, LLaMA-3):**
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
```

### 4.2 Tokenisation

Decoder-only models are trained with **left-to-right causal LM**. The training signal comes only from the **response** tokens (instruction tokens are masked with `-100` in `labels`):

$$\mathcal{L} = -\frac{1}{|y|}\sum_{t=1}^{|y|} \log p_\theta(y_t \,|\, x, y_{<t})$$

### 4.3 Data Collator

For decoder-only models, sequences are **left-padded** to uniform length within a batch (or right-padded with causal masking). The `DataCollatorForSeq2Seq` with `padding=True` handles this automatically.

**Key fields in a batch:**
| Field | Shape | Description |
|-------|-------|-------------|
| `input_ids` | $(B, T)$ | Token indices |
| `attention_mask` | $(B, T)$ | 1 = real token, 0 = pad |
| `labels` | $(B, T)$ | Targets; `-100` masks instruction tokens |

---

## 5. Training Loop

### 5.1 SFTTrainer (TRL)

```python
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622%

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=lora_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
```

### 5.2 Gradient Flow Through LoRA

During backprop, gradients flow through $h = W_0 x + \frac{\alpha}{r} BAx$:

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^\top \frac{\partial \mathcal{L}}{\partial h} x^\top$$

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal{L}}{\partial h} (Ax)^\top$$

$W_0$ receives no gradient update (`.requires_grad = False`).

### 5.3 Hyperparameters

| Param | Typical Value | Effect |
|-------|--------------|--------|
| `r` | 8–16 | Capacity; higher = more expressive |
| `lora_alpha` | 16–32 | Scale factor $\alpha/r$; keep $\alpha = 2r$ |
| `lora_dropout` | 0.05 | Regularisation |
| `target_modules` | q, v projections | Which layers get adapters |
| `learning_rate` | 2e-4 | Higher than full FT (only adapters update) |
| `batch_size` | 4–8 | Per GPU; gradient accumulate to effective 32–128 |
| `max_seq_length` | 2048 | Depends on task; longer = more GPU memory |

---

## 6. Evaluation Metrics

### 6.1 Perplexity

$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(w_t \mid w_{<t})\right)$$

Perplexity = exponential of average cross-entropy per token. Lower is better. **Not comparable across tokenisers** (different vocab sizes give different PPL scales).

**Interpretation:** PPL = 10 means the model is as uncertain as if it uniformly distributed probability over 10 choices at each step.

### 6.2 BLEU

BLEU (Bilingual Evaluation Understudy) measures n-gram precision between hypothesis and reference:

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

**Modified n-gram precision** (clips multiple matches to reference count):

$$p_n = \frac{\sum_{\text{ngrams}} \min(\text{count}_{\text{hyp}}, \text{count}_{\text{ref}})}{\sum_{\text{ngrams}} \text{count}_{\text{hyp}}}$$

**Brevity penalty** (penalises too-short hypotheses):

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \leq r \end{cases}$$

where $c$ = hypothesis length, $r$ = reference length. Uniform weights $w_n = 1/N$.

BLEU-4 ($N=4$) is the standard; correlates poorly with human judgement for open-ended generation.

### 6.3 ROUGE-L

ROUGE-L uses the **Longest Common Subsequence (LCS)**:

$$\text{LCS}(X, Y) = \max \text{ length of common subsequence}$$

$$P_{lcs} = \frac{\text{LCS}(hyp, ref)}{|hyp|}, \quad R_{lcs} = \frac{\text{LCS}(hyp, ref)}{|ref|}$$

$$\text{ROUGE-L} = \frac{(1+\beta^2) P_{lcs} R_{lcs}}{R_{lcs} + \beta^2 P_{lcs}} \quad (\beta=1 \text{ for F1})$$

LCS DP recurrence:

$$L[i][j] = \begin{cases} L[i-1][j-1] + 1 & \text{if } X[i] = Y[j] \\ \max(L[i-1][j], L[i][j-1]) & \text{otherwise} \end{cases}$$

### 6.4 Metric Comparison

| Metric | Measures | Reference Needed | Limitation |
|--------|----------|-----------------|------------|
| Perplexity | Token probability | No (uses model) | Tokeniser-dependent |
| BLEU | N-gram overlap | Yes | Ignores semantics |
| ROUGE-L | Subsequence overlap | Yes | Length-sensitive |
| BERTScore | Semantic similarity | Yes | Slow, requires BERT |
| Human eval | Overall quality | Yes | Expensive, subjective |

---

## 7. Adapter Merging & Deployment

### 7.1 Weight Merge Math

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} \cdot B \cdot A$$

Merge is lossless for LoRA (adapter is additive). After merge, the model behaves identically to the trained PEFT model but requires **no PEFT library at inference time**.

### 7.2 PEFT Merge API

```python
merged_model = model.merge_and_unload()  # returns nn.Module, not PeftModel
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

### 7.3 HuggingFace Hub Push

```python
merged_model.push_to_hub("username/model-name")
tokenizer.push_to_hub("username/model-name")
```

Adapter-only push (without merging):

```python
model.push_to_hub("username/adapter-name")  # saves adapter_config.json + adapter_model.bin only
```

### 7.4 Adapter Inspection

```json
// adapter_config.json
{
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj"],
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "base_model_name_or_path": "meta-llama/Llama-2-7b-hf"
}
```

---

## 8. Interview Q&A

**Q: Why does LoRA initialise $B=0$?**
So that $\Delta W = BA = 0$ at the start of training. This means the model begins at the pre-trained checkpoint, not at a random perturbation. Training from a stable starting point is critical for convergence.

**Q: What does the $\alpha/r$ scaling factor do?**
It decouples the effective learning rate of the LoRA update from the rank $r$. Without it, increasing $r$ would also increase the magnitude of $\Delta W$, requiring LR retuning. With $\alpha$ fixed, changing $r$ only changes capacity, not scale.

**Q: Why NF4 instead of INT4 for QLoRA?**
INT4 places quantisation levels uniformly across $[-1, 1]$. Neural network weights follow approximately $\mathcal{N}(0, \sigma^2)$, so most weight values cluster near zero. NF4 places more quantisation levels near the mode of the distribution (quantiles of $\mathcal{N}(0,1)$), minimising expected quantisation error under the assumption that weights are Gaussian.

**Q: What is double quantisation?**
NF4 stores one 32-bit `absmax` constant per block of 64 weights, costing 0.5 bits/weight. QLoRA quantises these constants themselves with 8-bit quantisation, reducing the constant overhead to ~0.127 bits/weight. Saves ~0.37 GB per 7B model.

**Q: Why can't you train the NF4 weights directly?**
Quantisation is non-differentiable — quantised values are integers with no useful gradient. The base weights are frozen and dequantised only during the forward pass (compute-time conversion to BF16). Gradients flow only through the LoRA adapters (which are in BF16).

**Q: What is the intrinsic dimension hypothesis?**
Aghajanyan et al. (2020) showed that fine-tuning objective surfaces have a low intrinsic dimensionality — most tasks can be solved by optimising in a subspace of 100–1000 dimensions, regardless of model size. LoRA operationalises this: the weight update $\Delta W = BA$ lives in a rank-$r$ subspace.

---

[← Module 08 — RAG Chatbot](08-rag.md)
