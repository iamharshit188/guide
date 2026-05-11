# Project 09 — Domain-Specific Instruction Tuner

> **Difficulty:** Advanced · **Module:** 09 — Fine-Tuning (LoRA/QLoRA)
> **Skills:** Dataset extraction, Alpaca format, LoRA theory, NF4 QLoRA simulation, PPL/BLEU/ROUGE, adapter merging

---

## What You'll Build

Extract Q&A pairs from the module guides → format as Alpaca JSON → BPE tokenize → simulate LoRA ($r=8$, $\alpha=16$) gradient updates with NumPy (always runs) + real PEFT/QLoRA (graceful skip) → evaluate PPL, BLEU-4, ROUGE-L before/after → merge adapter → print dataset stats, training curve, metrics table, merged model size.

---

## Skills Exercised

- Q&A extraction from structured Markdown (regex on `###` Q&A sections)
- Alpaca instruction format: `{"instruction":..., "input":"", "output":...}`
- Label masking: instruction tokens get `-100` (ignored in loss)
- LoRA math: $W' = W_0 + \frac{\alpha}{r}BA$, $B$ initialized to 0, $A \sim \mathcal{N}(0, \sigma^2)$
- NF4 quantization simulation: quantile-based binning of $\mathcal{N}(0,1)$
- PPL = $\exp(\text{mean cross-entropy})$; BLEU-4 from scratch; ROUGE-L LCS
- Adapter merging: $W_{\text{merged}} = W_0 + \frac{\alpha}{r}BA$

---

## Approach

### Phase 1 — Dataset extraction
```
for module in docs/*.md:
    find all Q&A blocks (### Q: ... **A:** ...)
    parse question + answer text
    format as Alpaca:
        {"instruction": question, "input": "", "output": answer,
         "source": module_file}

# also create synthetic pairs:
for each section header:
    instruction: "Explain {section_topic} in the context of ML interviews"
    output: first 3 paragraphs of section

print dataset stats:
    total examples, min/max/avg output length, examples per module
```

### Phase 2 — BPE tokenization
```
use Module 07 BPE tokenizer trained on the docs corpus
  (or fall back to char-level if BPE not available)

for each example:
    instruction_ids = tokenizer.encode(instruction)
    output_ids = tokenizer.encode(output)
    input_ids = instruction_ids + output_ids
    labels = [-100]*len(instruction_ids) + output_ids   # mask instruction
    pad/truncate to max_len=512
```

### Phase 3 — LoRA simulation (NumPy, always runs)
```
# simulate a single linear layer W0 (d_out × d_in)
W0 = rng.normal(0, 0.02, (d_out, d_in))   # frozen

# LoRA adapter
r, alpha = 8, 16
A = rng.normal(0, 0.01, (r, d_in))         # trainable
B = np.zeros((d_out, r))                    # trainable, init 0
scale = alpha / r                           # = 2.0

# forward: W_eff = W0 + scale * B @ A
# compute loss on a batch of token embeddings
# backward through B @ A only (W0 frozen)
# Adam update of A, B

for step in range(n_steps):
    x_batch = sample_batch(tokenized_data)
    logits = x_batch @ (W0 + scale * B @ A).T
    loss = cross_entropy(logits, targets)
    # backprop dL/dB, dL/dA
    print(f"step {step}: loss={loss:.4f}")
```

### Phase 4 — NF4 quantization simulation
```
# NF4: 16 quantization levels based on N(0,1) quantiles
import scipy.stats as st
quantiles = np.array([-1] + [st.norm.ppf(i/16) for i in range(1,16)] + [1])
centers = (quantiles[:-1] + quantiles[1:]) / 2   # 16 NF4 codes

def quantize_nf4(W):
    # scale W to [-1, 1]
    absmax = np.abs(W).max()
    W_norm = W / absmax
    # find nearest NF4 center for each weight
    codes = np.argmin(np.abs(W_norm[:,:,None] - centers), axis=2)
    W_quant = centers[codes] * absmax
    rmse = np.sqrt(np.mean((W - W_quant)**2))
    return W_quant, codes, rmse

print("INT4 RMSE:", quantize_int4(W0)[2])
print("NF4  RMSE:", quantize_nf4(W0)[2])   # should be lower
```

### Phase 5 — PEFT/QLoRA (graceful skip)
```python
try:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    # ... real QLoRA training
except ImportError:
    print("PEFT/BitsAndBytes not available — skipping real training")
    print("NumPy simulation results above are the primary output")
```

### Phase 6 — Evaluation + merge
```
# Evaluate before and after LoRA fine-tuning
for split in ["before", "after"]:
    ppl = compute_ppl(model, val_data)                    # exp(mean CE)
    bleu = bleu4_from_scratch(references, hypotheses)
    rouge = rouge_l_from_scratch(references, hypotheses)  # LCS-based
    print(f"{split}: PPL={ppl:.2f} BLEU-4={bleu:.4f} ROUGE-L={rouge:.4f}")

# merge adapter
W_merged = W0 + scale * B @ A
print(f"Merged model size: {W_merged.nbytes / 1024**2:.2f} MB")

# latency comparison
base_latency    = time_inference(W0, test_batch)
merged_latency  = time_inference(W_merged, test_batch)
adapter_latency = time_inference(W0, test_batch, adapter=(A, B, scale))
print(f"Base: {base_latency:.2f}ms  Merged: {merged_latency:.2f}ms  Adapter: {adapter_latency:.2f}ms")
```

---

## Checkpoints

| Phase | What correct output looks like |
|-------|-------------------------------|
| 1 | ≥50 examples across 9 modules; avg output length > 100 tokens |
| 2 | `labels[i] == -100` for all instruction positions; padding visible |
| 3 | Loss decreases over steps; B stays near-zero early (correct LoRA init) |
| 4 | NF4 RMSE < INT4 RMSE for any random weight matrix (by construction) |
| 5 | Either real PEFT results or clear "skipping" message with NumPy results |
| 6 | PPL decreases after fine-tuning; merged latency ≈ base latency (single matmul) |

---

## Extensions

1. **DPO loss** — implement Direct Preference Optimization: given (prompt, chosen, rejected) triplets, compute $\mathcal{L}_{\text{DPO}} = -\log\sigma(\beta(\log\pi_\theta(\text{chosen}) - \log\pi_{\text{ref}}(\text{chosen})) - \beta(\log\pi_\theta(\text{rejected}) - \log\pi_{\text{ref}}(\text{rejected})))$
2. **Different target modules** — in real PEFT, try LoRA on q_proj only vs. q_proj+v_proj vs. all linear layers; compare trainable parameter count and final PPL.
3. **Rank sensitivity analysis** — run LoRA at r∈{1,2,4,8,16,32}; plot trainable params vs. final loss; find the knee of the curve.

---

## Hints

<details><summary>Hint 1 — Extracting Q&A from module guides</summary>
The Q&A sections follow a pattern like <code>**Q:** question text\n\n**A:** answer text</code>. Use regex: <code>re.findall(r'\*\*Q:\*\* (.+?)\n+\*\*A:\*\* (.+?)(?=\n\n\*\*Q:|$)', text, re.DOTALL)</code>
</details>

<details><summary>Hint 2 — LoRA backward through B @ A</summary>
dL/dB = dL/dlogits @ x_batch @ A.T * scale<br>
dL/dA = B.T @ dL/dlogits @ x_batch * scale<br>
W0 gradient is computed but discarded (frozen).
</details>

<details><summary>Hint 3 — BLEU-4 from scratch</summary>
For n=1..4: count n-gram matches between hypothesis and reference, clip by reference count. Geometric mean of 4 n-gram precisions × brevity penalty. BP = exp(1 - ref_len/hyp_len) if hyp_len < ref_len, else 1.
</details>

<details><summary>Hint 4 — ROUGE-L LCS</summary>
LCS length via dynamic programming: dp[i][j] = dp[i-1][j-1]+1 if tokens match, else max(dp[i-1][j], dp[i][j-1]).<br>
ROUGE-L = F1 of LCS: precision = lcs/|hyp|, recall = lcs/|ref|, F1 = 2*P*R/(P+R).
</details>

<details><summary>Hint 5 — Why merged latency ≈ base latency</summary>
After merging, W_merged is just a single matrix — same shape as W0. Inference does one matmul. The adapter path (W0 + BA) does two matmuls. Merging eliminates the adapter overhead, which is why production deployments merge before serving.
</details>

---

*Back to [Module 09 — Fine-Tuning (LoRA/QLoRA)](../09-finetuning.md)*
