"""
Instruction dataset preparation: Alpaca/ChatML templates, tokenisation,
train/val split, data collator simulation, dataset statistics.
pip install numpy  (transformers optional)
"""

import re
import math
import random
from typing import List, Dict, Any, Tuple, Optional

RNG = random.Random(42)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

ALPACA_INPUT_TEMPLATE = (
    "Below is an instruction that describes a task, "
    "paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n{assistant}<|im_end|>"
)

LLAMA3_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
)


def format_alpaca(instruction: str, output: str, input_text: str = "") -> str:
    if input_text.strip():
        return ALPACA_INPUT_TEMPLATE.format(
            instruction=instruction, input=input_text, output=output)
    return ALPACA_TEMPLATE.format(instruction=instruction, output=output)


def format_chatml(system: str, user: str, assistant: str) -> str:
    return CHATML_TEMPLATE.format(system=system, user=user, assistant=assistant)


def format_llama3(system: str, user: str, assistant: str) -> str:
    return LLAMA3_TEMPLATE.format(system=system, user=user, assistant=assistant)


# ---------------------------------------------------------------------------
# Toy tokeniser (BPE approximation for demo — no tiktoken needed)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """
    Character-level tokeniser with a small fixed vocab.
    Approximates word-piece splitting for demo purposes.
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        # ASCII characters as base vocab (offset by 4 for specials)
        self.char2id = {chr(i): i + 4 for i in range(32, min(vocab_size + 28, 127))}
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.pad_token_id = self.PAD_ID
        self.eos_token_id = self.EOS_ID
        self.bos_token_id = self.BOS_ID

    def encode(self, text: str, add_special_tokens: bool = True,
               max_length: Optional[int] = None) -> List[int]:
        ids = ([self.BOS_ID] if add_special_tokens else [])
        for ch in text:
            ids.append(self.char2id.get(ch, self.UNK_ID))
        if add_special_tokens:
            ids.append(self.EOS_ID)
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        specials = set(self.SPECIAL.values())
        chars = []
        for i in ids:
            if skip_special_tokens and i in specials:
                continue
            chars.append(self.id2char.get(i, "?"))
        return "".join(chars)

    def __len__(self) -> int:
        return self.vocab_size


# ---------------------------------------------------------------------------
# Dataset record
# ---------------------------------------------------------------------------

class Sample:
    def __init__(self, text: str, source: str = ""):
        self.text = text
        self.source = source

    def __repr__(self):
        return f"Sample(len={len(self.text)}, source={self.source!r})"


# ---------------------------------------------------------------------------
# Build dataset from raw instruction/response pairs
# ---------------------------------------------------------------------------

RAW_DATA = [
    {
        "instruction": "Explain the difference between precision and recall.",
        "output": (
            "Precision = TP / (TP + FP): fraction of positive predictions that are correct. "
            "Recall = TP / (TP + FN): fraction of actual positives correctly predicted. "
            "High precision, low recall: few false positives but many missed positives. "
            "High recall, low precision: few missed positives but many false positives."
        ),
    },
    {
        "instruction": "What is gradient descent?",
        "output": (
            "Gradient descent iteratively updates parameters theta by moving opposite "
            "to the gradient of the loss: theta <- theta - lr * grad(L). "
            "The learning rate lr controls step size. SGD uses a single sample per step, "
            "mini-batch GD uses a subset, and full-batch GD uses all data."
        ),
    },
    {
        "instruction": "Define overfitting and how to prevent it.",
        "output": (
            "Overfitting: model learns training data noise, fails on unseen data "
            "(high variance, low bias). Prevention: regularisation (L1/L2), dropout, "
            "early stopping, data augmentation, cross-validation, reducing model capacity."
        ),
    },
    {
        "instruction": "Explain attention mechanism in transformers.",
        "output": (
            "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V. "
            "Q, K, V are linear projections of the input. "
            "The softmax over QK^T / sqrt(d_k) gives attention weights, "
            "scaled by sqrt(d_k) to prevent vanishing gradients in high dimensions."
        ),
    },
    {
        "instruction": "What is the vanishing gradient problem?",
        "output": (
            "In deep networks, gradients shrink exponentially through backprop via chain rule. "
            "With sigmoid (derivative max 0.25), stacking N layers multiplies gradients by "
            "at most 0.25^N. Early layers receive near-zero gradient, learning stalls. "
            "Solutions: ReLU, residual connections, batch normalisation, careful initialisation."
        ),
    },
    {
        "instruction": "Explain LoRA in one paragraph.",
        "output": (
            "LoRA freezes pre-trained weights W0 and adds a low-rank adapter W' = W0 + BA, "
            "where B in R^{m x r} and A in R^{r x n} with r << min(m,n). "
            "B is initialised to zero so delta W = 0 at training start. "
            "The forward pass computes h = W0 x + (alpha/r) B A x. "
            "Only A and B are trained, reducing trainable parameters by ~100-256x."
        ),
    },
    {
        "instruction": "What is BLEU score?",
        "output": (
            "BLEU measures n-gram precision between hypothesis and reference with a brevity penalty. "
            "BLEU = BP * exp(sum(w_n * log p_n)). Modified n-gram precision clips counts to "
            "avoid rewarding repetition. BP = exp(1 - r/c) if c <= r, else 1. "
            "BLEU-4 (n=1..4 with equal weights) is standard for translation evaluation."
        ),
    },
    {
        "instruction": "Explain the bias-variance tradeoff.",
        "output": (
            "Expected test error = Bias^2 + Variance + Irreducible noise. "
            "High bias (underfitting): model too simple, systematic error. "
            "High variance (overfitting): model too complex, sensitive to training data. "
            "Increasing model capacity reduces bias but increases variance."
        ),
    },
]


def build_dataset(raw: List[Dict], template: str = "alpaca") -> List[Sample]:
    samples = []
    for item in raw:
        if template == "alpaca":
            text = format_alpaca(item["instruction"], item["output"],
                                 item.get("input", ""))
        elif template == "chatml":
            text = format_chatml(
                system="You are a helpful AI assistant.",
                user=item["instruction"],
                assistant=item["output"],
            )
        elif template == "llama3":
            text = format_llama3(
                system="You are a helpful AI assistant.",
                user=item["instruction"],
                assistant=item["output"],
            )
        else:
            raise ValueError(f"Unknown template: {template}")
        samples.append(Sample(text, source=template))
    return samples


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def train_val_split(samples: List[Sample],
                    val_ratio: float = 0.2,
                    seed: int = 42) -> Tuple[List[Sample], List[Sample]]:
    indices = list(range(len(samples)))
    r = random.Random(seed)
    r.shuffle(indices)
    n_val = max(1, int(len(samples) * val_ratio))
    val_idx = set(indices[:n_val])
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


# ---------------------------------------------------------------------------
# Tokenisation + label masking
# ---------------------------------------------------------------------------

def tokenise_sample(sample: Sample, tokenizer: SimpleTokenizer,
                    max_length: int = 128,
                    response_marker: str = "### Response:\n") -> Dict[str, List[int]]:
    """
    Tokenise full text. Mask instruction tokens in labels with -100.
    Only response tokens contribute to the training loss.
    """
    ids = tokenizer.encode(sample.text, add_special_tokens=True,
                           max_length=max_length)
    labels = ids[:]

    # Find response start in the original text (approximate character split)
    marker_pos = sample.text.find(response_marker)
    if marker_pos != -1:
        instruction_text = sample.text[:marker_pos + len(response_marker)]
        # Encode instruction portion to find how many tokens to mask
        instr_ids = tokenizer.encode(instruction_text, add_special_tokens=True)
        mask_len = min(len(instr_ids), len(labels))
        for i in range(mask_len):
            labels[i] = -100

    attention_mask = [1] * len(ids)
    return {
        "input_ids": ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Data collator — pad to max length in batch
# ---------------------------------------------------------------------------

def collate_batch(batch: List[Dict[str, List[int]]],
                  pad_id: int = 0,
                  label_pad: int = -100) -> Dict[str, List[List[int]]]:
    """
    Right-pad input_ids, attention_mask, labels to the longest sequence in batch.
    For decoder-only models left-pad is preferred in real code, but right-pad
    is simpler to implement and works with causal masking.
    """
    max_len = max(len(item["input_ids"]) for item in batch)
    padded = {"input_ids": [], "attention_mask": [], "labels": []}

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        padded["input_ids"].append(item["input_ids"] + [pad_id] * pad_len)
        padded["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
        padded["labels"].append(item["labels"] + [label_pad] * pad_len)

    return padded


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def dataset_stats(samples: List[Sample], tokenizer: SimpleTokenizer,
                  max_length: int = 128, label: str = ""):
    lengths = [len(tokenizer.encode(s.text, max_length=max_length)) for s in samples]
    label_token_counts = []
    for s in samples:
        tok = tokenise_sample(s, tokenizer, max_length)
        n_labels = sum(1 for l in tok["labels"] if l != -100)
        label_token_counts.append(n_labels)

    print(f"\n  {label} ({len(samples)} samples)")
    print(f"    Sequence lengths: "
          f"min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f}")
    print(f"    Label tokens:    "
          f"min={min(label_token_counts)}, max={max(label_token_counts)}, "
          f"mean={sum(label_token_counts)/len(label_token_counts):.1f}")
    truncated = sum(1 for l in lengths if l >= max_length)
    print(f"    Truncated (>= {max_length}): {truncated}/{len(samples)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tokenizer = SimpleTokenizer(vocab_size=256)

    section("PROMPT TEMPLATES")
    ex = RAW_DATA[0]
    print("\n  --- Alpaca ---")
    print(format_alpaca(ex["instruction"], ex["output"])[:300])
    print("\n  --- ChatML ---")
    print(format_chatml("You are a helpful AI assistant.",
                        ex["instruction"], ex["output"])[:300])
    print("\n  --- LLaMA-3 ---")
    print(format_llama3("You are a helpful AI assistant.",
                        ex["instruction"], ex["output"])[:300])

    section("DATASET BUILD")
    for tmpl in ("alpaca", "chatml", "llama3"):
        ds = build_dataset(RAW_DATA, template=tmpl)
        print(f"\n  Template: {tmpl:8s} | {len(ds)} samples | "
              f"avg chars: {sum(len(s.text) for s in ds)//len(ds)}")

    section("TRAIN / VAL SPLIT")
    samples = build_dataset(RAW_DATA, template="alpaca")
    train, val = train_val_split(samples, val_ratio=0.2)
    print(f"\n  Total: {len(samples)} | Train: {len(train)} | Val: {len(val)}")

    section("TOKENISATION + LABEL MASKING")
    sample = train[0]
    tok = tokenise_sample(sample, tokenizer, max_length=128)
    print(f"\n  Sample text (first 120 chars):")
    print(f"    {sample.text[:120]!r}")
    print(f"\n  input_ids length: {len(tok['input_ids'])}")
    print(f"  Tokens masked (-100): {sum(1 for l in tok['labels'] if l == -100)}")
    print(f"  Label tokens:         {sum(1 for l in tok['labels'] if l != -100)}")
    print(f"\n  First 20 input_ids: {tok['input_ids'][:20]}")
    print(f"  First 20 labels:    {tok['labels'][:20]}")

    section("DATA COLLATOR")
    tokenised = [tokenise_sample(s, tokenizer, max_length=64) for s in train[:3]]
    batch = collate_batch(tokenised)
    print(f"\n  Batch of 3 samples padded to max length:")
    for key, rows in batch.items():
        shapes = [len(row) for row in rows]
        print(f"  {key:>15}: shapes={shapes}, all equal={len(set(shapes))==1}")

    section("DATASET STATISTICS")
    dataset_stats(train, tokenizer, max_length=128, label="Train")
    dataset_stats(val, tokenizer, max_length=128, label="Val")

    section("TEMPLATE COMPARISON")
    print(f"\n  {'Template':>10} {'Avg tokens':>12} {'Response %':>12}")
    print(f"  {'-'*38}")
    for tmpl in ("alpaca", "chatml", "llama3"):
        ds = build_dataset(RAW_DATA, template=tmpl)
        all_tok = [tokenise_sample(s, tokenizer, max_length=256) for s in ds]
        avg_tokens = sum(len(t["input_ids"]) for t in all_tok) / len(all_tok)
        avg_labels = sum(
            sum(1 for l in t["labels"] if l != -100) for t in all_tok
        ) / len(all_tok)
        pct = 100 * avg_labels / avg_tokens if avg_tokens else 0
        print(f"  {tmpl:>10} {avg_tokens:>12.1f} {pct:>11.1f}%")

    section("INSTRUCTION FORMAT ANALYSIS")
    print(f"\n  Key design decisions:")
    print(f"  1. Labels for instruction tokens → -100 (excluded from loss)")
    print(f"  2. Labels for response tokens → actual token ids (contribute to loss)")
    print(f"  3. This makes loss = -1/|y| * sum log p(y_t | x, y_<t)")
    print(f"  4. EOS token at end of response (model learns to stop generating)")
    print(f"  5. Right-pad with attention_mask=0 (causal mask ignores pad positions)")

    section("HF TRANSFORMERS INTEGRATION (GRACEFUL SKIP)")
    try:
        from transformers import AutoTokenizer
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False

    if not TRANSFORMERS_AVAILABLE:
        print("\n  transformers not installed — pip install transformers")
        print("  Real usage:")
        print("    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')")
        print("    tokenizer.pad_token = tokenizer.eos_token")
        print("    tokenized = tokenizer(text, max_length=2048, truncation=True,")
        print("                          return_tensors='pt')")
    else:
        print("\n  transformers available — would load real tokenizer here.")


if __name__ == "__main__":
    main()
