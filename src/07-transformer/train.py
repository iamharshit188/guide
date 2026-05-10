"""
Transformer training loop — PyTorch.
Covers: synthetic seq2seq task, transformer LR schedule, gradient clipping,
        checkpointing, label smoothing, loss curve, beam search vs greedy.
pip install torch numpy
"""

import math
import time
import os
import tempfile
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("pip install torch")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


if not TORCH_AVAILABLE:
    import sys; sys.exit(0)


# ---------------------------------------------------------------------------
# Transformer LR Schedule
# ---------------------------------------------------------------------------

class TransformerScheduler:
    """
    lr(t) = d_model^{-0.5} * min(t^{-0.5}, t * warmup_steps^{-1.5})
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        t = self.step_num
        lr = (self.d_model ** -0.5) * min(t ** -0.5, t * self.warmup ** -1.5)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def current_lr(self):
        t = max(1, self.step_num)
        return (self.d_model ** -0.5) * min(t ** -0.5, t * self.warmup ** -1.5)


# ---------------------------------------------------------------------------
# Label Smoothing Loss
# ---------------------------------------------------------------------------

class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing (ε=smoothing)."""

    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (N, vocab_size)
        targets : (N,)
        """
        V = self.vocab_size
        log_probs = F.log_softmax(logits, dim=-1)  # (N, V)

        # Smooth targets: (1-ε) on true class + ε/V everywhere
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / V)
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence + self.smoothing / V)

        loss = -(smooth_targets * log_probs).sum(dim=-1)  # (N,)

        # Mask padding
        mask = targets != self.ignore_index
        return loss[mask].mean()


# ---------------------------------------------------------------------------
# Synthetic seq2seq dataset: reverse sequence
# ---------------------------------------------------------------------------

def make_dataset(n_samples: int, seq_len: int, vocab_size: int,
                 bos: int, eos: int, pad: int, rng: torch.Generator):
    """Task: reverse a sequence of tokens."""
    special = {bos, eos, pad}
    data = []
    for _ in range(n_samples):
        length = torch.randint(seq_len // 2, seq_len + 1, (1,), generator=rng).item()
        src = torch.randint(len(special), vocab_size, (length,), generator=rng)
        tgt_content = src.flip(0)
        # src: [bos, tokens..., eos]
        src_seq = torch.cat([torch.tensor([bos]), src, torch.tensor([eos])])
        # tgt input: [bos, reversed...]
        tgt_in  = torch.cat([torch.tensor([bos]), tgt_content])
        # tgt output: [reversed..., eos]
        tgt_out = torch.cat([tgt_content, torch.tensor([eos])])
        data.append((src_seq, tgt_in, tgt_out))
    return data


def collate(batch, pad_id: int):
    """Pad batch to max length."""
    srcs, tgt_ins, tgt_outs = zip(*batch)
    def pad_seq(seqs):
        L = max(s.size(0) for s in seqs)
        return torch.stack([
            F.pad(s, (0, L - s.size(0)), value=pad_id) for s in seqs
        ])
    return pad_seq(srcs), pad_seq(tgt_ins), pad_seq(tgt_outs)


# ---------------------------------------------------------------------------
# Gradient clipping demo
# ---------------------------------------------------------------------------

def gradient_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search(model, src: torch.Tensor, bos_id: int, eos_id: int,
                beam_size: int = 3, max_len: int = 20,
                length_penalty: float = 0.6) -> list[int]:
    """Beam search for a single source sequence (B=1)."""
    model.eval()
    enc_out = model.encoder(src)  # (1, S, d)

    # Each beam: (score, [token_ids])
    beams = [(0.0, [bos_id])]
    completed = []

    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_id:
                completed.append((score, seq))
                continue
            tgt = torch.tensor([seq], dtype=torch.long, device=src.device)
            tgt_mask = model.causal_mask(tgt.size(1)).to(src.device)
            dec_out, _ = model.decoder(tgt, enc_out, tgt_mask=tgt_mask)
            logits = model.proj(dec_out[:, -1])         # (1, vocab)
            log_probs = F.log_softmax(logits, dim=-1)[0]
            topk_vals, topk_ids = log_probs.topk(beam_size)
            for lp, tok_id in zip(topk_vals.tolist(), topk_ids.tolist()):
                new_score = score + lp
                candidates.append((new_score, seq + [tok_id]))

        if not candidates:
            break
        # Keep top beam_size (length-normalised)
        candidates.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        beams = candidates[:beam_size]
        if all(seq[-1] == eos_id for _, seq in beams):
            completed.extend(beams)
            break

    completed.extend(beams)
    completed.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
    return completed[0][1] if completed else [bos_id, eos_id]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, dataset, criterion, optimizer, scheduler,
          batch_size, device, epochs, max_grad_norm, pad_id,
          log_every=10):
    model.train()
    history = []

    for epoch in range(epochs):
        # Shuffle
        indices = torch.randperm(len(dataset))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(dataset), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = [dataset[i] for i in batch_idx]
            src, tgt_in, tgt_out = collate(batch, pad_id)
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            optimizer.zero_grad()
            logits, _ = model(src, tgt_in)  # (B, T_tgt, vocab)

            # Flatten for loss
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), tgt_out.view(B * T))
            loss.backward()

            # Gradient clipping
            grad_n = gradient_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            after_clip = gradient_norm(model)

            lr = scheduler.step()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / n_batches
        history.append(avg)

        if (epoch + 1) % log_every == 0 or epoch < 3:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}  lr={scheduler.current_lr():.2e}")

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    rng = torch.Generator()
    rng.manual_seed(42)

    PAD, BOS, EOS = 0, 1, 2
    VOCAB_SIZE = 20
    SEQ_LEN    = 8
    D_MODEL    = 64
    N_HEADS    = 4
    D_FF       = 128
    N_LAYERS   = 2
    WARMUP     = 40
    MAX_LEN    = 16
    EPOCHS     = 60
    BATCH_SIZE = 32
    N_TRAIN    = 512
    N_VAL      = 64
    MAX_GRAD   = 1.0
    device = torch.device("cpu")

    # -----------------------------------------------------------------------
    section("SYNTHETIC DATASET: SEQUENCE REVERSAL")
    # -----------------------------------------------------------------------
    print(f"\n  Task: reverse a sequence of {SEQ_LEN} tokens (vocab_size={VOCAB_SIZE})")
    print(f"  Src: [BOS, a, b, c, d, EOS]")
    print(f"  Tgt: [BOS, d, c, b, a] → target out: [d, c, b, a, EOS]")
    train_data = make_dataset(N_TRAIN, SEQ_LEN, VOCAB_SIZE, BOS, EOS, PAD, rng)
    val_data   = make_dataset(N_VAL,   SEQ_LEN, VOCAB_SIZE, BOS, EOS, PAD, rng)
    print(f"\n  Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    # Show one example
    src0, ti0, to0 = train_data[0]
    print(f"  Example src    : {src0.tolist()}")
    print(f"  Example tgt_in : {ti0.tolist()}")
    print(f"  Example tgt_out: {to0.tolist()}")

    # -----------------------------------------------------------------------
    section("TRANSFORMER LR SCHEDULE")
    # -----------------------------------------------------------------------
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from model import Transformer

    dummy_model = Transformer(VOCAB_SIZE, VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS,
                              MAX_LEN, dropout=0.1)
    dummy_opt = torch.optim.Adam(dummy_model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    sched = TransformerScheduler(dummy_opt, D_MODEL, WARMUP)
    print(f"\n  d_model={D_MODEL}, warmup={WARMUP}")
    print(f"  {'Step':>6} {'LR':>12} {'Phase'}")
    for t in [1, 10, WARMUP // 2, WARMUP, WARMUP * 2, WARMUP * 5, WARMUP * 20]:
        old = sched.step_num
        sched.step_num = t
        lr = sched.current_lr()
        sched.step_num = old
        phase = "warmup" if t <= WARMUP else "decay"
        print(f"  {t:>6} {lr:>12.6f} {phase}")
    peak_step = WARMUP
    peak_lr = D_MODEL ** -0.5 * WARMUP ** -0.5
    print(f"\n  Peak LR at step {peak_step}: {peak_lr:.6f}")
    print(f"  Formula: d_model^{{-0.5}} × warmup^{{-0.5}} = {D_MODEL}^{{-0.5}} × {WARMUP}^{{-0.5}}")

    # -----------------------------------------------------------------------
    section("LABEL SMOOTHING")
    # -----------------------------------------------------------------------
    crit_smooth = LabelSmoothingLoss(VOCAB_SIZE, smoothing=0.1)
    crit_hard   = nn.CrossEntropyLoss(ignore_index=PAD)
    dummy_logits = torch.randn(10, VOCAB_SIZE)
    dummy_tgts   = torch.randint(3, VOCAB_SIZE, (10,))
    ls_loss = crit_smooth(dummy_logits, dummy_tgts)
    ce_loss = crit_hard(dummy_logits, dummy_tgts)
    print(f"\n  Hard CE loss      : {ce_loss.item():.4f}")
    print(f"  Label-smooth loss : {ls_loss.item():.4f}")
    print(f"  Smoothed loss distributes {0.1:.0%} prob mass uniformly across vocabulary.")
    print(f"  Prevents model from being overconfident → better generalisation.")

    # -----------------------------------------------------------------------
    section("GRADIENT CLIPPING DEMO")
    # -----------------------------------------------------------------------
    tiny = Transformer(VOCAB_SIZE, VOCAB_SIZE, 16, 2, 32, 1, MAX_LEN, dropout=0.0)
    src_d = torch.randint(3, VOCAB_SIZE, (4, 6))
    tgt_d = torch.randint(3, VOCAB_SIZE, (4, 5))
    logits_d, _ = tiny(src_d, tgt_d)
    loss_d = F.cross_entropy(logits_d.view(-1, VOCAB_SIZE), tgt_d.view(-1))
    loss_d.backward()
    before = gradient_norm(tiny)
    torch.nn.utils.clip_grad_norm_(tiny.parameters(), max_norm=1.0)
    after = gradient_norm(tiny)
    print(f"\n  Gradient norm before clipping : {before:.4f}")
    print(f"  Gradient norm after  clipping : {after:.4f}  (≤ 1.0)")
    print(f"  Clipped: {before > 1.0}")

    # -----------------------------------------------------------------------
    section("TRAINING")
    # -----------------------------------------------------------------------
    model = Transformer(VOCAB_SIZE, VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS,
                        MAX_LEN, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerScheduler(optimizer, D_MODEL, WARMUP)
    criterion = LabelSmoothingLoss(VOCAB_SIZE, smoothing=0.1, ignore_index=PAD)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {total_params:,}")
    print(f"  Device: {device}, epochs={EPOCHS}, batch={BATCH_SIZE}, warmup={WARMUP}")
    t0 = time.perf_counter()
    history = train(model, train_data, criterion, optimizer, scheduler,
                    BATCH_SIZE, device, EPOCHS, MAX_GRAD, PAD, log_every=10)
    elapsed = time.perf_counter() - t0
    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Final loss: {history[-1]:.4f}")

    # -----------------------------------------------------------------------
    section("VALIDATION — SEQUENCE REVERSAL ACCURACY")
    # -----------------------------------------------------------------------
    model.eval()
    n_correct = 0
    for src_v, ti_v, to_v in val_data[:32]:
        src_pt = src_v.unsqueeze(0).to(device)
        generated = model.greedy_decode(src_pt, BOS, EOS, max_len=SEQ_LEN + 4)
        gen_toks = generated[0].tolist()
        # Strip BOS/EOS
        gen_content = [t for t in gen_toks if t not in {BOS, EOS, PAD}]
        true_content = to_v[to_v != EOS].tolist()
        if gen_content == true_content:
            n_correct += 1
    acc = n_correct / 32
    print(f"\n  Reversal accuracy (greedy decode, {32} samples): {acc:.1%}")
    print(f"  (Task is learned if > 50% — small model, 60 epochs, CPU training)")

    # Show some examples
    print(f"\n  Example predictions:")
    for src_v, _, to_v in val_data[:4]:
        src_pt = src_v.unsqueeze(0).to(device)
        gen = model.greedy_decode(src_pt, BOS, EOS, max_len=SEQ_LEN + 4)[0].tolist()
        true_out = to_v.tolist()
        src_content = src_v[1:-1].tolist()  # strip BOS/EOS
        gen_content = [t for t in gen if t not in {BOS, EOS}]
        match = "✓" if gen_content == [t for t in true_out if t != EOS] else "✗"
        print(f"    src={src_content}  pred={gen_content}  true={[t for t in true_out if t != EOS]}  {match}")

    # -----------------------------------------------------------------------
    section("BEAM SEARCH vs GREEDY DECODE")
    # -----------------------------------------------------------------------
    print(f"\n  Comparing greedy vs beam (beam_size=3) on 8 validation samples:")
    greedy_correct = 0
    beam_correct = 0
    for src_v, _, to_v in val_data[:8]:
        src_pt = src_v.unsqueeze(0).to(device)
        true_content = [t for t in to_v.tolist() if t != EOS]

        # Greedy
        g = model.greedy_decode(src_pt, BOS, EOS, max_len=SEQ_LEN + 4)[0].tolist()
        g_content = [t for t in g if t not in {BOS, EOS}]

        # Beam
        b = beam_search(model, src_pt, BOS, EOS, beam_size=3, max_len=SEQ_LEN + 4)
        b_content = [t for t in b if t not in {BOS, EOS}]

        greedy_correct += int(g_content == true_content)
        beam_correct   += int(b_content == true_content)

    print(f"  Greedy correct: {greedy_correct}/8")
    print(f"  Beam   correct: {beam_correct}/8")

    # -----------------------------------------------------------------------
    section("CHECKPOINTING")
    # -----------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "transformer_ckpt.pt")
        torch.save({
            "epoch": EPOCHS,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_step": scheduler.step_num,
            "final_loss": history[-1],
            "config": {
                "vocab_size": VOCAB_SIZE,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "d_ff": D_FF,
                "n_layers": N_LAYERS,
            },
        }, ckpt_path)
        size_kb = os.path.getsize(ckpt_path) / 1024
        print(f"\n  Checkpoint saved: {ckpt_path}")
        print(f"  File size: {size_kb:.1f} KB")

        # Reload and verify
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2 = Transformer(VOCAB_SIZE, VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS,
                             MAX_LEN, dropout=0.0)
        model2.load_state_dict(ckpt["model_state"])
        model2.eval()

        # Check outputs match
        with torch.no_grad():
            src_check = val_data[0][0].unsqueeze(0)
            out1 = model.encoder(src_check)
            out2 = model2.encoder(src_check)
        diff = (out1 - out2).abs().max().item()
        print(f"  Load verification: max diff = {diff:.2e}  {'PASS' if diff < 1e-5 else 'FAIL'}")
        print(f"  Epoch saved: {ckpt['epoch']}, final loss: {ckpt['final_loss']:.4f}")

    # -----------------------------------------------------------------------
    section("LOSS CURVE (ASCII)")
    # -----------------------------------------------------------------------
    print(f"\n  Training loss (every 10 epochs):")
    sampled = [(i+1, history[i]) for i in range(0, EPOCHS, 10)]
    max_loss = max(v for _, v in sampled)
    for ep, v in sampled:
        bar_len = int((v / max_loss) * 40)
        bar = "█" * bar_len
        print(f"  epoch {ep:3d}: {v:.4f} {bar}")

    # -----------------------------------------------------------------------
    section("INFERENCE: DECODING STRATEGIES")
    # -----------------------------------------------------------------------
    print("""
  Greedy:     w_t = argmax P(w | w_<t)
    Fast, deterministic, but locally optimal — may miss high-prob sequences.

  Beam search: maintain B candidate sequences, expand each by top-B tokens.
    Score: sum(log P) / length^α  (length penalty prevents short-sequence bias)
    Better quality than greedy for translation; less useful for open-ended gen.

  Temperature: P_τ(w) ∝ exp(logit / τ)
    τ < 1: sharper distribution (more deterministic)
    τ > 1: flatter distribution (more random)
    τ = 1: unchanged

  Top-k:      sample from top k tokens only  (k=50 typical)
  Top-p (nucleus): sample from smallest set with cumulative P ≥ p  (p=0.9 typical)
    """)


if __name__ == "__main__":
    main()
