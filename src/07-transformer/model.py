"""
Full Transformer encoder-decoder in PyTorch.
Covers: embeddings + sinusoidal PE, encoder/decoder layers, MHA, FFN,
        LayerNorm, residuals, causal mask, cross-attention, greedy decode.
pip install torch numpy
"""

import math
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
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.WQ = nn.Linear(d_model, d_model, bias=False)
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)
        self.WO = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, seq, d_model)
        mask   : (batch, 1, seq_q, seq_k) — True means mask out
        """
        B, T_q, _ = Q.shape
        T_k = K.shape[1]

        # Project and split into heads
        def project_split(x, W):
            return W(x).view(B, -1, self.h, self.d_k).transpose(1, 2)

        q = project_split(Q, self.WQ)   # (B, h, T_q, d_k)
        k = project_split(K, self.WK)
        v = project_split(V, self.WV)

        # Scaled dot-product attention
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        w = self.dropout(F.softmax(scores, dim=-1))
        out = w @ v  # (B, h, T_q, d_k)

        # Concatenate heads and project
        out = out.transpose(1, 2).reshape(B, T_q, self.h * self.d_k)
        return self.WO(out), w


# ---------------------------------------------------------------------------
# Feed-Forward Sublayer
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Encoder Layer
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """Pre-LN: LayerNorm applied before each sublayer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention sublayer
        x_n = self.norm1(x)
        attn_out, _ = self.self_attn(x_n, x_n, x_n, mask=src_mask)
        x = x + self.drop(attn_out)
        # FFN sublayer
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Pre-LN decoder: masked self-attn + cross-attn + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        # Masked self-attention
        x_n = self.norm1(x)
        sa_out, _ = self.self_attn(x_n, x_n, x_n, mask=tgt_mask)
        x = x + self.drop(sa_out)
        # Cross-attention (Q from decoder, K/V from encoder)
        x_n = self.norm2(x)
        ca_out, cross_w = self.cross_attn(x_n, enc_out, enc_out, mask=src_mask)
        x = x + self.drop(ca_out)
        # FFN
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x, cross_w


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 d_ff: int, n_layers: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = SinusoidalPE(d_model, max_len, dropout)
        self.scale = math.sqrt(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.pe(self.embed(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 d_ff: int, n_layers: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = SinusoidalPE(d_model, max_len, dropout)
        self.scale = math.sqrt(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        x = self.pe(self.embed(tgt) * self.scale)
        cross_weights = []
        for layer in self.layers:
            x, cw = layer(x, enc_out, tgt_mask, src_mask)
            cross_weights.append(cw)
        return self.norm(x), cross_weights


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 128,
                 n_heads: int = 4, d_ff: int = 256, n_layers: int = 2,
                 max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_heads, d_ff, n_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, n_heads, d_ff, n_layers, max_len, dropout)
        self.proj = nn.Linear(d_model, tgt_vocab)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def causal_mask(sz: int) -> torch.Tensor:
        """(1, 1, sz, sz) — True in upper triangle (future positions)."""
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1).view(1, 1, sz, sz)

    def forward(self, src, tgt, src_key_padding=None):
        tgt_mask = self.causal_mask(tgt.size(1)).to(src.device)
        enc_out = self.encoder(src, src_mask=None)
        dec_out, cross_w = self.decoder(tgt, enc_out, tgt_mask=tgt_mask, src_mask=None)
        logits = self.proj(dec_out)  # (B, T_tgt, tgt_vocab)
        return logits, cross_w

    @torch.no_grad()
    def greedy_decode(self, src, bos_id: int, eos_id: int, max_len: int = 50):
        """Generate greedily from a single source sequence."""
        self.eval()
        enc_out = self.encoder(src, src_mask=None)
        B = src.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            tgt_mask = self.causal_mask(ys.size(1)).to(src.device)
            dec_out, _ = self.decoder(ys, enc_out, tgt_mask=tgt_mask)
            logits = self.proj(dec_out[:, -1])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break
        return ys


# ---------------------------------------------------------------------------
# Parameter count utility
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_module = {}
    for name, mod in model.named_children():
        by_module[name] = sum(p.numel() for p in mod.parameters())
    return {"total": total, "trainable": trainable, "by_module": by_module}


def main():
    torch.manual_seed(42)

    SRC_VOCAB, TGT_VOCAB = 100, 100
    D_MODEL, N_HEADS, D_FF, N_LAYERS = 64, 4, 128, 2
    MAX_LEN = 32

    section("MODEL ARCHITECTURE")
    model = Transformer(
        src_vocab=SRC_VOCAB, tgt_vocab=TGT_VOCAB,
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
        n_layers=N_LAYERS, max_len=MAX_LEN, dropout=0.0,
    )
    params = count_params(model)
    print(f"\n  d_model={D_MODEL}, n_heads={N_HEADS}, d_ff={D_FF}, n_layers={N_LAYERS}")
    print(f"  Total parameters : {params['total']:,}")
    print(f"  Trainable        : {params['trainable']:,}")
    print(f"\n  Parameters by submodule:")
    for name, cnt in params["by_module"].items():
        print(f"    {name:<12} {cnt:>8,}")
    theory = 12 * N_LAYERS * D_MODEL ** 2
    print(f"\n  Theory (12 × L × d²) = 12 × {N_LAYERS} × {D_MODEL}² = {theory:,}")
    print(f"  Actual transformer blocks ≈ {params['by_module']['encoder'] + params['by_module']['decoder']:,}")

    section("FORWARD PASS")
    B, S_LEN, T_LEN = 2, 10, 8
    src = torch.randint(4, SRC_VOCAB, (B, S_LEN))
    tgt = torch.randint(4, TGT_VOCAB, (B, T_LEN))
    logits, cross_w = model(src, tgt)
    print(f"\n  src shape   : {tuple(src.shape)}")
    print(f"  tgt shape   : {tuple(tgt.shape)}")
    print(f"  logits shape: {tuple(logits.shape)}  (batch, tgt_len, tgt_vocab)")
    print(f"  cross_attn weights [layer 0]: {tuple(cross_w[0].shape)}  (B, h, T_tgt, T_src)")
    print(f"  All logits finite: {logits.isfinite().all().item()}")

    section("ENCODER LAYER SHAPES")
    enc_layer = EncoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0)
    x = torch.randn(B, S_LEN, D_MODEL)
    x_enc = enc_layer(x)
    print(f"\n  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(x_enc.shape)}  (same shape — residual structure)")

    section("DECODER LAYER — CAUSAL MASK EFFECT")
    dec_layer = DecoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0)
    enc_out = torch.randn(B, S_LEN, D_MODEL)
    tgt_in = torch.randn(B, T_LEN, D_MODEL)
    causal = Transformer.causal_mask(T_LEN)
    print(f"\n  Causal mask (upper triangle = True, means masked out):")
    for row in causal[0, 0]:
        print("   ", " ".join(str(int(v)) for v in row))
    dec_out, cw = dec_layer(tgt_in, enc_out, tgt_mask=causal)
    print(f"\n  Decoder output shape: {tuple(dec_out.shape)}")
    print(f"  Cross-attn weights  : {tuple(cw.shape)}  (B, h, T_tgt, T_src)")

    section("SINUSOIDAL POSITIONAL ENCODING")
    pe_layer = SinusoidalPE(D_MODEL, MAX_LEN, dropout=0.0)
    x_dummy = torch.zeros(1, 8, D_MODEL)
    pe_out = pe_layer(x_dummy)
    # The PE values are the difference from zeros
    pe_vals = pe_out[0].detach().numpy()
    print(f"\n  PE shape: {pe_vals.shape}  (seq_len, d_model)")
    print(f"  PE range: [{pe_vals.min():.4f}, {pe_vals.max():.4f}]")
    print(f"  Consecutive position cosine similarity:")
    for pos in range(min(5, pe_vals.shape[0] - 1)):
        a, b = pe_vals[pos], pe_vals[pos + 1]
        sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        print(f"    pos {pos} vs {pos+1}: {sim:.4f}")

    section("GREEDY DECODING")
    BOS_ID, EOS_ID = 1, 2
    src_single = torch.randint(4, SRC_VOCAB, (1, 6))
    generated = model.greedy_decode(src_single, BOS_ID, EOS_ID, max_len=10)
    print(f"\n  Source tokens : {src_single[0].tolist()}")
    print(f"  Generated IDs : {generated[0].tolist()}")
    print(f"  Generated len : {generated.size(1)}  (includes BOS)")
    print(f"\n  Note: untrained model generates random IDs.")
    print(f"  After training on seq2seq task, greedy decode produces meaningful output.")

    section("WEIGHT INITIALIZATION — XAVIER UNIFORM")
    attn = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0)
    # Check init stats before training
    for name, p in attn.named_parameters():
        print(f"  {name:<10}  shape={tuple(p.shape)}  "
              f"mean={p.data.mean():.4f}  std={p.data.std():.4f}")

    section("PRE-LN vs POST-LN GRADIENT FLOW")
    # Pre-LN: gradient norm should be similar across layers
    model_pre = Transformer(SRC_VOCAB, TGT_VOCAB, D_MODEL, N_HEADS, D_FF, N_LAYERS,
                            MAX_LEN, dropout=0.0)
    logits2, _ = model_pre(src, tgt)
    loss = F.cross_entropy(logits2.view(-1, TGT_VOCAB), tgt.view(-1))
    loss.backward()
    print(f"\n  Loss (random model): {loss.item():.4f}")
    grad_norms = []
    for name, p in model_pre.named_parameters():
        if p.grad is not None:
            grad_norms.append((name, p.grad.norm().item()))
    # Show encoder vs decoder gradient norms
    enc_grads = [(n, g) for n, g in grad_norms if "encoder" in n]
    dec_grads = [(n, g) for n, g in grad_norms if "decoder" in n]
    print(f"  Encoder param gradient norms (Pre-LN):")
    for n, g in enc_grads[:4]:
        print(f"    {n:<50} {g:.4f}")
    print(f"  Decoder param gradient norms (Pre-LN):")
    for n, g in dec_grads[:4]:
        print(f"    {n:<50} {g:.4f}")
    print(f"\n  Pre-LN: gradient norms are consistent across layers.")
    print(f"  Post-LN would show decay in early layers at init.")

    section("TRANSFORMER ARCHITECTURE SUMMARY")
    print(f"""
  Encoder:
    - {N_LAYERS} × EncoderLayer(d={D_MODEL}, h={N_HEADS}, ff={D_FF})
    - Each: Pre-LN + Self-Attention + residual + Pre-LN + FFN + residual
    - No masking (bidirectional)

  Decoder:
    - {N_LAYERS} × DecoderLayer(d={D_MODEL}, h={N_HEADS}, ff={D_FF})
    - Each: Pre-LN + Masked-Self-Attn + residual
           + Pre-LN + Cross-Attn(Q=dec, K/V=enc) + residual
           + Pre-LN + FFN + residual
    - Causal mask prevents attending to future positions

  Output:
    - Linear projection: d_model → tgt_vocab
    - Cross-entropy loss during training
    - Softmax → greedy/beam/sample at inference
    """)


if __name__ == "__main__":
    main()
