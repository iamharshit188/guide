"""
BPE tokenizer from scratch — pure Python, no dependencies.
Covers: vocabulary initialisation, pair counting, merge rules, encode/decode,
        special tokens, unknown-word handling.
pip install (none required)
"""

import re
from collections import Counter, defaultdict

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer.

    Vocabulary = base characters + merged symbols + special tokens.
    Words are pre-split on whitespace; each word is represented as
    a sequence of characters with a word-boundary marker '</w>'.
    """

    SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def __init__(self, vocab_size=200):
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _word_to_chars(word: str) -> tuple[str, ...]:
        """'hello' → ('h', 'e', 'l', 'l', 'o</w>')"""
        if not word:
            return ()
        chars = list(word[:-1]) + [word[-1] + "</w>"]
        return tuple(chars)

    @staticmethod
    def _get_pairs(vocab_freq: dict[tuple, int]) -> Counter:
        """Count all adjacent symbol pairs weighted by word frequency."""
        pairs: Counter = Counter()
        for symbols, freq in vocab_freq.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_pair(pair: tuple[str, str],
                    vocab_freq: dict[tuple, int]) -> dict[tuple, int]:
        """Replace all occurrences of pair (a, b) with 'ab' in vocab."""
        a, b = pair
        merged = a + b
        new_vocab: dict[tuple, int] = {}
        for symbols, freq in vocab_freq.items():
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_vocab[tuple(new_symbols)] = freq
        return new_vocab

    def fit(self, texts: list[str]) -> "BPETokenizer":
        """Train BPE on a list of strings."""
        # Tokenise corpus to words, count word frequencies
        word_counts: Counter = Counter()
        for text in texts:
            for word in text.lower().split():
                word_counts[word] += 1

        # Initialise symbol vocabulary as character sequences
        vocab_freq: dict[tuple, int] = {
            self._word_to_chars(word): freq
            for word, freq in word_counts.items()
        }

        # Initial character vocabulary
        char_vocab: set[str] = set()
        for symbols in vocab_freq:
            char_vocab.update(symbols)

        # Build final vocab with specials first
        all_symbols = self.SPECIAL + sorted(char_vocab)
        n_merges = max(0, self.vocab_size - len(all_symbols))

        # BPE merge loop
        for _ in range(n_merges):
            pairs = self._get_pairs(vocab_freq)
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            self.merges.append(best)
            all_symbols.append(best[0] + best[1])
            vocab_freq = self._merge_pair(best, vocab_freq)

        # Build index maps
        self.vocab = {sym: i for i, sym in enumerate(all_symbols)}
        self.inv_vocab = {i: sym for sym, i in self.vocab.items()}
        self._trained = True
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _tokenise_word(self, word: str) -> list[str]:
        """Apply stored merge rules to tokenise a single word."""
        if not word:
            return []
        symbols = list(self._word_to_chars(word))
        for a, b in self.merges:
            merged = a + b
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str, add_special=True) -> list[int]:
        """Text → list of token IDs."""
        unk_id = self.vocab.get("<unk>", 1)
        tokens = []
        if add_special:
            tokens.append(self.vocab["<bos>"])
        for word in text.lower().split():
            for sym in self._tokenise_word(word):
                tokens.append(self.vocab.get(sym, unk_id))
        if add_special:
            tokens.append(self.vocab["<eos>"])
        return tokens

    def decode(self, ids: list[int]) -> str:
        """List of token IDs → text string."""
        specials = set(self.SPECIAL)
        parts = []
        for i in ids:
            sym = self.inv_vocab.get(i, "<unk>")
            if sym in specials:
                continue
            parts.append(sym)
        text = "".join(parts).replace("</w>", " ").strip()
        return text

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def vocab_size_actual(self) -> int:
        return len(self.vocab)

    def top_merges(self, n=10) -> list[tuple[str, str]]:
        return self.merges[:n]


# ---------------------------------------------------------------------------
# Demo corpus
# ---------------------------------------------------------------------------

CORPUS = [
    "the transformer architecture uses attention mechanisms",
    "attention is all you need for sequence to sequence tasks",
    "the encoder processes the input sequence with self attention",
    "the decoder generates output tokens one by one",
    "byte pair encoding builds a subword vocabulary from the corpus",
    "tokenization splits text into subword units for processing",
    "transformers replaced recurrent neural networks in natural language processing",
    "the feed forward layer applies a position wise transformation",
    "layer normalization stabilizes training in deep transformers",
    "the attention mechanism computes weighted sums of value vectors",
    "gradient clipping prevents exploding gradients during training",
    "learning rate warmup helps transformer training converge",
    "the decoder uses masked attention to prevent attending to future tokens",
    "cross attention connects encoder and decoder in seq to seq models",
    "bert uses bidirectional attention for language understanding",
    "gpt uses causal attention for language generation",
]


def main():
    section("BPE TOKENIZER TRAINING")
    tokenizer = BPETokenizer(vocab_size=120)
    tokenizer.fit(CORPUS)
    print(f"\n  Corpus size   : {len(CORPUS)} sentences")
    print(f"  Target vocab  : 120")
    print(f"  Actual vocab  : {tokenizer.vocab_size_actual()}")
    print(f"  Merge rules   : {len(tokenizer.merges)}")

    section("TOP MERGE RULES LEARNED")
    print(f"\n  First 20 merges (most frequent character pairs):")
    for i, (a, b) in enumerate(tokenizer.top_merges(20)):
        print(f"    {i+1:2d}. '{a}' + '{b}' → '{a+b}'")

    section("VOCABULARY SAMPLE")
    print(f"\n  Special tokens : {tokenizer.SPECIAL}")
    # Show some interesting subword tokens
    sample_syms = [s for s in tokenizer.vocab if len(s) > 2 and '</w>' not in s][:15]
    print(f"  Subword tokens (sample): {sample_syms}")
    eos_sym = [s for s in tokenizer.vocab if s.endswith('</w>')][:10]
    print(f"  Word-end tokens (sample): {eos_sym}")

    section("ENCODING EXAMPLES")
    test_sentences = [
        "the attention mechanism",
        "transformer encoder decoder",
        "unknown word supercalifragilistic",
    ]
    for sent in test_sentences:
        ids = tokenizer.encode(sent, add_special=True)
        tokens = [tokenizer.inv_vocab[i] for i in ids]
        decoded = tokenizer.decode(ids)
        print(f"\n  Input  : '{sent}'")
        print(f"  Tokens : {tokens}")
        print(f"  IDs    : {ids}")
        print(f"  Decode : '{decoded}'")

    section("ENCODE → DECODE ROUND-TRIP")
    print("\n  Testing encode/decode consistency on corpus sentences:")
    n_pass = 0
    for sent in CORPUS[:8]:
        ids = tokenizer.encode(sent, add_special=False)
        decoded = tokenizer.decode(ids)
        match = decoded.strip() == sent.strip()
        n_pass += int(match)
        status = "PASS" if match else "DIFF"
        print(f"  [{status}] '{sent[:45]}'")
        if not match:
            print(f"       got: '{decoded[:45]}'")
    print(f"\n  {n_pass}/{min(8, len(CORPUS))} round-trips exact match.")
    print("  Minor mismatches: multiple spaces collapse, expected.")

    section("BPE vs CHARACTER vs WORD TOKENIZATION")
    test = "tokenization and self attention"
    bpe_ids = tokenizer.encode(test, add_special=False)
    bpe_toks = [tokenizer.inv_vocab[i] for i in bpe_ids]
    char_toks = list(test.replace(" ", "▁"))
    word_toks = test.split()
    print(f"\n  Input: '{test}'")
    print(f"\n  Word-level   ({len(word_toks):2d} tokens): {word_toks}")
    print(f"  BPE-level    ({len(bpe_toks):2d} tokens): {bpe_toks}")
    print(f"  Char-level   ({len(char_toks):2d} tokens): {char_toks[:20]}...")

    section("BPE MERGE ALGORITHM STEP-BY-STEP")
    print("\n  Tracing BPE on mini corpus: ['low', 'lower', 'lowest']")
    mini = ["low", "lower", "lowest"]
    mini_vf = {BPETokenizer._word_to_chars(w): 1 for w in mini}
    print(f"\n  Initial:")
    for syms, f in mini_vf.items():
        print(f"    {syms}")
    for step in range(4):
        pairs = BPETokenizer._get_pairs(mini_vf)
        if not pairs:
            break
        best = pairs.most_common(1)[0]
        mini_vf = BPETokenizer._merge_pair(best[0], mini_vf)
        print(f"\n  Step {step+1}: merge {best[0]} (count={best[1]})")
        for syms in mini_vf:
            print(f"    {syms}")

    section("PRACTICAL NOTES")
    print("""
  GPT-4 (tiktoken cl100k_base):
    - Vocabulary size: 100,277
    - Operates on UTF-8 bytes → zero OOV for any text/code
    - Regex pre-tokeniser splits on whitespace, punctuation, digits

  LLaMA (SentencePiece BPE):
    - Vocabulary size: 32,000 (LLaMA 1/2), 128,000 (LLaMA 3)
    - Byte fallback for rare characters

  Rule of thumb: 1 token ≈ 0.75 English words ≈ 4 characters.
  100K token vocabulary provides good coverage for code + multilingual text.
    """)


if __name__ == "__main__":
    main()
