"""
Document ingestion and chunking strategies for RAG pipelines.
Covers: fixed-size chunking, recursive character splitting, semantic chunking,
        PDF/text loading, chunk metadata, overlap analysis.
pip install numpy
"""

import re
import os
import math
import tempfile
import numpy as np
from typing import List, Dict, Any


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Chunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"Chunk(len={len(self.text)}, meta={self.metadata})"


# ---------------------------------------------------------------------------
# Fixed-size chunking
# ---------------------------------------------------------------------------

def chunk_fixed(text: str, chunk_size: int = 512, overlap: int = 50,
                source: str = "unknown") -> List[Chunk]:
    """
    Split text into fixed-character windows with overlap.
    chunk_size chars with overlap chars of carry-forward context.
    """
    chunks = []
    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

    i = 0
    chunk_idx = 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        chunk_text = text[i:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                metadata={"source": source, "chunk_index": chunk_idx,
                          "strategy": "fixed", "start_char": i, "end_char": end}
            ))
            chunk_idx += 1
        i += stride
        if end == len(text):
            break

    return chunks


# ---------------------------------------------------------------------------
# Recursive character splitting
# ---------------------------------------------------------------------------

def chunk_recursive(text: str, max_chars: int = 512, overlap: int = 50,
                    separators: List[str] = None, source: str = "unknown",
                    _depth: int = 0) -> List[Chunk]:
    """
    Split text using a priority list of separators, recursing on large pieces.
    Separators tried in order: paragraph → sentence → word → character.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= max_chars:
        t = text.strip()
        if t:
            return [Chunk(text=t, metadata={"source": source, "strategy": "recursive",
                                            "depth": _depth})]
        return []

    sep = separators[0]
    remaining_seps = separators[1:]

    if sep == "":
        # Character-level fallback
        return chunk_fixed(text, chunk_size=max_chars, overlap=overlap, source=source)

    pieces = text.split(sep)
    chunks = []
    current = ""
    for piece in pieces:
        candidate = current + (sep if current else "") + piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current.strip():
                sub = chunk_recursive(current, max_chars, overlap, remaining_seps,
                                      source, _depth + 1)
                chunks.extend(sub)
            current = piece

    if current.strip():
        sub = chunk_recursive(current, max_chars, overlap, remaining_seps,
                              source, _depth + 1)
        chunks.extend(sub)

    # Re-index
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    return chunks


# ---------------------------------------------------------------------------
# Semantic chunking (cosine similarity between adjacent sentences)
# ---------------------------------------------------------------------------

def _simple_embed(text: str, dim: int = 64, rng=None) -> np.ndarray:
    """
    Deterministic character-frequency embedding (bag-of-chars, dim-truncated).
    Real pipelines use sentence-transformers here. This is purely for the demo
    to avoid requiring a network call or heavy model.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    # Build character frequency vector seeded by text content
    seed_val = sum(ord(c) * (i + 1) for i, c in enumerate(text[:200]))
    local_rng = np.random.default_rng(seed_val % (2**32))
    # Char n-gram bag-of-words into dim dimensions
    vec = np.zeros(dim)
    for i in range(len(text) - 1):
        bigram_hash = (ord(text[i]) * 31 + ord(text[i+1])) % dim
        vec[bigram_hash] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    # Add small noise based on seed for distinctiveness between similar texts
    noise = local_rng.standard_normal(dim) * 0.05
    vec = vec + noise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def chunk_semantic(text: str, threshold: float = 0.85, min_chunk_chars: int = 100,
                   source: str = "unknown") -> List[Chunk]:
    """
    Split text where cosine similarity between adjacent sentence embeddings
    drops below threshold. Uses simple character-level embedding as proxy;
    replace _simple_embed with sentence-transformers for production.
    """
    # Split into sentences
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [Chunk(text=text.strip(), metadata={"source": source, "strategy": "semantic",
                                                    "chunk_index": 0})]

    embeddings = [_simple_embed(s) for s in sentences]
    similarities = [_cosine(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]

    # Build chunks by splitting at low-similarity boundaries
    chunks = []
    current_sentences = [sentences[0]]
    chunk_idx = 0

    for i, sim in enumerate(similarities):
        if sim < threshold and len(" ".join(current_sentences)) >= min_chunk_chars:
            chunk_text = " ".join(current_sentences).strip()
            chunks.append(Chunk(
                text=chunk_text,
                metadata={"source": source, "strategy": "semantic",
                          "chunk_index": chunk_idx, "split_similarity": round(sim, 4)}
            ))
            chunk_idx += 1
            current_sentences = [sentences[i+1]]
        else:
            current_sentences.append(sentences[i+1])

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                metadata={"source": source, "strategy": "semantic",
                          "chunk_index": chunk_idx}
            ))

    return chunks


# ---------------------------------------------------------------------------
# Text loader (TXT and minimal PDF simulation)
# ---------------------------------------------------------------------------

def load_txt(path: str) -> str:
    """Load plain text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_text(path: str) -> str:
    """
    Load PDF using pypdf if available; fall back to treating file as text.
    pip install pypdf
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        print("  [pypdf not installed — treating as plaintext]")
        return load_txt(path)


def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf_text(path)
    return load_txt(path)


# ---------------------------------------------------------------------------
# Chunk analysis helpers
# ---------------------------------------------------------------------------

def chunk_stats(chunks: List[Chunk]) -> Dict[str, float]:
    lengths = [len(c.text) for c in chunks]
    return {
        "count": len(chunks),
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "mean_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "total_chars": sum(lengths),
    }


def overlap_fraction(chunks: List[Chunk]) -> float:
    """Estimate character-level overlap between consecutive chunks."""
    if len(chunks) < 2:
        return 0.0
    overlaps = []
    for i in range(len(chunks) - 1):
        a, b = chunks[i].text, chunks[i+1].text
        # Common suffix of a vs prefix of b
        overlap = 0
        max_check = min(200, len(a), len(b))
        for k in range(1, max_check + 1):
            if a[-k:] == b[:k]:
                overlap = k
        overlaps.append(overlap)
    return round(sum(overlaps) / (len(overlaps) * 100 + 1e-9), 4)


# ---------------------------------------------------------------------------
# Demo corpus
# ---------------------------------------------------------------------------

DEMO_TEXT = """
Retrieval-Augmented Generation (RAG) is a framework that enhances language models
by incorporating external knowledge retrieval. The model first retrieves relevant
documents from a corpus, then uses them to generate grounded responses.

The retrieval step typically uses dense vector search via embeddings. Each document
is encoded into a high-dimensional vector space. At query time, the query is encoded
using the same embedding model, and approximate nearest-neighbour search finds the
most semantically similar documents.

Chunking is the process of splitting documents into smaller segments before embedding.
The chunk size determines the granularity of retrieval. Smaller chunks enable precise
retrieval but may lack context. Larger chunks contain more context but dilute the
embedding signal. A typical chunk size is 256 to 512 tokens with 50-token overlap.

BM25 is a classic sparse retrieval algorithm based on term frequency and inverse
document frequency. It excels at exact keyword matching and is complementary to
dense retrieval. Hybrid systems combine both approaches to capture lexical and
semantic similarity simultaneously.

Evaluation of RAG systems uses metrics like faithfulness, answer relevancy, and
context recall. Faithfulness measures whether the answer is grounded in the retrieved
context. Answer relevancy measures whether the answer addresses the query. These
metrics can be estimated using an LLM as a judge, enabling automated evaluation
without human annotation.

Maximum Marginal Relevance (MMR) is a retrieval strategy that balances relevance
and diversity. It iteratively selects chunks that are relevant to the query but
dissimilar to already-selected chunks. This reduces redundancy in the context
window and improves answer quality when the corpus contains near-duplicate content.

The context window budget is a critical constraint. With a 4096-token window,
system prompts, query tokens, and retrieved chunks must all fit. Rerankers can
help by retrieving a large candidate set and filtering to the most relevant chunks,
effectively increasing recall while controlling context length.

Fine-tuning and RAG are complementary. Fine-tuning adapts the model's reasoning
style and output format. RAG provides up-to-date factual knowledge. Production
systems often combine both: a fine-tuned model with a RAG retrieval layer.
"""


def main():
    section("FIXED-SIZE CHUNKING")
    chunks_fixed = chunk_fixed(DEMO_TEXT, chunk_size=300, overlap=50, source="demo.txt")
    stats = chunk_stats(chunks_fixed)
    print(f"\n  chunk_size=300, overlap=50")
    print(f"  Count: {stats['count']}  |  Min: {stats['min_chars']}  "
          f"|  Max: {stats['max_chars']}  |  Mean: {stats['mean_chars']}")
    print(f"\n  First chunk ({len(chunks_fixed[0].text)} chars):")
    print(f"    {chunks_fixed[0].text[:120]}...")
    print(f"  Second chunk ({len(chunks_fixed[1].text)} chars):")
    print(f"    {chunks_fixed[1].text[:120]}...")

    # Show overlap between chunk[0] and chunk[1]
    overlap_chars = 50
    print(f"\n  Expected overlap region (~{overlap_chars} chars):")
    tail = chunks_fixed[0].text[-overlap_chars:]
    head = chunks_fixed[1].text[:overlap_chars]
    print(f"    end of chunk[0]: ...{tail!r}")
    print(f"    start of chunk[1]: {head!r}...")

    section("RECURSIVE CHARACTER SPLITTING")
    chunks_rec = chunk_recursive(DEMO_TEXT, max_chars=300, overlap=50, source="demo.txt")
    stats = chunk_stats(chunks_rec)
    print(f"\n  max_chars=300, overlap=50")
    print(f"  Count: {stats['count']}  |  Min: {stats['min_chars']}  "
          f"|  Max: {stats['max_chars']}  |  Mean: {stats['mean_chars']}")
    print(f"\n  First 3 chunks:")
    for i, c in enumerate(chunks_rec[:3]):
        print(f"    [{i}] depth={c.metadata.get('depth', 0)} | {len(c.text)} chars | "
              f"{c.text[:80]}...")

    section("SEMANTIC CHUNKING")
    # Lower threshold so we get splits on the demo text
    chunks_sem = chunk_semantic(DEMO_TEXT, threshold=0.92, min_chunk_chars=80, source="demo.txt")
    stats = chunk_stats(chunks_sem)
    print(f"\n  similarity_threshold=0.92, min_chunk_chars=80")
    print(f"  Count: {stats['count']}  |  Min: {stats['min_chars']}  "
          f"|  Max: {stats['max_chars']}  |  Mean: {stats['mean_chars']}")
    print(f"\n  Chunk boundaries (split_similarity where low ≈ topic shift):")
    for c in chunks_sem:
        sim = c.metadata.get("split_similarity", "n/a")
        print(f"    chunk[{c.metadata['chunk_index']}]  sim_before_split={sim}  "
              f"len={len(c.text)}  preview={c.text[:60]}...")

    section("FILE I/O — TXT LOADING VIA TEMPFILE")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                    encoding="utf-8") as f:
        f.write(DEMO_TEXT)
        tmp_path = f.name
    try:
        loaded = load_document(tmp_path)
        print(f"\n  Loaded {len(loaded)} chars from {os.path.basename(tmp_path)}")
        chunks_loaded = chunk_fixed(loaded, chunk_size=400, overlap=60,
                                    source=os.path.basename(tmp_path))
        print(f"  Chunked into {len(chunks_loaded)} fixed-size chunks")
        print(f"  Metadata sample: {chunks_loaded[0].metadata}")
    finally:
        os.unlink(tmp_path)

    section("CHUNK STRATEGY COMPARISON")
    strategies = {
        "Fixed-size (300/50)": chunk_fixed(DEMO_TEXT, 300, 50, "demo.txt"),
        "Fixed-size (512/0)":  chunk_fixed(DEMO_TEXT, 512, 0,  "demo.txt"),
        "Recursive (300/50)":  chunk_recursive(DEMO_TEXT, 300, 50, source="demo.txt"),
        "Semantic (0.92)":     chunk_semantic(DEMO_TEXT, 0.92, 80, "demo.txt"),
    }
    print(f"\n  {'Strategy':<25} {'Count':>6} {'Min':>6} {'Max':>6} {'Mean':>8}")
    print(f"  {'-'*55}")
    for name, chunks in strategies.items():
        s = chunk_stats(chunks)
        print(f"  {name:<25} {s['count']:>6} {s['min_chars']:>6} "
              f"{s['max_chars']:>6} {s['mean_chars']:>8.1f}")

    section("OVERLAP ANALYSIS")
    for name, chunks in strategies.items():
        frac = overlap_fraction(chunks)
        print(f"  {name:<25} overlap_fraction={frac:.4f}")

    section("METADATA STRUCTURE")
    print(f"\n  Example chunk metadata:")
    sample = chunk_fixed(DEMO_TEXT, 300, 50, "attention_paper.pdf")[2]
    for k, v in sample.metadata.items():
        print(f"    {k}: {v!r}")


if __name__ == "__main__":
    main()
