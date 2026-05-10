"""
Context injection, prompt construction, and LLM generation for RAG.
Covers: prompt templates, context window budget, token counting, mock LLM,
        OpenAI-compatible real API (graceful skip), citation formatting.
pip install numpy  (openai optional — pip install openai)
"""

import re
import math
import time
import textwrap
from typing import List, Dict, Any, Optional, Tuple


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Token counting (BPE approximation — no tiktoken dependency)
# ---------------------------------------------------------------------------

def approx_token_count(text: str) -> int:
    """
    Approximate BPE token count: ~4 chars/token for English prose.
    Real pipelines use tiktoken.encoding_for_model("gpt-4o").encode(text).
    """
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Retrieved chunk representation
# ---------------------------------------------------------------------------

class RetrievedChunk:
    def __init__(self, doc_id: str, text: str, score: float,
                 source: str = "unknown", page: int = 1, chunk_index: int = 0):
        self.doc_id = doc_id
        self.text = text
        self.score = score
        self.source = source
        self.page = page
        self.chunk_index = chunk_index

    def token_count(self) -> int:
        return approx_token_count(self.text)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise technical assistant. "
    "Answer using ONLY the provided context. "
    "If the answer is not in the context, respond exactly: 'I don't know based on the provided context.' "
    "Cite source filenames in your answer where relevant."
)

CONTEXT_BLOCK_TEMPLATE = "[{idx}] Source: {source} | Page: {page}\n{text}"

USER_PROMPT_TEMPLATE = (
    "Context:\n"
    "{context_blocks}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)


def build_context_block(chunk: RetrievedChunk, idx: int) -> str:
    return CONTEXT_BLOCK_TEMPLATE.format(
        idx=idx,
        source=chunk.source,
        page=chunk.page,
        text=chunk.text.strip(),
    )


def build_prompt(query: str, chunks: List[RetrievedChunk],
                 max_context_tokens: int = 2000) -> Tuple[str, str, int]:
    """
    Assemble system + user prompt, respecting context token budget.
    Returns: (system_prompt, user_prompt, total_token_estimate)
    """
    budget = max_context_tokens
    context_parts = []
    tokens_used = 0

    for i, chunk in enumerate(chunks):
        block = build_context_block(chunk, i + 1)
        block_tokens = approx_token_count(block)
        if tokens_used + block_tokens > budget:
            break
        context_parts.append(block)
        tokens_used += block_tokens

    context_str = "\n\n".join(context_parts)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context_blocks=context_str, query=query
    )
    system_tokens = approx_token_count(SYSTEM_PROMPT)
    user_tokens = approx_token_count(user_prompt)
    return SYSTEM_PROMPT, user_prompt, system_tokens + user_tokens


# ---------------------------------------------------------------------------
# Context window budget analysis
# ---------------------------------------------------------------------------

def context_budget_table(context_window: int = 4096,
                         chunk_sizes_tokens: List[int] = None) -> None:
    """
    Print budget breakdown for various chunk sizes and k values.
    """
    if chunk_sizes_tokens is None:
        chunk_sizes_tokens = [128, 256, 512]

    system_reserve = approx_token_count(SYSTEM_PROMPT)
    query_reserve = 50
    answer_reserve = 512
    overhead = system_reserve + query_reserve + answer_reserve
    available = context_window - overhead

    print(f"\n  Context window  : {context_window} tokens")
    print(f"  System prompt   : ~{system_reserve} tokens")
    print(f"  Query overhead  : ~{query_reserve} tokens")
    print(f"  Answer reserve  : ~{answer_reserve} tokens")
    print(f"  For context     : ~{available} tokens")
    print()
    print(f"  {'Chunk (tok)':>12} {'k=3':>8} {'k=5':>8} {'k=10':>8} {'k=20':>8}")
    print(f"  {'-'*48}")
    for chunk_tok in chunk_sizes_tokens:
        row = f"  {chunk_tok:>12}"
        for k in [3, 5, 10, 20]:
            needed = chunk_tok * k
            fits = "✓" if needed <= available else "✗"
            row += f"  {needed:>4}{fits:>3}"
        print(row)


# ---------------------------------------------------------------------------
# Mock LLM (deterministic, no API call)
# ---------------------------------------------------------------------------

class MockLLM:
    """
    Simulates LLM generation by extracting the most relevant sentence
    from the context that overlaps with the query terms.
    Produces deterministic output — no randomness.
    """

    def __init__(self, model: str = "mock-gpt-4o"):
        self.model = model
        self.calls = 0
        self.total_tokens = 0

    def _overlap_score(self, query_tokens: set, text: str) -> float:
        text_tokens = set(re.findall(r"[a-z]+", text.lower()))
        if not text_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens | text_tokens)

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.1, max_tokens: int = 256) -> Dict[str, Any]:
        self.calls += 1
        start = time.perf_counter()

        # Extract query from user_prompt
        query_match = re.search(r"Question:\s*(.+?)\s*\n\s*Answer:", user_prompt, re.DOTALL)
        query = query_match.group(1).strip() if query_match else ""
        query_tokens = set(re.findall(r"[a-z]+", query.lower()))

        # Extract context blocks from user_prompt
        context_match = re.search(r"Context:\n(.*?)\n\nQuestion:", user_prompt, re.DOTALL)
        context_text = context_match.group(1) if context_match else ""

        # Score each sentence in context by overlap with query
        sentences = re.split(r"(?<=[.!?])\s+", context_text)
        scored = [(self._overlap_score(query_tokens, s), s) for s in sentences
                  if len(s.strip()) > 20 and not s.startswith("[")]

        if not scored or max(s for s, _ in scored) < 0.05:
            answer = "I don't know based on the provided context."
        else:
            scored.sort(key=lambda x: x[0], reverse=True)
            # Combine top 2 sentences into a coherent answer
            top_sentences = [s for _, s in scored[:2] if s.strip()]
            answer = " ".join(top_sentences).strip()
            if not answer.endswith("."):
                answer += "."

        elapsed_ms = (time.perf_counter() - start) * 1000
        prompt_tokens = approx_token_count(system_prompt + user_prompt)
        completion_tokens = approx_token_count(answer)
        self.total_tokens += prompt_tokens + completion_tokens

        return {
            "answer": answer,
            "model": self.model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "latency_ms": round(elapsed_ms, 2),
        }


# ---------------------------------------------------------------------------
# OpenAI real LLM (graceful skip)
# ---------------------------------------------------------------------------

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAILLM:
    """
    OpenAI-compatible LLM wrapper. Falls back to mock if no API key.
    pip install openai
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("pip install openai")
        import os
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("Set OPENAI_API_KEY environment variable")
        self.client = openai.OpenAI(api_key=key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.1, max_tokens: int = 512) -> Dict[str, Any]:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        answer = response.choices[0].message.content
        return {
            "answer": answer,
            "model": self.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "latency_ms": round(elapsed_ms, 2),
        }


# ---------------------------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------------------------

def format_sources(chunks: List[RetrievedChunk], used_count: int) -> str:
    lines = ["Sources:"]
    for i, c in enumerate(chunks[:used_count], 1):
        lines.append(f"  [{i}] {c.source} (page {c.page}) — score {c.score:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full generator pipeline
# ---------------------------------------------------------------------------

class RAGGenerator:
    def __init__(self, llm, context_window: int = 4096, answer_reserve: int = 512):
        self.llm = llm
        self.context_window = context_window
        self.answer_reserve = answer_reserve

    def generate(self, query: str, chunks: List[RetrievedChunk],
                 temperature: float = 0.1) -> Dict[str, Any]:
        max_ctx = self.context_window - self.answer_reserve - 200
        sys_p, usr_p, total_tok = build_prompt(query, chunks, max_context_tokens=max_ctx)
        result = self.llm.generate(sys_p, usr_p, temperature=temperature,
                                   max_tokens=self.answer_reserve)
        result["prompt_tokens_estimate"] = total_tok
        result["chunks_used"] = min(len(chunks),
                                    sum(1 for c in chunks
                                        if approx_token_count(c.text) <= max_ctx))
        return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DEMO_CHUNKS = [
    RetrievedChunk("c0", "Scaled dot-product attention divides scores by sqrt(d_k) to prevent"
                         " vanishing gradients in softmax.", 0.91, "transformers.txt", 1, 0),
    RetrievedChunk("c1", "Multi-head attention projects Q, K, V into h subspaces and concatenates"
                         " all head outputs before the output projection.", 0.87, "transformers.txt", 2, 1),
    RetrievedChunk("c2", "BM25 is a probabilistic retrieval function using term frequency and"
                         " inverse document frequency with length normalisation.", 0.83, "retrieval.txt", 1, 2),
    RetrievedChunk("c3", "Hybrid search combines sparse BM25 scores with dense embedding"
                         " scores using Reciprocal Rank Fusion.", 0.79, "retrieval.txt", 2, 3),
    RetrievedChunk("c4", "RAG injects retrieved context into the language model prompt,"
                         " grounding the answer in external evidence.", 0.75, "rag.txt", 1, 4),
]


def main():
    llm = MockLLM(model="mock-gpt-4o")
    generator = RAGGenerator(llm, context_window=4096, answer_reserve=512)

    section("CONTEXT WINDOW BUDGET ANALYSIS")
    context_budget_table(context_window=4096, chunk_sizes_tokens=[128, 256, 512])

    section("PROMPT CONSTRUCTION")
    query = "How does the attention mechanism prevent gradient issues?"
    sys_p, usr_p, total_tok = build_prompt(query, DEMO_CHUNKS, max_context_tokens=2000)
    print(f"\n  Query: {query!r}")
    print(f"\n  System prompt ({approx_token_count(sys_p)} tokens):")
    print(f"    {sys_p}")
    print(f"\n  User prompt preview (first 500 chars):")
    for line in usr_p[:500].split("\n"):
        print(f"    {line}")
    print(f"\n  Total prompt tokens (estimate): {total_tok}")

    section("MOCK LLM — GENERATION")
    queries = [
        "How does the attention mechanism prevent gradient issues?",
        "What is BM25 and how does it normalise document length?",
        "How does hybrid search combine sparse and dense retrieval?",
        "What is the capital of France?",   # not in context → "I don't know"
    ]

    for q in queries:
        result = generator.generate(q, DEMO_CHUNKS, temperature=0.1)
        print(f"\n  Q: {q!r}")
        print(f"  A: {result['answer']}")
        print(f"     tokens={result['usage']['total_tokens']}  "
              f"latency={result['latency_ms']:.1f}ms  "
              f"chunks_used={result['chunks_used']}")

    section("TOKEN BUDGET PER CHUNK")
    print(f"\n  {'chunk_id':>10} {'chars':>7} {'~tokens':>8} {'score':>7} {'source'}")
    print(f"  {'-'*55}")
    for c in DEMO_CHUNKS:
        tok = approx_token_count(c.text)
        print(f"  {c.doc_id:>10} {len(c.text):>7} {tok:>8} {c.score:>7.4f} {c.source}")

    section("CITATION FORMATTING")
    print(f"\n  {format_sources(DEMO_CHUNKS, 4)}")

    section("TEMPERATURE EFFECT (MOCK)")
    q = "What does BM25 measure?"
    print(f"\n  Query: {q!r}")
    for temp in [0.0, 0.5, 1.0]:
        r = generator.generate(q, DEMO_CHUNKS, temperature=temp)
        print(f"  temp={temp}: {r['answer'][:100]}")

    section("PROMPT TEMPLATE VARIANTS")
    # Show that context placement matters
    templates = {
        "Context-first (standard)": (
            "Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"
        ),
        "Question-first": (
            "Question: {q}\n\nContext:\n{ctx}\n\nAnswer:"
        ),
    }
    ctx_text = "\n\n".join(build_context_block(c, i+1) for i, c in enumerate(DEMO_CHUNKS[:2]))
    q = "How does multi-head attention work?"
    print(f"\n  Query: {q!r}")
    for name, tmpl in templates.items():
        filled = tmpl.format(ctx=ctx_text, q=q)
        print(f"\n  Template: {name}")
        print(f"    ~{approx_token_count(filled)} tokens total")
        print(f"    Context position: {'before' if 'Context-first' in name else 'after'} question")

    section("OPENAI REAL API")
    if not OPENAI_AVAILABLE:
        print("\n  openai not installed — pip install openai")
    else:
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n  OPENAI_API_KEY not set — skipping real API call")
            print("  Set: export OPENAI_API_KEY=sk-...")
        else:
            try:
                real_llm = OpenAILLM(model="gpt-4o-mini")
                real_gen = RAGGenerator(real_llm)
                result = real_gen.generate(queries[0], DEMO_CHUNKS[:3])
                print(f"\n  Q: {queries[0]!r}")
                print(f"  A: {result['answer']}")
                print(f"     usage={result['usage']}")
            except Exception as e:
                print(f"\n  Real API error: {e}")

    section("SUMMARY — GENERATOR STATS")
    print(f"\n  Mock LLM calls    : {llm.calls}")
    print(f"  Total tokens used : {llm.total_tokens}")
    print(f"  Avg tokens/call   : {llm.total_tokens / max(llm.calls, 1):.0f}")


if __name__ == "__main__":
    main()
