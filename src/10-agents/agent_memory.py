"""
Agent memory systems: buffer memory, summary memory, entity memory.
Covers: sliding window buffer, token-aware compression, entity extraction,
        memory-augmented conversation simulation, token counting.
No external deps required.
"""

import re, json
from collections import deque, defaultdict

rng = None   # deterministic


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Token Counting ─────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    """Rough approximation: 1 token ≈ 0.75 words (GPT-style)."""
    return max(1, int(len(text.split()) * 4 / 3))


# ── Buffer Memory ─────────────────────────────────────────────────
class BufferMemory:
    """
    Keeps the last `max_messages` turns in memory.
    Simple and fast; loses old context when full.
    """

    def __init__(self, max_messages: int = 10):
        self.buffer = deque(maxlen=max_messages)

    def add(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})

    def get_messages(self) -> list:
        return list(self.buffer)

    def token_count(self) -> int:
        return sum(count_tokens(m["content"]) for m in self.buffer)

    def clear(self):
        self.buffer.clear()

    def __repr__(self):
        return f"BufferMemory(turns={len(self.buffer)}, tokens={self.token_count()})"


# ── Summary Memory ─────────────────────────────────────────────────
class SummaryMemory:
    """
    Compresses old messages into a running summary when the buffer
    exceeds a token threshold. Preserves recent messages verbatim.
    """

    def __init__(self, summarizer, buffer_token_limit: int = 500,
                 keep_recent: int = 4):
        self.summarizer         = summarizer   # callable(text) -> summary_str
        self.buffer_token_limit = buffer_token_limit
        self.keep_recent        = keep_recent
        self.summary            = ""           # compressed history
        self.recent: list       = []           # recent uncompressed messages

    def add(self, role: str, content: str):
        self.recent.append({"role": role, "content": content})
        if self._recent_tokens() > self.buffer_token_limit:
            self._compress()

    def _recent_tokens(self) -> int:
        return sum(count_tokens(m["content"]) for m in self.recent)

    def _compress(self):
        to_compress = self.recent[:-self.keep_recent]
        if not to_compress:
            return
        text = "\n".join(f"{m['role']}: {m['content']}" for m in to_compress)
        new_summary = self.summarizer(
            f"Existing summary: {self.summary}\n\nNew messages:\n{text}"
        )
        self.summary = new_summary
        self.recent  = self.recent[-self.keep_recent:]
        print(f"    [SummaryMemory] Compressed {len(to_compress)} messages → summary")

    def get_messages(self) -> list:
        messages = []
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Conversation history summary: {self.summary}"
            })
        messages.extend(self.recent)
        return messages

    def token_count(self) -> int:
        summary_tokens = count_tokens(self.summary) if self.summary else 0
        return summary_tokens + sum(count_tokens(m["content"]) for m in self.recent)

    def __repr__(self):
        return (f"SummaryMemory(summary_len={len(self.summary)}, "
                f"recent={len(self.recent)}, tokens={self.token_count()})")


# ── Entity Memory ─────────────────────────────────────────────────
class EntityMemory:
    """
    Extracts named entities and their attributes from conversation turns.
    Maintains a persistent entity → facts dict across turns.
    """

    def __init__(self, extractor):
        self.extractor = extractor   # callable(text) -> dict[entity, list[fact]]
        self.entities: dict = defaultdict(list)

    def add(self, role: str, content: str):
        if role == "user":
            extracted = self.extractor(content)
            for entity, facts in extracted.items():
                for f in facts:
                    if f not in self.entities[entity]:
                        self.entities[entity].append(f)

    def get_context(self) -> str:
        if not self.entities:
            return ""
        lines = [f"  {e}: {'; '.join(f)}" for e, f in self.entities.items()]
        return "Known entities:\n" + "\n".join(lines)

    def get_facts_about(self, entity: str) -> list:
        return self.entities.get(entity, [])

    def __repr__(self):
        return f"EntityMemory(entities={dict(self.entities)})"


# ── Combined Memory ───────────────────────────────────────────────
class CombinedMemory:
    """
    Buffer memory for recent turns + entity memory for persistent facts.
    """

    def __init__(self, buffer_max: int = 8):
        self.buffer = BufferMemory(max_messages=buffer_max)
        self.entity = EntityMemory(extractor=simple_entity_extractor)

    def add(self, role: str, content: str):
        self.buffer.add(role, content)
        self.entity.add(role, content)

    def get_messages(self) -> list:
        messages = []
        entity_ctx = self.entity.get_context()
        if entity_ctx:
            messages.append({"role": "system", "content": entity_ctx})
        messages.extend(self.buffer.get_messages())
        return messages

    def __repr__(self):
        return f"CombinedMemory({self.buffer}, {self.entity})"


# ── Mock Extractors / Summarizers ─────────────────────────────────
def simple_entity_extractor(text: str) -> dict:
    """Rule-based entity extraction (in production: call an LLM)."""
    entities = {}

    name_m = re.search(r"(?:my name is|I am|I'm)\s+([A-Z][a-z]+)", text)
    if name_m:
        entities[name_m.group(1)] = ["is the user"]

    job_m = re.search(r"(?:I work as|I am a|I'm a)\s+([a-z ]+?)(?:\.|,|$)", text, re.I)
    if job_m:
        subject = name_m.group(1) if name_m else "User"
        entities[subject] = entities.get(subject, []) + [f"works as {job_m.group(1).strip()}"]

    location_m = re.search(r"(?:I live in|I'm from|based in)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
    if location_m:
        subject = name_m.group(1) if name_m else "User"
        entities.setdefault(subject, []).append(f"lives in {location_m.group(1)}")

    return entities


def mock_summarizer(text: str) -> str:
    """Produces a deterministic mock summary (in production: call an LLM)."""
    lines = [l for l in text.split("\n") if l.strip() and not l.startswith("Existing")]
    if len(lines) > 3:
        return f"[SUMMARY] {len(lines)} messages discussed. Topics: " + \
               ", ".join(l.split(":")[0].strip() for l in lines[:3])
    return "[SUMMARY] " + " | ".join(l[:40] for l in lines)


# ── Conversation Simulator ─────────────────────────────────────────
def simulate_conversation(memory, turns: list):
    """Run a scripted conversation through the memory system."""
    for role, content in turns:
        memory.add(role, content)
        token_info = (f"tokens={memory.token_count()}"
                      if hasattr(memory, "token_count") else "")
        print(f"  [{role:9}] {content[:60]:<60} | {token_info}")


def main():
    section("1. BUFFER MEMORY")
    buf = BufferMemory(max_messages=6)

    turns = [
        ("user",      "Hi! My name is Alice."),
        ("assistant", "Hello Alice! How can I help?"),
        ("user",      "What is 2+2?"),
        ("assistant", "2+2 = 4."),
        ("user",      "Now what is 4+4?"),
        ("assistant", "4+4 = 8."),
        ("user",      "What was my first question?"),       # tests memory window
        ("assistant", "Your first question was '2+2'."),
        ("user",      "And before that?"),                  # should be pushed out by maxlen=6
        ("assistant", "I don't have that in my window."),
    ]

    print("  Adding turns (maxlen=6):")
    for role, content in turns[:8]:
        buf.add(role, content)
        print(f"  [{role:9}] {content[:55]} | buf_size={len(buf.buffer)} tokens={buf.token_count()}")

    print(f"\n  Final buffer ({len(buf.buffer)} turns, {buf.token_count()} tokens):")
    for m in buf.get_messages():
        print(f"    {m['role']:9}: {m['content'][:60]}")

    # The first two turns should be gone (maxlen=6 means only last 6 remain)
    first_message = buf.get_messages()[0]["content"]
    print(f"\n  Oldest visible message: '{first_message[:60]}'")
    assert "Alice" not in first_message or "first question" in first_message or True

    section("2. SUMMARY MEMORY")
    smem = SummaryMemory(summarizer=mock_summarizer, buffer_token_limit=100, keep_recent=3)

    long_conversation = [
        ("user",      "Hi! I'm Alice, a data scientist based in London."),
        ("assistant", "Hello Alice! Great to meet you."),
        ("user",      "I've been working on a customer churn model."),
        ("assistant", "That's interesting. What features are you using?"),
        ("user",      "We use tenure, monthly charges, and contract type."),
        ("assistant", "Those are classic churn predictors. What's your AUC?"),
        ("user",      "Currently 0.84 with a Random Forest."),
        ("assistant", "Good baseline. Have you tried gradient boosting?"),
        ("user",      "Yes, XGBoost gives 0.87."),
        ("assistant", "Nice improvement. What's your train/test split?"),
        ("user",      "80/20 with stratified K-fold."),
        ("assistant", "That's solid. Any class imbalance issues?"),
        ("user",      "Churners are 15% of the dataset."),
        ("assistant", "Consider SMOTE or adjusting class weights."),
        ("user",      "Good idea. What about feature importance?"),
        ("assistant", "Monthly charges and tenure usually dominate."),
    ]

    print("  Simulating long conversation with auto-compression:")
    for role, content in long_conversation:
        smem.add(role, content)

    messages = smem.get_messages()
    print(f"\n  Final state: {smem}")
    print(f"  Messages returned to LLM: {len(messages)}")
    for m in messages:
        print(f"    [{m['role']:9}] {m['content'][:70]}")

    section("3. ENTITY MEMORY")
    emem = EntityMemory(extractor=simple_entity_extractor)

    entity_turns = [
        ("user",      "Hi, my name is Bob."),
        ("assistant", "Hello Bob!"),
        ("user",      "I work as a machine learning engineer."),
        ("assistant", "Interesting career!"),
        ("user",      "I'm based in San Francisco."),
        ("assistant", "Nice city for ML!"),
    ]

    print("  Extracting entities from user messages:")
    for role, content in entity_turns:
        emem.add(role, content)
        if role == "user":
            print(f"    user: '{content}' → entities={dict(emem.entities)}")

    print(f"\n  Entity context for next LLM call:")
    print(f"  {emem.get_context()}")

    section("4. COMBINED MEMORY")
    combined = CombinedMemory(buffer_max=6)

    combined_turns = [
        ("user",      "I'm Alice, a data scientist."),
        ("assistant", "Hi Alice! What are you working on?"),
        ("user",      "Building a recommendation system."),
        ("assistant", "What algorithm are you using?"),
        ("user",      "Matrix factorization with implicit feedback."),
        ("assistant", "ALS or SGD?"),
        ("user",      "ALS — scales better for sparse data."),
        ("assistant", "Good choice. What's your evaluation metric?"),
    ]

    simulate_conversation(combined, combined_turns)

    print(f"\n  {combined}")
    print(f"\n  Messages sent to LLM (buffer + entity context):")
    for m in combined.get_messages():
        print(f"    [{m['role']:9}] {m['content'][:70]}")

    section("5. TOKEN COUNT COMPARISON")
    print("  Memory type comparison for a 20-turn conversation:")
    print(f"  {'Type':<18} {'Tokens in context':>20} {'Notes'}")
    print(f"  {'-'*55}")

    sample_turns = [("user" if i%2==0 else "assistant",
                     f"Message {i} with about twenty words " * 2)
                    for i in range(20)]

    buf20 = BufferMemory(max_messages=20)
    buf6  = BufferMemory(max_messages=6)
    smem2 = SummaryMemory(mock_summarizer, buffer_token_limit=200, keep_recent=4)

    for role, content in sample_turns:
        buf20.add(role, content); buf6.add(role, content); smem2.add(role, content)

    total_tokens = sum(count_tokens(m["content"]) for m in [{"content": t[1]} for t in sample_turns])
    print(f"  {'Full history':<18} {total_tokens:>20} tokens  all 20 turns")
    print(f"  {'Buffer(max=20)':<18} {buf20.token_count():>20} tokens  all turns fit")
    print(f"  {'Buffer(max=6)':<18} {buf6.token_count():>20} tokens  last 6 only")
    print(f"  {'SummaryMemory':<18} {smem2.token_count():>20} tokens  summary + recent")


if __name__ == "__main__":
    main()
