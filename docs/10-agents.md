# Module 10 — LLM Agents & Tool Use

> **Run the code:**
> ```bash
> cd src/10-agents
> python3.14 react_agent.py
> python3.14 tool_calling.py
> python3.14 agent_memory.py
> python3.14 agent_eval.py
> ```

---

## Prerequisites & Overview

**Time estimate:** 6–8 hours

| Prerequisite | From |
|-------------|------|
| Transformer decoder & autoregressive generation | Module 07 |
| Flask API patterns | Module 04 |
| Attention mechanism | Module 06 |

**Before you start:**
- [ ] Understand how the Transformer decoder generates one token at a time
- [ ] Know the shape of a `POST /chat/completions` request and response
- [ ] Understand JSON Schema (`type`, `properties`, `required`, `enum`)

**Module map:**

| Section | Core concept |
|---------|-------------|
| ReAct loop | Thought → Action → Observation interleaving |
| Tool schemas | JSON Schema function definitions |
| Chain-of-thought | Zero-shot, few-shot, self-consistency |
| Multi-step planning | Plan → Execute decomposition |
| Agent memory | Buffer ($N$ messages) + summary compression |
| Error recovery | Retry with error injected as observation |
| Evaluation | Trajectory accuracy, tool precision, answer F1 |

---

## ReAct — Reasoning + Acting

### The Core Loop

ReAct (Yao et al., 2022) interleaves chain-of-thought reasoning with tool actions. The LLM never takes an action without first generating a reasoning trace.

```
THOUGHT → ACTION → OBSERVATION → THOUGHT → ACTION → OBSERVATION → FINAL ANSWER
```

### Formal Definition

State $s_t = (q, h_t)$ where $q$ is the question and $h_t$ is the trajectory history:

$$h_t = (o_1, a_1, \text{obs}_1, \;o_2, a_2, \text{obs}_2, \;\ldots)$$

LLM policy $\pi_\theta$ generates reasoning $o_t$ then action $a_t$:

$$o_t \sim \pi_\theta(\cdot \mid s_t), \qquad a_t \sim \pi_\theta(\cdot \mid s_t, o_t)$$

Tool executor $\mathcal{E}$: $\;\text{obs}_t = \mathcal{E}(a_t)$

Termination condition: $a_t = \text{FINISH}[\text{answer}]$

### Prompt Template

```
You are an agent with access to tools. Use this format exactly:

Thought: reason about what to do next
Action: tool_name[input]
Observation: (filled by system after tool runs)
... (Thought/Action/Observation repeats)
Thought: I now know the final answer
Final Answer: the answer

Tools:
- calculator[expr]: evaluates a Python math expression
- search[query]: returns the top web result summary
- read_file[path]: reads a local file and returns its content

Question: {question}
```

### Python Implementation

```python
import re, math

def parse_action(action_str: str):
    m = re.match(r"(\w+)\[(.+)\]", action_str.strip())
    if not m:
        raise ValueError(f"Bad action format: {action_str}")
    return m.group(1), m.group(2)

class ReActAgent:
    def __init__(self, llm, tools: dict, max_steps: int = 10):
        self.llm      = llm       # callable(prompt: str) -> str
        self.tools    = tools     # {"tool_name": callable(input) -> str}
        self.max_steps = max_steps

    def run(self, question: str) -> str:
        history = f"Question: {question}\n"

        for step in range(self.max_steps):
            raw = self.llm(SYSTEM_PROMPT + history)

            # Extract Thought
            thought_m = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:)", raw, re.S)
            thought = thought_m.group(1).strip() if thought_m else ""

            # Check for Final Answer
            fa_m = re.search(r"Final Answer:\s*(.+)", raw, re.S)
            if fa_m:
                history += f"Thought: {thought}\nFinal Answer: {fa_m.group(1).strip()}\n"
                return fa_m.group(1).strip()

            # Extract Action
            action_m = re.search(r"Action:\s*(.+)", raw)
            if not action_m:
                break
            action_str = action_m.group(1).strip()

            history += f"Thought: {thought}\nAction: {action_str}\n"

            # Execute tool with error recovery
            try:
                tool_name, tool_input = parse_action(action_str)
                if tool_name not in self.tools:
                    observation = f"Error: unknown tool '{tool_name}'. Available: {list(self.tools)}"
                else:
                    observation = str(self.tools[tool_name](tool_input))
            except Exception as e:
                observation = f"Error: {e}. Rethink your approach."

            history += f"Observation: {observation}\n"
            print(f"  [step {step+1}] {action_str} → {observation[:80]}")

        return "Max steps reached."
```

---

## Tool Schemas — OpenAI Function Calling Format

### JSON Schema for a Tool

The OpenAI tools API format is the industry standard for structured tool use:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name, e.g. 'London'"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"]
        }
      },
      "required": ["city"]
    }
  }
}
```

The LLM outputs a structured `tool_calls` array; the application dispatches and injects `role: "tool"` messages back.

### Tool Registry Decorator

```python
import json, math

TOOL_REGISTRY: dict = {}

def tool(name, description, params):
    def decorator(fn):
        TOOL_REGISTRY[name] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {"name": name, "description": description, "parameters": params}
            }
        }
        return fn
    return decorator

@tool("calculator", "Evaluate a Python math expression",
      {"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]})
def calculator(expression: str) -> float:
    allowed = set("0123456789+-*/().^ eE")
    if not all(c in allowed for c in expression):
        raise ValueError("Invalid characters")
    return eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi, "log": math.log})

@tool("word_count", "Count words in a string",
      {"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})
def word_count(text: str) -> int:
    return len(text.split())

def dispatch_tool_call(tool_call_dict: dict) -> str:
    name = tool_call_dict["function"]["name"]
    args = json.loads(tool_call_dict["function"]["arguments"])
    if name not in TOOL_REGISTRY:
        return f"Unknown tool: {name}"
    try:
        return str(TOOL_REGISTRY[name]["fn"](**args))
    except Exception as e:
        return f"Tool error: {e}"
```

### Multi-Tool Request Cycle (with real API)

```python
def run_agentic_loop(client, user_message: str, max_rounds: int = 5) -> str:
    messages = [{"role": "user", "content": user_message}]
    tool_schemas = [t["schema"] for t in TOOL_REGISTRY.values()]

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content           # LLM done, no more tool calls

        for tc in msg.tool_calls:
            result = dispatch_tool_call(tc.model_dump())
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "Max rounds reached."
```

### Parallel Tool Calls

Modern LLMs can request multiple tool calls in a single response (parallel execution):

```python
# LLM response with 2 parallel tool calls:
# tool_calls = [
#   {id:"call_1", function:{name:"calculator", arguments:'{"expression":"2**10"}'}},
#   {id:"call_2", function:{name:"word_count", arguments:'{"text":"hello world"}'}},
# ]

import concurrent.futures

def dispatch_parallel(tool_calls: list) -> list:
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = {tc.id: pool.submit(dispatch_tool_call, tc.model_dump())
                   for tc in tool_calls}
    return [{"role":"tool","tool_call_id":tid,"content":f.result()}
            for tid, f in futures.items()]
```

---

## Chain-of-Thought Prompting

### Zero-Shot CoT

Appending "Let's think step by step." to a prompt improves multi-step reasoning across LLMs (Kojima et al., 2022). The model first generates a reasoning chain, then the final answer.

```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each.
   How many balls does he have now? Let's think step by step.

A: Roger starts with 5 balls.
   He bought 2 cans × 3 = 6 balls.
   5 + 6 = 11 balls.
   The answer is 11.
```

Extract final answer: `re.search(r"answer is (\d+)", response).group(1)`

### Few-Shot CoT

Provide 3–5 worked examples before the target question. Each example shows the full reasoning chain → final answer. More reliable than zero-shot for complex multi-step tasks.

```python
COT_EXAMPLES = [
    {"q": "John has 3 apples and gives 1 away. How many?",
     "a": "John starts with 3. Gives 1 away. 3 - 1 = 2. The answer is 2."},
    {"q": "A rectangle is 4cm × 6cm. What is its area?",
     "a": "Area = length × width = 4 × 6 = 24 sq cm. The answer is 24."},
]

def few_shot_cot_prompt(question: str) -> str:
    examples = "\n\n".join(f"Q: {e['q']}\nA: {e['a']}" for e in COT_EXAMPLES)
    return f"{examples}\n\nQ: {question}\nA:"
```

### Self-Consistency

Sample $k$ reasoning chains at temperature $T > 0$, extract final answers, return majority vote:

$$\hat{a} = \text{mode}(\{a_1, a_2, \ldots, a_k\})$$

Self-consistency (Wang et al., 2022) outperforms single-sample CoT by 5–15% on math and reasoning benchmarks.

```python
from collections import Counter

def self_consistent_answer(llm, question: str, k: int = 7, temp: float = 0.7) -> str:
    prompt = few_shot_cot_prompt(question)
    answers = [extract_final_answer(llm(prompt, temperature=temp)) for _ in range(k)]
    return Counter(answers).most_common(1)[0][0]
```

---

## Multi-Step Planning

### Plan-then-Execute

Decompose complex tasks into a numbered plan before tool execution:

```
System: First output a numbered plan. Then execute each step using tools.
        Format:
        Plan:
        1. step one
        2. step two
        ...
        Executing step 1: ...
        Result: ...
        Executing step 2: ...
        Final Answer: ...

User: Analyze the sales data in sales.csv and summarize the top 3 products.
```

### Hierarchical Planning

For long-horizon tasks, decompose recursively:

```
Level 1 (planner):  break task → subtasks
Level 2 (executor): for each subtask → tool calls
Level 3 (verifier): check output meets requirements
```

```python
class HierarchicalAgent:
    def __init__(self, planner_llm, executor_llm, tools, verifier_llm=None):
        self.planner  = planner_llm
        self.executor = executor_llm
        self.tools    = tools
        self.verifier = verifier_llm

    def run(self, task: str) -> str:
        # Phase 1: Plan
        plan_prompt = f"Break this task into ≤5 numbered subtasks:\n{task}"
        plan_text = self.planner(plan_prompt)
        subtasks = parse_numbered_list(plan_text)

        results = []
        for i, subtask in enumerate(subtasks):
            # Phase 2: Execute
            agent = ReActAgent(self.executor, self.tools, max_steps=5)
            result = agent.run(subtask)
            results.append(result)
            print(f"  Subtask {i+1} done: {result[:60]}")

        # Phase 3: Synthesize
        synthesis_prompt = (
            f"Original task: {task}\n"
            f"Subtask results:\n" +
            "\n".join(f"{i+1}. {r}" for i, r in enumerate(results)) +
            "\nSynthesize a final answer:"
        )
        return self.planner(synthesis_prompt)
```

---

## Agent Memory

### Buffer Memory (Last N Messages)

The simplest memory: keep the last $N$ messages in the context window.

```python
from collections import deque

class BufferMemory:
    def __init__(self, max_messages: int = 20):
        self.buffer = deque(maxlen=max_messages)

    def add(self, role: str, content: str):
        self.buffer.append({"role": role, "content": content})

    def get_messages(self) -> list:
        return list(self.buffer)

    def token_count(self) -> int:
        return sum(len(m["content"].split()) * 4 // 3   # rough token estimate
                   for m in self.buffer)
```

**Problem:** Older context is lost when maxlen is exceeded. Relevant early information (like the user's name or initial constraints) is forgotten.

### Summary Memory

When the buffer exceeds a token threshold, compress old messages into a summary:

```python
class SummaryMemory:
    def __init__(self, llm, buffer_limit: int = 2000, summary_tokens: int = 300):
        self.llm           = llm
        self.buffer_limit  = buffer_limit
        self.summary_tokens = summary_tokens
        self.summary       = ""              # compressed old context
        self.recent        = []             # recent messages (not yet summarized)

    def add(self, role: str, content: str):
        self.recent.append({"role": role, "content": content})
        if self._token_count(self.recent) > self.buffer_limit:
            self._compress()

    def _compress(self):
        to_compress = self.recent[:-4]  # keep last 4 messages as-is
        text = "\n".join(f"{m['role']}: {m['content']}" for m in to_compress)
        self.summary = self.llm(
            f"Summarize this conversation in {self.summary_tokens} tokens:\n{text}"
        )
        self.recent = self.recent[-4:]

    def get_messages(self) -> list:
        messages = []
        if self.summary:
            messages.append({"role": "system",
                             "content": f"Conversation summary: {self.summary}"})
        messages.extend(self.recent)
        return messages

    def _token_count(self, messages: list) -> int:
        return sum(len(m["content"].split()) * 4 // 3 for m in messages)
```

### Entity Memory

Extract and track named entities across turns:

```python
class EntityMemory:
    def __init__(self, llm):
        self.llm      = llm
        self.entities: dict = {}   # entity → facts

    def update(self, text: str):
        prompt = (
            f"Extract entity facts from this text as JSON "
            f'{{entity: [facts...]}}:\n{text}'
        )
        extracted = json.loads(self.llm(prompt))
        for entity, facts in extracted.items():
            if entity not in self.entities:
                self.entities[entity] = []
            self.entities[entity].extend(facts)

    def get_context(self) -> str:
        if not self.entities:
            return ""
        lines = [f"{e}: {'; '.join(f)}" for e, f in self.entities.items()]
        return "Known entities:\n" + "\n".join(lines)
```

---

## Error Recovery

### Retry with Error Observation

When a tool fails, inject the error as an observation so the agent can self-correct:

```python
# Instead of:
# raise on tool failure  ← bad

# Do:
try:
    result = tool(input)
    observation = result
except Exception as e:
    observation = (
        f"Error: {e}. "
        "Rethink your approach — maybe use a different tool or different input format."
    )
# Inject observation and let the agent retry
```

### Retry with Exponential Backoff (API rate limits)

```python
import time, random

def call_llm_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="claude-sonnet-4-6", messages=messages
            )
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise
```

### Validation Before Execution

```python
def safe_tool_call(tool_name: str, args: dict, schema: dict) -> str:
    """Validate args against JSON schema before calling tool."""
    # Check required fields
    required = schema["parameters"].get("required", [])
    missing = [r for r in required if r not in args]
    if missing:
        return f"Validation error: missing required fields {missing}"

    # Check enum constraints
    props = schema["parameters"].get("properties", {})
    for key, val in args.items():
        if key in props and "enum" in props[key]:
            if val not in props[key]["enum"]:
                return f"Validation error: '{val}' not in enum {props[key]['enum']}"

    return dispatch_tool_call({"function": {"name": tool_name, "arguments": json.dumps(args)}})
```

---

## Agent Evaluation

### Evaluation Dimensions

| Metric | Definition |
|--------|-----------|
| **Task success rate** | Fraction of tasks with correct final answer |
| **Trajectory accuracy** | Fraction of steps that match reference trajectory |
| **Tool precision** | Fraction of tool calls that were necessary + correct |
| **Tool recall** | Fraction of necessary tools that were actually called |
| **Steps efficiency** | Reference steps / actual steps (lower = more detours) |
| **Answer F1** | Token-level F1 between generated and reference answers |

### Ground-Truth Trajectory Comparison

```python
def trajectory_accuracy(predicted: list, reference: list) -> float:
    """
    predicted, reference: lists of (action_type, action_input) tuples
    Returns: fraction of reference actions that appear in predicted (order-free).
    """
    pred_set = set(predicted)
    ref_set  = set(reference)
    if not ref_set:
        return 1.0
    return len(pred_set & ref_set) / len(ref_set)

def answer_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens  = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    tp        = len(pred_tokens & ref_tokens)
    precision = tp / len(pred_tokens)
    recall    = tp / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

### Evaluation Dataset Format

```python
# Evaluation examples: each has question, reference trajectory, reference answer
EVAL_DATASET = [
    {
        "question": "What is 2^10 + sqrt(144)?",
        "reference_trajectory": [("calculator", "2**10"), ("calculator", "144**0.5")],
        "reference_answer": "1036.0",
        "reference_steps": 2,
    },
    {
        "question": "How many words are in 'The quick brown fox'?",
        "reference_trajectory": [("word_count", "The quick brown fox")],
        "reference_answer": "4",
        "reference_steps": 1,
    },
]

def evaluate_agent(agent, dataset):
    results = {"success": 0, "total": len(dataset), "f1s": [], "efficiencies": []}

    for ex in dataset:
        response, trajectory = agent.run_with_trajectory(ex["question"])
        f1     = answer_f1(response, ex["reference_answer"])
        n_steps = len(trajectory)
        efficiency = ex["reference_steps"] / n_steps if n_steps > 0 else 0
        success = f1 >= 0.8

        results["success"] += int(success)
        results["f1s"].append(f1)
        results["efficiencies"].append(efficiency)

    results["success_rate"] = results["success"] / results["total"]
    results["mean_f1"]      = sum(results["f1s"]) / len(results["f1s"])
    results["mean_efficiency"] = sum(results["efficiencies"]) / len(results["efficiencies"])
    return results
```

---

## Interview Q&A

**Q: What is the difference between ReAct and Plan-and-Execute?**
**A:** ReAct interleaves reasoning and action at every step — each thought immediately leads to one tool call. Plan-and-Execute first generates a complete plan (list of subtasks), then executes them, then synthesizes. Plan-and-Execute is better for long-horizon tasks where the full plan helps avoid dead ends. ReAct is better for tasks where the next step depends on the previous observation (adaptive). Most production agents hybridize: plan globally, execute locally with ReAct.

**Q: Why is tool input validation important before dispatching?**
**A:** LLMs hallucinate arguments. Without validation, a tool might receive an integer where a string is expected, or a missing required field, causing cryptic errors. Injecting a validation error back as an observation allows the agent to self-correct. Always validate against the JSON Schema before calling the underlying function.

**Q: What causes infinite loops in agents, and how do you prevent them?**
**A:** An agent loops when it repeatedly calls the same tool with the same input (usually because the tool returns an unhelpful result and the agent doesn't adapt). Prevention: (1) `max_steps` hard limit, (2) detect repeated (tool, input) pairs in the trajectory and inject "You already tried this — try a different approach," (3) temperature > 0 adds randomness that breaks deterministic loops.

**Q: How does buffer memory vs. summary memory trade off?**
**A:** Buffer memory retains exact wording but loses older context beyond `maxlen`. Summary memory preserves semantics of old context in compressed form but loses exact quotes and numbers. For tasks requiring precise recall (e.g., "what was the exact number I mentioned 20 messages ago?"), buffer is safer. For long conversations, summary avoids context overflow. A hybrid — buffer recent messages + summary of older ones — is the production standard.

**Q: How do parallel tool calls work and when are they beneficial?**
**A:** When the LLM identifies that multiple tool calls are independent (do not depend on each other's results), it returns them in a single response as an array of `tool_calls`. The application executes them concurrently and returns all results in a single `role: "tool"` batch. This reduces the number of LLM round-trips. It is beneficial when a task needs multiple data points that can be fetched simultaneously (e.g., weather in 3 cities, or 3 independent calculator operations).

**Q: How do you evaluate whether an agent is using tools appropriately?**
**A:** Three metrics: (1) **Tool precision** = fraction of tool calls that were necessary (penalizes unnecessary calls), (2) **Tool recall** = fraction of necessary tools that were actually invoked (penalizes missing tools), (3) **Steps efficiency** = reference steps / actual steps (penalizes verbose trajectories). Additionally, check trajectory against a reference with human-labeled "correct action sequences" for the task.

**Q: What is self-consistency and why does it work?**
**A:** Self-consistency samples $k$ independent reasoning chains (at temperature > 0) and returns the majority-vote answer. It works because CoT reasoning errors are stochastic — different incorrect paths lead to different wrong answers, while the correct reasoning path consistently reaches the correct answer. The majority vote amplifies the signal. It adds a $k\times$ compute cost but consistently outperforms greedy decoding by 5–15% on math benchmarks.

**Q: What is the ReAct agent's limitation for tasks requiring backtracking?**
**A:** ReAct is greedy — it commits to each action and cannot backtrack to an earlier state. If a wrong tool call leads to an irreversible state (e.g., deleting a file, sending an email), the agent cannot undo it. Mitigation: (1) use reversible tools wherever possible, (2) add a confirmation step before irreversible actions, (3) use tree-search variants like Tree-of-Thoughts for tasks where backtracking is essential.

---

## Resources

**Papers:**
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Yao et al., 2022
- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903) — Wei et al., 2022
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — Wang et al., 2022
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — Schick et al., 2023
- [Tree of Thoughts: Deliberate Problem Solving with LLMs](https://arxiv.org/abs/2305.10601) — Yao et al., 2023

**Documentation:**
- OpenAI Function Calling Guide — tool schema format and multi-turn examples
- Anthropic Claude Tool Use — identical JSON Schema format, parallel tools

**Books:**
- *Building LLM Powered Applications* — Valentino Zocca (2024) — Chapter 5: Agents

---

*Next: [Module 11 — Deployment & Production ML](11-deployment.md)*
