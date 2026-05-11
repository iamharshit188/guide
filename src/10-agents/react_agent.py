"""
ReAct (Reasoning + Acting) agent from scratch.
Covers: Thought/Action/Observation loop, tool dispatch, error recovery,
        chain-of-thought, self-consistency voting.
No external deps required — uses a mock LLM for demonstration.
"""

import re, math, time
from collections import Counter

rng_seed = 42


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Mock LLM ──────────────────────────────────────────────────────
class MockLLM:
    """
    Deterministic mock that simulates a ReAct-capable LLM.
    In production, replace with an API call to Claude/GPT/etc.
    """

    RESPONSES = {
        "2**10 + sqrt(144)": [
            "Thought: I need to compute 2^10 and sqrt(144) separately, then add them.\n"
            "Action: calculator[2**10]",

            "Thought: 2^10 = 1024. Now I need sqrt(144).\n"
            "Action: calculator[144**0.5]",

            "Thought: sqrt(144) = 12.0. So 1024 + 12 = 1036.\n"
            "Final Answer: 1036.0",
        ],
        "words in 'the quick brown fox'": [
            "Thought: I should count the words in the phrase.\n"
            "Action: word_count[the quick brown fox]",

            "Thought: There are 4 words.\n"
            "Final Answer: 4",
        ],
        "area of rectangle 6 by 4": [
            "Thought: Area = length × width = 6 × 4.\n"
            "Action: calculator[6*4]",

            "Thought: 6 × 4 = 24.\n"
            "Final Answer: 24",
        ],
    }

    def __init__(self):
        self._step_idx: dict = {}

    def __call__(self, prompt: str, temperature: float = 0.0) -> str:
        for key, responses in self.RESPONSES.items():
            if key in prompt.lower():
                idx = self._step_idx.get(key, 0)
                response = responses[min(idx, len(responses) - 1)]
                self._step_idx[key] = idx + 1
                return response
        return "Thought: I don't know how to answer this.\nFinal Answer: Unknown"

    def reset(self):
        self._step_idx = {}


# ── Tool Definitions ──────────────────────────────────────────────
def _calculator(expression: str) -> float:
    allowed = set("0123456789+-*/().^ eE")
    if not all(c in allowed for c in expression):
        raise ValueError(f"Unsafe expression: '{expression}'")
    return eval(expression, {"__builtins__": {}},
                {"sqrt": math.sqrt, "pi": math.pi, "log": math.log, "abs": abs})


def _word_count(text: str) -> int:
    return len(text.split())


def _read_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read(500)    # limit to first 500 chars
    except FileNotFoundError:
        return f"Error: file not found: {path}"


TOOLS = {
    "calculator": _calculator,
    "word_count":  _word_count,
    "read_file":   _read_file,
}

SYSTEM_PROMPT = """You are an agent with access to tools. Use this format exactly:

Thought: reason about what to do next
Action: tool_name[input]
Observation: (filled by system)
... (repeat)
Thought: I now know the final answer
Final Answer: the answer

Available tools:
- calculator[expr]: evaluates a Python math expression (use ** for powers)
- word_count[text]: counts words in text
- read_file[path]: reads a local file

"""


# ── Action Parser ──────────────────────────────────────────────────
def parse_action(action_str: str):
    m = re.match(r"(\w+)\[(.+)\]", action_str.strip())
    if not m:
        raise ValueError(f"Cannot parse action: '{action_str}'")
    return m.group(1), m.group(2)


# ── ReAct Agent ───────────────────────────────────────────────────
class ReActAgent:
    def __init__(self, llm, tools: dict, max_steps: int = 10):
        self.llm       = llm
        self.tools     = tools
        self.max_steps = max_steps
        self.trajectory = []

    def run(self, question: str) -> str:
        self.trajectory = []
        history = f"Question: {question}\n"

        for step in range(self.max_steps):
            raw = self.llm(SYSTEM_PROMPT + history)

            thought_m = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:)", raw, re.S)
            thought = thought_m.group(1).strip() if thought_m else ""

            fa_m = re.search(r"Final Answer:\s*(.+)", raw, re.S)
            if fa_m:
                answer = fa_m.group(1).strip()
                self.trajectory.append(("FINISH", answer))
                return answer

            action_m = re.search(r"Action:\s*(.+)", raw)
            if not action_m:
                return "Agent produced no action and no final answer."

            action_str = action_m.group(1).strip()
            history += f"Thought: {thought}\nAction: {action_str}\n"

            try:
                tool_name, tool_input = parse_action(action_str)
                if tool_name not in self.tools:
                    observation = (
                        f"Error: unknown tool '{tool_name}'. "
                        f"Available: {list(self.tools.keys())}"
                    )
                else:
                    observation = str(self.tools[tool_name](tool_input))
                    self.trajectory.append((tool_name, tool_input))
            except Exception as e:
                observation = f"Error: {e}. Try a different approach."
                self.trajectory.append(("ERROR", str(e)))

            history += f"Observation: {observation}\n"
            print(f"    Step {step+1}: {action_str[:50]} → {observation[:60]}")

        return "Max steps reached without final answer."


# ── Self-Consistency ──────────────────────────────────────────────
def self_consistent_answer(llm_factory, question: str, k: int = 3) -> str:
    """Sample k independent reasoning chains, return majority-vote answer."""
    answers = []
    for i in range(k):
        llm = llm_factory()
        agent = ReActAgent(llm, TOOLS, max_steps=8)
        ans = agent.run(question)
        answers.append(ans)
        print(f"    Sample {i+1}: {ans}")

    majority = Counter(answers).most_common(1)[0][0]
    return majority


# ── Trajectory Accuracy ───────────────────────────────────────────
def trajectory_accuracy(predicted: list, reference: list) -> float:
    if not reference:
        return 1.0
    pred_set = set(predicted)
    ref_set  = set(reference)
    return len(pred_set & ref_set) / len(ref_set)


def answer_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens  = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    tp = len(pred_tokens & ref_tokens)
    p  = tp / len(pred_tokens)
    r  = tp / len(ref_tokens)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def main():
    section("1. REACT AGENT — Basic Operation")
    llm = MockLLM()

    tasks = [
        {
            "question":   "What is 2**10 + sqrt(144)?",
            "ref_answer": "1036.0",
            "ref_traj":   [("calculator", "2**10"), ("calculator", "144**0.5")],
        },
        {
            "question":   "How many words in 'the quick brown fox'?",
            "ref_answer": "4",
            "ref_traj":   [("word_count", "the quick brown fox")],
        },
        {
            "question":   "What is the area of rectangle 6 by 4?",
            "ref_answer": "24",
            "ref_traj":   [("calculator", "6*4")],
        },
    ]

    for task in tasks:
        llm.reset()
        agent = ReActAgent(llm, TOOLS, max_steps=8)
        print(f"\n  Q: {task['question']}")
        answer = agent.run(task["question"])
        traj_acc = trajectory_accuracy(agent.trajectory, task["ref_traj"])
        f1 = answer_f1(answer, task["ref_answer"])
        print(f"  → Answer: {answer}")
        print(f"  → Traj accuracy: {traj_acc:.2f}  |  Answer F1: {f1:.2f}")

    section("2. ERROR RECOVERY")
    print("  Calling calculator with invalid input...")
    llm2 = MockLLM()
    agent2 = ReActAgent(llm2, TOOLS, max_steps=5)

    # Directly test error path
    try:
        result = _calculator("import os; os.system('ls')")
    except ValueError as e:
        print(f"  Calculator correctly rejected: {e}")

    # Test unknown tool handling
    class BadLLM:
        def __call__(self, prompt, temperature=0.0):
            if "Error" in prompt:
                return "Thought: The tool failed. Let me use calculator instead.\n" \
                       "Action: calculator[6*4]\n"
            return "Thought: I'll use unknown_tool.\nAction: unknown_tool[test]"

    bad_llm = BadLLM()
    print("\n  Agent with bad LLM (calls unknown tool first):")
    # Agent should handle unknown tool gracefully
    # We simulate just 2 steps manually
    obs = "Error: unknown tool 'unknown_tool'. Available: ['calculator', 'word_count', 'read_file']"
    print(f"  Observation after bad tool: {obs}")
    print("  Agent would retry with known tool on next step.")

    section("3. SELF-CONSISTENCY VOTING")
    print("  Running 3 independent samples on: 'What is 2**10 + sqrt(144)?'")
    majority = self_consistent_answer(MockLLM, "2**10 + sqrt(144)", k=3)
    print(f"  Majority vote answer: {majority}")

    section("4. CHAIN-OF-THOUGHT PROMPTS")
    cot_prompt = (
        "Q: John has 3 apples and gives 1 away. How many? "
        "Let's think step by step.\n"
        "A: John starts with 3. Gives 1. 3 - 1 = 2. The answer is 2.\n\n"
        "Q: What is 2**10 + sqrt(144)? Let's think step by step.\n"
        "A:"
    )
    print("  Few-shot CoT prompt structure:")
    print("  ", cot_prompt[:200], "...")
    print("\n  Zero-shot suffix: append 'Let's think step by step.' to any question")

    section("5. EVALUATION SUMMARY")
    eval_data = [
        {"question": "2**10 + sqrt(144)", "ref_answer": "1036.0",
         "ref_traj": [("calculator","2**10"),("calculator","144**0.5")]},
        {"question": "words in 'the quick brown fox'", "ref_answer": "4",
         "ref_traj": [("word_count","the quick brown fox")]},
    ]

    successes, f1s, effs = 0, [], []
    for ex in eval_data:
        llm.reset()
        agent = ReActAgent(llm, TOOLS, max_steps=6)
        ans = agent.run(ex["question"])
        f1  = answer_f1(ans, ex["ref_answer"])
        eff = len(ex["ref_traj"]) / max(len(agent.trajectory), 1)
        f1s.append(f1); effs.append(eff)
        successes += int(f1 >= 0.8)

    print(f"\n  Tasks: {len(eval_data)}")
    print(f"  Success rate (F1≥0.8): {successes}/{len(eval_data)}")
    print(f"  Mean answer F1:        {sum(f1s)/len(f1s):.3f}")
    print(f"  Mean step efficiency:  {sum(effs)/len(effs):.3f}")


if __name__ == "__main__":
    main()
