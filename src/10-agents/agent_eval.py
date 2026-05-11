"""
Agent evaluation framework: trajectory accuracy, tool metrics, answer F1,
self-consistency, and benchmark suite.
Covers: ground-truth comparison, tool precision/recall, step efficiency,
        aggregate metrics table.
No external deps required.
"""

import re, math
from collections import Counter

rng = None   # no randomness in evaluation


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Text Metrics ──────────────────────────────────────────────────
def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, reference: str) -> bool:
    return normalize(prediction) == normalize(reference)


def answer_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(normalize(prediction).split())
    ref_tokens  = set(normalize(reference).split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    tp = len(pred_tokens & ref_tokens)
    p  = tp / len(pred_tokens)
    r  = tp / len(ref_tokens)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def bleu_1(prediction: str, reference: str) -> float:
    """Unigram BLEU without brevity penalty."""
    pred_tokens = normalize(prediction).split()
    ref_tokens  = normalize(reference).split()
    if not pred_tokens:
        return 0.0
    ref_count = Counter(ref_tokens)
    matches = sum(min(count, ref_count.get(t, 0)) for t, count in Counter(pred_tokens).items())
    return matches / len(pred_tokens)


# ── Trajectory Metrics ─────────────────────────────────────────────
def trajectory_accuracy(predicted: list, reference: list) -> float:
    """
    Order-insensitive: fraction of reference (tool, input) pairs
    that appear in predicted trajectory.
    """
    if not reference:
        return 1.0
    pred_set = set(predicted)
    ref_set  = set(reference)
    return len(pred_set & ref_set) / len(ref_set)


def tool_precision(predicted: list, reference: list) -> float:
    """Fraction of predicted tool calls that were necessary (in reference)."""
    if not predicted:
        return 1.0
    ref_set = set(reference)
    return sum(1 for p in predicted if p in ref_set) / len(predicted)


def tool_recall(predicted: list, reference: list) -> float:
    """Fraction of necessary tool calls that were made."""
    if not reference:
        return 1.0
    pred_set = set(predicted)
    return sum(1 for r in reference if r in pred_set) / len(reference)


def step_efficiency(predicted_steps: int, reference_steps: int) -> float:
    """reference_steps / predicted_steps — 1.0 = optimal, < 1.0 = too many steps."""
    if predicted_steps == 0:
        return 0.0
    return reference_steps / predicted_steps


def tool_f1(predicted: list, reference: list) -> float:
    p = tool_precision(predicted, reference)
    r = tool_recall(predicted, reference)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


# ── Evaluation Dataset ─────────────────────────────────────────────
EVAL_DATASET = [
    {
        "id": "math_01",
        "question": "What is 2**10 + sqrt(144)?",
        "reference_answer": "1036.0",
        "reference_trajectory": [
            ("calculator", "2**10"),
            ("calculator", "sqrt(144)"),
        ],
        "reference_steps": 2,
        "category": "math",
    },
    {
        "id": "math_02",
        "question": "What is pi times 5 squared?",
        "reference_answer": "78.54",
        "reference_trajectory": [("calculator", "pi*5**2")],
        "reference_steps": 1,
        "category": "math",
    },
    {
        "id": "text_01",
        "question": "How many words are in 'the quick brown fox jumps over'?",
        "reference_answer": "6",
        "reference_trajectory": [("word_count", "the quick brown fox jumps over")],
        "reference_steps": 1,
        "category": "text",
    },
    {
        "id": "text_02",
        "question": "Convert 'hello world' to uppercase.",
        "reference_answer": "HELLO WORLD",
        "reference_trajectory": [("string_transform", "hello world")],
        "reference_steps": 1,
        "category": "text",
    },
    {
        "id": "multi_01",
        "question": "Give me the mean and std of [10, 20, 30, 40, 50].",
        "reference_answer": "mean 30.0 std 14.14",
        "reference_trajectory": [
            ("get_stats", "mean"),
            ("get_stats", "std"),
        ],
        "reference_steps": 2,
        "category": "multi_tool",
    },
    {
        "id": "multi_02",
        "question": "What is log(e^2) and what is sqrt(81)?",
        "reference_answer": "log e 2 = 2.0 and sqrt 81 = 9.0",
        "reference_trajectory": [
            ("calculator", "log(math.e**2)"),
            ("calculator", "sqrt(81)"),
        ],
        "reference_steps": 2,
        "category": "multi_tool",
    },
]


# ── Mock Agent Runs ───────────────────────────────────────────────
def mock_agent_run(example: dict, agent_type: str) -> dict:
    """
    Returns simulated agent results for different agent quality levels.
    agent_type: "optimal" | "verbose" | "wrong_tool" | "correct"
    """
    qid = example["id"]
    ref = example["reference_trajectory"]
    ref_ans = example["reference_answer"]

    if agent_type == "optimal":
        return {
            "predicted_answer":     ref_ans,
            "predicted_trajectory": ref,
            "predicted_steps":      example["reference_steps"],
        }

    if agent_type == "verbose":
        # correct answer but uses extra unnecessary steps
        extra = [("search", f"how to solve {qid}")] + ref
        return {
            "predicted_answer":     ref_ans,
            "predicted_trajectory": extra,
            "predicted_steps":      example["reference_steps"] + 1,
        }

    if agent_type == "wrong_tool":
        # calls tools but gets wrong result
        wrong_traj = [("search", f"answer to {qid}")]
        return {
            "predicted_answer":     "I searched but couldn't find it",
            "predicted_trajectory": wrong_traj,
            "predicted_steps":      1,
        }

    if agent_type == "partial":
        # uses first tool correctly but misses second
        return {
            "predicted_answer":     ref_ans.split()[0] + " (partial)",
            "predicted_trajectory": ref[:1],
            "predicted_steps":      1,
        }

    return {"predicted_answer": "unknown", "predicted_trajectory": [], "predicted_steps": 0}


# ── Evaluation Runner ──────────────────────────────────────────────
def evaluate_agent(agent_type: str, dataset: list) -> dict:
    metrics = {
        "exact_matches": 0,
        "f1s":           [],
        "bleus":         [],
        "traj_accs":     [],
        "tool_precs":    [],
        "tool_recs":     [],
        "tool_f1s":      [],
        "efficiencies":  [],
    }

    for ex in dataset:
        result = mock_agent_run(ex, agent_type)

        pred_ans  = result["predicted_answer"]
        pred_traj = result["predicted_trajectory"]
        pred_steps = result["predicted_steps"]

        ref_ans   = ex["reference_answer"]
        ref_traj  = ex["reference_trajectory"]
        ref_steps = ex["reference_steps"]

        metrics["exact_matches"] += int(exact_match(pred_ans, ref_ans))
        metrics["f1s"].append(answer_f1(pred_ans, ref_ans))
        metrics["bleus"].append(bleu_1(pred_ans, ref_ans))
        metrics["traj_accs"].append(trajectory_accuracy(pred_traj, ref_traj))
        metrics["tool_precs"].append(tool_precision(pred_traj, ref_traj))
        metrics["tool_recs"].append(tool_recall(pred_traj, ref_traj))
        metrics["tool_f1s"].append(tool_f1(pred_traj, ref_traj))
        metrics["efficiencies"].append(step_efficiency(pred_steps, ref_steps))

    n = len(dataset)

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    return {
        "agent_type":       agent_type,
        "exact_match_rate": metrics["exact_matches"] / n,
        "mean_answer_f1":   avg(metrics["f1s"]),
        "mean_bleu1":       avg(metrics["bleus"]),
        "mean_traj_acc":    avg(metrics["traj_accs"]),
        "mean_tool_prec":   avg(metrics["tool_precs"]),
        "mean_tool_rec":    avg(metrics["tool_recs"]),
        "mean_tool_f1":     avg(metrics["tool_f1s"]),
        "mean_efficiency":  avg(metrics["efficiencies"]),
        "n":                n,
    }


# ── Self-Consistency Simulation ───────────────────────────────────
def simulate_self_consistency(answers: list) -> tuple:
    """
    Returns (majority_answer, vote_fraction).
    """
    if not answers:
        return ("", 0.0)
    counts = Counter(answers)
    winner, count = counts.most_common(1)[0]
    return winner, count / len(answers)


# ── Per-Category Breakdown ────────────────────────────────────────
def evaluate_by_category(agent_type: str, dataset: list) -> dict:
    categories = set(ex["category"] for ex in dataset)
    by_cat = {}
    for cat in categories:
        cat_examples = [ex for ex in dataset if ex["category"] == cat]
        result = evaluate_agent(agent_type, cat_examples)
        by_cat[cat] = result
    return by_cat


def main():
    section("1. INDIVIDUAL METRICS")
    pairs = [
        ("1036.0", "1036.0",           "perfect"),
        ("1036.0", "1036",             "near match"),
        ("The answer is 1036", "1036", "verbose correct"),
        ("42",     "1036.0",           "wrong"),
        ("",       "1036.0",           "empty"),
    ]

    print(f"  {'Prediction':<25} {'Reference':<12} {'EM':>4} {'F1':>5} {'BLEU1':>6}")
    print(f"  {'-'*60}")
    for pred, ref, note in pairs:
        em   = exact_match(pred, ref)
        f1   = answer_f1(pred, ref)
        b1   = bleu_1(pred, ref)
        print(f"  {pred:<25} {ref:<12} {str(em):>4} {f1:5.3f} {b1:6.3f}  ({note})")

    section("2. TRAJECTORY METRICS")
    cases = [
        ([("calc","2**10"),("calc","sqrt(144)")],  [("calc","2**10"),("calc","sqrt(144)")], "perfect"),
        ([("calc","2**10"),("search","sqrt")],     [("calc","2**10"),("calc","sqrt(144)")], "partial"),
        ([("search","q"),("calc","2**10"),("calc","sqrt(144)")], [("calc","2**10"),("calc","sqrt(144)")], "verbose"),
        ([("search","q")],                         [("calc","2**10"),("calc","sqrt(144)")], "wrong"),
    ]

    print(f"  {'Case':<10} {'Traj Acc':>9} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Eff(2)':>8}")
    print(f"  {'-'*55}")
    for pred_t, ref_t, label in cases:
        ta  = trajectory_accuracy(pred_t, ref_t)
        prec = tool_precision(pred_t, ref_t)
        rec  = tool_recall(pred_t, ref_t)
        tf1  = tool_f1(pred_t, ref_t)
        eff  = step_efficiency(len(pred_t), 2)
        print(f"  {label:<10} {ta:9.3f} {prec:6.3f} {rec:6.3f} {tf1:6.3f} {eff:8.3f}")

    section("3. AGENT BENCHMARK — 4 AGENT TYPES")
    agent_types = ["optimal", "verbose", "partial", "wrong_tool"]
    all_results = [evaluate_agent(at, EVAL_DATASET) for at in agent_types]

    print(f"\n  {'Agent':<12} {'EM':>5} {'AnswF1':>7} {'TrajAcc':>8} {'ToolF1':>7} {'Eff':>6}")
    print(f"  {'-'*55}")
    for r in all_results:
        print(f"  {r['agent_type']:<12} "
              f"{r['exact_match_rate']:5.2f} "
              f"{r['mean_answer_f1']:7.3f} "
              f"{r['mean_traj_acc']:8.3f} "
              f"{r['mean_tool_f1']:7.3f} "
              f"{r['mean_efficiency']:6.3f}")

    section("4. PER-CATEGORY BREAKDOWN (optimal agent)")
    by_cat = evaluate_by_category("optimal", EVAL_DATASET)
    print(f"  {'Category':<14} {'n':>3} {'EM':>5} {'AnswF1':>7} {'TrajAcc':>9}")
    print(f"  {'-'*45}")
    for cat, r in sorted(by_cat.items()):
        print(f"  {cat:<14} {r['n']:>3} {r['exact_match_rate']:5.2f} "
              f"{r['mean_answer_f1']:7.3f} {r['mean_traj_acc']:9.3f}")

    section("5. SELF-CONSISTENCY SIMULATION")
    # Simulate 5 independent reasoning chains for 3 questions
    sc_scenarios = [
        ("What is 2**10 + sqrt(144)?", ["1036.0","1036.0","1036.0","1024.0","1036.0"]),
        ("What is pi * 5**2?",         ["78.54","78.54","78.5","78.54","79.0"]),
        ("Reverse of 'hello'?",        ["olleh","olleh","hello","olleh","olhel"]),
    ]

    for question, sampled_answers in sc_scenarios:
        winner, fraction = simulate_self_consistency(sampled_answers)
        dist = Counter(sampled_answers)
        print(f"\n  Q: {question}")
        print(f"  Samples: {sampled_answers}")
        print(f"  Majority: '{winner}' (vote fraction: {fraction:.2f})")
        print(f"  Distribution: {dict(dist)}")

    section("6. EVALUATION RECOMMENDATIONS")
    print("""
  When benchmarking an agent system:

  1. Exact Match is too strict for open-ended answers → use F1
  2. Trajectory accuracy catches "right answer, wrong method"
  3. Tool F1 = harmonic mean of precision and recall:
       - High precision: agent doesn't waste tool calls
       - High recall: agent doesn't miss necessary tools
  4. Step efficiency penalizes verbose chains
  5. Self-consistency improves reliability at k×cost
  6. Always evaluate per-category (math vs. text vs. multi-tool)
     to find the agent's specific failure modes

  Typical production thresholds:
    Answer F1 > 0.85  (task success)
    Tool F1 > 0.80    (tool usage correctness)
    Step efficiency > 0.75  (not too verbose)
    Self-consistency k=5 adds ~20% latency for ~8% accuracy gain
""")


if __name__ == "__main__":
    main()
