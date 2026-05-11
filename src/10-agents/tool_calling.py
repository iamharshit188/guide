"""
Tool schemas, JSON Schema validation, and the OpenAI function-calling format.
Covers: tool registry decorator, schema construction, argument validation,
        parallel tool dispatch, multi-turn tool loop simulation.
No external deps required — simulates the API cycle locally.
"""

import json, math, re, time
from concurrent.futures import ThreadPoolExecutor

rng = None   # no randomness needed


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Tool Registry ─────────────────────────────────────────────────
TOOL_REGISTRY: dict = {}


def tool(name: str, description: str, params: dict):
    """Decorator — registers function with its JSON Schema."""
    def decorator(fn):
        TOOL_REGISTRY[name] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": params,
                }
            }
        }
        return fn
    return decorator


# ── Tool Implementations ──────────────────────────────────────────
@tool(
    name="calculator",
    description="Evaluate a Python math expression. Use ** for powers, sqrt/log/pi available.",
    params={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A safe math expression, e.g. '2**10 + sqrt(144)'"
            }
        },
        "required": ["expression"]
    }
)
def calculator(expression: str) -> float:
    allowed = set("0123456789+-*/().^ eE")
    if not all(c in allowed for c in expression):
        raise ValueError(f"Unsafe characters in expression: '{expression}'")
    return eval(expression, {"__builtins__": {}},
                {"sqrt": math.sqrt, "log": math.log, "pi": math.pi, "abs": abs, "round": round})


@tool(
    name="word_count",
    description="Count the number of words in a text string.",
    params={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to count words in"}
        },
        "required": ["text"]
    }
)
def word_count(text: str) -> int:
    return len(text.split())


@tool(
    name="string_transform",
    description="Transform a string using a specified operation.",
    params={
        "type": "object",
        "properties": {
            "text":      {"type": "string"},
            "operation": {
                "type": "string",
                "enum": ["upper", "lower", "reverse", "title"],
                "description": "Transformation to apply"
            }
        },
        "required": ["text", "operation"]
    }
)
def string_transform(text: str, operation: str) -> str:
    ops = {"upper": str.upper, "lower": str.lower, "title": str.title,
           "reverse": lambda s: s[::-1]}
    if operation not in ops:
        raise ValueError(f"Unknown operation: {operation}")
    return ops[operation](text)


@tool(
    name="get_stats",
    description="Compute basic statistics (mean, min, max, std) for a list of numbers.",
    params={
        "type": "object",
        "properties": {
            "numbers": {
                "type": "array",
                "items": {"type": "number"},
                "description": "List of numbers"
            },
            "stat": {
                "type": "string",
                "enum": ["mean", "min", "max", "std", "all"],
                "description": "Which statistic to compute"
            }
        },
        "required": ["numbers", "stat"]
    }
)
def get_stats(numbers: list, stat: str) -> dict | float:
    n = len(numbers)
    if n == 0:
        raise ValueError("Empty list")
    mean = sum(numbers) / n
    variance = sum((x - mean)**2 for x in numbers) / n
    results = {
        "mean": mean,
        "min":  min(numbers),
        "max":  max(numbers),
        "std":  variance ** 0.5,
    }
    if stat == "all":
        return results
    return results[stat]


# ── JSON Schema Validator ─────────────────────────────────────────
def validate_args(args: dict, schema: dict) -> list:
    """Returns list of validation error strings (empty = valid)."""
    errors = []
    params = schema["parameters"]
    properties = params.get("properties", {})
    required   = params.get("required", [])

    for req in required:
        if req not in args:
            errors.append(f"Missing required field: '{req}'")

    for key, val in args.items():
        if key not in properties:
            continue
        prop = properties[key]
        expected_type = prop.get("type")

        type_map = {"string": str, "number": (int, float), "integer": int,
                    "boolean": bool, "array": list, "object": dict}

        if expected_type and expected_type in type_map:
            if not isinstance(val, type_map[expected_type]):
                errors.append(f"Field '{key}': expected {expected_type}, got {type(val).__name__}")

        if "enum" in prop and val not in prop["enum"]:
            errors.append(f"Field '{key}': '{val}' not in enum {prop['enum']}")

    return errors


# ── Tool Dispatcher ───────────────────────────────────────────────
def dispatch(tool_call: dict) -> str:
    """Execute a tool call (dict with 'function.name' and 'function.arguments')."""
    fn_info = tool_call.get("function", {})
    name    = fn_info.get("name", "")
    raw_args = fn_info.get("arguments", "{}")

    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: '{name}'"})

    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON arguments: {e}"})

    registry_entry = TOOL_REGISTRY[name]
    errors = validate_args(args, registry_entry["schema"]["function"])
    if errors:
        return json.dumps({"error": "Validation failed", "details": errors})

    try:
        result = registry_entry["fn"](**args)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Parallel Tool Dispatch ─────────────────────────────────────────
def dispatch_parallel(tool_calls: list) -> list:
    """
    Execute multiple independent tool calls concurrently.
    Returns list of {role, tool_call_id, content} messages.
    """
    def run_one(tc):
        result = dispatch(tc)
        return {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result}

    with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as pool:
        results = list(pool.map(run_one, tool_calls))

    return results


# ── Mock Multi-Turn Tool Loop ─────────────────────────────────────
class MockFunctionCallingLLM:
    """
    Simulates an LLM that uses function calling for specific prompts.
    Emits structured tool_calls then a final content response.
    """

    PLANS = {
        "statistics": [
            # Round 1: LLM requests 2 parallel tool calls
            {"tool_calls": [
                {"id": "call_1", "function": {"name": "get_stats",
                  "arguments": '{"numbers": [10, 20, 30, 40, 50], "stat": "mean"}'}},
                {"id": "call_2", "function": {"name": "get_stats",
                  "arguments": '{"numbers": [10, 20, 30, 40, 50], "stat": "std"}'}},
            ]},
            # Round 2: LLM produces final answer using tool results
            {"content": "The mean is 30.0 and the standard deviation is 14.14."},
        ],
        "transform": [
            {"tool_calls": [
                {"id": "call_1", "function": {"name": "string_transform",
                  "arguments": '{"text": "hello world", "operation": "upper"}'}},
            ]},
            {"content": "The uppercase version is HELLO WORLD."},
        ],
        "math": [
            {"tool_calls": [
                {"id": "call_1", "function": {"name": "calculator",
                  "arguments": '{"expression": "2**10"}'}},
                {"id": "call_2", "function": {"name": "calculator",
                  "arguments": '{"expression": "sqrt(144)"}'}},
            ]},
            {"content": "2^10 = 1024 and sqrt(144) = 12.0, so their sum is 1036.0."},
        ],
    }

    def __init__(self):
        self._round: dict = {}

    def respond(self, messages: list, task_key: str) -> dict:
        """Simulate one LLM response turn."""
        r = self._round.get(task_key, 0)
        plan = self.PLANS.get(task_key, [{"content": "I don't know."}])
        response = plan[min(r, len(plan) - 1)]
        self._round[task_key] = r + 1
        return response


def run_multi_turn_tool_loop(llm: MockFunctionCallingLLM, task_key: str,
                              user_msg: str, max_rounds: int = 5) -> str:
    messages = [{"role": "user", "content": user_msg}]
    print(f"  User: {user_msg}")

    for round_num in range(max_rounds):
        response = llm.respond(messages, task_key)

        if "tool_calls" in response:
            tool_calls = response["tool_calls"]
            print(f"\n  [Round {round_num+1}] LLM requests {len(tool_calls)} tool call(s):")
            for tc in tool_calls:
                print(f"    → {tc['function']['name']}({tc['function']['arguments'][:60]})")

            # Execute (parallel for multiple independent calls)
            start = time.perf_counter()
            if len(tool_calls) > 1:
                tool_results = dispatch_parallel(tool_calls)
            else:
                tool_results = [{"role": "tool", "tool_call_id": tool_calls[0]["id"],
                                 "content": dispatch(tool_calls[0])}]
            elapsed_ms = (time.perf_counter() - start) * 1000

            print(f"  [Tool results in {elapsed_ms:.1f}ms]:")
            for r in tool_results:
                print(f"    ← {r['content'][:80]}")

            messages.extend(tool_results)

        elif "content" in response:
            print(f"\n  [Round {round_num+1}] Final answer: {response['content']}")
            return response["content"]

    return "Max rounds reached."


def main():
    section("1. TOOL REGISTRY + SCHEMAS")
    print(f"  Registered tools: {list(TOOL_REGISTRY.keys())}")
    for name, entry in TOOL_REGISTRY.items():
        schema = entry["schema"]["function"]
        req = schema["parameters"].get("required", [])
        print(f"  [{name}] required={req}  desc={schema['description'][:50]}...")

    section("2. JSON SCHEMA VALIDATION")
    test_cases = [
        # (tool_name, args, expect_error)
        ("calculator", {"expression": "2**10"}, False),
        ("calculator", {}, True),                              # missing required
        ("string_transform", {"text": "hi", "operation": "upper"}, False),
        ("string_transform", {"text": "hi", "operation": "shout"}, True),  # bad enum
        ("get_stats", {"numbers": [1,2,3], "stat": "mean"}, False),
        ("get_stats", {"numbers": "not_a_list", "stat": "mean"}, True),    # wrong type
    ]
    for name, args, expect_err in test_cases:
        schema = TOOL_REGISTRY[name]["schema"]["function"]
        errors = validate_args(args, schema)
        status = "PASS" if bool(errors) == expect_err else "FAIL"
        print(f"  [{status}] {name}({args}): errors={errors or 'none'}")

    section("3. TOOL DISPATCH — SINGLE CALLS")
    single_calls = [
        {"function": {"name": "calculator",   "arguments": '{"expression": "2**10 + sqrt(144)"}'}},
        {"function": {"name": "word_count",   "arguments": '{"text": "the quick brown fox"}'}},
        {"function": {"name": "string_transform", "arguments": '{"text": "hello", "operation": "upper"}'}},
        {"function": {"name": "get_stats",    "arguments": '{"numbers": [1,2,3,4,5], "stat": "all"}'}},
        {"function": {"name": "calculator",   "arguments": '{"expression": "import os"}'}},  # blocked
    ]
    for tc in single_calls:
        result = dispatch(tc)
        print(f"  {tc['function']['name']}: {result[:80]}")

    section("4. PARALLEL TOOL DISPATCH")
    parallel_calls = [
        {"id": "c1", "function": {"name": "calculator",
          "arguments": '{"expression": "2**10"}'}},
        {"id": "c2", "function": {"name": "calculator",
          "arguments": '{"expression": "3**5"}'}},
        {"id": "c3", "function": {"name": "word_count",
          "arguments": '{"text": "the quick brown fox jumps"}'}},
    ]
    print(f"  Dispatching {len(parallel_calls)} calls concurrently...")
    start = time.perf_counter()
    results = dispatch_parallel(parallel_calls)
    ms = (time.perf_counter() - start) * 1000
    print(f"  Completed in {ms:.1f}ms:")
    for r in results:
        print(f"    tool_call_id={r['tool_call_id']}: {r['content']}")

    section("5. MULTI-TURN TOOL LOOP SIMULATION")
    llm = MockFunctionCallingLLM()

    print("\n--- Task: Compute statistics ---")
    run_multi_turn_tool_loop(llm, "statistics",
                             "What is the mean and standard deviation of [10,20,30,40,50]?")

    print("\n--- Task: Math with 2 parallel calls ---")
    run_multi_turn_tool_loop(llm, "math",
                             "What is 2^10 and sqrt(144)? Give me both.")

    print("\n--- Task: String transform ---")
    run_multi_turn_tool_loop(llm, "transform",
                             "Convert 'hello world' to uppercase.")

    section("6. ALL TOOL SCHEMAS (JSON)")
    for name, entry in TOOL_REGISTRY.items():
        print(f"\n  {name}:")
        print("  " + json.dumps(entry["schema"], indent=2).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
