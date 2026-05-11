# Project 10: Multi-Agent Crypto Analyst

**Difficulty:** Expert  
**Module:** 10 (Agents & Tool Use)

## 📌 The Challenge
Build a hierarchical agentic system using ReAct loops. The system evaluates a cryptocurrency target by deploying "Researcher", "Quant", and "Summarizer" agents relying exclusively on simple web scraping tools and mock price tools.

## 📖 The Approach
1. **Tool Definition**: Create strictly defined JSON schemas for tools like `get_current_price(symbol)` and `fetch_recent_news(symbol)`.
2. **ReAct Loop Orchestration**: Implement the (Thought ➔ Action ➔ Observation) loop from scratch using standard chat completion APIs, parsing the agent's output.
3. **Hierarchical Routing**: Create a "Manager" context prompt that routes the user's initial prompt either to the Quant (price logic) or the Researcher (news logic) before passing it to the Summarizer.
4. **Memory Injection**: Ensure context is managed effectively so the final Summarizer has access to the observation strings returned by the underlying tools.

## ✅ Checkpoints
- [ ] Define Python schemas and dummy functions for 3 tools.
- [ ] Parse a JSON blob securely out of an LLM generation.
- [ ] Create a `run_agent(prompt)` function that loops until the agent outputs a `FINISH` action.
- [ ] Orchestrate 2 agents in sequence.
