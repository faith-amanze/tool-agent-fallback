"""
Tool-using agent, version 2: the LLM chooses the tool (function calling).

Differences from tool_agent.py (v1, regex router):
  - The model decides: calculator, search, or answer directly.
  - If the router call itself fails, we fall back to the v1 rule-based agent.
  - Everything else is reused from v1: the tools, the 2s timeout, the
    low-confidence check, the LLM fallback, and the route logging.

Routes logged: TOOL, FALLBACK, DIRECT, FAILED, plus RULE_ROUTER when the
LLM router failed and the v1 regex router handled the query instead.

Run:  python tool_agent_v2.py      (needs GROQ_API_KEY and `pip install groq`)
"""
import json
import os

import tool_agent as v1
from tool_agent import log, ToolError, run_tool, safe_llm, MIN_CONFIDENT_LEN

ROUTER_PROMPT = (
    "You are a demo AI agent with two tools. Use `calculator` for any arithmetic or math "
    "expression. Use `search` for factual lookups about places, topics, or current "
    "information. For greetings or questions about yourself, answer directly without a tool. "
    "If asked who you are, say you are a demo AI agent with a calculator and a search tool, "
    "without naming any underlying model or company. "
    "If you answer directly, use at most 3 sentences of plain text and never invent facts."
)

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate a math expression such as 'sqrt(144) + 5' or '12 * 8'.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Look up factual information about a topic, place or event.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]
ARG_NAME = {"calculator": "expression", "search": "query"}


def get_client():
    """Return a Groq client, or None if no API key / package is available."""
    if not os.getenv("GROQ_API_KEY"):
        return None
    from groq import Groq
    return Groq()


def agent(query: str) -> dict:
    client = get_client()
    if client is None:
        log.warning("no GROQ_API_KEY: using v1 rule-based agent")
        return v1.agent(query)

    # 1. Let the model decide whether to call a tool
    try:
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            max_tokens=400, temperature=0.2,
            tools=TOOL_SCHEMAS, tool_choice="auto",
            messages=[{"role": "system", "content": ROUTER_PROMPT},
                      {"role": "user", "content": query}])
        msg = resp.choices[0].message
    except Exception as e:
        log.error("LLM router failed (%s); using v1 rule-based router", type(e).__name__)
        result = v1.agent(query)
        result["source"] = f"RULE_ROUTER/{result['source']}"
        return result

    # 2. No tool call -> the model answered directly
    if not msg.tool_calls:
        log.info("query=%r -> LLM chose: direct", query)
        return {"answer": (msg.content or "").strip(), "source": "DIRECT", "reason": "model answered directly"}

    # 3. Tool call -> run it with the same safety nets as v1
    call = msg.tool_calls[0]
    name = call.function.name
    log.info("query=%r -> LLM chose tool: %s", query, name)
    try:
        if name not in v1.TOOLS:
            raise ToolError(f"model asked for unknown tool '{name}'")
        try:
            args = json.loads(call.function.arguments or "{}")
            tool_input = str(args[ARG_NAME[name]])
        except (ValueError, KeyError) as e:
            raise ToolError(f"bad tool arguments: {e}")
        output = run_tool(name, tool_input)
        if name == "search" and len(output) < MIN_CONFIDENT_LEN:
            raise ToolError(f"low confidence: output too short ({len(output)} chars)")
        return {"answer": output, "source": "TOOL", "reason": name}
    except ToolError as e:
        log.warning("tool '%s' failed -> falling back to LLM (%s)", name, e)
        answer, ok = safe_llm(query)
        if ok:
            answer = f"(Tool unavailable, answering from general knowledge) {answer}"
        return {"answer": answer, "source": "FALLBACK" if ok else "FAILED", "reason": str(e)}


if __name__ == "__main__":
    tests = [
        "What is sqrt(144) + 5?",
        "What is 12 * 8?",
        "What is 10 / 0?",
        "Search 404 test",
        "Search timeout test",
        "Search quantum gardening",
        "Tell me something about Tunisia",
        "Who are you?",
    ]
    for t in tests:
        r = agent(t)
        print(f"\nQ: {t}\n[{r['source']}] ({r['reason']})\nA: {r['answer']}")
