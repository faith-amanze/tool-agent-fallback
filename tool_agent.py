"""
Tool-using agent with fallback logic.

Tools:    calculator (safe AST evaluator), search (simulated, can fail)
Routing:  rule-based router picks calculator / search / direct LLM
Fallback: tool error, timeout, or low-confidence output -> answer with the LLM
Logging:  every answer is tagged TOOL, FALLBACK or DIRECT

Run:  python tool_agent.py
Uses Groq if GROQ_API_KEY is set (pip install groq);
otherwise a stub LLM is used so the script runs anywhere.
"""
import ast
import logging
import math
import operator
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("agent")

TOOL_TIMEOUT_S = 2.0
MIN_CONFIDENT_LEN = 40  # search output shorter than this is treated as low confidence


class ToolError(Exception):
    """Raised when a tool fails or returns something unusable."""


# ---------------------------------------------------------------- Tool 1: Calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round,
          "sin": math.sin, "cos": math.cos, "log": math.log}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculator(query: str) -> str:
    expr = re.sub(r"(?i)^(what is|what's|calculate|compute)\s+", "", query.strip()).rstrip("?= ")
    expr = expr.replace("^", "**").replace("x", "*") if re.fullmatch(r"[\dx\s.+\-*/()]+", expr) else expr.replace("^", "**")
    try:
        result = _eval(ast.parse(expr, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError, OverflowError) as e:
        raise ToolError(f"calculator could not evaluate '{expr}': {e}")
    return str(int(result)) if float(result).is_integer() else f"{result:.6g}"


# ---------------------------------------------------------------- Tool 2: Search (simulated)
_KB = {
    "tunisia": "Tunisia is a North African country on the Mediterranean coast, bordered by "
               "Algeria and Libya. Its capital is Tunis, and it is home to the ruins of Carthage.",
    "python": "Python is a high-level, general-purpose programming language known for its "
              "readable syntax and large ecosystem, widely used in data science and web backends.",
    "nigeria": "Nigeria is a West African country and the most populous in Africa. Its capital "
               "is Abuja and its largest city is Lagos.",
}


def search(query: str) -> str:
    q = query.lower()
    if "404" in q or "error" in q:
        raise ToolError("search backend returned HTTP 404")        # simulated failure
    if "timeout" in q:
        time.sleep(TOOL_TIMEOUT_S + 1)                              # simulated hang
    for key, snippet in _KB.items():
        if key in q:
            return snippet
    return "No results."                                            # weak/irrelevant output


TOOLS = {"calculator": calculator, "search": search}


def run_tool(name: str, query: str) -> str:
    """Run a tool with a timeout; normalise every failure into ToolError."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(TOOLS[name], query)
        try:
            return future.result(timeout=TOOL_TIMEOUT_S)
        except FutureTimeout:
            raise ToolError(f"{name} timed out after {TOOL_TIMEOUT_S}s")


# ---------------------------------------------------------------- LLM (with offline stub)
SYSTEM_PROMPT = (
    "You are a demo AI agent. Reply directly in plain text; you cannot call any tools "
    "in this reply, so never try to search or use functions. "
    "If asked who you are, say you are a demo AI agent that normally uses a calculator "
    "and a search tool, without naming any underlying model or company. "
    "Answer in at most 3 sentences with no tables or headings. "
    "If you do not know something or cannot verify it, say so plainly. "
    "Never invent papers, links, people, or statistics."
)
BARE_PROMPT = "Answer in at most 3 plain-text sentences. If you are unsure, say so."


def llm_answer(query: str) -> str:
    if os.getenv("GROQ_API_KEY"):
        from groq import Groq
        client = Groq()
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        last_err = None
        # try the full prompt first, then a minimal one (some models misbehave on long prompts)
        for system in (SYSTEM_PROMPT, BARE_PROMPT):
            try:
                resp = client.chat.completions.create(
                    model=model, max_tokens=400, temperature=0.3,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": query}])
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                log.warning("LLM attempt failed (%s); retrying with simpler prompt", type(e).__name__)
        raise last_err
    if re.search(r"(?i)who are you", query):
        return "I'm a small demo agent that can use a calculator and a search tool, or answer directly."
    return f"[stub LLM] Direct answer to: {query!r}"


# ---------------------------------------------------------------- Router
def choose_route(query: str) -> str:
    q = query.lower()
    if re.search(r"\d", q) and re.search(r"[+\-*/^x]|sqrt|sin\(|cos\(|log\(", q) \
            and not q.startswith("search"):
        return "calculator"
    if re.search(r"\b(search|look up|find|latest|news|tell me (something )?about)\b", q):
        return "search"
    return "direct"


# ---------------------------------------------------------------- Agent
def safe_llm(query: str) -> tuple[str, bool]:
    """Call the LLM; if it also fails, return a plain message instead of crashing."""
    try:
        return llm_answer(query), True
    except Exception as e:  # network, auth, bad model name, rate limit...
        log.error("LLM call failed: %s: %s", type(e).__name__, e)
        return "Sorry, I couldn't answer that right now (tool and LLM both failed).", False


def agent(query: str) -> dict:
    route = choose_route(query)
    log.info("query=%r -> router chose: %s", query, route)

    if route == "direct":
        answer, ok = safe_llm(query)
        return {"answer": answer, "source": "DIRECT" if ok else "FAILED", "reason": "no tool needed"}

    try:
        output = run_tool(route, query)
        # confidence-based fallback: tool 'worked' but output is too thin to trust
        if route == "search" and len(output) < MIN_CONFIDENT_LEN:
            raise ToolError(f"low confidence: output too short ({len(output)} chars)")
        return {"answer": output, "source": "TOOL", "reason": route}
    except ToolError as e:
        log.warning("tool '%s' failed -> falling back to LLM (%s)", route, e)
        answer, ok = safe_llm(query)
        if ok:
            answer = f"(Tool unavailable, answering from general knowledge) {answer}"
        return {"answer": answer, "source": "FALLBACK" if ok else "FAILED", "reason": str(e)}


if __name__ == "__main__":
    tests = [
        "What is sqrt(144) + 5?",              # tool: calculator
        "What is 12 * 8?",                     # tool: calculator
        "What is 10 / 0?",                     # calculator error -> fallback
        "Search 404 test",                     # search error -> fallback
        "Search timeout test",                 # search timeout -> fallback
        "Search quantum gardening",            # weak output -> confidence fallback
        "Tell me something about Tunisia",     # tool: search
        "Who are you?",                        # direct
    ]
    for t in tests:
        r = agent(t)
        print(f"\nQ: {t}\n[{r['source']}] ({r['reason']})\nA: {r['answer']}")
