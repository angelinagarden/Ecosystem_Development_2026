"""
Multi-step LLM agent with one tool.

Same triad. Two upgrades vs agent 2:
  1. Up to MAX_STEPS thinking steps per URL (classical ReAct loop).
  2. One tool `fetch_url` — the agent can open the page before answering.
"""

import subprocess
import urllib.request
import ssl
from urllib.parse import urlparse

# macOS Python often lacks a wired-up cert store. Teaching demo only.
_SSL_CTX = ssl._create_unverified_context()


# ---------- PERCEPTION ----------
def perceive(url: str) -> dict:
    parsed = urlparse(url)
    return {
        "url": url,
        "scheme": parsed.scheme,
        "domain": parsed.netloc.lower(),
        "path": parsed.path.lower(),
    }


# ---------- TOOLS ----------
# The agent's only reach into the world. Errors are returned as strings
# so the agent stays alive on bad URLs / network failures.
def fetch_url(url: str, max_chars: int = 4000) -> str:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LLMAgent/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as response:
            raw = response.read(max_chars * 4)
        return raw.decode("utf-8", errors="replace")[:max_chars]
    except Exception as e:
        return f"[fetch failed: {e}]"


# Name → function. The LLM only emits names; code dispatches here.
# Adding a tool = adding one row.
TOOLS = {
    "fetch_url": fetch_url,
}


# ---------- THINKING ----------
# System prompt = the thinking block's "OS": lists tools, defines the
# TOOL/ANSWER protocol, forbids anything else. Classical ReAct: LLM
# proposes, Python executes, observation goes back into context.
SYSTEM = """You are an agent that classifies web resources.

You have ONE tool available:
  fetch_url(url) -> returns the first 4000 characters of the page at that URL.

On each turn, respond with EXACTLY ONE line, in ONE of two formats:
  TOOL: fetch_url <url>
  ANSWER: <one short sentence describing what the URL is>

Rules:
- If you have not fetched the URL yet, respond with TOOL first.
- After you receive the fetched content, respond with ANSWER.
- Do not invent other tools. Do not use any built-in tools.
- ANSWER should be ONE short English sentence describing what the page is.
"""


def think(context: str) -> str:
    """Ask the LLM for its next line. `claude -p` is stateless — we
    rebuild the whole prompt (SYSTEM + running context) each call."""
    prompt = SYSTEM + "\n\n=== CURRENT CONTEXT ===\n" + context
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        return f"ANSWER: [claude error: {result.stderr.strip() or 'unknown'}]"
    # LLM sometimes prefaces its answer — take the first valid line.
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("TOOL:") or line.startswith("ANSWER:"):
            return line
    return "ANSWER: " + result.stdout.strip()


# ---------- TRACING ----------
# Minimal observability — flip to False to silence. Without this, a
# probabilistic brain is a black box during a lecture.
TRACE = True


def trace(msg: str) -> None:
    if TRACE:
        print(f"  [trace] {msg}")


# ---------- AGENT LOOP ----------
# Up to MAX_STEPS think-dispatch cycles per URL. Context grows with
# tool observations between steps and is thrown away between URLs.
MAX_STEPS = 4


def agent_step(url: str) -> str:
    percept = perceive(url)
    trace(f"new percept: {percept['url']}")
    context = f"URL to classify: {percept['url']}"

    for step in range(1, MAX_STEPS + 1):
        trace(f"step {step}: thinking...")
        decision = think(context)
        trace(f"step {step}: LLM decided -> {decision[:120]}")

        if decision.startswith("TOOL:"):
            body = decision[len("TOOL:"):].strip()
            tool_name, _, arg = body.partition(" ")
            arg = arg.strip()
            if tool_name not in TOOLS:
                trace(f"step {step}: unknown tool '{tool_name}', aborting")
                return f"[agent tried unknown tool: {tool_name}]"
            trace(f"step {step}: running {tool_name}({arg})")
            observation = TOOLS[tool_name](arg)
            trace(
                f"step {step}: observation received "
                f"({len(observation)} chars, starts with: {observation[:80]!r})"
            )
            context += (
                f"\n\nStep {step}: called {tool_name}({arg})"
                f"\nObservation (first 4000 chars):\n{observation}"
            )
            continue

        if decision.startswith("ANSWER:"):
            trace(f"step {step}: agent answered, done")
            return decision[len("ANSWER:"):].strip()

        trace(f"step {step}: malformed decision, aborting")
        return f"[malformed decision: {decision}]"

    trace(f"step limit ({MAX_STEPS}) reached, giving up")
    return "[step limit reached without an answer]"


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for url in sys.argv[1:]:
            print(agent_step(url))
        sys.exit(0)

    print("Multi-step LLM agent ready. Paste a URL and press Enter.")
    print("(each URL triggers multiple LLM calls — expect ~10-30 seconds)")
    print("Empty line to quit.")
    while True:
        try:
            url = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not url:
            break
        print(agent_step(url))
