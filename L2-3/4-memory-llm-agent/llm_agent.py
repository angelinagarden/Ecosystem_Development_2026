"""
LLM agent with memory.

Same multi-step loop as agent 3, plus persistent MEMORY (dict backed by
a JSON file). Two input modes: a URL (cache or classify) or free text
(ask memory). Delete `agent_memory.json` to wipe.
"""

import subprocess
import urllib.request
from urllib.parse import urlparse
import ssl
import json
import pathlib

_SSL_CTX = ssl._create_unverified_context()


# ---------- MEMORY ----------
# A plain dict that outlives agent_step() calls, mirrored to a JSON file
# so it also survives process restarts.
MEMORY: dict = {}
MEMORY_FILE = pathlib.Path(__file__).parent / "agent_memory.json"


def load_memory() -> None:
    """Called once at startup."""
    if MEMORY_FILE.exists():
        try:
            data = json.loads(MEMORY_FILE.read_text())
            MEMORY.update(data)
            print(f"  [memory] loaded {len(MEMORY)} entries from {MEMORY_FILE.name}")
        except Exception as e:
            print(f"  [memory] file exists but is broken ({e}), starting empty")
    else:
        print(f"  [memory] no {MEMORY_FILE.name} yet — starting empty")


def save_memory() -> None:
    """Called after every new entry."""
    MEMORY_FILE.write_text(json.dumps(MEMORY, indent=2, ensure_ascii=False))


# ---------- PERCEPTION ----------
def perceive(url: str) -> dict:
    parsed = urlparse(url)
    return {
        "url": url,
        "scheme": parsed.scheme,
        "domain": parsed.netloc.lower(),
        "path": parsed.path.lower(),
    }


def is_url(text: str) -> bool:
    return text.startswith(("http://", "https://"))


# ---------- TOOLS ----------
def fetch_url(url: str, max_chars: int = 4000) -> str:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LLMAgent/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as response:
            raw = response.read(max_chars * 4)
        text = raw.decode("utf-8", errors="replace")
        return text[:max_chars]
    except Exception as e:
        return f"[fetch failed: {e}]"


TOOLS = {"fetch_url": fetch_url}


# ---------- THINKING (classification loop) ----------
# Unchanged from agent 3.
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
- ANSWER should be ONE short English sentence describing what the page is
  (e.g. "The GitHub repository for the Anthropic Python SDK.").
"""


def think(context: str) -> str:
    prompt = SYSTEM + "\n\n=== CURRENT CONTEXT ===\n" + context
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        return f"ANSWER: [claude error: {result.stderr.strip() or 'unknown'}]"
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("TOOL:") or line.startswith("ANSWER:"):
            return line
    return "ANSWER: " + result.stdout.strip()


# ---------- MEMORY-ANSWERING ----------
# Non-URL input is treated as a question about memory. The whole MEMORY
# is dumped into the prompt; no tools, pure recall. (Agent 5 replaces
# this dump with a tool-mediated search.)
MEMORY_SYSTEM = """You are an agent that has classified some URLs.
Below is the entire contents of your memory: every URL you have seen
and how you classified it. The user is asking you a question about
your memory.

Rules:
- Answer briefly and directly, based ONLY on the memory below.
- If your memory is empty, say "I don't remember anything yet."
- If the user's question cannot be answered from memory, say so.
- Do NOT fetch anything. Do NOT use any tools.
- Answer in English, in one or two short sentences.
"""


def answer_question(question: str) -> str:
    if not MEMORY:
        return "I don't remember anything yet — you haven't asked me about any URL."
    memory_dump = "\n".join(f"- {url} -> {answer}" for url, answer in MEMORY.items())
    prompt = (
        MEMORY_SYSTEM
        + "\n\n=== MEMORY ===\n" + memory_dump
        + "\n\n=== USER QUESTION ===\n" + question
    )
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        return f"[claude error: {result.stderr.strip() or 'unknown'}]"
    return result.stdout.strip()


# ---------- TRACING ----------
TRACE = True


def trace(msg: str) -> None:
    if TRACE:
        print(f"  [trace] {msg}")


# ---------- CLASSIFICATION LOOP ----------
# The multi-step loop from agent 3, lifted out so the dispatcher below
# can call it only on cache miss.
MAX_STEPS = 4


def classify_url(url: str) -> str:
    percept = perceive(url)
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


# ---------- AGENT LOOP ----------
# Dispatcher: URL → cache or classify; non-URL → ask memory. The triad
# itself is unchanged — memory is a layer on top.
def agent_step(input_text: str) -> str:
    trace(f"new percept: {input_text[:100]}")

    if is_url(input_text):
        if input_text in MEMORY:
            trace("MEMORY HIT — skipping loop entirely")
            return f"(from memory) {MEMORY[input_text]}"
        trace("memory miss — running full classification loop")
        answer = classify_url(input_text)
        MEMORY[input_text] = answer
        save_memory()
        trace(
            f"saved to MEMORY (size now: {len(MEMORY)} entries) "
            f"— written to {MEMORY_FILE.name}"
        )
        return answer

    trace("input is not a URL — treating as memory question")
    return answer_question(input_text)


# ---------- MAIN ----------
if __name__ == "__main__":
    import sys

    load_memory()

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(agent_step(arg))
            print()
        sys.exit(0)

    print("Memory LLM agent ready.")
    print("  - Paste a URL to classify (cached after the first time).")
    print("  - Type a question ('what have you seen?') to query memory.")
    print("  - Empty line or Ctrl+C to quit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        print(agent_step(text))
