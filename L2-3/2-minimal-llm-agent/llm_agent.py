"""
Minimal LLM agent.

Same triad as agent 1. Only `think()` changes: rules → LLM via `claude -p`.
No API key — we shell out to Claude Code in headless mode.
"""

import subprocess
from urllib.parse import urlparse


# ---------- PERCEPTION ----------
def perceive(url: str) -> dict:
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "domain": parsed.netloc.lower(),
        "path": parsed.path.lower(),
    }


# ---------- THINKING ----------
# The prompt plays the role RULES played in agent 1: it programs the
# brain. Rules were Python; this is English. Allowed-list is closed on
# purpose — swap it out and the brain's behaviour changes.
PROMPT_TEMPLATE = (
    "You are a URL classifier. Given the parts of a URL below, respond "
    "with EXACTLY ONE lowercase English word describing what kind of "
    "resource it is. Allowed words: video, code, paper, reference, "
    "social, blog, news, docs, shop, forum, unknown. "
    "Output the word only. No punctuation. No explanation.\n\n"
    "scheme: {scheme}\n"
    "domain: {domain}\n"
    "path: {path}"
)


def think(percept: dict) -> str:
    prompt = PROMPT_TEMPLATE.format(**percept)
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        return f"error: {result.stderr.strip() or 'claude call failed'}"
    # LLMs sometimes return extra tokens — take the first, normalize.
    raw = result.stdout.strip().split()
    if not raw:
        return "unknown"
    return raw[0].strip(".,!?:;\"'").lower()


# ---------- ACTION ----------
def act(decision: str, url: str) -> str:
    return f"[{decision}] {url}"


# ---------- AGENT LOOP ----------
def agent_step(url: str) -> str:
    return act(think(perceive(url)), url)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for url in sys.argv[1:]:
            print(agent_step(url))
        sys.exit(0)

    print("LLM agent ready. Paste a URL and press Enter. Empty line to quit.")
    print("(each answer takes a few seconds — LLM round-trip)")
    while True:
        try:
            url = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not url:
            break
        print(agent_step(url))
