"""
Simple reflex agent (Russell & Norvig, ch. 2).

Perceive → think → act. Rules only, no LLM.
Task: given a URL, say what kind of link it is.
"""

from urllib.parse import urlparse


# ---------- PERCEPTION ----------
# Raw URL → structured percept. The thinking block only sees this dict,
# never the raw string.
def perceive(url: str) -> dict:
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "domain": parsed.netloc.lower(),
        "path": parsed.path.lower(),
    }


# ---------- THINKING ----------
# Condition-action pairs. First match wins. No memory, no planning.
RULES = [
    (lambda p: p["scheme"] not in ("http", "https"), "not-a-web-link"),
    (lambda p: "youtube.com" in p["domain"] or "youtu.be" in p["domain"], "video"),
    (lambda p: "github.com" in p["domain"], "code"),
    (lambda p: "arxiv.org" in p["domain"] or p["path"].endswith(".pdf"), "paper"),
    (lambda p: "wikipedia.org" in p["domain"], "reference"),
    (lambda p: "x.com" in p["domain"] or "twitter.com" in p["domain"], "social"),
    (lambda p: "medium.com" in p["domain"] or "substack.com" in p["domain"], "blog"),
]


def think(percept: dict) -> str:
    for condition, action in RULES:
        if condition(percept):
            return action
    return "unknown"


# ---------- ACTION ----------
# The agent's visible effect. Trivial here; in bigger agents this is
# where a tool call / write / message would go.
def act(decision: str, url: str) -> str:
    return f"[{decision}] {url}"


# ---------- AGENT LOOP ----------
# One tick of the triad. The outer while-loop in __main__ runs many ticks.
def agent_step(url: str) -> str:
    return act(think(perceive(url)), url)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for url in sys.argv[1:]:
            print(agent_step(url))
        sys.exit(0)

    print("Reflex agent ready. Paste a URL and press Enter. Empty line to quit.")
    while True:
        try:
            url = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not url:
            break
        print(agent_step(url))
