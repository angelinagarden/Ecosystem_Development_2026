"""
Typed-memory LLM agent.

Same triad as agent 4. Three changes:

  1. `fetch_url` returns a TAGGED dict ({type: html/pdf/image/json/
     redirect/text/error, ...}) instead of raw str. Side effect: PDF /
     PNG / MP4 null-byte crashes from agent 3 go away.

  2. MEMORY now stores CARDS: {url -> {title, author, date, type,
     summary}}. Any field may be null.

  3. `answer_question` is a ReAct loop over two memory tools —
     `memory_stats` and `search_memory` — instead of dumping the whole
     MEMORY into the prompt. Context scales with MEMORY_SEARCH_LIMIT,
     not |MEMORY|. (This is the architectural change.)
"""

import subprocess
import urllib.request
import urllib.error
import urllib.parse
from urllib.parse import urlparse
import ssl
import json
import pathlib
import re
import shutil
import tempfile
import os

_SSL_CTX = ssl._create_unverified_context()


# ---------- NO-AUTO-REDIRECT OPENER ----------
# urllib follows 3xx by default; we want redirects visible as a step
# in the loop. Returning None aborts the default follow — the response
# then surfaces as an HTTPError that fetch_url turns into type=redirect.
class _NoAutoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=_SSL_CTX),
    _NoAutoRedirect(),
)


# ---------- MEMORY ----------
# Same shape as agent 4, but values are cards (dicts) instead of
# one-sentence labels.
MEMORY: dict = {}
MEMORY_FILE = pathlib.Path(__file__).parent / "agent_memory.json"


def load_memory() -> None:
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


# ---------- PER-PERCEPT CACHE ----------
# Raw bytes stashed by fetch_url so a same-percept extract_meta /
# extract_pdf_text can reuse them. Cleared at the start of every
# build_card() — never outlives a single percept.
_PERCEPT_CACHE: dict = {}


# ---------- TOOLS ----------

MAX_BODY_BYTES = 16000
HTML_PREVIEW_CHARS = 500


def _parse_png_dims(raw: bytes) -> str:
    """PNG dimensions live in the IHDR chunk at bytes 16..24."""
    try:
        if len(raw) >= 24 and raw[:8] == b"\x89PNG\r\n\x1a\n":
            w = int.from_bytes(raw[16:20], "big")
            h = int.from_bytes(raw[20:24], "big")
            return f"{w}x{h}"
    except Exception:
        pass
    return "?"


def _parse_jpeg_dims(raw: bytes) -> str:
    """Walk JPEG markers to a Start-Of-Frame segment and read W/H."""
    try:
        if raw[:2] != b"\xff\xd8":
            return "?"
        i = 2
        sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                       0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while i < len(raw) - 9:
            if raw[i] != 0xFF:
                return "?"
            marker = raw[i + 1]
            seg_len = int.from_bytes(raw[i + 2:i + 4], "big")
            if marker in sof_markers:
                h = int.from_bytes(raw[i + 5:i + 7], "big")
                w = int.from_bytes(raw[i + 7:i + 9], "big")
                return f"{w}x{h}"
            i += 2 + seg_len
    except Exception:
        pass
    return "?"


def _looks_like_json(raw: bytes) -> bool:
    stripped = raw.lstrip()
    return stripped[:1] in (b"{", b"[")


def fetch_url(url: str) -> dict:
    """HTTP GET, then classify the response by magic bytes (first) and
    Content-Type (fallback). Returns a JSON-safe tagged dict —
    `type` ∈ {html, pdf, image, json, redirect, text, error}."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LLMAgent/1.0)"},
        )
        resp = _OPENER.open(req, timeout=15)
    except urllib.error.HTTPError as e:
        # _NoAutoRedirect routes 3xx here.
        if 300 <= e.code < 400:
            loc = e.headers.get("Location")
            if loc:
                return {
                    "type": "redirect",
                    "status": e.code,
                    "to": urllib.parse.urljoin(url, loc),
                    "url": url,
                }
        return {"type": "error", "msg": f"HTTP {e.code} {e.reason}", "url": url}
    except Exception as e:
        return {"type": "error", "msg": str(e), "url": url}

    headers = {k.lower(): v for k, v in resp.headers.items()}
    content_type = headers.get("content-type", "").lower()
    raw = resp.read(MAX_BODY_BYTES)
    resp.close()
    _PERCEPT_CACHE[url] = raw

    # Magic bytes first — servers lie about Content-Type.
    if raw.startswith(b"%PDF"):
        return {"type": "pdf", "size_hint": len(raw), "url": url}

    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return {
            "type": "image",
            "format": "png",
            "dims": _parse_png_dims(raw),
            "url": url,
        }
    if raw[:3] == b"\xff\xd8\xff":
        return {
            "type": "image",
            "format": "jpeg",
            "dims": _parse_jpeg_dims(raw),
            "url": url,
        }

    # Fall back to Content-Type / shape.
    if "application/json" in content_type or _looks_like_json(raw):
        try:
            parsed = json.loads(raw.decode("utf-8", errors="replace"))
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
            if len(pretty) > 4000:
                pretty = pretty[:4000] + "\n... (truncated)"
            return {"type": "json", "body": pretty, "url": url}
        except Exception:
            pass

    text = raw.decode("utf-8", errors="replace")
    low = text[:200].lower()
    if ("text/html" in content_type
            or "<!doctype html" in low
            or "<html" in low):
        # Short preview on purpose — the LLM is expected to call
        # extract_meta next for the full card.
        return {
            "type": "html",
            "preview": text[:HTML_PREVIEW_CHARS],
            "size": len(raw),
            "url": url,
        }

    return {"type": "text", "body": text[:4000], "url": url}


def extract_meta(url: str) -> dict:
    """Parse <title>, <meta author>, <time>, OpenGraph fields out of an
    HTML page. Reuses the per-percept cache if available. Any field
    may be None. Regex-based — a real agent would use a proper parser."""
    raw = _PERCEPT_CACHE.get(url)
    if raw is None:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; LLMAgent/1.0)"},
            )
            with _OPENER.open(req, timeout=15) as resp:
                raw = resp.read(MAX_BODY_BYTES)
            _PERCEPT_CACHE[url] = raw
        except Exception as e:
            return {"error": f"fetch failed: {e}", "url": url}

    body = raw.decode("utf-8", errors="replace")

    def _find(pattern):
        m = re.search(pattern, body, flags=re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None

    out = {
        "title": _find(r"<title[^>]*>(.*?)</title>"),
        "author": (
            _find(r'<meta[^>]+(?:name|property)=["\']author["\'][^>]+content=["\']([^"\']+)["\']')
            or _find(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\']author["\']')
            or _find(r'<meta[^>]+property=["\']article:author["\'][^>]+content=["\']([^"\']+)["\']')
            or _find(r'<meta[^>]+name=["\']citation_author["\'][^>]+content=["\']([^"\']+)["\']')
        ),
        "date": (
            # name/property first, content second
            _find(r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']')
            or _find(r'<meta[^>]+property=["\']og:article:published_time["\'][^>]+content=["\']([^"\']+)["\']')
            or _find(r'<meta[^>]+name=["\']citation_date["\'][^>]+content=["\']([^"\']+)["\']')
            or _find(r'<meta[^>]+name=["\']citation_online_date["\'][^>]+content=["\']([^"\']+)["\']')
            or _find(r'<meta[^>]+name=["\']citation_publication_date["\'][^>]+content=["\']([^"\']+)["\']')
            or _find(r'<meta[^>]+name=["\']date["\'][^>]+content=["\']([^"\']+)["\']')
            or _find(r'<meta[^>]+name=["\']pubdate["\'][^>]+content=["\']([^"\']+)["\']')
            # content first, name/property second (some CMSes emit this order)
            or _find(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']article:published_time["\']')
            or _find(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_date["\']')
            or _find(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']date["\']')
            # HTML5 fallback
            or _find(r'<time[^>]+datetime=["\']([^"\']+)["\']')
        ),
        "og_title": _find(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']'),
        "og_description": _find(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']'),
        "url": url,
    }
    if out["title"]:
        out["title"] = re.sub(r"\s+", " ", out["title"]).strip()
    return out


def extract_pdf_text(url: str, max_chars: int = 4000) -> dict:
    """Run `pdftotext` on the PDF (first ~2 pages). Returns
    {available: False, reason: ...} if the tool isn't installed or
    the fetch fails — the agent then falls back gracefully."""
    if not shutil.which("pdftotext"):
        return {"available": False, "reason": "pdftotext not installed", "url": url}

    raw = _PERCEPT_CACHE.get(url)
    # Per-percept cache caps at 16 KB — too small for a real PDF.
    if raw is None or not raw.startswith(b"%PDF") or len(raw) < 50_000:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; LLMAgent/1.0)"},
            )
            with _OPENER.open(req, timeout=30) as resp:
                raw = resp.read(2_000_000)
        except Exception as e:
            return {"available": False, "reason": f"fetch failed: {e}", "url": url}

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(raw)
            pdf_path = f.name
    except Exception as e:
        return {"available": False, "reason": f"tempfile failed: {e}", "url": url}

    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-l", "2", pdf_path, "-"],
            capture_output=True, text=True, timeout=20,
        )
    finally:
        try:
            os.unlink(pdf_path)
        except Exception:
            pass

    if result.returncode != 0:
        return {
            "available": False,
            "reason": f"pdftotext failed: {result.stderr.strip()[:200]}",
            "url": url,
        }

    text = (result.stdout or "")[:max_chars]
    return {"available": True, "body": text, "url": url}


TOOLS = {
    "fetch_url": fetch_url,
    "extract_meta": extract_meta,
    "extract_pdf_text": extract_pdf_text,
}


# ---------- MEMORY TOOLS ----------
# Separate registry from TOOLS so the memory-question branch can't
# accidentally hit the network, and the URL branch can't accidentally
# search memory.

MEMORY_SEARCH_LIMIT = 5


def memory_stats(_arg: str = "") -> dict:
    """Scope of memory: total size, counts by card type, list of URLs.
    Used for "how many X?" / "do you remember Y?" questions without
    dumping card contents into the prompt."""
    counts: dict = {}
    for card in MEMORY.values():
        t = card.get("type") or "unknown"
        counts[t] = counts.get(t, 0) + 1
    return {
        "size": len(MEMORY),
        "types": counts,
        "urls": list(MEMORY.keys()),
    }


def search_memory(query: str) -> dict:
    """Lowercase substring search across url + title + author + summary
    + type. Returns up to MEMORY_SEARCH_LIMIT matches plus a `total`.
    Dumb on purpose — the teaching point is that the agent calls a
    search tool at all, not the quality of the ranker."""
    q = (query or "").lower().strip()
    if not q:
        return {"query": query, "total": 0, "returned": 0, "matches": {}}

    hits = []
    for url, card in MEMORY.items():
        hay = " ".join([
            url,
            str(card.get("title") or ""),
            str(card.get("author") or ""),
            str(card.get("summary") or ""),
            str(card.get("type") or ""),
        ]).lower()
        if q in hay:
            hits.append((url, card))

    total = len(hits)
    hits = hits[:MEMORY_SEARCH_LIMIT]
    return {
        "query": query,
        "total": total,
        "returned": len(hits),
        "matches": {url: card for url, card in hits},
    }


MEMORY_TOOLS = {
    "memory_stats": memory_stats,
    "search_memory": search_memory,
}


# ---------- THINKING (card-building loop) ----------
# Three tools, labelled multi-line card in the ANSWER block instead of
# a one-liner. Same TOOL/ANSWER protocol as agent 3.
SYSTEM = """You are an agent that builds a METADATA CARD for a web resource.

You have THREE tools:
  fetch_url(url)
      HTTP GET the URL. Returns a tagged observation:
        type=html     - call extract_meta(url) next, then answer.
        type=pdf      - call extract_pdf_text(url) next, then answer.
        type=image    - you already have everything (format + dims). Answer.
        type=json     - the body IS the data. Answer.
        type=redirect - call fetch_url(<to>) and continue.
        type=text     - treat as plain text. Answer from the body.
        type=error    - report the error in the summary field. Answer.
  extract_meta(url)
      For HTML pages. Returns {title, author, date, og_title, og_description}.
      Any field may be null.
  extract_pdf_text(url)
      For PDFs. Returns {available: true/false, body?}. If unavailable,
      answer with type=pdf and summary like 'PDF document (text not extracted)'.

On each turn respond with EXACTLY ONE of these two shapes:

  Shape A (call a tool, ONE line):
    TOOL: <tool_name> <arg>

  Shape B (final answer, SIX lines in this exact order):
    ANSWER:
    TITLE: <string or NULL>
    AUTHOR: <string or NULL>
    DATE: <string or NULL>
    TYPE: <one of: paper, code, video, blog, news, docs, forum, image, pdf, data, reference, social, shop, redirect, error, unknown>
    SUMMARY: <one short sentence>

Rules:
- Use the literal word NULL (uppercase) for missing fields.
- Do not invent other tools. Do not use any built-in tools.
- For html observations, you MUST call extract_meta before answering.
- For pdf observations, you MUST call extract_pdf_text before answering.
- For redirect observations, call fetch_url on the target URL.
"""


def think(context: str) -> str:
    """Ask the LLM for its next move. The ANSWER is multi-line (the
    labelled card), so we return the whole stdout and let the loop
    parse it."""
    prompt = SYSTEM + "\n\n=== CURRENT CONTEXT ===\n" + context
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        return (
            "ANSWER:\n"
            "TITLE: NULL\n"
            "AUTHOR: NULL\n"
            "DATE: NULL\n"
            "TYPE: error\n"
            f"SUMMARY: claude error: {result.stderr.strip() or 'unknown'}"
        )
    return result.stdout.strip()


# ---------- ANSWER PARSING ----------
# Forgiving: unknown lines ignored, missing fields → None, NULL-ish
# values normalized to None.
CARD_KEYS = ("title", "author", "date", "type", "summary")


def _parse_card(blob: str) -> dict:
    card = {k: None for k in CARD_KEYS}
    for line in blob.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if key in card:
            if val.lower() in ("null", "none", "n/a", "unknown", "—", "-", ""):
                card[key] = None
            else:
                card[key] = val
    return card


def _decision_kind(decision: str) -> str:
    """'tool' / 'answer' / 'malformed'."""
    first = decision.strip().splitlines()[0].strip() if decision.strip() else ""
    if first.startswith("TOOL:"):
        return "tool"
    if first.upper().startswith("ANSWER"):
        return "answer"
    return "malformed"


# ---------- MEMORY-ANSWERING (search loop) ----------
# Same TOOL/ANSWER protocol as build_card, but the toolbox is
# MEMORY_TOOLS instead of TOOLS. The prompt no longer contains the
# memory dump — it describes HOW TO QUERY memory. This is the
# architectural change vs agent 4.
MEMORY_SYSTEM = """You are an agent answering a question about your own memory.

Your memory is a set of METADATA CARDS indexed by URL. Each card has:
  title, author, date, type, summary.

You have TWO tools for querying memory. You CANNOT see memory directly —
you MUST call a tool to look at anything.

  memory_stats
      No argument. Returns {size, types (counts by type), urls (list of
      all URLs)}. Use for SCOPE questions: "how many papers?", "which
      types do you remember?", "do you remember X?" when "X" is a URL.

  search_memory <query>
      Substring match across url/title/author/summary/type. Returns up
      to 5 matching cards plus the total match count. Use for CONTENT
      questions: "who wrote the GPT-3 paper?", "the article about Y",
      any name/topic/author lookup.

On each turn respond with EXACTLY ONE of these two shapes:

  Shape A (call a tool, ONE line):
    TOOL: <tool_name> <arg>
    (memory_stats takes no argument; search_memory takes a short
    keyword or phrase — not the whole question.)

  Shape B (final answer):
    ANSWER: <one or two short sentences, English>

Rules:
- Prefer specific keywords in search_memory. If one search returns 0
  matches, try a different keyword before giving up.
- Base the answer ONLY on tool results. Do not invent fields.
- If memory_stats shows size=0, answer: "I don't remember anything yet."
- Do not invent other tools. Do not use any built-in tools.
"""


MAX_MEMORY_STEPS = 5


def _run_memory_tool(name: str, arg: str) -> dict:
    """memory_stats ignores its arg; search_memory takes it."""
    fn = MEMORY_TOOLS[name]
    if name == "memory_stats":
        return fn()
    return fn(arg)


def answer_question(question: str) -> str:
    """Loop with MEMORY_TOOLS. ANSWER is a free-form sentence, not a
    labelled card — we return everything after 'ANSWER:' as-is."""
    if not MEMORY:
        return "I don't remember anything yet — you haven't shown me any URL."

    context = f"User question: {question}"

    for step in range(1, MAX_MEMORY_STEPS + 1):
        trace(f"mem step {step}: thinking...")
        prompt = MEMORY_SYSTEM + "\n\n=== CURRENT CONTEXT ===\n" + context
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            return f"[claude error: {result.stderr.strip() or 'unknown'}]"
        decision = result.stdout.strip()
        first_line = decision.splitlines()[0][:120] if decision else "(empty)"
        trace(f"mem step {step}: LLM decided -> {first_line}")
        kind = _decision_kind(decision)

        if kind == "tool":
            body = decision.splitlines()[0].strip()[len("TOOL:"):].strip()
            tool_name, _, arg = body.partition(" ")
            arg = arg.strip()
            if tool_name not in MEMORY_TOOLS:
                trace(f"mem step {step}: unknown tool '{tool_name}', aborting")
                return f"[agent tried unknown memory tool: {tool_name}]"
            trace(f"mem step {step}: running {tool_name}({arg!r})")
            observation = _run_memory_tool(tool_name, arg)
            trace(
                f"mem step {step}: observation keys={list(observation.keys())}"
            )
            context += (
                f"\n\nStep {step}: called {tool_name}({arg})"
                f"\nObservation:\n{_format_observation(observation)}"
            )
            continue

        if kind == "answer":
            trace(f"mem step {step}: agent answered, done")
            # Strip the leading 'ANSWER:' prefix; return the rest as-is.
            first = decision.splitlines()[0]
            tail = first.split(":", 1)[1].strip() if ":" in first else ""
            rest = "\n".join(decision.splitlines()[1:]).strip()
            return (tail + ("\n" + rest if rest else "")).strip() or "(empty answer)"

        trace(f"mem step {step}: malformed decision, aborting")
        return f"[malformed decision: {decision[:120]}]"

    trace(f"mem step limit ({MAX_MEMORY_STEPS}) reached, giving up")
    return "[step limit reached without an answer]"


# ---------- TRACING ----------
TRACE = True


def trace(msg: str) -> None:
    if TRACE:
        print(f"  [trace] {msg}")


# ---------- CARD-BUILDING LOOP ----------
# Same shape as agent 4's classify_url, bumped to MAX_STEPS = 7 so a
# redirect → fetch → extract chain can complete.
MAX_STEPS = 7


def _format_observation(obs: dict) -> str:
    return json.dumps(obs, indent=2, ensure_ascii=False)


def _error_card(summary: str) -> dict:
    return {
        "title": None,
        "author": None,
        "date": None,
        "type": "error",
        "summary": summary,
    }


def build_card(url: str) -> dict:
    """URL → card dict. 2–6 LLM calls depending on the resource type."""
    _PERCEPT_CACHE.clear()

    percept = perceive(url)
    context = f"URL to classify: {percept['url']}"

    for step in range(1, MAX_STEPS + 1):
        trace(f"step {step}: thinking...")
        decision = think(context)
        first_line = decision.splitlines()[0][:120] if decision else "(empty)"
        trace(f"step {step}: LLM decided -> {first_line}")
        kind = _decision_kind(decision)

        if kind == "tool":
            body = decision.strip().splitlines()[0].strip()[len("TOOL:"):].strip()
            tool_name, _, arg = body.partition(" ")
            arg = arg.strip()
            if tool_name not in TOOLS:
                trace(f"step {step}: unknown tool '{tool_name}', aborting")
                return _error_card(f"agent tried unknown tool: {tool_name}")
            trace(f"step {step}: running {tool_name}({arg})")
            observation = TOOLS[tool_name](arg)
            trace(
                f"step {step}: observation type={observation.get('type', '?')}, "
                f"keys={list(observation.keys())}"
            )
            context += (
                f"\n\nStep {step}: called {tool_name}({arg})"
                f"\nObservation:\n{_format_observation(observation)}"
            )
            continue

        if kind == "answer":
            trace(f"step {step}: agent answered, done")
            lines = decision.strip().splitlines()
            if lines and lines[0].strip().upper().startswith("ANSWER"):
                lines = lines[1:]
            return _parse_card("\n".join(lines))

        trace(f"step {step}: malformed decision, aborting")
        return _error_card(f"malformed decision: {decision[:120]}")

    trace(f"step limit ({MAX_STEPS}) reached, giving up")
    return _error_card("step limit reached without an answer")


# ---------- AGENT LOOP ----------
# Dispatcher: URL → build card (or cache hit); non-URL → memory question.
def _fmt_card(card: dict) -> str:
    return (
        f"title:   {card.get('title')}\n"
        f"author:  {card.get('author')}\n"
        f"date:    {card.get('date')}\n"
        f"type:    {card.get('type')}\n"
        f"summary: {card.get('summary')}"
    )


def agent_step(input_text: str) -> str:
    trace(f"new percept: {input_text[:100]}")

    if is_url(input_text):
        if input_text in MEMORY:
            trace("MEMORY HIT — skipping the loop entirely")
            return "(from memory)\n" + _fmt_card(MEMORY[input_text])

        trace("memory miss — running full card-building loop")
        card = build_card(input_text)
        MEMORY[input_text] = card
        save_memory()
        trace(
            f"saved to MEMORY (size now: {len(MEMORY)} entries) "
            f"— written to {MEMORY_FILE.name}"
        )
        return _fmt_card(card)

    trace("input is not a URL — treating as a memory question")
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

    print("Typed-memory LLM agent ready.")
    print("  - Paste a URL to build a card (cached after the first time).")
    print("  - Type a question ('how many papers?') to query the memory.")
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
