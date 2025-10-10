# interactive_cli.py
"""
Interactive CLI for AIResumeAnalyzer

Commands:
  - search <free-text query>
  - ingest            # optional: trigger ingest script (if you want)
  - count             # count points in Qdrant if configured
  - quit / exit
  - help
"""

import sys
import time
import traceback
from typing import Optional

# Import project functions (adjust names if you renamed them)
try:
    from parse_query_gemma import parse_query_with_gemma_local as parse_query_with_gemma
except Exception:
    parse_query_with_gemma = None

try:
    # semantic_search should be implemented in chatbot_poc.py and extended to accept filters
    from chatbot_poc import semantic_search, QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT
except Exception:
    # graceful fallback if import fails
    semantic_search = None
    QDRANT_COLLECTION = None
    QDRANT_HOST = None
    QDRANT_PORT = None

# Optional: small local regex fallback if LLM parsing fails
import re
def _fallback_parse_simple(query: str) -> dict:
    """Cheap fallback: find numeric years and obvious skills/locations via regex."""
    q = query or ""
    # years pattern: "5 years", ">= 5 years", "min 3 yrs"
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:years?|yrs?)', q, flags=re.I)
    years = float(m.group(1)) if m else None
    # naive skills: look for common tokens (extend as needed)
    common_skills = ["python","pytorch","tensorflow","spark","sql","java","c++","ml","machine learning","data engineer","ml engineer"]
    skills = []
    for s in common_skills:
        if re.search(r'\b' + re.escape(s) + r'\b', q, flags=re.I):
            skills.append(s if " " not in s else s.split()[0])  # simple normalization
    # naive location: "from X" or "in X" or trailing token
    loc_m = re.search(r'\bfrom\s+([A-Za-z0-9\-\_ ]{2,40})', q, flags=re.I) or re.search(r'\bin\s+([A-Za-z0-9\-\_ ]{2,40})', q, flags=re.I)
    loc = loc_m.group(1).strip() if loc_m else None
    # role detection simple
    role = None
    if re.search(r'\bml engineer\b', q, flags=re.I) or re.search(r'\bmachine learning\b', q, flags=re.I):
        role = "ml engineer"
    elif re.search(r'\bdata engineer\b', q, flags=re.I):
        role = "data engineer"
    return {
        "role": role,
        "skills": [s.lower() for s in skills],
        "locations": [loc] if loc else [],
        "min_years_experience": years,
        "must_have_keywords": [],
        "must_not_keywords": [],
        "raw_text": q
    }


def _format_candidate(candidate: dict) -> str:
    """Given a candidate dict (point return), format a readable line."""
    pl = (candidate.get("payload") or {}) or {}
    name = pl.get("candidate_name") or pl.get("filename") or "Unknown"

    # Prefer a cleaned numeric total_years_experience if present
    tys = pl.get("total_years_experience")
    yrs_display = "N/A"
    if isinstance(tys, (int, float)):
        # clamp to sensible range
        if 0 <= tys <= 60:
            yrs_display = f"{tys:.1f} yrs"
        else:
            yrs_display = "N/A"
    else:
        # fallback: if experience is list, show role count
        exp = pl.get("experience")
        if isinstance(exp, list):
            yrs_display = f"{len(exp)} roles"

    # If someone accidentally stored a list in 'total_years_experience'
    # or another variable, handle it gracefully:
    if isinstance(tys, (list, tuple)):
        yrs_display = f"{len(tys)} roles"

    # skills
    skills = pl.get("skills_set") or pl.get("skills") or []
    if isinstance(skills, (list, tuple)):
        skills_str = ", ".join(map(str, skills[:8]))
    else:
        skills_str = str(skills)

    # locations
    locs = pl.get("locations") or []
    if isinstance(locs, (list, tuple)) and locs:
        loc_str = str(locs[0])
    elif isinstance(locs, str) and locs:
        loc_str = locs
    else:
        loc_str = "Unknown"

    filename = pl.get("filename") or "n/a"
    score = candidate.get("score")
    score_str = f" score={score:.3f}" if isinstance(score, (float, int)) else ""

    return f"{name} — {yrs_display} — {loc_str} — skills: {skills_str} — file: {filename}{score_str}"



def _print_results(results: list):
    if not results:
        print("No matching candidates found.")
        return
    print(f"✅ Found {len(results)} result(s). Showing top {min(len(results), 10)}:")
    for i, c in enumerate(results[:10], start=1):
        try:
            print(f"{i}. {_format_candidate(c)}")
        except Exception:
            print(f"{i}. [error formatting candidate] {c}")


def search_from_free_text(user_query: str, top_k: int = 20):
    """
    High-level integration:
      - parse user query using Gemma LLM (if available) -> filters
      - call semantic_search(query, filter_skill=..., filter_location=..., min_years_experience=...)
      - return list of candidate dicts
    """
    # 1) parse using Gemma LLM if present
    parsed = None
    if parse_query_with_gemma:
        try:
            parsed = parse_query_with_gemma(user_query)
            # basic validation
            if not isinstance(parsed, dict) or "raw_text" not in parsed:
                parsed = None
        except Exception as e:
            print("LLM parse failed, falling back to regex extractor. Error:", e)
            parsed = None

    if parsed is None:
        parsed = _fallback_parse_simple(user_query)

    # Normalize filters for semantic_search
    skill = None
    if parsed.get("skills"):
        try:
            skill = parsed["skills"][0].strip().lower()
        except Exception:
            skill = None
    loc = None
    if parsed.get("locations"):
        try:
            loc = parsed["locations"][0].strip()
        except Exception:
            loc = None
    minyrs = parsed.get("min_years_experience")

    # Build a semantic query string: include role to bias embedding
    semantic_query = user_query
    if parsed.get("role"):
        semantic_query = f"{parsed['role']} {user_query}"

    if semantic_search is None:
        raise RuntimeError("semantic_search is not imported. Check chatbot_poc.py and its exports.")

    try:
        results = semantic_search(
            query=semantic_query,
            top_k=top_k,
            debug=False,
            filter_skill=skill,
            filter_location=loc,
            min_years_experience=minyrs
        )
    except TypeError:
        # older semantic_search signature: try without filter args (best-effort)
        print("semantic_search signature mismatch; attempting call without structured filters.")
        results = semantic_search(query=semantic_query, top_k=top_k)

    return results


def _main_loop():
    print("AIResumeAnalyzer interactive CLI. Type 'help' for commands.")
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not line:
            continue

        cmd, *rest = line.split(" ", 1)
        cmd = cmd.lower()

        if cmd in ("quit", "exit"):
            print("bye.")
            return

        if cmd == "help":
            print("Commands:")
            print("  search <free-text query>   - search resumes using natural language")
            print("  ingest                     - run ingestion script (if you want to rebuild indexes)")
            print("  count                      - show Qdrant count (if client available in chatbot_poc)")
            print("  quit / exit")
            continue

        if cmd == "count":
            # optional: try to import qdrant client from chatbot_poc or call a helper
            try:
                from qdrant_client import QdrantClient
                from chatbot_poc import QDRANT_HOST as host, QDRANT_PORT as port, QDRANT_COLLECTION as coll
                client = QdrantClient(host=host, port=port)
                c = client.count(collection_name=coll, exact=True)
                total = getattr(c, "count", c if isinstance(c, int) else None)
                print("Total points in collection:", total)
            except Exception as e:
                print("Count failed:", e)
            continue

        if cmd == "ingest":
            # optional: run ingest_resume.py main; warn user
            print("Running ingest (this will recreate collection) ...")
            try:
                import ingest_resume as ingest_mod
                ingest_mod.main()
            except Exception as e:
                print("Ingest failed:", e)
                traceback.print_exc()
            continue

        if cmd == "search":
            if not rest:
                print("Usage: search <your query>")
                continue
            user_query = rest[0].strip()
            if not user_query:
                print("Empty query.")
                continue
            t0 = time.time()
            try:
                results = search_from_free_text(user_query, top_k=20)
                t1 = time.time()
                print(f"[search time: {t1 - t0:.2f}s]")
                _print_results(results)
            except Exception as e:
                print("Search failed:", e)
                traceback.print_exc()
            continue

        print("Unknown command. Type 'help' for available commands.")


if __name__ == "__main__":
    _main_loop()
