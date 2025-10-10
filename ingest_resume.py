# ingest_resume.py
"""
Run this once (or when you add new resumes) to create/verify the Qdrant collection
and upload resume chunks with embeddings.

Usage:
    python ingest_resume.py
"""

import os as _os
import uuid as _uuid
import re as _re
import math as _math
import datetime as _datetime
from typing import List as _List

from model import init_skill_model

try:
    from qdrant_client import QdrantClient as _QdrantClient
    from qdrant_client.http import models as qm
except Exception as _e:
    _QdrantClient = None
    qm = None
    print("qdrant-client not available:", _e)
# at top of ingest_resume.py
try:
    # pull canonicalization helpers from chatbot_poc.py (adjust module name if different)
    from chatbot_poc import post_process_skills, map_token_to_canonical
except Exception:
    post_process_skills = None
    map_token_to_canonical = None


# try to import values from chatbot_poc if available (preferred),
# but fall back to sensible defaults so this file can run standalone.
try:
    from chatbot_poc import (
        RESUME_FOLDER as _RESUME_FOLDER,
        QDRANT_HOST as _QDRANT_HOST,
        QDRANT_PORT as _QDRANT_PORT,
        QDRANT_COLLECTION as _QDRANT_COLLECTION,
        embedding_model as _embedding_model,           # may be None if not initialized there
        parse_resume_file as _parse_resume_file,
    )
except Exception as _e:
    # defaults (adjust as needed)
    _RESUME_FOLDER = "./resumes"
    _QDRANT_HOST = "localhost"
    _QDRANT_PORT = 6333
    _QDRANT_COLLECTION = "resumes_collection"
    _embedding_model = None
    # provide a very small fallback parser if chatbot_poc.parse_resume_file not available
    def _parse_resume_file(path: str) -> dict:
        # minimal parser: return filename as candidate_name and full_text via fallback reader
        return {"candidate_name": _os.path.splitext(_os.path.basename(path))[0], "candidate_id": None, "email": None,
                "skills": {}, "locations": [], "sections": {}, "experience": [], "full_text": ""}

_BATCH = 128

def _fallback_read_text(path: str) -> str:
    try:
        if path.lower().endswith('.pdf'):
            try:
                import fitz as _fitz
                doc = _fitz.open(path)
                pages = [p.get_text('text') for p in doc]
                return "\n".join(pages)
            except Exception:
                with open(path, 'rb') as f:
                    return str(f.read()[:5000], errors='replace')
        elif path.lower().endswith('.docx'):
            try:
                import docx as _docx
                d = _docx.Document(path)
                return "\n".join(p.text for p in d.paragraphs)
            except Exception:
                with open(path, 'rb') as f:
                    return str(f.read()[:5000], errors='replace')
        else:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
    except Exception as e:
        print("Warning: fallback text read failed for", path, ":", e)
        return ""

# ----------------- New helper: estimate total years -----------------
def _estimate_total_years_from_experience(exp_list):
    """
    Best-effort: sum durations from parsed experience entries.
    Accepts list of dicts or strings. Returns float years (1 decimal).
    """
    if not exp_list:
        return 0.0
    total_days = 0
    today = _datetime.date.today()

    range_re = _re.compile(r'(\d{4})\s*[-–—]\s*(Present|present|Current|current|(\d{4}))')
    years_re = _re.compile(r'(\d+(?:\.\d+)?)\s+years?', flags=_re.I)
    year_only_re = _re.compile(r'(\d{4})')

    for e in exp_list:
        try:
            if isinstance(e, dict):
                txt = " ".join([str(e.get(k,"")) for k in ("dates","raw","summary","title") if e.get(k)])
            else:
                txt = str(e)
            txt = txt.strip()
            if not txt:
                continue

            # explicit range "2018-2021" or "2019 - Present"
            m = range_re.search(txt)
            if m:
                try:
                    start = int(m.group(1))
                    if m.group(2).lower() in ("present", "current"):
                        end = today.year
                    else:
                        end = int(m.group(2))
                    days = max(0, ( _datetime.date(end,1,1) - _datetime.date(start,1,1) ).days)
                    total_days += days
                    continue
                except Exception:
                    pass

            # "X years"
            m2 = years_re.search(txt)
            if m2:
                try:
                    yrs = float(m2.group(1))
                    total_days += int(yrs * 365)
                    continue
                except Exception:
                    pass

            # fallback: two year tokens
            yrs_found = year_only_re.findall(txt)
            if len(yrs_found) >= 2:
                try:
                    y1 = int(yrs_found[0]); y2 = int(yrs_found[-1])
                    days = max(0, (_datetime.date(y2,1,1) - _datetime.date(y1,1,1)).days)
                    total_days += days
                    continue
                except Exception:
                    pass

        except Exception:
            continue

    years = round(total_days / 365.0, 1)
    if years < 0:
        years = 0.0
    return float(years)
# ----------------- end helper -----------------

def _normalize_vec(v):
    try:
        n = _math.sqrt(sum((float(x) ** 2) for x in v))
        if n > 0:
            return [float(x) / n for x in v]
    except Exception:
        pass
    return v

def _role_focused_text(parsed: dict, full_text: str) -> str:
    """
    Build a compact, role-focused text for candidate-level vector:
    - title / headline
    - summary / objective
    - top skills
    Fallback: first ~120 words of full_text if above is empty.
    """
    sections = parsed.get("sections") or {}
    title = str(sections.get("title") or sections.get("headline") or "").strip()
    summary = str(sections.get("summary") or sections.get("objective") or "").strip()

    skills = parsed.get("skills") or []
    if isinstance(skills, dict):
        skill_terms = list(skills.keys())
    elif isinstance(skills, (list, tuple)):
        skill_terms = [str(s) for s in skills if s]
    else:
        skill_terms = []

    parts = []
    if title:
        parts.append(title)
    if summary:
        parts.append(summary)
    if skill_terms:
        parts.append("skills: " + ", ".join(skill_terms[:80]))

    role_text = " | ".join([p for p in parts if p]).strip()
    if not role_text:
        role_text = " ".join((full_text.split() or [])[:120])

    return role_text

def _smart_chunk(full_text: str):
    """
    Chunk the resume text with paragraph merge + windowing.
    Returns a list of chunk strings.
    """
    # params
    min_words = 40
    max_words = 400
    window = 300
    stride = 200
    per_file_chunk_cap = 120

    paras = [p.strip() for p in _re.split(r'\n{2,}', full_text) if p.strip()]

    # merge until min_words
    merged = []
    cur = ""
    for p in paras:
        cur = (cur + "\n\n" + p).strip() if cur else p
        if len(cur.split()) >= min_words:
            merged.append(cur)
            cur = ""
    if cur:
        merged.append(cur)

    if not merged:
        if full_text:
            merged = [full_text[:400]]
        else:
            return []

    # split very large chunks via sliding windows
    final_chunks = []
    for ch in merged:
        words = ch.split()
        if len(words) <= max_words:
            final_chunks.append(ch)
        else:
            i = 0
            L = len(words)
            while i < L:
                final_chunks.append(" ".join(words[i:i + window]))
                if i + window >= L:
                    break
                i += stride

    if len(final_chunks) > per_file_chunk_cap:
        final_chunks = final_chunks[:per_file_chunk_cap]

    return final_chunks
def _gather_points(resume_folder: str) -> _List[dict]:
    """
    Parse resumes, compute chunk embeddings, and attach a role-focused
    _candidate_vector to each chunk payload for candidate-level reranking.

    Improvements:
     - build a canonical `skills_set` for each candidate using post_process_skills()
       (or a conservative fallback) so Qdrant payloads contain normalized tokens.
     - ensure deterministic candidate_id (email or filename).
     - limit chunks per file to `per_file_chunk_cap`.
    """
    # Attempt to import post-processing utilities from chatbot_poc; tolerate failure
    try:
        from chatbot_poc import post_process_skills, map_token_to_canonical
    except Exception:
        post_process_skills = None
        map_token_to_canonical = None

    files = [f for f in _os.listdir(resume_folder) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    print(f"Found {len(files)} files in '{resume_folder}'")
    if not files:
        raise FileNotFoundError(f"No resume files found in '{resume_folder}'. Put .pdf/.docx/.txt files there.")

    if _embedding_model is None:
        raise RuntimeError("Embedding model not initialized. Call init_skill_model(...) before gathering points.")

    dim = _embedding_model.get_sentence_embedding_dimension()
    points = []

    # chunking params (tune if needed)
    min_words = 40
    max_words = 400
    window = 300
    stride = 200
    per_file_chunk_cap = 120

    # helper to do conservative normalization when post_process_skills is not available
    def _fallback_normalize_skills(raw_skills, full_text):
        out = []
        seen = set()
        if isinstance(raw_skills, dict):
            vals = []
            for v in raw_skills.values():
                if isinstance(v, (list, tuple)):
                    vals.extend([str(x) for x in v if x])
                else:
                    vals.append(str(v))
            raw_list = vals
        elif isinstance(raw_skills, (list, tuple)):
            raw_list = list(raw_skills)
        elif raw_skills:
            raw_list = [str(raw_skills)]
        else:
            raw_list = []

        for s in raw_list:
            if not s: continue
            t = str(s).strip()
            # remove leading/trailing quotes/parenthesis artifacts
            t = _re.sub(r'^[\s"\'`]+|[\s"\'`]+$', '', t)
            t = _re.sub(r'[\(\)]+$', '', t).strip()  # drop trailing parentheses leftovers
            # drop obviously long descriptive phrases
            if len(t.split()) > 6:
                # try to extract embedded tech tokens
                kws = _re.findall(r'\b(python|scala|java|spark|hadoop|kafka|hive|airflow|sql|snowflake|dynamodb|databricks|azure|aws|gcp|bigquery|dbt)\b', t, flags=re.I)
                if not kws:
                    continue
                else:
                    for k in kws:
                        k = k.lower()
                        if k not in seen:
                            seen.add(k); out.append(k)
                    continue

            t = t.lower()
            # filter trivial tokens
            if t in ("and","or","with","experience","skills","skill","years","year","the","a","an","in","of","for","is","are"):
                continue
            if t not in seen:
                seen.add(t); out.append(t)
        return out

    for fn in files:
        path = _os.path.join(resume_folder, fn)
        print("Parsing:", fn)
        try:
            parsed = _parse_resume_file(path)
        except Exception as e:
            print(f"Error parsing {fn}: {e}")
            parsed = {
                "candidate_name": _os.path.splitext(fn)[0], "candidate_id": None, "email": None,
                "skills": {}, "locations": [], "sections": {}, "experience": [], "full_text": ""
            }

        full_text = parsed.get("full_text") or ""
        if not full_text:
            print(f"[{fn}] empty full_text after parsing; using fallback reader.")
            try:
                full_text = _fallback_read_text(path)
            except Exception as fe:
                print(f"Fallback read failed for {fn}: {fe}")
                full_text = ""
            parsed["full_text"] = full_text

        # deterministic candidate_id: prefer explicit id -> email -> filename (lowercased)
        cand_id = parsed.get("candidate_id") or parsed.get("email") or _os.path.basename(fn)
        if cand_id:
            cand_id = str(cand_id).strip()
        else:
            cand_id = _os.path.basename(fn)
        candidate_id_norm = cand_id.lower()

        # Compute candidate-level role text and candidate vector (once per file)
        role_text = _role_focused_text(parsed, full_text)
        try:
            cand_vec = _embedding_model.encode(role_text)
            if hasattr(cand_vec, "tolist"):
                cand_vec = cand_vec.tolist()
            cand_vec = _normalize_vec(cand_vec)
        except Exception as e:
            print(f"[{fn}] candidate vector build failed:", e)
            cand_vec = None

        # Precompute total_years and normalized skills_set once per file (faster)
        total_years_for_file = _estimate_total_years_from_experience(parsed.get("experience") or [])

        # raw skills from parser
        raw_skills = parsed.get("skills") or []

        # Build canonical skills_set:
        skills_set = []
        try:
            if post_process_skills is not None:
                processed = post_process_skills(raw_skills, full_text=full_text, whitelist=None, fuzzy_cutoff=0.78)
                # post_process_skills returns canonical or normalized tokens; ensure dedupe & lower
                seen = set()
                for s in processed:
                    if not s: continue
                    s2 = _re.sub(r'^[\W_]+|[\W_]+$', '', str(s)).strip().lower()
                    if not s2: continue
                    # drop overly long phrase artifacts
                    if len(s2.split()) > 6:
                        # try map_token_to_canonical for embedded tech token if available
                        if map_token_to_canonical is not None:
                            m = map_token_to_canonical(s2, fuzzy_cutoff=85.0)
                            if m:
                                s2 = m
                            else:
                                continue
                        else:
                            continue
                    if s2 not in seen:
                        seen.add(s2); skills_set.append(s2)
            else:
                skills_set = _fallback_normalize_skills(raw_skills, full_text)
        except Exception as e:
            print(f"[{fn}] skill post-processing failed: {e}")
            skills_set = _fallback_normalize_skills(raw_skills, full_text)

        # Ensure fallback non-empty skills_set if nothing found (very conservative)
        if not skills_set and raw_skills:
            skills_set = _fallback_normalize_skills(raw_skills, full_text)

        # limit and dedupe
        skills_set = list(dict.fromkeys(skills_set))[:200]

        # chunking (and cap number of chunks per file)
        chunks = _smart_chunk(full_text)
        if not chunks:
            fallback = full_text[:400]
            if fallback:
                chunks = [fallback]
            else:
                print(f"[{fn}] no usable text found; skipping file.")
                continue

        if per_file_chunk_cap and len(chunks) > per_file_chunk_cap:
            # heuristics: keep evenly spaced chunks (avoid biasing to head)
            step = max(1, int(len(chunks) / per_file_chunk_cap))
            chunks = [chunks[i] for i in range(0, len(chunks), step)][:per_file_chunk_cap]

        # embed each chunk
        for chunk in chunks:
            try:
                vec = _embedding_model.encode(chunk)
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                if not isinstance(vec, list) or len(vec) != dim:
                    raise ValueError(f"Bad embedding dim {0 if not vec else len(vec)} (expected {dim})")
                vec = _normalize_vec(vec)
            except Exception as e:
                print(f"[{fn}] embedding error for a chunk; skipping. Err:", e)
                continue

            # Build payload: keep human-readable 'skills' and canonical 'skills_set'
            payload = {
               "filename": fn,
               "candidate_name": parsed.get("candidate_name") or _os.path.splitext(fn)[0],
               "candidate_id": candidate_id_norm,
               "email": parsed.get("email"),
               "skills": parsed.get("skills"),                # human-readable list (unchanged)
               "skills_set": skills_set,                      # canonical normalized tokens (lower-case)
               "locations": parsed.get("locations"),
               "total_years_experience": parsed.get("total_years_experience", total_years_for_file or 0.0),
               "sections": parsed.get("sections"),
               "experience": parsed.get("experience"),
               "full_text": parsed.get("full_text"),
               "text": chunk[:3000],
               "_candidate_vector": cand_vec,
           }

            points.append({"id": str(_uuid.uuid4()), "vector": vec, "payload": payload})

    return points


def _ensure_collection(client: _QdrantClient, collection_name: str, dim: int):
    """
    Recreate the collection with the correct vector size and COSINE distance.
    Safe to call every ingest; it replaces the collection.
    """
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        print(f"Collection '{collection_name}' created/recreated with dim={dim}, metric=COSINE")
    except Exception as e:
        print("Collection ensure/recreate warning:", e)

def _upsert_points(client: _QdrantClient, points: _List[dict], collection_name: str = _QDRANT_COLLECTION):
    for i in range(0, len(points), _BATCH):
        batch = points[i:i + _BATCH]
        try:
            ps = [qm.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in batch]
            client.upsert(collection_name=collection_name, points=ps, wait=True)
            print(f"Upserted batch {i // _BATCH + 1} ({len(batch)} points)")
        except Exception as e:
            print("Qdrant upsert error:", e)

def main():
    # check qdrant client exists
    if _QdrantClient is None:
        print("qdrant-client is not installed or failed to import. Install with `pip install qdrant-client`.")
        return

    # If chatbot_poc didn't set _embedding_model, initialize Gemma here explicitly.
    global _embedding_model
    if _embedding_model is None:
        print("Embedding model not loaded from chatbot_poc; initializing Gemma via model.init_skill_model()")
        try:
            # change device to "cuda" if you'd like GPU
            _embedding_model = init_skill_model(model_id="google/gemma-3-270m-it", backend="transformers", device=None, normalize=True)
        except Exception as e:
            print("Failed to initialize Gemma transformer model:", e)
            print("Try installing 'transformers' and 'torch', or switch to sentence-transformers backend.")
            return

    # sanity check embedding dim
    try:
        dim = _embedding_model.get_sentence_embedding_dimension()
        print("Embedding dimension:", dim)
    except Exception as e:
        print("Failed to get embedding dimension from model:", e)
        return

    print("Connecting to Qdrant:", _QDRANT_HOST, _QDRANT_PORT)
    client = _QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT)

    _ensure_collection(client, _QDRANT_COLLECTION, dim)

    points = _gather_points(_RESUME_FOLDER)
    print(f"Gathered {len(points)} points from resumes.")
    if not points:
        print("No points to upsert (check parsing and embeddings).")
        return

    _upsert_points(client, points, collection_name=_QDRANT_COLLECTION)

    # Sanity check count:
    try:
        c = client.count(collection_name=_QDRANT_COLLECTION, exact=True)
        total = getattr(c, "count", c if isinstance(c, int) else None)
        print("Total points in collection:", total)
    except Exception as e:
        print("Count check failed:", e)

    print("Ingestion finished.")

if __name__ == "__main__":
    main()
