# ======================= ingest_resume.py (UPDATED) =======================
"""
Run this when you add/refresh resumes to (re)create the Qdrant collection
and upload resume chunks with embeddings.

Usage:
    python ingest_resume.py
"""

import os as _os
import uuid as _uuid
import re as _re
import math as _math
from typing import List as _List

try:
    from qdrant_client import QdrantClient as _QdrantClient
    from qdrant_client.http import models as qm
except Exception as _e:
    _QdrantClient = None
    qm = None
    print("qdrant-client not available:", _e)

# Bring in your app config + embedding model + parser
from chatbot_poc import (
    RESUME_FOLDER as _RESUME_FOLDER,
    QDRANT_HOST as _QDRANT_HOST,
    QDRANT_PORT as _QDRANT_PORT,
    QDRANT_COLLECTION as _QDRANT_COLLECTION,
    embedding_model as _embedding_model,
    parse_resume_file as _parse_resume_file,
)

_BATCH = 128  # upsert batch size


def _fallback_read_text(path: str) -> str:
    """Very robust text fallback for PDF/DOCX/TXT."""
    try:
        if path.lower().endswith('.pdf'):
            try:
                import fitz as _fitz  # PyMuPDF
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
    """
    files = [f for f in _os.listdir(resume_folder) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    print(f"Found {len(files)} files in '{resume_folder}'")
    if not files:
        raise FileNotFoundError(
            f"No resume files found in '{resume_folder}'. Put .pdf/.docx/.txt files there."
        )

    if _embedding_model is None:
        raise RuntimeError("Embedding model not initialized in chatbot_poc.embedding_model")

    dim = _embedding_model.get_sentence_embedding_dimension()
    points = []

    for fn in files:
        path = _os.path.join(resume_folder, fn)
        print("Parsing:", fn)
        try:
            parsed = _parse_resume_file(path)
        except Exception as e:
            print(f"Error parsing {fn}: {e}")
            parsed = {
                "candidate_name": _os.path.splitext(fn)[0],
                "candidate_id": None,
                "email": None,
                "skills": {},
                "locations": [],
                "sections": {},
                "experience": [],
                "full_text": "",
            }

        full_text = parsed.get("full_text") or ""
        if not full_text:
            print(f"[{fn}] empty full_text after parsing; using fallback reader.")
            full_text = _fallback_read_text(path)
            parsed["full_text"] = full_text

        # role-focused candidate text + vector (once per file)
        role_text = _role_focused_text(parsed, full_text)
        try:
            cand_vec = _embedding_model.encode(role_text)
            if hasattr(cand_vec, "tolist"):
                cand_vec = cand_vec.tolist()
            cand_vec = _normalize_vec(cand_vec)
        except Exception as e:
            print(f"[{fn}] candidate vector build failed:", e)
            cand_vec = None

        # chunking
        chunks = _smart_chunk(full_text)
        if not chunks:
            # absolute fallback (use very small slice)
            fallback = full_text[:400]
            if fallback:
                chunks = [fallback]
            else:
                print(f"[{fn}] no usable text found; skipping file.")
                continue

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

            payload = {
                "filename": fn,
                "candidate_name": parsed.get("candidate_name"),
                "candidate_id": parsed.get("candidate_id"),
                "email": parsed.get("email"),
                "skills": parsed.get("skills"),
                "locations": parsed.get("locations"),
                "sections": parsed.get("sections"),
                "experience": parsed.get("experience"),
                "full_text": parsed.get("full_text"),
                "text": chunk[:3000],
                "_candidate_vector": cand_vec,        # role-focused vector (normalized)
                "_candidate_role_text": role_text,    # for debugging/visibility
                # Optional: if you ever want to fallback-average chunk vectors at query time:
                # "_vector": vec,
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
    if _embedding_model is None:
        print("Embedding model not loaded. Initialize it in chatbot_poc.py before ingest.")
        return

    print("Connecting to Qdrant:", _QDRANT_HOST, _QDRANT_PORT)
    client = _QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT)

    dim = _embedding_model.get_sentence_embedding_dimension()
    _ensure_collection(client, _QDRANT_COLLECTION, dim)

    points = _gather_points(_RESUME_FOLDER)
    print(f"Gathered {len(points)} points from resumes.")
    if not points:
        print("No points to upsert (check parsing and embeddings).")
        return

    _upsert_points(client, points, collection_name=_QDRANT_COLLECTION)

    # Sanity count
    try:
        c = client.count(collection_name=_QDRANT_COLLECTION, exact=True)
        total = getattr(c, "count", c if isinstance(c, int) else None)
        print("Total points in collection:", total)
    except Exception as e:
        print("Count check failed:", e)

    print("Ingestion finished.")


if __name__ == "__main__":
    main()
