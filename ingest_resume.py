# ingest_resume.py
"""
Run this once (or when you add new resumes) to create the Qdrant collection
and upload resume chunks with embeddings.

Usage:
    python ingest_resume.py
"""

import os
import uuid
import re
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Import helpers from chatbot_poc (make sure chatbot_poc.py no longer recreates the collection at import)
from chatbot_poc import (
    RESUME_FOLDER,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    embedding_model,
    parse_resume_file,
)

BATCH = 128

def ensure_collection(client: QdrantClient, dim: int, collection_name: str = QDRANT_COLLECTION):
    try:
        # create if not exists
        client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        print(f"Creating collection '{collection_name}' with dim={dim}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

def _simple_read_file_text(path: str) -> str:
    """Fallback: attempt to read file contents as plain text if parser returns empty full_text."""
    try:
        if path.lower().endswith(".pdf"):
            try:
                import fitz
                doc = fitz.open(path)
                pages = [p.get_text("text") for p in doc]
                return "\n".join(pages)
            except Exception:
                # fallback to binary read
                with open(path, "rb") as f:
                    return str(f.read()[:5000], errors="replace")
        elif path.lower().endswith(".docx"):
            try:
                import docx
                doc = docx.Document(path)
                texts = [p.text for p in doc.paragraphs]
                return "\n".join(texts)
            except Exception:
                with open(path, "rb") as f:
                    return str(f.read()[:5000], errors="replace")
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
    except Exception as e:
        print("Warning: fallback text read failed for", path, ":", e)
        return ""

def gather_points_from_resumes(resume_folder: str) -> List[dict]:
    files = [f for f in os.listdir(resume_folder) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    if not files:
        raise FileNotFoundError(f"No resume files found in '{resume_folder}'. Put .pdf/.docx/.txt files there.")
    points = []
    for fn in files:
        path = os.path.join(resume_folder, fn)
        print("Parsing:", fn)
        try:
            parsed = parse_resume_file(path)
        except Exception as e:
            print(f"Error parsing {fn}: {e}")
            # create minimal parsed structure so we can continue
            parsed = {
                "candidate_name": os.path.splitext(fn)[0],
                "candidate_id": None,
                "email": None,
                "skills": {},
                "locations": [],
                "sections": {},
                "experience": [],
                "full_text": ""
            }

        # ensure full_text exists (fallback read)
        full_text = parsed.get("full_text") or ""
        if not full_text:
            print(f"DEBUG: parsed.full_text empty for {fn} — attempting fallback text read.")
            full_text = _simple_read_file_text(path)
            parsed["full_text"] = full_text or ""

        # debug: show whether experience was parsed
        exp_list = parsed.get("experience") or []
        if not exp_list:
            print(f"DEBUG: no experience parsed for {fn} (experience list is empty).")
        else:
            print(f"DEBUG: parsed {len(exp_list)} experience entries for {fn}.")

        # chunk by double-newline to keep points small
        chunks = [c.strip() for c in re.split(r'\n{2,}', full_text) if c.strip()]
        if not chunks:
            chunks = [full_text[:1000] or (parsed.get("sections") or {}).get("summary", "")[:1000]]

        for chunk in chunks:
            try:
                vec = embedding_model.encode(chunk).tolist()
            except Exception as e:
                print("Embedding error for chunk (falling back to empty vector):", e)
                # create a zero vector of appropriate dim if possible
                try:
                    dim = embedding_model.get_sentence_embedding_dimension()
                    vec = [0.0] * dim
                except Exception:
                    vec = []

            payload = {
                "filename": fn,
                "candidate_name": parsed.get("candidate_name"),
                "candidate_id": parsed.get("candidate_id"),
                "email": parsed.get("email"),
                "skills": parsed.get("skills"),
                "locations": parsed.get("locations"),
                "sections": parsed.get("sections"),
                "experience": parsed.get("experience"),   # <- include parsed experience
                "full_text": parsed.get("full_text"),    # <- include full text for debugging/details
                "text": chunk[:3000]
            }
            points.append({"id": str(uuid.uuid4()), "vector": vec, "payload": payload})
    return points

def upsert_points(client: QdrantClient, points: List[dict], collection_name: str = QDRANT_COLLECTION):
    for i in range(0, len(points), BATCH):
        batch = points[i:i+BATCH]
        try:
            client.upsert(collection_name=collection_name, points=batch)
            print(f"Upserted batch {i // BATCH + 1} ({len(batch)} points)")
        except Exception as e:
            print("Qdrant upsert error:", e)

def main():
    if embedding_model is None:
        print("Embedding model not loaded. Ensure SentenceTransformer loaded in chatbot_poc.py.")
        return
    print("Connecting to Qdrant:", QDRANT_HOST, QDRANT_PORT)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    dim = embedding_model.get_sentence_embedding_dimension()
    ensure_collection(client, dim, collection_name=QDRANT_COLLECTION)
    points = gather_points_from_resumes(RESUME_FOLDER)
    print(f"Gathered {len(points)} points from resumes.")
    upsert_points(client, points, collection_name=QDRANT_COLLECTION)
    print("Ingestion finished.")

if __name__ == "__main__":
    main()

