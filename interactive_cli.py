# interactive_cli.py
"""
Interactive CLI for AIResumeAnalyzer with semantic search and
candidate-level semantic reranking (no keyword boosts).

Usage:
    python interactive_cli.py
"""

import os
import sys
import pickle
from typing import Optional, Tuple, List, Dict
import numpy as np

# Optional: set to True to print extra diagnostics
DEBUG = True

# Qdrant client (used for sanity checks and building location index)
try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None

# ----------------------- Tunables -----------------------
LOC_NGRAM_MAX = 4
LOC_MATCH_THRESHOLD = 0.72     # cosine threshold to accept a phrase as a location
INTENT_MARGIN = 0.05           # margin between include/exclude intent scores
TOP_K_SEARCH = 50              # how many chunk-hits to pull before per-candidate aggregation
LOC_CACHE_PATH = ".loc_index.pkl"

# ---------------------- Model init ----------------------
try:
    from model import init_skill_model
except Exception as e:
    init_skill_model = None
    print("Warning: model.init_skill_model failed to import:", e)

# Import your main module so we reuse its config + functions
try:
    import chatbot_poc as cp
except Exception as e:
    print("Failed to import chatbot_poc. Ensure chatbot_poc.py is in PYTHONPATH.", e)
    raise

# Convenience handles from chatbot_poc
INGEST_FN = getattr(cp, "ingest_resumes_to_qdrant", None)
SEMANTIC_SEARCH = getattr(cp, "semantic_search", None)
PARSE_FN = getattr(cp, "parse_resume_file", None)
QDRANT_COLLECTION = getattr(cp, "QDRANT_COLLECTION", "resumes_collection")
RESUME_FOLDER = getattr(cp, "RESUME_FOLDER", "./resumes")
EMBEDDING_BACKEND = getattr(cp, "EMBEDDING_BACKEND", os.getenv("EMBEDDING_BACKEND", "gemma"))

# Gemma (or other) model id (env overridable)
GEMMA_MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-3-270m-it")


def _ensure_embedding_model(device: Optional[str] = None, normalize: bool = True):
    """
    Initialize embedding model via model.init_skill_model and inject into chatbot_poc.embedding_model
    if not already present. Must match the model used during ingest.
    """
    if getattr(cp, "embedding_model", None) is not None:
        return cp.embedding_model

    if init_skill_model is None:
        print("init_skill_model() not available (model.py may be missing). Cannot initialize embeddings.")
        return None

    try:
        print(f"Initializing embedding model '{GEMMA_MODEL_ID}' (backend=transformers) ...")
        model = init_skill_model(model_id=GEMMA_MODEL_ID, backend="transformers", device=device, normalize=normalize)
        cp.embedding_model = model
        print("Embedding model initialized. Dim:", model.get_sentence_embedding_dimension())
        return model
    except Exception as e:
        print("Failed to initialize embedding model:", e)
        return None


# ---------------------- Embedding utils ----------------------
def embed_texts(texts: List[str], model, batch: int = 64) -> List[np.ndarray]:
    """
    Encode list of texts using model.encode and return normalized numpy vectors.
    """
    out: List[np.ndarray] = []
    if not texts: 
        return out
    for i in range(0, len(texts), batch): 
        b = texts[i:i + batch]
        vs = model.encode(b)
        # force to 2D
        arr = np.asarray(vs)
        if arr.ndim == 1:
            arr = arr[None, :]
        for row in arr:
            v = np.asarray(row, dtype=float).flatten()
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            out.append(v)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------- Intent / Location ----------------------
_INCLUDE_EXAMPLES = ["find", "show me", "give me", "pick", "search", "list", "find me resumes"]
_EXCLUDE_EXAMPLES = ["exclude", "don't include", "filter out", "not in", "without", "exclude resumes", "remove"]


def detect_intent_semantic(query: str, embedding_model) -> str:
    """
    Return "include" or "exclude" by semantic similarity to tiny prototype sets.
    """
    if embedding_model is None or not query.strip():
        return "include"
    qv = embed_texts([query], embedding_model)
    if not qv:
        return "include"
    qv = qv[0]
    inc_vecs = embed_texts(_INCLUDE_EXAMPLES, embedding_model)
    exc_vecs = embed_texts(_EXCLUDE_EXAMPLES, embedding_model)
    if not inc_vecs or not exc_vecs:
        return "include"
    inc_score = float(np.max(np.stack(inc_vecs, axis=0) @ qv))
    exc_score = float(np.max(np.stack(exc_vecs, axis=0) @ qv))
    return "exclude" if exc_score > inc_score + INTENT_MARGIN else "include"


def build_location_index_from_collection(qc_client, collection_name: str, embedding_model) -> Tuple[List[str], np.ndarray]:
    """
    Read unique location strings from collection payloads and embed them once.
    """
    loc_set = set()
    try:
        offset = None
        while True:
            resp, offset = qc_client.scroll(collection_name=collection_name, limit=500, with_payload=True, offset=offset)
            if not resp:
                break
            for p in resp:
                payload = getattr(p, "payload", None) or (p.get("payload") if isinstance(p, dict) else {})
                locs = payload.get("locations") or payload.get("locations_extracted") or []
                if isinstance(locs, str):
                    locs = [locs]
                for l in locs or []:
                    v = str(l).strip().lower()
                    if v:
                        loc_set.add(v)
            if offset is None:
                break
    except Exception:
        pass

    locations = sorted(loc_set)
    if not locations:
        dim = embedding_model.get_sentence_embedding_dimension() if embedding_model else 0
        return [], np.zeros((0, dim))

    vecs = embed_texts(locations, embedding_model, batch=64)
    if not vecs:
        dim = embedding_model.get_sentence_embedding_dimension()
        return locations, np.zeros((0, dim))
    loc_vecs = np.stack(vecs, axis=0)
    return locations, loc_vecs


def extract_location_semantic(query: str, embedding_model, locations: List[str], loc_vecs: np.ndarray) -> Optional[str]:
    """
    Find a location phrase in query by n-gram semantic match to known locations.
    """
    if not query.strip() or not locations or loc_vecs.size == 0:
        return None
    tokens = query.strip().split()
    ngrams: List[str] = []
    for n in range(1, min(LOC_NGRAM_MAX, len(tokens)) + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i:i + n]))

    if not ngrams:
        return None

    ng_vecs = embed_texts(ngrams, embedding_model)
    best = None
    best_score = -1.0
    for phrase, vec in zip(ngrams, ng_vecs):
        scores = loc_vecs @ vec
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        if score >= LOC_MATCH_THRESHOLD and score > best_score:
            best = locations[idx]
            best_score = score
    return best


# --------------------------- CLI ---------------------------
def main():
    print("AIResumeAnalyzer CLI")
    print("Collection:", QDRANT_COLLECTION)
    print("Resume folder:", RESUME_FOLDER)
    print("Embedding backend:", EMBEDDING_BACKEND, ("-> " + GEMMA_MODEL_ID if EMBEDDING_BACKEND == "gemma" else ""))

    # Ensure embedding model exists
    embedding_model = _ensure_embedding_model(device=None, normalize=True)
    if embedding_model is None:
        print("Warning: embedding model not initialized. Ingest/search may fail.")

    # Sanity: index dim vs encoder dim
    try:
        qc = getattr(cp, "qdrant_client", None)
        if qc is None and QdrantClient is not None:
            qc = QdrantClient(host=getattr(cp, "QDRANT_HOST", "localhost"),
                              port=getattr(cp, "QDRANT_PORT", 6333))
        collection_dim = None
        if qc is not None:
            try:
                info = qc.get_collection(QDRANT_COLLECTION)
                collection_dim = getattr(getattr(info, "vectors_config", None), "size", None)
            except Exception:
                pass
        encoder_dim = None
        if embedding_model is not None:
            try:
                encoder_dim = embedding_model.get_sentence_embedding_dimension()
            except Exception:
                v = np.asarray(embedding_model.encode("test"))
                encoder_dim = int(v.shape[-1])
        print(f"Qdrant dim: {collection_dim} | encoder dim: {encoder_dim}")
        if collection_dim is not None and encoder_dim is not None and int(collection_dim) != int(encoder_dim):
            print("ERROR: Vector dimension mismatch. Recreate collection & re-ingest with the current encoder.")
            sys.exit(1)
    except Exception as e:
        print("Warning: dim sanity check failed:", e)

    # Build/load semantic location index
    LOCATIONS: List[str] = []
    LOC_VECS = None
    try:
        if qc is None and QdrantClient is not None:
            qc = QdrantClient(host=getattr(cp, "QDRANT_HOST", "localhost"),
                              port=getattr(cp, "QDRANT_PORT", 6333))
        if qc is not None:
            if os.path.exists(LOC_CACHE_PATH):
                try:
                    with open(LOC_CACHE_PATH, "rb") as fh:
                        LOCATIONS, LOC_VECS = pickle.load(fh)
                    if DEBUG:
                        print(f"[DEBUG] loaded location cache: {len(LOCATIONS)} entries")
                except Exception:
                    LOCATIONS, LOC_VECS = build_location_index_from_collection(qc, QDRANT_COLLECTION, cp.embedding_model)
            else:
                LOCATIONS, LOC_VECS = build_location_index_from_collection(qc, QDRANT_COLLECTION, cp.embedding_model)
                try:
                    with open(LOC_CACHE_PATH, "wb") as fh:
                        pickle.dump((LOCATIONS, LOC_VECS), fh)
                except Exception:
                    pass
            print(f"Location index: {len(LOCATIONS)} unique values")
    except Exception as e:
        print("Warning: failed to build location index:", e)
        LOCATIONS, LOC_VECS = [], np.zeros((0, embedding_model.get_sentence_embedding_dimension() if embedding_model else 0))

    print("Type 'help' for commands.\n")

    # ---------------- REPL ----------------
    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        if not cmd:
            continue

        low = cmd.lower()

        if low in ("exit", "quit"):
            break

        if low in ("help", "h", "?"):
            print("Commands:\n  ingest\n  search <query>\n  parse <path>\n  exit\n")
            continue

        if low == "ingest":
            if not os.path.isdir(RESUME_FOLDER):
                print(f"Resume folder '{RESUME_FOLDER}' not found. Put .pdf/.docx/.txt files there.")
                continue
            if INGEST_FN:
                try:
                    INGEST_FN(RESUME_FOLDER)
                except TypeError:
                    try:
                        INGEST_FN()
                    except Exception as e:
                        print("Ingest failed:", e)
                except Exception as e:
                    print("Ingest failed:", e)
            else:
                print("ingest_resumes_to_qdrant() not found in chatbot_poc.")
            # invalidate location cache on new ingest
            try:
                if os.path.exists(LOC_CACHE_PATH):
                    os.remove(LOC_CACHE_PATH)
            except Exception:
                pass
            continue

        if low.startswith("parse "):
            path = cmd[6:].strip().strip('"')
            if not path:
                print("Provide a file path.")
                continue
            if not os.path.exists(path):
                print("File not found:", path)
                continue
            if PARSE_FN:
                try:
                    parsed = PARSE_FN(path)
                except Exception as e:
                    print("Parse failed:", e)
                    continue
            else:
                print("parse_resume_file() not available.")
                continue

            print("Name:", parsed.get("candidate_name"))
            print("Email:", parsed.get("email"))
            skills = parsed.get("skills", []) or []
            if isinstance(skills, dict):
                skills_sample = list(skills.keys())[:30]
            else:
                skills_sample = skills[:30]
            print("Skills (sample):", ", ".join(skills_sample))
            print("Locations:", parsed.get("locations"))
            print("Experience entries:", len(parsed.get("experience") or []))
            continue

        # ---------------------- SEARCH ----------------------
        if low.startswith("search "):
            raw = cmd[len("search "):].strip()
            if not raw:
                print("Provide a query: search python airflow spark aws")
                continue

            search_fn = SEMANTIC_SEARCH or getattr(cp, "semantic_search", None)
            if not search_fn:
                print("semantic_search() not available in chatbot_poc.")
                continue

            if DEBUG:
                print("[DEBUG] interactive_cli will call search_fn = chatbot_poc.semantic_search")

            # Detect semantic intent + location
            intent = detect_intent_semantic(raw, cp.embedding_model) if cp.embedding_model else "include"
            matched_loc = extract_location_semantic(raw, cp.embedding_model, LOCATIONS, LOC_VECS) if cp.embedding_model is not None and LOC_VECS is not None else None

            # Run vector search (optionally filter by location payload)
            try:
                if matched_loc and intent == "include":
                    try:
                        hits = search_fn(raw, top_k=TOP_K_SEARCH, filter_location=matched_loc)
                    except TypeError:
                        hits = search_fn(raw, top_k=TOP_K_SEARCH)
                        hits = [h for h in hits if any(matched_loc in str(x).lower() for x in (h.get("payload", {}).get("locations") or []))]
                elif matched_loc and intent == "exclude":
                    hits = search_fn(raw, top_k=TOP_K_SEARCH)
                    hits = [h for h in hits if not any(matched_loc in str(x).lower() for x in (h.get("payload", {}).get("locations") or []))]
                else:
                    hits = search_fn(raw, top_k=TOP_K_SEARCH)
            except Exception as e:
                print("Search failed:", e)
                continue

            hits = (hits or [])[:TOP_K_SEARCH]
            if not hits:
                print("No results.")
                continue

            # Build query vector once (normalized)
            query_vec = None
            try:
                qv = cp.embedding_model.encode(raw)
                qv = np.asarray(qv, dtype=float).flatten()
                nq = np.linalg.norm(qv)
                query_vec = qv / (nq + 1e-12) if nq > 0 else None
            except Exception:
                query_vec = None

            # Group by candidate key
            groups: Dict[str, Dict] = {}
            for h in hits:
                pl = (h.get("payload") or {}) or {}
                # candidate identifier priority
                if pl.get("candidate_id"):
                    cid = str(pl.get("candidate_id")).strip().lower()
                elif pl.get("email"):
                    cid = str(pl.get("email")).strip().lower()
                elif pl.get("candidate_name"):
                    cid = str(pl.get("candidate_name")).strip().lower()
                else:
                    cid = str(pl.get("filename") or h.get("id") or "").strip().lower()
                if not cid:
                    cid = "__unknown__"

                rec = groups.setdefault(cid, {"scores": [], "payloads": [], "best_snippet": None, "best_score": -1e9})
                sc = h.get("score") if h.get("score") is not None else 0.0
                try:
                    sc = float(sc)
                except Exception:
                    sc = 0.0
                rec["scores"].append(sc)
                rec["payloads"].append(pl)
                if sc > rec["best_score"]:
                    rec["best_score"] = sc
                    rec["best_snippet"] = (pl.get("text") or "")[:600]

            # Candidate-level rerank (purely semantic)
            FINAL_W_SEM = 1.0 # prioritize candidate semantic similarity
            FINAL_W_AGG = 0.0  # small contribution from chunk aggregation

            final_candidates: List[Dict] = []
            for cid, info in groups.items():
                scores_all = info.get("scores", []) or []
                # map possible [-1,1] to [0,1] just in case
                if any(s < 0 for s in scores_all):
                    scores_all = [(s + 1.0) / 2.0 for s in scores_all]

                scores_sorted = sorted(scores_all, reverse=True)
                top_scores = scores_sorted[:8]  # take up to top-8 chunks
                max_top = float(top_scores[0]) if top_scores else 0.0
                avg_top = float(sum(top_scores)) / max(len(top_scores), 1) if top_scores else 0.0
                agg_score = 0.6 * max_top + 0.4 * avg_top

                # fetch candidate vector if present, else fallback to mean of chunk vectors if available
                cand_vec = None
                for pl in info["payloads"]:
                    if isinstance(pl.get("_candidate_vector"), (list, tuple)) and len(pl["_candidate_vector"]) > 0:
                        arr = np.asarray(pl["_candidate_vector"], dtype=float).flatten()
                        n = np.linalg.norm(arr)
                        cand_vec = arr / (n + 1e-12) if n > 0 else None
                        if cand_vec is not None:
                            break

                if cand_vec is None:
                    # optional fallback if you stored individual chunk vectors in payload (e.g., "_vector")
                    chunk_vecs = []
                    for pl in info["payloads"]:
                        v = pl.get("_vector")
                        if isinstance(v, (list, tuple)) and len(v) > 0:
                            a = np.asarray(v, dtype=float).flatten()
                            n = np.linalg.norm(a)
                            if n > 0:
                                a = a / n
                            chunk_vecs.append(a)
                    if chunk_vecs:
                        cand_vec = np.mean(np.stack(chunk_vecs, axis=0), axis=0)
                        n = np.linalg.norm(cand_vec)
                        if n > 0:
                            cand_vec = cand_vec / n

                sem_score01 = 0.0
                if query_vec is not None and cand_vec is not None:
                    sem = cosine(query_vec, cand_vec)     # cosine in [-1,1]
                    sem_score01 = (sem + 1.0) / 2.0       # map to [0,1] for blending

                final_score = FINAL_W_SEM * sem_score01 + FINAL_W_AGG * agg_score

                final_candidates.append({
                    "cid": cid,
                    "final": final_score,
                    "sem": sem_score01,
                    "agg": agg_score,
                    "payloads": info["payloads"],
                    "best_snippet": info.get("best_snippet"),
                })

            final_candidates.sort(key=lambda x: x["final"], reverse=True)

            # Print results (top 12 for convenience)
            shown = 0
            for i, cand in enumerate(final_candidates[:12], 1):
                # choose a representative payload (first is fine; purely for filename/name/locations)
                p = (cand.get("payloads") or [{}])[0]
                name = p.get("candidate_name") or ""
                fname = p.get("filename") or ""
                locs = p.get("locations") or []
                snippet = (cand.get("best_snippet") or "").replace("\n", " ").strip()
                print(f"{i}. {fname or name}  score={cand['final']:.4f} (agg={cand['agg']:.4f} sem={cand['sem']:.4f})")
                if snippet:
                    print("   =>", snippet[:300], "..." if len(snippet) > 300 else "")
                shown += 1
            print(f"\nReturned {shown} candidates (showing top {shown}).")
            continue

        print("Unknown command. Type 'help'.")


if __name__ == "__main__":
    main()
