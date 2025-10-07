# ======================= chatbot_poc.py (UPDATED) =======================
# Full resume analyzer CLI: parse files, extract skills (model + heuristics), ingest to Qdrant,
# semantic search, and JSON output for details & search.
# Put this file next to model.py and your 'resumes' folder and skills.csv.

import os
# Avoid TF being pulled in by transformers on Windows
os.environ["TRANSFORMERS_NO_TF"] = "1"
import re
import io 
import logging
from typing import List, Dict, Any, Optional
import json
import csv
import time
from difflib import get_close_matches
import numpy as _np
from collections import defaultdict
import uuid as _uuid

# ---------------- Optional deps (guarded imports) ----------------
try:
    import spacy 
except Exception as e:
    spacy = None
try:
    import fitz  # PyMuPDF
except Exception: 
    fitz = None
try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None
    print("Warning: sentence-transformers not available:", e)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
except Exception as e:
    QdrantClient = None
    qm = None
    print("Warning: qdrant-client not available:", e)

# rapidfuzz optional for faster fuzzy matching
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    RAPIDFUZZ = True
except Exception:
    RAPIDFUZZ = False

# import model initializer (your local model.py)
try:
    from model import init_skill_model
except Exception as e:
    init_skill_model = None
    print("Warning: could not import init_skill_model from model.py:", e)

# ---------------- CONFIG ----------------
HF_LOCAL_SNAPSHOT = None

# Default to a small, fast model; can override via EMBEDDING_MODEL env var
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "resumes_collection")

RESUME_FOLDER = os.getenv("RESUME_FOLDER", "resumes")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_trf")
SKILLS_CSV_PATH = os.getenv("SKILLS_CSV_PATH", "categorized.csv")

# ---------------- NLP init ----------------
if spacy is not None:
    try:
        nlp = spacy.load(SPACY_MODEL)
        print(f"spaCy model '{SPACY_MODEL}' loaded.")
    except Exception as e:
        print("spaCy load failed; using blank model:", e)
        nlp = spacy.blank("en")
else:
    print("spaCy not installed; using minimal tokenizer.")
    class _Dummy:
        def __call__(self, x): return []
        @property
        def ents(self): return []
    nlp = _Dummy()

# ---------------- init skill model (amjad/jobbert etc.) ----------------
skill_pipe = None
if init_skill_model is not None:
    try:
        skill_pipe = init_skill_model(local_snapshot=HF_LOCAL_SNAPSHOT, device=-1)
    except Exception as e:
        print("init_skill_model error:", e)
        skill_pipe = None
print("skill_pipe available:", bool(skill_pipe))

# ---------------- Embeddings backend (SentenceTransformer OR Gemma) ----------------
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").lower()  # or "gemma"
GEMMA_LOCAL_PATH = os.getenv("GEMMA_LOCAL_PATH")  # optional local path

class _SentenceTransformerWrapper:
    def __init__(self, model_name):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed")
        self._m = SentenceTransformer(model_name)
        self._dim = self._m.get_sentence_embedding_dimension()
    def encode(self, texts):
        single = isinstance(texts, str)
        vecs = self._m.encode([texts] if single else texts,
                              convert_to_numpy=True, normalize_embeddings=True)
        return vecs[0] if single else vecs
    def get_sentence_embedding_dimension(self):
        return int(self._dim)

import numpy as np

class GemmaEmbedder:
    """
    Sentence embedding using a Gemma causal LM by mean-pooling the last hidden states.
    Works with any HF Gemma checkpoint (e.g., google/gemma-2-2b-it, gemma-2-9b-it, gemma-3-270m-it).
    """
    def __init__(self, model_id: str = None, device: str = None, dtype: str = "auto", max_length: int = 1024):
        self.model_id = model_id or os.getenv("GEMMA_MODEL_ID", "google/gemma-3-270m-it")
        self.max_length = max_length
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto" if dtype == "auto" else getattr(torch, dtype),
            device_map=None,
        )
        self.model.eval()
        self.model.to(self.device)
        torch.set_grad_enabled(False)

        # Dimension = hidden size from config
        self.dim = getattr(self.model.config, "hidden_size", None)
        if not self.dim:
            tmp_inp = self.tokenizer("x", return_tensors="pt").to(self.device)
            out = self.model(**tmp_inp, output_hidden_states=True)
            self.dim = out.hidden_states[-1].shape[-1]

    def get_sentence_embedding_dimension(self):
        return int(self.dim)

    @torch.inference_mode()
    def encode(self, texts, batch_size: int = 16, normalize: bool = True):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tok = self.tokenizer(
                batch, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**tok, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]      # [B, T, H]
            mask = tok.attention_mask.unsqueeze(-1)      # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)     # [B, H]
            counts = mask.sum(dim=1).clamp(min=1)        # [B, 1]
            pooled = summed / counts                     # [B, H]
            vecs = pooled.detach().float().cpu().numpy()
            if normalize:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                vecs = vecs / norms
            embs.append(vecs)
        out = np.vstack(embs)
        return out[0] if single else out  # [N, H]

# Choose backend
embedding_model = None
if EMBEDDING_BACKEND == "gemma":
    embedding_model = GemmaEmbedder(
        model_id=os.getenv("GEMMA_MODEL_ID", "google/gemma-3-270m-it"),
        max_length=int(os.getenv("GEMMA_MAX_LEN", "1024"))
    )
    EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
    def embed_texts(texts):
        return embedding_model.encode(texts)
else:
    # Fallback: MiniLM from sentence_transformers (and actually ASSIGN embedding_model)
    _minilm_id = os.getenv("MINILM_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = _SentenceTransformerWrapper(_minilm_id)
    EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
    def embed_texts(texts):
        return embedding_model.encode(texts)

# Qdrant init (collection creation is handled by ingest_resume.py)
qdrant_client = None
if QdrantClient is not None and embedding_model is not None:
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print("Qdrant client initialized (collection creation handled by ingest_resume.py).")
    except Exception as e:
        print("Qdrant init failed:", e)
        qdrant_client = None

# --- CSV-based skill taxonomy (fallbacks + helpers) ----------------------
SKILL_TAXONOMY = {
    "programming_languages": ["python","java","c","c++","c#","javascript","typescript","go","scala","ruby","php","r","swift","kotlin"],
    "cloud": ["aws","amazon web services","azure","microsoft azure","gcp","google cloud","google cloud platform"],
    "databases": ["mysql","postgresql","postgres","oracle","sql server","mongodb","redis","dynamodb","bigquery","snowflake","redshift"],
    "data_engineering": ["hadoop","spark","pyspark","hive","airflow","databricks","dbt","kafka","sqoop"],
    "devops_ci_cd": ["docker","kubernetes","jenkins","gitlab ci","github actions","circleci","terraform","ansible","azure devops"],
    "web_frameworks": ["react","angular","vue","spring","django","flask","express","node","asp.net"],
    "analytics_and_viz": ["tableau","powerbi","looker","matplotlib","seaborn"],
    "testing_and_quality": ["pytest","junit","selenium","cucumber"],
    "storage_and_format": ["parquet","avro","orc","csv","json"],
    "other_tech": ["git","github","gitlab","bitbucket","rest","graphql","elasticsearch"],
    "soft_skills": ["communication","leadership","teamwork","collaboration","problem solving","adaptability","management","presentation","mentoring"],
}
CANONICAL_SKILLS: set = set()
SKILL_ALIASES: Dict[str, str] = {}
SKILL_TO_CATEGORY: Dict[str, str] = {}
CANONICAL_LIST: List[str] = []
CSV_SKILLS_LOADED = False
_FLAT_SKILL_TO_CATEGORY: Dict[str, str] = {}
_SOFT_SKILL_KEYWORDS: set = set()

def load_skills_from_csv(path: str = SKILLS_CSV_PATH) -> int:
    global CANONICAL_SKILLS, SKILL_ALIASES, SKILL_TO_CATEGORY, CANONICAL_LIST, CSV_SKILLS_LOADED
    CANONICAL_SKILLS = set(); SKILL_ALIASES = {}; SKILL_TO_CATEGORY = {}; CANONICAL_LIST = []
    CSV_SKILLS_LOADED = False

    if not path:
        print("load_skills_from_csv: no path provided")
        return 0

    if not os.path.exists(path):
        print(f"load_skills_from_csv: file not found at '{path}'")
        return 0

    def _cell_to_str(cell):
        """
        Safely convert a CSV cell to a stripped string.
        Handles None, list, numbers, etc.
        """
        if cell is None:
            return ""
        # If the CSV parser returned a list (weird), join it
        if isinstance(cell, (list, tuple)):
            try:
                cell = ", ".join(str(x) for x in cell)
            except Exception:
                cell = str(cell)
        try:
            return str(cell).strip()
        except Exception:
            return ""

    try:
        with open(path, newline='', encoding='utf-8', errors='replace') as f:
            sample = f.read(8192)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
                reader = csv.DictReader(f, dialect=dialect)
            except Exception:
                f.seek(0)
                reader = csv.DictReader(f)

            # header_map: lowercase->original
            header_map = {h.lower(): h for h in (reader.fieldnames or []) if h}

            # pick skill/category/alias columns robustly
            skill_col = header_map.get("skill") or header_map.get("skills") or header_map.get("name") or header_map.get("canonical") or ((reader.fieldnames or [None])[0] if reader.fieldnames else None)
            cat_col = header_map.get("category") or header_map.get("cat") or header_map.get("group") or header_map.get("type")
            alias_col = header_map.get("alias") or header_map.get("aliases") or header_map.get("synonyms") or header_map.get("alt")

            rows = 0
            for r in reader:
                rows += 1
                # Try to safely fetch values even if the row mapping is odd
                try:
                    raw_skill = _cell_to_str(r.get(skill_col) if skill_col in (r or {}) else None)
                except Exception:
                    # fallback: try to find first non-empty cell
                    raw_skill = ""
                    if isinstance(r, dict):
                        for v in r.values():
                            s = _cell_to_str(v)
                            if s:
                                raw_skill = s
                                break

                raw_cat = ""
                raw_alias = ""
                if isinstance(r, dict):
                    try:
                        raw_cat = _cell_to_str(r.get(cat_col)) if cat_col else ""
                    except Exception:
                        raw_cat = ""
                    try:
                        raw_alias = _cell_to_str(r.get(alias_col)) if alias_col else ""
                    except Exception:
                        raw_alias = ""

                if not raw_skill:
                    continue

                canon = raw_skill.lower()
                CANONICAL_SKILLS.add(canon)
                if raw_cat:
                    SKILL_TO_CATEGORY[canon] = raw_cat.lower()
                if raw_alias:
                    # split aliases by common delimiters and handle list-like strings
                    for a in _re.split(r'[;,/]+', raw_alias):
                        a = a.strip().lower()
                        if a:
                            SKILL_ALIASES[a] = canon

            CANONICAL_LIST = sorted(list(CANONICAL_SKILLS))
            CSV_SKILLS_LOADED = True
            print(f"Loaded {len(CANONICAL_SKILLS)} canonical skills from '{path}' and {len(SKILL_ALIASES)} aliases.")
            return len(CANONICAL_SKILLS)
    except Exception as e:
        print("Error reading skills CSV:", e)
        return 0

# load once (like before)
_loaded_count = load_skills_from_csv(SKILLS_CSV_PATH)


def _rebuild_flat_lookup_from_csv():
    global _FLAT_SKILL_TO_CATEGORY, _SOFT_SKILL_KEYWORDS
    _FLAT_SKILL_TO_CATEGORY = {}
    _SOFT_SKILL_KEYWORDS = set()

    if CSV_SKILLS_LOADED and CANONICAL_LIST:
        for canon in CANONICAL_LIST:
            cat = SKILL_TO_CATEGORY.get(canon, "") or ""
            key = canon.lower().strip()
            _FLAT_SKILL_TO_CATEGORY[key] = cat or "other_tech"

        for alias, canon in SKILL_ALIASES.items():
            key = alias.lower().strip()
            can_cat = SKILL_TO_CATEGORY.get(canon, "") or ""
            _FLAT_SKILL_TO_CATEGORY[key] = can_cat or "other_tech"

        for canon, cat in SKILL_TO_CATEGORY.items():
            if cat and ("soft" in cat or cat in ("soft_skills", "soft")):
                _SOFT_SKILL_KEYWORDS.add(canon.lower())
    else:
        for cat, toks in SKILL_TAXONOMY.items():
            if cat == "soft_skills":
                for s in toks: _SOFT_SKILL_KEYWORDS.add(s.lower())
            else:
                for t in toks: _FLAT_SKILL_TO_CATEGORY[t.lower()] = cat

_rebuild_flat_lookup_from_csv()
print(f"Flat lookup built. Flat keys: {len(_FLAT_SKILL_TO_CATEGORY)}; soft keywords: {len(_SOFT_SKILL_KEYWORDS)}")
if _loaded_count == 0:
    print("WARNING: No canonical skills loaded from categorized.csv. Check file contents/encoding and SKILLS_CSV_PATH.")
else:
    print("SUCCESS: canonical skills loaded.")

# ---------------- Canonical mapping helpers ----------------
_skill_emb_cache: Optional[Dict[str, _np.ndarray]] = None

def map_token_to_canonical(token: str, fuzzy_cutoff: float = 85.0, embed_cutoff: float = 0.78) -> Optional[str]:
    if not token: return None
    tok = token.strip().lower()
    if not tok: return None
    if tok in SKILL_ALIASES: return SKILL_ALIASES[tok]
    if tok in CANONICAL_SKILLS: return tok

    if CANONICAL_LIST:
        if RAPIDFUZZ: 
            try:
                match, score, _ = rf_process.extractOne(tok, CANONICAL_LIST, scorer=rf_fuzz.token_sort_ratio)
                if match and score >= fuzzy_cutoff:
                    return match
            except Exception:
                pass
        else:
            try:
                matches = get_close_matches(tok, CANONICAL_LIST, n=1, cutoff=(fuzzy_cutoff/100.0))
                if matches:
                    return matches[0]
            except Exception:
                pass

    global _skill_emb_cache
    if embedding_model is not None and CANONICAL_LIST:
        try:
            if _skill_emb_cache is None:
                _skill_emb_cache = {s: _np.array(embedding_model.encode(s), dtype=_np.float32) for s in CANONICAL_LIST}
            v_tok = _np.array(embedding_model.encode(tok), dtype=_np.float32)
            best_sim = 0.0; best_skill = None
            for s, vec in _skill_emb_cache.items():
                denom = (_np.linalg.norm(v_tok) * _np.linalg.norm(vec))
                if denom == 0: continue
                sim = float(_np.dot(v_tok, vec) / denom)
                if sim > best_sim:
                    best_sim = sim; best_skill = s
            if best_sim >= embed_cutoff:
                return best_skill
        except Exception:
            pass
    return None

# ---------------- Section splitting & headings ----------------
def split_into_sections(text: str):
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    if not lines:
        return [("body", text or "")]
    heading_indices = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s: continue
        if (len(s) < 90) and (s.isupper() or re.match(r'^[A-Z][a-z]+( [A-Za-z0-9&/-]+)*$', s) or (':' in s and len(s.split(':')[0].split()) <= 6)):
            heading_indices.append((i, s))
    if not heading_indices:
        for i, ln in enumerate(lines):
            if ':' in ln and len(ln) < 120:
                heading_indices.append((i, ln.split(':',1)[0].strip()))
    if not heading_indices:
        return [("body", text or "")]
    sections = []
    for idx, (i, hdr) in enumerate(heading_indices):
        start = i + 1
        end = heading_indices[idx+1][0] if idx+1 < len(heading_indices) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        sections.append((hdr.strip(), body))
    merged = []
    for hdr, body in sections:
        if merged and len(body.split()) < 3:
            prev_hdr, prev_body = merged[-1]
            merged[-1] = (prev_hdr, (prev_body + "\n" + hdr + "\n" + body).strip())
        else:
            merged.append((hdr, body))
    return merged

_HEADINGS_PLAIN = [
    "professional experience","work experience","experience","employment history","work history",
    "professional history","experience & qualifications","experience and qualifications",
    "work & experience","professional experience and qualifications"
]

def canonicalize_heading(h: str):
    if not h: return "body"
    hu = h.upper()
    if "SKILL" in hu: return "skills"
    if "EXPERIENCE" in hu or "EMPLOY" in hu: return "experience"
    if "EDUC" in hu: return "education"
    if "CERTIF" in hu or "LICENSE" in hu: return "certifications"
    if "PROJECT" in hu: return "projects"
    if "CONTACT" in hu or "EMAIL" in hu or "PHONE" in hu: return "contact"
    if "SUMMARY" in hu or "PROFILE" in hu or "OBJECTIVE" in hu: return "summary"
    try:
        h_low = h.strip().lower()
        close = get_close_matches(h_low, _HEADINGS_PLAIN, n=1, cutoff=0.75)
        if close:
            cand = close[0]
            if "experience" in cand: return "experience"
            if "skill" in cand: return "skills"
            if "education" in cand: return "education"
            if "project" in cand: return "projects"
            if "contact" in cand: return "contact"
            if "summary" in cand or "profile" in cand: return "summary"
    except Exception:
        pass
    return h.lower().strip()

# ---------------- Skill extraction ----------------
def extract_skills_from_section(text: str) -> List[str]:
    if not text: return []
    lines = [ln.strip(" •\t-") for ln in text.splitlines() if ln.strip()]
    skills = set()
    for ln in lines:
        if ',' in ln and len(ln) < 300:
            parts = [p.strip().lower() for p in ln.split(',') if p.strip()]
            skills.update(parts); continue
        if '|' in ln or '/' in ln or '·' in ln:
            parts = re.split(r'[|/·]', ln)
            skills.update([p.strip().lower() for p in parts if p.strip()]); continue
        tokens = re.findall(r'[A-Za-z0-9\+\-#\.]{2,}', ln)
        if tokens:
            skills.update([t.lower() for t in tokens if t.lower() not in ("and","with","experience","years","proficient")])
    cleaned = sorted({s for s in skills if len(s) > 1})
    return cleaned

def extract_skills_from_section_combined(section_text: str, full_text: str = "") -> List[str]:
    model_skills = []
    heur_skills = []
    try:
        heur_skills = extract_skills_from_section(section_text or full_text or "")
    except Exception as e:
        heur_skills = []; print("Heuristic extraction error:", e)

    text_for_model = (full_text or section_text or "").strip()
    if not text_for_model and section_text:
        text_for_model = section_text
    try:
        if skill_pipe is not None and text_for_model:
            chunk = text_for_model[:12000]
            ents = skill_pipe(chunk)
            for ent in ents:
                w = ent.get("word") or ent.get("entity_group") or ent.get("entity") or ent.get("label") or ent.get("text") or ""
                if w:
                    w = w.replace("Ġ", " ").replace("##", "").strip(" ,.;:-()[]\"'")
                    if w:
                        model_skills.append(w.strip().lower())
    except Exception as e:
        print("skill_pipe run error:", e)
        model_skills = []

    merged = []
    seen = set()
    for m in model_skills:
        mk = re.sub(r'^[\W_]+|[\W_]+$', '', m).strip().lower()
        if mk and mk not in seen:
            seen.add(mk); merged.append(mk)
    for h in heur_skills:
        hk = re.sub(r'^[\W_]+|[\W_]+$', '', h).strip().lower()
        if hk and hk not in seen:
            seen.add(hk); merged.append(hk)

    cleaned = post_process_skills(merged, full_text=text_for_model, whitelist=None)
    try:
        categorized = categorize_amjad_skills(cleaned, fuzzy=True)
        print_categorized(categorized)
    except Exception as e:
        print('Skill categorization error:', e)
    return cleaned

# --- Categorization helpers (condensed) ---------------------------------
def categorize_amjad_skills(amjad_skills, fuzzy=True, fuzz_cutoff=80):
    canon_set = set([s for s in CANONICAL_LIST]) if CANONICAL_LIST else set()
    aliases = {k.lower(): v.lower() for k, v in SKILL_ALIASES.items()} if SKILL_ALIASES else {}
    canon_to_cat = {k.lower(): v.lower() for k, v in SKILL_TO_CATEGORY.items()} if SKILL_TO_CATEGORY else {}
    result = {"technical": {k: [] for k in [
        "programming_languages","cloud","databases","data_engineering","devops_ci_cd","web_frameworks",
        "analytics_and_viz","testing_and_quality","storage_and_format","other_tech"]},
        "soft": [], "other": []}
    unmatched = []
    canon_keys = sorted(list(canon_set))
    for raw in amjad_skills:
        orig = str(raw).strip()
        if not orig: continue
        tok = orig.lower()
        matched_canon = None
        if tok in canon_set:
            matched_canon = tok
        elif tok in aliases:
            matched_canon = aliases[tok]
        else:
            for c in canon_keys:
                if c and (c in tok or tok in c):
                    matched_canon = c; break
        if not matched_canon and fuzzy and canon_keys:
            if RAPIDFUZZ:
                matches = rf_process.extract(tok, canon_keys, scorer=rf_fuzz.ratio, score_cutoff=fuzz_cutoff, limit=3)
                if matches: matched_canon = matches[0][0]
            else:
                close = get_close_matches(tok, canon_keys, n=1, cutoff=fuzz_cutoff/100.0)
                if close: matched_canon = close[0]
        if matched_canon:
            cat = canon_to_cat.get(matched_canon, "")
            mapped = cat if cat else None
            if mapped == "soft":
                result["soft"].append(orig)
            elif mapped in result["technical"]:
                result["technical"][mapped].append(orig)
            else:
                result["technical"]["other_tech"].append(orig)
        else:
            unmatched.append(orig)
            result["other"].append(orig)
    for k in list(result["technical"].keys()):
        result["technical"][k] = sorted(dict.fromkeys(result["technical"][k]))
    result["soft"] = sorted(dict.fromkeys(result["soft"]))
    result["other"] = sorted(dict.fromkeys(result["other"]))
    result["_meta"] = {"unmatched_count": len(unmatched), "unmatched_samples": unmatched[:40]}
    return result

def print_categorized(result):
    print("Categorized skills:")
    for bucket, skills in result["technical"].items():
        if skills:
            print(f" {bucket}:")
            for sk in skills:
                print(" -", sk)
    if result["soft"]:
        print("Soft:")
        for sk in result["soft"]:
            print(" -", sk)
    if result["other"]:
        print("Other (not in CSV):")
        for sk in result["other"]:
            print(" -", sk)

def post_process_skills(raw_list: List[str], full_text: str = "", whitelist: set = None, fuzzy_cutoff: float = 0.78) -> List[str]:
    if not raw_list: return []
    out = []
    email_re = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-z]{2,}', re.I)
    phone_re = re.compile(r'(\+?\d[\d\-\s()]{5,}\d)')
    seen = set()
    for tok in raw_list:
        if not tok: continue
        t = tok.strip()
        if email_re.search(t) or phone_re.search(t): continue
        if re.fullmatch(r'\d{4}', t) or re.fullmatch(r'[\d\-\./]{2,}', t): continue

        if t.lower().startswith("and ") or " and " in t.lower():
            parts = [p.strip() for p in re.split(r'\band\b', t, flags=re.I) if p.strip()]
            for p in parts:
                mapped = map_token_to_canonical(p, fuzzy_cutoff=85.0)
                if mapped and mapped not in seen:
                    out.append(mapped); seen.add(mapped)
            continue

        if len(t.split()) > 6:
            techs = re.findall(r'\b(java|python|scala|spark|hadoop|sql|pyspark|aws|azure|gcp|bigquery|jenkins|docker|kubernetes|dbt|databricks|snowflake|tableau)\b', t, flags=re.I)
            if techs:
                for tt in techs:
                    mapped = map_token_to_canonical(tt, fuzzy_cutoff=90.0)
                    if mapped and mapped not in seen:
                        out.append(mapped); seen.add(mapped)
                continue
            else:
                continue

        tl = t.lower()
        if tl in ("and","with","experience","skills","skill","years","year","the","a","an","in","on","of","for","is","are"):
            continue

        mapped = map_token_to_canonical(tl, fuzzy_cutoff=85.0)
        if mapped:
            if mapped not in seen:
                out.append(mapped); seen.add(mapped)
            continue

        if any(x in tl for x in ("python","java","scala","spark","hadoop","sql","aws","azure","gcp","docker","kubernetes","jenkins","react","node","django","flask")):
            norm = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', tl)
            if norm and norm not in seen:
                out.append(norm); seen.add(norm)
            continue
    return out

# ---------------- Text extraction ----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATE_RANGE_RE = re.compile(
    r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4}|\d{1,2}/\d{4})\s*(?:[-–—to]{1,4})\s*(?:Present|present|Current|current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4}|\d{1,2}/\d{4})',
    flags=re.I
)
SINGLE_DATE_RE = re.compile(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4}|\d{1,2}/\d{4}', flags=re.I)
_BULLET_RE = re.compile(r'^[\u2022\-\*\•\u25D8\u00B0\u00B70-9\)]\s+')

def _is_probably_binary(s: bytes) -> bool:
    if not s: return False
    if b'PK\x03\x04' in s[:4096]: return True
    total = len(s)
    nonprint = sum(1 for c in s if c < 32 and c not in (9,10,13))
    if total > 0 and (nonprint / float(total)) > 0.15:
        return True
    return False

def _clean_text_for_output(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[\t\f\v]+', ' ', text)
    text = ''.join(ch if (31 < ord(ch) < 127 or ord(ch) == 10) else ' ' for ch in text)
    text = re.sub(r'\n\s+\n', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def safe_text_extract(path: str) -> str:
    text = ""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".docx":
            try:
                import docx2txt
                text = docx2txt.process(path) or ""
                text = _clean_text_for_output(text)
                if text: return text
            except Exception:
                pass
            try:
                from docx import Document
                doc = Document(path)
                paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                        if row_text: paragraphs.append(row_text)
                text = "\n".join(paragraphs)
                text = _clean_text_for_output(text)
                if text: return text
            except Exception:
                pass
            try:
                import zipfile, xml.etree.ElementTree as ET
                with zipfile.ZipFile(path, 'r') as z:
                    if "word/document.xml" in z.namelist():
                        raw = z.read("word/document.xml")
                        raw = raw.decode('utf-8', errors='ignore') if isinstance(raw, bytes) else str(raw)
                        try:
                            root = ET.fromstring(raw)
                            texts = []
                            for node in root.iter():
                                tag = getattr(node, 'tag', '')
                                if tag.endswith('}t'):
                                    if node.text: texts.append(node.text)
                            text = "\n".join(t.strip() for t in texts if t and t.strip())
                            text = _clean_text_for_output(text)
                            if text: return text
                        except Exception:
                            plain = re.sub(r'<[^>]+>', ' ', raw)
                            plain = _clean_text_for_output(plain)
                            if plain: return plain
            except Exception:
                pass
            return ""

        if ext == ".pdf":
            try:
                import pdfplumber
                pages = []
                with pdfplumber.open(path) as pdf:
                    for p in pdf.pages:
                        try:
                            pt = p.extract_text() or ""
                        except Exception:
                            pt = ""
                        if pt and pt.strip():
                            pages.append(pt)
                text = "\n\n".join(pages).strip()
                text = _clean_text_for_output(text)
                if text: return text
            except Exception:
                pass
            try:
                from pdfminer.high_level import extract_text
                text = extract_text(path) or ""
                text = _clean_text_for_output(text)
                if text: return text
            except Exception:
                pass
            return ""

        # Plain text or other
        try:
            with open(path, 'rb') as fh:
                raw = fh.read()
            if _is_probably_binary(raw):
                raise ValueError("File looks binary; try specialized extractor")
            try:
                candidate = raw.decode('utf-8')
            except Exception:
                try:
                    candidate = raw.decode('latin-1')
                except Exception:
                    candidate = ""
            text = _clean_text_for_output(candidate)
            if text: return text
        except Exception:
            pass

        try:
            import textract
            raw = textract.process(path)
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='ignore')
            text = _clean_text_for_output(str(raw))
            if text: return text
        except Exception:
            pass
    except Exception as e:
        log.debug("safe_text_extract error for %s: %s", path, e)
    return ""

# ---------------- Candidate/name/contact/location helpers ----------------
def extract_locations_from_text(text: str):
    try:
        doc = nlp(text or "")
        locs = set()
        for ent in getattr(doc, "ents", []):
            if getattr(ent, "label_", "") in ("GPE", "LOC"):
                locs.add(ent.text.strip())
        return [l.lower() for l in sorted(locs)] if locs else []
    except Exception:
        return []

def extract_candidate_name_from_text(full_text: str, filename: str = None):
    lines = [ln.strip() for ln in (full_text or "").splitlines() if ln.strip()]
    skip_kw = re.compile(r'(?i)\b(resume|cv|curriculum vitae|profile|project history|objective|summary|contact|phone|email|address|linkedin|github)\b')
    for ln in lines[:12]:
        if skip_kw.search(ln): continue
        words = ln.split()
        if 1 < len(words) <= 4 and not any(ch.isdigit() for ch in ln):
            if all((w and w[0].isupper()) for w in words) or ln.isupper():
                return " ".join([w.capitalize() for w in ln.split()])
    try:
        doc = nlp("\n".join(lines[:30]))
        persons = [ent.text.strip() for ent in getattr(doc, "ents", []) if getattr(ent, "label_", "") == "PERSON"]
        for p in sorted(persons, key=lambda s: -len(s)):
            if not skip_kw.search(p):
                return p
    except Exception:
        pass
    if filename:
        base = os.path.splitext(os.path.basename(filename))[0]
        base_clean = re.sub(r'[_\-.]+', ' ', base)
        base_clean = re.sub(r'(?i)\b(resume|cv|final|de|profile)\b', '', base_clean)
        if base_clean.strip():
            return " ".join([w.capitalize() for w in base_clean.split()])
    return "Unknown"

# ---------------- Parsing & packaging ----------------
def extract_experience_from_section(text: str):
    if not text or not isinstance(text, str): return []
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    return [{"raw": p, "title": "", "company": "", "dates": [], "location": "", "summary": p} for p in paras]

def categorize_skills_for_resume(skills: List[str], full_text: str = "") -> Dict[str, Any]:
    technical_out: Dict[str, List[str]] = {k: [] for k in [
        "programming_languages","cloud","databases","data_engineering","devops_ci_cd",
        "web_frameworks","analytics_and_viz","testing_and_quality","storage_and_format","other_tech"
    ]}
    soft_out: List[str] = []
    other_out: List[str] = []
    for tok in skills:
        t = tok.lower().strip()
        if t in _SOFT_SKILL_KEYWORDS:
            soft_out.append(tok); continue
        mapped = _FLAT_SKILL_TO_CATEGORY.get(t)
        if mapped and mapped in technical_out:
            technical_out[mapped].append(tok); continue
        if any(x in t for x in ("communication","leadership","teamwork","mentoring","presentation","collaboration")):
            soft_out.append(tok); continue
        substr_to_bucket = {
            "python": "programming_languages","java": "programming_languages","c++": "programming_languages",
            "sql": "databases","mysql": "databases","postgres": "databases","mongodb": "databases",
            "aws": "cloud","azure": "cloud","gcp": "cloud","spark": "data_engineering","hadoop": "data_engineering",
            "airflow": "data_engineering","docker": "devops_ci_cd","kubernetes": "devops_ci_cd","jenkins": "devops_ci_cd",
            "react": "web_frameworks","angular": "web_frameworks","django": "web_frameworks","flask": "web_frameworks",
            "tableau": "analytics_and_viz","powerbi": "analytics_and_viz","pytest": "testing_and_quality",
            "parquet": "storage_and_format","json": "storage_and_format","csv": "storage_and_format"
        }
        bucket = None
        for sub, b in substr_to_bucket.items():
            if sub in t:
                bucket = b; break
        if bucket:
            technical_out[bucket].append(tok)
        else:
            if 1 <= len(t.split()) <= 4:
                other_out.append(tok)
    for k in list(technical_out.keys()):
        technical_out[k] = sorted(dict.fromkeys(technical_out[k]))
    soft_out = sorted(dict.fromkeys(soft_out))
    other_out = sorted(dict.fromkeys(other_out))
    return {"technical": technical_out, "soft": soft_out, "other": other_out}

def format_candidate_json(parsed_payload: Dict[str, Any]) -> str:
    name = parsed_payload.get("candidate_name") or parsed_payload.get("candidate_id") or "Unknown"
    email = parsed_payload.get("email")
    phone = parsed_payload.get("phone")
    linkedin = parsed_payload.get("linkedin")
    locations_raw = parsed_payload.get('locations') or []
    locations = [l for l in locations_raw if l]
    summary = (parsed_payload.get("summary") or "")[:2000]
    skills_list = parsed_payload.get("skills") or []
    categorized = categorize_skills_for_resume(skills_list, full_text=parsed_payload.get("full_text",""))
    out = {
        "candidate_name": name, "email": email, "phone": phone, "linkedin": linkedin,
        "locations": locations, "summary": summary, "skills": categorized,
        "experience": parsed_payload.get("experience", [])
    }
    return json.dumps(out, indent=2, ensure_ascii=False)


def semantic_search(query: str,
                             top_k: int = 20,
                             qdrant_limit: int = 500,
                             debug: bool = False) -> List[Dict[str, Any]]:
    """
    OPTION A: Pure embedding-based candidate rerank (no chunk-score blending, no role hardcoding).

    Strategy:
      - Embed the query.
      - Query Qdrant for many chunk hits (keep payload vectors if present).
      - Group chunks by candidate id.
      - For each candidate, build a candidate vector:
          1) prefer payload['_candidate_vector'] if present,
          2) else mean(chunk payload vectors _vector/vector/embedding),
          3) else embed a short concatenation of snippet texts.
      - Compute cosine(candidate_vector, query_vector) and rank descending.
      - Return top_k results as [{"score": sem_sim, "payload": rep_payload}, ...].
    """
    global qdrant_client, embedding_model, QDRANT_COLLECTION
    import numpy as _np, uuid as _uuid

    if qdrant_client is None or embedding_model is None:
        if debug:
            print("[option_a] qdrant_client or embedding_model not initialized.")
        return []

    # 1) embed query and normalize
    try:
        qv = embedding_model.encode(query)
        qv = _np.asarray(qv, dtype=float).flatten()
        qv = qv / ( _np.linalg.norm(qv) + 1e-12 )
    except Exception as e:
        if debug:
            print("[option_a] failed to embed query:", e)
        return []

    # 2) Qdrant search (try modern/legacy signatures)
    resp = None
    try:
        resp = qdrant_client.search(collection_name=QDRANT_COLLECTION,
                                   query_vector=qv.tolist(),
                                   limit=max(qdrant_limit, top_k),
                                   with_payload=True)
    except TypeError:
        try:
            resp = qdrant_client.search(collection_name=QDRANT_COLLECTION,
                                       vector=qv.tolist(),
                                       top=max(qdrant_limit, top_k),
                                       with_payload=True)
        except Exception:
            try:
                resp = qdrant_client.query_points(collection_name=QDRANT_COLLECTION,
                                                  vector=qv.tolist(),
                                                  limit=max(qdrant_limit, top_k),
                                                  with_payload=True)
            except Exception as e:
                if debug:
                    print("[option_a] qdrant search fallback failed:", e)
                return []
    except Exception as e:
        try:
            resp = qdrant_client.query_points(collection_name=QDRANT_COLLECTION,
                                              vector=qv.tolist(),
                                              limit=max(qdrant_limit, top_k),
                                              with_payload=True)
        except Exception as e2:
            if debug:
                print("[option_a] qdrant search error:", e, e2)
            return []

    if not resp:
        if debug:
            print("[option_a] no response from qdrant search.")
        return []

    # 3) normalize hits to list of dicts
    hits = []
    for r in resp:
        try:
            if hasattr(r, "payload"):
                payload = r.payload or {}
                score = getattr(r, "score", None)
                pid = getattr(r, "id", None) or getattr(r, "point_id", None)
            elif isinstance(r, dict):
                payload = r.get("payload") or r.get("point", {}).get("payload") or {}
                score = r.get("score")
                pid = r.get("id") or r.get("point", {}).get("id")
            else:
                payload = dict(getattr(r, "payload", {}) or {})
                score = getattr(r, "score", None)
                pid = getattr(r, "id", None)
        except Exception:
            payload = {}
            score = None
            pid = None

        hits.append({"id": pid or str(_uuid.uuid4()), "score": float(score) if score is not None else 0.0, "payload": payload})

    if not hits:
        if debug:
            print("[option_a] no hits.")
        return []

    # 4) group by candidate id and collect vectors/texts
    groups = {}
    for h in hits:
        pl = h.get("payload") or {}
        cid = pl.get("candidate_id") or pl.get("email") or pl.get("candidate_name") or h.get("id")
        cid = str(cid).strip().lower() if cid else str(_uuid.uuid4())
        entry = groups.setdefault(cid, {"payloads": [], "vectors": [], "texts": []})

        # candidate-level vector if present
        cand_v = pl.get("_candidate_vector") or pl.get("candidate_vector")
        if cand_v is not None:
            try:
                arr = _np.asarray(cand_v, dtype=float).flatten()
                n = _np.linalg.norm(arr)
                if n > 0:
                    arr = arr / n
                entry["vectors"].append(arr)
            except Exception:
                pass

        # per-chunk vector fields
        for vf in ("_vector", "vector", "embedding"):
            v = pl.get(vf)
            if v is not None:
                try:
                    arr = _np.asarray(v, dtype=float).flatten()
                    n = _np.linalg.norm(arr)
                    if n > 0:
                        arr = arr / n
                    entry["vectors"].append(arr)
                    break
                except Exception:
                    continue

        # text snippet fallback
        text = pl.get("text") or pl.get("snippet") or pl.get("summary") or ""
        if text:
            entry["texts"].append(str(text)[:2000])

        entry["payloads"].append(pl)

    # 5) build candidate vectors and compute sem sim
    candidates = []
    for cid, info in groups.items():
        cand_vec = None
        if info["vectors"]:
            try:
                cand_vec = _np.mean(_np.stack(info["vectors"], axis=0), axis=0)
                cand_vec = cand_vec / (_np.linalg.norm(cand_vec) + 1e-12)
            except Exception:
                cand_vec = None

        # fallback: embed small concatenated text
        if cand_vec is None and embedding_model is not None and info["texts"]:
            try:
                sample = " ".join(info["texts"][:4])
                ev = embedding_model.encode(sample)
                ev = _np.asarray(ev, dtype=float).flatten()
                ev = ev / (_np.linalg.norm(ev) + 1e-12)
                cand_vec = ev
            except Exception:
                cand_vec = None

        sem_sim = 0.0
        if cand_vec is not None:
            sem_sim = float(_np.dot(cand_vec, qv) / ((_np.linalg.norm(cand_vec) * _np.linalg.norm(qv)) + 1e-12))

        # representative payload: pick payload from first member or best available
        rep_payload = None
        if info["payloads"]:
            rep_payload = info["payloads"][0]
        candidates.append({"candidate_id": cid, "sem_sim": sem_sim, "payload": rep_payload})

        if debug:
            print(f"[option_a:debug] cid={cid[:30]} sem_sim={sem_sim:.6f} vectors={len(info['vectors'])} texts={len(info['texts'])}")

    # 6) sort by sem_sim descending and return top_k
    candidates_sorted = sorted(candidates, key=lambda x: x["sem_sim"], reverse=True)[:top_k]
    results = [{"score": float(c["sem_sim"]), "payload": c.get("payload") or {}} for c in candidates_sorted]
    return results

def parse_resume_file(path: str) -> dict:
    parsed = {
        "candidate_name": None, "candidate_id": os.path.basename(path),
        "email": None, "phone": None, "full_text": "", "locations": [],
        "sections": {}, "skills": [], "skills_by_category": {"technical": {}, "soft": [], "other": []},
        "experience": []
    }
    try:
        full_text = safe_text_extract(path) or ""
        parsed["full_text"] = full_text

        if full_text:
            m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', full_text)
            parsed["email"] = m.group(0) if m else None
            m2 = re.search(r'(\+?\d[\d\-\.\s\(\)]{7,}\d)', full_text)
            parsed["phone"] = m2.group(0).strip() if m2 else None

        try:
            parsed_name = extract_candidate_name_from_text(parsed.get("full_text",""), filename=path)
            if parsed_name: parsed["candidate_name"] = parsed_name
        except Exception:
            pass

        parsed_sections = {}
        if full_text:
            try:
                sections = split_into_sections(full_text)
                for hdr, body in sections:
                    key = canonicalize_heading(hdr) or "body"
                    parsed_sections[key] = parsed_sections.get(key, "") + ("\n" + body if parsed_sections.get(key) else body)
            except Exception:
                parsed_sections["body"] = full_text
        parsed["sections"] = parsed_sections

        try:
            skills_section_text = (parsed_sections.get("skills") or "").strip()
            if skills_section_text:
                extracted_skills = extract_skills_from_section_combined(skills_section_text, full_text=full_text)
            else:
                extracted_skills = extract_skills_from_section_combined(full_text, full_text=full_text)
            cleaned = []
            seen = set()
            for s in (extracted_skills or []):
                st = str(s).strip()
                st = re.sub(r'^[\W_]+|[\W_]+$', '', st)
                if not st: continue
                key = st.lower()
                if key not in seen:
                    seen.add(key); cleaned.append(st) 
            parsed["skills"] = cleaned
            parsed["skills_by_category"] = categorize_skills_for_resume(parsed["skills"], full_text=full_text)
        except Exception as e:
            log.debug("skill extraction failed for %s: %s", path, e)
            parsed["skills"] = []
            parsed["skills_by_category"] = {"technical": {}, "soft": [], "other": []}

        exp_text = ""
        try:
            if parsed["sections"].get("experience"):
                exp_text = parsed["sections"].get("experience", "").strip()
            else:
                for k, v in parsed["sections"].items():
                    if v and ("experience" in (k or "").lower() or "employment" in (k or "").lower() or "work" in (k or "").lower()):
                        exp_text = v.strip(); break
            if exp_text:
                parsed["experience"] = extract_experience_from_section(exp_text)
        except Exception as e:
            log.debug("experience extraction failed: %s", e)
            parsed["experience"] = []

        parsed["locations"] = extract_locations_from_text(full_text)
    except Exception as e:
        log.debug("parse_resume_file error for %s: %s", path, e)
    return parsed
