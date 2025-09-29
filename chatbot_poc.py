# chatbot_poc.py
# Full resume analyzer CLI: parse files, extract skills (model + heuristics), ingest to Qdrant,
# semantic search, and JSON output for details & search.
# Put this file next to model.py and your 'resumes' folder and skills.csv.

import os
# Avoid TF being pulled in by transformers on Windows
os.environ["TRANSFORMERS_NO_TF"] = "1"
import re
import io
import logging
from typing import List
import re
import uuid
import json
import csv
import time
from typing import List, Dict, Any, Optional
from difflib import get_close_matches

import fitz  # PyMuPDF
import docx
import spacy
import numpy as _np

# optional libs
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None
    print("Warning: sentence-transformers not available:", e)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import VectorParams, Distance
except Exception as e:
    QdrantClient = None
    VectorParams = None
    Distance = None
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
# e.g., r"C:\Users\annen\.cache\huggingface\hub\models--amjad-awad--skill-extractor\snapshots\..." if you have it
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "resumes_collection")
RESUME_FOLDER = os.getenv("RESUME_FOLDER", "resumes")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_trf")
SKILLS_CSV_PATH = os.getenv("SKILLS_CSV_PATH", "categorized.csv")

# ---------------- NLP init ----------------
try:
    nlp = spacy.load(SPACY_MODEL)
    print(f"spaCy model '{SPACY_MODEL}' loaded.")
except Exception as e:
    print("spaCy load failed; using blank model:", e)
    nlp = spacy.blank("en")

# ---------------- init skill model (amjad/jobbert etc.) ----------------
skill_pipe = None
if init_skill_model is not None:
    try:
        skill_pipe = init_skill_model(local_snapshot=HF_LOCAL_SNAPSHOT, device=-1)
    except Exception as e:
        print("init_skill_model error:", e)
        skill_pipe = None
print("skill_pipe available:", bool(skill_pipe))

# ---------------- embeddings + qdrant init ----------------
embedding_model = None
if SentenceTransformer is not None:
    try:
        print("Loading embedding model:", EMBEDDING_MODEL)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding dim:", embedding_model.get_sentence_embedding_dimension())
    except Exception as e:
        print("Embedding load failed:", e)
        embedding_model = None

qdrant_client = None
if QdrantClient is not None and embedding_model is not None:
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print("Qdrant client initialized (collection creation handled by ingest_resume.py).")
    except Exception as e:
        print("Qdrant init failed:", e)
        qdrant_client = None

# --- Legacy canonical skill placeholders (kept empty because CSV is primary source) ---
CANONICAL_SKILLS = set()
SKILL_ALIASES = {}
SKILL_TO_CATEGORY = {}
CANONICAL_LIST = []
CSV_SKILLS_LOADED = False


# ---------------- Skill taxonomy (fallback) ----------------
SKILL_TAXONOMY = {
    "programming_languages": ["python", "java", "c", "c++", "c#", "javascript", "typescript", "go", "scala", "ruby", "php", "r", "swift", "kotlin"],
    "cloud": ["aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud", "google cloud platform"],
    "databases": ["mysql", "postgresql", "postgres", "oracle", "sql server", "mongodb", "redis", "dynamodb", "bigquery", "snowflake", "redshift"],
    "data_engineering": ["hadoop", "spark", "pyspark", "hive", "airflow", "databricks", "dbt", "kafka", "sqoop"],
    "devops_ci_cd": ["docker", "kubernetes", "jenkins", "gitlab ci", "github actions", "circleci", "terraform", "ansible", "azure devops"],
    "web_frameworks": ["react", "angular", "vue", "spring", "django", "flask", "express", "node", "asp.net"],
    "analytics_and_viz": ["tableau", "powerbi", "looker", "matplotlib", "seaborn"],
    "testing_and_quality": ["pytest", "junit", "selenium", "cucumber"],
    "storage_and_format": ["parquet", "avro", "orc", "csv", "json"],
    "other_tech": ["git", "github", "gitlab", "bitbucket", "rest", "graphql", "elasticsearch"],
    "soft_skills": ["communication", "leadership", "teamwork", "collaboration", "problem solving", "adaptability", "management", "presentation", "mentoring"]
}
_FLAT_SKILL_TO_CATEGORY = {}
# If CSV whitelist loaded, prefer that mapping; otherwise fall back to SKILL_TAXONOMY
if CANONICAL_SKILLS:
    for canon, cat in SKILL_TO_CATEGORY.items():
        _FLAT_SKILL_TO_CATEGORY[canon.lower()] = cat
    # include aliases
    for alias, canon in SKILL_ALIASES.items():
        _FLAT_SKILL_TO_CATEGORY[alias.lower()] = SKILL_TO_CATEGORY.get(canon, "")
else:
    for cat, toks in SKILL_TAXONOMY.items():
        for t in toks:
            _FLAT_SKILL_TO_CATEGORY[t.lower()] = cat
# soft keywords
_SOFT_SKILL_KEYWORDS = set()
if CANONICAL_SKILLS:
    for k,v in SKILL_TO_CATEGORY.items():
        if v and "soft" in v:
            _SOFT_SKILL_KEYWORDS.add(k)
else:
    _SOFT_SKILL_KEYWORDS = set(SKILL_TAXONOMY.get("soft_skills", []))


# --- CSV-based skill categorizer -------------------------------------------------
import difflib

CSV_SKILLS_PATH = os.getenv("CSV_SKILLS_PATH", "categorized..csv")

def normalize_skill(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r'[\s]+', ' ', re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', s))
    return s

def load_skill_lookup(csv_path=CSV_SKILLS_PATH):
    """
    Load CSV into lookup: normalized_skill -> category (lowercase)
    """
    lookup = {}
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = [h.lower() for h in (reader.fieldnames or [])]
            skill_col = next((h for h in headers if h in ('skill','skills','name','canonical')), headers[0] if headers else None)
            cat_col = next((h for h in headers if h in ('category','cat','group','type')), None)
            for r in reader:
                raw = (r.get(skill_col) or "").strip()
                if not raw:
                    continue
                n = normalize_skill(raw)
                cat = (r.get(cat_col) or "").strip().lower() if cat_col else ""
                lookup[n] = cat
    except Exception as e:
        print("Could not load skill CSV:", e)
    return lookup

# Map dataset categories to top-level buckets
CATEGORY_MAPPING = {
    "programming_languages": "programming_languages",
    "databases": "databases",
    "data_engineering": "data_engineering",
    "analytics_visualization": "analytics_and_viz",
    "analytics_and_viz": "analytics_and_viz",
    "testing_tools": "testing_and_quality",
    "frameworks_libraries": "web_frameworks",
    "frontend": "web_frameworks",
    "web": "web_frameworks",
    "cloud_platforms": "cloud",
    "cloud_services_aws": "cloud",
    "ci_cd": "devops_ci_cd",
    "infrastructure_as_code": "devops_ci_cd",
    "containers_orchestration": "devops_ci_cd",
    "big_data_tools": "data_engineering",
    "message_brokers_streaming": "data_engineering",
    "monitoring_logging": "devops_ci_cd",
    "misc_tools": "other_tech",
    "mobile": "other_tech",
    "graphics_design": "other_tech",
    "office_productivity": "other_tech",
    "networking": "other_tech",
    "security": "other_tech",
    "version_control": "other_tech",
    "ide_tools": "other_tech",
    "certifications": "other_tech",
    "methodologies": "other_tech",
    "project_management": "other_tech",
    "data_science_ml": "other_tech",
    "soft_skills": "soft",
    "soft": "soft",
}

TECH_BUCKETS = [
    "programming_languages", "cloud", "databases", "data_engineering",
    "devops_ci_cd", "web_frameworks", "analytics_and_viz",
    "testing_and_quality", "storage_and_format", "other_tech"
]

_skill_lookup_cache = None

# ----- robust categorizer with fuzzy & substring matching -----
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _HAVE_RAPIDFUZZ = True
except Exception:
    _HAVE_RAPIDFUZZ = False
import difflib

def _norm(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'[\s]+', ' ', s)
    s = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', s)
    return s

def categorize_amjad_skills(amjad_skills, fuzzy=True, fuzz_cutoff=80):
    """
    Map Amjad-extracted tokens to your CSV categories using multiple strategies:
      - exact normalized match
      - alias match (SKILL_ALIASES)
      - substring match (token contains canonical or canonical contained in token)
      - fuzzy match (rapidfuzz if available, else difflib)
    Returns dict: {'technical': {...}, 'soft': [...], 'other': [...], '_meta': {...}}
    """
    # ensure lookup structures exist
    canon_set = set([_norm(x) for x in CANONICAL_LIST]) if CANONICAL_LIST else set()
    aliases = {k.lower(): v.lower() for k, v in SKILL_ALIASES.items()} if SKILL_ALIASES else {}
    canon_to_cat = {k.lower(): v.lower() for k, v in SKILL_TO_CATEGORY.items()} if SKILL_TO_CATEGORY else {}

    # prepare for fuzzy choices
    canon_keys = sorted(list(canon_set))
    result = {"technical": {k: [] for k in TECH_BUCKETS}, "soft": [], "other": []}
    unmatched = []

    for raw in amjad_skills:
        orig = str(raw).strip()
        if not orig:
            continue
        tok = _norm(orig)
        matched_canon = None

        # 1) exact normalized
        if tok in canon_set:
            matched_canon = tok
        # 2) alias map
        elif tok in aliases:
            matched_canon = aliases[tok]
        # 3) substring matches (prefer exact containment)
        else:
            # token contains a known canonical skill (e.g., 'python3' -> 'python')
            for c in canon_keys:
                if c and (c in tok or tok in c):
                    matched_canon = c
                    break

        # 4) fuzzy match as last resort
        if not matched_canon and fuzzy and canon_keys:
            if _HAVE_RAPIDFUZZ:
                # rapidfuzz returns list of tuples (match, score, index)
                matches = rf_process.extract(tok, canon_keys, scorer=rf_fuzz.ratio, score_cutoff=fuzz_cutoff, limit=3)
                if matches:
                    matched_canon = matches[0][0]
            else:
                close = difflib.get_close_matches(tok, canon_keys, n=1, cutoff=fuzz_cutoff/100.0)
                if close:
                    matched_canon = close[0]

        if matched_canon:
            cat = canon_to_cat.get(matched_canon, "")
            mapped = CATEGORY_MAPPING.get(cat, None)
            if mapped == "soft":
                result["soft"].append(orig)
            elif mapped in TECH_BUCKETS:
                result["technical"][mapped].append(orig)
            else:
                # if no mapping found treat as other_tech
                result["technical"]["other_tech"].append(orig)
        else:
            unmatched.append(orig)
            result["other"].append(orig)

    # dedupe & sort
    for k in result["technical"]:
        result["technical"][k] = sorted(dict.fromkeys(result["technical"][k]))
    result["soft"] = sorted(dict.fromkeys(result["soft"]))
    result["other"] = sorted(dict.fromkeys(result["other"]))
    result["_meta"] = {"unmatched_count": len(unmatched), "unmatched_samples": unmatched[:40]}
    return result

def print_categorized(result):
    print("Categorized skills:")
    for bucket, skills in result["technical"].items():
        if skills:
            print(f"  {bucket}:")
            for sk in skills:
                print("    -", sk)
    if result["soft"]:
        print("Soft:")
        for sk in result["soft"]:
            print("  -", sk)
    if result["other"]:
        print("Other (not in CSV):")
        for sk in result["other"]:
            print("  -", sk)

# -----------------------------------------------------------------------------

# ---------------- Load skills CSV as canonical whitelist ----------------
CANONICAL_SKILLS = set()
SKILL_ALIASES = {}       # alias_lower -> canonical_lower
SKILL_TO_CATEGORY = {}   # canonical -> category
CANONICAL_LIST = []
def load_skills_from_csv(path: str = "categorized.csv"):
    """
    Robust CSV loader that populates:
      - CANONICAL_SKILLS (set of canonical lower-case skill strings)
      - SKILL_ALIASES (alias_lower -> canonical_lower)
      - SKILL_TO_CATEGORY (canonical_lower -> category_lower)
      - CANONICAL_LIST (sorted list)
    Returns number of canonical skills loaded.
    """
    global CANONICAL_SKILLS, SKILL_ALIASES, SKILL_TO_CATEGORY, CANONICAL_LIST, CSV_SKILLS_LOADED
    CANONICAL_SKILLS = set()
    SKILL_ALIASES = {}
    SKILL_TO_CATEGORY = {}
    CANONICAL_LIST = []
    CSV_SKILLS_LOADED = False

    if not path:
        print("load_skills_from_csv: no path provided")
        return 0
    if not os.path.exists(path):
        print(f"load_skills_from_csv: file not found at '{path}'")
        return 0

    try:
        with open(path, newline='', encoding='utf-8', errors='replace') as csvfile:
            sample = csvfile.read(8192)
            csvfile.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
                reader = csv.DictReader(csvfile, dialect=dialect)
            except Exception:
                csvfile.seek(0)
                reader = csv.DictReader(csvfile)

            # build header mapping: lowercase -> original header name
            original_fieldnames = list(reader.fieldnames or [])
            header_map = {h.lower(): h for h in original_fieldnames}

            headers_lower = list(header_map.keys())
            # decide skill and category columns using lowercase names
            skill_col_lower = next((h for h in headers_lower if h in ("skill","skills","name","canonical")), None)
            cat_col_lower   = next((h for h in headers_lower if h in ("category","cat","group","type")), None)
            alias_col_lower = next((h for h in headers_lower if h in ("alias","aliases","synonyms","alt")), None)

            # fallback to first column if not detected
            if not skill_col_lower and original_fieldnames:
                skill_col_orig = original_fieldnames[0]
            else:
                skill_col_orig = header_map.get(skill_col_lower)

            cat_col_orig = header_map.get(cat_col_lower) if cat_col_lower else None
            alias_col_orig = header_map.get(alias_col_lower) if alias_col_lower else None

            row_count = 0
            for row in reader:
                row_count += 1
                if isinstance(row, dict):
                    raw_skill = (row.get(skill_col_orig) or "").strip() if skill_col_orig else ""
                    raw_cat   = (row.get(cat_col_orig) or "").strip() if cat_col_orig else ""
                    raw_alias = (row.get(alias_col_orig) or "").strip() if alias_col_orig else ""
                else:
                    raw_skill = str(row[0]).strip() if row else ""
                    raw_cat = ""
                    raw_alias = ""

                if not raw_skill:
                    continue
                canon = raw_skill.strip().lower()
                CANONICAL_SKILLS.add(canon)
                if raw_cat:
                    SKILL_TO_CATEGORY[canon] = raw_cat.strip().lower()
                if raw_alias:
                    for a in re.split(r'[;,/]+', raw_alias):
                        a = a.strip().lower()
                        if a:
                            SKILL_ALIASES[a] = canon

        CANONICAL_LIST = sorted(list(CANONICAL_SKILLS))
        CSV_SKILLS_LOADED = True
        print(f"Loaded {len(CANONICAL_SKILLS)} canonical skills from '{path}' and {len(SKILL_ALIASES)} aliases. (scanned rows: {row_count})")
        if CANONICAL_LIST:
            print("Sample canonical skills:", CANONICAL_LIST[:10])
        return len(CANONICAL_SKILLS)
    except Exception as e:
        print("Error reading skills CSV:", e)
        return 0

SKILLS_CSV_PATH = "categorized.csv"  # you said this is the filename
_loaded_count = load_skills_from_csv(SKILLS_CSV_PATH)
_FLAT_SKILL_TO_CATEGORY = {}
_SOFT_SKILL_KEYWORDS = set()

def _rebuild_flat_lookup_from_csv():
    global _FLAT_SKILL_TO_CATEGORY, _SOFT_SKILL_KEYWORDS
    _FLAT_SKILL_TO_CATEGORY = {}
    if CSV_SKILLS_LOADED:
        # map canonical skills to mapped top-level CATEGORY_MAPPING when possible
        for canon in CANONICAL_LIST:
            cat = SKILL_TO_CATEGORY.get(canon, "") or ""
            mapped = CATEGORY_MAPPING.get(cat, cat) if cat else ""
            # normalized key
            key = canon.lower().strip()
            if mapped:
                _FLAT_SKILL_TO_CATEGORY[key] = mapped
            else:
                # put uncategorized into other_tech
                _FLAT_SKILL_TO_CATEGORY[key] = "other_tech"
        # aliases -> map alias to same category as canonical
        for alias, canon in SKILL_ALIASES.items():
            key = alias.lower().strip()
            can_cat = SKILL_TO_CATEGORY.get(canon, "") or ""
            mapped = CATEGORY_MAPPING.get(can_cat, can_cat) if can_cat else "other_tech"
            _FLAT_SKILL_TO_CATEGORY[key] = mapped
        # build soft keywords set from any CSV category mapping labelled as soft
        for canon, cat in SKILL_TO_CATEGORY.items():
            if cat and ("soft" in cat or cat in ("soft_skills", "soft")):
                _SOFT_SKILL_KEYWORDS.add(canon.lower())
    else:
        # fallback to your hardcoded taxonomy
        for cat, toks in SKILL_TAXONOMY.items():
            if cat == "soft_skills":
                for s in toks: _SOFT_SKILL_KEYWORDS.add(s.lower())
            else:
                for t in toks:
                    _FLAT_SKILL_TO_CATEGORY[t.lower()] = cat

_rebuild_flat_lookup_from_csv()
print(f"Flat lookup built. Flat keys: {len(_FLAT_SKILL_TO_CATEGORY)}; soft keywords: {len(_SOFT_SKILL_KEYWORDS)}")
if _loaded_count == 0:
    print("WARNING: No canonical skills loaded from categorized.csv. Check file contents/encoding and SKILLS_CSV_PATH.")
else:
    print("SUCCESS: canonical skills loaded.")

# ---------- helper: map token to canonical skill ----------
_skill_emb_cache = None
# ---------- Replace existing filter_out_skills_from_locations and location extractors ----------

def filter_out_skills_from_locations(locations):
    """
    Given a list of location strings, remove any that are actually recognized skills
    (checks canonical whitelist, aliases, and fuzzy-mapped skills).
    Returns cleaned list (original casing preserved).
    """
    clean = []
    for loc in (locations or []):
        if not loc: 
            continue
        norm = loc.strip().lower()
        # direct check against canonical skill set / aliases
        if norm in CANONICAL_SKILLS:
            continue
        if norm in SKILL_ALIASES:
            continue
        # also try fuzzy/canonical mapping
        mapped = map_token_to_canonical(norm, fuzzy_cutoff=85.0, embed_cutoff=0.78)
        if mapped:
            # if token maps to a skill, skip it
            continue
        clean.append(loc)
    return clean

def extract_locations_from_text(text: str):
    """
    Use spaCy to extract GPE/LOC and then remove anything that maps to a skill.
    Returns lowercase list (sorted, unique).
    """
    if not text:
        return []
    doc = nlp(text)
    locs = set()
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC", "FAC"):  # include FAC in case addresses are recognized
            cand = ent.text.strip()
            if not cand:
                continue
            # normalize and check against skill lists
            norm = cand.lower()
            if norm in CANONICAL_SKILLS or norm in SKILL_ALIASES:
                # skip skill-like tokens
                continue
            # try fuzzy/embedding mapping to be safe
            try:
                mapped = map_token_to_canonical(norm, fuzzy_cutoff=90.0, embed_cutoff=0.85)
                if mapped:
                    continue
            except Exception:
                pass
            locs.add(norm)
    return [l for l in sorted(locs)] if locs else []



def map_token_to_canonical(token: str, fuzzy_cutoff: float = 85.0, embed_cutoff: float = 0.78) -> Optional[str]:
    if not token: return None
    tok = token.strip().lower()
    if not tok: return None
    if tok in SKILL_ALIASES: return SKILL_ALIASES[tok]
    if tok in CANONICAL_SKILLS: return tok
    # fuzzy match
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
    # embedding fallback
    global _skill_emb_cache
    if embedding_model is not None and CANONICAL_LIST:
        try:
            if _skill_emb_cache is None:
                _skill_emb_cache = {}
                for s in CANONICAL_LIST:
                    _skill_emb_cache[s] = _np.array(embedding_model.encode(s), dtype=_np.float32)
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

# ---------------- helpers: splitting/parse/payload ----------------
def split_into_sections(text: str):
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    if not lines: return [("body", text or "")]
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

# New: a list of canonical heading phrases (plain strings) for fuzzy matching
_HEADINGS_PLAIN = [
    "professional experience", "work experience", "experience", "employment history",
    "work history", "professional history", "experience & qualifications", "experience and qualifications",
    "work & experience", "professional experience and qualifications"
]

def canonicalize_heading(h: str):
    """
    Turn a raw section header into a canonical key. Uses case-insensitive checks
    and fuzzy matching to tolerate typos like 'PORFESSIONAL EXPERIENCE'.
    """
    if not h: return "body"
    hu = h.upper()
    # quick exact checks
    if "SKILL" in hu: return "skills"
    if "EXPERIENCE" in hu or "EMPLOY" in hu: return "experience"
    if "EDUC" in hu: return "education"
    if "CERTIF" in hu or "LICENSE" in hu: return "certifications"
    if "PROJECT" in hu: return "projects"
    if "CONTACT" in hu or "EMAIL" in hu or "PHONE" in hu: return "contact"
    if "SUMMARY" in hu or "PROFILE" in hu or "OBJECTIVE" in hu: return "summary"

    # fuzzy match against common headings (lowercased)
    try:
        h_low = h.strip().lower()
        close = get_close_matches(h_low, _HEADINGS_PLAIN, n=1, cutoff=0.75)
        if close:
            cand = close[0]
            if "experience" in cand:
                return "experience"
            if "skill" in cand:
                return "skills"
            if "education" in cand:
                return "education"
            if "project" in cand:
                return "projects"
            if "contact" in cand:
                return "contact"
            if "summary" in cand or "profile" in cand:
                return "summary"
    except Exception:
        pass
    return h.lower().strip()

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

# (rest of the file remains unchanged)
# For brevity we append the remainder of the original chatbot_poc.py unchanged below.



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
        # drop pure year/dates or tokens that are clearly not skills
        if re.fullmatch(r'\d{4}', t) or re.fullmatch(r'[\d\-\./]{2,}', t): continue
        # drop sentence fragments that start with 'and ' or 'and,' etc.
        if t.lower().startswith("and ") or " and " in t.lower():
            # try splitting into parts and keep tech token parts
            parts = [p.strip() for p in re.split(r'\band\b', t, flags=re.I) if p.strip()]
            if len(parts) == 1:
                continue
            for p in parts:
                mapped = map_token_to_canonical(p, fuzzy_cutoff=85.0)
                if mapped and mapped not in seen:
                    out.append(mapped); seen.add(mapped)
            continue
        # long multi-word sentence -> try extract common tech substrings
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
        # drop very generic tokens
        if tl in ("and","with","experience","skills","skill","years","year","the","a","an","in","on","of","for","is","are"):
            continue
        # try map to canonical (strict)
        mapped = map_token_to_canonical(tl, fuzzy_cutoff=85.0)
        if mapped:
            if mapped not in seen:
                out.append(mapped); seen.add(mapped)
            continue
        # allow some tech-like tokens even if not in whitelist (fallback)
        if any(x in tl for x in ("python","java","scala","spark","hadoop","sql","aws","azure","gcp","docker","kubernetes","jenkins","react","node","django","flask")):
            norm = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', tl)
            if norm and norm not in seen:
                out.append(norm); seen.add(norm)
            continue
    return out

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
                # transformers pipeline returns dicts; keys vary by pipeline version
                w = ent.get("word") or ent.get("entity_group") or ent.get("entity") or ent.get("label") or ""
                if not w:
                    # some NER pipelines give 'entity' and 'score' and 'start'/'end'; use span from chunk if available
                    try:
                        w = ent.get("text") or ""
                    except Exception:
                        w = ""
                if w:
                    w = w.replace("Ġ", " ").replace("##", "").strip(" ,.;:-()[]\"'")
                    if w:
                        model_skills.append(w.strip().lower())
    except Exception as e:
        print("skill_pipe run error:", e)
        model_skills = []
    print("DEBUG model_skills (sample):", model_skills[:40])
    print("DEBUG heur_skills (sample):", heur_skills[:40])
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
    print("DEBUG post_process_skills -> cleaned (sample 60):", cleaned[:60])
    try:
        categorized = categorize_amjad_skills(cleaned, fuzzy=True)
        print_categorized(categorized)
    except Exception as e:
        print('Skill categorization error:', e)
    return cleaned

def extract_pdf_text_blocks(pdf_path: str):
    full_text = ""
    blocks = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text("text")
            full_text += page_text + "\n"
            parts = [p for p in re.split(r'\n{2,}', page_text) if p.strip()]
            for p in parts:
                lines = [l for l in p.splitlines() if l.strip()]
                is_table_like = False
                if len(lines) >= 3:
                    avg_len = sum(len(l) for l in lines) / max(1, len(lines))
                    if avg_len < 60:
                        is_table_like = True
                blocks.append({"text": p, "is_likely_table": is_table_like})
        doc.close()
    except Exception as e:
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()
        except Exception as e2:
            print("PDF read error:", e, e2)
    return full_text, blocks

def extract_education_from_section(text: str):
    if not text: return {"schools": [], "degrees": []}
    schools = set(); degrees = set()
    degree_patterns = re.compile(r'\b(bachelor|b\.sc|bsc|bachelor of|master|m\.sc|msc|master of|phd|doctor|mba|bs|ms)\b', re.I)
    for ln in [l.strip() for l in text.splitlines() if l.strip()]:
        low = ln.lower()
        if degree_patterns.search(low):
            degrees.add(ln)
            if ' from ' in low:
                schools.add(low.split(' from ', 1)[1].strip())
            elif ' at ' in low:
                schools.add(low.split(' at ', 1)[1].strip())
            elif ',' in ln:
                schools.add(ln.split(',')[-1].strip())
            else:
                m = re.search(r'((university|college|institute)[^,;\n]*)', ln, re.I)
                if m: schools.add(m.group(1).strip())
        else:
            m = re.search(r'((university|college|institute)[^,;\n]*)', ln, re.I)
            if m: schools.add(m.group(1).strip())
    return {"schools": [s.lower() for s in sorted(schools)], "degrees": [d for d in sorted(degrees)]}

# ---- START REPLACEMENT: improved experience parsing helpers ----
import re

# conservative date patterns
DATE_RANGE_RE = re.compile(
    r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4})\s*(?:[-–—to]{1,4})\s*(?:Present|present|Current|current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4})',
    flags=re.I
)
SINGLE_DATE_RE = re.compile(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4}|\d{1,2}/\d{4}', flags=re.I)

_BULLET_RE = re.compile(r'^[\u2022\-\*\•\u25D8\u00B0\u00B70-9\)]\s+')


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# improved date regexes
DATE_RANGE_RE = re.compile(
    r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4}|\d{1,2}/\d{4})\s*(?:[-–—to]{1,4})\s*(?:Present|present|Current|current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4}|\d{1,2}/\d{4})',
    flags=re.I
)
SINGLE_DATE_RE = re.compile(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4}|\d{1,2}/\d{4}', flags=re.I)

_BULLET_RE = re.compile(r'^[\u2022\-\*\•\u25D8\u00B0\u00B70-9\)]\s+')

def _is_probably_binary(s: bytes) -> bool:
    # simple heuristic: contains ZIP magic or lots of non-printables
    if not s:
        return False
    if b'PK\x03\x04' in s[:4096]:
        return True
    # count non-printable fraction
    total = len(s)
    nonprint = sum(1 for c in s if c < 32 and c not in (9,10,13))
    if total > 0 and (nonprint / float(total)) > 0.15:
        return True
    return False

def _clean_text_for_output(text: str) -> str:
    if not text:
        return ""
    # replace many control chars and normalize whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[\t\f\v]+', ' ', text)
    # remove weird nulls / high-control characters
    text = ''.join(ch if (31 < ord(ch) < 127 or ord(ch) == 10) else ' ' for ch in text)
    # collapse multiple blank lines
    text = re.sub(r'\n\s+\n', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def safe_text_extract(path: str) -> str:
    """
    Robust multi-strategy text extraction:
      - docx: docx2txt -> python-docx -> zip/xml (`word/document.xml`)
      - pdf: pdfplumber -> pdfminer.six
      - fallback: textract (but avoid returning raw PK/XML dumps)
      - if result looks like raw binary or xml dump, return empty string (safer)
    """
    text = ""
    ext = os.path.splitext(path)[1].lower()
    try:
        # ---------------- DOCX ----------------
        if ext == ".docx":
            # 1) docx2txt
            try:
                import docx2txt
                text = docx2txt.process(path) or ""
                text = _clean_text_for_output(text)
                if text:
                    return text
            except Exception:
                pass

            # 2) python-docx (python-docx package)
            try:
                from docx import Document
                doc = Document(path)
                paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
                # also capture tables (common in resumes)
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                        if row_text:
                            paragraphs.append(row_text)
                text = "\n".join(paragraphs)
                text = _clean_text_for_output(text)
                if text:
                    return text
            except Exception:
                pass

            # 3) as last resort read word/document.xml from the docx zip and parse XML cleanly
            try:
                import zipfile
                import xml.etree.ElementTree as ET
                with zipfile.ZipFile(path, 'r') as z:
                    if "word/document.xml" in z.namelist():
                        raw = z.read("word/document.xml")
                        if isinstance(raw, bytes) and _is_probably_binary(raw):
                            raw = raw.decode('utf-8', errors='ignore')
                        else:
                            raw = raw.decode('utf-8', errors='ignore') if isinstance(raw, bytes) else str(raw)
                        try:
                            root = ET.fromstring(raw)
                            texts = []
                            for node in root.iter():
                                tag = getattr(node, 'tag', '')
                                if tag.endswith('}t') or tag.endswith('}tx') or tag.endswith('}T'):
                                    if node.text:
                                        texts.append(node.text)
                            text = "\n".join(t.strip() for t in texts if t and t.strip())
                            text = _clean_text_for_output(text)
                            if text:
                                return text
                        except Exception:
                            plain = re.sub(r'<[^>]+>', ' ', raw)
                            plain = _clean_text_for_output(plain)
                            if plain:
                                return plain
            except Exception:
                pass

            return ""

        # ---------------- PDF ----------------
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
                if text:
                    return text
            except Exception:
                pass

            try:
                from pdfminer.high_level import extract_text
                text = extract_text(path) or ""
                text = _clean_text_for_output(text)
                if text:
                    return text
            except Exception:
                pass

            return ""

        # ---------------- OTHER (txt, doc, unknown) ----------------
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
            if text:
                return text
        except Exception:
            pass

        # try textract (last resort)
        try:
            import textract
            raw = textract.process(path)
            if isinstance(raw, bytes):
                if _is_probably_binary(raw):
                    return ""
                raw = raw.decode('utf-8', errors='ignore')
            text = _clean_text_for_output(str(raw))
            if text:
                return text
        except Exception:
            pass

    except Exception as e:
        log.debug("safe_text_extract error for %s: %s", path, e)

    return ""


# ---------------------------
# Rest of file content (unchanged sections)
# ---------------------------

def _looks_like_sentence_fragment(line: str) -> bool:
    if not line:
        return False
    s = line.strip()
    if not s:
        return False
    if s[0].islower():
        return True
    if s.endswith('.') or s.endswith('…') or s.endswith(';'):
        return True
    if len(s.split()) > 14:
        return True
    if re.search(r'\b(manage|managed|design|designed|lead|led|architect|develop|developed|perform|performed|optimi|deploy|deployed)\b', s, flags=re.I):
        return True
    return False

def split_experience_entries(section_text: str):
    if not section_text:
        return []
    sec = section_text.replace('\r\n', '\n')
    sec = sec.replace('•', '- ').replace('\u2022', '- ')
    paras = [p.strip() for p in re.split(r'\n{2,}', sec) if p.strip()]
    entries = []
    for para in paras:
        try:
            # gather date matches safely
            dates_found = list(DATE_RANGE_RE.finditer(para)) + list(SINGLE_DATE_RE.finditer(para))
            if len(dates_found) > 1:
                # split at subsequent date anchors
                cut_positions = [m.start() for m in dates_found[1:]] + [len(para)]
                start = 0
                for pos in cut_positions:
                    piece = para[start:pos].strip()
                    if piece and len(piece) > 12:
                        entries.append(piece)
                    start = pos
                continue
            lines = [ln.rstrip() for ln in para.splitlines() if ln.strip()]
            if not lines:
                continue
            if len(lines) == 1:
                single = lines[0].strip()
                if len(single) < 12 and entries:
                    entries[-1] = entries[-1] + "\n" + single
                else:
                    entries.append(single)
                continue
            i = 0
            while i < len(lines):
                if _BULLET_RE.match(lines[i]) or lines[i].startswith('- '):
                    bullets = []
                    while i < len(lines) and (_BULLET_RE.match(lines[i]) or lines[i].startswith('- ')):
                        bullets.append(lines[i]); i += 1
                    if entries:
                        entries[-1] = entries[-1] + "\n" + "\n".join(bullets)
                    else:
                        entries.append("\n".join(bullets))
                else:
                    header_lines = [lines[i].strip()]; j = i + 1
                    while j < len(lines) and not _BULLET_RE.match(lines[j]) and not _looks_like_sentence_fragment(lines[j]) and len(header_lines) < 3:
                        header_lines.append(lines[j].strip()); j += 1
                    bullets = []
                    while j < len(lines) and (_BULLET_RE.match(lines[j]) or lines[j].startswith('- ')):
                        bullets.append(lines[j]); j += 1
                    block = "\n".join(header_lines + bullets).strip()
                    if len(block) < 20 and _looks_like_sentence_fragment(block):
                        if entries:
                            entries[-1] = entries[-1] + "\n" + block
                        else:
                            entries.append(block)
                    else:
                        entries.append(block)
                    i = j
        except Exception as e:
            log.debug("split_experience_entries error: %s", e)
            # fallback: append whole paragraph
            entries.append(para)
    # final merge of tiny fragments
    cleaned = []
    for e in entries:
        es = e.strip()
        if not cleaned:
            cleaned.append(es); continue
        first_line = es.splitlines()[0] if es else ""
        if _looks_like_sentence_fragment(first_line) and len(es.split()) < 20:
            cleaned[-1] = cleaned[-1] + "\n" + es
        else:
            cleaned.append(es)
    cleaned = [c for c in cleaned if len(c.strip()) > 8]
    return cleaned


import re

# Keep your DATE regexes (or replace with these)
DATE_RANGE_RE = re.compile(
    r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4}|\d{1,2}/\d{4})\s*(?:[-–—to]{1,4})\s*(?:Present|present|Current|current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4}|\d{1,2}/\d{4})',
    flags=re.I
)
SINGLE_DATE_RE = re.compile(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4}|\d{1,2}/\d{4}', flags=re.I)

# US state abbreviations + common full names heuristic for location detection
_STATE_RE = r'\b(?:A[LKZR]|C[AOT]|D[EC]|F[L]|G[A]|H[I]|I[ADLN]|K[SY]|L[A]|M[ADEHINOPST]|N[CEHILOPRSTV]|O[HKR]|P[A]|R[I]|S[CD]|T[NX]|U[T]|V[AIT]|W[AIY])\b'
# also matches full state names (partial list) — keep loose
STATE_NAME_RE = r'\b(?:alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|new york|new jersey|new mexico|north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|virginia|washington|west virginia|wisconsin|wyoming)\b'

# location patterns: "City, ST" or "City, State"
LOC_RE = re.compile(r'([A-Za-z .\'\-]{2,60}),\s*([A-Za-z]{2,20})')

# separators that commonly separate title and company on same line
SEPARATORS = re.compile(r'\s*[-—–|@:\|]\s*')

def _clean_line(s: str) -> str:
    return re.sub(r'[\u00A0\u200B]+', ' ', s).strip().strip('•-–—|')

def parse_experience_entry(entry_text: str, full_text: str = None):
    """
    Improved conservative parser that attempts to extract title/company/location.
    Returns dict: {raw, title, company, dates, location, summary}
    """
    res = {"raw": entry_text, "title": "", "company": "", "dates": [], "location": "", "summary": ""}
    if not entry_text or not isinstance(entry_text, str):
        return res

    # normalize and split
    text = entry_text.replace('\r\n', '\n').replace('\t', ' ')
    lines = [ _clean_line(l) for l in text.splitlines() if l.strip() ]
    if not lines:
        return res

    # 1) extract dates first (prefer ranges)
    dates = []
    for m in DATE_RANGE_RE.finditer(text):
        dates.append(m.group(0).strip())
    if not dates:
        for m in SINGLE_DATE_RE.finditer(text):
            dates.append(m.group(0).strip())
    # dedupe & preserve order
    seen = set(); keep = []
    for d in dates:
        if d not in seen:
            keep.append(d); seen.add(d)
    res["dates"] = keep

    # 2) try to find a separate location line (City, ST or City, State)
    loc = ""
    loc_idx = -1
    for i, ln in enumerate(lines[:3]):  # location typically near top lines
        m = LOC_RE.search(ln)
        if m:
            # ensure second group is likely state abbreviation or state name
            grp = m.group(2)
            if re.fullmatch(r'[A-Za-z]{2}', grp) or re.search(STATE_NAME_RE, grp, flags=re.I):
                loc = m.group(0)
                loc_idx = i
                break
    if loc:
        res["location"] = loc

    # 3) handle first line patterns:
    first = lines[0]
    # If first line contains a clear separator and not just a sentence, split title/company
    if SEPARATORS.search(first) and len(first.split()) <= 20:
        parts = SEPARATORS.split(first, maxsplit=1)
        # sometimes splits into many pieces; treat first piece as title, last piece as company
        if len(parts) >= 2:
            title_candidate = parts[0].strip()
            company_candidate = parts[1].strip()
            # sanity checks: company candidate should not look like a sentence fragment (no verbs)
            if len(company_candidate.split()) <= 10 or re.search(r'\b(inc|llc|corp|ltd|co|company|solutions|technologies|systems|inc\.)\b', company_candidate, flags=re.I):
                res["title"] = title_candidate
                res["company"] = company_candidate
            else:
                # fallback: treat as title and push remainder to summary
                res["title"] = title_candidate

    # 4) if still empty title, maybe the first line *is* the title (short and capitalized)
    if not res["title"]:
        if len(first.split()) <= 8 and not re.search(r'[,.]', first):  # short single header-like line
            res["title"] = first

    # 5) if company still empty, maybe second line is company (and not a date or location)
    if not res["company"] and len(lines) > 1:
        second = lines[1]
        # skip if second looks like a date line or is location we already captured
        if not DATE_RANGE_RE.search(second) and not SINGLE_DATE_RE.search(second) and not res["location"] == second:
            # heuristic: company lines are shortish and often contain company cues or are ALL CAPS
            if len(second.split()) <= 8 or second.isupper() or re.search(r'\b(inc|llc|corp|ltd|company|co\.|solutions|technologies|systems|inc\.)\b', second, flags=re.I):
                res["company"] = second
                # if third line looks like a location, capture it
                if len(lines) > 2 and LOC_RE.search(lines[2]):
                    res["location"] = LOC_RE.search(lines[2]).group(0)

    # 6) if company empty but first line was "Company, City" style (company then comma)
    if not res["company"] and ',' in first and len(first.split(',')) <= 3:
        comps = [c.strip() for c in first.split(',') if c.strip()]
        # if first token is likely company (contains Inc/LLC etc), set it
        if re.search(r'\b(inc|llc|corp|ltd|co\.|company)\b', comps[0], flags=re.I) or comps[0].isupper():
            res["company"] = comps[0]
            if len(comps) > 1:
                res["location"] = ', '.join(comps[1:])

    # 7) location fallback: if we matched a location earlier, keep it; else scan all lines for city,state
    if not res["location"]:
        for i, ln in enumerate(lines[:4]):
            m = LOC_RE.search(ln)
            if m:
                res["location"] = m.group(0)
                break

    # 8) summary: everything but header/company/date/location lines
    skip_indices = set()
    # mark first line if used as title/company
    if res["title"] and lines:
        skip_indices.add(0)
    if res["company"] and len(lines) > 1:
        # if company matched same as first line (title/company in one line), do not skip extra
        if lines[0] and SEPARATORS.search(lines[0]) and res["company"] in lines[0]:
            skip_indices.add(0)
        else:
            # find the index where company appears
            for idx, ln in enumerate(lines[:4]):
                if res["company"] and res["company"].lower() == ln.lower():
                    skip_indices.add(idx); break
    # skip date-only lines
    for idx, ln in enumerate(lines):
        if DATE_RANGE_RE.search(ln) or SINGLE_DATE_RE.fullmatch(ln):
            skip_indices.add(idx)
    # skip location line if identical
    for idx, ln in enumerate(lines):
        if res["location"] and res["location"].lower() == ln.lower():
            skip_indices.add(idx)
    # build summary from remaining
    summary_pieces = [lines[i] for i in range(len(lines)) if i not in skip_indices]
    # If first line contained both title and company separated, remove the company part from summary
    if summary_pieces and SEPARATORS.search(summary_pieces[0]):
        # no-op: split_experience_blocks should have put bullets after header; keep as-is
        pass
    res["summary"] = ' '.join(summary_pieces).strip()

    # final normalization: strip stray separators from fields
    for k in ("title", "company", "location", "summary"):
        if isinstance(res[k], str):
            res[k] = re.sub(r'^[\-\|:]+\s*', '', res[k])
            res[k] = re.sub(r'\s+[\-\|:]+$', '', res[k])
            res[k] = res[k].strip()

    # if title empty but summary begins with Title - Company pattern, attempt one more try
    if not res["title"] and res["summary"]:
        m = SEPARATORS.split(res["summary"], maxsplit=1)
        if len(m) >= 2 and len(m[0].split()) <= 8:
            res["title"] = m[0].strip()
            res["company"] = res["company"] or m[1].strip()
            # update summary remove header chunk
            res["summary"] = re.sub(re.escape(m[0]) + r'\s*[-—–|:]\s*' + re.escape(m[1]), '', res["summary"], count=1).strip()

    return res


def extract_experience_from_section(text: str):
    if not text or not isinstance(text, str):
        return []
    blocks = split_experience_entries(text)
    parsed = []
    for b in blocks:
        try:
            parsed.append(parse_experience_entry(b, full_text=text))
        except Exception as e:
            log.debug("extract_experience_from_section parse error: %s", e)
            parsed.append({"raw": b, "title": "", "company": "", "dates": [], "location": "", "summary": b})
    if not parsed:
        return [{"raw": text, "title": "", "company": "", "dates": [], "location": "", "summary": text[:1200]}]
    return parsed

def extract_contact_from_section(text: str):
    if not text: return {}
    email = re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-z]{2,}', text)
    phone = re.search(r'(\+?\d[\d\s().-]{7,}\d)', text)
    linkedin = re.search(r'(linkedin\.com/[^\s,;]+)', text, re.I)
    return {"email": email.group(0) if email else None, "phone": phone.group(0).strip() if phone else None, "linkedin": linkedin.group(0) if linkedin else None}

def extract_locations_from_text(text: str):
    doc = nlp(text or "")
    locs = set()
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"): locs.add(ent.text.strip())
    return [l.lower() for l in sorted(locs)] if locs else []

def extract_candidate_name_from_text(full_text: str, filename: str = None):
    lines = [ln.strip() for ln in (full_text or "").splitlines() if ln.strip()]
    skip_kw = re.compile(r'(?i)\b(resume|cv|curriculum vitae|profile|project history|objective|summary|contact|phone|email|address|linkedin|github)\b')
    for ln in lines[:12]:
        if skip_kw.search(ln): continue
        words = ln.split()
        if 1 < len(words) <= 4 and not any(ch.isdigit() for ch in ln):
            if all(w[0].isupper() for w in words) or ln.isupper():
                return " ".join([w.capitalize() for w in ln.split()])
    doc = nlp("\n".join(lines[:30]))
    persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
    for p in sorted(persons, key=lambda s: -len(s)):
        if not skip_kw.search(p): return p
    if filename:
        base = os.path.splitext(os.path.basename(filename))[0]
        base_clean = re.sub(r'[_\-.]+', ' ', base)
        base_clean = re.sub(r'(?i)\b(resume|cv|final|de|profile)\b', '', base_clean)
        if base_clean.strip(): return " ".join([w.capitalize() for w in base_clean.split()])
    return "Unknown"

def parse_resume_file(path: str) -> dict:
    """
    Robust resume parser that returns a structured dict:
      {
        "candidate_name": str,
        "candidate_id": str,
        "email": str|None,
        "phone": str|None,
        "full_text": str,
        "locations": list,
        "sections": dict,            # canonicalized heading -> section text
        "skills": list,              # flat list of skills (strings)
        "skills_by_category": dict,  # output from categorize_skills_for_resume
        "experience": list,          # parsed experience entries (dicts)
        ... (other helper fields)
      }

    Relies on global helpers defined elsewhere in the module:
      safe_text_extract, split_into_sections, canonicalize_heading,
      extract_skills_from_section_combined, categorize_skills_for_resume,
      extract_experience_from_section, extract_contact_from_section,
      extract_candidate_name_from_text, load_skills_from_csv, SKILLS_CSV_PATH,
      CANONICAL_SKILLS, SKILL_ALIASES, SKILL_TO_CATEGORY, CANONICAL_LIST,
      CSV_SKILLS_LOADED
    """
    global _FLAT_SKILL_TO_CATEGORY, _SOFT_SKILL_KEYWORDS, CANONICAL_LIST, CSV_SKILLS_LOADED
    parsed = {
        "candidate_name": None,
        "candidate_id": os.path.basename(path),
        "email": None,
        "phone": None,
        "full_text": "",
        "locations": [],
        "sections": {},
        "skills": [],
        "skills_by_category": {"technical": {}, "soft": [], "other": []},
        "experience": []
    }

    # Ensure CSV skills are loaded and flat lookup is rebuilt
    try:
        if not CSV_SKILLS_LOADED:
            load_skills_from_csv(SKILLS_CSV_PATH)
    except Exception:
        # ignore; load_skills_from_csv already logs
        pass

    # Rebuild _FLAT_SKILL_TO_CATEGORY and _SOFT_SKILL_KEYWORDS from CSV if available
    try:
        _FLAT_SKILL_TO_CATEGORY = {}
        _SOFT_SKILL_KEYWORDS = set()
        if CSV_SKILLS_LOADED and CANONICAL_LIST:
            for canon in CANONICAL_LIST:
                cat = SKILL_TO_CATEGORY.get(canon, "") or ""
                mapped = CATEGORY_MAPPING.get(cat, cat) if cat else ""
                key = canon.lower().strip()
                if mapped:
                    _FLAT_SKILL_TO_CATEGORY[key] = mapped
                else:
                    _FLAT_SKILL_TO_CATEGORY[key] = "other_tech"
            for alias, canon in SKILL_ALIASES.items():
                key = alias.lower().strip()
                can_cat = SKILL_TO_CATEGORY.get(canon, "") or ""
                mapped = CATEGORY_MAPPING.get(can_cat, can_cat) if can_cat else "other_tech"
                _FLAT_SKILL_TO_CATEGORY[key] = mapped
            for canon, cat in SKILL_TO_CATEGORY.items():
                if cat and ("soft" in cat or cat in ("soft_skills", "soft")):
                    _SOFT_SKILL_KEYWORDS.add(canon.lower())
        else:
            # fallback to built-in SKILL_TAXONOMY (if present)
            for cat, toks in SKILL_TAXONOMY.items():
                if cat == "soft_skills":
                    for s in toks:
                        _SOFT_SKILL_KEYWORDS.add(s.lower())
                else:
                    for t in toks:
                        _FLAT_SKILL_TO_CATEGORY[t.lower()] = cat
    except Exception:
        # non-fatal; continue with whatever mapping we have
        pass

    try:
        # Extract full text (safe extractor)
        full_text = safe_text_extract(path) or ""
        # discard obvious raw xml/zip dumps returned by fallback extractors
        if full_text:
            if re.search(r'PK\x03\x04', full_text) or len(re.findall(r'<\/?w:t', full_text)) > 5:
                log.warning("safe_text_extract returned probable raw DOCX/XML for %s — discarding extracted text.", path)
                full_text = ""
            else:
                # check non-printable fraction
                num_chars = len(full_text)
                if num_chars > 0:
                    nonprint = sum(1 for ch in full_text if ord(ch) < 32 and ch not in ('\n', '\t', '\r'))
                    if (nonprint / float(num_chars)) > 0.12:
                        log.warning("safe_text_extract returned content with too many non-printables for %s — discarding.", path)
                        full_text = ""
        parsed["full_text"] = full_text

        # Quick contact heuristics from full_text
        if full_text:
            m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', full_text)
            parsed["email"] = m.group(0) if m else None
            m2 = re.search(r'(\+?\d[\d\-\.\s\(\)]{7,}\d)', full_text)
            parsed["phone"] = m2.group(0).strip() if m2 else None

        # Candidate name - best effort using helper or heuristics
        try:
            parsed_name = extract_candidate_name_from_text(parsed.get("full_text",""), filename=path)
            if parsed_name:
                parsed["candidate_name"] = parsed_name
        except Exception:
            # fallback: take top non-empty line not containing 'resume'
            if full_text:
                for ln in (l.strip() for l in full_text.splitlines() if l.strip())[:8]:
                    if not re.search(r'resume|curriculum vitae|cv|profile|summary|objective', ln, flags=re.I):
                        if len(ln.split()) <= 6 and not any(ch.isdigit() for ch in ln):
                            parsed["candidate_name"] = ln
                            break

        # Split into sections and canonicalize headings
        parsed_sections = {}
        if full_text:
            try:
                sections = split_into_sections(full_text)
                for hdr, body in sections:
                    key = canonicalize_heading(hdr)
                    if not key:
                        key = "body"
                    # accumulate same headings
                    parsed_sections[key] = parsed_sections.get(key, "") + ("\n" + body if parsed_sections.get(key) else body)
            except Exception:
                # fallback: keep whole text as "body"
                parsed_sections["body"] = full_text
        parsed["sections"] = parsed_sections

        # SKILL EXTRACTION: prefer explicit skills section, otherwise analyze full_text
        try:
            skills_section_text = (parsed_sections.get("skills") or "").strip()
            if skills_section_text:
                extracted_skills = extract_skills_from_section_combined(skills_section_text, full_text=full_text)
            else:
                # Use full text as fallback — this is heavier but necessary for many resumes without clear Skills header
                extracted_skills = extract_skills_from_section_combined(full_text, full_text=full_text)
            # Normalization: ensure list of unique lowercase-ish strings, but preserve readable case when possible
            cleaned = []
            seen = set()
            if isinstance(extracted_skills, (list, tuple)):
                for s in extracted_skills:
                    if not s: continue
                    st = str(s).strip()
                    # remove stray punctuation
                    st = re.sub(r'^[\W_]+|[\W_]+$', '', st)
                    if not st: continue
                    key = st.lower()
                    if key not in seen:
                        seen.add(key)
                        cleaned.append(st)
            else:
                cleaned = []
            parsed["skills"] = cleaned
            # Categorize into technical/soft/other
            try:
                parsed["skills_by_category"] = categorize_skills_for_resume(parsed["skills"], full_text=full_text)
            except Exception as e:
                log.debug("categorize_skills_for_resume error: %s", e)
                parsed["skills_by_category"] = {"technical": {}, "soft": [], "other": []}
        except Exception as e:
            log.debug("skill extraction failed for %s: %s", path, e)
            parsed["skills"] = []
            parsed["skills_by_category"] = {"technical": {}, "soft": [], "other": []}

        # Experience extraction — prefer explicit 'experience' canonical section; fallback to heuristics
        exp_text = ""
        try:
            if parsed["sections"].get("experience"):
                exp_text = parsed["sections"].get("experience", "").strip()
            else:
                # find any section that looks like experience
                for k, v in parsed["sections"].items():
                    if v and ("experience" in (k or "").lower() or "employment" in (k or "").lower() or "work" in (k or "").lower()):
                        exp_text = v.strip(); break
            # if still empty, try regex heuristics on full_text
            if not exp_text and full_text:
                m3 = re.search(r'(work experience|professional experience|employment history|experience)\s*[:\n]\s*(.+)', full_text, flags=re.I|re.S)
                if m3:
                    exp_text = m3.group(2).strip()
            parsed["sections"]["experience"] = exp_text
            parsed["experience"] = extract_experience_from_section(exp_text) if exp_text else []
        except Exception as e:
            log.debug("experience extraction error for %s: %s", path, e)
            parsed["experience"] = []

        # Locations: try sections -> spaCy NER fallback
        locs = []
        try:
            # if there's a contact section, prefer scanning it first
            contact_block = parsed["sections"].get("contact") or parsed["sections"].get("header") or ""
            if contact_block:
                locs = extract_locations_from_text(contact_block)
            if not locs and parsed["sections"]:
                # scan header-ish sections for LOC
                for k in ("contact", "summary", "profile", "header"):
                    b = parsed["sections"].get(k)
                    if b:
                        locs = extract_locations_from_text(b)
                        if locs:
                            break
            if not locs and full_text:
                locs = extract_locations_from_text(full_text)
        except Exception:
            locs = []
        parsed["locations"] = locs

        # Finally, ensure email/phone present, else try extract from contact section
        try:
            if (not parsed.get("email") or not parsed.get("phone")) and parsed["sections"].get("contact"):
                cinfo = extract_contact_from_section(parsed["sections"].get("contact"))
                if not parsed.get("email"):
                    parsed["email"] = cinfo.get("email")
                if not parsed.get("phone"):
                    parsed["phone"] = cinfo.get("phone")
        except Exception:
            pass

    except Exception as e:
        log.debug("parse_resume_file overall error for %s: %s", path, e)

    return parsed

# ===== END: replacement =====
# ----------------- Categorization & JSON formatting -----------------
def categorize_skills_for_resume(skills: List[str], full_text: str = "", fuzzy_cutoff: float = 0.72) -> Dict[str, Any]:
    """
    Categorize a flat list of skill tokens into structured buckets.

    - Uses _FLAT_SKILL_TO_CATEGORY mapping (rebuilt from CSV or taxonomy).
    - Falls back to fuzzy matching via difflib.get_close_matches.
    - If category maps to a technical bucket put in technical; if maps to soft put in soft;
      otherwise put in other.
    - Performs simple token heuristics for uncovered tokens (common tech substrings).
    """
    # prepare outputs
    technical_out = {sub: [] for sub in ("programming_languages","cloud","databases","data_engineering","devops_ci_cd","web_frameworks","analytics_and_viz","testing_and_quality","storage_and_format","other_tech")}
    soft_out = []
    other_out = []

    if not skills:
        return {"technical": technical_out, "soft": [], "other": []}

    # normalize input tokens
    normalized = []
    for s in skills:
        if not s: continue
        tok = s.strip()
        tok = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', tok)  # strip punctuation edges
        tok = tok.strip().lower()
        if tok:
            normalized.append(tok)

    # convenience local maps
    flat_map = _FLAT_SKILL_TO_CATEGORY or {}
    flat_keys = list(flat_map.keys())

    for tok in normalized:
        assigned = False
        cat = None

        # direct mapping from flat_map
        if tok in flat_map:
            cat = flat_map[tok]

        # fuzzy fallback to flat_map keys
        if cat is None and flat_keys:
            try:
                close = get_close_matches(tok, flat_keys, n=1, cutoff=fuzzy_cutoff)
                if close:
                    cat = flat_map.get(close[0])
            except Exception:
                pass

        # normalize category label if present
        if isinstance(cat, str):
            cat_norm = cat.strip().lower()
        else:
            cat_norm = None

        # if category appears to be a tech bucket
        if cat_norm in technical_out:
            if tok not in technical_out[cat_norm]:
                technical_out[cat_norm].append(tok)
            assigned = True

        # handle common synonyms where CSV used 'soft' or 'soft_skills'
        elif cat_norm in ("soft", "soft_skills", "soft-skills"):
            if tok not in soft_out:
                soft_out.append(tok)
            assigned = True

        # other explicit non-tech categories -> other_out
        elif cat_norm:
            # map via CATEGORY_MAPPING if possible
            mapped = CATEGORY_MAPPING.get(cat_norm, cat_norm)
            if mapped in technical_out:
                if tok not in technical_out[mapped]:
                    technical_out[mapped].append(tok)
                assigned = True
            elif mapped in ("soft", "soft_skills", "soft-skills"):
                if tok not in soft_out:
                    soft_out.append(tok)
                assigned = True
            else:
                if tok not in other_out:
                    other_out.append(tok)
                assigned = True

        # heuristics if still unassigned
        if not assigned:
            # soft-keyword detection
            if any(sk in tok for sk in _SOFT_SKILL_KEYWORDS):
                if tok not in soft_out:
                    soft_out.append(tok)
                assigned = True

        if not assigned:
            # match known tech substrings -> map to reasonable bucket
            substr_to_bucket = {
                # programming languages
                "python": "programming_languages", "java": "programming_languages", "c++": "programming_languages", "c#": "programming_languages",
                "javascript": "programming_languages", "typescript": "programming_languages", "scala": "programming_languages", "go": "programming_languages",
                "ruby": "programming_languages", "php": "programming_languages", "r ": "programming_languages", "swift": "programming_languages", "kotlin": "programming_languages",
                # cloud
                "aws": "cloud", "amazon web services": "cloud", "azure": "cloud", "gcp": "cloud", "google cloud": "cloud",
                # data engineering / big data
                "spark": "data_engineering", "hadoop": "data_engineering", "kafka": "data_engineering", "airflow": "data_engineering", "databricks": "data_engineering",
                # databases
                "sql": "databases", "mysql": "databases", "postgres": "databases", "postgresql": "databases", "mongodb": "databases", "oracle": "databases",
                # devops/ci_cd
                "docker": "devops_ci_cd", "kubernetes": "devops_ci_cd", "jenkins": "devops_ci_cd", "terraform": "devops_ci_cd", "ansible": "devops_ci_cd",
                # web frameworks / frontend
                "react": "web_frameworks", "angular": "web_frameworks", "vue": "web_frameworks", "django": "web_frameworks", "flask": "web_frameworks", "spring": "web_frameworks",
                # analytics and viz
                "tableau": "analytics_and_viz", "powerbi": "analytics_and_viz", "matplotlib": "analytics_and_viz", "seaborn": "analytics_and_viz",
                # testing
                "pytest": "testing_and_quality", "junit": "testing_and_quality", "selenium": "testing_and_quality",
                # storage/formats
                "parquet": "storage_and_format", "avro": "storage_and_format", "json": "storage_and_format", "csv": "storage_and_format"
            }
            found_bucket = None
            for sub, bucket in substr_to_bucket.items():
                if sub in tok:
                    found_bucket = bucket
                    break
            if found_bucket:
                if tok not in technical_out[found_bucket]:
                    technical_out[found_bucket].append(tok)
                assigned = True

        # final fallback: place short tokens into other_out
        if not assigned:
            if 1 <= len(tok.split()) <= 4:
                if tok not in other_out:
                    other_out.append(tok)

    # dedupe and sort lists
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
    locations = filter_out_skills_from_locations(locations_raw)
    summary = (parsed_payload.get("summary") or "")[:2000]
    skills_list = parsed_payload.get("skills") or []
    categorized = categorize_skills_for_resume(skills_list, full_text=parsed_payload.get("full_text",""))
    out = {
        "candidate_name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "locations": locations,
        "summary": summary,
        "skills": categorized,
        "experience": parsed_payload.get("experience", [])
    }
    return json.dumps(out, indent=2, ensure_ascii=False)

# ----------------- Qdrant ingestion & search -----------------
def ingest_resumes_to_qdrant(resume_folder: str = RESUME_FOLDER):
    if qdrant_client is None or embedding_model is None:
        print("Qdrant or embedding model not initialized; cannot ingest.")
        return
    if not os.path.exists(resume_folder):
        print(f"Resume folder '{resume_folder}' missing; create and add files.")
        return
    files = [f for f in os.listdir(resume_folder) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    if not files:
        print("No resume files found in resumes/ - add pdf/docx/txt files and re-run.")
        return
    points = []
    for fn in files:
        path = os.path.join(resume_folder, fn)
        print("Parsing:", fn)
        parsed = parse_resume_file(path)
        chunks = [c.strip() for c in re.split(r'\n{2,}', parsed.get("full_text") or "") if c.strip()]
        if not chunks:
            chunks = [(parsed.get("full_text") or "")[:1000]]
        for chunk in chunks:
            vec = embedding_model.encode(chunk).tolist()
            payload = {
                "filename": fn,
                "candidate_name": parsed.get("candidate_name"),
                "candidate_id": parsed.get("candidate_id"),
                "email": parsed.get("email"),
                "skills": parsed.get("skills"),
                "locations": parsed.get("locations"),
                "sections": parsed.get("sections"),
                "experience": parsed.get("experience"),       # <- add this
                "full_text": parsed.get("full_text"),
                "text": chunk[:3000]
            }
            points.append({"id": str(uuid.uuid4()), "vector": vec, "payload": payload})
    BATCH = 128
    for i in range(0, len(points), BATCH):
        batch = points[i:i+BATCH]
        try:
            qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
        except Exception as e:
            print("Qdrant upsert error:", e)
    print(f"Ingested {len(points)} chunks from {len(files)} files into Qdrant.")

def qdrant_semantic_search(query: str, limit: int = 10, filter_candidate_name: str = None, filter_location: str = None):
    """
    Robust search wrapper returning list of {'score':..., 'payload':...}
    Accepts different qdrant client return types.
    """
    if qdrant_client is None or embedding_model is None:
        print("Qdrant or embedding model not initialized; cannot search.")
        return []
    try:
        qvec = embedding_model.encode(query).tolist()
    except Exception as e:
        print("Embedding encode failed:", e); return []
    try:
        resp = qdrant_client.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=200, with_payload=True)
    except Exception as e:
        print("Qdrant search error:", e)
        try:
            resp = qdrant_client.query_points(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=200, with_payload=True)
        except Exception as e2:
            print("Fallback qdrant query failed:", e2); return []
    hits = []
    for r in resp:
        payload = {}
        score = None
        pid = None
        if hasattr(r, "payload"):
            try: payload = r.payload or {}
            except Exception: payload = {}
            score = getattr(r, "score", None)
            pid = getattr(r, "id", None) or getattr(r, "point_id", None)
        elif isinstance(r, dict):
            payload = r.get("payload") or r.get("point", {}).get("payload") or {}
            score = r.get("score", None)
            pid = r.get("id") or r.get("point", {}).get("id")
            if not payload and "result" in r and isinstance(r["result"], dict):
                payload = r["result"].get("payload") or payload
                if score is None:
                    score = r["result"].get("score", score)
        else:
            try:
                payload = getattr(r, "payload", {}) or {}
                score = getattr(r, "score", None)
                pid = getattr(r, "id", None)
            except Exception:
                payload = {}; score = None; pid = None
        if not isinstance(payload, dict):
            try:
                payload = payload.__dict__
            except Exception:
                try: payload = dict(payload)
                except Exception: payload = payload
        if filter_candidate_name:
            cn = (payload.get("candidate_name") or "").strip().lower()
            if filter_candidate_name.lower() not in cn: continue
        if filter_location:
            locs = payload.get("locations") or []
            if isinstance(locs, str): locs = [locs]
            locs = [str(l).lower() for l in (locs or [])]
            if not any(filter_location.lower() in l for l in locs): continue
        hits.append({"id": pid, "score": score, "payload": payload})
    agg = {}
    for h in hits:
        pl = h["payload"] or {}
        cid = (pl.get("candidate_id") or pl.get("email") or pl.get("candidate_name") or h.get("id") or str(uuid.uuid4())).lower()
        prev = agg.get(cid)
        if prev is None:
            agg[cid] = {"score": h["score"], "payload": pl}
        else:
            prev_score = prev.get("score")
            this_score = h.get("score")
            if this_score is not None and (prev_score is None or this_score > prev_score):
                agg[cid] = {"score": this_score, "payload": pl}
    ranked = sorted(agg.values(), key=lambda x: (x.get("score") is not None, x.get("score")), reverse=True)
    return ranked[:limit]

# ----------------- CLI -----------------
def interactive_cli():
    print("--- AI Resume Analyzer (Qdrant + skill model) ---")
    print("Commands: ingest | search <query> [::location] | details <name> | exit")
    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye."); break
        if not cmd: continue
        if cmd.lower() in ("exit", "quit"): break
        if cmd.lower() == "ingest":
            ingest_resumes_to_qdrant(RESUME_FOLDER); continue
        if cmd.lower().startswith("search "):
            q = cmd[len("search "):].strip()
            parts = q.split("::")
            query = parts[0].strip()
            filter_loc = parts[1].strip() if len(parts) > 1 else None
            results = qdrant_semantic_search(query, limit=10, filter_location=filter_loc)
            results_json = []
            for r in results:
                p = r["payload"] or {}
                parsed_out = {
                    "candidate_name": p.get("candidate_name"),
                    "email": p.get("email"),
                    "phone": None,
                    "linkedin": None,
                    "locations": p.get("locations") or [],
                    "skills": p.get("skills") or []
                }
                sec = p.get("sections") or {}
                if isinstance(sec, dict):
                    contact_block = sec.get("contact") or ""
                    cinfo = extract_contact_from_section(contact_block or p.get("text",""))
                    parsed_out["phone"] = cinfo.get("phone")
                    parsed_out["linkedin"] = cinfo.get("linkedin")

                # ensure locations are cleaned (remove skills)
                locations_raw = parsed_out.get("locations") or []
                parsed_out["locations"] = filter_out_skills_from_locations(locations_raw)

                cat = categorize_skills_for_resume(parsed_out.get("skills") or [], full_text="")
                results_json.append({
                    "candidate_name": parsed_out.get("candidate_name"),
                    "email": parsed_out.get("email"),
                    "phone": parsed_out.get("phone"),
                    "locations": parsed_out.get("locations"),
                    "skills": cat,
                    "experience": p.get("experience") or [] 
                })
            print(json.dumps(results_json, indent=2, ensure_ascii=False)); continue
        if cmd.lower().startswith("details "):
            name = cmd[len("details "):].strip().lower()
            res = qdrant_semantic_search(name, limit=200)
            if not res:
                print(json.dumps({"error": "No candidate found"}, indent=2)); continue
            chosen = None
            for r in res:
                p = r["payload"]
                if name == (p.get("candidate_name") or "").strip().lower():
                    chosen = p; break
            if not chosen: chosen = res[0]["payload"]
            phone = chosen.get("phone"); linkedin = chosen.get("linkedin")
            if not phone or not linkedin:
                sec = chosen.get("sections") or {}
                if isinstance(sec, dict):
                    contact_txt = sec.get("contact") or ""
                    c = extract_contact_from_section(contact_txt or chosen.get("text",""))
                    phone = phone or c.get("phone"); linkedin = linkedin or c.get("linkedin")
            parsed_payload = {
                "candidate_name": chosen.get("candidate_name"),
                "email": chosen.get("email"),
                "phone": phone,
                "linkedin": linkedin,
                "locations": chosen.get("locations"),
                "summary": (chosen.get("text") or "")[:800],
                "skills": chosen.get("skills") or [],
                "experience": chosen.get("experience") or [], 
                "full_text": chosen.get("text") or ""
            }
            print(format_candidate_json(parsed_payload)); continue
        print("Unknown command. Use: ingest | search <query> [::location] | details <name> | exit")

if __name__ == "__main__":
    print("Starting chatbot_poc.py")
    print("skill_pipe:", bool(skill_pipe))
    print("embedding_model:", bool(embedding_model))
    print("qdrant_client:", bool(qdrant_client))
    interactive_cli()


