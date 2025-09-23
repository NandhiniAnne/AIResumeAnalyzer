# ===== Model loading for Amjad skill extractor (put near other imports / init) =====
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Option A: if you previously downloaded with snapshot_download, use that local path
# (this avoids re-downloading). Otherwise pipeline will load from HF hub.
try:
    hf_local_path = snapshot_download("amjad-awad/skill-extractor", repo_type="model")
    print("Using local HF snapshot at:", hf_local_path)
except Exception as e:
    print("Could not snapshot_download amjad-awad/skill-extractor (will attempt from hub):", e)
    hf_local_path = None

# Create tokenizer/model/pipeline
skill_pipe = None
try:
    if hf_local_path:
        tokenizer_skill = AutoTokenizer.from_pretrained(hf_local_path)
        model_skill = AutoModelForTokenClassification.from_pretrained(hf_local_path)
    else:
        # fallback: load directly from hub
        tokenizer_skill = AutoTokenizer.from_pretrained("amjad-awad/skill-extractor")
        model_skill = AutoModelForTokenClassification.from_pretrained("amjad-awad/skill-extractor")

    skill_pipe = pipeline("ner", model=model_skill, tokenizer=tokenizer_skill, grouped_entities=True)
    # smoke test so you see something at startup
    try:
        print("Skill model test output:",
              skill_pipe("Experienced in Python, PySpark, AWS and Docker."))
    except Exception as e:
        print("Skill pipeline test failed:", e)
except Exception as e:
    print("Failed to load skill extractor model:", e)
    skill_pipe = None

# chatbot_poc.py (modified to load both en_core_web_trf + amjad-awad/skill-extractor)
import numpy as _np
from huggingface_hub import snapshot_download
import importlib.util
from skill_taxonomy import SKILL_TAXONOMY, _FLAT_SKILL_TO_CATEGORY
from skill_taxonomy import categorize_skills_for_resume, attach_categorized_skills_to_candidate
import os
import re
import uuid
from collections import defaultdict
from difflib import get_close_matches
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
import docx  # python-docx
import spacy
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Qdrant imports: support both qdrant_client and qdrant_client.http.models depending on version
from qdrant_client import QdrantClient
try:
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue, ScoredPoint
except Exception:
    # if older style, import via qdrant_client.models
    from qdrant_client import models as qmodels
    Filter = qmodels.Filter
    FieldCondition = qmodels.FieldCondition
    MatchValue = qmodels.MatchValue
    ScoredPoint = None

# -------- CONFIGURATION --------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "final_hybrid_chunk_collection")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# default spaCy model used for general NLP tasks (NER/GPE/name parsing). If en_core_web_trf is installed, great.
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_trf")
RESUME_FOLDER = os.getenv("RESUME_FOLDER", "resumes")
SKILLS_FILE = os.getenv("SKILLS_FILE", "skills.txt")  # optional
# HF spaCy skill model repo id (can be overwritten via env)
HF_SKILL_MODEL = os.getenv("HF_SKILL_MODEL", "amjad-awad/skill-extractor")

# -------- INIT --------
print("Loading embedding model and qdrant client (spaCy models will load shortly)...")
# We'll lazily load spaCy models with try_load_spacy_models()
nlp = None           # general spaCy model (prefer en_core_web_trf)
nlp_skill = None     # specialized spaCy skill extractor (amjad-awad)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
print("Embedding model & Qdrant client ready. SpaCy models will load on startup.")

# Heading mapping and heuristics (kept same)
HEADING_MAP = {
    "SUMMARY": "summary", "PROFILE": "summary", "OBJECTIVE": "summary",
    "PROFESSIONAL SUMMARY": "summary",
    "EXPERIENCE": "experience", "WORK EXPERIENCE": "experience", "PROFESSIONAL EXPERIENCE": "experience", "EMPLOYMENT HISTORY": "experience",
    "EDUCATION": "education", "ACADEMIC": "education",
    "CERTIFICATIONS": "certifications", "LICENSES": "certifications",
    "SKILLS": "skills", "TECHNICAL SKILLS": "skills", "TECH SKILLS": "skills",
    "PROJECTS": "projects",
    "CONTACT": "contact", "CONTACT INFORMATION": "contact",
    "ACHIEVEMENTS": "achievements", "AWARDS": "achievements",
    "LANGUAGES": "languages", "ADDITIONAL INFORMATION": "additional", "ADDITIONAL INFO": "additional",
}

SECTION_SYNONYMS = {
    "skills": ["skill", "skills", "technical skills", "tech skills", "expertise", "proficiencies"],
    "experience": ["experience", "work experience", "professional experience", "employment", "jobs", "work history"],
    "education": ["education", "degree", "bachelor", "master", "phd", "university", "college", "school"],
    "certifications": ["certification", "certifications", "certificate", "certificates", "certs", "license", "licenses", "credentials"],
    "projects": ["project", "projects", "portfolio"],
    "summary": ["summary", "profile", "objective", "about", "about me"],
    "contact": ["contact", "email", "phone", "linkedin"],
    "publications": ["publication", "publications", "papers"],
    "languages": ["language", "languages"],
    "additional": ["additional information", "additional info", "other", "miscellaneous"]
}
# reverse index
SECTION_KEYWORDS = {kw: sec for sec, kws in SECTION_SYNONYMS.items() for kw in kws}

def clamp01(x: float) -> float:
    """Clamp a number to the [0,1] range."""
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def normalize_token(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r'[\W_]+', ' ', s)  # keep letters/digits, convert punctuation -> space
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def canonicalize_heading(h):
    h_up = h.upper().strip()
    if h_up in HEADING_MAP:
        return HEADING_MAP[h_up]
    if "SKILL" in h_up: return "skills"
    if "EDUC" in h_up: return "education"
    if "EXPERI" in h_up or "EMPLOY" in h_up: return "experience"
    if "CERTIF" in h_up or "LICENSE" in h_up: return "certifications"
    if "PROJECT" in h_up: return "projects"
    if "CONTACT" in h_up or "EMAIL" in h_up or "PHONE" in h_up: return "contact"
    if "SUMMARY" in h_up or "PROFILE" in h_up or "OBJECTIVE" in h_up: return "summary"
    return h.lower().strip()

def split_into_sections(text):
    """
    Tolerant header detection: looks for short lines in Title Case or ALL CAPS and splits to header/body.
    Returns list of (header, body).
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    if not lines:
        return [("body", text)]

    heading_indices = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        # heuristics: short line, title case OR uppercase OR contains colon and short
        if (len(s) < 80) and (s.isupper() or re.match(r'^[A-Z][a-z]+( [A-Za-z0-9&/-]+)*$', s) or (':' in s and len(s.split(':')[0].split()) <= 5)):
            heading_indices.append((i, s))

    # fallback: colon-cued headings if nothing found
    if not heading_indices:
        for i, line in enumerate(lines):
            if ':' in line and len(line) < 120:
                heading_indices.append((i, line.split(':', 1)[0].strip()))

    if not heading_indices:
        return [("body", text)]

    sections = []
    for idx, (i, hdr) in enumerate(heading_indices):
        start = i + 1
        end = heading_indices[idx + 1][0] if idx + 1 < len(heading_indices) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        sections.append((hdr.strip(), body))

    # merge tiny bodies into previous
    merged = []
    for hdr, body in sections:
        if merged and len(body.split()) < 3:
            prev_hdr, prev_body = merged[-1]
            merged[-1] = (prev_hdr, (prev_body + "\n" + hdr + "\n" + body).strip())
        else:
            merged.append((hdr, body))
    return merged

# ------------------------------
# LOAD BOTH spaCy MODELS (en_core_web_trf + HF skill model)
# ------------------------------
def try_load_spacy_models():
    """
    Try to load:
      - general spaCy model into `nlp` (prefer SPACY_MODEL env, default en_core_web_trf)
      - HF spaCy skill model into `nlp_skill` (HF_SKILL_MODEL env)
    Returns tuple (nlp_loaded, nlp_skill_loaded)
    """
    global nlp, nlp_skill
    nlp = None
    nlp_skill = None

    # 1) Try general model
    if SPACY_MODEL == "en_core_web_trf":
        try:
            print("[spacy] Trying to load en_core_web_trf for general NLP...")
            nlp = spacy.load("en_core_web_trf")
            print("[spacy] Loaded en_core_web_trf as general NLP model.")
        except Exception as e:
            print("[spacy] Could not load en_core_web_trf locally:", e)
            # attempt to load a small installed model as fallback
            try:
                print("[spacy] Trying to load 'en_core_web_sm' as fallback...")
                nlp = spacy.load("en_core_web_sm")
                print("[spacy] Loaded en_core_web_sm as general NLP model.")
            except Exception as e2:
                print("[spacy] Could not load en_core_web_sm either:", e2)
                nlp = None
    else:
        # custom SPACY_MODEL name or local dir
        try:
            print(f"[spacy] Trying to load SPACY_MODEL='{SPACY_MODEL}' ...")
            # if SPACY_MODEL is a directory use spacy.load, else try to load by name
            if os.path.isdir(SPACY_MODEL):
                nlp = spacy.load(SPACY_MODEL)
            else:
                nlp = spacy.load(SPACY_MODEL)
            print(f"[spacy] Loaded general spaCy model: {SPACY_MODEL}")
        except Exception as e:
            print(f"[spacy] Failed to load SPACY_MODEL '{SPACY_MODEL}':", e)
            # fallback to small model
            try:
                nlp = spacy.load("en_core_web_sm")
                print("[spacy] Loaded en_core_web_sm as fallback general model.")
            except Exception as e2:
                print("[spacy] No spaCy general model available:", e2)
                nlp = None

    # 2) Try HF skill model (amjad-awad/skill-extractor by default)
    try:
        print(f"[spacy] Attempting to download/load HF spaCy skill model '{HF_SKILL_MODEL}' ...")
        model_path = snapshot_download(HF_SKILL_MODEL, repo_type="model")
        try:
            nlp_skill = spacy.load(model_path)
            print(f"[spacy] Loaded HF skill extractor from: {model_path}")
        except Exception as e:
            print("[spacy] Failed to load HF skill model as spaCy model:", e)
            nlp_skill = None
    except Exception as e:
        print("[spacy] Could not download HF skill model:", e)
        nlp_skill = None

    return nlp is not None, nlp_skill is not None

# ------------------------------
# Combined skill extraction using both models
# ------------------------------
# words we definitely don't want to treat as skills
NON_SKILL_WORDS = {
    "ability","accepted","developed","provided","provides","providing","worked",
    "experience","experienced","years","year","team","business","clients","manager",
    "management","lead","leading","responsible","responsibility","work"
}

def cleanup_span(s: str) -> str:
    s2 = s.strip()
    s2 = re.sub(r'^[\W_]+|[\W_]+$', '', s2)
    return s2

def is_valid_skill_span(span_text: str, trf_doc=None) -> bool:
    st = span_text.strip()
    if not st:
        return False
    if st.lower() in NON_SKILL_WORDS:
        return False
    if trf_doc is None:
        return True
    # attempt to locate the span in trf_doc and check token POS
    txt = trf_doc.text.lower()
    low = st.lower()
    start = txt.find(low)
    if start == -1:
        return True
    end = start + len(low)
    overlapping = [t for t in trf_doc if not (t.idx + len(t.text) <= start or t.idx >= end)]
    if overlapping and all(t.pos_ in ("VERB","AUX","ADV") for t in overlapping):
        return False
    if any(t.pos_ in ("NOUN","PROPN","SYM") for t in overlapping):
        return True
    return True

def extract_skills_from_section_combined(text: str, use_whitelist: bool = False, whitelist: set = None) -> List[str]:
    """
    Robust extractor for skills sections:
     - deterministic parse of label: list lines (Programming Languages:, Front-End Technologies:, ...)
     - union with nlp_skill spans (if available)
     - light POS/verb filtering using nlp (if available)
     - normalize/dedupe
    """
    if not text:
        return []

    NON_SKILL_WORDS = {"ability","experience","years","team","work","responsible","provides","provided","developed"}
    candidates = []

    # 1) Deterministic parse: common "Label: item, item, item" patterns
    for line in text.splitlines():
        ln = line.strip(" •\t- ")
        if not ln:
            continue
        # If the line contains ":" and RHS contains commas -> very likely a skill list
        if ':' in ln:
            left, right = ln.split(':', 1)
            if ',' in right or right.strip().count(' ') < 6:  # right-hand list or short list
                parts = re.split(r'[,\|/·;]', right)
                for p in parts:
                    p2 = p.strip().lower()
                    if p2 and p2 not in NON_SKILL_WORDS:
                        candidates.append(p2)
                continue
        # If line begins with "-" or bullet and contains commas, split
        if ',' in ln and len(ln) < 300:
            parts = [p.strip().lower() for p in ln.split(',') if p.strip()]
            candidates.extend([p for p in parts if p not in NON_SKILL_WORDS])
            continue
        # pipe/slash separated
        if '|' in ln or '/' in ln or '·' in ln:
            parts = re.split(r'[|/·]', ln)
            candidates.extend([p.strip().lower() for p in parts if p.strip() and p.strip().lower() not in NON_SKILL_WORDS])
            continue

    # 2) Model-based extraction (if available) -- add those spans too
    try:
        if nlp_skill is not None:
            doc = nlp_skill(text)
            for ent in doc.ents:
                lab = getattr(ent, "label_", "") or getattr(ent, "label", "")
                if lab and ("skill" in str(lab).lower() or "tech" in str(lab).lower()):
                    cand = re.sub(r'^[\W_]+|[\W_]+$', '', ent.text).strip().lower()
                    if cand and cand not in NON_SKILL_WORDS:
                        candidates.append(cand)
    except Exception:
        pass

    # 3) Light POS filter via nlp (reject pure verbs/adverbs), but *don't* be overly strict
    final = []
    trf_doc = None
    if nlp is not None:
        try:
            trf_doc = nlp(text)
        except Exception:
            trf_doc = None

    for cand in candidates:
        cand_clean = cand.strip().strip(',.')
        if not cand_clean:
            continue
        if trf_doc is not None:
            # locate tokens overlapping candidate; if all are verbs/adverbs -> reject
            txt = trf_doc.text.lower()
            pos = txt.find(cand_clean.lower())
            if pos != -1:
                end = pos + len(cand_clean)
                overlapping = [t for t in trf_doc if not (t.idx + len(t.text) <= pos or t.idx >= end)]
                if overlapping and all(t.pos_ in ("VERB","AUX","ADV") for t in overlapping):
                    continue
        if cand_clean in NON_SKILL_WORDS:
            continue
        # optional whitelist enforcement
        if use_whitelist and whitelist:
            if cand_clean not in whitelist:
                continue
        final.append(cand_clean)

    # Deduplicate preserving order
    seen = set()
    ordered = []
    for s in final:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered
# -------------------- Stricter Skill post-processing helper (REPLACE existing) --------------------
import re
from difflib import get_close_matches

# tuned regexes
_PHONE_RE = re.compile(r'(\+?\d[\d\-\s().]{6,}\d)')
_EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-z]{2,}', re.I)
_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')
# tech token regex: common tech words and allow short tokens like "sql", "rdd", "ci/cd", "etl"
_TECH_TOKEN_RE = re.compile(
    r'\b(java|python|javascript|typescript|react|angular|node|go|golang|c\+\+|c#|scala|sql|pyspark|spark|hadoop|hive|hiveql|'
    r'kafka|airflow|apache spark|apache kafka|jenkins|git|github|gitlab|azure|aws|gcp|google bigquery|bigquery|redshift|snowflake|'
    r'databricks|dbt|kubernetes|docker|hdfs|sqoop|oozie|parquet|avro|json|nosql|mongodb|cassandra|elasticsearch|redis|'
    r'powerbi|tableau|lookml|power bi|pl/sql|plsql|t-sql|tsql|unix|linux|ubuntu|centos|shell|bash|pyspark|scala|spark sql)\b', re.I)

# small english stopwords set (extend as needed)
_STOPWORDS = {
    "and","or","the","a","an","in","on","of","for","with","to","from","such","as","is","are","be","by","at",
    "i","we","they","you","he","she","it","this","that","these","those","have","has","had","do","does","did",
    "will","would","should","can","could","may","might","our","their","its","but","not","into","about","per",
    "using","used","using","include","includes","including","experience","years","year","responsible","responsibilities",
    "skilled","skill","skills","expertise","proficient","proficiently","knowledge","knowledgeable","developed","developing",
    "implement","implementing","implemented","ability","abilities","strong","hands","on","work","works","working","worked"
}

# Toggle: if True, only keep tokens that match (or fuzzy-match) the provided whitelist.
STRICT_WHITELIST = True

def post_process_skills(raw_skills, full_text=None, whitelist: set = None, fuzzy_cutoff: float = 0.78, too_long_words: int = 8):
    """
    Stricter cleaning of raw skill candidates. Returns deduped list of short canonical tokens.
    - STRICT_WHITELIST global toggles whitelist-only behaviour.
    """
    def norm(s: str) -> str:
        if not s: return ""
        s2 = s.strip()
        s2 = re.sub(r'\s+', ' ', s2)
        s2 = re.sub(r'^[\W_]+|[\W_]+$', '', s2)
        return s2

    wl_map = None
    if whitelist:
        wl_map = {w.lower(): w for w in whitelist}

    cleaned = []
    seen = set()
    full_text_low = (full_text or "").lower()

    for raw in (raw_skills or []):
        s = norm(raw)
        if not s:
            continue
        s_low = s.lower()

        # Remove emails and phones quickly
        if _EMAIL_RE.search(s_low) or _PHONE_RE.search(s_low):
            continue

        # Remove garbage long digit strings (IDs, long phone shards)
        if re.search(r'\d{4,}', s_low) and not re.search(r'python3|java8|java11|c\+\+|c#', s_low):
            # allow e.g. "python3", "java8" but not long digit sequences
            if len(re.findall(r'\d', s_low)) > 3:
                continue

        # remove year tokens or short date fragments
        if _YEAR_RE.search(s_low) and len(s_low.split()) <= 3:
            continue

        # strip leading/trailing common noise words produced by sentence capture (e.g. "and", "including", "such as")
        s_low = re.sub(r'^(and|including|including:|such as|such|with|using|use)\s+', '', s_low)
        s_low = re.sub(r'\s+(and|including|including:|such as|such|with|using|use)$', '', s_low)
        s_low = s_low.strip()
        if not s_low:
            continue

        # If the candidate is a long sentence-like fragment, try to extract tech tokens from it
        if len(s_low.split()) > too_long_words:
            found = set(m.group(0).strip() for m in _TECH_TOKEN_RE.finditer(s_low))
            # also try token-level extraction: words with punctuation like 'ci/cd' or 'rdd'
            token_like = re.findall(r'[A-Za-z\+#\./-]{2,}', s_low)
            for t in token_like:
                if len(t) <= 2:
                    continue
                if _TECH_TOKEN_RE.search(t):
                    found.add(t)
            if not found:
                # attempt to capture short alphanumeric tokens (e.g., 'spark', 'jenkins')
                candidates = [tk.lower() for tk in token_like if 3 <= len(tk) <= 30]
                found.update(candidates[:6])
            for tk in found:
                tk_norm = tk.strip().lower()
                # map to whitelist if available
                if wl_map and tk_norm in wl_map:
                    out = wl_map[tk_norm]
                else:
                    out = tk_norm
                if out and out.lower() not in seen:
                    seen.add(out.lower()); cleaned.append(out)
            # skip adding the original long fragment
            continue

        # Drop very short tokens (1-2 chars) unless in whitelist or matched by tech regex (e.g., 'sql', 'ci')
        if len(s_low) <= 2:
            if wl_map and s_low in wl_map:
                out = wl_map[s_low]
                if out.lower() not in seen:
                    seen.add(out.lower()); cleaned.append(out)
            elif _TECH_TOKEN_RE.search(s_low):
                if s_low not in seen:
                    seen.add(s_low); cleaned.append(s_low)
            else:
                continue

        # Drop plain stopwords and filler tokens
        if s_low in _STOPWORDS:
            continue

        # Prefer multi-word canonical tokens to be in whitelist or present verbatim in resume
        if len(s_low.split()) > 1:
            if wl_map:
                # map by exact or fuzzy
                if s_low in wl_map:
                    out = wl_map[s_low]
                else:
                    close = get_close_matches(s_low, wl_map.keys(), n=1, cutoff=fuzzy_cutoff)
                    if close:
                        out = wl_map[close[0]]
                    else:
                        # keep if appears verbatim in resume text and contains tech token
                        if s_low in full_text_low and _TECH_TOKEN_RE.search(s_low):
                            out = s
                        else:
                            # otherwise skip multiword phrase (too noisy)
                            continue
            else:
                # no whitelist — allow the phrase only if it contains a tech token
                if not _TECH_TOKEN_RE.search(s_low):
                    continue
                out = s

        else:
            # Single-word (length 3+): keep if passes tech regex OR maps to whitelist OR is not a stopword and looks techy
            if wl_map and s_low in wl_map:
                out = wl_map[s_low]
            elif _TECH_TOKEN_RE.search(s_low):
                out = s_low
            else:
                # use spaCy POS heuristic to drop verbs/adverbs (if available)
                try:
                    if nlp is not None:
                        doc = nlp(s)
                        # if most tokens are verbs/adverbs then skip
                        pos_good = sum(1 for t in doc if t.pos_ in ("NOUN", "PROPN", "ADJ", "SYM"))
                        pos_total = max(1, sum(1 for t in doc if t.is_alpha or t.pos_))
                        if pos_good / pos_total < 0.35:
                            continue
                except Exception:
                    pass
                out = s_low

        # enforce strict whitelist mode if requested
        if STRICT_WHITELIST and wl_map:
            out_low = out.lower()
            if out_low not in wl_map:
                # try fuzzy match
                close = get_close_matches(out_low, wl_map.keys(), n=1, cutoff=fuzzy_cutoff)
                if close:
                    out = wl_map[close[0]]
                else:
                    continue

        # final normalization and dedupe
        out_norm = re.sub(r'^[\W_]+|[\W_]+$', '', str(out)).strip()
        if not out_norm:
            continue
        out_low = out_norm.lower()
        if out_low not in seen:
            seen.add(out_low)
            cleaned.append(out_norm)

    # final cleanup: sort mostly by insertion order preserved in list; optionally prefer canonical case from whitelist
    if wl_map:
        # map lower->canonical if available
        final = []
        for c in cleaned:
            key = c.lower()
            if key in wl_map:
                final.append(wl_map[key])
            else:
                final.append(c)
        return final
    return cleaned


# small extractors (note: extract_skills_from_section replaced above)
def extract_education_from_section(text):
    if not text: return {"schools": [], "degrees": []}
    schools = set()
    degrees = set()
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
                if m:
                    schools.add(m.group(1).strip())
        else:
            m = re.search(r'((university|college|institute)[^,;\n]*)', ln, re.I)
            if m:
                schools.add(m.group(1).strip())
    return {"schools": [s.lower() for s in sorted(schools)], "degrees": [d for d in sorted(degrees)]}

def extract_experience_from_section(text):
    if not text:
        return []
    jobs = []
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    i = 0
    year_re = re.compile(r'\b\d{4}\b')
    separator_re = re.compile(r'[—–\-\@\|,()]')
    date_token_re = re.compile(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[^\d]*\d{4}\b|\b\d{4}\b|\d{2}/\d{4})', re.I)

    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue

        looks_like_entry = bool(year_re.search(ln)) or bool(separator_re.search(ln)) or (len(ln.split()) <= 8 and ln[0].isupper())

        if looks_like_entry:
            dates = date_token_re.findall(ln)
            parts = re.split(r'[-–—@,|()]+', ln)
            title_candidate = parts[0].strip() if parts else ln
            company_candidate = parts[1].strip() if len(parts) > 1 else ""
            j = i + 1
            summary_lines = []
            while j < len(lines):
                nxt = lines[j].strip()
                if year_re.search(nxt) or separator_re.search(nxt) or (len(nxt.split()) <= 6 and nxt.isupper()):
                    break
                summary_lines.append(nxt)
                j += 1
            jobs.append({
                "title": title_candidate,
                "company": company_candidate,
                "dates": dates,
                "summary": " ".join(summary_lines)[:1200]
            })
            i = j
        else:
            i += 1

    if not jobs:
        jobs.append({"title": "", "company": "", "dates": [], "summary": text[:1000]})
    return jobs

def extract_certifications_from_section(text):
    if not text: return []
    lines = [ln.strip(" •\t-") for ln in text.splitlines() if ln.strip()]
    certs = set()
    for ln in lines:
        if re.search(r'\b(certified|certificate|certification|cissp|cism|cisa|pmp|ccna|ccnp)\b', ln, re.I) or len(ln.split()) <= 8:
            certs.add(ln.strip())
    return sorted(certs)

def extract_contact_from_section(text):
    if not text: return {}
    email = re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-z]{2,}', text)
    phone = re.search(r'(\+?\d[\d\s().-]{7,}\d)', text)
    linkedin = re.search(r'(linkedin\.com/[^\s,;]+)', text, re.I)
    return {"email": email.group(0) if email else None, "phone": phone.group(0).strip() if phone else None, "linkedin": linkedin.group(0) if linkedin else None}

# normalize name functions unchanged
def normalize_name(name):
    if not name: return "Unknown"
    name = re.sub(r'(?i)\b(resume|cv|curriculum vitae|profile|project history|final|de|candidate)\b', '', name)
    name = re.sub(r'[_\-\.\d]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.title() if name else "Unknown"

def extract_candidate_name_from_text(full_text, filename=None):
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    skip_kw = re.compile(r'(?i)\b(resume|cv|curriculum vitae|profile|project history|objective|summary|contact|phone|email|address|linkedin|github)\b')
    for ln in lines[:12]:
        if skip_kw.search(ln): continue
        words = ln.split()
        if 1 < len(words) <= 4 and not any(ch.isdigit() for ch in ln):
            if all(w[0].isupper() for w in words) or ln.isupper():
                return normalize_name(ln)
    snippet = "\n".join(lines[:30])
    doc = nlp(snippet) if nlp is not None else None
    if doc is not None:
        persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
        for p in sorted(persons, key=lambda s: -len(s)):
            if not skip_kw.search(p):
                return normalize_name(p)
    if filename:
        base = os.path.splitext(os.path.basename(filename))[0]
        base_clean = re.sub(r'[_\-\.]+', ' ', base)
        base_clean = re.sub(r'(?i)\b(resume|cv|final|de|profile)\b', '', base_clean)
        if base_clean.strip():
            return normalize_name(base_clean)
    return "Unknown"

# parse resume file (unchanged except location of extract_skills_from_section)
def extract_pdf_text_blocks(path):
    """
    Use PyMuPDF (fitz) to extract page blocks and mark table-like blocks.
    Returns:
      - full_text (concatenated plain text)
      - blocks_info: list of dicts {page, bbox, text, block_no, is_likely_table}
    """
    blocks_info = []
    full_text = ""
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"Error opening PDF {path} with fitz: {e}")
        return "", []

    for pno, page in enumerate(doc):
        try:
            pd = page.get_text("dict")
        except Exception:
            txt = page.get_text() or ""
            full_text += txt + "\n"
            blocks_info.append({"page": pno, "bbox": None, "text": txt, "block_no": 0, "is_likely_table": False})
            continue

        page_blocks = pd.get("blocks", [])
        for b_idx, blk in enumerate(page_blocks):
            block_text_lines = []
            if blk.get("type", 0) == 0:
                for line in blk.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    line_text = line_text.strip()
                    if line_text:
                        block_text_lines.append(line_text)
            else:
                continue

            block_text_join = "\n".join(block_text_lines).strip()
            if not block_text_join:
                continue
            full_text += block_text_join + "\n"

            # heuristics for table-like blocks
            comma_count = block_text_join.count(",")
            pipe_count = block_text_join.count("|")
            short_lines = sum(1 for l in block_text_lines if len(l.split()) <= 6)
            is_table = False
            if comma_count >= 3 or pipe_count >= 2 or short_lines >= max(3, len(block_text_lines)//2):
                is_table = True

            bbox = blk.get("bbox", None)
            blocks_info.append({
                "page": pno,
                "bbox": bbox,
                "text": block_text_join,
                "block_no": b_idx,
                "is_likely_table": is_table
            })
    try:
        doc.close()
    except Exception:
        pass
    return full_text, blocks_info


def parse_resume_file(file_path):
    """
    Read PDF / DOCX / TXT, detect table-like PDF blocks (skills), merge into sections,
    then extract skills and categorize them.
    """
    full_text = ""
    skills_section_text = ""  # optional merged skills text from PDF tables

    # Read file content (PDF / DOCX / TXT)
    if file_path.lower().endswith(".pdf"):
        try:
            # Use block extraction to preserve columns / tables
            full_text_blocks, pdf_blocks = extract_pdf_text_blocks(file_path)
            full_text = full_text_blocks

            # Try to detect explicit "skills" blocks: look for header cues first
            skills_section_candidates = []
            for i, blk in enumerate(pdf_blocks):
                low = (blk.get("text") or "").lower()
                # header-like block
                if ("technical skills" in low or "technical skill" in low or
                    low.strip().startswith("skills") or "technical skillset" in low):
                    gathered = [blk["text"]]
                    # look ahead a few blocks for likely table/list blocks
                    for j in range(i+1, min(i+4, len(pdf_blocks))):
                        if pdf_blocks[j].get("is_likely_table") or len(pdf_blocks[j].get("text","").splitlines()) <= 8:
                            gathered.append(pdf_blocks[j]["text"])
                    skills_section_candidates.append("\n".join(gathered))

            # If no header-cued blocks, scan table-like blocks for tech tokens
            if not skills_section_candidates:
                tech_token_re = re.compile(r'\b(java|python|javascript|react|aws|azure|docker|kubernetes|sql|spark|hadoop|scala|pyspark|hive)\b', re.I)
                for blk in pdf_blocks:
                    if blk.get("is_likely_table") and tech_token_re.search(blk.get("text", "")):
                        skills_section_candidates.append(blk["text"])

            if skills_section_candidates:
                skills_section_text = "\n\n".join(skills_section_candidates)
            else:
                # leave skills_section_text empty so the fallback extractor will scan full_text
                skills_section_text = ""

        except Exception as e:
            print(f"Error reading PDF {file_path} with enhanced reader: {e}")
            # fallback to simple page.get_text()
            try:
                doc = fitz.open(file_path)
                for page in doc:
                    full_text += page.get_text() + "\n"
                doc.close()
            except Exception as e2:
                print(f"Fallback error reading PDF {file_path}: {e2}")

    elif file_path.lower().endswith(".docx"):
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text and para.text.strip():
                    full_text += para.text + "\n"
            # include table cells (useful for skill tables)
            try:
                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            if cell.text and cell.text.strip():
                                full_text += cell.text + "\n"
            except Exception:
                pass
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            full_text = ""

    # ---- sections parsing ----
    raw_sections = split_into_sections(full_text)
    sections_canonical = {}
    for header, body in raw_sections:
        key = canonicalize_heading(header)
        if key in sections_canonical:
            sections_canonical[key] += "\n\n" + body
        else:
            sections_canonical[key] = body

    # If we detected table-based skills in a PDF, merge them into the canonical skills section
    if skills_section_text:
        if "skills" in sections_canonical:
            # avoid duplicating identical content
            if skills_section_text not in sections_canonical["skills"]:
                sections_canonical["skills"] += "\n\n" + skills_section_text
        else:
            sections_canonical["skills"] = skills_section_text

    # ---- skills extraction and categorization ----
    # ---- extract raw skills (model + heuristics) ----
    try:
        raw_skills = extract_skills_from_section_combined(sections_canonical.get("skills", ""), full_text=full_text)
    except TypeError:
        raw_skills = extract_skills_from_section_combined(sections_canonical.get("skills", ""))

    # ---- build a whitelist (if skill_taxonomy exists) ----
    whitelist = None
    try:
        whitelist_set = set()
        try:
            # if you imported SKILL_TAXONOMY from skill_taxonomy.py
            for cat, toks in SKILL_TAXONOMY.items():
                for tkn in toks:
                    whitelist_set.add(tkn.lower())
            whitelist = whitelist_set if whitelist_set else None
        except Exception:
            try:
                whitelist = set(k.lower() for k in _FLAT_SKILL_TO_CATEGORY.keys())
            except Exception:
                whitelist = None
    except Exception:
        whitelist = None

    # ---- post-process raw skills to remove sentences, emails, phones, dates, etc. ----
    skills = post_process_skills(raw_skills, full_text=full_text, whitelist=whitelist, fuzzy_cutoff=0.78, too_long_words=6)

    # ---- categorize cleaned skills using taxonomy ----
    try:
        categorized = categorize_skills_for_resume(skills, full_text=full_text, fuzzy_cutoff=0.7)
    except TypeError:
        categorized = categorize_skills_for_resume(skills, full_text=full_text)

    # ---- other extractions ----
    education_info = extract_education_from_section(sections_canonical.get("education", ""))
    experience = extract_experience_from_section(sections_canonical.get("experience", ""))
    certifications = extract_certifications_from_section(sections_canonical.get("certifications", ""))
    contact = extract_contact_from_section(sections_canonical.get("contact", full_text))
    summary = sections_canonical.get("summary", "")

    # ---- candidate name / id ----
    candidate_name = extract_candidate_name_from_text(full_text, filename=file_path)
    email = contact.get("email")
    candidate_id = (email or candidate_name or os.path.splitext(os.path.basename(file_path))[0]).lower()

    # ---- build parsed dict to return ----
    parsed = {
        "full_text": full_text,
        "candidate_name": candidate_name,
        "candidate_id": candidate_id,
        "email": email,
        "phone": contact.get("phone"),
        "linkedin": contact.get("linkedin"),
        "skills": skills,                      # cleaned skills (post-processed)
        "skills_categorized": categorized,     # taxonomy output (dict)
        "locations": extract_locations_from_text(full_text),
        "education": education_info.get("schools", []),
        "education_degrees": education_info.get("degrees", []),
        "experience": experience,
        "certifications": certifications,
        "sections": sections_canonical,
        "summary": summary,
    }

    return parsed



# small helper to collect GPEs (unchanged)
def extract_locations_from_text(text):
    doc = nlp(text) if nlp is not None else None
    locs = set()
    if doc is not None:
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                locs.add(ent.text.strip())
    return [l.lower() for l in sorted(locs)] if locs else []

# ------------------ Qdrant ingestion (unchanged) ------------------
def ingest_resumes():
    print(f"Ingesting resumes from folder '{RESUME_FOLDER}'...")
    supported_formats = (".pdf", ".docx", ".txt")
    if not os.path.exists(RESUME_FOLDER):
        os.makedirs(RESUME_FOLDER, exist_ok=True)
        print(f"Folder '{RESUME_FOLDER}' created. Put resumes there and re-run.")
        return

    files = [f for f in os.listdir(RESUME_FOLDER) if f.lower().endswith(supported_formats)]
    if not files:
        print("No resume files found in resumes/ - add pdf/docx/txt files and re-run.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    points = []
    for filename in files:
        path = os.path.join(RESUME_FOLDER, filename)
        print(f"Parsing: {filename}")
        parsed = parse_resume_file(path)
        chunks = splitter.split_text(parsed["full_text"] or "")
        if not chunks:
            chunks = [parsed["full_text"] or ""]

        for chunk in chunks:
            vec = embedding_model.encode(chunk).tolist()
            payload = {
                "filename": filename,
                "candidate_name": parsed["candidate_name"],
                "candidate_id": parsed["candidate_id"],
                "email": parsed["email"],
                "phone": parsed["phone"],
                "linkedin": parsed["linkedin"],
                "skills": parsed["skills"],
                "locations": parsed["locations"],
                "education": parsed["education"],
                "certifications": parsed["certifications"],
                "sections": parsed["sections"],
                "text": chunk,
            }
            points.append({"id": str(uuid.uuid4()), "vector": vec, "payload": payload})

    # create/recreate collection
    try:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": embedding_model.get_sentence_embedding_dimension(), "distance": "Cosine"},
        )
    except Exception:
        # older client may use models.VectorParams
        try:
            from qdrant_client import models
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=embedding_model.get_sentence_embedding_dimension(), distance=models.Distance.COSINE),
            )
        except Exception as e:
            print("Warning: couldn't call recreate_collection with either signature:", e)

    # upsert points
    try:
        # prefer http client upsert signature
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    except Exception:
        # older client uses different models.PointStruct
        try:
            from qdrant_client import models
            pts = []
            for p in points:
                pts.append(models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]))
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=pts, wait=True)
        except Exception as e:
            print("Failed to upsert points:", e)
            raise

    print(f"Ingested {len(points)} chunks from {len(files)} files.")

# ------------------ aggregation & helpers (unchanged) ------------------
def aggregate_hits(points):
    grouped = defaultdict(lambda: {
        "filenames": set(),
        "candidate_name": None,
        "emails": set(),
        "phones": set(),
        "linkedin": set(),
        "skills": set(),
        "locations": set(),
        "education": set(),
        "certifications": set(),
        "experience": [],       # <-- ensure experience list exists and is merged
        "sections": {},
        "contexts": []
    })
    for p in points:
        payload = p.payload if hasattr(p, "payload") else p.get("payload", {})
        cid = (payload.get("candidate_id") or payload.get("email") or payload.get("candidate_name") or payload.get("filename") or str(uuid.uuid4())).lower()
        g = grouped[cid]

        if payload.get("candidate_name"):
            g["candidate_name"] = payload.get("candidate_name")
        if payload.get("email"):
            g["emails"].add(payload.get("email"))
        if payload.get("phone"):
            g["phones"].add(payload.get("phone"))
        if payload.get("linkedin"):
            g["linkedin"].add(payload.get("linkedin"))

        for s in (payload.get("skills") or []):
            if s:
                g["skills"].add(s.lower())
        for l in (payload.get("locations") or []):
            if l:
                g["locations"].add(l.lower())
        for ed in (payload.get("education") or []):
            if ed:
                g["education"].add(ed.lower())
        for c in (payload.get("certifications") or []):
            if c:
                g["certifications"].add(c)

        # merge experience entries if present in payload
        for job in (payload.get("experience") or []):
            if job:
                if job not in g["experience"]:
                    g["experience"].append(job)

        # merge sections dicts (keep section text; prefer existing if present)
        sec = payload.get("sections") or {}
        if isinstance(sec, dict):
            for k, v in sec.items():
                if k not in g["sections"]:
                    g["sections"][k] = v
                else:
                    if v and v not in g["sections"][k]:
                        g["sections"][k] += "\n\n" + v

        if payload.get("filename"):
            g["filenames"].add(payload.get("filename"))

        if payload.get("text"):
            g["contexts"].append(payload.get("text"))
    return grouped

class SimplePoint:
    def __init__(self, id_, payload):
        self.id = id_
        self.payload = payload

def normalize_qdrant_points(resp_points):
    normalized = []
    for p in resp_points:
        if hasattr(p, "payload"):
            normalized.append(p)
        elif isinstance(p, dict):
            pid = p.get("id") or p.get("point_id") or p.get("point", {}).get("id")
            payload = p.get("payload") or p.get("point", {}).get("payload") or {}
            normalized.append(SimplePoint(pid, payload))
        else:
            try:
                normalized.append(SimplePoint(p.id, p.payload))
            except Exception:
                continue
    return normalized

# ------------------ query helpers (unchanged) ------------------
def extract_name_from_query(query_text, all_known_names):
    doc = nlp(query_text) if nlp is not None else None
    persons = [ent.text.strip().lower() for ent in doc.ents if ent.label_ == "PERSON"] if doc is not None else []
    if persons:
        person = persons[0]
        for name in all_known_names:
            if person == name or all(tok in name for tok in person.split()):
                return name
        match = get_close_matches(person, all_known_names, n=1, cutoff=0.7)
        if match:
            return match[0]
    q_tokens = set(re.findall(r'\w+', query_text.lower()))
    for name in all_known_names:
        name_tokens = set(name.split())
        if name_tokens and name_tokens.issubset(q_tokens):
            return name
    candidates = []
    for name in all_known_names:
        name_tokens = name.split()
        if not name_tokens:
            continue
        overlap = sum(1 for t in name_tokens if t in q_tokens) / len(name_tokens)
        if overlap >= 0.6:
            candidates.append((overlap, name))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    joined = " ".join(sorted(q_tokens))
    match = get_close_matches(joined, all_known_names, n=1, cutoff=0.6)
    return match[0] if match else None

def map_query_to_section(query_text):
    q = query_text.lower()
    for kw, sec in SECTION_KEYWORDS.items():
        if kw in q:
            return sec
    prefixes = ("what are", "what is", "list", "show", "give me", "provide", "display", "give")
    if any(q.startswith(p) for p in prefixes):
        for kw, sec in SECTION_KEYWORDS.items():
            if kw in q:
                return sec
    doc = nlp(query_text) if nlp is not None else None
    if doc is not None:
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in SECTION_KEYWORDS:
                return SECTION_KEYWORDS[lemma]
    return None

# Qdrant query by section unchanged (kept as-is)
def qdrant_query_by_section(section_key, text_filter=None, limit=500):
    results = []
    try:
        fv = FieldCondition(key=f"sections.{section_key}", match=MatchValue(value=text_filter.lower() if text_filter else ""))
        filt = Filter(must=[fv]) if text_filter else None
        try:
            resp = qdrant_client.query_points(collection_name=COLLECTION_NAME, query_filter=filt, limit=limit, with_payload=True, with_vector=False)
            resp_points = resp.get("result", resp.get("points", []))
            results = normalize_qdrant_points(resp_points)
        except Exception:
            points, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=2000)
            pts = normalize_qdrant_points(points)
            filtered = []
            for p in pts:
                secobj = p.payload.get("sections") or {}
                sec_text = ""
                if isinstance(secobj, dict):
                    sec_text = secobj.get(section_key, "")
                if sec_text:
                    if text_filter:
                        if text_filter.lower() in sec_text.lower():
                            filtered.append(p)
                    else:
                        filtered.append(p)
            results = filtered
    except Exception:
        try:
            points, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=2000)
            pts = normalize_qdrant_points(points)
            filtered = []
            for p in pts:
                secobj = p.payload.get("sections") or {}
                sec_text = ""
                if isinstance(secobj, dict):
                    sec_text = secobj.get(section_key, "")
                if sec_text:
                    if text_filter:
                        if text_filter.lower() in sec_text.lower():
                            filtered.append(p)
                    else:
                        filtered.append(p)
            results = filtered
        except Exception as e:
            print("Error querying qdrant by section:", e)
            results = []
    return results
def extract_pdf_text_blocks(path):
    """
    Use PyMuPDF (fitz) to pull structured page blocks and spans.
    Returns:
      - full_text (concatenated plain text)
      - blocks_info: list of dicts {page, bbox, text, block_no, is_likely_table}
    Heuristic for 'likely_table': block contains many commas/pipes OR many short lines.
    """
    blocks_info = []
    full_text = ""
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"Error opening PDF {path} with fitz: {e}")
        return "", []

    for pno, page in enumerate(doc):
        try:
            pd = page.get_text("dict")  # structured dict with blocks -> lines -> spans
        except Exception:
            # fallback to simple text
            txt = page.get_text() or ""
            full_text += txt + "\n"
            blocks_info.append({"page": pno, "bbox": None, "text": txt, "block_no": 0, "is_likely_table": False})
            continue

        page_blocks = pd.get("blocks", [])
        for b_idx, blk in enumerate(page_blocks):
            # collect text of all spans in block
            block_text = []
            if blk.get("type", 0) == 0:  # text block
                for line in blk.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    # normalize common control chars
                    line_text = line_text.strip()
                    if line_text:
                        block_text.append(line_text)
            else:
                # image or other block -> skip text extraction
                continue

            block_text_join = "\n".join(block_text).strip()
            if not block_text_join:
                continue
            full_text += block_text_join + "\n"

            # heuristics to detect table-like block:
            # - many commas or pipes
            # - multiple short lines (like cells)
            comma_count = block_text_join.count(",")
            pipe_count = block_text_join.count("|")
            short_lines = sum(1 for l in block_text if len(l.split()) <= 6)
            is_table = False
            if comma_count >= 3 or pipe_count >= 2 or short_lines >= max(3, len(block_text)//2):
                is_table = True

            bbox = blk.get("bbox", None)
            blocks_info.append({
                "page": pno,
                "bbox": bbox,
                "text": block_text_join,
                "block_no": b_idx,
                "is_likely_table": is_table
            })
    try:
        doc.close()
    except Exception:
        pass
    return full_text, blocks_info


def handle_personal_details_query(query_text: str, grouped: dict) -> bool:
    q = (query_text or "").lower()
    contact_keywords = (
        "personal details", "personal detail", "personal info", "contact details",
        "contact info", "contact information", "personal information",
        "email", "phone", "mobile", "linkedin", "contact"
    )
    if not any(k in q for k in contact_keywords):
        return False

    if not grouped:
        print("No candidates found to show contact details for.")
        return True

    email_re = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-z]{2,}', re.I)
    phone_re = re.compile(r'(\+?\d[\d\s().\-]{7,}\d)', re.I)
    linkedin_re = re.compile(r'(linkedin\.com/[^\s,;]+)', re.I)

    printed = 0
    for cid, info in grouped.items():
        if printed >= 5:
            break
        name = info.get("candidate_name") or cid

        emails = sorted(info.get("emails") or [])
        phones = sorted(info.get("phones") or [])
        linkedin = sorted(info.get("linkedin") or [])

        if not emails or not phones or not linkedin:
            sec = info.get("sections") or {}
            if isinstance(sec, dict):
                for k, v in sec.items():
                    if not emails:
                        emails.extend(email_re.findall(v or ""))
                    if not phones:
                        phones.extend(phone_re.findall(v or ""))
                    if not linkedin:
                        linkedin.extend(linkedin_re.findall(v or ""))
            for ctx in info.get("contexts", [])[:10]:
                if not emails:
                    emails.extend(email_re.findall(ctx or ""))
                if not phones:
                    phones.extend(phone_re.findall(ctx or ""))
                if not linkedin:
                    linkedin.extend(linkedin_re.findall(ctx or ""))
                if emails and phones and linkedin:
                    break

        emails = sorted(set(emails))
        phones = sorted({re.sub(r'\s+', ' ', p).strip() for p in phones})
        linkedin = sorted(set(linkedin))

        emails_str = ", ".join(emails) if emails else "Not found"
        phones_str = ", ".join(phones) if phones else "Not found"
        linkedin_str = ", ".join(linkedin) if linkedin else "Not found"
        locations = ", ".join(sorted(list(info.get("locations") or []))) if info.get("locations") else "Not found"

        print(f"\nCandidate: {name}")
        print(f"  Email   : {emails_str}")
        print(f"  Phone   : {phones_str}")
        print(f"  LinkedIn: {linkedin_str}")
        print(f"  Location: {locations}")

        ctxs = info.get("contexts", []) or []
        if ctxs:
            snippet = ctxs[0].replace("\n", " ").strip()
            if len(snippet) > 300:
                snippet = snippet[:300].rsplit(" ", 1)[0] + "..."
            print("  Summary :", snippet)

        printed += 1

    return True

# ------------------ main search flow (unchanged) ------------------
def search_resumes(query_text):
    print(f"\nAnalyzing query: '{query_text}'")

    try:
        points, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=2000)
        all_pts = normalize_qdrant_points(points)
    except Exception:
        all_pts = []

    if not all_pts:
        print("No data found in Qdrant. Run ingestion first.")
        return

    all_names = sorted({(p.payload.get("candidate_name") or "").strip().lower() for p in all_pts if p.payload.get("candidate_name")})
    all_locations = set()
    all_skills = set()
    for p in all_pts:
        for loc in (p.payload.get("locations") or []):
            try:
                all_locations.add(loc.lower())
            except Exception:
                pass
        for sk in (p.payload.get("skills") or []):
            try:
                all_skills.add(sk.lower())
            except Exception:
                pass

    section_target = map_query_to_section(query_text)
    if section_target:
        print(f"Detected section intent: {section_target}")
        candidate_name = extract_name_from_query(query_text, all_names)
        if candidate_name:
            candidate_points = [p for p in all_pts if (p.payload.get("candidate_name") or "").strip().lower() == candidate_name]
            grouped = aggregate_hits(candidate_points)
            if handle_personal_details_query(query_text, grouped):
                return
            for k, info in grouped.items():
                print(f"\nCandidate: {info.get('candidate_name')}")
                sec_text = ""
                sec_dict = info.get("sections") or {}
                if isinstance(sec_dict, dict):
                    sec_text = sec_dict.get(section_target, "")
                if sec_text:
                    print(f"{section_target.title()}:")
                    print(sec_text.strip()[:3000])
                else:
                    print(f"{section_target.title()}: Not found in candidate sections.")
                print("-" * 50)
            return

        section_points = qdrant_query_by_section(section_target)
        if not section_points:
            print("No direct section-tagged points found; performing semantic search + section post-filter.")
            try:
                qvec = embedding_model.encode(query_text).tolist()
                sem = qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=qvec, limit=40, with_payload=True)
                sem_norm = normalize_qdrant_points(sem)
            except Exception:
                sem_norm = []
            filtered = []
            for p in sem_norm:
                secobj = p.payload.get("sections") or {}
                sec_text = secobj.get(section_target, "") if isinstance(secobj, dict) else ""
                if sec_text and section_target in secobj:
                    filtered.append(p)
            section_points = filtered

        if not section_points:
            print(f"No resumes found with section '{section_target}'.")
            return

        grouped = aggregate_hits(section_points)
        for k, info in grouped.items():
            print(f"\nCandidate: {info.get('candidate_name')}")
            sec_text = ""
            sec_dict = info.get("sections") or {}
            if isinstance(sec_dict, dict):
                sec_text = sec_dict.get(section_target, "")
            if sec_text:
                print(f"{section_target.title()}:")
                print(sec_text.strip()[:3000])
            else:
                if section_target == "certifications" and info.get("certifications"):
                    print(", ".join(info.get("certifications")))
                else:
                    print("Not found.")
            print("-" * 50)
        return

    query_lower = query_text.lower()
    name_from_query = extract_name_from_query(query_text, all_names)
    if name_from_query:
        results = [p for p in all_pts if (p.payload.get("candidate_name") or "").strip().lower() == name_from_query]
        print(f"Resolved candidate name from query: {name_from_query}")
    else:
        q_loc = None
        doc_q = nlp(query_text) if nlp is not None else None
        if doc_q is not None:
            for ent in doc_q.ents:
                if ent.label_ == "GPE":
                    q_loc = ent.text.strip().lower()
                    break
        if not q_loc:
            for w in re.findall(r'\w+', query_text.lower()):
                m = get_close_matches(w, list(all_locations), n=1, cutoff=0.85)
                if m:
                    q_loc = m[0]; break

        if q_loc:
            results = [p for p in all_pts if q_loc in [loc.lower() for loc in (p.payload.get("locations") or [])]]
            print(f"Exact location match found: {q_loc}")
        else:
            matched_skills = [s for s in all_skills if s in query_lower]
            if matched_skills:
                results = [p for p in all_pts if any(s in [sk.lower() for sk in (p.payload.get("skills") or [])] for s in matched_skills)]
                print(f"Exact skill match found: {matched_skills}")
            else:
                try:
                    qvec = embedding_model.encode(query_text).tolist()
                    sem = qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=qvec, limit=10, with_payload=True)
                    results = normalize_qdrant_points(sem)
                except Exception:
                    results = []
                print("No exact match, using semantic search.")

    if not results:
        print("No relevant information found.")
        return

    grouped = aggregate_hits(results)

    if name_from_query:
        target = name_from_query.strip().lower()
        filtered = {}
        for cid, info in grouped.items():
            cand_name = (info.get("candidate_name") or "").strip().lower()
            if cand_name == target:
                filtered[cid] = info
        if filtered:
            grouped = filtered

    try:
        if handle_personal_details_query(query_text, grouped):
            return
    except Exception as e:
        print("Warning: personal-details handler error:", e)


    for key, info in grouped.items():
        print(f"\n📄 Files: {', '.join(info['filenames']) if info['filenames'] else 'Unknown'}")
        print(f"Candidate: {info.get('candidate_name')}")
        print(f"Emails: {', '.join(info['emails']) if info['emails'] else 'Not found'}")
        print(f"Phones: {', '.join(info['phones']) if info['phones'] else 'Not found'}")
        print(f"LinkedIn: {', '.join(info['linkedin']) if info['linkedin'] else 'Not found'}")
        print(f"Skills: {sorted(list(info['skills']))}")
        print(f"Locations: {sorted(list(info['locations']))}")
        print(f"Education: {sorted(list(info['education']))}")
        if info['contexts']:
            print("Context:", info['contexts'][0][:500].replace("\n", " ").strip(), "...")
        print("-" * 50)

# Ranking and other functions remain unchanged from your original file.
# If your original file contains compute_final_score_v2, semantic functions, rank_candidates_v2, integrate_ranking_and_print_v2 etc,
# make sure they remain appended below this point exactly as before, otherwise ranking will raise errors.

# ------------------ main entrypoint ------------------
if __name__ == "__main__":
    # load both spaCy models at startup
    try:
        nlp_ok, nlp_skill_ok = try_load_spacy_models()
        print(f"spaCy general model loaded: {nlp_ok}, skill model loaded: {nlp_skill_ok}")
    except Exception as e:
        print("Error loading spaCy models:", e)
        nlp = None
        nlp_skill = None

    SKIP_INGEST = os.getenv("SKIP_INGEST", "0") in ("1", "true", "True")

    if not os.path.exists(RESUME_FOLDER):
        os.makedirs(RESUME_FOLDER, exist_ok=True)
        print(f"Please add resumes to the '{RESUME_FOLDER}' directory (pdf/docx/txt) and re-run.")
    else:
        if True: # <-- CHANGE THIS LINE
            ingest_resumes()

    print("\n--- AI Resume Analyzer (section-aware) ---")
    print("Examples: 'what are certifications', 'what skills does dexter have', 'provide me pranay reddy personal details'")
    print("Type 'exit' to quit.")

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        search_resumes(q)
