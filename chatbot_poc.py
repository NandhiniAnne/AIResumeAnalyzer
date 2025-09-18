import numpy as _np
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
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_trf")
RESUME_FOLDER = os.getenv("RESUME_FOLDER", "resumes")
SKILLS_FILE = os.getenv("SKILLS_FILE", "skills.txt")  # optional

# -------- INIT --------
print("Loading models and qdrant client...")
nlp = spacy.load(SPACY_MODEL)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
print("Ready.")
# Heading mapping and heuristics
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

# small extractors
def extract_skills_from_section(text):
    if not text: return []
    lines = [ln.strip(" •\t-") for ln in text.splitlines() if ln.strip()]
    skills = set()
    for ln in lines:
        if ',' in ln and len(ln) < 250:
            parts = [p.strip().lower() for p in ln.split(',') if p.strip()]
            skills.update(parts)
            continue
        if '|' in ln or '/' in ln or '·' in ln:
            parts = re.split(r'[|/·]', ln)
            skills.update([p.strip().lower() for p in parts if p.strip()])
            continue
        tokens = re.findall(r'[A-Za-z+#\.\-]{2,}', ln)
        if tokens:
            skills.update([t.lower() for t in tokens if t.lower() not in ("and","with","experience","years","proficient")])
    cleaned = sorted({s for s in skills if len(s) > 1})
    return cleaned

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
    """
    Try to parse experience bullets into list of jobs (title, company, dates, summary).
    Robust against unicode dashes and empty lines.
    """
    if not text:
        return []
    jobs = []
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    i = 0
    # precompile regexes
    year_re = re.compile(r'\b\d{4}\b')
    # include common dash characters and escape hyphen safely
    separator_re = re.compile(r'[—–\-\@\|,()]')
    date_token_re = re.compile(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[^\d]*\d{4}\b|\b\d{4}\b|\d{2}/\d{4})', re.I)

    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue

        # heuristics: line has a year OR contains separator characters OR looks like a short title/company line
        looks_like_entry = bool(year_re.search(ln)) or bool(separator_re.search(ln)) or (len(ln.split()) <= 8 and ln[0].isupper())

        if looks_like_entry:
            dates = date_token_re.findall(ln)
            # split by common separators to get possible title/company parts
            parts = re.split(r'[-–—@,|()]+', ln)
            title_candidate = parts[0].strip() if parts else ln
            company_candidate = parts[1].strip() if len(parts) > 1 else ""
            # collect following bullet lines as summary until next likely header-like line
            j = i + 1
            summary_lines = []
            while j < len(lines):
                nxt = lines[j].strip()
                # stop if next line looks like a new entry (year/separator/short all-caps)
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

    # fallback: if nothing detected, return the whole section as a single experience entry
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

# ------------------ Candidate & parsing helpers ------------------

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
    # spaCy fallback
    snippet = "\n".join(lines[:30])
    doc = nlp(snippet)
    persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
    for p in sorted(persons, key=lambda s: -len(s)):
        if not skip_kw.search(p):
            return normalize_name(p)
    # filename fallback
    if filename:
        base = os.path.splitext(os.path.basename(filename))[0]
        base_clean = re.sub(r'[_\-\.]+', ' ', base)
        base_clean = re.sub(r'(?i)\b(resume|cv|final|de|profile)\b', '', base_clean)
        if base_clean.strip():
            return normalize_name(base_clean)
    return "Unknown"

# ------------------ Parse resume (full flow) ------------------

def parse_resume_file(file_path):
    full_text = ""
    if file_path.lower().endswith(".pdf"):
        try:
            doc = fitz.open(file_path)
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
    elif file_path.lower().endswith(".docx"):
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                full_text += para.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
        except Exception:
            full_text = ""

    # sections parsing
    raw_sections = split_into_sections(full_text)
    sections_canonical = {}
    for header, body in raw_sections:
        key = canonicalize_heading(header)
        if key in sections_canonical:
            sections_canonical[key] += "\n\n" + body
        else:
            sections_canonical[key] = body

    skills = extract_skills_from_section(sections_canonical.get("skills", ""))
    education_info = extract_education_from_section(sections_canonical.get("education", ""))
    experience = extract_experience_from_section(sections_canonical.get("experience", ""))
    certifications = extract_certifications_from_section(sections_canonical.get("certifications", ""))
    contact = extract_contact_from_section(sections_canonical.get("contact", full_text))
    summary = sections_canonical.get("summary", "")

    candidate_name = extract_candidate_name_from_text(full_text, filename=file_path)
    email = contact.get("email")
    candidate_id = (email or candidate_name or os.path.splitext(os.path.basename(file_path))[0]).lower()

    return {
        "full_text": full_text,
        "candidate_name": candidate_name,
        "candidate_id": candidate_id,
        "email": email,
        "phone": contact.get("phone"),
        "linkedin": contact.get("linkedin"),
        "skills": skills,
        "locations": extract_locations_from_text(full_text := full_text),  # helper defined below
        "education": education_info.get("schools", []),
        "education_degrees": education_info.get("degrees", []),
        "experience": experience,
        "certifications": certifications,
        "sections": sections_canonical,
    }

# ------------------ small helper to collect GPEs ------------------

def extract_locations_from_text(text):
    doc = nlp(text)
    locs = set()
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            locs.add(ent.text.strip())
    return [l.lower() for l in sorted(locs)] if locs else []

# ------------------ Qdrant ingestion ------------------

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

# ------------------ aggregation & helpers ------------------
def aggregate_hits(points):
    """
    Group point-level payloads into candidate-level records.

    Returns a dict: candidate_id -> aggregated info dict
    """
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
                # avoid duplicates (simple object equality)
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

# convert Qdrant response dict->point-like object
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
            # dict with 'id' and 'payload' or nested structure
            pid = p.get("id") or p.get("point_id") or p.get("point", {}).get("id")
            payload = p.get("payload") or p.get("point", {}).get("payload") or {}
            normalized.append(SimplePoint(pid, payload))
        else:
            # unknown format — try to access attributes
            try:
                normalized.append(SimplePoint(p.id, p.payload))
            except Exception:
                continue
    return normalized

# ------------------ query helpers (name detection, section mapping) ------------------

def extract_name_from_query(query_text, all_known_names):
    doc = nlp(query_text)
    persons = [ent.text.strip().lower() for ent in doc.ents if ent.label_ == "PERSON"]
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
    doc = nlp(query_text)
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in SECTION_KEYWORDS:
            return SECTION_KEYWORDS[lemma]
    return None

# Qdrant query-by-section
def qdrant_query_by_section(section_key, text_filter=None, limit=500):
    """
    Fetch points where payload.sections contains section_key (client-side filtering fallback supported).
    """
    results = []
    # try server-side filter if possible
    try:
        fv = FieldCondition(key=f"sections.{section_key}", match=MatchValue(value=text_filter.lower() if text_filter else ""))
        filt = Filter(must=[fv]) if text_filter else None
        # use query_points if available
        try:
            resp = qdrant_client.query_points(collection_name=COLLECTION_NAME, query_filter=filt, limit=limit, with_payload=True, with_vector=False)
            resp_points = resp.get("result", resp.get("points", []))
            results = normalize_qdrant_points(resp_points)
        except Exception:
            # fallback to scroll and filter
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
        # last-resort: scroll and client-side filter
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
def handle_personal_details_query(query_text: str, grouped: dict) -> bool:
    """
    If the query clearly asks for personal/contact details, print contact fields
    (email, phone, linkedin, locations) for the candidate(s) in `grouped`.
    If contact fields are missing from grouped payload, scan 'sections' and 'contexts'
    for email/phone/linkedin using regexes as a fallback.
    Returns True if the handler printed something (so caller can return/skip ranking).
    """
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

    # regexes for fallback scanning
    email_re = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-z]{2,}', re.I)
    phone_re = re.compile(r'(\+?\d[\d\s().\-]{7,}\d)', re.I)
    linkedin_re = re.compile(r'(linkedin\.com/[^\s,;]+)', re.I)

    printed = 0
    for cid, info in grouped.items():
        if printed >= 5:
            break
        name = info.get("candidate_name") or cid

        # Primary fields from aggregated payload
        emails = sorted(info.get("emails") or [])
        phones = sorted(info.get("phones") or [])
        linkedin = sorted(info.get("linkedin") or [])

        # Fallback: scan sections and contexts if some fields missing
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

        # dedupe + cleanup
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
# ------------------ main search flow ------------------
def search_resumes(query_text):
    print(f"\nAnalyzing query: '{query_text}'")

    # load all points (used for building known name/location/skill lists)
    try:
        points, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=2000)
        all_pts = normalize_qdrant_points(points)
    except Exception:
        all_pts = []

    if not all_pts:
        print("No data found in Qdrant. Run ingestion first.")
        return

    # build quick lookup sets
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

    # 0) section-intent detection (e.g., "what are certifications")
    section_target = map_query_to_section(query_text)
    if section_target:
        print(f"Detected section intent: {section_target}")
        # check if query also contains a name
        candidate_name = extract_name_from_query(query_text, all_names)
        if candidate_name:
            candidate_points = [p for p in all_pts if (p.payload.get("candidate_name") or "").strip().lower() == candidate_name]
            grouped = aggregate_hits(candidate_points)
            # allow personal details handler for name-scoped section requests
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

        # else search the section across all resumes (try server-side then semantic fallback)
        section_points = qdrant_query_by_section(section_target)
        if not section_points:
            # semantic fallback and post-filter by section contents
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

    # If not a section-targeted query, proceed with name/location/skill/semantic priority:
    query_lower = query_text.lower()
    name_from_query = extract_name_from_query(query_text, all_names)
    if name_from_query:
        # explicit name -> restrict results to that candidate's points
        results = [p for p in all_pts if (p.payload.get("candidate_name") or "").strip().lower() == name_from_query]
        print(f"Resolved candidate name from query: {name_from_query}")
    else:
        # try location detection via spaCy GPE or fuzzy match
        q_loc = None
        doc_q = nlp(query_text)
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
            # exact skill matching
            matched_skills = [s for s in all_skills if s in query_lower]
            if matched_skills:
                results = [p for p in all_pts if any(s in [sk.lower() for sk in (p.payload.get("skills") or [])] for s in matched_skills)]
                print(f"Exact skill match found: {matched_skills}")
            else:
                # semantic fallback
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

    # aggregate chunk-level points into candidate-level grouped dict
    grouped = aggregate_hits(results)

    # If user named a candidate, narrow grouped to that candidate (so handlers operate on the intended person)
    if name_from_query:
        target = name_from_query.strip().lower()
        filtered = {}
        for cid, info in grouped.items():
            cand_name = (info.get("candidate_name") or "").strip().lower()
            if cand_name == target:
                filtered[cid] = info
        if filtered:
            grouped = filtered

    # personal/contact details handler: if query asks for contact info, handle and return
    try:
        if handle_personal_details_query(query_text, grouped):
            return
    except Exception as e:
        # don't fail hard on handler error; continue with ranking fallback
        print("Warning: personal-details handler error:", e)

    # attempt to rank candidates intelligently if query contains skills/years/seniority
    try:
        did_rank = integrate_ranking_and_print_v2(grouped, query_text, all_skills)
        if did_rank:
            return
    except Exception as e:
        print("Ranking error:", e)

    # fallback: print aggregated candidate info (no ranking requested)
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



# ------------------ Improved deterministic ranking (offline) ------------------

import datetime, json


# Configurable weights
RANKING_WEIGHTS = {
    "primary": 0.40,
    "secondary": 0.10,
    "seniority": 0.15,
    "experience": 0.15,
    "soft": 0.05,
    "quality": 0.05,
    "semantic": 0.10
}

# Skill synonyms loader (expects skills.txt lines like: python:py,python3)
_SKILL_SYNONYMS = {}
if os.path.exists(SKILLS_FILE):
    try:
        with open(SKILLS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.strip().startswith("#"):
                    continue
                parts = line.strip().split(":")
                if len(parts) == 2:
                    canon = parts[0].strip().lower()
                    aliases = [a.strip().lower() for a in parts[1].split(",")]
                    _SKILL_SYNONYMS[canon] = aliases
    except Exception as e:
        print("Warning: could not load skills file:", e)

def normalize_skill(s: str) -> str:
    s = s.lower().strip()
    for canon, aliases in _SKILL_SYNONYMS.items():
        if s == canon or s in aliases:
            return canon
    return s

def skill_present(candidate: Dict, skill: str) -> bool:
    sk = normalize_skill(skill)
    allskills = [(normalize_skill(s) if s else "") for s in (candidate.get("skills_primary") or []) + (candidate.get("skills_secondary") or [])]
    if sk in allskills:
        return True
    text_fields = " ".join([candidate.get("summary","") or ""] + (candidate.get("contexts") or [])).lower()
    if re.search(r'\b' + re.escape(sk) + r'\b', text_fields):
        return True
    if sk in text_fields:
        return True
    return False

# Improved experience parser
_MONTH_MAP = {m.lower(): i+1 for i,m in enumerate(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec"]
)}
def parse_date_token(tok: str):
    try:
        low = tok.lower()
        m = re.match(r'(\\w+)\\s+(\\d{4})', low)
        if m:
            mon = _MONTH_MAP.get(m.group(1)[:3], 1)
            return datetime.date(int(m.group(2)), mon, 1)
        if re.match(r'\\d{4}', low):
            return datetime.date(int(low),1,1)
        if re.match(r'\\d{2}/\\d{4}', low):
            parts = low.split("/")
            return datetime.date(int(parts[1]), int(parts[0]), 1)
    except Exception:
        return None
    return None

def compute_experience_years(experience_entries: list) -> float:
    total_months = 0
    for job in experience_entries or []:
        dates = job.get("dates") or []
        if len(dates) >= 2:
            d1 = parse_date_token(dates[0])
            d2 = parse_date_token(dates[-1])
            if d1 and d2:
                months = (d2.year - d1.year)*12 + (d2.month - d1.month)
                if months > 0:
                    total_months += months
    return round(total_months/12.0,1) if total_months>0 else 0.0

# Chunk-level semantic similarity
def aggregate_chunk_similarities(contexts: list, query_text: str) -> float:
    if not contexts:
        return 0.0
    try:
        qv = _np.array(embedding_model.encode(query_text), dtype=_np.float32)
        sims = []
        for ctx in contexts[:10]:
            sv = _np.array(embedding_model.encode(ctx), dtype=_np.float32)
            denom = (_np.linalg.norm(qv) * _np.linalg.norm(sv))
            if denom>0:
                sims.append(float(_np.dot(qv, sv)/denom))
        if not sims:
            return 0.0
        return 0.7*max(sims) + 0.3*(_np.mean(sorted(sims,reverse=True)[:3]))
    except Exception:
        return 0.0
def compute_skill_fraction(candidate: Dict, req_skills: List[str]) -> float:
    """Return fraction of required skills present in candidate's skills lists."""
    if not req_skills:
        return 0.0
    # collect candidate skills normalized
    skills = [s.lower() for s in (candidate.get("skills_primary") or []) + (candidate.get("skills_secondary") or [])]
    if not skills:
        # fallback: search candidate summary/contexts for skill tokens
        text_fields = " ".join([candidate.get("summary","") or ""] + (candidate.get("contexts") or [])).lower()
        matched = 0
        for rs in req_skills:
            if rs.lower() in text_fields:
                matched += 1
        return matched / len(req_skills) if matched else 0.0

    matched = 0
    for rs in req_skills:
        rs_low = rs.lower()
        # exact or contained match
        if any(rs_low == s or rs_low in s or s in rs_low for s in skills):
            matched += 1
    return matched / len(req_skills)

# Override compute_final_score to include improved exp + chunk sims
def compute_final_score(candidate: Dict,
                        req_primary: List[str],
                        req_secondary: List[str],
                        req_soft: List[str],
                        req_min_years: float,
                        req_seniority: str,
                        query_text: str,
                        weights: Dict[str, float] = None) -> Tuple[float, Dict]:
    w = weights or RANKING_WEIGHTS

    primary_score = compute_skill_fraction(candidate, req_primary)
    secondary_score = compute_skill_fraction(candidate, req_secondary)
    soft_score = compute_skill_fraction(candidate, req_soft)

    rel_exp = candidate.get("relevant_experience_years", 0.0)
    experience_score = clamp01(rel_exp / req_min_years) if req_min_years > 0 else clamp01(rel_exp/5.0)

    seniority_score = seniority_match(candidate, req_seniority)
    quality_score = clamp01(candidate.get("resume_quality_score", 0.5))

    sem_score_chunks = aggregate_chunk_similarities(candidate.get("contexts", []), query_text)
    semantic_score = sem_score_chunks

    combined = (
        w["primary"] * primary_score +
        w["secondary"] * secondary_score +
        w["seniority"] * seniority_score +
        w["experience"] * experience_score +
        w["soft"] * soft_score +
        w["quality"] * quality_score +
        w["semantic"] * semantic_score
    )

    final_score = clamp01(combined) * 100.0

    breakdown = {
        "primary": primary_score,
        "secondary": secondary_score,
        "seniority": seniority_score,
        "experience": experience_score,
        "soft": soft_score,
        "quality": quality_score,
        "semantic": semantic_score,
        "final": final_score
    }
    return final_score, breakdown
def seniority_match(candidate: Dict, req_seniority: str) -> float:
    """Return a score [0,1] for how well candidate's seniority matches the requested one."""
    if not req_seniority:
        return 0.5  # neutral if not requested
    cand_sen = (candidate.get("seniority_estimate") or "").lower()
    req = req_seniority.lower()
    if not cand_sen:
        return 0.5
    if cand_sen == req:
        return 1.0
    # related groupings: manager > senior > mid > junior
    groups = {
        "manager": ["manager", "director", "head", "vp", "vice president"],
        "senior": ["senior", "sr", "lead", "principal", "staff"],
        "mid": ["mid", "software engineer", "engineer"],
        "junior": ["junior", "jr", "associate", "intern"]
    }
    # exact-match fallback otherwise approximate
    if req in groups:
        if any(tok in cand_sen for tok in groups[req]):
            return 0.8
    # partial matches
    if req in cand_sen or cand_sen in req:
        return 0.8
    return 0.0
def infer_seniority_from_text(text: str) -> str:
    """Infer a seniority level from free-text (simple heuristic)."""
    if not text:
        return ""
    low = text.lower()
    if any(k in low for k in ("manager", "director", "head of", "vp ", "vice president")):
        return "manager"
    if any(k in low for k in ("senior", "sr.", "sr ", "lead", "principal", "staff", "distinguished")):
        return "senior"
    if any(k in low for k in ("junior", "jr.", "jr ", "entry", "associate", "intern")):
        return "junior"
    return "mid"

# Enhance build_candidate_records_from_grouped to compute exp and carry contexts
def build_candidate_records_from_grouped(grouped: Dict) -> List[Dict]:
    candidates = []
    for cid, info in grouped.items():
        skills = sorted(list(info.get("skills") or []))
        summary = ""
        if info.get("sections") and isinstance(info.get("sections"), dict):
            summary = info["sections"].get("summary", "") or (" ".join(info.get("contexts")[:2]) if info.get("contexts") else "")
        exp_entries = info.get("experience", [])
        years = compute_experience_years(exp_entries)
        senior = infer_seniority_from_text(summary + " " + (info["sections"].get("experience","") if info.get("sections") else ""))
        quality = 0.5
        if info.get("emails") or info.get("phones"):
            quality += 0.25
        if len(skills) >= 3:
            quality += 0.25
        cand = {
            "candidate_id": cid,
            "name": info.get("candidate_name"),
            "skills_primary": skills,
            "skills_secondary": [],
            "summary": summary,
            "seniority_estimate": senior,
            "relevant_experience_years": years,
            "resume_quality_score": quality,
            "contexts": info.get("contexts", [])
        }
        candidates.append(cand)
    return candidates

# Improved integrate_ranking_and_print to show evidence
def integrate_ranking_and_print(grouped: Dict, query_text: str, all_skills: set):
    candidates = build_candidate_records_from_grouped(grouped)
    req_primary = []
    try:
        qdoc = nlp(query_text)
        candidates_sk = set()
        for nc in list(qdoc.noun_chunks) + [ent for ent in qdoc.ents]:
            tok = normalize_token(nc.text)
            if len(tok.split()) <= 5 and len(tok) > 1:
                candidates_sk.add(tok)
        for w in re.findall(r'\\w+', query_text.lower()):
            if len(w) > 1:
                candidates_sk.add(w)
        for cand in sorted(candidates_sk, key=lambda s: -len(s)):
            match = get_close_matches(cand, list(all_skills), n=1, cutoff=0.6)
            if match and match[0] not in req_primary:
                req_primary.append(match[0])
    except Exception:
        req_primary = [s for s in all_skills if s in query_text.lower()]
    min_years = 0
    m = re.search(r'(\\d+)\\s*\\+?\\s*(?:years|yrs)', query_text.lower())
    if m:
        min_years = int(m.group(1))
    req_sen = ""
    for s in ("senior","junior","mid","lead","manager"):
        if s in query_text.lower():
            req_sen = s; break

    ranked = rank_candidates(candidates, req_primary, [], [], min_years, req_sen, query_text, top_k=20)
    if not ranked:
        print("No candidates to rank.")
        return False
    print("\\nRanked candidates (top results):")
    for r in ranked:
        print(f"{r.get('name','(no-name)')} - Score: {r.get('_score',0):.1f}")
        bd = r.get("_breakdown",{})
        print(f"  Breakdown: primary={bd.get('primary',0):.2f}, exp={bd.get('experience',0):.2f}, seniority={bd.get('seniority',0):.2f}, semantic={bd.get('semantic',0):.2f}")
        print(f"  Skills: {r.get('skills_primary')}")
        # show one evidence snippet
        if r.get("contexts"):
            snippet = r["contexts"][0][:250].replace("\\n"," ")
            print(f"  Evidence: {snippet}...")
        print("-"*50)
    return True

# ------------------ end improved ranking ------------------




# ------------------ Enhanced resume ranking module (v2) ------------------
from typing import List, Dict, Tuple
import math, os, json, re
import numpy as _np

RANKING_WEIGHTS_V2 = {
    "primary": 0.40,
    "secondary": 0.10,
    "seniority": 0.15,
    "experience": 0.20,
    "soft": 0.03,
    "quality": 0.04,
    "semantic": 0.08
}

EMB_CACHE_PATH = ".embedding_cache_v2.json"

def _load_json(path):
    try:
        if os.path.exists(path):
            with open(path,"r",encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}

def _save_json(path, obj):
    try:
        with open(path,"w",encoding="utf-8") as f:
            json.dump(obj, f)
    except Exception:
        pass

_emb_cache_v2 = _load_json(EMB_CACHE_PATH)

def semantic_similarity_v2(candidate: Dict, query_text: str, query_vec=None) -> float:
    try:
        if query_vec is None:
            qvec = embedding_model.encode(query_text)
        else:
            qvec = _np.array(query_vec, dtype=_np.float32)
        cid = candidate.get("candidate_id") or candidate.get("name") or ""
        if cid in _emb_cache_v2:
            cand_vec = _np.array(_emb_cache_v2[cid], dtype=_np.float32)
        else:
            texts = []
            if candidate.get("summary"):
                texts.append(candidate.get("summary"))
            if candidate.get("contexts"):
                texts.extend(candidate.get("contexts")[:6])
            if not texts:
                return 0.0
            vecs = []
            for t in texts:
                try:
                    v = embedding_model.encode(t)
                    vecs.append(_np.array(v, dtype=_np.float32))
                except Exception:
                    continue
            if not vecs:
                return 0.0
            cand_vec = _np.mean(_np.stack(vecs, axis=0), axis=0)
            _emb_cache_v2[cid] = cand_vec.tolist()
            _save_json(EMB_CACHE_PATH, _emb_cache_v2)
        denom = (_np.linalg.norm(cand_vec) * _np.linalg.norm(_np.array(qvec, dtype=_np.float32)))
        if denom == 0:
            return 0.0
        sim = float(_np.dot(cand_vec, _np.array(qvec, dtype=_np.float32)) / denom)
        return clamp01((sim + 1.0) / 2.0)
    except Exception:
        # fallback
        q_tokens = [t.lower() for t in re.findall(r'\w+', query_text) if len(t)>2]
        if not q_tokens:
            return 0.0
        matched = 0
        summary = candidate.get("summary","").lower()
        for t in q_tokens:
            if t in summary:
                matched += 1
        return clamp01(matched / len(q_tokens))

def compute_final_score_v2(candidate: Dict,
                        req_primary: List[str],
                        req_secondary: List[str],
                        req_soft: List[str],
                        req_min_years: float,
                        req_seniority: str,
                        query_text: str,
                        weights: Dict[str, float] = None,
                        query_vec=None) -> Tuple[float, Dict]:
    w = weights or RANKING_WEIGHTS_V2

    primary_score = compute_skill_fraction(candidate, req_primary)
    secondary_score = compute_skill_fraction(candidate, req_secondary)
    soft_score = compute_skill_fraction(candidate, req_soft)

    rel_exp = candidate.get("relevant_experience_years", 0.0)
    experience_score = clamp01(rel_exp / req_min_years) if req_min_years > 0 else 1.0

    seniority_score = 1.0 if req_seniority and req_seniority == candidate.get("seniority_estimate") else (0.6 if candidate.get("seniority_estimate") else 0.5)
    quality_score = clamp01(candidate.get("resume_quality_score", 0.5))
    semantic_score = semantic_similarity_v2(candidate, query_text, query_vec=query_vec)

    combined = (
        w["primary"] * primary_score +
        w["secondary"] * secondary_score +
        w["seniority"] * seniority_score +
        w["experience"] * experience_score +
        w["soft"] * soft_score +
        w["quality"] * quality_score +
        w["semantic"] * semantic_score
    )

    final_score = clamp01(combined) * 100.0
    breakdown = {
        "primary": primary_score, "secondary": secondary_score, "seniority": seniority_score,
        "experience": experience_score, "soft": soft_score, "quality": quality_score, "semantic": semantic_score,
        "final": final_score
    }
    return final_score, breakdown

def rank_candidates_v2(candidates: List[Dict],
                    req_primary: List[str],
                    req_secondary: List[str],
                    req_soft: List[str],
                    req_min_years: float,
                    req_seniority: str,
                    query_text: str,
                    top_k: int = 20) -> List[Dict]:
    scored = []
    qvec = None
    try:
        qvec = embedding_model.encode(query_text)
    except Exception:
        qvec = None
    for c in candidates:
        sc, breakdown = compute_final_score_v2(c, req_primary, req_secondary, req_soft, req_min_years, req_seniority, query_text, query_vec=qvec)
        entry = c.copy()
        entry["_score"] = sc
        entry["_breakdown"] = breakdown
        scored.append(entry)
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_k]

def integrate_ranking_and_print_v2(grouped: Dict, query_text: str, all_skills: set):
    # build candidate records using existing builder
    candidates = build_candidate_records_from_grouped(grouped)
    # extract requested skills similar to previous approach
    req_primary = []
    try:
        qdoc = nlp(query_text)
        candidates_set = set()
        for nc in list(qdoc.noun_chunks) + [ent for ent in qdoc.ents]:
            tok = normalize_token(nc.text)
            if len(tok.split()) <= 5 and len(tok) > 1:
                candidates_set.add(tok)
        for w in re.findall(r'\w+', query_text.lower()):
            if len(w) > 1:
                candidates_set.add(w)
        for cand in sorted(candidates_set, key=lambda s: -len(s)):
            match = get_close_matches(cand, list(all_skills), n=1, cutoff=0.6)
            if match and match[0] not in req_primary:
                req_primary.append(match[0])
    except Exception:
        req_primary = [s for s in all_skills if s in query_text.lower()]

    m = re.search(r'(\d+)\s*\+?\s*(?:years|yrs)', query_text.lower())
    min_years = int(m.group(1)) if m else 0
    req_sen = ""
    for s in ("senior","junior","mid","lead","manager"):
        if s in query_text.lower():
            req_sen = s; break

    if not (req_primary or min_years or req_sen):
        return False

    ranked = rank_candidates_v2(candidates, req_primary, [], [], min_years, req_sen, query_text, top_k=20)
    if not ranked:
        print("No candidates to rank.")
        return True

    print("\nRanked candidates (v2) (top results):")
    for r in ranked:
        print(f"{r.get('name','(no-name)')} - Score: {r.get('_score',0):.1f}")
        bd = r.get("_breakdown",{})
        print(f"  Breakdown: primary={bd.get('primary',0):.2f}, experience={bd.get('experience',0):.2f}, seniority={bd.get('seniority',0):.2f}, semantic={bd.get('semantic',0):.2f}")
        print(f"  Years: {r.get('relevant_experience_years')} | Seniority: {r.get('seniority_estimate')} | Quality: {r.get('resume_quality_score')}")
        if r.get('skills_primary'):
            print(f"  Skills: {r.get('skills_primary')}")
        if r.get('evidence'):
            print("  Evidence:")
            for ev in r.get('evidence',[])[:3]:
                print("   -", ev.strip()[:240].replace('\\n',' '))
        print("-"*40)
    return True
if __name__ == "__main__":
    SKIP_INGEST = os.getenv("SKIP_INGEST", "0") in ("1", "true", "True")

    # ensure resume folder exists
    if not os.path.exists(RESUME_FOLDER):
        os.makedirs(RESUME_FOLDER, exist_ok=True)
        print(f"Please add resumes to the '{RESUME_FOLDER}' directory (pdf/docx/txt) and re-run.")
    else:
        if not SKIP_INGEST:
            ingest_resumes()
        else:
            print("Skipping ingestion (SKIP_INGEST set). Using existing Qdrant data.")

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









