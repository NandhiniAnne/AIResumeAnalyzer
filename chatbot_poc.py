import os
import fitz  # PyMuPDF
import docx  # python-docx
import uuid
import re
import spacy
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import defaultdict
from difflib import get_close_matches

# --- 1. CONFIGURATION ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "final_hybrid_chunk_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_trf"
RESUME_FOLDER = "resumes"

# --- 2. INITIALIZATION ---
print("Loading models and clients...")
nlp = spacy.load(SPACY_MODEL)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
print("Models and clients loaded.")

# --- 3. UTILITY FUNCTIONS ---

def extract_candidate_name(text):
    """Extract candidate name from top lines, fallback to SpaCy PERSON entity."""
    lines = text.strip().splitlines()[:3]
    for line in lines:
        line_clean = line.strip()
        words = line_clean.split()
        capitalized_words = [w for w in words if w[0].isupper()]
        if 1 < len(capitalized_words) <= 4:  # 2-4 capitalized words likely name
            return " ".join(capitalized_words)
    # fallback SpaCy
    doc_nlp = nlp(text)
    person_entities = [ent.text.strip() for ent in doc_nlp.ents if ent.label_ == "PERSON"]
    return max(person_entities, key=len) if person_entities else "Unknown"

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r"(\+?\d[\d\s().-]{7,}\d)", text)
    return match.group(0) if match else None

def extract_linkedin(text):
    match = re.search(r"(https?://)?(www\.)?linkedin\.com/[^\s]+", text, re.I)
    return match.group(0) if match else None

def extract_skills(text):
    # simple keyword extraction for demonstration
    skill_keywords = ["python","java","sql","spark","hadoop","azure","etl","data engineering","scala"]
    text_lower = text.lower()
    return [s for s in skill_keywords if s in text_lower]

def extract_locations(text):
    doc_nlp = nlp(text)
    locations = []
    current = []
    for ent in doc_nlp.ents:
        if ent.label_ == "GPE":
            current.append(ent.text.strip())
        else:
            if current:
                locations.append(" ".join(current))
                current = []
    if current:
        locations.append(" ".join(current))
    return list(set([loc.lower() for loc in locations])) or ["not found"]

# --- 4. PARSE RESUME FILES ---
def parse_resume_file(file_path):
    full_text = ""
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
    elif file_path.lower().endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            full_text += para.text + "\n"

    return {
        "full_text": full_text,
        "candidate_name": extract_candidate_name(full_text),
        "email": extract_email(full_text),
        "phone": extract_phone(full_text),
        "linkedin": extract_linkedin(full_text),
        "skills": extract_skills(full_text),
        "locations": extract_locations(full_text)
    }

# --- 5. INGEST RESUMES INTO QDRANT ---
def ingest_resumes():
    print(f"Starting resume ingestion from folder: '{RESUME_FOLDER}'")
    supported_formats = (".pdf", ".docx")
    resume_files = [f for f in os.listdir(RESUME_FOLDER) if f.lower().endswith(supported_formats)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    points_to_upload = []

    for filename in resume_files:
        path = os.path.join(RESUME_FOLDER, filename)
        print(f"Processing and chunking {filename}...")
        parsed = parse_resume_file(path)
        chunks = text_splitter.split_text(parsed["full_text"])

        for chunk in chunks:
            points_to_upload.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding_model.encode(chunk).tolist(),
                    payload={
                        "filename": filename,
                        "candidate_name": parsed["candidate_name"],
                        "email": parsed["email"],
                        "phone": parsed["phone"],
                        "linkedin": parsed["linkedin"],
                        "skills": parsed["skills"],
                        "locations": parsed["locations"],
                        "text": chunk,
                    },
                )
            )

    if points_to_upload:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points_to_upload, wait=True)

    print(f"Successfully ingested {len(points_to_upload)} chunks from {len(resume_files)} resumes.")

# --- 6. SEARCH RESUMES ---
def extract_query_location(query_text, known_locations=None):
    doc_nlp = nlp(query_text)
    for ent in doc_nlp.ents:
        if ent.label_ == "GPE":
            return ent.text.strip().lower()
    if known_locations:
        words = query_text.split()
        for word in words:
            match = get_close_matches(word.lower(), known_locations, n=1, cutoff=0.8)
            if match:
                return match[0]
    return None

def search_resumes(query_text):
    """
    Search resumes with optional exact name, location, or skill filter.
    Exact match takes priority over semantic search.
    """
    print(f"\nAnalyzing query: '{query_text}'")

    # --- 1. Scroll all points from Qdrant ---
    points, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=1000)

    # --- 2. Prepare metadata lists ---
    all_names = [p.payload.get("candidate_name", "").lower() for p in points]
    all_locations = set()
    all_skills = set()
    for p in points:
        all_locations.update([loc.lower() for loc in p.payload.get("locations", [])])
        all_skills.update([s.lower() for s in p.payload.get("skills", [])])

    # --- 3. Check if query matches a name exactly ---
    query_lower = query_text.lower()
    exact_name_matches = [p for p in points if p.payload.get("candidate_name", "").lower() == query_lower]

    if exact_name_matches:
        results = exact_name_matches
        print("Exact candidate name match found.")
    else:
        # --- 4. Check for exact location match ---
        query_location = extract_query_location(query_text, known_locations=all_locations)
        if query_location:
            results = [p for p in points if query_location in [loc.lower() for loc in p.payload.get("locations", [])]]
            print(f"Exact location match found: {query_location}")
        else:
            # --- 5. Check for exact skill match ---
            matched_skills = [s for s in all_skills if s in query_lower]
            if matched_skills:
                results = [p for p in points if any(s in [skill.lower() for skill in p.payload.get("skills", [])] for s in matched_skills)]
                print(f"Exact skill match found: {matched_skills}")
            else:
                # --- 6. Fallback to semantic search ---
                query_vector = embedding_model.encode(query_text).tolist()
                search_results = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=10,
                    with_payload=True
                )
                results = search_results
                print("No exact match, using semantic search.")

    # --- 7. Display results ---
    if not results:
        print("No relevant information found matching the criteria.")
        return

    for p in results:
        payload = p.payload
        print(f"\n📄 {payload.get('filename', 'Unknown')}")
        print(f"Candidate: {payload.get('candidate_name', 'Unknown')}")
        print(f"Email: {payload.get('email', 'Not found')}")
        print(f"Phone: {payload.get('phone', 'Not found')}")
        print(f"LinkedIn: {payload.get('linkedin', 'Not found')}")
        print(f"Skills: {payload.get('skills', [])}")
        print(f"Locations: {payload.get('locations', [])}")
        print(f"Context: {payload.get('text', '')[:300]}...")
        print("-" * 50)

# --- 7. MAIN EXECUTION ---
if __name__ == "__main__":
    ingest_resumes()
    print("\n--- AI Resume Analyzer Chatbot ---")
    print("Ask me anything about the resumes. Type 'exit' to quit.")

    while True:
        user_query = input("\n> ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not user_query:
            continue
        search_resumes(user_query)


