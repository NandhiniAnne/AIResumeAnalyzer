import os
import re
import fitz 
import spacy
import hashlib
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
# CONFIGURATION 
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "final_resume_collection"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
SPACY_MODEL = 'en_core_web_sm'
RESUME_FOLDER = 'resumes'
# INITIALIZATION
print("Loading models and clients...")
nlp = spacy.load(SPACY_MODEL)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
print("Models and clients loaded.")
# CORE FUNCTIONS
def parse_resume_pdf(pdf_path):
    """
    Final, corrected parser. Uses layout-aware text extraction and
    includes regex for finding email and phone numbers.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))
        for b in blocks:
            full_text += b[4]
    doc.close()
    sections = {
        'full_text': full_text,
        'candidate_name': "Unknown",
        'email': "Not Found",
        'phone': "Not Found"
    }
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\(?\b[2-9][0-9]{2}\)?[-. ]?\b[2-9][0-9]{2}[-. ]?\b[0-9]{4}\b'
    
    email_match = re.search(email_pattern, full_text)
    if email_match:
        sections['email'] = email_match.group(0)

    phone_match = re.search(phone_pattern, full_text)
    if phone_match:
        sections['phone'] = phone_match.group(0)

    #Section Parsing Logic 
    section_headings = [
        'OBJECTIVE', 'EXPERIENCE', 'EDUCATION', 'SKILLS', 'PROJECTS',
        'CERTIFICATIONS', 'CERTIFICATES', 'AWARDS', 'PUBLICATIONS', 
        'ADDITIONAL INFORMATION', 'EXTRACURRICULAR ACTIVITIES', 'WORK EXPERIENCE'
    ]
    
    pattern = r'^\s*(' + '|'.join(section_headings) + r')\s*$'
    lines = full_text.split('\n')
    
    if sections['candidate_name'] == "Unknown":
        for line in lines[:5]:
            clean_line = line.strip()
            if "@" not in clean_line and "linkedin.com" not in clean_line and not any(char.isdigit() for char in clean_line):
                if 4 < len(clean_line) < 30:
                    sections['candidate_name'] = clean_line
                    break
    
    current_section_name = "HEADER"
    sections[current_section_name] = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        match = re.match(pattern, line_stripped, re.IGNORECASE)
        if match and len(line_stripped) < 30:
            current_section_name = match.group(1).upper().replace("WORK EXPERIENCE", "EXPERIENCE")
            sections[current_section_name] = []
        else:
            cleaned_line = re.sub(r'^\s*[\•\\-\*]\s*', '', line_stripped)
            if cleaned_line:
                if current_section_name not in sections:
                    sections[current_section_name] = []
                sections[current_section_name].append(cleaned_line)
    return sections
def setup_qdrant_collection():
    """Sets up the collection with multiple named vectors."""
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "skills": models.VectorParams(size=384, distance=models.Distance.COSINE),
            "experience": models.VectorParams(size=384, distance=models.Distance.COSINE),
        }
    )

def ingest_resumes():
    print(f"Starting resume ingestion from folder: '{RESUME_FOLDER}'")
    resume_files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith('.pdf')]
    points_to_upload = []
    
    for filename in resume_files:
        path = os.path.join(RESUME_FOLDER, filename)
        print(f"Processing {filename}...")
        
        with open(path, 'rb') as f:
            file_bytes = f.read()
            hash_hex = hashlib.sha256(file_bytes).hexdigest()
            resume_id = str(uuid.UUID(hash_hex[:32]))

        parsed_data = parse_resume_pdf(path)
        
        skills_text_block = " ".join(parsed_data.get('SKILLS', []))
        vector_skills = embedding_model.encode(skills_text_block or "No skills found").tolist()
        vector_experience = embedding_model.encode(parsed_data['full_text']).tolist()
        
        payload = {key.lower(): " ".join(value) if isinstance(value, list) else value for key, value in parsed_data.items()}
        payload["filename"] = filename
        
        doc_nlp = nlp(payload['full_text'])
        payload['location'] = "Not Found"
        for ent in doc_nlp.ents:
            if ent.label_ == "GPE":
                payload['location'] = ent.text
                break
        
        points_to_upload.append(
            models.PointStruct(
                id=resume_id,
                vector={"skills": vector_skills, "experience": vector_experience},
                payload=payload
            )
        )
        
    if points_to_upload:
        setup_qdrant_collection()
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME, points=points_to_upload, wait=True
        )
    print(f"Successfully ingested {len(points_to_upload)} resumes.")

def search_resumes(query_text):
    print(f"\nAnalyzing query: '{query_text}'")
    doc_nlp = nlp(query_text)
    
    person_name = None
    for ent in doc_nlp.ents:
        if ent.label_ == "PERSON":
            person_name = ent.text
            break

    if not person_name:
        all_points, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=1000, with_payload=["candidate_name"])
        known_names = [point.payload['candidate_name'] for point in all_points if 'candidate_name' in point.payload]
        query_lower = query_text.lower()
        for name in known_names:
            first_name = name.split()[0].lower()
            if len(first_name) > 2 and first_name in query_lower:
                person_name = name
                break

    if person_name:
        print(f"Detected PERSON entity: '{person_name}'. Switching to QA mode.")

    #Candidate Search Mode 
    print("No person detected. Proceeding with Candidate Search.")
    query_location = None
    main_query_text = query_text
    for ent in doc_nlp.ents:
        if ent.label_ == "GPE":
            query_location = ent.text
            main_query_text = main_query_text.replace(ent.text, "").strip()
            break
            
    is_semantic_search = bool(main_query_text and not main_query_text.isspace())
    print(f"Detected Location: {query_location}, Semantic Query: {main_query_text if is_semantic_search else 'N/A (Location only)'}")

    search_filter = None
    if query_location:
        search_filter = models.Filter(must=[
            models.FieldCondition(key="location", match=models.MatchText(text=query_location))
        ])

    candidate_scores = {}
    if is_semantic_search:
        query_vector = embedding_model.encode(main_query_text).tolist()
        search_requests = [
            models.SearchRequest(vector=models.NamedVector(name="skills", vector=query_vector), filter=search_filter, limit=5, with_payload=True),
            models.SearchRequest(vector=models.NamedVector(name="experience", vector=query_vector), filter=search_filter, limit=5, with_payload=True)
        ]
        search_results = qdrant_client.search_batch(collection_name=COLLECTION_NAME, requests=search_requests)
        
        for i, result_set in enumerate(search_results):
            for hit in result_set:
                if hit.id not in candidate_scores:
                    candidate_scores[hit.id] = {"score": 0, "payload": hit.payload}
                score_weight = 1.5 if i == 0 else 1.0
                candidate_scores[hit.id]["score"] += hit.score * score_weight
    elif query_location:
        scroll_result, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, scroll_filter=search_filter, limit=10, with_payload=True)
        for hit in scroll_result:
             candidate_scores[hit.id] = {"score": 1.0, "payload": hit.payload}
    
    print("\n--- Top Candidates ---")
    if not candidate_scores:
        print("No candidates found matching all criteria.")
        return

    sorted_candidates = sorted(candidate_scores.values(), key=lambda x: x['score'], reverse=True)
    for candidate in sorted_candidates:
        payload = candidate['payload']
        # The skills_list is created by the updated ingest_resumes logic
        skills_list = payload.get('skills', '').split('\n')
        
        print(f"  📄 Filename: {payload['filename']}, Location: {payload['location']} (Score: {candidate['score']:.2f})")
        
        # FINAL FIX: Context-aware output logic
        if is_semantic_search:
             print(f"     Relevant Skills: {', '.join(skills_list[:3])}...")
        else:
             print(f"     (Match based on location filter)")

#MAIN EXECUTION 
if __name__ == "__main__":
    ingest_resumes()
    print("\n--- AI Resume Analyzer Chatbot ---")
    print("Ask me to find candidates or ask a question about a specific candidate. Type 'exit' to quit.")

    while True:
        user_query = input("\n> ")
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        if not user_query:
            continue
        search_resumes(user_query)