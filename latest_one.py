import os
import re
import fitz  # PyMuPDF
import docx  # python-docx
import hashlib
import uuid
import ollama
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# --- 1. CONFIGURATION ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "llm_resume_collection"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'llama3:8b'
RESUME_FOLDER = 'resumes'

# --- 2. INITIALIZATION ---
print("Loading models and clients...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
print("Models and clients loaded.")

# --- 3. CORE FUNCTIONS ---

def parse_resume_file(file_path):
    """
    Universal parser for PDF and DOCX files. Extracts text, sections,
    and contact details.
    """
    full_text = ""
    if file_path.endswith('.pdf'):
        doc = fitz.open(file_path)
        for page in doc:
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))
            for b in blocks:
                full_text += b[4]
        doc.close()
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            full_text += para.text + '\n'
    
    # --- The rest of the parsing logic is the same ---
    sections = {
        'full_text': full_text, 'candidate_name': "Unknown",
        'email': "Not Found", 'phone': "Not Found"
    }

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\(?\b[2-9][0-9]{2}\)?[-. ]?\b[2-9][0-9]{2}[-. ]?\b[0-9]{4}\b'
    
    email_match = re.search(email_pattern, full_text)
    if email_match: sections['email'] = email_match.group(0)
    phone_match = re.search(phone_pattern, full_text)
    if phone_match: sections['phone'] = phone_match.group(0)

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
        if not line_stripped: continue
        match = re.match(pattern, line_stripped, re.IGNORECASE)
        if match and len(line_stripped) < 30:
            current_section_name = match.group(1).upper().replace("WORK EXPERIENCE", "EXPERIENCE")
            sections[current_section_name] = []
        else:
            cleaned_line = re.sub(r'^\s*[\•\\-\*]\s*', '', line_stripped)
            if cleaned_line:
                if current_section_name not in sections: sections[current_section_name] = []
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
    """Ingests PDF and DOCX files from the resumes folder."""
    print(f"Starting resume ingestion from folder: '{RESUME_FOLDER}'")
    supported_formats = ('.pdf', '.docx')
    resume_files = [f for f in os.listdir(RESUME_FOLDER) if f.lower().endswith(supported_formats)]
    points_to_upload = []
    
    for filename in resume_files:
        path = os.path.join(RESUME_FOLDER, filename)
        print(f"Processing {filename}...")
        
        with open(path, 'rb') as f:
            file_bytes = f.read()
            hash_hex = hashlib.sha256(file_bytes).hexdigest()
            resume_id = str(uuid.UUID(hash_hex[:32]))

        parsed_data = parse_resume_file(path)
        
        skills_text_block = " ".join(parsed_data.get('SKILLS', []))
        vector_skills = embedding_model.encode(skills_text_block or "No skills found").tolist()
        vector_experience = embedding_model.encode(parsed_data['full_text']).tolist()
        
        payload = {key.lower(): " ".join(value) if isinstance(value, list) else value for key, value in parsed_data.items()}
        payload["filename"] = filename
        
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

def get_query_intent_with_llm(query_text):
    """Uses a local LLM to parse the user's query into a structured format."""
    prompt = f"""
    You are an expert query parser for a resume database. Analyze the user's query and extract entities into a JSON object.
    The JSON object must have the following keys:
    - "intent": Can be "search_candidate" or "question_about_candidate".
    - "person_name": The full name of a person, if mentioned. Otherwise, null.
    - "information_requested": The specific piece of information asked for (e.g., "skills", "certifications", "phone"), if the intent is "question_about_candidate". Otherwise, null.
    - "skills": A list of skills or job roles mentioned. Otherwise, an empty list.
    - "location": The geographic location mentioned, if any. Otherwise, null.

    User Query: "{query_text}"

    JSON Output:
    """
    
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0}
        )
        json_string = response['message']['content']
        # Clean up potential markdown formatting
        if json_string.startswith("```json"):
            json_string = json_string[7:-3].strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return None

def search_resumes(query_text):
    """
    Final unified function that uses an LLM for intent recognition.
    """
    print(f"\nAnalyzing query with LLM: '{query_text}'")
    intent_data = get_query_intent_with_llm(query_text)

    if not intent_data:
        print("Sorry, I could not understand your request.")
        return

    # --- QA Mode ---
    if intent_data.get("intent") == "question_about_candidate":
        person_name = intent_data.get("person_name")
        info_requested = intent_data.get("information_requested")
        if not person_name:
            print("You asked a question, but didn't specify a candidate's name.")
            return

        print(f"Detected intent: QA for '{person_name}'. Looking for '{info_requested}'.")
        # Find the candidate
        all_candidates, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=1000, with_payload=True)
        found_candidate_payload = None
        for point in all_candidates:
            if person_name.lower() in point.payload.get('candidate_name', '').lower():
                found_candidate_payload = point.payload
                break
        
        if not found_candidate_payload:
            print(f"Sorry, I could not find a resume for '{person_name}'.")
            return
        
        print("\n--- Answer ---")
        if info_requested and found_candidate_payload.get(info_requested.lower()):
            print(f"Here is the {info_requested.upper()} for {found_candidate_payload['filename']}:")
            print(f"  - {found_candidate_payload[info_requested.lower()]}")
        else:
            print(f"I found {found_candidate_payload['filename']}, but couldn't find the specific section '{info_requested}'.")
        return

    # --- Candidate Search Mode ---
    elif intent_data.get("intent") == "search_candidate":
        location = intent_data.get("location")
        skills = intent_data.get("skills", [])
        print(f"Detected intent: Search. Location: {location}, Skills: {skills}")

        search_filter = None
        if location:
            search_filter = models.Filter(must=[
                models.FieldCondition(key="location", match=models.MatchText(text=location))
            ])
        
        main_query_text = " ".join(skills)
        if not main_query_text:
            print("Please specify some skills or a role to search for.")
            return

        query_vector = embedding_model.encode(main_query_text).tolist()
        search_requests = [
            models.SearchRequest(vector=models.NamedVector(name="skills", vector=query_vector), filter=search_filter, limit=5, with_payload=True),
            models.SearchRequest(vector=models.NamedVector(name="experience", vector=query_vector), filter=search_filter, limit=5, with_payload=True)
        ]
        search_results = qdrant_client.search_batch(collection_name=COLLECTION_NAME, requests=search_requests)
        
        candidate_scores = {}
        for i, result_set in enumerate(search_results):
            for hit in result_set:
                if hit.id not in candidate_scores:
                    candidate_scores[hit.id] = {"score": 0, "payload": hit.payload}
                score_weight = 1.5 if i == 0 else 1.0
                candidate_scores[hit.id]["score"] += hit.score * score_weight
        
        print("\n--- Top Candidates ---")
        if not candidate_scores:
            print("No candidates found matching all criteria.")
            return

        sorted_candidates = sorted(candidate_scores.values(), key=lambda x: x['score'], reverse=True)
        for candidate in sorted_candidates:
            payload = candidate['payload']
            print(f"  📄 Filename: {payload['filename']}, Location: {payload.get('location', 'N/A')} (Score: {candidate['score']:.2f})")
            print(f"     Relevant Skills: {payload.get('skills', 'N/A')[:100]}...")

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    ingest_resumes()
    print("\n--- AI Resume Analyzer Chatbot (LLM-Powered) ---")
    print("Ask me to find candidates or ask a question about a specific candidate. Type 'exit' to quit.")

    while True:
        user_query = input("\n> ")
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        if not user_query:
            continue
        search_resumes(user_query)
