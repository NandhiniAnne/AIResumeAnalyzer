import os
import re
import random
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# --- Config ---
RESUME_FOLDER = "resumes"          # folder containing PDF resumes
COLLECTION_NAME = "resumes"
MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"

# --- Initialize ---
client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(MODEL_NAME)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={"size": model.get_sentence_embedding_dimension(), "distance": "Cosine"}
)

# --- Helper functions ---
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group() if match else None

def extract_phone(text):
    match = re.search(r'\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})', text)
    return match.group() if match else None

def extract_area_code(phone):
    if phone:
        return phone[:3]
    return None

def extract_city(text):
    cities = ['Tucson', 'Phoenix', 'Kansas', 'Mesa']  # Add more cities
    for city in cities:
        if city.lower() in text.lower():
            return city
    return None

def extract_university(text):
    universities = [
        'University of Arizona', 'University of Missouri', 'MIT', 'Stanford'
    ]  # Add more universities
    for uni in universities:
        if uni.lower() in text.lower():
            return uni
    return None

def parse_resume(file_path):
    text = extract_text_from_pdf(file_path)
    vector = model.encode(text).tolist()
    
    metadata = {
        "file": os.path.basename(file_path),
        "name": os.path.splitext(os.path.basename(file_path))[0],
        "email": extract_email(text),
        "phone": extract_phone(text),
        "area_code": extract_area_code(extract_phone(text)),
        "city": extract_city(text),
        "university": extract_university(text)
    }
    
    # Qdrant requires integer ID
    point_id = random.randint(1, 1_000_000_000)
    
    return PointStruct(id=point_id, vector=vector, payload=metadata)

# --- Ingest resumes ---
for resume_file in os.listdir(RESUME_FOLDER):
    if resume_file.endswith(".pdf"):
        full_path = os.path.join(RESUME_FOLDER, resume_file)
        point = parse_resume(full_path)
        client.upsert(collection_name=COLLECTION_NAME, points=[point])
print("Resume ingestion complete!")