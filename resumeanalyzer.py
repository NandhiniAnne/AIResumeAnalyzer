from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# --- Config ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "resumes"
MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize Qdrant client and embedding model
client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(MODEL_NAME)

# --- Query function ---
def query_resumes(query_text=None, city=None, area_code=None, university=None, top_k=5):
    """
    Query resumes in Qdrant with optional exact filters and semantic search.
    
    Parameters:
        query_text: str (optional) - free text query for semantic search
        city: str (optional) - exact city filter
        area_code: str (optional) - exact phone area code filter
        university: str (optional) - exact university filter
        top_k: int - number of results to return
    """
    
    # Build exact-match filter
    must_conditions = []
    if city:
        must_conditions.append(FieldCondition(key="city", match=MatchValue(value=city)))
    if area_code:
        must_conditions.append(FieldCondition(key="area_code", match=MatchValue(value=area_code)))
    if university:
        must_conditions.append(FieldCondition(key="university", match=MatchValue(value=university)))
    
    filter_cond = Filter(must=must_conditions) if must_conditions else None
    
    # Encode query text to vector
    query_vector = model.encode(query_text).tolist() if query_text else None
    
    # Perform search
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        filter=filter_cond  # works in latest qdrant-client
    )
    
    # Display results
    if not results:
        print("No resumes matched your query.")
        return

    for res in results:
        payload = res.payload
        print(f"File: {payload.get('file')}")
        print(f"Name: {payload.get('name')}")
        print(f"Email: {payload.get('email')}")
        print(f"Phone: {payload.get('phone')}")
        print(f"City: {payload.get('city')}")
        print(f"University: {payload.get('university')}")
        print(f"Score: {res.score:.4f}")
        print("-" * 50)

# --- Example queries ---
if __name__ == "__main__":
    print("Query: Who is from Tucson?")
    query_resumes(city="Tucson")
    
    print("\nQuery: Who has area code 520?")
    query_resumes(area_code="520")
    
    print("\nQuery: Who studied at University of Missouri?")
    query_resumes(university="University of Missouri")
    
    print("\nQuery: Who knows Python and Machine Learning? (semantic search)")
    query_resumes(query_text="Python Machine Learning", top_k=5)
    
    print("\nQuery: Who from Tucson knows Python? (combined filter + semantic)")
    query_resumes(query_text="Python", city="Tucson", top_k=5)