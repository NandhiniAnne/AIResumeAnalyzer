from huggingface_hub import snapshot_download
print("Downloading spaCy skill model (this may take a moment)...")
path = snapshot_download("amjad-awad/skill-extractor", repo_type="model")
print("Model downloaded to:", path)