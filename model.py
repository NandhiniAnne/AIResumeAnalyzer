# model.py
"""
Embedding model wrapper that exposes:
  - init_skill_model(model_id=..., device=..., normalize=True)
  - returned object supports:
      .encode(text_or_list) -> list or numpy array (or list of lists)
      .get_sentence_embedding_dimension() -> int

Default uses HuggingFace transformers + Gemma model (google/gemma-3-270m-it).
If transformers is not available, optionally falls back to sentence_transformers.
"""

from typing import List, Union, Optional
import numpy as np

# Try preferred backend (transformers). If not present, fallback to sentence_transformers.
try:
    from transformers import AutoTokenizer, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False


class _BaseEmbeddingModel:
    def encode(self, texts: Union[str, List[str]]):
        raise NotImplementedError

    def get_sentence_embedding_dimension(self) -> int:
        raise NotImplementedError


if _HAS_TRANSFORMERS:
    import torch

    class GemmaTransformersModel(_BaseEmbeddingModel):
        def __init__(self, model_id: str = "google/gemma-3-270m-it", device: Optional[str] = None, normalize: bool = True):
            """
            model_id: HF model id for Gemma (change to your preferred variant).
            device: 'cpu' or 'cuda' or None (auto-select).
            normalize: whether to L2-normalize output embeddings.
            """
            self.model_id = model_id
            self.normalize = normalize

            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            # Load tokenizer & model
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            self.model = AutoModel.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()

            # hidden size from model config if available
            hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
            # fallback to embedding dim derived from model params
            self._dim = hidden_size or self.model.config.hidden_size

        def _mean_pooling(self, model_output, attention_mask):
            """
            Standard mean pooling of token embeddings using attention mask.
            model_output.last_hidden_state shape: (batch, seq_len, hidden)
            attention_mask shape: (batch, seq_len)
            """
            token_embeddings = model_output.last_hidden_state  # (batch, seq_len, hidden)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

        def encode(self, texts: Union[str, List[str]], batch_size: int = 16) -> List[List[float]]:
            """
            Encode input text (single string or list of strings) and return list of vectors (python lists).
            """
            single = False
            if isinstance(texts, str):
                texts = [texts]
                single = True
            vectors = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=1024)
                    input_ids = enc["input_ids"].to(self.device)
                    attention_mask = enc["attention_mask"].to(self.device)
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                    pooled = self._mean_pooling(out, attention_mask)  # tensor (batch, hidden)
                    pooled = pooled.cpu().numpy()
                    for vec in pooled:
                        v = np.asarray(vec, dtype=float)
                        if self.normalize:
                            norm = np.linalg.norm(v)
                            if norm > 0:
                                v = v / norm
                        vectors.append(v.tolist())
            return vectors[0] if single else vectors

        def get_sentence_embedding_dimension(self) -> int:
            return int(self._dim)


if _HAS_SENTENCE_TRANSFORMERS:
    class STModelWrapper(_BaseEmbeddingModel):
        def __init__(self, model_id: str = "all-MiniLM-L6-v2", device: Optional[str] = None, normalize: bool = True):
            self.model = SentenceTransformer(model_id, device=device or ('cuda' if hasattr(self.model, 'cuda') else 'cpu'))
            self.normalize = normalize
            # sentence-transformers has get_sentence_embedding_dimension
            try:
                self._dim = self.model.get_sentence_embedding_dimension()
            except Exception:
                # derive dimension by running a sample encode
                sample = self.model.encode("hello")
                self._dim = len(sample)

        def encode(self, texts: Union[str, List[str]], batch_size: int = 16) -> List[List[float]]:
            single = False
            if isinstance(texts, str):
                texts = [texts]
                single = True
            vecs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
            if self.normalize:
                vecs = [ (v / np.linalg.norm(v)).tolist() if np.linalg.norm(v) > 0 else v.tolist() for v in np.asarray(vecs) ]
            else:
                vecs = [v.tolist() for v in np.asarray(vecs)]
            return vecs[0] if single else vecs

        def get_sentence_embedding_dimension(self) -> int:
            return int(self._dim)


# in model.py (replace or update existing init_skill_model)
def init_skill_model(model_id: str = "google/gemma-3-270m-it",
                     backend: str = "auto",
                     device: Optional[str] = None,
                     normalize: bool = True,
                     **kwargs) -> _BaseEmbeddingModel:
    """
    Initialize and return an embedding model wrapper.

    Extra kwargs (e.g. local_snapshot, trust_remote_code) are accepted and ignored
    so callers that pass extra options won't break.
    """
    # quietly accept/back-compat any extra kwargs (like local_snapshot)
    # You may log unsupported kwargs if desired:
    if kwargs:
        # simple debug print — safe to remove or change to logging
        print("init_skill_model: ignoring extra kwargs:", list(kwargs.keys()))

    backend = (backend or "auto").lower()
    if backend == "transformers":
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers is not installed. Install with `pip install transformers`")
        return GemmaTransformersModel(model_id=model_id, device=device, normalize=normalize)

    if backend == "sentence_transformers":
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers is not installed. Install with `pip install sentence-transformers`")
        return STModelWrapper(model_id=model_id, device=device, normalize=normalize)

    # auto
    if _HAS_TRANSFORMERS:
        try:
            return GemmaTransformersModel(model_id=model_id, device=device, normalize=normalize)
        except Exception as e:
            if _HAS_SENTENCE_TRANSFORMERS:
                return STModelWrapper(device=device
                                      , normalize=normalize)
            raise e
    elif _HAS_SENTENCE_TRANSFORMERS:
        return STModelWrapper(device=device, normalize=normalize)
    else:
        raise ImportError("No supported embedding backend installed. Install `transformers` or `sentence-transformers`.")


# If this file is executed directly, demonstrate a quick smoke test (won't run on import)
if __name__ == "__main__":
    print("Initializing embedding model (smoke test)...")
    try:
        m = init_skill_model()
        print("Embedding dim:", m.get_sentence_embedding_dimension())
        s = "This is a quick test for data engineer resume search."
        v = m.encode(s)
        print("Vector length:", len(v), "First 8 dims:", v[:8] if isinstance(v, list) else v[0][:8])
    except Exception as ex:
        print("Failed to initialize embedding model:", ex)

