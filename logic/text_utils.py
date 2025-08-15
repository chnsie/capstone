
import tiktoken
from typing import List

def _get_encoder(model: str = "gpt-4o-mini"):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def chunk_by_tokens(text: str, tokens_per_chunk: int = 400, overlap: int = 50, model: str = "gpt-4o-mini") -> List[str]:
    enc = _get_encoder(model)
    toks = enc.encode(text or "")
    chunks = []
    start = 0
    while start < len(toks):
        end = min(start + tokens_per_chunk, len(toks))
        chunks.append(enc.decode(toks[start:end]))
        if end == len(toks):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c.strip()]
