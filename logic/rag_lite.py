
import os, json, time
from typing import List, Dict, Any, Optional
import numpy as np

from helper_functions.llm import get_embedding
from logic.text_utils import chunk_by_tokens
from logic.docx_to_guides import read_docx_paragraphs

DOCX_PATH = "data/Delivery of Digital Products.docx"
GUIDES_JSON = "data/guides.json"

DOCX_INDEX = "index/docx_index.json"
GUIDES_INDEX = "index/guides_index.json"
META_LITE = "index/meta_lite.json"

# --- coercion helpers ---
def _to_text(v):
    if v is None:
        return ""
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return " ".join([_to_text(x) for x in v if x is not None])
    if isinstance(v, dict):
        for k in ("name","title","text","value","label"):
            if k in v and isinstance(v[k], (str,int,float,bool)):
                return str(v[k])
        try:
            return " ".join([_to_text(x) for x in v.values() if x is not None])
        except Exception:
            return str(v)
    return str(v)

def _to_words_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (int,float,bool)):
        return [str(v)]
    if isinstance(v, list):
        out = []
        for x in v:
            out.extend(_to_words_list(x))
        return out
    if isinstance(v, dict):
        out = []
        for val in v.values():
            out.extend(_to_words_list(val))
        return out
    return [str(v)]

def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

def _save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-10
    return v / norms

def _embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype="float32")
    embs = get_embedding(texts)
    mat = np.array(embs, dtype="float32")
    return _normalize(mat)

def build_docx_index():
    if not os.path.exists(DOCX_PATH):
        return {"vecs": [], "chunks": [], "meta": []}
    paras = read_docx_paragraphs(DOCX_PATH)
    text = "\n".join([t for _, t in paras])
    chunks = chunk_by_tokens(text, tokens_per_chunk=380, overlap=40)
    vecs = _embed_texts(chunks).tolist()
    meta = [{"source": "docx", "title": "Delivery of Digital Products", "phase": "" , "chunk_id": f"docx#c{i+1}"} for i,_ in enumerate(chunks)]
    idx = {"vecs": vecs, "chunks": chunks, "meta": meta}
    _save_json(DOCX_INDEX, idx)
    return idx

def _summarize_guide(g: Dict[str, Any]) -> str:
    parts = [
        _to_text(g.get("guide_name") or g.get("title") or ""),
        _to_text(g.get("purpose") or ""),
        " ".join(_to_words_list(g.get("tags"))),
        _to_text(g.get("audience") or g.get("owner") or ""),
        _to_text(g.get("phase") or ""),
        " ".join(_to_words_list(g.get("key_terms"))),
        " ".join(_to_words_list(g.get("use_case"))),
        " ".join(_to_words_list(g.get("common_questions"))),
        _to_text(g.get("context") or ""),
    ]
    flat = [p for p in parts if isinstance(p, str) and p.strip()]
    return " | ".join(flat)

def build_guides_index():
    data = _load_json(GUIDES_JSON) or {}
    guides = data.get("guides") or data.get("Guides") or (data if isinstance(data, list) else [])
    items = []
    for g in guides:
        gid = g.get("id") or g.get("doc_id") or g.get("guide_id") or (g.get("title") or "guide").lower().replace(" ","-")
        title = _to_text(g.get("title") or g.get("guide_name") or "Untitled")
        url = _to_text(g.get("url") or "")
        phase = _to_text(g.get("phase") or "")
        tags = _to_words_list(g.get("tags"))
        audience = _to_text(g.get("audience") or g.get("target_audience") or g.get("targetAudience") or g.get("owner") or "")
        purpose = _to_text(g.get("purpose") or "")
        key_terms = _to_words_list(g.get("key_terms") or g.get("keywords") or g.get("keyTerms"))
        use_case = _to_words_list(g.get("use_case"))
        common_qs = _to_words_list(g.get("common_questions"))
        text = _summarize_guide(g)
        if not text.strip():
            continue
        items.append({
            "id": gid,
            "title": title,
            "url": url,
            "phase": phase,
            "tags": tags,
            "audience": audience,
            "purpose": purpose,
            "key_terms": key_terms,
            "use_case": use_case,
            "common_questions": common_qs,
            "text": text
        })
    vecs = _embed_texts([it["text"] for it in items]).tolist()
    idx = {"vecs": vecs, "items": items}
    _save_json(GUIDES_INDEX, idx)
    return idx

def ensure_built():
    prev = _load_json(META_LITE) or {}
    prev_d = prev.get("docx_mtime", 0.0)
    prev_g = prev.get("guides_mtime", 0.0)
    cur_d = _mtime(DOCX_PATH)
    cur_g = _mtime(GUIDES_JSON)
    rebuilt = False
    if not os.path.exists(DOCX_INDEX) or cur_d != prev_d:
        build_docx_index(); rebuilt = True
    if not os.path.exists(GUIDES_INDEX) or cur_g != prev_g:
        build_guides_index(); rebuilt = True
    if rebuilt:
        _save_json(META_LITE, {"docx_mtime": cur_d, "guides_mtime": cur_g, "built_at": int(time.time())})
    return {"status": "rebuilt" if rebuilt else "up-to-date"}

def _cosine_search(query: str, mat: np.ndarray) -> np.ndarray:
    q = _embed_texts([query])
    if q.shape[1] != mat.shape[1]:
        return np.zeros((mat.shape[0],), dtype="float32")
    return (mat @ q.T)[:,0]

def search_docx(query: str, phase: Optional[str] = None, top_k: int = 6):
    idx = _load_json(DOCX_INDEX) or {"vecs": [], "chunks": [], "meta": []}
    if not idx["vecs"]:
        return []
    mat = np.array(idx["vecs"], dtype="float32")
    scores = _cosine_search(query, mat)
    if phase:
        tokens = phase.lower().split()
        for i, chunk in enumerate(idx["chunks"]):
            if any(t in chunk.lower() for t in tokens):
                scores[i] += 0.05
    order = list(np.argsort(-scores)[:top_k])
    return [{"text": idx["chunks"][i], "meta": idx["meta"][i], "score": float(scores[i])} for i in order]

def search_guides(query: str, role: Optional[str] = None, phase: Optional[str] = None, keywords: Optional[List[str]] = None, top_k: int = 8):
    idx = _load_json(GUIDES_INDEX) or {"vecs": [], "items": []}
    if not idx["vecs"]:
        return []
    mat = np.array(idx["vecs"], dtype="float32")
    base = _cosine_search(query, mat)
    scores = base.copy()
    for i, it in enumerate(idx["items"]):
        if phase and (it.get("phase") or "").lower() == (phase or "").lower():
            scores[i] += 0.10
        if role and role.lower() in (it.get("audience") or "").lower():
            scores[i] += 0.05
        hay = " ".join([
            " ".join(_to_words_list(it.get("tags"))),
            " ".join(_to_words_list(it.get("key_terms"))),
            " ".join(_to_words_list(it.get("use_case"))),
            " ".join(_to_words_list(it.get("common_questions"))),
            _to_text(it.get("text",""))
        ]).lower()
        if keywords:
            overlap = sum(1 for k in keywords if k.lower() in hay)
            scores[i] += 0.03 * min(overlap, 4)
        if not (it.get("url") or "").strip():
            scores[i] -= 0.01

    order = list(np.argsort(-scores))[:top_k*2]
    out, seen = [], set()
    for i in order:
        it = idx["items"][i]
        title = it.get("title") or "Untitled"
        if title in seen: continue
        seen.add(title)
        out.append({"item": it, "score": float(scores[i])})
        if len(out) >= top_k: break
    return out
