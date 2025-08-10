from __future__ import annotations
import os, json, hashlib
from typing import List, Dict, Any

# Where we store cached indices
INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "index_cache"))
os.makedirs(INDEX_DIR, exist_ok=True)

def _fingerprint_docs(docs: List[Dict[str, Any]], model_name: str) -> str:
    """Stable short fingerprint of docs + model (first 16 hex chars)."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for d in docs:
        h.update((d.get("id","") + "|" + d.get("company","") + "|" + d.get("period","")).encode("utf-8"))
        # limit text to keep hash cheap but stable
        h.update((d.get("text","")[:2000]).encode("utf-8"))
    return h.hexdigest()[:16]

def _paths(fp: str):
    base = os.path.join(INDEX_DIR, fp)
    return {
        "emb":  base + ".npy",
        "meta": base + ".meta.json",
        "model": base + ".model.txt",
        "faiss": base + ".faiss",
    }

def save_index(fp: str, emb, meta: List[Dict[str,Any]], model_name: str):
    import numpy as np, faiss
    p = _paths(fp)
    np.save(p["emb"], emb.astype("float32"))
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(p["model"], "w", encoding="utf-8") as f:
        f.write(model_name)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    faiss.write_index(index, p["faiss"])

def load_index(fp: str):
    import numpy as np, faiss
    p = _paths(fp)
    if not (os.path.exists(p["emb"]) and os.path.exists(p["meta"]) and os.path.exists(p["model"]) and os.path.exists(p["faiss"])):
        return None
    emb = np.load(p["emb"])
    with open(p["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(p["model"], "r", encoding="utf-8") as f:
        model_name = f.read().strip()
    index = faiss.read_index(p["faiss"])
    return {"emb": emb, "meta": meta, "model_name": model_name, "index": index}

def get_or_build_index(docs: List[Dict[str,Any]], model_name: str = "all-MiniLM-L6-v2"):
    """
    Return {'index','meta','model','model_name'} using disk cache when available.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np, re, faiss

    # Build passages & meta
    passages, meta = [], []
    def _split_paragraphs(text: str):
        paras = re.split(r"\n{2,}|(?<=\.)\s*\n", text or "")
        return [p.strip() for p in paras if p and len(p.strip()) > 10]
    for d in docs:
        for para in _split_paragraphs(d.get("text","")):
            passages.append(para)
            meta.append({"doc": d, "para": para})

    if not passages:
        return {"index": None, "meta": [], "model": None, "model_name": model_name}

    fp = _fingerprint_docs(docs, model_name)
    cached = load_index(fp)
    if cached:
        model = SentenceTransformer(cached["model_name"])
        return {"index": cached["index"], "meta": cached["meta"], "model": model, "model_name": cached["model_name"]}

    # build fresh
    model = SentenceTransformer(model_name)
    emb = model.encode(passages, normalize_embeddings=True, convert_to_numpy=True)
    save_index(fp, emb, meta, model_name)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    return {"index": index, "meta": meta, "model": model, "model_name": model_name}
