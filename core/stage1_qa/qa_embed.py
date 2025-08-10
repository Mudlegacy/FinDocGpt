from __future__ import annotations
from typing import List, Dict, Any
import re

def _split_paragraphs(text: str) -> List[str]:
    paras = re.split(r"\n{2,}|(?<=\.)\s*\n", text)
    return [p.strip() for p in paras if p and len(p.strip()) > 10]

def extract_answer_span(passage: str, question: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", passage.strip())
    if re.search(r"\d", question) or any(w in question.lower() for w in ["how much","revenue","eps","growth","percent","%"]):
        best = max(sents, key=lambda s: len(re.findall(r"[\d,\.%]+", s))) if sents else passage
        return best.strip()
    return sents[0].strip() if sents else passage.strip()

def answer_question_embed(question: str, docs: List[Dict[str, Any]], top_k: int = 3):
    """
    Embedding-based retrieval with FAISS + sentence-transformers.
    Uses a disk-cached FAISS index (fast after first run).
    Falls back to TF-IDF if deps/cache fail.
    """
    try:
        import numpy as np, faiss  # noqa
        from sentence_transformers import SentenceTransformer  # noqa
        from core.stage1_qa.index_cache import get_or_build_index
        from core.schemas import Stage1QA, Stage1QAAnswer, Evidence
    except Exception:
        from core.stage1_qa.qa import answer_question as tfidf_answer
        return tfidf_answer(question, docs)

    # Build/load cached index + model
    try:
        built = get_or_build_index(docs, model_name="all-MiniLM-L6-v2")
        index = built["index"]; meta = built["meta"]; model = built["model"]
        if not index or not meta or not model:
            from core.stage1_qa.qa import answer_question as tfidf_answer
            return tfidf_answer(question, docs)
    except Exception:
        from core.stage1_qa.qa import answer_question as tfidf_answer
        return tfidf_answer(question, docs)

    # Encode question and search
    try:
        q_emb = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        D, I = index.search(q_emb.astype("float32"), max(1, top_k))
        if I.size == 0 or I[0][0] < 0:
            from core.stage1_qa.qa import answer_question as tfidf_answer
            return tfidf_answer(question, docs)
        best = meta[int(I[0][0])]
        span = extract_answer_span(best["para"], question)
        ev = [Evidence(source=best["doc"]["id"], snippet=best["para"][:400], location=best["doc"].get("location_hint",""))]
        return Stage1QA(
            answer=Stage1QAAnswer(value=span),
            evidence=ev,
            confidence="medium",
            notes_short="Embeddings+FAISS top match (cached index). Falls back to TF-IDF if unavailable.",
            viz_spec={"type":"key_value_table","fields":{"Answer":"answer.value","Evidence":"evidence[0].snippet"}}
        )
    except Exception:
        from core.stage1_qa.qa import answer_question as tfidf_answer
        return tfidf_answer(question, docs)

# --- helper: return top-k passages with similarity scores (uses cached index) ---
def topk_passages_embed(question: str, docs: List[Dict[str, Any]], k: int = 3):
    try:
        import numpy as np  # noqa
        from core.stage1_qa.index_cache import get_or_build_index
    except Exception:
        return []

    try:
        built = get_or_build_index(docs, model_name="all-MiniLM-L6-v2")
        index = built["index"]; meta = built["meta"]; model = built["model"]
        if not index or not meta or not model:
            return []
        q_emb = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        D, I = index.search(q_emb.astype("float32"), min(k, len(meta)))
        out = []
        for rank, (score, idx) in enumerate(zip(D[0].tolist(), I[0].tolist()), start=1):
            if idx < 0:
                continue
            m = meta[int(idx)]
            out.append({
                "rank": rank,
                "score": float(score),
                "source": m["doc"]["id"],
                "snippet": m["para"][:600]
            })
        return out
    except Exception:
        return []
