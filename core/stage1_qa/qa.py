from __future__ import annotations
from typing import List, Dict, Any
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from core.schemas import Stage1QA, Stage1QAAnswer, Evidence

def _split_paragraphs(text: str) -> List[str]:
    paras = re.split(r'\n{2,}|(?<=\.)\s*\n', text)
    return [p.strip() for p in paras if p and len(p.strip()) > 10]

def retrieve_best_passage(question: str, docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    corpus, meta = [], []
    for d in docs:
        for para in _split_paragraphs(d['text']):
            corpus.append(para)
            meta.append({'doc': d, 'para': para})
    if not corpus:
        return []
    vect = TfidfVectorizer(stop_words='english', max_features=8000)
    X = vect.fit_transform(corpus + [question])
    doc_mat = X[:-1]
    q_vec = X[-1]
    sims = cosine_similarity(doc_mat, q_vec)
    idxs = np.argsort(-sims.flatten())[:top_k]
    return [meta[i] for i in idxs]

def extract_answer_span(passage: str, question: str) -> str:
    sents = re.split(r'(?<=[.!?])\s+', passage.strip())
    if re.search(r'\d', question) or any(w in question.lower() for w in ['how much','revenue','eps','growth','percent','%']):
        best = max(sents, key=lambda s: len(re.findall(r'[\d,\.%]+', s)))
        return best.strip()
    return sents[0].strip() if sents else passage.strip()

def answer_question(question: str, docs: List[Dict[str, Any]]) -> Stage1QA:
    top = retrieve_best_passage(question, docs, top_k=3)
    if not top:
        return Stage1QA(answer=Stage1QAAnswer(value='INSUFFICIENT_DATA'), evidence=[], confidence='low', notes_short='No passages found.', viz_spec=None)
    best = top[0]
    span = extract_answer_span(best['para'], question)
    ev = [Evidence(source=best['doc']['id'], snippet=best['para'][:400], location=best['doc'].get('location_hint',''))]
    return Stage1QA(answer=Stage1QAAnswer(value=span), evidence=ev, confidence='medium', notes_short='Top TF-IDF passage with answer span.', viz_spec={'type':'key_value_table','fields':{'Answer':'answer.value','Evidence':'evidence[0].snippet'}})
