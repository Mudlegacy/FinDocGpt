from __future__ import annotations
from typing import List, Dict, Any
import re
from collections import Counter
from core.schemas import Stage1Sentiment, Stage1SentimentAnswer, Evidence, SentimentDriver, Stage1SentimentSections

POS_WORDS = set('growth profit profitable strong improvement expansion benefit record exceeded beat optimistic robust increasing'.split())
NEG_WORDS = set('decline loss losses weak headwind risk risks uncertainty impairment miss lawsuit investigation decreasing negative'.split())

def simple_sentiment_score(text: str) -> float:
    tokens = re.findall(r'[a-zA-Z\']+', text.lower())
    c = Counter(tokens)
    pos = sum(c[w] for w in POS_WORDS)
    neg = sum(c[w] for w in NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    score = (pos - neg) / total
    return max(-1.0, min(1.0, score))

def top_drivers(text: str, k: int = 5):
    tokens = re.findall(r'[a-zA-Z\']+', text.lower())
    c = Counter(tokens)
    drivers = []
    for w in sorted(POS_WORDS, key=lambda w: -c[w])[:k]:
        if c[w] > 0:
            drivers.append(SentimentDriver(phrase=w, impact='+'))
    for w in sorted(NEG_WORDS, key=lambda w: -c[w])[:k]:
        if c[w] > 0:
            drivers.append(SentimentDriver(phrase=w, impact='-'))
    return drivers[:k]

def split_sections(text: str):
    lower = text.lower()
    out_idx = lower.find('outlook')
    res_idx = lower.find('results')
    outlook = text[out_idx: out_idx+500] if out_idx != -1 else ''
    results = text[res_idx: res_idx+500] if res_idx != -1 else ''
    return outlook, results

def analyze_sentiment(docs: List[Dict[str, Any]]) -> Stage1Sentiment:
    if not docs:
        return Stage1Sentiment(answer=Stage1SentimentAnswer(overall={'label':'neutral','score':0.0}, drivers=[], sections=Stage1SentimentSections()), evidence=[], confidence='low', notes_short='No documents provided.')
    joined = '\n\n'.join(d['text'][:5000] for d in docs)
    score = simple_sentiment_score(joined)
    label = 'positive' if score > 0.2 else 'negative' if score < -0.2 else 'neutral'
    drivers = top_drivers(joined, k=5)
    out, res = split_sections(joined)
    ev = [Evidence(source=docs[0]['id'], snippet=docs[0]['text'][:300], location=docs[0].get('location_hint',''))]
    return Stage1Sentiment(answer=Stage1SentimentAnswer(overall={'label':label,'score':round(score,3)}, drivers=drivers, sections=Stage1SentimentSections(outlook=out, results=res)), evidence=ev, confidence='medium', notes_short='Lexicon-based sentiment; consider upgrading later.')
