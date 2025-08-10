from __future__ import annotations
from typing import List, Dict, Any
import os
from PyPDF2 import PdfReader

def _read_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def _read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or '')
            except Exception:
                pages.append('')
        return '\n'.join(pages)
    except Exception:
        return ''

def load_docs(paths: List[str], default_company: str = 'UnknownCo', default_period: str = 'Unknown') -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == '.txt':
            text = _read_txt(p)
        elif ext == '.pdf':
            text = _read_pdf(p)
        else:
            continue
        doc_id = os.path.basename(p)
        docs.append({
    "id": doc_id,
    "company": default_company,
    "period": default_period,
    "text": text,
    "source_url": None,
    "location_hint": "file:"+doc_id,
    "path": os.path.abspath(p)
})
    return docs
