from __future__ import annotations
from typing import List, Dict, Any
import os

def extract_tables_pdf(pdf_path: str, max_tables: int = 3):
    """
    Returns a list of Pandas DataFrames extracted from the PDF.
    Tries Camelot first (if available), then pdfplumber.
    """
    out = []
    # Try camelot (optional)
    try:
        import camelot
        tables = camelot.read_pdf(pdf_path, flavor="stream", pages="1-end")
        for i, t in enumerate(tables):
            if i >= max_tables: break
            out.append(t.df)
    except Exception:
        pass

    # Fallback to pdfplumber
    if not out:
        try:
            import pdfplumber
            import pandas as pd
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tbls = page.extract_tables() or []
                    for t in tbls:
                        df = pd.DataFrame(t)
                        out.append(df)
                        if len(out) >= max_tables:
                            break
                    if len(out) >= max_tables:
                        break
        except Exception:
            pass
    return out
