from __future__ import annotations
from typing import Dict, Any, List
import json
from core.llm.openai_client import get_client

def refine_rationales(signals: Dict[str,Any], raw_bullets: List[str], model: str = "gpt-4.1") -> List[str]:
    client = get_client()
    prompt = f"""
    You are summarizing trading rationales for a demo. Rewrite the bullets to be crisp, non-hype, and grounded in the signals.

    Signals JSON:
    {json.dumps(signals, indent=2)}

    Raw bullets:
    {json.dumps(raw_bullets, indent=2)}

    Return a list of 2-4 short bullets. No extra text.
    """
    resp = client.responses.create(model=model, input=prompt)
    txt = resp.output_text.strip()
    # simple parse: split by lines / bullets
    lines = [l.strip("•- ").strip() for l in txt.splitlines() if l.strip()]
    return [l for l in lines if l][:4]
