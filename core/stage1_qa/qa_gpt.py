from __future__ import annotations
from typing import List, Dict, Any
import textwrap, json

from core.schemas import Stage1QA, Stage1QAAnswer, Evidence
from core.llm.openai_client import get_client

def _topk(docs: List[Dict[str, Any]], question: str, k: int = 3) -> List[Dict[str, Any]]:
    try:
        from core.stage1_qa.qa_embed import topk_passages_embed
        return topk_passages_embed(question, docs, k=k)
    except Exception:
        from core.stage1_qa.qa import answer_question
        qa = answer_question(question, docs)
        e = qa.evidence[0] if qa.evidence else None
        return [{"rank": 1, "score": 1.0, "source": getattr(e, "source", "unknown"), "snippet": getattr(e, "snippet", "")}]

def _build_prompt(question: str, passages: List[Dict[str, Any]]) -> str:
    ctx = "\n\n".join([f"[{i+1}] source={p['source']}\n{p['snippet']}" for i,p in enumerate(passages)])
    return textwrap.dedent(f"""
    You are an evidence-first financial QA assistant.
    Answer ONLY from the provided snippets. If the answer is missing, return "INSUFFICIENT_DATA".

    Question:
    {question}

    Evidence snippets:
    {ctx}
    """)

def _call_gpt_json(prompt: str, model: str = "gpt-4.1") -> dict:
    """
    Try Responses API with JSON schema; if not supported (older SDK), fall back to Chat Completions JSON mode.
    Always returns a dict with keys: answer, evidence_sources, confidence.
    """
    client = get_client()

    schema = {
        "name": "stage1_qa_schema",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "evidence_sources": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string", "enum": ["low","medium","high"]}
            },
            "required": ["answer","evidence_sources","confidence"],
            "additionalProperties": False
        },
        "strict": True
    }

    # Try new Responses API first
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            response_format={"type":"json_schema","json_schema":schema}
        )
        try:
            return json.loads(resp.output_text)
        except Exception:
            # very old SDK shapes
            first = resp.output[0].content[0]
            txt = first.get("text") or first.get("output_text") or ""
            return json.loads(txt) if txt else {}
    except TypeError:
        # Older SDK: no response_format; fall back to Chat Completions JSON mode
        pass
    except AttributeError:
        # Older SDK: no Responses API at all
        pass

    # Chat Completions fallback (JSON mode, no schema enforcement—use instructions)
    sys_prompt = (
        "Return ONLY a JSON object with keys: answer (string), evidence_sources (array of strings), confidence (low|medium|high). "
        "Do not include any extra text."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content": sys_prompt},
            {"role":"user","content": prompt}
        ],
        response_format={"type":"json_object"}
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        # final safety
        return {"answer":"INSUFFICIENT_DATA","evidence_sources":[],"confidence":"low"}

def answer_question_gpt41(question: str, docs: List[Dict[str, Any]], k: int = 3, model: str = "gpt-4.1") -> Stage1QA:
    passages = _topk(docs, question, k=k)
    prompt = _build_prompt(question, passages)
    out = _call_gpt_json(prompt, model=model)

    answer = out.get("answer","INSUFFICIENT_DATA")
    conf = out.get("confidence","low")

    ev_list = []
    if passages:
        ev_list.append(Evidence(
            source=passages[0]["source"],
            snippet=passages[0]["snippet"][:400],
            location=""
        ))

    return Stage1QA(
        answer=Stage1QAAnswer(value=answer),
        evidence=ev_list,
        confidence=conf,
        notes_short="GPT-4.1 synthesis over top-k retrieved passages (with JSON fallback).",
        viz_spec={"type":"key_value_table","fields":{"Answer":"answer.value","Evidence":"evidence[0].snippet"}}
    )
