# core/chat/chatbot.py
from __future__ import annotations
from typing import Dict, Any, List
import json
from core.llm.openai_client import get_client

def _build_context(ss: Dict[str, Any]) -> Dict[str, Any]:
    """Collects the latest run outputs from session_state into one compact dict."""
    return {
        "qa": ss.get("qa_json"),
        "sentiment": ss.get("sent_json"),
        "anomalies": ss.get("an_json"),
        "forecast": ss.get("fc_json"),
        "decision": ss.get("dec_json"),
    }

def _framework_note() -> str:
    return (
        "Framework: (1) Retrieval-QA with evidence; (2) Document sentiment; "
        "(3) Simple anomaly flags; (4) ARIMA forecast on daily returns with 80/95% intervals and backtest "
        "(RMSE/sMAPE/MAPE); (5) Rule-based decision with position size, stop-loss, take-profit, review date. "
        "All outputs are JSON for auditability. Not financial advice."
    )

def chat_reply(user_message: str, ss: Dict[str, Any], model: str = "gpt-4.1") -> str:
    """
    Produces an assistant reply grounded in the current run (stored in session_state).
    Keeps language clear and avoids giving financial advice.
    """
    ctx = _build_context(ss)
    framework = _framework_note()

    # Gentle guardrails: summarize from context; if missing pieces, say so.
    system_prompt = (
        "You are FinDocGPT's explainer. Use ONLY the supplied context to answer questions about the pipeline, "
        "signals, and decision. Keep responses concise, structured, and educational. "
        "Do not provide investment advice. If a value is missing in context, say what stage to run."
    )

    # Build a compact summary to help GPT stay grounded
    summary_bits: List[str] = []
    try:
        # Decision
        dec = (ctx.get("decision") or {}).get("answer", {})
        if dec:
            summary_bits.append(
                f"Decision: action={dec.get('action')} size={dec.get('position_size_pct')}% "
                f"stop={dec.get('risk_management',{}).get('stop_loss_pct')}% "
                f"tp={dec.get('risk_management',{}).get('take_profit_pct')}%"
            )
            r = dec.get("rationale_bullets") or []
            if r:
                summary_bits.append("Rationales: " + " | ".join(r[:4]))
        # Forecast
        fc = (ctx.get("forecast") or {}).get("answer", {})
        if fc:
            bt = fc.get("backtest") or {}
            summary_bits.append(
                f"Forecast backtest: RMSE={bt.get('RMSE')} MAPE={bt.get('MAPE')} sMAPE={bt.get('sMAPE')} "
                "(lower is better)."
            )
            fclist = fc.get("forecast") or []
            if fclist:
                last = fclist[-1]
                summary_bits.append(
                    f"Next-horizon return: mean={last.get('mean')} pi95={last.get('pi95')} pi80={last.get('pi80')}"
                )
        # Sentiment
        sent = (ctx.get("sentiment") or {}).get("answer", {}).get("overall")
        if sent:
            summary_bits.append(f"Sentiment: {sent.get('label')} ({sent.get('score')}).")
        # Anomalies
        an = (ctx.get("anomalies") or {}).get("answer", {})
        if an and an.get("flags"):
            labels = [f.get("label") for f in an["flags"]]
            summary_bits.append("Risk flags: " + ", ".join(labels))
        # QA
        qa = (ctx.get("qa") or {}).get("answer", {})
        if qa:
            summary_bits.append(f"QA answer: {qa.get('value')}")
    except Exception:
        pass

    run_summary = "\n".join(summary_bits) if summary_bits else "No run summary available yet."

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Use the following CONTEXT to answer the question.\n\n"
                f"CONTEXT.framework:\n{framework}\n\n"
                f"CONTEXT.run_summary:\n{run_summary}\n\n"
                f"CONTEXT.full_json:\n{json.dumps(ctx, indent=2)}\n\n"
                f"USER QUESTION:\n{user_message}"
            ),
        },
    ]

    client = get_client()
    # Use Chat Completions for broad SDK compatibility
    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()
