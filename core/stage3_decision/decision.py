from __future__ import annotations
from typing import Dict, Any, List
from datetime import date, timedelta
from core.schemas import Stage3Decision, Stage3DecisionAnswer, RiskManagement, Evidence

def decide(signals: Dict[str, Any], policy: Dict[str, Any] | None = None) -> Stage3Decision:
    policy = policy or {"max_position_pct":5, "stop_loss_pct":8, "take_profit_pct":15, "min_confidence":"medium"}

    mean_return = signals.get("forecast_return_next_h",{}).get("mean", 0.0)
    pi95 = signals.get("forecast_return_next_h",{}).get("pi95", [None, None])
    lo95, hi95 = pi95[0], pi95[1]
    risk_flags: List[str] = signals.get("risk_flags", [])
    sent = signals.get("sentiment_overall")  # {'label': 'positive|neutral|negative', 'score': float in [-1,1]} or None

    action = "HOLD"
    rationales: List[str] = []

    if mean_return is not None and lo95 is not None and hi95 is not None:
        if mean_return > 0 and lo95 > 0 and not risk_flags:
            action = "BUY"
            rationales.append("Forecast distribution positive: mean > 0 and 95% lower bound > 0.")
        elif mean_return < 0 and hi95 < 0:
            action = "SELL"
            rationales.append("Forecast distribution negative: mean < 0 and 95% upper bound < 0.")
        else:
            action = "HOLD"
            rationales.append("Forecast not decisive at 95% confidence or risk flags present.")

    # Position sizing: start from max, reduce for risks & negative sentiment
    pos = float(policy["max_position_pct"])
    if risk_flags:
        pos -= 1.0
        rationales.append(f"Reduced size due to risk flags: {', '.join(risk_flags)}.")
    if sent and isinstance(sent, dict):
        s_score = float(sent.get("score", 0.0))
        s_label = str(sent.get("label", "neutral"))
        rationales.append(f"Sentiment signal: {s_label} ({s_score:+.2f}).")
        if s_score < -0.2:
            pos -= 1.0
            rationales.append("Further reduced size due to negative sentiment.")
        elif s_score > 0.2 and action == "BUY":
            rationales.append("Positive sentiment aligns with BUY signal.")

    # Clamp position size
    pos = max(1.0, min(pos, float(policy["max_position_pct"])))

    rm = RiskManagement(
        stop_loss_pct=float(policy["stop_loss_pct"]),
        take_profit_pct=float(policy["take_profit_pct"]),
        review_date=(date.today() + timedelta(days=14)).isoformat()
    )

    ans = Stage3DecisionAnswer(
        action=action,
        position_size_pct=pos,
        entry="market",
        risk_management=rm,
        rationale_bullets=rationales if rationales else ["No strong signals; default HOLD."]
    )

    return Stage3Decision(
        answer=ans,
        evidence=[Evidence(source="stage2_forecast", snippet="Signal summary")],
        confidence="medium",
        notes_short="Educational demo; not financial advice.",
        viz_spec={"type":"badge_action","fields":{"action":"answer.action"}}
    )
