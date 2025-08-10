from core.schemas import Stage1QA, Stage1QAAnswer, Evidence, Stage1Anomaly, Stage1AnomalyAnswer, Stage3Decision, Stage3DecisionAnswer, RiskManagement
def test_stage1qa_schema():
    qa = Stage1QA(answer=Stage1QAAnswer(value='42'), evidence=[Evidence(source='doc1', snippet='answer 42', location='p1')], confidence='medium', notes_short='ok')
    assert qa.task == 'stage1_qa' and qa.answer.value == '42'
def test_anomaly_empty():
    an = Stage1Anomaly(answer=Stage1AnomalyAnswer(flags=[]), evidence=[], confidence='low', notes_short='')
    assert an.task == 'stage1_anomaly'
def test_decision_schema():
    rm = RiskManagement(stop_loss_pct=8, take_profit_pct=15, review_date='2025-08-01')
    dec = Stage3Decision(answer=Stage3DecisionAnswer(action='HOLD', position_size_pct=5, entry='market', risk_management=rm, rationale_bullets=['x']), evidence=[], confidence='medium', notes_short='demo')
    assert dec.task == 'stage3_decision'
