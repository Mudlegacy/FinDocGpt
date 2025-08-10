from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional, Literal, Tuple, Dict, Any

class Evidence(BaseModel):
    source: str
    snippet: str
    location: Optional[str] = None

class Stage1QAAnswer(BaseModel):
    value: Optional[str] = None
    unit: Optional[str] = None
    period: Optional[str] = None
    table: Optional[List[Dict[str, Any]]] = None

class Stage1QA(BaseModel):
    task: Literal['stage1_qa'] = 'stage1_qa'
    answer: Stage1QAAnswer
    evidence: List[Evidence] = []
    confidence: Literal['low','medium','high'] = 'low'
    notes_short: str = ''
    viz_spec: Optional[Dict[str, Any]] = None

class SentimentDriver(BaseModel):
    phrase: str
    impact: Literal['+','-']

class Stage1SentimentSections(BaseModel):
    outlook: str = ''
    results: str = ''

class Stage1SentimentAnswer(BaseModel):
    overall: Dict[str, Any]
    drivers: List[SentimentDriver]
    sections: Stage1SentimentSections

class Stage1Sentiment(BaseModel):
    task: Literal['stage1_sentiment'] = 'stage1_sentiment'
    answer: Stage1SentimentAnswer
    evidence: List[Evidence] = []
    confidence: Literal['low','medium','high'] = 'low'
    notes_short: str = ''

class AnomalyFlag(BaseModel):
    period: str
    delta_pct: float
    z: float
    label: Literal['risk','opportunity']

class Stage1AnomalyAnswer(BaseModel):
    flags: List[AnomalyFlag]

class Stage1Anomaly(BaseModel):
    task: Literal['stage1_anomaly'] = 'stage1_anomaly'
    answer: Stage1AnomalyAnswer
    evidence: List[Evidence] = []
    confidence: Literal['low','medium','high'] = 'low'
    notes_short: str = ''

class ForecastPoint(BaseModel):
    date: str
    mean: float
    pi80: Tuple[float, float]
    pi95: Tuple[float, float]

class Stage2ForecastAnswer(BaseModel):
    forecast: List[ForecastPoint]
    model: Dict[str, Any]
    backtest: Dict[str, float]

class Stage2Forecast(BaseModel):
    task: Literal['stage2_forecast'] = 'stage2_forecast'
    answer: Stage2ForecastAnswer
    evidence: List[Evidence] = []
    confidence: Literal['low','medium','high'] = 'low'
    notes_short: str = ''
    viz_spec: Optional[Dict[str, Any]] = None

class RiskManagement(BaseModel):
    stop_loss_pct: float
    take_profit_pct: float
    review_date: str

class Stage3DecisionAnswer(BaseModel):
    action: Literal['BUY','SELL','HOLD']
    position_size_pct: float
    entry: str
    risk_management: RiskManagement
    rationale_bullets: List[str]

class Stage3Decision(BaseModel):
    task: Literal['stage3_decision'] = 'stage3_decision'
    answer: Stage3DecisionAnswer
    evidence: List[Evidence] = []
    confidence: Literal['low','medium','high'] = 'low'
    notes_short: str = ''
    viz_spec: Optional[Dict[str, Any]] = None
