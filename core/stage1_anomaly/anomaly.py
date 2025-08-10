from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from core.schemas import Stage1Anomaly, Stage1AnomalyAnswer, AnomalyFlag

def compute_anomalies(metric_series: List[Dict[str, float]], z_threshold: float = 2.0) -> Stage1Anomaly:
    if len(metric_series) < 4:
        return Stage1Anomaly(answer=Stage1AnomalyAnswer(flags=[]), evidence=[], confidence='low', notes_short='Need at least 4 periods.')
    values = np.array([p['value'] for p in metric_series], dtype=float)
    periods = [p['period'] for p in metric_series]
    pct = np.zeros_like(values)
    pct[1:] = (values[1:] - values[:-1]) / np.where(values[:-1]==0, 1, values[:-1])
    std = pct.std() if pct.std() != 0 else 1.0
    z = (pct - pct.mean()) / std
    flags = []
    for i in range(len(values)):
        if abs(z[i]) >= z_threshold:
            label = 'opportunity' if pct[i] > 0 else 'risk'
            flags.append(AnomalyFlag(period=periods[i], delta_pct=float(pct[i]), z=float(z[i]), label=label))
    return Stage1Anomaly(answer=Stage1AnomalyAnswer(flags=flags), evidence=[], confidence='medium' if flags else 'low', notes_short='Z-score on pct change vs series.')
