from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from core.schemas import Stage2Forecast, Stage2ForecastAnswer, Evidence, ForecastPoint

def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError('No data fetched. Check ticker or date range.')
    df = df[['Close']].rename(columns={'Close':'close'})
    df.index = pd.to_datetime(df.index)
    df['return'] = df['close'].pct_change().fillna(0.0)
    return df

def rolling_backtest(series: pd.Series, order=(1,0,0), folds: int = 5, horizon: int = 5) -> Dict[str, float]:
    if len(series) < folds * horizon + 20:
        folds = max(1, min(folds, (len(series)-20)//horizon))
    preds, trues = [], []
    for k in range(folds):
        split = len(series) - (folds-k)*horizon
        train, test = series.iloc[:split], series.iloc[split: split+horizon]
        if len(test) == 0 or len(train) < 30:
            continue
        model = ARIMA(train, order=order)
        fit = model.fit()
        fc = fit.forecast(steps=len(test))
        preds.extend(fc.values.tolist())
        trues.extend(test.values.tolist())
    if not trues:
        return {'MAPE': float('nan'), 'sMAPE': float('nan'), 'RMSE': float('nan')}
    trues = np.array(trues); preds = np.array(preds)
    eps = 1e-8
    mape = float(np.mean(np.abs((trues - preds) / (np.abs(trues)+eps)))*100.0)
    smape = float(np.mean(2*np.abs(preds - trues) / (np.abs(trues)+np.abs(preds)+eps))*100.0)
    rmse = float(np.sqrt(np.mean((preds - trues)**2)))
    return {'MAPE': round(mape,3), 'sMAPE': round(smape,3), 'RMSE': round(rmse,6)}

def forecast_returns(ticker: str, start: str, end: str, horizon: int = 5) -> Stage2Forecast:
    df = fetch_prices(ticker, start, end)
    series = df['return']
    order = (1,0,0)
    model = ARIMA(series, order=order)
    fit = model.fit()
    fc = fit.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)
    conf80 = fc.conf_int(alpha=0.20)
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='B')
    points: List[ForecastPoint] = []
    for i, d in enumerate(idx):
        l95, u95 = conf.iloc[i,0], conf.iloc[i,1]
        l80, u80 = conf80.iloc[i,0], conf80.iloc[i,1]
        points.append(ForecastPoint(date=d.strftime('%Y-%m-%d'), mean=float(mean.iloc[i]), pi80=(float(l80), float(u80)), pi95=(float(l95), float(u95))))
    bt = rolling_backtest(series, order=order, folds=5, horizon=min(5, horizon))
    ev = [Evidence(source='market_api', snippet=f'yfinance {ticker} from {start} to {end}', location=None)]
    return Stage2Forecast(answer=Stage2ForecastAnswer(forecast=points, model={'family':'ARIMA','order':order,'features':[]}, backtest=bt), evidence=ev, confidence='medium', notes_short='ARIMA on daily returns; add exogenous features later.', viz_spec={'type':'line_forecast','fields':{'date':'date','y':['mean','pi80','pi95']}})
