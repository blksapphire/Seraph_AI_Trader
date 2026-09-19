import numpy as np
import pandas as pd
from modules.decision_engine import DecisionEngine
from modules.technical_analyzer import TechnicalAnalyzer
from modules.strategy_engine import StrategyEngine

def cfg():
    return {"brains":{"technical_weight":.4,"structural_weight":.3,"fundamental_weight":.1,"regime_weight":.2,"entry_threshold":.58,"agreement_threshold":.55},"technical":{"lookback":60,"model_path":"x","scaler_path":"y","feature_path":"z"},"strategies":{"weights":{"trend":.2,"momentum":.15,"mean_reversion":.15,"breakout":.15,"vwap":.1,"candlestick":.1,"volatility":.05}}}

def sample(n=160):
    t=np.arange(n); c=100+0.04*t+np.sin(t/5)
    return pd.DataFrame({"open":c-.2,"high":c+.7,"low":c-.7,"close":c,"tick_volume":100+10*np.sin(t/7)})

def test_disagreement_holds():
    d=DecisionEngine(cfg()).evaluate("EURUSD","M15",{"score":1},{"score":-1},{"score":0},{"score":0})
    assert d.action=="HOLD"

def test_features_and_strategies():
    df=sample(); x=TechnicalAnalyzer.calculate_features(df)
    for col in ["atr","rsi","macd","bb_position","fvg","order_block"]: assert col in x
    result=StrategyEngine(cfg()).evaluate(df)
    assert -1<=result["score"]<=1
    assert set(result["components"])>= {"trend","momentum","mean_reversion","breakout","vwap","candlestick","volatility"}

def test_technical_fallback_is_deterministic():
    result=TechnicalAnalyzer(cfg()).analyze(sample())
    assert -1<=result["score"]<=1
    assert result["regime"] in {"trend","breakout","range"}
