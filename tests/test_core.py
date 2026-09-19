import json
from modules.decision_engine import DecisionEngine
from modules.technical_analyzer import TechnicalAnalyzer

def config():
    return {
        "brains": {"technical_weight":.35,"structural_weight":.35,"fundamental_weight":.15,"regime_weight":.15,"entry_threshold":.58,"agreement_threshold":.5},
        "technical": {"lookback":60,"model_path":"x","scaler_path":"y","feature_path":"z"},
        "system_identity":{"name":"test"},
    }

def test_decision_hold_on_disagreement():
    d=DecisionEngine(config()).evaluate("EURUSD","M15",
        {"score":1},{"score":-1},{"score":0},{"score":0})
    assert d.action=="HOLD"

def test_features_have_expected_columns():
    import pandas as pd, numpy as np
    n=100
    c=np.linspace(100,110,n)+np.sin(np.arange(n))
    df=pd.DataFrame({"open":c,"high":c+1,"low":c-1,"close":c,"tick_volume":np.ones(n)*100})
    x=TechnicalAnalyzer.calculate_features(df)
    for col in ["atr","rsi","macd","bb_position","fvg","order_block"]:
        assert col in x
