import numpy as np
import pandas as pd
from modules.decision_engine import DecisionEngine
from modules.technical_analyzer import TechnicalAnalyzer
from modules.strategy_engine import StrategyEngine
from modules.wyckoff_analyzer import WyckoffAnalyzer
from modules.trade_database import TradeDatabase

def cfg():
    return {"brains":{"technical_weight":.4,"structural_weight":.3,"fundamental_weight":.1,"regime_weight":.2,"entry_threshold":.58,"agreement_threshold":.55,"max_fundamental_disagreement":.65},"technical":{"lookback":60,"model_path":"x","scaler_path":"y","feature_path":"z"},"strategies":{"weights":{"trend":.16,"momentum":.12,"mean_reversion":.1,"breakout":.12,"vwap":.08,"candlestick":.07,"volatility":.04,"ma_crossover":.08,"rsi_momentum":.07,"wyckoff":.16}},"wyckoff":{"range_lookback":40,"volume_period":20,"volume_spike_factor":1.5},"database":{"path":"runtime/test_seraph.db"}}

def sample(n=180):
    t=np.arange(n); c=100+0.04*t+np.sin(t/5)
    return pd.DataFrame({"open":c-.2,"high":c+.7,"low":c-.7,"close":c,"tick_volume":100+10*np.sin(t/7)})

def test_disagreement_holds():
    assert DecisionEngine(cfg()).evaluate("EURUSD","M15",{"score":1},{"score":-1},{"score":0},{"score":0}).action=="HOLD"

def test_features_and_r_strategy_ensemble():
    result=StrategyEngine(cfg()).evaluate(sample())
    assert -1<=result["score"]<=1
    assert set(result["components"])>= {"ma_crossover","rsi_momentum","wyckoff"}

def test_wyckoff_output():
    result=WyckoffAnalyzer(cfg()).analyze(sample())
    assert -1<=result["score"]<=1 and "phase" in result

def test_technical_fallback():
    result=TechnicalAnalyzer(cfg()).analyze(sample())
    assert -1<=result["score"]<=1 and result["regime"] in {"trend","breakout","range"}

def test_sqlite_trade_memory(tmp_path):
    c=cfg(); c["database"]["path"]=str(tmp_path/"seraph.db"); db=TradeDatabase(c); db.event("EURUSD","test",{"ok":1}); assert len(db.recent())==0
