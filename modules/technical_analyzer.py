import json, logging, os
import numpy as np
import pandas as pd
from modules.strategy_engine import StrategyEngine
FEATURES=["return_1","return_3","atr","rsi","macd","macd_signal","macd_hist","bb_position","ema_fast_dist","ema_slow_dist","volume_z","range_pct","fvg","order_block"]
class TechnicalAnalyzer:
    def __init__(self,config):
        self.cfg=config; self.model=None; self.scaler=None; self.features=FEATURES.copy(); self.strategies=StrategyEngine(config); self.rl=None; self.load_model()
        if config.get("rl",{}).get("enabled"):
            try:
                from learning_agent.agent import RLAgent
                self.rl=RLAgent(config["rl"].get("model_path"))
            except Exception as exc: logging.warning("RL unavailable: %s",exc)
    @staticmethod
    def calculate_features(df):
        x=df.copy(); c=x["close"]; x["return_1"]=c.pct_change(); x["return_3"]=c.pct_change(3)
        tr=pd.concat([x["high"]-x["low"],(x["high"]-c.shift()).abs(),(x["low"]-c.shift()).abs()],axis=1).max(axis=1); x["atr"]=tr.rolling(14).mean()
        delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean(); x["rsi"]=100-(100/(1+gain/(loss+1e-9)))
        e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); x["macd"]=e12-e26; x["macd_signal"]=x["macd"].ewm(span=9,adjust=False).mean(); x["macd_hist"]=x["macd"]-x["macd_signal"]
        mid,std=c.rolling(20).mean(),c.rolling(20).std(); x["bb_position"]=(c-(mid-2*std))/(4*std+1e-9); x["ema_fast_dist"]=c/e12-1; x["ema_slow_dist"]=c/e26-1
        vm,vs=x["tick_volume"].rolling(30).mean(),x["tick_volume"].rolling(30).std(); x["volume_z"]=(x["tick_volume"]-vm)/(vs+1e-9); x["range_pct"]=(x["high"]-x["low"])/c
        x["fvg"]=0.0; x.loc[x["high"].shift(2)<x["low"],"fvg"]=1.0; x.loc[x["low"].shift(2)>x["high"],"fvg"]=-1.0
        body=(x["close"]-x["open"]).abs(); avg=body.rolling(20).mean(); x["order_block"]=0.0
        x.loc[(x["close"].shift()<x["open"].shift())&((x["close"]-x["open"])>1.5*avg),"order_block"]=1.0; x.loc[(x["close"].shift()>x["open"].shift())&((x["open"]-x["close"])>1.5*avg),"order_block"]=-1.0
        return x
    def load_model(self):
        try:
            import joblib; p=self.cfg["technical"]["model_path"]; s=self.cfg["technical"]["scaler_path"]
            if os.path.exists(p) and os.path.exists(s):
                self.model=joblib.load(p); self.scaler=joblib.load(s); f=self.cfg["technical"]["feature_path"]
                if os.path.exists(f): self.features=json.load(open(f))
        except Exception as exc: logging.warning("Technical model unavailable: %s",exc)
    def analyze(self,df):
        x=self.calculate_features(df).replace([np.inf,-np.inf],np.nan).dropna(); lookback=self.cfg["technical"]["lookback"]
        if len(x)<lookback:return {"score":0.0,"confidence":0.0,"narrative":"Insufficient technical history","regime":"unknown","strategies":{}}
        strategy=self.strategies.evaluate(df); model_score=None
        if self.model is not None and self.scaler is not None:
            try:
                row=x[self.features].iloc[[-1]]; z=self.scaler.transform(row); raw=float(self.model.predict_proba(z)[0,1]) if hasattr(self.model,"predict_proba") else float(np.asarray(self.model.predict(z)).ravel()[0]); model_score=2*raw-1
            except Exception as exc: logging.warning("Technical ML prediction failed: %s",exc)
        score=strategy["score"] if model_score is None else float(np.clip(.55*strategy["score"]+.30*model_score+.15*strategy["wyckoff"]["score"],-1,1))
        if self.rl is not None:
            try:
                obs=self.calculate_features(df).replace([np.inf,-np.inf],np.nan).dropna().tail(1).iloc[0].to_numpy(dtype=np.float32); rl=self.rl.predict(obs); score=float(np.clip(.8*score+.2*rl["score"],-1,1))
            except Exception as exc: logging.warning("RL prediction failed: %s",exc)
        regime="trend" if abs(strategy["components"].get("trend",0))>.45 else "breakout" if abs(strategy["components"].get("breakout",0))>.65 else "range"
        return {"score":score,"confidence":min(1,abs(score)*.8+strategy.get("agreement",0)*.2),"narrative":strategy["narrative"],"regime":regime,"strategies":strategy["components"],"wyckoff":strategy["wyckoff"],"ml_score":model_score}
