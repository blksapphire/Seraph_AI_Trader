import numpy as np
import pandas as pd

class StructuralAnalyzer:
    def __init__(self, config):
        p=config.get("structural_parameters",{}); self.lookback=int(p.get("swing_point_lookback",20)); self.atr_mult=float(p.get("bos_choch_threshold_atr",1.2))

    def analyze(self,df):
        x=df.copy(); tr=pd.concat([x["high"]-x["low"],(x["high"]-x["close"].shift()).abs(),(x["low"]-x["close"].shift()).abs()],axis=1).max(axis=1); atr=tr.rolling(14).mean().iloc[-1]
        if not np.isfinite(atr) or len(x)<self.lookback+3: return {"score":0.0,"confidence":0.0,"narrative":"Insufficient structure data"}
        prev=x.iloc[-self.lookback-1:-1]; hi,lo=prev["high"].max(),prev["low"].min(); c=x.iloc[-1]; score=0.; events=[]
        if c["high"]>hi and c["close"]<hi: score-=.55; events.append("buy-side sweep")
        if c["low"]<lo and c["close"]>lo: score+=.55; events.append("sell-side sweep")
        if c["close"]>hi+self.atr_mult*atr: score+=1.; events.append("bullish BOS")
        elif c["close"]<lo-self.atr_mult*atr: score-=1.; events.append("bearish BOS")
        if not events: score=float(np.clip((c["close"]-x["close"].iloc[-10])/(atr*5+1e-9),-1,1)); events.append("structure neutral")
        score=float(np.clip(score,-1,1)); return {"score":score,"confidence":abs(score),"narrative":", ".join(events)}
