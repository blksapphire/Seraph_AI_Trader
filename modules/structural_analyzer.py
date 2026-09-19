import numpy as np
import pandas as pd

class StructuralAnalyzer:
    def __init__(self,config):
        p=config.get("structural_parameters",{})
        self.lookback=int(p.get("swing_point_lookback",20)); self.atr_mult=float(p.get("bos_choch_threshold_atr",1.2))

    def analyze(self,df):
        x=df.copy(); tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1); x["atr"]=tr.rolling(14).mean()
        if len(x)<self.lookback+20:return {"score":0.0,"confidence":0.0,"narrative":"Insufficient structure data","events":[]}
        c=x.iloc[-1]; atr=float(x.atr.iloc[-1]); prev=x.iloc[-self.lookback-1:-1]; hi=float(prev.high.max()); lo=float(prev.low.min()); score=0.; events=[]
        if c.high>hi and c.close<hi: score-=.60; events.append("buy-side liquidity sweep")
        if c.low<lo and c.close>lo: score+=.60; events.append("sell-side liquidity sweep")
        if c.close>hi+self.atr_mult*atr: score+=1.; events.append("bullish BOS")
        elif c.close<lo-self.atr_mult*atr: score-=1.; events.append("bearish BOS")
        # Equal-high/low proxy and displacement.
        recent=x.iloc[-5:]; recent_hi=float(recent.high.max()); recent_lo=float(recent.low.min())
        body=abs(c.close-c.open)
        if body>atr*1.2:
            score += .25 if c.close>c.open else -.25; events.append("bullish displacement" if c.close>c.open else "bearish displacement")
        # Swing trend confirmation.
        fast=float(x.close.rolling(10).mean().iloc[-1]); slow=float(x.close.rolling(30).mean().iloc[-1])
        trend=float(np.tanh((fast-slow)/(atr+1e-9)))
        score=.70*float(np.clip(score,-1,1))+.30*trend
        if not events: events.append("range / no structural break")
        score=float(np.clip(score,-1,1))
        return {"score":score,"confidence":min(1,abs(score)),"narrative":", ".join(events), "events":events, "swing_high":hi, "swing_low":lo, "trend":trend}
