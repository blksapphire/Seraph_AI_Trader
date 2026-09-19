import numpy as np
import pandas as pd

class StrategyEngine:
    """Independent, explainable price-action strategies. Each returns [-1, 1]."""
    def __init__(self, config):
        self.weights=config.get("strategies",{}).get("weights",{"trend":.20,"momentum":.15,"mean_reversion":.15,"breakout":.15,"vwap":.10,"candlestick":.10,"volatility":.05})
        total=sum(float(v) for v in self.weights.values()) or 1
        self.weights={k:float(v)/total for k,v in self.weights.items()}

    @staticmethod
    def _atr(x):
        tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1)
        return tr.rolling(14).mean()

    def evaluate(self,df):
        x=df.copy(); c=x.close
        if len(x)<60:return {"score":0.0,"confidence":0.0,"components":{},"narrative":"Insufficient strategy history"}
        atr=self._atr(x); a=float(atr.iloc[-1])
        if not np.isfinite(a) or a<=0:return {"score":0.0,"confidence":0.0,"components":{},"narrative":"Invalid ATR"}
        ema20=c.ewm(span=20,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean()
        trend=float(np.tanh((ema20.iloc[-1]-ema50.iloc[-1])/(a*1.5)))
        macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); signal=macd.ewm(span=9,adjust=False).mean()
        momentum=float(np.tanh((macd.iloc[-1]-signal.iloc[-1])/(a/2+1e-9)))
        gain=c.diff().clip(lower=0).rolling(14).mean(); loss=(-c.diff().clip(upper=0)).rolling(14).mean()
        rsi=100-(100/(1+gain/(loss+1e-9)))
        mid=c.rolling(20).mean(); std=c.rolling(20).std(); z=(c.iloc[-1]-mid.iloc[-1])/(std.iloc[-1]+1e-9)
        mean_rev=float(np.clip(-z/2,-1,1)) if abs(z)<2.5 else float(-np.sign(z))
        hi=x.high.iloc[-21:-1].max(); lo=x.low.iloc[-21:-1].min()
        breakout=1.0 if c.iloc[-1]>hi else -1.0 if c.iloc[-1]<lo else float(np.clip((c.iloc[-1]-(hi+lo)/2)/(a*2+1e-9),-1,1))
        typical=(x.high+x.low+x.close)/3; v=x.tick_volume.replace(0,np.nan).fillna(1)
        vwap=(typical*v).rolling(50).sum()/v.rolling(50).sum()
        vwap_score=float(np.tanh((c.iloc[-1]-vwap.iloc[-1])/(a+1e-9)))
        last=x.iloc[-1]; body=abs(last.close-last.open); rng=max(last.high-last.low,1e-9); upper=last.high-max(last.open,last.close); lower=min(last.open,last.close)-last.low
        candle=0.0
        if body/rng<.25 and lower/rng>.55:candle=.65
        elif body/rng<.25 and upper/rng>.55:candle=-.65
        elif last.close>last.open and body/rng>.65:candle=.45
        elif last.close<last.open and body/rng>.65:candle=-.45
        atr_slow=float(atr.rolling(50).mean().iloc[-1]); volatility=float(np.clip((a/(atr_slow+1e-9)-1)*3,-1,1))
        components={"trend":trend,"momentum":momentum,"mean_reversion":mean_rev,"breakout":breakout,"vwap":vwap_score,"candlestick":candle,"volatility":volatility}
        score=float(np.clip(sum(self.weights.get(k,0)*v for k,v in components.items()),-1,1))
        active=[v for k,v in components.items() if k!="volatility" and abs(v)>=.2]; agreement=sum(1 for v in active if v*score>0)/len(active) if active else 0
        return {"score":score,"confidence":min(1,abs(score)*.75+agreement*.25),"agreement":agreement,"components":components,"narrative":" | ".join(f"{k}={v:+.2f}" for k,v in components.items())}
