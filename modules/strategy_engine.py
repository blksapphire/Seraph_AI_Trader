import numpy as np
import pandas as pd
from modules.wyckoff_analyzer import WyckoffAnalyzer

class StrategyEngine:
    """Unified deterministic ensemble. Every component is normalized to [-1, 1]."""
    def __init__(self,config):
        self.cfg=config
        default={"trend":.16,"momentum":.12,"mean_reversion":.10,"breakout":.12,"vwap":.08,"candlestick":.07,"volatility":.04,"ma_crossover":.08,"rsi_momentum":.07,"wyckoff":.16}
        self.weights=config.get("strategies",{}).get("weights",default)
        total=sum(max(0,float(v)) for v in self.weights.values()) or 1
        self.weights={k:max(0,float(v))/total for k,v in self.weights.items()}
        self.wyckoff=WyckoffAnalyzer(config)
    @staticmethod
    def _atr(x):
        tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1)
        return tr.rolling(14).mean()
    def evaluate(self,df):
        x=df.copy(); c=x.close
        if len(x)<60:return {"score":0.0,"confidence":0.0,"agreement":0.0,"components":{},"narrative":"Insufficient strategy history"}
        atr=self._atr(x); a=float(atr.iloc[-1])
        if not np.isfinite(a) or a<=0:return {"score":0.0,"confidence":0.0,"agreement":0.0,"components":{},"narrative":"Invalid ATR"}
        ema20=c.ewm(span=20,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean()
        trend=float(np.tanh((ema20.iloc[-1]-ema50.iloc[-1])/(a*1.5)))
        macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); sig=macd.ewm(span=9,adjust=False).mean()
        momentum=float(np.tanh((macd.iloc[-1]-sig.iloc[-1])/(a/2+1e-9)))
        mid=c.rolling(20).mean(); std=c.rolling(20).std(); z=(c.iloc[-1]-mid.iloc[-1])/(std.iloc[-1]+1e-9)
        mean_rev=float(np.clip(-z/2,-1,1)) if abs(z)<2.5 else float(-np.sign(z))
        hi=x.high.iloc[-21:-1].max(); lo=x.low.iloc[-21:-1].min()
        breakout=1.0 if c.iloc[-1]>hi else -1.0 if c.iloc[-1]<lo else float(np.clip((c.iloc[-1]-(hi+lo)/2)/(a*2+1e-9),-1,1))
        typical=(x.high+x.low+x.close)/3; v=x.tick_volume.replace(0,np.nan).fillna(1)
        vwap=(typical*v).rolling(50).sum()/v.rolling(50).sum(); vwap_score=float(np.tanh((c.iloc[-1]-vwap.iloc[-1])/(a+1e-9)))
        last=x.iloc[-1]; body=abs(last.close-last.open); rng=max(last.high-last.low,1e-9); upper=last.high-max(last.open,last.close); lower=min(last.open,last.close)-last.low
        candle=.65 if body/rng<.25 and lower/rng>.55 else -.65 if body/rng<.25 and upper/rng>.55 else .45 if last.close>last.open and body/rng>.65 else -.45 if last.close<last.open and body/rng>.65 else 0.0
        atr_slow=float(atr.rolling(50).mean().iloc[-1]); volatility=float(np.clip((a/(atr_slow+1e-9)-1)*3,-1,1))
        ma_s=c.rolling(20).mean(); ma_l=c.rolling(50).mean()
        ma=1.0 if ma_s.iloc[-2]<=ma_l.iloc[-2] and ma_s.iloc[-1]>ma_l.iloc[-1] else -1.0 if ma_s.iloc[-2]>=ma_l.iloc[-2] and ma_s.iloc[-1]<ma_l.iloc[-1] else float(np.tanh((ma_s.iloc[-1]-ma_l.iloc[-1])/(a+1e-9)))
        delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean(); rsi=100-100/(1+gain/(loss+1e-9))
        rsi_score=1.0 if rsi.iloc[-2]<30<=rsi.iloc[-1] else -1.0 if rsi.iloc[-2]>70>=rsi.iloc[-1] else float(np.clip((50-rsi.iloc[-1])/25,-1,1))
        wy=self.wyckoff.analyze(x)
        components={"trend":trend,"momentum":momentum,"mean_reversion":mean_rev,"breakout":breakout,"vwap":vwap_score,"candlestick":candle,"volatility":volatility,"ma_crossover":ma,"rsi_momentum":rsi_score,"wyckoff":wy["score"]}
        score=float(np.clip(sum(self.weights.get(k,0)*v for k,v in components.items()),-1,1))
        active=[v for k,v in components.items() if abs(v)>=.2 and k!="volatility"]; agreement=sum(1 for v in active if v*score>0)/len(active) if active else 0
        return {"score":score,"confidence":min(1,abs(score)*.7+agreement*.2+wy.get("confidence",0)*.1),"agreement":agreement,"components":components,"wyckoff":wy,"narrative":" | ".join(f"{k}={v:+.2f}" for k,v in components.items())}
