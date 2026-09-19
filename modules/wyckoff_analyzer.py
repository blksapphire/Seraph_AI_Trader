import numpy as np
import pandas as pd

class WyckoffAnalyzer:
    """Lightweight, dependency-free Wyckoff phase/event detector."""
    def __init__(self, config):
        self.cfg=config.get("wyckoff",{})
        self.lookback=int(self.cfg.get("range_lookback",40))
        self.volume_period=int(self.cfg.get("volume_period",20))
        self.spike=float(self.cfg.get("volume_spike_factor",1.5))

    def analyze(self, df):
        if len(df)<max(30,self.lookback+2):
            return {"score":0.0,"confidence":0.0,"state":"unknown","phase":"None","events":[],"narrative":"Insufficient Wyckoff history"}
        x=df.copy()
        vol=x.get("tick_volume",x.get("volume",pd.Series(1,index=x.index))).astype(float)
        vma=vol.rolling(self.volume_period).mean()
        recent=x.tail(self.lookback); high=float(recent.high.max()); low=float(recent.low.min()); close=float(x.close.iloc[-1])
        prev=x.close.iloc[-2]; last=x.iloc[-1]; avg=float(vma.iloc[-1]) if np.isfinite(vma.iloc[-1]) else float(vol.tail(self.volume_period).mean())
        events=[]; score=0.0
        if last.low < low and last.close > low and last.tick_volume < avg if "tick_volume" in last.index else False:
            events.append("spring"); score += .8
        if last.high > high and last.close < high and last.tick_volume < avg if "tick_volume" in last.index else False:
            events.append("upthrust"); score -= .8
        if close>high and vol.iloc[-1]>avg*self.spike: events.append("SOS"); score += 1.0
        if close<low and vol.iloc[-1]>avg*self.spike: events.append("SOW"); score -= 1.0
        slope=float(np.polyfit(np.arange(min(20,len(x))),x.close.tail(min(20,len(x))),1)[0])
        atr=float((x.high-x.low).rolling(14).mean().iloc[-1] or 0)
        trend=np.tanh(slope/(atr/10+1e-9))
        score=float(np.clip(.65*score+.35*trend,-1,1))
        width=max(high-low,1e-9); pos=(close-low)/width
        phase="accumulation" if pos<.35 and score>=0 else "distribution" if pos>.65 and score<=0 else "markup" if score>.35 else "markdown" if score<-.35 else "range"
        state=events[-1] if events else phase
        confidence=float(min(1,.45*abs(score)+.55*min(1,len(events)/2)))
        return {"score":score,"confidence":confidence,"state":state,"phase":phase,"events":events,"narrative":f"Wyckoff {phase}; events={','.join(events) if events else 'none'}; score={score:+.2f}"}
