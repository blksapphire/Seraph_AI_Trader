import numpy as np
def evaluate(df,period=14,oversold=30,overbought=70):
    d=df.close.diff(); g=d.clip(lower=0).rolling(period).mean(); l=(-d.clip(upper=0)).rolling(period).mean(); r=100-100/(1+g/(l+1e-9))
    if len(df)<period+2 or not np.isfinite(r.iloc[-1]):return {"score":0.0,"confidence":0.0,"signal":"HOLD","narrative":"Insufficient RSI history"}
    score=1 if r.iloc[-2]<oversold<=r.iloc[-1] else -1 if r.iloc[-2]>overbought>=r.iloc[-1] else 0
    return {"score":float(score),"confidence":float(abs(score)),"signal":"BUY" if score>0 else "SELL" if score<0 else "HOLD","narrative":f"RSI={r.iloc[-1]:.1f}"}
