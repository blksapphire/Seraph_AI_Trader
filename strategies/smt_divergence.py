import numpy as np
def evaluate(df,corr_df,lookback=30):
    if len(df)<lookback or len(corr_df)<lookback:return {"score":0.0,"confidence":0.0,"signal":"HOLD","narrative":"Insufficient SMT history"}
    a=df.tail(lookback); b=corr_df.tail(lookback)
    score=0
    if a.high.iloc[-1]>=a.high.max() and b.low.iloc[-1]>b.low.min(): score=-1
    elif a.low.iloc[-1]<=a.low.min() and b.high.iloc[-1]<b.high.max(): score=1
    return {"score":float(score),"confidence":float(abs(score)),"signal":"BUY" if score>0 else "SELL" if score<0 else "HOLD","narrative":"SMT divergence" if score else "No SMT divergence"}
