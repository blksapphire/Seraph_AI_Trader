import pandas as pd
def evaluate(df,short=20,long=50):
    s=df.close.rolling(short).mean(); l=df.close.rolling(long).mean()
    if len(df)<long+1:return {"score":0.0,"confidence":0.0,"signal":"HOLD","narrative":"Insufficient MA history"}
    cross=1 if s.iloc[-2]<=l.iloc[-2] and s.iloc[-1]>l.iloc[-1] else -1 if s.iloc[-2]>=l.iloc[-2] and s.iloc[-1]<l.iloc[-1] else 0
    return {"score":float(cross),"confidence":float(abs(cross)),"signal":"BUY" if cross>0 else "SELL" if cross<0 else "HOLD","narrative":"MA crossover"}
