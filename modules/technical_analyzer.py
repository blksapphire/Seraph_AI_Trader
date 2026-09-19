import json, logging, os
import numpy as np
import pandas as pd

FEATURES=["return_1","return_3","atr","rsi","macd","macd_signal","macd_hist","bb_position","ema_fast_dist","ema_slow_dist","volume_z","range_pct","fvg","order_block"]

class TechnicalAnalyzer:
    def __init__(self,config):
        self.cfg=config; self.model=None; self.scaler=None; self.features=FEATURES.copy(); self.load_model()

    @staticmethod
    def calculate_features(df):
        x=df.copy(); c=x["close"]
        x["return_1"]=c.pct_change(); x["return_3"]=c.pct_change(3)
        tr=pd.concat([x["high"]-x["low"],(x["high"]-c.shift()).abs(),(x["low"]-c.shift()).abs()],axis=1).max(axis=1); x["atr"]=tr.rolling(14).mean()
        delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean(); x["rsi"]=100-(100/(1+gain/(loss+1e-9)))
        e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); x["macd"]=e12-e26; x["macd_signal"]=x["macd"].ewm(span=9,adjust=False).mean(); x["macd_hist"]=x["macd"]-x["macd_signal"]
        mid,std=c.rolling(20).mean(),c.rolling(20).std(); x["bb_position"]=(c-(mid-2*std))/(4*std+1e-9)
        x["ema_fast_dist"]=c/e12-1; x["ema_slow_dist"]=c/e26-1
        vm,vs=x["tick_volume"].rolling(30).mean(),x["tick_volume"].rolling(30).std(); x["volume_z"]=(x["tick_volume"]-vm)/(vs+1e-9)
        x["range_pct"]=(x["high"]-x["low"])/c
        x["fvg"]=0.0; x.loc[x["high"].shift(2)<x["low"],"fvg"]=1.0; x.loc[x["low"].shift(2)>x["high"],"fvg"]=-1.0
        body=(x["close"]-x["open"]).abs(); avg=body.rolling(20).mean(); x["order_block"]=0.0
        x.loc[(x["close"].shift()<x["open"].shift())&((x["close"]-x["open"])>1.5*avg),"order_block"]=1.0
        x.loc[(x["close"].shift()>x["open"].shift())&((x["open"]-x["close"])>1.5*avg),"order_block"]=-1.0
        return x

    def load_model(self):
        try:
            import joblib
            p=self.cfg["technical"]["model_path"]; s=self.cfg["technical"]["scaler_path"]
            if os.path.exists(p) and os.path.exists(s):
                self.model=joblib.load(p); self.scaler=joblib.load(s)
                f=self.cfg["technical"]["feature_path"]
                if os.path.exists(f):
                    with open(f) as fh:self.features=json.load(fh)
                logging.info("Technical model loaded")
        except Exception as exc:logging.warning("Technical model unavailable: %s",exc)

    def analyze(self,df):
        x=self.calculate_features(df).replace([np.inf,-np.inf],np.nan).dropna()
        lookback=self.cfg["technical"]["lookback"]
        if len(x)<lookback:return {"score":0.0,"confidence":0.0,"narrative":"Insufficient technical history","regime":"unknown"}
        q=x.iloc[-1]; trend=float(np.tanh(q["ema_fast_dist"]*120+q["ema_slow_dist"]*60)); momentum=float(np.tanh(q["macd_hist"]/(q["atr"]+1e-9)*4)); rsi=float(np.clip((q["rsi"]-50)/25,-1,1))
        rule=float(np.clip(.45*trend+.35*momentum+.20*rsi,-1,1)); model_score=None
        if self.model is not None and self.scaler is not None:
            try:
                seq=x[self.features].tail(lookback); z=self.scaler.transform(seq)
                if hasattr(self.model,"predict_proba"):
                    raw=float(self.model.predict_proba(z)[-1,1])
                else:
                    raw=float(np.asarray(self.model.predict(z)).ravel()[0])
                model_score=2*raw-1
            except Exception as exc:logging.warning("Model prediction failed: %s",exc)
        score=rule if model_score is None else float(np.clip(.45*rule+.55*model_score,-1,1))
        regime="trend" if abs(trend)>.45 else "momentum" if abs(momentum)>.45 else "range"
        return {"score":score,"confidence":abs(score),"narrative":f"{regime} | RSI {q['rsi']:.1f} | MACD {momentum:+.2f}","regime":regime}
