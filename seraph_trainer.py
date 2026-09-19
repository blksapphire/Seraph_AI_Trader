import json,os,logging
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from modules.mt5_client import MT5Client
from modules.technical_analyzer import TechnicalAnalyzer,FEATURES

class SeraphTrainer:
    def __init__(self,config_path="config.json"):
        with open(config_path) as f:self.config=json.load(f)
        self.client=MT5Client(self.config); self.tech=TechnicalAnalyzer(self.config)

    def train(self,symbol=None,timeframe=None):
        symbol=symbol or self.config["market"]["symbols"][0]; timeframe=timeframe or self.config["market"]["primary_timeframe"]; self.client.connect()
        try:df=self.client.rates(symbol,timeframe,self.config["training"]["historical_bars"])
        finally:self.client.close()
        x=self.tech.calculate_features(df).replace([np.inf,-np.inf],np.nan).dropna().copy(); horizon=int(self.config["training"]["horizon_bars"])
        x["target"]=(x["close"].shift(-horizon)>x["close"]).astype(int); x=x.iloc[:-horizon].dropna(subset=FEATURES+["target"])
        split=int(len(x)*(1-self.config["training"]["validation_fraction"])); Xtr,Xte=x[FEATURES].iloc[:split],x[FEATURES].iloc[split:]; ytr,yte=x.target.iloc[:split],x.target.iloc[split:]
        if len(Xtr)<self.config["technical"]["min_training_rows"] or ytr.nunique()<2:raise RuntimeError("Not enough diverse clean rows to train")
        scaler=StandardScaler(); Xtrz=scaler.fit_transform(Xtr); Xtez=scaler.transform(Xte)
        model=HistGradientBoostingClassifier(max_iter=250,max_depth=5,learning_rate=.05,l2_regularization=.2,random_state=42); model.fit(Xtrz,ytr)
        acc=model.score(Xtez,yte) if len(Xte) else 0.
        os.makedirs(os.path.dirname(self.config["technical"]["model_path"]) or ".",exist_ok=True); joblib.dump(model,self.config["technical"]["model_path"]); joblib.dump(scaler,self.config["technical"]["scaler_path"])
        with open(self.config["technical"]["feature_path"],"w") as f:json.dump(FEATURES,f)
        logging.info("Technical model trained: rows=%d chronological_holdout_accuracy=%.3f",len(x),acc); print(f"trained {symbol} {timeframe}: {len(x)} rows, holdout accuracy={acc:.3f}")

if __name__=="__main__":SeraphTrainer().train()
