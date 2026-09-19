import argparse,json
import numpy as np
from modules.mt5_client import MT5Client
from modules.technical_analyzer import TechnicalAnalyzer
from modules.structural_analyzer import StructuralAnalyzer
from modules.decision_engine import DecisionEngine

class Replay:
    def __init__(self,cfg):
        self.cfg=cfg; self.client=MT5Client(cfg); self.tech=TechnicalAnalyzer(cfg); self.struct=StructuralAnalyzer(cfg); self.engine=DecisionEngine(cfg)

    def run(self,symbol,timeframe,bars=5000):
        self.client.connect()
        try: df=self.client.rates(symbol,timeframe,bars)
        finally: self.client.close()
        warm=max(self.cfg["technical"]["lookback"],60); trades=[]; position=None
        for i in range(warm,len(df)-1):
            window=df.iloc[:i+1]; t=self.tech.analyze(window); s=self.struct.analyze(window); d=self.engine.evaluate(symbol,timeframe,t,s,{"score":0.0},{"score":self.struct.analyze(window)["score"]})
            if position is None and d.action!="HOLD":
                atr=float(self.tech.calculate_features(window)["atr"].iloc[-1]); entry=float(df["close"].iloc[i])
                stop=entry-(atr*1.8 if d.action=="BUY" else -atr*1.8); target=entry+(atr*2.7 if d.action=="BUY" else -atr*2.7)
                position={"side":d.action,"entry":entry,"stop":stop,"target":target,"opened":df.index[i]}
            elif position is not None:
                hi,lo=float(df["high"].iloc[i+1]),float(df["low"].iloc[i+1]); exit_price=None
                if position["side"]=="BUY":
                    if lo<=position["stop"]: exit_price=position["stop"]
                    elif hi>=position["target"]: exit_price=position["target"]
                else:
                    if hi>=position["stop"]: exit_price=position["stop"]
                    elif lo<=position["target"]: exit_price=position["target"]
                if exit_price is not None:
                    pnl=(exit_price-position["entry"])*(1 if position["side"]=="BUY" else -1)
                    trades.append({"side":position["side"],"entry":position["entry"],"exit":exit_price,"pnl":pnl,"opened":str(position["opened"]),"closed":str(df.index[i+1])}); position=None
        wins=sum(t["pnl"]>0 for t in trades)
        return {"symbol":symbol,"timeframe":timeframe,"trades":len(trades),"wins":wins,"win_rate":wins/len(trades) if trades else 0,"net_price_move":sum(t["pnl"] for t in trades),"trades_detail":trades}

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--symbol",default="XAUUSD"); p.add_argument("--timeframe",default="M15"); p.add_argument("--bars",type=int,default=5000); a=p.parse_args()
    with open("config.json") as f: cfg=json.load(f)
    print(json.dumps(Replay(cfg).run(a.symbol,a.timeframe,a.bars),indent=2))
