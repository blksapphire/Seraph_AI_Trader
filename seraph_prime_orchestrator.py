import json,logging,os,time
import MetaTrader5 as mt5
from modules.mt5_client import MT5Client
from modules.technical_analyzer import TechnicalAnalyzer
from modules.structural_analyzer import StructuralAnalyzer
from modules.fundamental_analyzer import FundamentalAnalyzer
from modules.decision_engine import DecisionEngine
from modules.risk_engine import RiskEngine
from modules.state_store import StateStore
from modules.trade_database import TradeDatabase
from modules.notifier import Notifier

class SeraphPrime:
    def __init__(self,config_path="config.json"):
        self.config=json.load(open(config_path)); os.makedirs(os.path.dirname(self.config["runtime"]["log_file"]) or ".",exist_ok=True)
        logging.basicConfig(filename=self.config["runtime"]["log_file"],level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s")
        self.mt5=MT5Client(self.config); self.tech=TechnicalAnalyzer(self.config); self.struct=StructuralAnalyzer(self.config); self.fund=FundamentalAnalyzer(self.config)
        self.decision=DecisionEngine(self.config); self.risk=RiskEngine(self.config); self.state=StateStore(self.config); self.db=TradeDatabase(self.config); self.notify=Notifier(self.config); self.running=True
        self._last_signal={}
    def _atr(self,df): return float(self.tech.calculate_features(df)["atr"].iloc[-1])
    def _paper(self,symbol,decision,df):
        tick=self.mt5.tick(symbol); info=self.mt5.ensure_symbol(symbol); price=tick.ask if decision.action=="BUY" else tick.bid; sl,tp=self.risk.levels(decision.action,price,self._atr(df))
        return {"executed":False,"mode":"paper","action":decision.action,"entry":price,"sl":sl,"tp":tp,"reason":"paper mode - no order sent"}
    def _execute(self,symbol,decision,df):
        tick=self.mt5.tick(symbol); info=self.mt5.ensure_symbol(symbol); account=self.mt5.account(); positions=self.mt5.positions(); spread=(tick.ask-tick.bid)/info.point
        ok,reason=self.risk.allowed(account,positions,symbol,spread)
        if not ok:return {"executed":False,"mode":"live","reason":reason}
        price=tick.ask if decision.action=="BUY" else tick.bid; sl,tp=self.risk.levels(decision.action,price,self._atr(df)); volume=self.risk.volume(account,info,price,sl)
        request=self.risk.request(symbol,decision.action,price,sl,tp,volume); check=self.mt5.order_check(request)
        if check is None:return {"executed":False,"mode":"live","reason":"order_check returned None","mt5_error":str(mt5.last_error())}
        if getattr(check,"retcode",0) not in (0,mt5.TRADE_RETCODE_DONE): return {"executed":False,"mode":"live","reason":"order_check rejected","retcode":getattr(check,"retcode",None)}
        result=self.mt5.send(request); ok=getattr(result,"retcode",0) in (mt5.TRADE_RETCODE_DONE,mt5.TRADE_RETCODE_PLACED)
        return {"executed":ok,"mode":"live","retcode":getattr(result,"retcode",None),"deal":getattr(result,"deal",None),"sl":sl,"tp":tp,"volume":volume}
    def cycle(self,symbol):
        m15=self.mt5.rates(symbol,self.config["market"]["primary_timeframe"],self.config["market"]["bars"]); htf_df=self.mt5.rates(symbol,self.config["market"]["higher_timeframe"],max(500,self.config["market"]["bars"]//3))
        tech=self.tech.analyze(m15); struct=self.struct.analyze(m15); fund=self.fund.analyze(self.config["fundamental"]["currencies"]); htf=self.struct.analyze(htf_df)
        d=self.decision.evaluate(symbol,self.config["market"]["primary_timeframe"],tech,struct,fund,htf); key=f"{symbol}:{d.action}:{round(d.score,2)}"
        duplicate=key==self._last_signal.get(symbol); self._last_signal[symbol]=key
        if duplicate and d.action!="HOLD": execution={"executed":False,"mode":self.config.get("mode","paper"),"reason":"signal deduplicated"}
        elif d.action=="HOLD": execution={"executed":False,"mode":self.config.get("mode","paper"),"reason":"hold"}
        elif self.config.get("mode","paper")=="paper": execution=self._paper(symbol,d,m15)
        elif self.config.get("mode")=="live": execution=self._execute(symbol,d,m15)
        else: execution={"executed":False,"mode":self.config.get("mode"),"reason":"invalid mode"}
        payload={"status":"running","mode":self.config.get("mode","paper"),"symbol":symbol,"decision":self.decision.dict(d),"brains":{"technical":tech,"structural":struct,"fundamental":fund,"higher_timeframe":htf},"execution":execution}
        self.state.write_status(payload); self.state.journal(payload); self.db.event(symbol,"decision",payload)
        if execution.get("executed") or (d.action!="HOLD" and not duplicate): self.notify.send(f"{symbol} {d.action} score={d.score:+.2f} conf={d.confidence:.2f} | {d.rationale}","Seraph signal")
        return payload
    def run(self):
        self.mt5.connect()
        try:
            while self.running:
                for symbol in self.config["market"]["symbols"]:
                    try:self.cycle(symbol)
                    except Exception as exc:logging.exception("Cycle failed for %s: %s",symbol,exc)
                time.sleep(self.config["market"]["poll_seconds"])
        finally:self.mt5.close()
if __name__=="__main__":SeraphPrime().run()
