import math
import MetaTrader5 as mt5
class RiskEngine:
    def __init__(self,config): self.cfg=config["risk"]; self.magic=int(config["mt5"]["magic"])
    def allowed(self,account,positions,symbol,spread_points):
        if spread_points>self.cfg["max_spread_points"]: return False,"spread too high"
        if len(positions)>=self.cfg["max_open_positions"]: return False,"max open positions reached"
        if sum(1 for p in positions if p.get("symbol")==symbol)>=self.cfg["max_symbol_positions"]: return False,"symbol position limit reached"
        balance=float(account.get("balance",0)); equity=float(account.get("equity",balance))
        if balance and (balance-equity)/balance>=self.cfg["max_drawdown"]: return False,"account drawdown limit reached"
        return True,"ok"
    def levels(self,side,price,atr):
        sl_dist=atr*self.cfg["atr_stop_multiplier"]; tp_dist=max(atr*self.cfg["atr_target_multiplier"],sl_dist*self.cfg["min_rr"]); return (price-sl_dist,price+tp_dist) if side=="BUY" else (price+sl_dist,price-tp_dist)
    def volume(self,account,info,entry,stop):
        risk_money=float(account.get("balance",0))*self.cfg["risk_per_trade"]; distance=abs(entry-stop); tick_size=float(info.trade_tick_size or info.point); tick_value=float(info.trade_tick_value or 0)
        if distance<=0 or tick_size<=0 or tick_value<=0:return float(info.volume_min)
        raw=risk_money/(distance/tick_size*tick_value); step=float(info.volume_step or info.volume_min); return round(max(float(info.volume_min),min(float(info.volume_max),math.floor(raw/step)*step)),8)
    def request(self,symbol,side,price,sl,tp,volume):
        return {"action":mt5.TRADE_ACTION_DEAL,"symbol":symbol,"volume":volume,"type":mt5.ORDER_TYPE_BUY if side=="BUY" else mt5.ORDER_TYPE_SELL,"price":price,"sl":sl,"tp":tp,"deviation":20,"magic":self.magic,"comment":"Seraph-Prime-v2","type_time":mt5.ORDER_TIME_GTC,"type_filling":getattr(mt5,"ORDER_FILLING_IOC",mt5.ORDER_FILLING_RETURN)}
