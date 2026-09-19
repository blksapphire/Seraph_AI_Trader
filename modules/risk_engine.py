import math
import MetaTrader5 as mt5

class RiskEngine:
    def __init__(self, config):
        self.cfg = config["risk"]
        self.magic = int(config["mt5"]["magic"])

    def allowed(self, account, positions, symbol, spread_points):
        if spread_points > self.cfg["max_spread_points"]:
            return False, "spread too high"
        if len(positions) >= self.cfg["max_open_positions"]:
            return False, "max open positions reached"
        if sum(1 for p in positions if p.get("symbol") == symbol) >= self.cfg["max_symbol_positions"]:
            return False, "symbol position already exists"
        balance = float(account.get("balance", 0))
        equity = float(account.get("equity", balance))
        if balance and (balance - equity) / balance >= self.cfg["max_daily_loss"]:
            return False, "equity drawdown limit reached"
        return True, "ok"

    def levels(self, side, price, atr):
        stop_dist = atr * self.cfg["atr_stop_multiplier"]
        target_dist = atr * self.cfg["atr_target_multiplier"]
        if side == "BUY":
            return price - stop_dist, price + target_dist
        return price + stop_dist, price - target_dist

    def volume(self, account, symbol_info, entry, stop):
        risk_money = float(account.get("balance", 0)) * self.cfg["risk_per_trade"]
        distance = abs(entry - stop)
        tick_size = float(symbol_info.trade_tick_size or symbol_info.point)
        tick_value = float(symbol_info.trade_tick_value or 0)
        if distance <= 0 or tick_value <= 0 or tick_size <= 0:
            return float(symbol_info.volume_min)
        raw = risk_money / (distance / tick_size * tick_value)
        step = float(symbol_info.volume_step or symbol_info.volume_min)
        vol = max(float(symbol_info.volume_min),
                  min(float(symbol_info.volume_max), math.floor(raw / step) * step))
        return round(vol, 8)

    def request(self, symbol, side, price, sl, tp, volume):
        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        filling = getattr(mt5, "ORDER_FILLING_IOC", mt5.ORDER_FILLING_RETURN)
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic,
            "comment": "Seraph-Prime-v2",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
