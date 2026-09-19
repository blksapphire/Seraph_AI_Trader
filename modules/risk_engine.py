import math


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
            return False, "symbol position limit reached"
        balance = float(account.get("balance", 0))
        equity = float(account.get("equity", balance))
        if balance and (balance - equity) / balance >= self.cfg["max_drawdown"]:
            return False, "account drawdown limit reached"
        return True, "ok"

    def levels(self, side, price, atr):
        sl_dist = atr * self.cfg["atr_stop_multiplier"]
        tp_dist = max(atr * self.cfg["atr_target_multiplier"], sl_dist * self.cfg["min_rr"])
        return (price - sl_dist, price + tp_dist) if side == "BUY" else (price + sl_dist, price - tp_dist)

    def volume(self, account, info, entry, stop):
        risk_money = float(account.get("balance", 0)) * self.cfg["risk_per_trade"]
        distance = abs(entry - stop)
        point = float(info.get("point", 0) or 0)
        tick_size = float(info.get("trade_tick_size", 0) or point)
        tick_value = float(info.get("trade_tick_value", 0) or 0)
        if distance <= 0 or tick_size <= 0 or tick_value <= 0:
            return float(info.get("volume_min", 0.01) or 0.01)

        raw = risk_money / (distance / tick_size * tick_value)
        min_volume = float(info.get("volume_min", 0.01) or 0.01)
        max_volume = float(info.get("volume_max", min_volume) or min_volume)
        step = float(info.get("volume_step", min_volume) or min_volume)
        return round(max(min_volume, min(max_volume, math.floor(raw / step) * step)), 8)

    def request(self, symbol, side, price, sl, tp, volume):
        return {
            "symbol": symbol,
            "volume": volume,
            "side": side,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic,
            "comment": "Seraph-Prime-v3",
            "type_time": "GTC",
            "type_filling": "IOC",
        }
