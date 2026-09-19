import logging
from datetime import datetime, timezone
import MetaTrader5 as mt5
import pandas as pd

TF = {name: getattr(mt5, f"TIMEFRAME_{name}") for name in ("M1","M5","M15","M30","H1","H4","D1")}

class MT5Client:
    def __init__(self, config):
        self.cfg = config["mt5"]
        self.connected = False

    def connect(self):
        kwargs = {}
        if self.cfg.get("path"): kwargs["path"] = self.cfg["path"]
        if self.cfg.get("timeout_ms"): kwargs["timeout"] = self.cfg["timeout_ms"]
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        login = int(self.cfg.get("login") or 0)
        if login and self.cfg.get("password") and self.cfg.get("server"):
            if not mt5.login(login, password=self.cfg["password"], server=self.cfg["server"]):
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
        self.connected = True
        logging.info("MT5 connected")
        return True

    def close(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False

    def ensure_symbol(self, symbol):
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Unknown symbol: {symbol}")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select {symbol}")
        return mt5.symbol_info(symbol)

    def rates(self, symbol, timeframe, count):
        self.ensure_symbol(symbol)
        raw = mt5.copy_rates_from_pos(symbol, TF[timeframe], 0, count)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"No rates for {symbol} {timeframe}: {mt5.last_error()}")
        df = pd.DataFrame(raw)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df.set_index("time")

    def tick(self, symbol):
        self.ensure_symbol(symbol)
        value = mt5.symbol_info_tick(symbol)
        if value is None:
            raise RuntimeError(f"No tick for {symbol}")
        return value

    def account(self):
        info = mt5.account_info()
        return info._asdict() if info else {}

    def positions(self, symbol=None):
        rows = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        return [x._asdict() for x in rows] if rows else []

    def order_check(self, request):
        return mt5.order_check(request)

    def send(self, request):
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"order_send returned None: {mt5.last_error()}")
        return result

    @staticmethod
    def utc_now():
        return datetime.now(timezone.utc)
