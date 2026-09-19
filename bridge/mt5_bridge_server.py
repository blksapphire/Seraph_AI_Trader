#!/usr/bin/env python3
"""
Seraph MT5 bridge.

This file runs with Windows Python inside the same Wine prefix as MetaTrader 5.
The Linux Seraph process talks to it over localhost HTTP.

Environment:
  SERAPH_MT5_BRIDGE_TOKEN       Shared bearer token (required)
  MT5_TERMINAL_PATH              Optional path to terminal64.exe
  MT5_LOGIN                      Optional account login
  MT5_PASSWORD                   Optional account password
  MT5_SERVER                     Optional broker server
  SERAPH_BRIDGE_ALLOW_TRADING    Must be "1" before /order_send is allowed
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import MetaTrader5 as mt5


HOST = os.getenv("SERAPH_MT5_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.getenv("SERAPH_MT5_BRIDGE_PORT", "8765"))
TOKEN = os.getenv("SERAPH_MT5_BRIDGE_TOKEN", "")
ALLOW_TRADING = os.getenv("SERAPH_BRIDGE_ALLOW_TRADING", "0") == "1"

TF = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if hasattr(value, "_asdict"):
        return plain(value._asdict())
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def error_text(default="MT5 operation failed"):
    err = mt5.last_error()
    return f"{default}: {err}"


def ensure_symbol(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Unknown symbol: {symbol}")
    if not info.visible and not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select {symbol}: {error_text()}")
    return mt5.symbol_info(symbol)


def connect():
    kwargs = {}
    path = os.getenv("MT5_TERMINAL_PATH", "").strip()
    if path:
        kwargs["path"] = path

    if not mt5.initialize(**kwargs):
        raise RuntimeError(error_text("MT5 initialize failed"))

    login = os.getenv("MT5_LOGIN", "").strip()
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "").strip()
    if login and password and server:
        if not mt5.login(int(login), password=password, server=server):
            raise RuntimeError(error_text("MT5 login failed"))


def semantic_request(req):
    side = str(req.get("side", "")).upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")

    filling_name = str(req.get("type_filling", "IOC")).upper()
    filling_map = {
        "FOK": mt5.ORDER_FILLING_FOK,
        "IOC": mt5.ORDER_FILLING_IOC,
        "RETURN": mt5.ORDER_FILLING_RETURN,
    }

    return {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": req["symbol"],
        "volume": float(req["volume"]),
        "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": float(req["price"]),
        "sl": float(req["sl"]),
        "tp": float(req["tp"]),
        "deviation": int(req.get("deviation", 20)),
        "magic": int(req.get("magic", 260901)),
        "comment": str(req.get("comment", "Seraph-Prime")),
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_map.get(filling_name, mt5.ORDER_FILLING_IOC),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "SeraphMT5Bridge/1.0"

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)

    def authorized(self):
        if self.path == "/health":
            return True
        if not TOKEN:
            return False
        return self.headers.get("Authorization", "") == f"Bearer {TOKEN}"

    def send_json(self, code, payload):
        raw = json.dumps(plain(payload), separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        try:
            if self.path == "/health":
                self.send_json(200, {
                    "ok": True,
                    "connected": bool(mt5.terminal_info()),
                    "trading_allowed": ALLOW_TRADING,
                    "version": plain(mt5.version()),
                })
                return

            if not self.authorized():
                self.send_json(401, {"error": "unauthorized"})
                return

            if self.path == "/account":
                info = mt5.account_info()
                if info is None:
                    self.send_json(503, {"error": error_text("No MT5 account")})
                    return
                self.send_json(200, {"account": info._asdict()})
                return

            self.send_json(404, {"error": "not found"})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def do_POST(self):
        try:
            if not self.authorized():
                self.send_json(401, {"error": "unauthorized"})
                return

            body = self.read_json()

            if self.path == "/connect":
                connect()
                self.send_json(200, {"connected": True, "version": plain(mt5.version())})
                return

            if self.path == "/shutdown":
                mt5.shutdown()
                self.send_json(200, {"connected": False})
                return

            if self.path == "/symbol":
                symbol = str(body["symbol"])
                self.send_json(200, {"symbol": plain(ensure_symbol(symbol))})
                return

            if self.path == "/rates":
                symbol = str(body["symbol"])
                timeframe = str(body["timeframe"]).upper()
                count = int(body["count"])
                ensure_symbol(symbol)
                if timeframe not in TF:
                    raise ValueError(f"Unsupported timeframe: {timeframe}")
                raw = mt5.copy_rates_from_pos(symbol, TF[timeframe], 0, count)
                if raw is None or len(raw) == 0:
                    raise RuntimeError(error_text(f"No rates for {symbol} {timeframe}"))
                rows = []
                for row in raw:
                    rows.append({
                        name: plain(row[name])
                        for name in raw.dtype.names
                    })
                self.send_json(200, {"rates": rows})
                return

            if self.path == "/tick":
                symbol = str(body["symbol"])
                ensure_symbol(symbol)
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    raise RuntimeError(error_text(f"No tick for {symbol}"))
                self.send_json(200, {"tick": tick._asdict()})
                return

            if self.path == "/positions":
                symbol = body.get("symbol")
                rows = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
                self.send_json(200, {"positions": [x._asdict() for x in rows] if rows else []})
                return

            if self.path in {"/order_check", "/order_send"}:
                request = semantic_request(body["request"])
                check = mt5.order_check(request) if self.path == "/order_check" else None

                if self.path == "/order_check":
                    if check is None:
                        self.send_json(200, {
                            "result": {
                                "ok": False,
                                "retcode": None,
                                "comment": error_text("order_check failed"),
                            }
                        })
                        return
                    self.send_json(200, {
                        "result": {
                            "ok": True,
                            "retcode": int(check.retcode),
                            "comment": str(getattr(check, "comment", "")),
                            "balance": float(getattr(check, "balance", 0) or 0),
                            "equity": float(getattr(check, "equity", 0) or 0),
                            "margin": float(getattr(check, "margin", 0) or 0),
                            "margin_free": float(getattr(check, "margin_free", 0) or 0),
                            "request": plain(getattr(check, "request", {})),
                        }
                    })
                    return

                if not ALLOW_TRADING:
                    self.send_json(403, {
                        "error": "live order execution is disabled on the MT5 bridge; set SERAPH_BRIDGE_ALLOW_TRADING=1"
                    })
                    return

                result = mt5.order_send(request)
                if result is None:
                    self.send_json(200, {
                        "result": {
                            "ok": False,
                            "retcode": None,
                            "comment": error_text("order_send failed"),
                        }
                    })
                    return

                self.send_json(200, {
                    "result": {
                        "ok": int(result.retcode) in {
                            mt5.TRADE_RETCODE_DONE,
                            mt5.TRADE_RETCODE_PLACED,
                        },
                        "retcode": int(result.retcode),
                        "deal": int(getattr(result, "deal", 0) or 0),
                        "order": int(getattr(result, "order", 0) or 0),
                        "comment": str(getattr(result, "comment", "")),
                        "request": plain(getattr(result, "request", {})),
                    }
                })
                return

            self.send_json(404, {"error": "not found"})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})


if __name__ == "__main__":
    print(f"Seraph MT5 bridge listening on http://{HOST}:{PORT}", flush=True)
    print(f"Trading execution: {'ENABLED' if ALLOW_TRADING else 'DISABLED'}", flush=True)
    if not TOKEN:
        print("WARNING: SERAPH_MT5_BRIDGE_TOKEN is not set; all non-health requests will be rejected.", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
