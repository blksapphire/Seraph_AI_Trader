import logging
from datetime import datetime, timezone
import os

import pandas as pd
import requests


class MT5Client:
    """
    Linux-side client for the MT5 bridge running inside the Wine environment.

    Seraph itself never imports MetaTrader5. The bridge owns the native MT5
    Python package and exposes a small authenticated HTTP API on localhost.
    """

    def __init__(self, config):
        self.cfg = config["mt5"]
        self.base_url = self.cfg.get("bridge_url", "http://127.0.0.1:8765").rstrip("/")
        self.token = os.getenv(self.cfg.get("bridge_token_env", "SERAPH_MT5_BRIDGE_TOKEN"), "")
        self.timeout = max(float(self.cfg.get("timeout_ms", 60000)) / 1000.0, 1.0)
        self.session = requests.Session()
        self.connected = False

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request(self, method, path, payload=None):
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"MT5 bridge unavailable at {self.base_url}: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"MT5 bridge returned non-JSON response ({response.status_code})"
            ) from exc

        if response.status_code >= 400:
            raise RuntimeError(data.get("error", f"MT5 bridge HTTP {response.status_code}"))
        return data

    def connect(self):
        data = self._request("POST", "/connect", {})
        if not data.get("connected"):
            raise RuntimeError(data.get("error", "MT5 bridge connection failed"))
        self.connected = True
        logging.info("MT5 bridge connected")
        return True

    def close(self):
        if self.connected:
            try:
                self._request("POST", "/shutdown", {})
            finally:
                self.connected = False
                self.session.close()

    def ensure_symbol(self, symbol):
        return self._request("POST", "/symbol", {"symbol": symbol})["symbol"]

    def rates(self, symbol, timeframe, count):
        data = self._request(
            "POST",
            "/rates",
            {"symbol": symbol, "timeframe": timeframe, "count": int(count)},
        )
        rows = data.get("rates", [])
        if not rows:
            raise RuntimeError(data.get("error", f"No rates for {symbol} {timeframe}"))
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df.set_index("time")

    def tick(self, symbol):
        return self._request("POST", "/tick", {"symbol": symbol})["tick"]

    def account(self):
        return self._request("GET", "/account").get("account", {})

    def positions(self, symbol=None):
        data = self._request("POST", "/positions", {"symbol": symbol})
        return data.get("positions", [])

    def order_check(self, request):
        return self._request("POST", "/order_check", {"request": request}).get("result", {})

    def send(self, request):
        return self._request("POST", "/order_send", {"request": request}).get("result", {})

    @staticmethod
    def utc_now():
        return datetime.now(timezone.utc)
