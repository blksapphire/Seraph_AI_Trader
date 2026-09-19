import json,logging,os,time

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
    def __init__(self, config_path="config.json"):
        self.config = json.load(open(config_path))
        os.makedirs(os.path.dirname(self.config["runtime"]["log_file"]) or ".", exist_ok=True)
        logging.basicConfig(
            filename=self.config["runtime"]["log_file"],
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        self.mt5 = MT5Client(self.config)
        self.tech = TechnicalAnalyzer(self.config)
        self.struct = StructuralAnalyzer(self.config)
        self.fund = FundamentalAnalyzer(self.config)
        self.decision = DecisionEngine(self.config)
        self.risk = RiskEngine(self.config)
        self.state = StateStore(self.config)
        self.db = TradeDatabase(self.config)
        self.notify = Notifier(self.config)
        self.running = True
        self._last_signal = {}

    def _atr(self, df):
        return float(self.tech.calculate_features(df)["atr"].iloc[-1])

    def _paper(self, symbol, decision, df):
        tick = self.mt5.tick(symbol)
        info = self.mt5.ensure_symbol(symbol)
        price = float(tick["ask"] if decision.action == "BUY" else tick["bid"])
        sl, tp = self.risk.levels(decision.action, price, self._atr(df))
        return {
            "executed": False,
            "mode": "paper",
            "action": decision.action,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "reason": "paper mode - signal generated, no order sent",
        }

    def _execute(self, symbol, decision, df):
        tick = self.mt5.tick(symbol)
        info = self.mt5.ensure_symbol(symbol)
        account = self.mt5.account()
        positions = self.mt5.positions()
        point = float(info.get("point", 0) or 0)
        if point <= 0:
            return {"executed": False, "mode": "live", "reason": "invalid symbol point"}

        spread = (float(tick["ask"]) - float(tick["bid"])) / point
        ok, reason = self.risk.allowed(account, positions, symbol, spread)
        if not ok:
            return {"executed": False, "mode": "live", "reason": reason}

        price = float(tick["ask"] if decision.action == "BUY" else tick["bid"])
        sl, tp = self.risk.levels(decision.action, price, self._atr(df))
        volume = self.risk.volume(account, info, price, sl)
        request = self.risk.request(symbol, decision.action, price, sl, tp, volume)

        check = self.mt5.order_check(request)
        if not check.get("ok") or not check.get("approved"):
            return {
                "executed": False,
                "mode": "live",
                "reason": "order_check failed",
                "retcode": check.get("retcode"),
                "comment": check.get("comment"),
            }

        result = self.mt5.send(request)
        return {
            "executed": bool(result.get("ok")),
            "mode": "live",
            "retcode": result.get("retcode"),
            "deal": result.get("deal"),
            "order": result.get("order"),
            "comment": result.get("comment"),
            "sl": sl,
            "tp": tp,
            "volume": volume,
        }

    def cycle(self, symbol):
        m15 = self.mt5.rates(
            symbol,
            self.config["market"]["primary_timeframe"],
            self.config["market"]["bars"],
        )
        htf_df = self.mt5.rates(
            symbol,
            self.config["market"]["higher_timeframe"],
            max(500, self.config["market"]["bars"] // 3),
        )
        tech = self.tech.analyze(m15)
        struct = self.struct.analyze(m15)
        fund = self.fund.analyze(self.config["fundamental"]["currencies"])
        htf = self.struct.analyze(htf_df)
        decision = self.decision.evaluate(
            symbol,
            self.config["market"]["primary_timeframe"],
            tech,
            struct,
            fund,
            htf,
        )

        key = f"{symbol}:{decision.action}:{round(decision.score, 2)}"
        duplicate = key == self._last_signal.get(symbol)
        self._last_signal[symbol] = key

        if duplicate and decision.action != "HOLD":
            execution = {
                "executed": False,
                "mode": self.config.get("mode", "paper"),
                "reason": "signal deduplicated",
            }
        elif decision.action == "HOLD":
            execution = {
                "executed": False,
                "mode": self.config.get("mode", "paper"),
                "reason": "hold",
            }
        elif self.config.get("mode", "paper") == "paper":
            execution = self._paper(symbol, decision, m15)
        elif self.config.get("mode") == "live":
            execution = self._execute(symbol, decision, m15)
        else:
            execution = {
                "executed": False,
                "mode": self.config.get("mode"),
                "reason": "invalid mode",
            }

        payload = {
            "status": "running",
            "mode": self.config.get("mode", "paper"),
            "symbol": symbol,
            "decision": self.decision.dict(decision),
            "brains": {
                "technical": tech,
                "structural": struct,
                "fundamental": fund,
                "higher_timeframe": htf,
            },
            "execution": execution,
        }
        self.state.write_status(payload)
        self.state.journal(payload)
        self.db.event(symbol, "decision", payload)

        if execution.get("executed") or (decision.action != "HOLD" and not duplicate):
            self.notify.send(
                f"{symbol} {decision.action} score={decision.score:+.2f} "
                f"conf={decision.confidence:.2f} | {decision.rationale}",
                "Seraph signal",
            )
        return payload

    def run(self):
        self.mt5.connect()
        try:
            while self.running:
                for symbol in self.config["market"]["symbols"]:
                    try:
                        self.cycle(symbol)
                    except Exception as exc:
                        logging.exception("Cycle failed for %s: %s", symbol, exc)
                time.sleep(self.config["market"]["poll_seconds"])
        finally:
            self.mt5.close()


if __name__ == "__main__":
    SeraphPrime().run()
