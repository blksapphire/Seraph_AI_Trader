from dataclasses import dataclass, asdict

@dataclass
class Decision:
    action: str
    score: float
    confidence: float
    agreement: float
    rationale: list
    symbol: str
    timeframe: str

class DecisionEngine:
    def __init__(self, config):
        b = config["brains"]
        self.weights = {
            "technical": b["technical_weight"],
            "structural": b["structural_weight"],
            "fundamental": b["fundamental_weight"],
            "regime": b["regime_weight"],
        }
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Brain weights must sum to more than zero")
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.threshold = float(b["entry_threshold"])
        self.agreement_threshold = float(b["agreement_threshold"])

    def evaluate(self, symbol, timeframe, tech, struct, fund, htf):
        values = {
            "technical": float(tech["score"]),
            "structural": float(struct["score"]),
            "fundamental": float(fund["score"]),
            "regime": float(htf["score"]),
        }
        total = sum(values[k] * self.weights[k] for k in values)
        directional = [v for v in values.values() if abs(v) >= 0.15]
        agreement = (
            sum(1 for v in directional if v * total > 0) / len(directional)
            if directional else 0.0
        )
        confidence = min(1.0, abs(total) * 0.8 + agreement * 0.2)
        action = "HOLD"
        if total >= self.threshold and agreement >= self.agreement_threshold:
            action = "BUY"
        elif total <= -self.threshold and agreement >= self.agreement_threshold:
            action = "SELL"
        rationale = [f"{k}: {v:+.2f}" for k, v in values.items()]
        return Decision(action, total, confidence, agreement, rationale, symbol, timeframe)

    @staticmethod
    def dict(decision):
        return asdict(decision)
