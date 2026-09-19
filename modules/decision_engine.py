from dataclasses import dataclass,asdict
@dataclass
class Decision:
    action:str; score:float; confidence:float; agreement:float; rationale:list; symbol:str; timeframe:str
class DecisionEngine:
    def __init__(self,config):
        b=config["brains"]; self.weights={k:float(b[f"{k}_weight"]) for k in ("technical","structural","fundamental","regime")}; total=sum(self.weights.values()) or 1; self.weights={k:v/total for k,v in self.weights.items()}; self.threshold=float(b["entry_threshold"]); self.agreement_threshold=float(b["agreement_threshold"]); self.max_fundamental_gap=float(b.get("max_fundamental_disagreement",.65))
    def evaluate(self,symbol,timeframe,tech,struct,fund,htf):
        values={"technical":float(tech.get("score",0)),"structural":float(struct.get("score",0)),"fundamental":float(fund.get("score",0)),"regime":float(htf.get("score",0))}; total=sum(values[k]*self.weights[k] for k in values); directional=[v for v in values.values() if abs(v)>=.15]; agreement=sum(1 for v in directional if v*total>0)/len(directional) if directional else 0; confidence=min(1,abs(total)*.75+agreement*.25); action="HOLD"; reasons=[f"{k}: {v:+.2f}" for k,v in values.items()]
        if abs(values["fundamental"])>=self.max_fundamental_gap and values["fundamental"]*total<0: reasons.append("fundamental contradiction -> HOLD")
        elif total>=self.threshold and agreement>=self.agreement_threshold: action="BUY"
        elif total<=-self.threshold and agreement>=self.agreement_threshold: action="SELL"
        return Decision(action,total,confidence,agreement,reasons,symbol,timeframe)
    @staticmethod
    def dict(d): return asdict(d)
