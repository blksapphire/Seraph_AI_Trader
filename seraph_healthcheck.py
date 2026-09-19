import json,os,sys
def main():
    with open("config.json") as f:c=json.load(f)
    errors=[]
    if c.get("mode") not in {"paper","live"}:errors.append("mode")
    if not c.get("market",{}).get("symbols"):errors.append("symbols")
    if not 0<float(c["risk"]["risk_per_trade"])<=.02:errors.append("risk_per_trade")
    if float(c["risk"]["min_rr"])<1:errors.append("min_rr")
    for p in [c["runtime"]["status_file"],c["runtime"]["journal_file"],c["runtime"]["log_file"]]:os.makedirs(os.path.dirname(p) or ".",exist_ok=True)
    if errors:print("CONFIG INVALID:",", ".join(errors));return 1
    print("Seraph configuration: OK");print("Mode:",c["mode"],"| Symbols:",", ".join(c["market"]["symbols"]));print("Live execution:", "ENABLED" if c["mode"]=="live" else "DISABLED");return 0
if __name__=="__main__":sys.exit(main())
