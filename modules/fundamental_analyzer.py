import logging,time,requests

class FundamentalAnalyzer:
    def __init__(self,config):
        self.cfg=config["fundamental"]; self.cache={}; self.sentiment=None
        if self.cfg.get("enabled") and self.cfg.get("news_api_key"):
            try:
                from transformers import pipeline
                self.sentiment=pipeline("sentiment-analysis",model=self.cfg["sentiment_model"])
            except Exception as exc: logging.warning("FinBERT unavailable: %s",exc)

    def _score(self,text):
        if not self.sentiment: return 0.
        try:
            out=self.sentiment(text[:512])[0]; label=out["label"].lower(); value=float(out["score"])
            return value if label=="positive" else -value if label=="negative" else 0.
        except Exception: return 0.

    def analyze(self,currencies):
        if not self.cfg.get("enabled") or not self.cfg.get("news_api_key"): return {"score":0.,"confidence":0.,"narrative":"Fundamental brain disabled/not configured","headlines":[]}
        key=",".join(sorted(currencies)); now=time.time()
        if key in self.cache and now-self.cache[key]["ts"]<self.cfg["cache_minutes"]*60: return self.cache[key]["value"]
        total=count=0.; headlines=[]
        try:
            for cur in currencies:
                r=requests.get("https://newsapi.org/v2/everything",params={"q":cur,"apiKey":self.cfg["news_api_key"],"language":"en","sortBy":"publishedAt","pageSize":10},timeout=self.cfg["timeout_seconds"]); r.raise_for_status()
                for a in r.json().get("articles",[]):
                    title=a.get("title") or ""
                    if title: total+=self._score(title); count+=1; headlines.append(title)
            result={"score":total/count if count else 0.,"confidence":min(1.,count/10),"narrative":headlines[0] if headlines else "No relevant headlines","headlines":headlines[:5]}
        except Exception as exc: logging.warning("News analysis failed: %s",exc); result={"score":0.,"confidence":0.,"narrative":"News unavailable","headlines":[]}
        self.cache[key]={"ts":now,"value":result}; return result
