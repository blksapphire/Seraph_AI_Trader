import os, numpy as np

class RLAgent:
    """Optional Stable-Baselines3 adapter. Deterministic stack remains primary."""
    def __init__(self,model_path=None):
        self.model=None; self.model_path=model_path
        if model_path and os.path.exists(model_path):
            try:
                from stable_baselines3 import PPO
                self.model=PPO.load(model_path)
            except Exception: self.model=None
    def predict(self,observation):
        if self.model is None:return {"score":0.0,"confidence":0.0,"action":"HOLD"}
        action,_=self.model.predict(np.asarray(observation,dtype=np.float32),deterministic=True)
        action=int(np.asarray(action).item()); name=("HOLD","BUY","SELL")[max(0,min(2,action))]
        return {"score":1.0 if name=="BUY" else -1.0 if name=="SELL" else 0.0,"confidence":0.5,"action":name}
