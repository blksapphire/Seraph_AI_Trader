import argparse, json, os
import numpy as np
from modules.mt5_client import MT5Client
from modules.technical_analyzer import TechnicalAnalyzer
from learning_agent.trading_environment import TradingEnvironment

def main():
    p=argparse.ArgumentParser(); p.add_argument("--symbol",default="XAUUSD"); p.add_argument("--timeframe",default="M15"); p.add_argument("--bars",type=int,default=10000); p.add_argument("--steps",type=int,default=10000); p.add_argument("--out",default="models/seraph_ppo")
    a=p.parse_args(); cfg=json.load(open("config.json")); client=MT5Client(cfg); client.connect()
    try: df=client.rates(a.symbol,a.timeframe,a.bars)
    finally: client.close()
    features=TechnicalAnalyzer.calculate_features(df).replace([np.inf,-np.inf],np.nan).dropna()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        import gymnasium as gym
        class GymAdapter(gym.Env):
            def __init__(self,data):
                self.env=TradingEnvironment(data); n=len(self.env.observation()); self.action_space=gym.spaces.Discrete(3); self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(n,),dtype=np.float32)
            def reset(self,seed=None,options=None): super().reset(seed=seed); return self.env.reset(),{}
            def step(self,action):
                o,r,d,i=self.env.step(int(action)); return o,r,d,False,i
        env=GymAdapter(features); check_env(env,warn=True); model=PPO("MlpPolicy",env,verbose=1); model.learn(total_timesteps=a.steps); os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True); model.save(a.out); print("saved",a.out)
    except ImportError: raise SystemExit("Install optional RL dependencies: pip install -r requirements-rl.txt")
if __name__=="__main__": main()
