import numpy as np
import pandas as pd

class TradingEnvironment:
    """Optional-RL-compatible environment without importing Gym at module load."""
    ACTIONS=("HOLD","BUY","SELL")
    def __init__(self,features,initial_balance=10000.0,cost=0.0002):
        self.df=features.reset_index(drop=True).copy(); self.initial_balance=initial_balance; self.cost=cost; self.reset()
    def reset(self):
        self.i=1; self.balance=self.initial_balance; self.position=0; self.entry=0.0; self.done=False
        return self.observation()
    def observation(self):
        row=self.df.iloc[min(self.i,len(self.df)-1)]
        vals=np.asarray(row.select_dtypes(include=[np.number]),dtype=np.float32) if hasattr(row,"select_dtypes") else np.asarray(row,dtype=np.float32)
        return np.nan_to_num(vals,nan=0,posinf=0,neginf=0)
    def step(self,action):
        if self.i>=len(self.df)-1:return self.observation(),0.0,True,{}
        old=float(self.df.close.iloc[self.i]); self.i+=1; price=float(self.df.close.iloc[self.i]); reward=0.0
        if self.position and self.position != (1 if action==1 else -1 if action==2 else 0):
            reward=(price-self.entry)/self.entry*self.position-self.cost
            self.balance*=1+reward; self.position=0; self.entry=0
        if self.position==0 and action in (1,2):
            self.position=1 if action==1 else -1; self.entry=price; self.balance*=1-self.cost
        reward=float(np.clip(reward,-1,1)); self.done=self.i>=len(self.df)-1
        return self.observation(),reward,self.done,{"balance":self.balance,"price":price}
