import json, os, sqlite3, time

class TradeDatabase:
    """Persistent trade/event memory. SQLite keeps it local and queryable."""
    def __init__(self, config):
        self.path=config.get("database",{}).get("path","runtime/seraph.db")
        os.makedirs(os.path.dirname(self.path) or ".",exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,timestamp REAL,symbol TEXT,event TEXT,payload TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY,timestamp REAL,symbol TEXT,side TEXT,entry REAL,exit REAL,sl REAL,tp REAL,volume REAL,pnl REAL,r_multiple REAL,decision TEXT,features TEXT)""")
            c.commit()
    def event(self,symbol,event,payload):
        with sqlite3.connect(self.path) as c:c.execute("INSERT INTO events(timestamp,symbol,event,payload) VALUES(?,?,?,?)",(time.time(),symbol,event,json.dumps(payload,default=str)))
    def trade(self,record):
        keys=("timestamp","symbol","side","entry","exit","sl","tp","volume","pnl","r_multiple","decision","features")
        vals=[record.get(k) for k in keys]
        with sqlite3.connect(self.path) as c:c.execute("INSERT INTO trades(timestamp,symbol,side,entry,exit,sl,tp,volume,pnl,r_multiple,decision,features) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(*vals, json.dumps(record.get("features",{}),default=str)))
    def recent(self,limit=100):
        with sqlite3.connect(self.path) as c:return c.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
