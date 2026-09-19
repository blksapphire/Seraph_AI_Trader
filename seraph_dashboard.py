import json
import dash
from dash import dcc,html,dash_table
from dash.dependencies import Input,Output
with open("config.json") as f:config=json.load(f)
app=dash.Dash(__name__); app.title="Seraph-Prime v2"
app.layout=html.Div([dcc.Interval(id="tick",interval=config["dashboard"]["refresh_seconds"]*1000),html.H1("SERAPH-PRIME // CONTROL"),html.Div(id="status"),html.Div(id="decision"),html.H3("Brains"),dash_table.DataTable(id="brains",columns=[{"name":x,"id":x} for x in ["brain","score","confidence","narrative"]]),html.H3("Execution"),html.Pre(id="execution"),html.H3("Log"),html.Pre(id="log",style={"height":"300px","overflowY":"scroll"})],style={"fontFamily":"Arial","maxWidth":"1200px","margin":"30px auto"})
@app.callback([Output("status","children"),Output("decision","children"),Output("brains","data"),Output("execution","children"),Output("log","children")],[Input("tick","n_intervals")])
def refresh(_):
    try:
        with open(config["runtime"]["status_file"]) as f:s=json.load(f)
        d=s.get("decision",{}); b=s.get("brains",{}); rows=[{"brain":k,"score":round(v.get("score",0),3),"confidence":round(v.get("confidence",0),3),"narrative":v.get("narrative","")} for k,v in b.items()]
        try:logs=open(config["runtime"]["log_file"]).read()[-12000:]
        except Exception:logs="No log yet."
        return f"STATUS: {s.get('status','unknown')} | {s.get('symbol','-')}",f"{d.get('action','HOLD')} | score={d.get('score',0):+.3f} | confidence={d.get('confidence',0):.3f} | agreement={d.get('agreement',0):.3f}",rows,json.dumps(s.get("execution",{}),indent=2),logs
    except Exception as e:return "STATUS: offline","No decision yet",[],"",f"Waiting for runtime state: {e}"
if __name__=="__main__":app.run(debug=False,host=config["dashboard"]["host"],port=config["dashboard"]["port"])
