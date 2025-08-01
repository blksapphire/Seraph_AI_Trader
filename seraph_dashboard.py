# seraph_dashboard.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import MetaTrader5 as mt5
import json
import pandas as pd

# --- Load Config ---
with open("config.json", 'r') as f:
    config = json.load(f)
AI_NAME = config["system_identity"]["name"]

# --- Initialize Dash App ---
app = dash.Dash(__name__)
app.title = f"{AI_NAME} Dashboard"

app.layout = html.Div(style={'backgroundColor': '#1E1E1E', 'color': '#E0E0E0', 'fontFamily': 'Monospace'}, children=[
    dcc.Interval(id='interval-component', interval=config['dashboard']['refresh_interval_seconds'] * 1000, n_intervals=0),
    html.H1(f"{AI_NAME}: Multi-Brain AI Dashboard", style={'textAlign': 'center', 'color': '#4CAF50'}),
    
    html.Div(id='live-status', style={'textAlign': 'center', 'fontSize': '20px', 'marginBottom': '20px'}),
    html.Div(id='confidence-breakdown', style={'textAlign': 'center', 'fontSize': '18px', 'marginBottom': '20px', 'padding': '10px', 'border': '1px solid #444', 'borderRadius': '5px'}),
    
    html.H2("Account Information", style={'borderBottom': '1px solid #444'}),
    html.Div(id='account-info'),
    
    html.H2("Open Positions", style={'borderBottom': '1px solid #444'}),
    html.Div(id='positions-table'),
    
    html.H2("System Log", style={'borderBottom': '1px solid #444'}),
    html.Pre(id='log-viewer', style={'border': '1px solid #444', 'height': '200px', 'overflowY': 'scroll', 'backgroundColor': '#111', 'padding': '5px'}),
    
    html.Button('EMERGENCY STOP', id='emergency-stop-button', n_clicks=0, style={'width': '100%', 'backgroundColor': '#B71C1C', 'color': 'white', 'fontWeight': 'bold', 'padding': '15px', 'fontSize': '20px', 'border': 'none'})
])

# ... (connect_mt5_for_dashboard function remains the same) ...

@app.callback(
    [Output('live-status', 'children'),
     Output('confidence-breakdown', 'children'),
     Output('account-info', 'children'),
     Output('positions-table', 'children'),
     Output('log-viewer', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    status_text = f"Status: {AI_NAME} not running or status file not found."
    confidence_text = "Confidence Scores: N/A"
    
    # 1. Update Status & Confidence from status file
    try:
        with open(config["system_files"]["status_file"], 'r') as f:
            status_data = json.load(f)
            status_text = f"Status: {status_data.get('status', 'N/A')} - {status_data.get('message', 'N/A')}"
            
            # Display the crucial breakdown of scores
            scores = status_data.get('scores', {})
            final_score = scores.get('final', 0)
            color = '#4CAF50' if final_score > 0 else '#F44336' if final_score < 0 else '#9E9E9E'
            confidence_text = [
                html.B(f"Final Confidence: {final_score:.2f}", style={'color': color}),
                html.Br(),
                f"TA: {scores.get('tech', 0):.2f} | ",
                f"SMC: {scores.get('struct', 0):.2f} | ",
                f"FA: {scores.get('fund', 0):.2f}"
            ]
    except Exception:
        pass # Keep default text
    
    # ... (Rest of the dashboard update logic for account info, positions, logs is the same) ...
    # It will connect to MT5, get data, and then shutdown the connection.
    
    return status_text, confidence_text, info_text, table, logs

# ... (Emergency stop callback remains the same) ...

if __name__ == '__main__':
    app.run_server(debug=False, host=config['dashboard']['host'], port=config['dashboard']['port'])