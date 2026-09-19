# Seraph-Prime Unified

Local-first multi-brain MT5 trading research/execution system.

The unified architecture keeps the **Seraph intelligence stack on Linux** while running the **native MetaTrader 5 Python integration inside the same Wine environment as the MT5 terminal**. A small authenticated localhost bridge connects the two.

**Default mode is PAPER. Live order execution requires explicit bridge + config enablement.**

## Architecture

```
Linux Mint
  │
  └─ Seraph Python
       ├─ Strategy ensemble
       ├─ Technical / SMC / Wyckoff
       ├─ Fundamental / HTF
       ├─ ML / optional RL
       ├─ Decision engine
       ├─ Risk engine
       └─ Dashboard / SQLite / journal
              │
              │ HTTP localhost
              ▼
       Wine MT5 Bridge
              │
              ▼
       MetaTrader 5 terminal
              │
              ▼
       Broker / Demo / Live account
```

MetaQuotes documents the Python integration around Windows Python and the MT5 terminal. On Linux, MT5 itself can run through Wine; this project therefore isolates the native MetaTrader5 Python package inside the Wine-side bridge instead of trying to install it into Linux Python.

## Modes

**Paper mode** still consumes real-time market data from the MT5 terminal, but Seraph does **not** call `order_send`. It generates and records hypothetical entries/SL/TP.

**Live mode** calculates position size and risk controls in Linux, sends an order-check request through the bridge, then sends the order only when the bridge is explicitly allowed to execute.

## Linux setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python seraph_healthcheck.py
pytest -q
```

Set the same bridge token in your Linux shell and Wine-side environment:

```bash
export SERAPH_MT5_BRIDGE_TOKEN='replace-with-a-long-random-token'
```

Start the Wine-side bridge with Windows Python from the same Wine prefix as MT5.

## Bridge

The bridge is:

`bridge/mt5_bridge_server.py`

Required Wine-side environment:

- `SERAPH_MT5_BRIDGE_TOKEN`
- optional `MT5_TERMINAL_PATH`
- optional `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`

For safety, live execution is disabled unless:

```bash
export SERAPH_BRIDGE_ALLOW_TRADING=1
```

The Linux config already points to:

`http://127.0.0.1:8765`

## Validation

Start the bridge, then from Linux:

```bash
python seraph_healthcheck.py
curl http://127.0.0.1:8765/health
python seraph_prime_orchestrator.py
```

In paper mode, a generated signal is logged but no trade is sent.

## Other capabilities

- Wyckoff phase/event analysis
- MA crossover and RSI momentum
- SMT divergence helper
- SQLite trade/event memory
- optional PPO/RL overlay
- Discord/email notifications
- explainable strategy ensemble
- SMC/BOS/liquidity/FVG/order-block analysis
- ML directional model
- multi-timeframe decision engine
- risk sizing and execution guards
- dashboard, healthcheck and CI tests

## Safety

- Paper mode is the default.
- The bridge rejects live `order_send` unless explicitly enabled.
- Never commit bridge/account credentials.
- Review demo behaviour before enabling live trading.
