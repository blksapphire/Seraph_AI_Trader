# Seraph-Prime v2

Local-first multi-brain trading research and execution system for MetaTrader 5.

**Default mode is PAPER.** Live execution is explicitly opt-in. This project is for research and demo validation; model confidence is not a guarantee of trading performance.

## Architecture
MT5 → Market Data → Strategy Ensemble → Technical/SMC/Fundamental/HTF Brains → Decision Engine → Risk Engine → Paper/Live Executor → Journal + Dashboard

## Strategies
- EMA trend following
- MACD momentum
- RSI/Bollinger mean reversion
- Donchian-style breakout
- VWAP deviation
- candlestick/rejection patterns
- ATR volatility regime
- liquidity sweeps
- break of structure / displacement
- higher-timeframe structure confirmation
- optional ML probability overlay

Every strategy produces a normalized directional score and is exposed in runtime state.

## Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python seraph_healthcheck.py

MT5 must be installed/running and accessible to Python for market-data, training, replay and execution commands.

## Train
python seraph_trainer.py

The ML model is an additional probability signal; the deterministic strategy ensemble remains usable without it.

## Run paper mode
Keep "mode": "paper" in config.json.
python seraph_prime_orchestrator.py

The system analyzes every configured symbol and calculates hypothetical entry/SL/TP levels without sending orders.

## Dashboard
python seraph_dashboard.py

## Replay/backtest
python seraph_backtester.py --symbol XAUUSD --timeframe M15 --bars 5000
python seraph_backtester.py --symbol GBPJPY --timeframe M15 --bars 5000
python seraph_backtester.py --symbol EURUSD --timeframe M15 --bars 5000

Replay results are research metrics, not proof of profitability.

## Tests
pytest -q
python -m compileall -q .

GitHub Actions is CI only: it checks syntax and unit tests. It is not intended to run the MT5 trader or place trades.

## Next engineering layer
1. Cost-aware backtester
2. Trade lifecycle manager and minimum-hold enforcement
3. Persistent trade memory/outcome labeling
4. Session and economic-event filters
5. Walk-forward optimization
6. Probability calibration
7. Regime classifier
8. Ensemble/reward learning
9. RL experiments after the baseline is measurable

Never enable live mode until paper/replay behaviour has been inspected on a demo account.
