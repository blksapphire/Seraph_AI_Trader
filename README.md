# Seraph-Prime v2

Seraph-Prime is a local-first multi-brain trading research and execution system for MetaTrader 5.

> **Research/demo software.** The default mode is `paper`. Do not treat model confidence as a guarantee of future returns. Validate on historical data, replay/backtests and a demo account before enabling live execution.

## Architecture

`MT5 -> Market Data -> Technical / Structural / Fundamental / HTF brains -> Decision Engine -> Risk Engine -> Executor -> Journal + Dashboard`

### Brains
- **Technical:** normalized trend/momentum/volatility features with an optional trained ML model and a deterministic fallback.
- **Structural:** volatility-aware liquidity sweep and break-of-structure analysis.
- **Fundamental:** cached NewsAPI headlines with optional FinBERT sentiment; failures degrade to neutral rather than stopping the trader.
- **HTF confirmation:** H1 structure is evaluated independently before an M15 decision is accepted.

### Decision layer
Every cycle records the final directional score, confidence, brain agreement and per-brain evidence. The action is only BUY/SELL when the configured threshold and agreement requirement are both met; otherwise it is HOLD.

### Risk layer
The new runtime includes percentage-based risk sizing, ATR-derived stop/target levels, spread limits, maximum open positions, per-symbol position limits and an equity drawdown guard. Live orders are passed through MT5 order checking and the returned trade result is inspected. MT5 documents `initialize`, market-data functions, `order_check` and `order_send` as the Python integration path. citeturn0search1turn0search0

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure `config.json`. Keep it in `paper` mode while validating.

Train the technical model:
```bash
python seraph_trainer.py
```

Run Seraph:
```bash
python seraph_prime_orchestrator.py
```

Run the dashboard in another terminal:
```bash
python seraph_dashboard.py
```

## Training design

The trainer uses a chronological holdout rather than randomly shuffling observations. Time-series validation should preserve temporal order; scikit-learn's `TimeSeriesSplit` is one standard implementation for this problem class. citeturn0search11

The current model is intentionally a lightweight gradient-boosting baseline. This gives us a measurable baseline before introducing a larger LSTM/RL system. The ML layer can be replaced without changing the decision, risk or execution interfaces.

## Roadmap
1. Historical replay/backtesting with spread, slippage and commission.
2. Trade memory and post-trade outcome labeling.
3. Multi-timeframe feature store and regime classifier.
4. Walk-forward evaluation and parameter search.
5. Ensemble calibration.
6. Reinforcement-learning experiments only after a stable, measurable baseline.
