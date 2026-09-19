# Seraph-Prime Unified

Local-first multi-brain MT5 trading research/execution system. The unified branch pulls the useful R-Reasoning capabilities into the cleaner v2 architecture instead of running two competing systems.

**Default mode is PAPER. Live execution remains opt-in.**

## Unified stack

MT5 → Market Data → Strategy Ensemble → Technical + SMC + Wyckoff + Fundamental + HTF → optional ML/RL overlays → Decision Engine → Risk Engine → Paper/Live → SQLite trade memory + JSON journal + notifications → Dashboard

## Imported R-Reasoning capabilities

- Wyckoff phase/event analysis
- MA crossover and RSI momentum strategies
- SMT divergence strategy helper
- SQLite trade/event memory
- Optional reinforcement-learning environment and PPO adapter
- RL training entry point: `train_rl.py`
- Discord/email notifications using environment secrets
- Strategy-oriented backtest dependency kept optional
- Existing R-Reasoning ideas are adapted to the v2 interfaces; duplicate MT5/risk/orchestrator stacks were not copied

## Existing v2 capabilities retained

- Explainable strategy ensemble
- SMC/BOS/liquidity/FVG/order-block analysis
- ML directional model
- multi-timeframe decision engine
- risk sizing and execution guards
- paper/live separation
- dashboard, healthcheck and CI tests

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python seraph_healthcheck.py
pytest -q
```

For RL experiments only:

```bash
pip install -r requirements-rl.txt
python train_rl.py --symbol XAUUSD --timeframe M15 --bars 10000
```

For the optional backtesting.py adapter:

```bash
pip install -r requirements-backtest.txt
```

## Run paper mode

Leave `"mode": "paper"` in `config.json`.

```bash
python seraph_prime_orchestrator.py
```

The deterministic ensemble remains the primary decision layer. RL is disabled by default and can only influence decisions when a trained model is explicitly configured.

## Persistent learning memory

Runtime trade/event records are stored in:

- `runtime/seraph.db` — SQLite
- `runtime/trade_journal.jsonl` — append-only journal

The database is intended to become the source for closed-trade outcome labeling and future reward/ensemble learning.

## Notifications

Notifications are disabled by default. Discord webhooks are configured in `config.json`; email credentials are supplied through `SERAPH_EMAIL_PASSWORD`. No secrets should be committed.

## Backtesting

```bash
python seraph_backtester.py --symbol XAUUSD --timeframe M15 --bars 5000
```

Backtest results are research metrics, not evidence of future profitability.

## Safety

- Paper mode is the default.
- RL is optional and off by default.
- Notification credentials are environment-based.
- Do not enable live trading until replay and demo behaviour have been reviewed.
- GitHub Actions validates code/tests; it does not run the MT5 trader.
