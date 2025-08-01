# Seraph-Prime: A Multi-Brain AI Trading System

Seraph-Prime is a professional-grade, multi-strategy algorithmic trading framework. It moves beyond simple indicators by operating on a principle of **Weighted Evidence**, synthesizing intelligence from three distinct analytical domains to form a single, unified trading decision.

**[CRITICAL RISK WARNING]**
This is a highly complex and experimental system. Its failure modes are unpredictable. The potential for rapid and complete financial loss is extreme. You are the operator and are solely responsible for all outcomes. **DO NOT DEPLOY ON A LIVE ACCOUNT.**

## The Seraph-Prime Architecture

Seraph-Prime is not a single AI; it is a committee of experts led by an orchestrator.

1.  **🧠 The Technical Brain (`Seraph-TA`):** A quantitative analyst using an LSTM Neural Network trained on a rich set of indicators (RSI, MACD, Bollinger Bands) and patterns (Fair Value Gaps).
2.  **👁️ The Structural Brain (`Seraph-SMC`):** A price action specialist interpreting the market through the lens of ICT and Wyckoff. It identifies liquidity sweeps and breaks in market structure.
3.  **📰 The Fundamental Brain (`Seraph-FA`):** A macroeconomic analyst that ingests real-time financial news and uses a financial NLP model (FinBERT) to gauge market sentiment.
4.  **👑 The Orchestrator (`Seraph-Prime`):** The master strategist. It polls each brain for its analysis, weighs their scores according to your defined strategy in `config.json`, and executes high-conviction trades.

## Operational Workflow

**1. Configuration: Define Your Strategy**
-   Get your API key from [newsapi.org](https://newsapi.org) and add it to `config.json`.
-   **Crucially, adjust the `strategy_weights`.** A weight of `1.0` for TA and `0` for others makes it a pure technical bot. A 50/50 split between TA and SMC ignores news. You control the bot's "personality" here.

**2. Train the Technical Brain**
-   Run `python seraph_trainer.py` to train the core LSTM model on technical data. This only needs to be done once initially and then periodically.

**3. Deploy the Orchestrator**
-   Ensure your MT5 terminal is running under Wine.
-   Launch the master system: `python seraph_prime_orchestrator.py`

**4. Monitor Mission Control**
-   Launch the dashboard: `python seraph_dashboard.py`
-   Observe the synthesized Confidence Score and the contributing narratives from each brain. This is your window into its "thought" process.