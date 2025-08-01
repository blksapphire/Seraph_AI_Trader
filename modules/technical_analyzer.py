# modules/technical_analyzer.py
import logging
import pandas as pd
import numpy as np

class TechnicalAnalyzer:
    def __init__(self, config):
        self.config = config
        self.ai_name = config["system_identity"]["name"]
        # The LSTM model will be loaded by the Orchestrator and passed to the analyze method
        self.model = None 
        self.scaler = None
        self.feature_columns = None

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a rich set of technical indicators and SMC signals.
        This is the core feature engineering pipeline.
        """
        logging.info("Calculating advanced technical features...")

        # Standard Technical Indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10) # Add epsilon to avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + (bb_std * 2)
        df['bb_lower'] = df['sma_20'] - (bb_std * 2)

        # Programmatic Smart Money Concepts (SMC)
        df['fvg'] = 0
        for i in range(2, len(df)):
            if df['high'].iloc[i-2] < df['low'].iloc[i]:
                df.loc[df.index[i-1], 'fvg'] = 1
            elif df['low'].iloc[i-2] > df['high'].iloc[i]:
                df.loc[df.index[i-1], 'fvg'] = -1

        df['order_block'] = 0
        strong_move_threshold = df['close'].diff().abs().mean() * 1.5
        for i in range(1, len(df)):
            if df['close'].iloc[i-1] < df['open'].iloc[i-1] and (df['close'].iloc[i] - df['open'].iloc[i]) > strong_move_threshold:
                 df.loc[df.index[i-1], 'order_block'] = 1
            elif df['close'].iloc[i-1] > df['open'].iloc[i-1] and (df['open'].iloc[i] - df['close'].iloc[i]) > strong_move_threshold:
                 df.loc[df.index[i-1], 'order_block'] = -1
        
        return df

    def analyze(self, df: pd.DataFrame, model, scaler, feature_columns) -> dict:
        """
        Analyzes the latest data point using the trained LSTM model.
        Returns a score and narrative.
        """
        if model is None or scaler is None or feature_columns is None:
            return {'score': 0, 'narrative': 'TA model not loaded'}
        
        # Prepare the final sequence for prediction
        lookback = self.config["model_architecture"]["lookback_period"]
        latest_data = df[feature_columns].tail(lookback)
        
        if len(latest_data) < lookback:
            return {'score': 0, 'narrative': 'Not enough data for TA sequence'}
            
        scaled_data = scaler.transform(latest_data)
        X_pred = np.array([scaled_data])

        # Get prediction from the neural network
        prediction_raw = model.predict(X_pred)[0][0]

        # The raw prediction is a probability (0 to 1). We convert it to a score (-1 to 1).
        # This gives us a directional bias from the model.
        score = (prediction_raw - 0.5) * 2
        
        narrative = f"LSTM Prediction: {'Bullish' if score > 0 else 'Bearish'} ({prediction_raw:.2%})"
        
        return {'score': score, 'narrative': narrative}