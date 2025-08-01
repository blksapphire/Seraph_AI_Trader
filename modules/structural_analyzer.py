# modules/structural_analyzer.py
import pandas as pd

class StructuralAnalyzer:
    def __init__(self, config):
        self.config = config
        self.lookback = config['structural_parameters']['swing_point_lookback']

    def analyze(self, df: pd.DataFrame):
        """Analyzes market structure for ICT/Wyckoff concepts."""
        # Calculate ATR for dynamic thresholds
        atr = (df['high'] - df['low']).rolling(window=14).mean().iloc[-1]
        
        # Find recent swing highs and lows
        recent_high = df['high'].rolling(self.lookback).max().iloc[-2]
        recent_low = df['low'].rolling(self.lookback).min().iloc[-2]

        last_candle = df.iloc[-1]
        score = 0
        narrative = "Neutral"

        # 1. Liquidity Sweep check
        if last_candle['high'] > recent_high and last_candle['close'] < recent_high:
            score -= 0.5
            narrative = "Bearish Liquidity Sweep"
        if last_candle['low'] < recent_low and last_candle['close'] > recent_low:
            score += 0.5
            narrative = "Bullish Liquidity Sweep"
            
        # 2. Break of Structure (BOS) / Change of Character (CHoCH) check
        bos_threshold = self.config['structural_parameters']['bos_choch_threshold_atr'] * atr
        if last_candle['close'] > recent_high + bos_threshold:
            score += 1.0
            narrative = "Bullish Break of Structure"
        if last_candle['close'] < recent_low - bos_threshold:
            score -= 1.0
            narrative = "Bearish Break of Structure"

        # Normalize score to be between -1 and 1
        final_score = max(-1.0, min(1.0, score))
        return {'score': final_score, 'narrative': narrative}