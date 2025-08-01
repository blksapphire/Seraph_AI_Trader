# seraph_trainer.py
import logging
import json
import pickle
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from modules.technical_analyzer import TechnicalAnalyzer # Import the brain

class SeraphTrainer:
    def __init__(self, config_path="config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.ai_name = self.config["system_identity"]["name"]
        
        logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - [{self.ai_name} Trainer] - %(levelname)s - %(message)s')
        
        # Instantiate the analyzer to use its feature calculation logic
        self.tech_analyzer = TechnicalAnalyzer(self.config)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        logging.info(f"Training Manager for {self.ai_name} initialized.")

    def connect_mt5(self):
        # ... (Same Linux-friendly connection logic as before) ...
        pass

    def get_and_prepare_data(self):
        """Fetches a large dataset and uses the TechnicalAnalyzer for feature engineering."""
        logging.info("Fetching training data...")
        symbol = self.config["trading_parameters"]["symbol"]
        timeframe = getattr(mt5, self.config["trading_parameters"]["timeframe"])
        bars = self.config["training_settings"]["historical_data_bars"]
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        
        if rates is None:
            logging.error("Failed to download training data.")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Use the shared, centralized feature calculation method
        df = self.tech_analyzer.calculate_features(df)
        df.dropna(inplace=True)
        logging.info(f"Data prepared with {len(df.columns)} features.")
        return df

    def create_sequences_and_save_scaler(self, df):
        """Prepares data for the LSTM and saves the scaler and feature config."""
        feature_columns = [
            'close', 'high', 'low', 'open', 'tick_volume',
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'rsi',
            'bb_upper', 'bb_lower', 'fvg', 'order_block'
        ]
        available_features = [col for col in feature_columns if col in df.columns]
        
        scaled_data = self.scaler.fit_transform(df[available_features])
        
        # Save for the live bot
        with open('scaler.pkl', 'wb') as f: pickle.dump(self.scaler, f)
        with open('feature_columns.json', 'w') as f: json.dump(available_features, f)
        logging.info("Scaler and feature list saved for live deployment.")

        lookback = self.config["model_architecture"]["lookback_period"]
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, :])
            y.append(1 if df['close'].iloc[i] > df['close'].iloc[i-1] else 0)
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Builds the LSTM neural network."""
        units = self.config["model_architecture"]["lstm_units"]
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units, return_sequences=False),
            Dropout(0.2),
            Dense(units // 2, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logging.info("LSTM model architecture compiled.")
        return model

    def train_model(self):
        """Orchestrates the entire model training and saving process."""
        logging.info(f"--- {self.ai_name.upper()} MODEL TRAINING PROTOCOL INITIATED ---")
        # ... (Connect to MT5) ...
        
        data = self.get_and_prepare_data()
        if data is None: return

        X, y = self.create_sequences_and_save_scaler(data)
        self.model = self.build_model((X.shape[1], X.shape[2]))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=self.config["model_architecture"]["model_path"], save_best_only=True, monitor='val_loss')
        ]
        
        self.model.fit(X, y, 
            epochs=self.config["training_settings"]["epochs"],
            batch_size=self.config["training_settings"]["batch_size"],
            validation_split=self.config["training_settings"]["validation_split"],
            callbacks=callbacks, verbose=1)
        
        logging.info(f"--- TRAINING COMPLETE. Best model saved to {self.config['model_architecture']['model_path']} ---")
        # ... (Shutdown MT5) ...

if __name__ == "__main__":
    trainer = SeraphTrainer()
    trainer.train_model()