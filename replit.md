# Zargul Trader 2.0

## Overview

Zargul Trader 2.0 is a sophisticated algorithmic trading system that uses machine learning to predict cryptocurrency price movements and execute automated trades. The system continuously analyzes market data for multiple cryptocurrencies (BTC-USD, ETH-USD, SOL-USD), generates AI-powered trading signals, and manages positions across different timeframes and strategies. It features multi-strategy trading capabilities, market regime detection, comprehensive risk management, and automated model retraining to adapt to changing market conditions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Trading Framework
The system is built around a modular architecture with five main components that work together continuously. The main trading loop runs every 5 minutes, analyzing each configured asset and making trading decisions based on AI predictions, market conditions, and risk parameters.

### AI-Powered Prediction Engine
The prediction system uses bidirectional LSTM neural networks with attention mechanisms to analyze market data. Models are trained for each asset and strategy combination, taking sequences of 45 time periods with ~27 technical indicators per period. The system includes confidence calibration using Platt scaling and supports multiple trading strategies (main, swing, scalp) with different timeframes and risk profiles.

### Multi-Source Data Integration
Market data is fetched from TwelveData as the primary source with Yahoo Finance as fallback. Social sentiment data is integrated from LunarCrush API. The system calculates extensive technical indicators including RSI, MACD, Bollinger Bands, volume analysis, CMF, VWAP, and OBV. Feature engineering creates normalized and derived indicators for model input.

### Market Regime Detection
The system implements sophisticated market regime classification using ADX for trend strength and Shannon entropy for market randomness. Three regimes are detected: Trending (strong directional movement), Ranging (sideways movement), and Chaotic (high entropy/unpredictable). Trading signals are filtered based on regime favorability to avoid unfavorable market conditions.

### Risk Management Framework
Comprehensive risk management includes minimum confidence thresholds, ATR-based minimum price movement requirements, position sizing based on volatility, and dynamic stop-loss/take-profit calculations. Multi-timeframe confirmation ensures lower timeframe signals align with higher timeframe trends before execution.

### Data Persistence Layer
The system uses SQLite as the primary database for trade storage and CSV files for human-readable logging. All trades, positions, and system states are persisted with full audit trails. Database operations include trade lifecycle management and position tracking.

### Automated Model Management
An auto-trainer system runs on schedule (configurable, default weekly) or triggers when model confidence drops below thresholds. The system automatically retrains all models with fresh data and includes cooldown periods to prevent excessive retraining.

### Analysis and Optimization Tools
The system includes backtesting capabilities for strategy validation, feature importance analysis using SHAP values, hyperparameter optimization using Optuna, and comprehensive diagnostic tools for system health monitoring.

## External Dependencies

### Market Data Providers
- **TwelveData API**: Primary source for OHLCV cryptocurrency data with configurable timeframes
- **Yahoo Finance**: Fallback data source when TwelveData is unavailable
- **LunarCrush API**: Social sentiment and engagement metrics for cryptocurrency analysis

### Machine Learning Stack
- **TensorFlow/Keras**: Neural network framework for LSTM model training and inference
- **scikit-learn**: Data preprocessing, scaling, and model evaluation utilities
- **SHAP**: Model explainability and feature importance analysis
- **Optuna**: Hyperparameter optimization for model tuning

### Data Processing Libraries
- **pandas**: Data manipulation and time series analysis
- **NumPy**: Numerical computations and array operations
- **pandas-ta**: Technical indicator calculations

### Visualization and Analysis
- **Matplotlib/Seaborn**: Chart generation and data visualization
- **Plotly**: Interactive plotting for analysis tools

### Utility Libraries
- **requests**: HTTP client for API communications
- **joblib**: Model serialization and caching
- **python-dotenv**: Environment variable management for API keys
- **ccxt/python-binance**: Alternative exchange connectivity (configured but not actively used)

### Database and Storage
- **SQLite**: Embedded database for trade and position storage
- **CSV logging**: Human-readable backup logging system

The system is designed to be self-contained and can run continuously with minimal external dependencies beyond the API services for data feeds.