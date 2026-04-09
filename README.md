# Ethereum Price Prediction with LSTM

A bachelor's thesis project (2021) that uses an LSTM neural network to predict
Ethereum (ETH/USDT) daily closing prices from historical Binance data.

> **Archival note:** This project was originally written in January 2021 as
> part of my undergraduate work. It is preserved here as a historical artifact.
> Some dependencies (notably `pandas.DataFrame.append` and the older
> `python-binance` API) have since changed, so the live data-fetching path in
> `main.py` may no longer run on modern environments. The bundled CSV
> (`ETHUSDT-1d-data.csv`) still works for offline training.

## Overview

The project has two parts:

1. **`main.py`** — connects to the Binance API and downloads historical OHLCV
   candles for a given symbol and interval, saving them as a CSV.
2. **`lstm.py`** — loads the CSV, preprocesses the close-price series, trains
   an LSTM model, and produces:
   - a fit plot on the training set,
   - a fit plot on the test set,
   - a recursive 30-day forward forecast.

## Model

- **Architecture:** single `LSTM(256)` layer → `Dense(1)`
- **Loss / optimizer:** MSE / Adam
- **Lookback window:** 100 days
- **Train/test split:** 80 / 20
- **Scaling:** `MinMaxScaler` fit on the training portion only (to avoid
  test-set leakage)
- **Forecast horizon:** 30 days, generated recursively by feeding each new
  prediction back into the input window

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Binance API client + historical kline downloader |
| `lstm.py` | Data preprocessing, LSTM model, training, forecasting, plots |
| `ETHUSDT-1d-data.csv` | Cached daily OHLCV data used for training |
| `.gitignore` | Excludes secrets, caches, virtualenvs |

## Running

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow python-binance python-dateutil
python lstm.py
```

If you want to refresh the CSV from Binance, set your own API key/secret in
`main.py`. **Never commit real keys** — use environment variables or a
gitignored `.env` file.

## Honest disclaimer

This is a learning project, not a trading system.

LSTMs trained on raw price history tend to learn the trivial "tomorrow ≈
today" rule, which makes the test-set plot look impressively close to reality
but contributes very little real predictive power. Crypto prices behave
close to a random walk, and a naive `price[t] = price[t-1]` baseline often
ties or beats LSTM models on this kind of task. The 30-day recursive forecast
also compounds error at every step, so it should be read as a trend
extrapolation, not a prediction.

**Do not trade real money based on this model.**

## Lessons learned (the hard way)

- **Never fit the scaler on the full dataset before splitting.** The original
  code did this and silently leaked test-set statistics into training. The
  current version fits `MinMaxScaler` on the training portion only.

## Author

A. Can Kaytaz — bachelor's project, 2021.
