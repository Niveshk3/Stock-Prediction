import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os, joblib

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

SEQUENCE_LEN = 60
FORECAST_DAYS = 30


def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def get_company_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        currency = info.get("currency", "USD")
        symbols = {
            "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£",
            "JPY": "¥", "CNY": "¥", "HKD": "HK$", "CAD": "C$",
            "AUD": "A$", "SGD": "S$", "KRW": "₩", "BRL": "R$",
            "MXN": "MX$", "CHF": "Fr", "SEK": "kr", "NOK": "kr"
        }
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "currency": currency,
            "currency_symbol": symbols.get(currency, currency + " "),
        }
    except Exception:
        return {"name": ticker}


def add_technical_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * std
    df['BB_Lower'] = df['BB_Mid'] - 2 * std
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df


FEATURES = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower', 'Return']


def build_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_or_load(ticker, df):
    model_path  = os.path.join(MODELS_DIR, f'{ticker}_lstm.keras')
    scaler_path = os.path.join(MODELS_DIR, f'{ticker}_scaler.pkl')

    df_feat = add_technical_indicators(df)[FEATURES]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat.values)

    X, y = build_sequences(scaled, SEQUENCE_LEN)
    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model  = load_model(model_path)
        scaler = joblib.load(scaler_path)
    else:
        model = build_lstm_model((SEQUENCE_LEN, len(FEATURES)))
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=50, batch_size=32, callbacks=[es], verbose=0)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

    preds_scaled = model.predict(X_test, verbose=0).flatten()
    dummy = np.zeros((len(preds_scaled), len(FEATURES)))
    dummy[:, 0] = preds_scaled
    preds = scaler.inverse_transform(dummy)[:, 0]

    dummy2 = np.zeros((len(y_test), len(FEATURES)))
    dummy2[:, 0] = y_test
    actuals = scaler.inverse_transform(dummy2)[:, 0]

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mape = float(np.mean(np.abs((actuals - preds) / (actuals + 1e-9))) * 100)

    return model, scaler, scaled, {"rmse": round(rmse, 2), "mape": round(mape, 2)}


def forecast(model, scaler, scaled, days=FORECAST_DAYS):
    seq = scaled[-SEQUENCE_LEN:].copy()
    preds = []
    for _ in range(days):
        inp = seq.reshape(1, SEQUENCE_LEN, len(FEATURES))
        p_scaled = model.predict(inp, verbose=0)[0, 0]
        next_row = seq[-1].copy()
        next_row[0] = p_scaled
        seq = np.vstack([seq[1:], next_row])
        dummy = np.zeros((1, len(FEATURES)))
        dummy[0, 0] = p_scaled
        price = scaler.inverse_transform(dummy)[0, 0]
        preds.append(round(float(price), 2))
    return preds


def sklearn_forecast(df, days=FORECAST_DAYS):
    from sklearn.linear_model import LinearRegression
    prices = df['Close'].values
    X = np.arange(len(prices)).reshape(-1, 1)
    model = LinearRegression().fit(X, prices)
    future_X = np.arange(len(prices), len(prices) + days).reshape(-1, 1)
    return [round(float(p), 2) for p in model.predict(future_X)]


def run_prediction(ticker, period="1y", forecast_days=FORECAST_DAYS):
    df   = get_stock_data(ticker, period)
    info = get_company_info(ticker)

    history_prices = [round(float(p), 2) for p in df['Close'].tolist()]
    history_dates  = [str(d.date()) for d in df.index]

    if TF_AVAILABLE and len(df) >= SEQUENCE_LEN + 20:
        model, scaler, scaled, metrics = train_or_load(ticker, df)
        future_prices = forecast(model, scaler, scaled, forecast_days)
        model_used = "LSTM"
    else:
        future_prices = sklearn_forecast(df, forecast_days)
        metrics = {}
        model_used = "LinearRegression"

    current_price = history_prices[-1]
    price_7d  = future_prices[6]  if len(future_prices) >= 7 else future_prices[-1]
    price_30d = future_prices[-1]
    pct_7d    = round((price_7d  - current_price) / current_price * 100, 2)
    pct_30d   = round((price_30d - current_price) / current_price * 100, 2)

    signal = (
        "STRONG BUY"  if pct_30d > 5  else
        "BUY"         if pct_30d > 1  else
        "STRONG SELL" if pct_30d < -5 else
        "SELL"        if pct_30d < -1 else
        "HOLD"
    )

    last_date    = df.index[-1]
    future_dates = [str((last_date + pd.Timedelta(days=i+1)).date()) for i in range(forecast_days)]

    # OHLCV for candlestick + volume charts
    ohlcv = [
        {
            "date":   str(d.date()),
            "open":   round(float(row['Open']),   2),
            "high":   round(float(row['High']),   2),
            "low":    round(float(row['Low']),    2),
            "close":  round(float(row['Close']),  2),
            "volume": int(row['Volume'])
        }
        for d, row in df.iterrows()
    ]

    # Technical indicators
    df_ind = add_technical_indicators(df)
    indicators = {
        "dates":       [str(d.date()) for d in df_ind.index],
        "rsi":         [round(float(v), 2) for v in df_ind['RSI'].tolist()],
        "macd":        [round(float(v), 2) for v in df_ind['MACD'].tolist()],
        "macd_signal": [round(float(v), 2) for v in df_ind['MACD_Signal'].tolist()],
        "bb_upper":    [round(float(v), 2) for v in df_ind['BB_Upper'].tolist()],
        "bb_lower":    [round(float(v), 2) for v in df_ind['BB_Lower'].tolist()],
        "bb_mid":      [round(float(v), 2) for v in df_ind['BB_Mid'].tolist()],
    }

    return {
        "ticker":     ticker,
        "company":    info,
        "model_used": model_used,
        "current_price": current_price,
        "history":    {"dates": history_dates, "prices": history_prices},
        "forecast":   {"dates": future_dates,  "prices": future_prices},
        "ohlcv":      ohlcv,
        "indicators": indicators,
        "metrics":    {"price_7d": price_7d, "pct_7d": pct_7d,
                       "price_30d": price_30d, "pct_30d": pct_30d,
                       "signal": signal, **metrics},
    }