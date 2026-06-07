import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from predictor import run_prediction
import requests

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '').strip()

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return jsonify({"status": "Stock Predictor API is running", "version": "1.0"})

@app.route('/search', methods=['GET'])
def search():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({"results": []})
    try:
        import yfinance as yf
        results = []
        ticker = yf.Ticker(q)
        # Use yfinance search
        import requests as req
        res = req.get(
            'https://query2.finance.yahoo.com/v1/finance/search',
            params={'q': q, 'quotesCount': 10, 'newsCount': 0},
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        data = res.json()
        quotes = data.get('quotes', [])
        for item in quotes:
            symbol = item.get('symbol', '')
            name   = item.get('longname') or item.get('shortname') or symbol
            exch   = item.get('exchDisp', '')
            typ    = item.get('typeDisp', '')
            if symbol:
                results.append({'symbol': symbol, 'name': name, 'exchange': exch, 'type': typ})
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'results': [], 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()

    system_prompt = """You are a concise, trustworthy financial analyst AI. Provide a balanced stock-market analysis with clear trend, risk, and outlook guidance. Keep the tone professional, factual, and easy to understand. Do not make guaranteed predictions or give personalized financial advice. """

    prompt = f"""Analyze the forecast for {data['ticker']}:"
Current: ${data['cur']} | 7-day: ${data['p7']} ({data['pct7']}%) | 30-day: ${data['p30']} ({data['pct30']}%)
Signal: {data['signal']} | RMSE: {data.get('rmse', 'N/A')}
Write 3-4 sentences on trend, risks, and outlook.
Then: CONFIDENCE: [Low/Medium/High]
Then: TAGS: [tag1, tag2, tag3]"""

    models = [
        'google/gemma-4-26b-a4b-it:free',
        'meta-llama/llama-3.2-3b-instruct:free',
        'microsoft/phi-3-mini-128k-instruct:free',
        'qwen/qwen3-0.6b:free',
    ]

    for model in models:
        try:
            if not OPENROUTER_API_KEY:
                raise ValueError('OPENROUTER_API_KEY is not set')

            response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
                headers={
                    'Authorization': 'Bearer ' + os.environ.get('OPENROUTER_API_KEY', ''),
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'openrouter/auto',
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': prompt}
                    ]
                },
                timeout=30
            )
            result = response.json()
            print("OpenRouter response:", result.get('error', 'OK'))

            if 'choices' in result:
                text = result['choices'][0]['message']['content']
                return jsonify({'content': [{'type': 'text', 'text': text}]})

        except Exception as e:
            print(f"OpenRouter failed: {e}")

    return jsonify({'content': [{'type': 'text', 'text': f"CONFIDENCE: Medium\nTAGS: Analysis unavailable, Try again later\n{data['ticker']} forecast: {data['signal']} signal with 30-day target of ${data['p30']} ({data['pct30']}% change from current price of ${data['cur']}."}]})

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker', '').strip().upper()
    period = request.args.get('period', '1y')
    days   = int(request.args.get('days', 30))

    if not ticker:
        return jsonify({"error": "Please provide a ticker symbol, e.g. ?ticker=AAPL"}), 400

    valid_periods = ['3mo', '6mo', '1y', '2y']
    if period not in valid_periods:
        period = '1y'

    days = max(7, min(days, 60))

    try:
        result = run_prediction(ticker, period=period, forecast_days=days)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Stock Predictor API on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', debug=False, port=port)