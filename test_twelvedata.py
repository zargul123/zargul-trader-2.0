import requests
import json
from scripts.config import TWELVEDATA_API_KEY, TWELVEDATA_MAPPING

def test_symbol(symbol, mapped_symbol):
    print(f'\n{"="*50}')
    print(f'Testing {symbol} -> {mapped_symbol}')
    print(f'{"="*50}')

    url = 'https://api.twelvedata.com/time_series'
    params = {
        'symbol': mapped_symbol,
        'interval': '1h',
        'apikey': TWELVEDATA_API_KEY,
        'outputsize': 10,
        'format': 'JSON'
    }

    print(f'API Key: {TWELVEDATA_API_KEY[:10] if TWELVEDATA_API_KEY else "NOT SET"}...')
    print(f'Request URL: {url}')
    print(f'Request Params: {params}')

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f'Status Code: {response.status_code}')
        print(f'Response Headers: {dict(response.headers)}')

        if response.status_code == 200:
            try:
                data = response.json()
                print(f'Response JSON Keys: {list(data.keys())}')

                if 'values' in data:
                    print(f'✅ Values found: {len(data["values"])} data points')
                    if data['values']:
                        print(f'Sample data point: {data["values"][0]}')
                else:
                    print('❌ No "values" key found')

                if 'message' in data:
                    print(f'API Message: {data["message"]}')
                if 'status' in data:
                    print(f'API Status: {data["status"]}')
                if 'code' in data:
                    print(f'API Code: {data["code"]}')

            except json.JSONDecodeError:
                print(f'❌ Invalid JSON response: {response.text[:200]}...')
        else:
            print(f'❌ HTTP Error: {response.text}')

    except Exception as e:
        print(f'❌ Request failed: {e}')

# Test API key validity first
print(f'TwelveData API Key: {TWELVEDATA_API_KEY if TWELVEDATA_API_KEY else "NOT CONFIGURED"}')

# Test each symbol
for symbol, mapped in TWELVEDATA_MAPPING.items():
    test_symbol(symbol, mapped)