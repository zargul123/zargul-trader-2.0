import requests
import time
from functools import lru_cache

class SocialAnalyzer:
    """
    A dedicated client for fetching social metrics from the LunarCrush API.
    Includes caching to avoid redundant API calls.
    """
    def __init__(self, api_key: str, base_url: str):
        """
        Initializes the analyzer with the API key and base URL.

        Args:
            api_key: The LunarCrush API key.
            base_url: The base URL for the LunarCrush API endpoint.
        """
        if not api_key:
            raise ValueError("LunarCrush API key is not set. Please check your environment variables.")
        
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        })

    # Cache results for 15 minutes (900 seconds) to prevent hitting rate limits
    # and to ensure the same social data is used for the duration of an analysis cycle.
    @lru_cache(maxsize=128)
    def get_social_metrics(self, symbol: str, ttl_hash=None) -> dict:
        """
        Fetches key social metrics for a given asset symbol from the LunarCrush API.
        The ttl_hash is used to bypass the cache for time-sensitive calls.

        Args:
            symbol: The asset symbol (e.g., "BTC", "ETH").
            ttl_hash: A hash representing the current time slice, to control caching.

        Returns:
            A dictionary containing the requested social metrics, or an empty dictionary on failure.
        """
        del ttl_hash # Unused, but necessary for the caching mechanism
        
        # The LunarCrush API typically uses the asset's symbol, not the pair (e.g., 'BTC' instead of 'BTC-USD')
        asset_symbol = symbol.split('-')[0]
        
        endpoint = f"{self.base_url}/public/coins/list/v2"
        params = {
            'limit': 1000, # Fetch a large list to ensure our assets are included
            'sort': 'market_cap_rank'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if 'data' in data and 'coins' in data['data']:
                # Find our specific coin in the list
                for coin in data['data']['coins']:
                    if coin.get('s') == asset_symbol:
                        # Found the coin, return its metrics
                        return {key: coin.get(key, 0) for key in coin}
                
                print(f"⚠️ LunarCrush: Symbol {asset_symbol} not found in the returned list.")
                return {}
            else:
                print(f"⚠️ LunarCrush: 'data' or 'coins' key not found in response for {asset_symbol}")
                return {}

        except requests.exceptions.RequestException as e:
            print(f"❌ LunarCrush API request failed for {asset_symbol}: {e}")
            return {}
        except (KeyError, IndexError) as e:
            print(f"❌ Failed to parse LunarCrush response for {asset_symbol}: {e}")
            return {}

# Helper function to bypass lru_cache for time-sensitive data
def get_ttl_hash(seconds=900):
    """
    Returns the current time slice. Used to invalidate the cache.
    Default is 15 minutes (900 seconds).
    """
    return round(time.time() / seconds)
