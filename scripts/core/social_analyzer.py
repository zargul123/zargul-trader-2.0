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
        Fetches key social metrics for a given asset symbol from the LunarCrush API
        using the modern v4 endpoint.
        The ttl_hash is used to bypass the cache for time-sensitive calls.

        Args:
            symbol: The asset symbol (e.g., "BTC", "ETH").
            ttl_hash: A hash representing the current time slice, to control caching.

        Returns:
            A dictionary containing the requested social metrics, or an empty dictionary on failure.
        """
        del ttl_hash # Unused, but necessary for the caching mechanism
        
        asset_symbol = symbol.split('-')[0]
        
        # --- CORRECTED V4 ENDPOINT ---
        # The new API allows direct querying for a specific symbol's data.
        endpoint = f"{self.base_url}/coins/{asset_symbol}"
        params = {
            'data': 'assets' # Requesting the 'assets' data points which include social metrics
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            # The new structure is typically nested under 'data' and 'assets'
            if 'data' in data and 'assets' in data:
                metrics = data['data']['assets']
                # The keys might be slightly different in v4, so we safely get them.
                # Example: 'galaxy_score', 'alt_rank', 'social_volume_24h', etc.
                return {key: metrics.get(key, 0) for key in metrics}
            else:
                print(f"⚠️ LunarCrush: 'data' or 'assets' key not found in v4 response for {asset_symbol}")
                return {}

        except requests.exceptions.RequestException as e:
            print(f"❌ LunarCrush API request failed for {asset_symbol}: {e}")
            return {}
        except (KeyError, IndexError) as e:
            print(f"❌ Failed to parse LunarCrush v4 response for {asset_symbol}: {e}")
            return {}

# Helper function to bypass lru_cache for time-sensitive data
def get_ttl_hash(seconds=900):
    """
    Returns the current time slice. Used to invalidate the cache.
    Default is 15 minutes (900 seconds).
    """
    return round(time.time() / seconds)
