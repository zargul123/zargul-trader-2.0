import time
from functools import wraps

class RateLimiter:
    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.last_called = {}
        self.cooldown = 60 / calls_per_minute

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # Enforce cooldown
            if func_name in self.last_called:
                elapsed = time.time() - self.last_called[func_name]
                if elapsed < self.cooldown:
                    sleep_time = self.cooldown - elapsed
                    time.sleep(sleep_time)

            result = func(*args, **kwargs)
            self.last_called[func_name] = time.time()
            return result
        return wrapper

# Initialize with TwelveData limits (8 calls/min)
twelvedata_limiter = RateLimiter(calls_per_minute=8)