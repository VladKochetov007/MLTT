from functools import wraps
from threading import RLock
from enum import Enum
from contextlib import contextmanager
from functools import _make_key
from collections import OrderedDict


class CacheMode(Enum):
    """
    Enumerate for cache mode.
    - `READ_WRITE`: Default: read from and write to cache
    - `READ_ONLY`: Read from cache, but do not write new values
    - `DISABLED`: Fully bypass cache
    """
    READ_WRITE = "READ_WRITE"
    READ_ONLY = "READ_ONLY"
    DISABLED = "DISABLED"

_CACHE_MODE = CacheMode.READ_WRITE

@contextmanager
def cache_mode(mode: CacheMode | str = CacheMode.READ_WRITE):
    """
    Caching context manager.
    Modes:
        - `READ_WRITE`
        - `READ_ONLY`
        - `DISABLED`
    """
    global _CACHE_MODE
    prev_mode = _CACHE_MODE
    _CACHE_MODE = mode
    try:
        yield
    finally:
        _CACHE_MODE = prev_mode

def conditional_lru_cache(maxsize=128, typed=False):
    def decorator(func):        
        cache = {}
        lock = RLock()
        # Use OrderedDict for O(1) access and LRU tracking
        cache_order = OrderedDict()

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = _make_key(args, kwargs, typed)
            global _CACHE_MODE

            if _CACHE_MODE == CacheMode.DISABLED:
                return func(*args, **kwargs)

            with lock:
                if key in cache:
                    # Move to end (most recently used) in O(1)
                    cache_order.move_to_end(key)
                    return cache[key]

                try:
                    result = func(*args, **kwargs)
                except Exception:
                    # Don't cache failed function calls
                    raise

                if _CACHE_MODE == CacheMode.READ_WRITE:
                    cache[key] = result
                    cache_order[key] = None  # Add to end
                    
                    # Evict least recently used if cache is full
                    if len(cache_order) > maxsize:
                        oldest_key = next(iter(cache_order))
                        del cache[oldest_key]
                        del cache_order[oldest_key]

                return result

        def cache_clear():
            with lock:
                cache.clear()
                cache_order.clear()

        wrapper.cache_clear = cache_clear
        return wrapper

    return decorator
