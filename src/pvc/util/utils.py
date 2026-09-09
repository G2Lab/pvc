from typing import Any, Dict, List
import time
from functools import wraps

def _expect_keys(d: Dict[str, Any], keys: List[str], ctx: str = ""):
    for k in keys:
        if k not in d:
            raise KeyError(f"Missing key '{k}' in {ctx or 'object'}; got keys={list(d.keys())}")

def time_profile(func):
    """
    Decorator that measures and prints the execution time of a function,
    automatically labeled with the function name.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"[TIME] {func.__name__} took {elapsed:.6f} seconds")
    return wrapper

def format_bytes(size_bytes):
    """Format bytes into human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def _log(msg):
    """Print with flush and timestamp to ensure immediate output in SLURM."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)