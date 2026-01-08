"""Utility decorators for error handling and retries."""

import functools
import time
from typing import Callable, Any


def robust_llm_call(func: Callable) -> Callable:
    """
    A standard Python decorator that retries the decorated function 
    if the LLM server disconnects or returns a 503/404/Connection error.
    
    Args:
        func: The function to wrap with retry logic
        
    Returns:
        Wrapped function with exponential backoff retry logic
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        max_retries = 5
        base_wait_time = 2  # Start with 2 seconds
        
        for attempt in range(max_retries):
            try:
                # Attempt to execute the decorated method (e.g., forward)
                return func(*args, **kwargs)
                
            except Exception as e:
                error_msg = str(e)
                
                # Check for common server-side transient errors
                # We also catch "404" specifically because sometimes routers momentarily lose the model
                is_transient = (
                    "503" in error_msg or 
                    "Service Temporarily Unavailable" in error_msg or 
                    "Connection error" in error_msg or
                    "404" in error_msg  # Sometimes helpful for momentary router glitches
                )
                
                if is_transient:
                    wait_time = base_wait_time * (2 ** attempt)  # Exponential backoff: 2, 4, 8...
                    print(
                        f"\n[System] Connection dropped in '{func.__name__}'. "
                        f"Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    # If it's a real code error (e.g., TypeError), crash immediately
                    raise e
                    
        raise Exception(f"Max retries ({max_retries}) reached. The server is persistently unavailable.")
        
    return wrapper