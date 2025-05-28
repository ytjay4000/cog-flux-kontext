from contextlib import contextmanager
import time

@contextmanager
def print_timing(operation_name: str):
    """Context manager to time and print the execution time of a block of code.

    Args:
        operation_name: A descriptive name for the operation being timed
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        print(f"{operation_name} took {elapsed_time:.2f} seconds")
