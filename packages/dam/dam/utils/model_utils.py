import asyncio
import functools
import logging
from typing import Any, Callable

import torch  # For torch.cuda.OutOfMemoryError

logger = logging.getLogger(__name__)


def oom_retry_batch_adjustment(
    max_retries: int = 3, batch_size_reduction_factor: float = 0.5, min_batch_size: int = 1
) -> Callable:
    """
    Decorator for an async function that performs batched model inference.
    It catches PyTorch CUDA OOM errors and retries with a smaller batch size.

    The decorated function is expected to:
    1. Accept a `batch_size` keyword argument.
    2. Process a list of items (e.g., `items_to_process: List[Any]`).
       The decorator doesn't directly interact with how items are batched internally
       by the decorated function, but it adjusts the `batch_size` kwarg for retries.
       It's up to the decorated function to respect this `batch_size`.
    3. Raise `torch.cuda.OutOfMemoryError` (or a compatible error) on OOM.

    Args:
        max_retries: Maximum number of retries after OOM.
        batch_size_reduction_factor: Factor to reduce batch_size by on each retry (e.g., 0.5 for halving).
        min_batch_size: The minimum batch size to attempt.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            original_batch_size = kwargs.get("batch_size")
            if original_batch_size is None:
                logger.warning(
                    f"oom_retry_batch_adjustment used on {func.__name__} but 'batch_size' kwarg not found. "
                    "Retry logic will not adjust batch size effectively."
                )
                # Proceed without batch adjustment, but retries might still happen if OOM occurs
                # and the function can somehow recover or if the error is transient.
                # However, without batch_size, it can't adapt.
                current_batch_size = -1  # Indicates no batch_size to adjust
            else:
                current_batch_size = original_batch_size

            retries = 0
            last_exception = None

            while retries <= max_retries:
                try:
                    if current_batch_size != -1:  # Only update kwarg if it was originally present
                        kwargs["batch_size"] = current_batch_size

                    logger.debug(
                        f"Attempting {func.__name__} with batch_size: {current_batch_size if current_batch_size != -1 else 'N/A'}. Retry: {retries}/{max_retries}"
                    )
                    return await func(*args, **kwargs)

                except torch.cuda.OutOfMemoryError as e:  # Specific to PyTorch CUDA
                    last_exception = e
                    logger.warning(
                        f"OOM error in {func.__name__} with batch_size {current_batch_size if current_batch_size != -1 else 'N/A'}. "
                        f"Retrying ({retries + 1}/{max_retries}). Error: {e}"
                    )
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) reached for {func.__name__} after OOM. Giving up.")
                        raise last_exception

                    if current_batch_size == -1:  # Cannot adjust batch size
                        logger.error(
                            f"OOM in {func.__name__}, but no 'batch_size' kwarg to adjust. Retrying without change, may loop."
                        )
                        await asyncio.sleep(1)  # Small delay before retrying same params
                        continue

                    new_batch_size = int(current_batch_size * batch_size_reduction_factor)
                    current_batch_size = max(min_batch_size, new_batch_size)

                    if current_batch_size == kwargs.get("batch_size") and current_batch_size == min_batch_size:
                        # If batch size is already at minimum and still OOMing
                        logger.error(
                            f"OOM error in {func.__name__} even with minimum batch_size {min_batch_size}. Giving up."
                        )
                        raise last_exception

                    logger.info(f"Reducing batch size to {current_batch_size} for next attempt.")
                    # Optional: Add a small delay or GPU cache clearing before retrying
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    await asyncio.sleep(0.5)  # Brief pause

                except Exception as e:  # Catch other errors
                    logger.error(f"Non-OOM exception in {func.__name__} during retry wrapper: {e}", exc_info=True)
                    raise  # Re-throw non-OOM errors immediately

            # Should not be reached if logic is correct, but as a fallback:
            if last_exception:
                raise last_exception
            # This line would only be hit if original_batch_size was None and no exceptions occurred,
            # which implies the initial call succeeded.
            # However, the return is inside the try block.
            # Adding for completeness, though it implies an issue if reached.
            return None

        return wrapper

    return decorator


# Example of how a service function might use this:
# class MyService:
#     @oom_retry_batch_adjustment(max_retries=2, batch_size_reduction_factor=0.5)
#     async def process_items_with_model(self, items: List[Any], model: Any, batch_size: int):
#         results = []
#         for i in range(0, len(items), batch_size):
#             batch = items[i:i+batch_size]
#             # Simulate model processing that might OOM
#             # if len(batch) > 10 and torch.cuda.is_available(): # Simulate OOM with larger batches
#             #     raise torch.cuda.OutOfMemoryError("Simulated OOM")
#             logger.info(f"Processing batch of size {len(batch)} (requested {batch_size})")
#             await asyncio.sleep(0.1 * len(batch)) # Simulate work
#             results.extend([f"processed_{item}" for item in batch])
#         return results

# async def main_example_oom():
#     service = MyService()
#     my_items = list(range(50)) # 50 items
#     mock_model = "my_model_instance"

#     try:
#         # Initial call with a potentially large batch size
#         processed_results = await service.process_items_with_model(my_items, mock_model, batch_size=32)
#         print(f"Successfully processed: {len(processed_results)} items.")
#     except Exception as e:
#         print(f"Failed after retries: {e}")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     # To run example:
#     # asyncio.run(main_example_oom())
#     # You'd need to manually trigger the OOM inside the example process_items_with_model
#     # for the retry logic to kick in, e.g. by uncommenting the simulated OOM line.
#     pass

__all__ = ["oom_retry_batch_adjustment"]
