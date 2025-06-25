import logging
import os
import sys

DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(level: int | str | None = None) -> None:
    """
    Sets up basic logging for the DAM application.

    Args:
        level: The logging level to set. Can be an integer (e.g., logging.INFO),
               a string (e.g., "INFO"), or None. If None, it tries to get
               the level from the DAM_LOG_LEVEL environment variable,
               defaulting to DEFAULT_LOG_LEVEL.
    """
    if level is None:
        env_level = os.environ.get("DAM_LOG_LEVEL", "").upper()
        if hasattr(logging, env_level):
            log_level = getattr(logging, env_level)
        else:
            log_level = DEFAULT_LOG_LEVEL
            if env_level:  # If DAM_LOG_LEVEL was set but invalid
                # Use a basic print here, as logging might not be fully set up
                # or to ensure this specific warning is always visible.
                print(
                    f"Warning: Invalid DAM_LOG_LEVEL '{env_level}'. Defaulting to {logging.getLevelName(log_level)}.",
                    file=sys.stderr,
                )
    elif isinstance(level, str):
        log_level_name = level.upper()
        if hasattr(logging, log_level_name):
            log_level = getattr(logging, log_level_name)
        else:
            log_level = DEFAULT_LOG_LEVEL
            print(
                f"Warning: Invalid log level string '{level}'. Defaulting to {logging.getLevelName(log_level)}.",
                file=sys.stderr,
            )
    else:
        log_level = level

    # Get the root logger or a specific application logger
    # Using a named logger is often better to avoid interfering with other libraries.
    app_logger = logging.getLogger("dam")
    app_logger.setLevel(log_level)

    # Prevent duplicate handlers if setup_logging is called multiple times
    if not app_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        app_logger.addHandler(handler)

    # You might also want to configure the root logger if you want to see logs
    # from dependencies, or ensure it doesn't interfere.
    # For now, focusing on the application's logger.


if __name__ == "__main__":
    # Example usage and test
    print("Testing logging configuration...")
    os.environ["DAM_LOG_LEVEL"] = "DEBUG"
    setup_logging()
    logger = logging.getLogger("dam.core.logging_config_test")
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")

    print("\nTesting with invalid level string...")
    setup_logging(level="INVALID_LEVEL_STRING")  # Should print warning to stderr
    logger.info("Info message after attempting invalid level string.")

    print("\nTesting with direct level INFO...")
    # Need to clear handlers for re-configuration in this test script context
    logging.getLogger("dam").handlers.clear()
    setup_logging(level=logging.INFO)
    logger.debug("This debug message should NOT be seen if level is INFO.")
    logger.info("This info message SHOULD be seen if level is INFO.")

    # Test default without env var
    if "DAM_LOG_LEVEL" in os.environ:
        del os.environ["DAM_LOG_LEVEL"]
    logging.getLogger("dam").handlers.clear()
    print("\nTesting with default log level (no env var)...")
    setup_logging()
    logger.info(
        f"Default log level test. Current level: {logging.getLevelName(logging.getLogger('dam').getEffectiveLevel())}"
    )
    logger.debug("This debug message should not be seen with default INFO level.")

    # Test specific string level
    logging.getLogger("dam").handlers.clear()
    print("\nTesting with specific string level 'WARNING'...")
    setup_logging(level="WARNING")
    effective_level_name = logging.getLevelName(logging.getLogger('dam').getEffectiveLevel())
    logger.info(f"Test log at WARNING. Effective: {effective_level_name}. Info should not appear.")
    logger.warning("This warning message should be seen.")

    print("\nLogging configuration test complete.")
