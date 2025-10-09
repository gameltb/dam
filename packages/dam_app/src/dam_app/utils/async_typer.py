"""A custom Typer class that supports async functions for commands and callbacks."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from functools import partial, wraps
from typing import Any

from typer import Typer


class AsyncTyper(Typer):
    """A Typer subclass that allows async functions to be used as commands."""

    @staticmethod
    def maybe_run_async(decorator: Callable[..., Any], func: Callable[..., Any]) -> Any:
        """
        Wrap a function to be run in an asyncio event loop if it is a coroutine.

        Args:
            decorator: The Typer decorator to apply.
            func: The function to wrap.

        Returns:
            The decorated function.

        """
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            def runner(*args: Any, **kwargs: Any) -> Any:
                # This is the simplest version. If a test calling this is async,
                # it will fail with "asyncio.run() cannot be called from a running event loop".
                # If the test is sync, this will work.
                return asyncio.run(func(*args, **kwargs))

            decorator(runner)
        else:
            decorator(func)
        return func

    def callback(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        """
        Create a callback that supports async functions.

        Returns:
            A decorated callback function.

        """
        decorator = super().callback(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)

    def command(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        """
        Create a command that supports async functions.

        Returns:
            A decorated command function.

        """
        decorator = super().command(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)
