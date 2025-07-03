from __future__ import annotations

import asyncio
import inspect
from functools import partial, wraps
from typing import Any, Callable

from typer import Typer


class AsyncTyper(Typer):
    @staticmethod
    def maybe_run_async(decorator: Callable, func: Callable) -> Any:
        # Always pass the function (async or sync) directly to the underlying
        # Typer decorator. We will rely on Typer's own handling of async functions
        # when invoked via CliRunner.
        # The decorator (e.g., Typer.command) should handle wrapping if needed
        # and return the callable that Typer expects.
        return decorator(func)

    def callback(self, *args: Any, **kwargs: Any) -> Any:
        decorator = super().callback(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)

    def command(self, *args: Any, **kwargs: Any) -> Any:
        decorator = super().command(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)
