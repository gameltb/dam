# pyright: ignore
from __future__ import annotations

import asyncio
import inspect
from functools import partial, wraps
from typing import Any, Callable

from typer import Typer


class AsyncTyper(Typer):
    @staticmethod
    def maybe_run_async(decorator: Callable, func: Callable) -> Any:
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

    def callback(self, *args: Any, **kwargs: Any) -> Any:
        decorator = super().callback(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)

    def command(self, *args: Any, **kwargs: Any) -> Any:
        decorator = super().command(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)
