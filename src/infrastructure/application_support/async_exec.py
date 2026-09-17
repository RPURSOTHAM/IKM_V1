"""Async execution helpers used across pipeline stages."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")

_DEFAULT_EXECUTOR: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    global _DEFAULT_EXECUTOR
    if _DEFAULT_EXECUTOR is None:
        _DEFAULT_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="infra-async")
    return _DEFAULT_EXECUTOR


async def run_sync(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking function off the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor(), lambda: fn(*args, **kwargs))


async def gather_bounded(
    coros: Sequence[Awaitable[T]],
    *,
    limit: int = 8,
    return_exceptions: bool = True,
) -> list[Any]:
    """Like asyncio.gather but with a concurrency limit."""
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _wrap(coro: Awaitable[T]) -> Any:
        async with semaphore:
            try:
                return await coro
            except Exception as exc:
                if return_exceptions:
                    return exc
                raise

    return list(await asyncio.gather(*[_wrap(c) for c in coros]))


async def map_sync(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    limit: int = 8,
    return_exceptions: bool = True,
) -> list[Any]:
    """Apply a sync function to many items concurrently via threads."""
    coros = [run_sync(fn, item) for item in items]
    return await gather_bounded(coros, limit=limit, return_exceptions=return_exceptions)


def run_async(coro: Awaitable[T]) -> T:
    """Run a coroutine from sync code when no event loop is running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "run_async() cannot block inside a running event loop; await the coroutine instead"
    )
