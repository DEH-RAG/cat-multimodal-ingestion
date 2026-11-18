"""The heavy image embedding runs on the core's ingestion lane, not on a hook.

Regression for the stale copy that dispatched through the removed
``run_in_ingestion_executor`` hook: no plugin in the fleet provides it, so
``MadHatter.execute_hook`` raised ``Hook run_in_ingestion_executor not present
in any plugin`` and the image points were silently dropped (the hook runner
swallows the error).

Two paths are pinned:

- the core exposes ``cat.rabbit_hole.run_in_ingestion_executor`` -> it is used;
- the import fails (older core) -> ``asyncio.to_thread`` runs the callable.

The sentinel returned by the callable proves WHICH path ran: a test that only
asserted "no exception" would also pass on the broken hook path, because the
hook runner swallows the error and returns ``None``.
"""

import asyncio
import os
import sys
import threading
import types

import pytest

os.environ.setdefault("CAT_CRYPTO_SALT", "test-salt")
os.environ.setdefault("CAT_CRYPTO_KEY", "test-key-0123456789")

import cat.rabbit_hole as rabbit_hole

from cat.plugins.cat_multimodal_ingestion import ingestion


class _HookOnlyCat:
    """A cat whose plugin manager only knows the removed hook.

    Any ``execute_hook`` call raises the exact production error, so a
    regression back to the hook path fails these tests with the real message
    instead of a generic attribute error.
    """

    class _PluginManager:
        async def execute_hook(self, hook_name, *args, **kwargs):
            raise Exception(f"Hook {hook_name} not present in any plugin")

    plugin_manager = _PluginManager()


@pytest.mark.asyncio
async def test_core_ingestion_lane_is_used(monkeypatch):
    """When the core exposes the lane, the callable runs through it."""
    lane_calls = []

    async def fake_lane(func, *args):
        lane_calls.append((func, args))
        return func(*args)

    async def forbidden_to_thread(*args, **kwargs):
        raise AssertionError("asyncio.to_thread must not be used when the core lane exists")

    monkeypatch.setattr(rabbit_hole, "run_in_ingestion_executor", fake_lane)
    monkeypatch.setattr(asyncio, "to_thread", forbidden_to_thread)

    def heavy():
        return "core-lane-sentinel"

    result = await ingestion._run_in_ingestion_executor(_HookOnlyCat(), heavy)

    assert result == "core-lane-sentinel"
    assert len(lane_calls) == 1
    assert lane_calls[0][0] is heavy


@pytest.mark.asyncio
async def test_falls_back_to_to_thread_when_core_lane_is_missing(monkeypatch):
    """When the core lane cannot be imported, the callable runs on a thread."""
    # Force ``from cat.rabbit_hole import run_in_ingestion_executor`` to fail:
    # the module is present but does not expose the name -> ImportError.
    monkeypatch.setitem(sys.modules, "cat.rabbit_hole", types.ModuleType("cat.rabbit_hole"))

    thread_calls = []
    real_to_thread = asyncio.to_thread

    async def recording_to_thread(func, *args, **kwargs):
        thread_calls.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", recording_to_thread)

    caller_thread = threading.get_ident()
    worker_threads = []

    def heavy(value):
        worker_threads.append(threading.get_ident())
        return f"thread-sentinel:{value}"

    result = await ingestion._run_in_ingestion_executor(_HookOnlyCat(), heavy, "arg")

    assert result == "thread-sentinel:arg"
    assert len(thread_calls) == 1
    assert worker_threads and worker_threads[0] != caller_thread
