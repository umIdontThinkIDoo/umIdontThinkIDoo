"""Regression test for the MCP connection owner-task teardown.

stdio/sse/http MCP transports run anyio task groups whose cancel scopes are
bound to the task that entered them. The manager used to enter the context in
the connect task and call aclose() from the shutdown/request task, which logged
"Attempted to exit cancel scope in a different task than it was entered in" on
every disconnect. The connection lifecycle is now owned by one task that both
enters and exits the context. This test connects to a real builtin stdio server
in one task and disconnects in another, asserting a clean teardown.
"""
import asyncio
import logging
import os
import sys

import pytest

pytest.importorskip("mcp")
from src.mcp_manager import McpManager

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MEMORY_SERVER = os.path.join(_BASE, "mcp_servers", "memory_server.py")


@pytest.mark.skipif(not os.path.exists(_MEMORY_SERVER), reason="builtin memory server missing")
async def test_connect_and_disconnect_in_different_tasks_is_clean(caplog):
    mgr = McpManager()

    # Connect inside a dedicated task (mimics the startup task).
    async def _connect():
        return await mgr.connect_server(
            "memory", "Built-in: Memory", "stdio",
            command=sys.executable, args=[_MEMORY_SERVER],
            env={"PYTHONPATH": _BASE},
        )

    with caplog.at_level(logging.WARNING, logger="src.mcp_manager"):
        ok = await asyncio.create_task(_connect())
        assert ok is True
        assert mgr._tools.get("memory"), "tools should be discovered"

        # Tool call from a *different* task than the owner must still work.
        tool_name = mgr._tools["memory"][0]["name"]
        result = await asyncio.create_task(
            mgr.call_tool(f"mcp__memory__{tool_name}", {})
        )
        assert "exit_code" in result

        # Disconnect from yet another task — the old code raised the cancel-scope
        # error here.
        await asyncio.create_task(mgr.disconnect_server("memory"))
        await asyncio.sleep(0.1)

    cancel_scope_warnings = [
        r.getMessage() for r in caplog.records
        if "cancel scope" in r.getMessage().lower()
    ]
    assert cancel_scope_warnings == [], cancel_scope_warnings

    # Teardown fully cleaned up internal state.
    assert "memory" not in mgr._sessions
    assert "memory" not in mgr._connect_tasks
    assert "memory" not in mgr._close_events
    assert "memory" not in mgr._connections


@pytest.mark.skipif(not os.path.exists(_MEMORY_SERVER), reason="builtin memory server missing")
async def test_disconnect_all_tears_down_every_owner_task():
    mgr = McpManager()
    await mgr.connect_server(
        "memory", "Built-in: Memory", "stdio",
        command=sys.executable, args=[_MEMORY_SERVER], env={"PYTHONPATH": _BASE},
    )
    assert mgr._connect_tasks.get("memory") is not None
    await mgr.disconnect_all()
    assert mgr._sessions == {}
    assert mgr._connect_tasks == {}
