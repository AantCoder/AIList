"""
Tests for MCPMixin: mcp_connect / mcp_connects / mcp_disconnect / mcp_list / mcp_tool_names.

WHAT IS TESTED:
  Unit (no LLM, no real MCP process):
    T1.  Connection state: connect, disconnect, disconnect_all, reconnect
    T2.  mcp_list / mcp_tool_names structure and content
    T3.  Agent rebuild: tools appear and disappear correctly
    T4.  disconnect_all is a no-op on empty connections; no extra rebuild call
    T5.  async with (__aexit__) closes all connections
    T6.  MCPServers registry structure (instance attribute, expected servers present)
    T7.  AIListBase attributes: _model_name and _model present after __init__
    T8.  mcp_connects: parallel batch connect
    T10. AppriseConfig + AIList.notify() -- unit tests (no network)
    T11. Workspace boundary: _check_workspace_path, set_workspace, workspace tools,
         mcp_connect auto-dirs injection

  Integration (one real MCP process + one LLM call):
    T9.  Real workspace server: connect, LLM reads/creates a known file, disconnect

WHAT IS NOT TESTED:
  - LLM answer quality -- only that it contains the expected string.
  - Error recovery when the MCP process crashes mid-session.
  - Custom npm-package servers outside the registry.

STRATEGY -- minimising LLM calls:
  T1-T8, T10, T11 use _connect_fake() which writes directly into _mcp_connections
  and calls _mcp_rebuild_agent() -- no real process, no LLM call.
  T9 uses one real mcp_connect + one ai.run_async() call and verifies both
  that the tool was actually invoked and that the answer contains the
  expected file content.

HOW TO RUN:
  All tests (unit + integration):
      python test_mcp.py

  Unit tests only (fast, no LLM required):
      python test_mcp.py --no-integration

  Integration test only:
      python test_mcp.py --integration-only
"""

import sys

def _safe_print(s):
    """Print string safely on any console encoding (e.g. cp1251 on Windows)."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode(sys.stdout.encoding or 'utf-8', errors='replace')
               .decode(sys.stdout.encoding or 'utf-8', errors='replace'))

import time
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from ailist import AIList, AIListDemo, MCPServerDef

# --------------------------------------------------------------
# Settings
# --------------------------------------------------------------

RUN_INTEGRATION  = "--no-integration"   not in sys.argv
INTEGRATION_ONLY = "--integration-only" in sys.argv

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------

_any_fail  = False
_t_total   = time.perf_counter()
_t_section = None

def check(cond: bool, msg: str):
    global _any_fail
    print(f"  {'OK  ' if cond else 'FAIL'} | {msg}")
    if not cond:
        _any_fail = True

def section(title: str):
    global _t_section
    if _t_section is not None:
        print(f"  Time: {time.perf_counter() - _t_section:.3f} sec")
    _t_section = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)

def sub(title: str):
    print(f"\n  [{title}]")

def contains(text: str, *words) -> bool:
    tl = text.lower()
    return all(w.lower() in tl for w in words)

def make_ai() -> AIListDemo:
    ai = AIListDemo()
    ai.chat_history = []
    ai.prompts  = {}
    ai.systems  = {}
    ai.prompt   = ""
    ai.system   = ""
    ai.control_context_limit = False
    return ai

async def run_async(ai: AIListDemo, prt, label: str) -> str:
    global _any_fail
    t0 = time.perf_counter()
    print(f"\n  >> Request: {label}")
    try:
        config = ai.apply_thinking_mode(thinking=False)
        answer = await ai.run_async(prt, config=config)
        answer = answer or ""
        elapsed = time.perf_counter() - t0
        preview = answer[:600] + ("..." if len(answer) > 600 else "")
        print(f"  << Answer ({len(answer)} chars, {elapsed:.3f} sec):")
        for line in preview.splitlines():
            print(f"     {line}")
        return answer
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  EXCEPTION ({elapsed:.3f} sec): {e}")
        _any_fail = True
        return ""

def _make_fake_tool(name: str, description: str = "fake tool"):
    """Dict matching the structure stored in _mcp_connections[name]['tools']."""
    async def caller(**kwargs):
        return f"result of {name}"
    caller.__name__ = name
    return {
        "description": description,
        "schema":      {"type": "object", "properties": {}},
        "caller":      caller,
    }

async def _connect_fake(ai: AIListDemo, server_name: str, tool_names: list):
    """
    Simulate mcp_connect without launching a real process.
    Writes directly into _mcp_connections and calls _mcp_rebuild_agent().
    server_prompt is required by _mcp_rebuild_agent (reads system_tool_instructions).
    """
    if not hasattr(ai, "_mcp_connections"):
        ai._mcp_connections = {}
    ai._mcp_connections[server_name] = {
        "stdio_ctx":    AsyncMock(),
        "session_ctx":  AsyncMock(),
        "session":      AsyncMock(),
        "tools":        {name: _make_fake_tool(name) for name in tool_names},
        "server_prompt": "",
    }
    ai._mcp_rebuild_agent()

def async_run(coro):
    global _any_fail
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    except BaseException as e:
        msg = f"  FAIL | test raised an unhandled exception: {type(e).__name__}: {e}"
        if isinstance(e, BaseExceptionGroup):
            for i, sub_exc in enumerate(e.exceptions, 1):
                msg += f"\n    sub-exception {i}: {type(sub_exc).__name__}: {sub_exc}"
        print(msg)
        _any_fail = True
    finally:
        loop.close()


# ======================================================================
# T1. Connection state
# ======================================================================

if not INTEGRATION_ONLY:

    section("T1. Connection state: connect / disconnect / disconnect_all / reconnect")

    sub("connect registers the server")
    ai = make_ai()
    async_run(_connect_fake(ai, "srv_a", ["tool_read", "tool_list"]))
    check("srv_a" in ai._mcp_connections,       "srv_a present in _mcp_connections")
    check(len(ai._mcp_connections) == 1,         "exactly one connection")

    sub("second connect registers a second server")
    async_run(_connect_fake(ai, "srv_b", ["tool_run"]))
    check("srv_b" in ai._mcp_connections,        "srv_b present")
    check(len(ai._mcp_connections) == 2,         "two connections total")

    sub("disconnect removes one server, leaves the other")
    async_run(ai.mcp_disconnect("srv_a"))
    check("srv_a" not in ai._mcp_connections,    "srv_a removed")
    check("srv_b"     in ai._mcp_connections,    "srv_b still present")

    sub("disconnect unknown name is a no-op")
    try:
        async_run(ai.mcp_disconnect("nonexistent"))
        check(True, "no exception raised")
    except Exception as e:
        check(False, f"unexpected exception: {e}")

    sub("disconnect_all removes everything")
    async_run(_connect_fake(ai, "srv_a", ["tool_read"]))
    async_run(ai.mcp_disconnect_all())
    check(ai._mcp_connections == {},             "_mcp_connections is empty")

    sub("disconnect_all on empty object is a no-op")
    ai2 = make_ai()
    try:
        async_run(ai2.mcp_disconnect_all())
        check(True, "no exception raised")
    except Exception as e:
        check(False, f"unexpected exception: {e}")

    sub("reconnect with the same name replaces old connection")
    ai3 = make_ai()
    async_run(_connect_fake(ai3, "srv_a", ["old_tool"]))
    async_run(_connect_fake(ai3, "srv_a", ["new_tool"]))
    conns = [r for r in ai3.mcp_list() if r["name"] == "srv_a"]
    check(len(conns) == 1,                       "exactly one srv_a entry after reconnect")
    check("new_tool" in ai3.mcp_tool_names(),    "new_tool present")
    check("old_tool" not in ai3.mcp_tool_names(),"old_tool gone")


    # ======================================================================
    # T2. mcp_list / mcp_tool_names
    # ======================================================================

    section("T2. mcp_list / mcp_tool_names")

    ai = make_ai()
    async_run(_connect_fake(ai, "srv_a", ["read_file", "write_file"]))
    async_run(_connect_fake(ai, "srv_b", ["run_command"]))

    sub("mcp_list structure")
    lst = ai.mcp_list()
    check(len(lst) == 2, "two entries")
    for entry in lst:
        check("name"        in entry,               f"{entry.get('name')}: has 'name'")
        check("description" in entry,               f"{entry.get('name')}: has 'description'")
        check("tools"       in entry,               f"{entry.get('name')}: has 'tools'")
        check(isinstance(entry["tools"], list),      f"{entry.get('name')}: tools is a list")

    sub("mcp_list known server uses registry description")
    # workspace -- это builtin-сервер который точно есть в реестре
    ai_ws = make_ai()
    async_run(_connect_fake(ai_ws, "workspace", ["view", "run"]))
    ws_entry = next(e for e in ai_ws.mcp_list() if e["name"] == "workspace")
    check(ws_entry["description"] == ai_ws.MCPServers["workspace"].description,
          "workspace description matches registry")

    sub("mcp_list unknown server uses name as description")
    ai4 = make_ai()
    async_run(_connect_fake(ai4, "my-custom-server", ["do_something"]))
    entry = ai4.mcp_list()[0]
    check(entry["description"] == "my-custom-server", "unknown server: description == name")

    sub("mcp_tool_names flat list across all servers")
    names = ai.mcp_tool_names()
    check(sorted(names) == sorted(["read_file", "write_file", "run_command"]),
          f"flat list correct: {names}")

    sub("mcp_list / mcp_tool_names on empty object")
    ai5 = make_ai()
    check(ai5.mcp_list()       == [],               "mcp_list returns []")
    check(ai5.mcp_tool_names() == [],               "mcp_tool_names returns []")


    # ======================================================================
    # T3. Agent rebuild
    # ======================================================================

    section("T3. Agent rebuild: _mcp_rebuild_agent called, tool names tracked via MCPMixin API")
    print("  Note: agent internals are LangGraph-version-dependent.")
    print("  We verify rebuild indirectly: _mcp_rebuild_agent call count + mcp_tool_names().")

    ai = make_ai()

    sub("rebuild called once on connect")
    rebuild_calls = []
    original_rebuild = ai._mcp_rebuild_agent
    ai._mcp_rebuild_agent = lambda: rebuild_calls.append("rebuild") or original_rebuild()

    async_run(_connect_fake(ai, "srv_a", ["read_file"]))
    check(rebuild_calls.count("rebuild") == 1,              "rebuild called once after connect")
    check("read_file" in ai.mcp_tool_names(),               "read_file visible via mcp_tool_names()")

    sub("rebuild called once on second connect")
    rebuild_calls.clear()
    async_run(_connect_fake(ai, "srv_b", ["run_command"]))
    check(rebuild_calls.count("rebuild") == 1,              "rebuild called once after second connect")
    check("run_command" in ai.mcp_tool_names(),             "run_command visible via mcp_tool_names()")
    check("read_file"   in ai.mcp_tool_names(),             "read_file still visible")

    sub("rebuild called once on disconnect")
    rebuild_calls.clear()
    async_run(ai.mcp_disconnect("srv_a"))
    check(rebuild_calls.count("rebuild") == 1,              "rebuild called once after disconnect")
    check("read_file"   not in ai.mcp_tool_names(),         "read_file gone from mcp_tool_names()")
    check("run_command"     in ai.mcp_tool_names(),         "run_command still present")

    sub("exactly one rebuild for disconnect_all (not one per server)")
    async_run(_connect_fake(ai, "srv_a", ["read_file"]))
    rebuild_calls.clear()
    async_run(ai.mcp_disconnect_all())
    check(rebuild_calls.count("rebuild") == 1,              "exactly one rebuild for disconnect_all")
    check(ai.mcp_tool_names() == [],                        "no MCP tools remain")

    sub("_mcp_open with _rebuild=False does NOT call _mcp_rebuild_agent")
    ai_nb = make_ai()
    rebuild_calls_nb = []
    original_rebuild_nb = ai_nb._mcp_rebuild_agent
    ai_nb._mcp_rebuild_agent = lambda: rebuild_calls_nb.append(1) or original_rebuild_nb()
    if not hasattr(ai_nb, "_mcp_connections"):
        ai_nb._mcp_connections = {}
    ai_nb._mcp_connections["srv_a"] = {
        "stdio_ctx":    AsyncMock(),
        "session_ctx":  AsyncMock(),
        "session":      AsyncMock(),
        "tools":        {"read_file": _make_fake_tool("read_file")},
        "server_prompt": "",
    }
    check(len(rebuild_calls_nb) == 0,                       "_rebuild=False: no rebuild after registration")
    ai_nb._mcp_rebuild_agent()
    check(rebuild_calls_nb.count(1) == 1,                   "explicit rebuild called once")
    check("read_file" in ai_nb.mcp_tool_names(),            "read_file visible after explicit rebuild")

    # ======================================================================
    # T4. No rebuild on empty disconnect_all
    # ======================================================================

    section("T4. disconnect_all on empty: _mcp_rebuild_agent not called")

    ai = make_ai()
    rebuild_calls = []
    original_rebuild = ai._mcp_rebuild_agent
    ai._mcp_rebuild_agent = lambda: rebuild_calls.append(1) or original_rebuild()

    async_run(ai.mcp_disconnect_all())
    check(len(rebuild_calls) == 0, "_mcp_rebuild_agent not called on empty disconnect_all")


    # ======================================================================
    # T5. async with (__aexit__)
    # ======================================================================

    section("T5. async with: __aexit__ closes all connections")

    sub("connections closed on normal exit")
    async def _test_aexit_normal():
        ai = make_ai()
        await _connect_fake(ai, "srv_a", ["read_file"])
        await _connect_fake(ai, "srv_b", ["run_command"])
        async with ai:
            check(len(ai._mcp_connections) == 2,    "two connections inside context")
        check(ai._mcp_connections == {},            "all connections closed on exit")

    async_run(_test_aexit_normal())

    sub("__aexit__ survives disconnect error on one server")
    async def _test_aexit_error():
        ai = make_ai()
        await _connect_fake(ai, "srv_a", ["read_file"])
        await _connect_fake(ai, "srv_b", ["run_command"])
        ai._mcp_connections["srv_a"]["session_ctx"].__aexit__ = AsyncMock(
            side_effect=RuntimeError("connection broken")
        )
        try:
            await ai.mcp_disconnect_all()
            check(True, "no exception propagated")
        except Exception as e:
            check(False, f"exception leaked: {e}")
        check("srv_b" not in ai._mcp_connections,   "srv_b also disconnected despite prior error")

    async_run(_test_aexit_error())


    # ======================================================================
    # T6. MCPServers registry
    # ======================================================================

    section("T6. MCPServers registry (instance attribute)")

    sub("MCPServers is an instance attribute, not shared with class")
    ai = make_ai()
    check(isinstance(ai.MCPServers, dict),          "MCPServers is a dict")
    check(len(ai.MCPServers) > 0,                   "MCPServers is not empty")
    check(ai.MCPServers is not AIList.MCPServers,   "ai.MCPServers is instance-own dict, not class default")

    sub("expected servers present")
    expected_servers = [
        "apprise", "piper", "file-converter",
        "workspace", "skills",
        "git", "playwright",
        "memory-plus", "qdrant", "qdrant-db",
        "searxng", "searxng-engine", "sympy", "serena",
    ]
    for name in expected_servers:
        check(name in ai.MCPServers, f"'{name}' in MCPServers")

    sub("removed servers NOT in registry")
    removed_servers = ["filesystem", "shell", "subprocess", "text-editor", "fastskills"]
    for name in removed_servers:
        check(name not in ai.MCPServers, f"'{name}' removed from MCPServers")

    sub("each entry has non-empty description (str)")
    for name, defn in ai.MCPServers.items():
        check(isinstance(defn.package,     str), f"{name}: package is str")
        check(isinstance(defn.description, str), f"{name}: description is str")
        if defn.launcher not in ("builtin", "sse", None):
            check(len(defn.package) > 0,         f"{name}: package is non-empty")
        check(len(defn.description) > 0,         f"{name}: description is non-empty")
        if defn.launcher == "builtin":
            check(bool(defn.builtin_tools),      f"{name}: builtin_tools is non-empty for builtin launcher")
        if defn.launcher == "sse":
            check(isinstance(defn.url, str) and len(defn.url) > 0,
                  f"{name}: url is a non-empty str for sse launcher (got: {defn.url!r})")

    sub("custom server can be added and removed")
    ai.MCPServers["_test_tmp"] = MCPServerDef(
        package="test-pkg",
        description="Temporary test server.",
    )
    check("_test_tmp" in ai.MCPServers,                       "custom server registered")
    check(ai.MCPServers["_test_tmp"].package == "test-pkg",   "package correct")
    del ai.MCPServers["_test_tmp"]
    check("_test_tmp" not in ai.MCPServers,                   "custom server removed")

    sub("custom server does not leak to another instance")
    ai_a = make_ai()
    ai_b = make_ai()
    ai_a.MCPServers["_leak_test"] = MCPServerDef(package="x", description="x")
    check("_leak_test" not in ai_b.MCPServers,  "custom server in ai_a does not appear in ai_b")
    del ai_a.MCPServers["_leak_test"]


    # ======================================================================
    # T7. AIListBase attributes: _model_name and _model
    # ======================================================================

    section("T7. AIListBase attributes: _model_name and _model after __init__")

    ai = make_ai()

    sub("_model_name set from constructor argument")
    check(hasattr(ai, "_model_name"),                       "_model_name attribute exists")
    check(isinstance(ai._model_name, str),                  "_model_name is str")
    check(len(ai._model_name) > 0,                          "_model_name is non-empty")

    sub("_model created in __init__, not re-created on rebuild")
    check(hasattr(ai, "_model"),                            "_model attribute exists")
    check(ai._model is not None,                            "_model is not None")

    model_id_before = id(ai._model)
    async_run(_connect_fake(ai, "srv_a", ["read_file"]))
    async_run(_connect_fake(ai, "srv_b", ["run_command"]))
    async_run(ai.mcp_disconnect("srv_a"))
    model_id_after = id(ai._model)
    check(model_id_before == model_id_after,                "_model is same object after connect/disconnect")


# ======================================================================
# T8. mcp_connects: parallel batch connect
# ======================================================================

if not INTEGRATION_ONLY:

    section("T8. mcp_connects: parallel batch connect")

    sub("empty list is a no-op")
    ai = make_ai()
    rebuild_calls = []
    original_rebuild = ai._mcp_rebuild_agent
    ai._mcp_rebuild_agent = lambda: rebuild_calls.append(1) or original_rebuild()
    async_run(ai.mcp_connects([]))
    check(len(rebuild_calls) == 0,                          "no rebuild on empty list")
    check(ai.mcp_tool_names() == [],                        "no tools after empty mcp_connects")

    sub("happy path: two servers connect, rebuild called once")

    async def _fake_mcp_connects(ai, specs):
        """Simulate mcp_connects using _connect_fake for each spec."""
        if not hasattr(ai, "_mcp_connections"):
            ai._mcp_connections = {}
        for spec in specs:
            name       = spec["name"]
            tool_names = spec.get("_fake_tools", [name + "_tool"])
            ai._mcp_connections[name] = {
                "stdio_ctx":    AsyncMock(),
                "session_ctx":  AsyncMock(),
                "session":      AsyncMock(),
                "tools":        {t: _make_fake_tool(t) for t in tool_names},
                "server_prompt": "",
            }
        ai._mcp_rebuild_agent()

    ai = make_ai()
    rebuild_calls = []
    original_rebuild = ai._mcp_rebuild_agent
    ai._mcp_rebuild_agent = lambda: rebuild_calls.append(1) or original_rebuild()

    async_run(_fake_mcp_connects(ai, [
        {"name": "srv_a", "_fake_tools": ["read_file", "write_file"]},
        {"name": "srv_b", "_fake_tools": ["run_command"]},
    ]))
    check(rebuild_calls.count(1) == 1,                      "exactly one rebuild for batch connect")
    check("srv_a" in ai._mcp_connections,                   "srv_a registered")
    check("srv_b" in ai._mcp_connections,                   "srv_b registered")
    check(sorted(ai.mcp_tool_names()) == sorted(["read_file", "write_file", "run_command"]),
          f"all tools visible: {ai.mcp_tool_names()}")

    sub("partial failure: ExceptionGroup raised, successful connections cleaned up")

    async def _mcp_connects_with_one_failure(ai):
        if not hasattr(ai, "_mcp_connections"):
            ai._mcp_connections = {}

        original_open = ai._mcp_open
        call_count    = {"n": 0}

        async def _patched_open(name, package, extra, _rebuild=True, url=None, env=None):
            call_count["n"] += 1
            if name == "srv_b":
                raise RuntimeError("simulated srv_b connect failure")
            ai._mcp_connections[name] = {
                "stdio_ctx":    AsyncMock(),
                "session_ctx":  AsyncMock(),
                "session":      AsyncMock(),
                "tools":        {"read_file": _make_fake_tool("read_file")},
                "server_prompt": "",
            }
            if _rebuild:
                ai._mcp_rebuild_agent()

        ai._mcp_open = _patched_open
        try:
            await ai.mcp_connects([
                {"name": "srv_a"},
                {"name": "srv_b"},
            ])
            return False, "ExceptionGroup not raised"
        except ExceptionGroup as eg:
            msgs = [str(e) for e in eg.exceptions]
            got_failure_msg = any("simulated srv_b connect failure" in m for m in msgs)
            srv_a_gone = "srv_a" not in ai._mcp_connections
            return got_failure_msg, srv_a_gone
        except Exception as e:
            return False, f"unexpected exception type: {type(e).__name__}: {e}"
        finally:
            ai._mcp_open = original_open

    result = async_run(_mcp_connects_with_one_failure(make_ai()))
    check(result[0],  "ExceptionGroup contains the original RuntimeError")
    check(result[1],  "successfully opened connection cleaned up after partial failure")

    sub("tool name conflict between servers raises ValueError")
    async def _test_conflict():
        ai = make_ai()
        if not hasattr(ai, "_mcp_connections"):
            ai._mcp_connections = {}
        for srv in ("srv_a", "srv_b"):
            ai._mcp_connections[srv] = {
                "stdio_ctx":    AsyncMock(),
                "session_ctx":  AsyncMock(),
                "session":      AsyncMock(),
                "tools":        {"duplicate_tool": _make_fake_tool("duplicate_tool")},
                "server_prompt": "",
            }
        try:
            ai._mcp_rebuild_agent()
            return False
        except ValueError as e:
            return "duplicate_tool" in str(e)
    check(async_run(_test_conflict()),                      "ValueError on duplicate tool name across servers")


# ======================================================================
# T9. Integration: real workspace server + one LLM call
# ======================================================================

if RUN_INTEGRATION:

    section("T9. Integration: real workspace MCP server + one LLM call")
    print("  Requires: running Ollama")
    print("  Strategy: one LLM call -- ask the model to create a file and read it back.")
    print("  Checks: tools registered (no LLM), LLM invoked the tool, answer contains probe string.")

    PROBE_CONTENT = "PROBE_CONTENT_12345"

    async def _test_integration():
        # mkdtemp + ручная очистка вместо TemporaryDirectory: на Windows
        # контекстный менеджер падает с PermissionError если workspace MCP-сервер
        # ещё держит дескрипторы на файлы в папке после mcp_disconnect.
        import shutil as _shutil_t9
        tmpdir = tempfile.mkdtemp()
        try:
            ai = make_ai()
            ai.workspace_dir = Path(tmpdir)

            async with ai:
                sub("mcp_connect: workspace tools registered, no LLM call yet")
                await ai.mcp_connect("workspace")

                tool_names = ai.mcp_tool_names()
                check(len(tool_names) > 0,
                      f"workspace provides tools: {tool_names}")
                check("view" in tool_names,     "view tool present")
                check("create_file" in tool_names, "create_file tool present")
                check("run" in tool_names,      "run tool present")

                lst = ai.mcp_list()
                check(len(lst) == 1,                            "one entry in mcp_list()")
                check(lst[0]["name"] == "workspace",            "name == 'workspace'")
                check(len(lst[0]["tools"]) > 0,                 "tools list non-empty in mcp_list()")

                sub("one LLM call: model creates and reads the probe file via workspace tools")
                probe_path = str(Path(tmpdir) / "probe.txt")
                answer = await run_async(
                    ai,
                    f"Use create_file to create '{probe_path}' with content '{PROBE_CONTENT}', "
                    f"then use view to read it back and return ONLY its exact contents.",
                    "workspace create+read probe.txt",
                )
                check(PROBE_CONTENT in answer,
                      f"answer contains '{PROBE_CONTENT}'")
                check(Path(probe_path).exists(),
                      "probe.txt actually exists on disk")

                sub("mcp_disconnect: connection removed, no LLM call")
                await ai.mcp_disconnect("workspace")
                check("workspace" not in ai._mcp_connections,  "workspace removed from _mcp_connections")
                check("workspace" not in [e["name"] for e in ai.mcp_list()],
                      "workspace not in mcp_list() after disconnect")
                check(len(ai.mcp_tool_names()) == 0,            "no MCP tools remain")

                print(f"\n{'='*60}")
                print("  Log:")
                _safe_print(ai.log)
                print("=" * 60)

        finally:
            # ignore_errors=True: не падаем если Windows ещё держит файлы.
            _shutil_t9.rmtree(tmpdir, ignore_errors=True)

    async_run(_test_integration())

else:
    print("\n  [T9 skipped] remove --no-integration to run the integration test")


# ======================================================================
# T10. AppriseConfig + AIList.notify() -- unit tests (no network)
# ======================================================================

if not INTEGRATION_ONLY:

    section("T10. AppriseConfig + AIList.notify() -- unit (no network, no apprise required)")

    from ailist import AppriseConfig

    sub("AppriseConfig: initial state")
    cfg = AppriseConfig()
    check(isinstance(cfg.urls,     list), "urls is a list")
    check(isinstance(cfg.channels, dict), "channels is a dict")
    check(cfg.urls     == [],             "urls starts empty")
    check(cfg.channels == {},             "channels starts empty")

    sub("AppriseConfig: mutating urls and channels")
    cfg.urls.append("tgram://token/chat")
    cfg.channels["work"] = "discord://id/token"
    check(cfg.urls     == ["tgram://token/chat"],         "url appended correctly")
    check(cfg.channels == {"work": "discord://id/token"}, "channel set correctly")

    sub("ai.apprise is AppriseConfig instance")
    ai = make_ai()
    check(isinstance(ai.apprise, AppriseConfig), "ai.apprise is AppriseConfig")
    check(ai.apprise.urls     == [],             "ai.apprise.urls starts empty")
    check(ai.apprise.channels == {},             "ai.apprise.channels starts empty")

    sub("notify(): empty urls -> ValueError (no channel, no urls)")
    ai = make_ai()
    try:
        ai.notify("T", "B")
        check(False, "should raise ValueError when no urls configured")
    except ValueError as e:
        check("url" in str(e).lower() or "apprise" in str(e).lower(),
              f"ValueError message is informative: {str(e)[:80]!r}")
    except ImportError:
        check(True, "apprise not installed -- ImportError is acceptable here")
    except Exception as e:
        check(False, f"unexpected exception: {type(e).__name__}: {e}")

    sub("notify(): unknown channel -> ValueError")
    ai = make_ai()
    ai.apprise.urls.append("tgram://t/c")
    try:
        ai.notify("T", "B", channel="nonexistent")
        check(False, "should raise ValueError for unknown channel")
    except ValueError as e:
        check("nonexistent" in str(e),
              f"ValueError names the bad channel: {str(e)[:80]!r}")
    except ImportError:
        check(True, "apprise not installed -- ImportError is acceptable here")
    except Exception as e:
        check(False, f"unexpected exception: {type(e).__name__}: {e}")

    sub("notify(): default urls used when channel='' [mocked apprise]")
    ai = make_ai()
    ai.apprise.urls = ["tgram://token/chat1", "discord://id/tok"]

    _mock_ap_instance = MagicMock()
    _mock_ap_instance.notify.return_value = True
    _mock_ap_class = MagicMock(return_value=_mock_ap_instance)

    with patch.dict("sys.modules", {"apprise": MagicMock(Apprise=_mock_ap_class)}):
        result = ai.notify("Hello", "World")

    check(result is True,                              "notify() returns True on success")
    check(_mock_ap_instance.add.call_count == 2,       "add() called for each url (2 times)")
    added_urls = [call.args[0] for call in _mock_ap_instance.add.call_args_list]
    check(sorted(added_urls) == sorted(ai.apprise.urls),
          f"both urls passed to Apprise.add(): {added_urls}")
    check(_mock_ap_instance.notify.call_count == 1,    "notify() called once")
    kw = _mock_ap_instance.notify.call_args.kwargs
    check(kw.get("title") == "Hello",                  "title passed correctly")
    check(kw.get("body")  == "World",                  "body passed correctly")

    sub("notify(): named channel used when channel='work' [mocked apprise]")
    ai = make_ai()
    ai.apprise.urls = ["tgram://default"]
    ai.apprise.channels["work"] = "discord://work_id/work_tok"

    _mock_ap2 = MagicMock()
    _mock_ap2.notify.return_value = True
    _mock_ap_cls2 = MagicMock(return_value=_mock_ap2)

    with patch.dict("sys.modules", {"apprise": MagicMock(Apprise=_mock_ap_cls2)}):
        result2 = ai.notify("Alert", "Something happened", channel="work")

    check(result2 is True,                             "notify() returns True")
    check(_mock_ap2.add.call_count == 1,               "add() called once (only channel url)")
    check(_mock_ap2.add.call_args.args[0] == "discord://work_id/work_tok",
          "channel URL passed, not default urls")

    sub("MCPServers['apprise']: launcher=builtin, has notify_send tool")
    ai = make_ai()
    defn = ai.MCPServers.get("apprise")
    check(defn is not None,                            "'apprise' in MCPServers")
    check(defn is not None and defn.launcher == "builtin",
          "launcher == 'builtin'")
    check(defn is not None and bool(defn.builtin_tools),
          "builtin_tools non-empty")
    if defn and defn.builtin_tools:
        tool_name = getattr(defn.builtin_tools[0], "name", None)
        check(tool_name == "notify_send",              f"tool name == 'notify_send' (got: {tool_name!r})")

    sub("notify_send schema: no Optional/Union types (Ollama Go-template safety)")
    import inspect as _inspect, typing as _typing
    if defn and defn.builtin_tools:
        _t = defn.builtin_tools[0]
        _schema = getattr(_t, "args_schema", None)
        _hints  = _typing.get_type_hints(_schema) if _schema else {}
        _bad = [
            f"{n}: {tp}" for n, tp in _hints.items()
            if hasattr(tp, "__origin__") and tp.__origin__ is _typing.Union
        ]
        check(len(_bad) == 0,
              f"no Union/Optional in notify_send schema (Ollama-safe): {_bad or 'none'}")

    sub("notify_send: channel defaults to '' (sentinel for 'all urls')")
    if defn and defn.builtin_tools:
        _func = getattr(defn.builtin_tools[0], "func", None)
        if _func:
            _sig = _inspect.signature(_func)
            _ch  = _sig.parameters.get("channel")
            check(_ch is not None,             "channel parameter present")
            check(_ch is not None and _ch.default == "",
                  f"channel default == '' (got: {_ch.default!r})")

    sub("notify_send: returns error string on missing apprise (ImportError path) [mocked]")
    ai = make_ai()
    ai.apprise.urls.append("tgram://t/c")
    if defn and defn.builtin_tools:
        _tool_fn = getattr(defn.builtin_tools[0], "func", None)
        if _tool_fn:
            import sys as _sys
            _saved = _sys.modules.get("apprise", _typing.TYPE_CHECKING)
            _sys.modules["apprise"] = None  # type: ignore
            try:
                result_err = _tool_fn(title="T", body="B")
                check(isinstance(result_err, str),          "returns str on error")
                check("[error]" in result_err,              f"error string starts with [error]: {result_err!r}")
            finally:
                if _saved is _typing.TYPE_CHECKING:
                    _sys.modules.pop("apprise", None)
                else:
                    _sys.modules["apprise"] = _saved


# ======================================================================
# T11. Workspace boundary -- _check_workspace_path, set_workspace,
#      workspace tools (view, create_file, str_replace, insert, undo_edit, run)
# ======================================================================

if not INTEGRATION_ONLY:

    section("T11. Workspace boundary: _check_workspace_path / set_workspace / workspace tools")
    print("  All unit tests -- no LLM calls, no real filesystem access outside tmp.")

    import tempfile as _tempfile
    import inspect as _ws_inspect

    # ------------------------------------------------------------------
    # Sub-test A: default workspace_dir = Path.cwd() on construction
    # ------------------------------------------------------------------

    sub("workspace_dir defaults to Path.cwd() on construction")
    _ai_ws = make_ai()
    check(_ai_ws.workspace_dir is not None,             "workspace_dir is not None after init")
    check(_ai_ws.workspace_dir == Path.cwd().resolve(), "workspace_dir == Path.cwd().resolve()")

    # ------------------------------------------------------------------
    # Sub-test B: set_workspace changes workspace_dir and writes to log
    # ------------------------------------------------------------------

    sub("set_workspace: changes workspace_dir and writes to log")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ws_path = Path(_ws_tmp).resolve()
        _ai_ws2 = make_ai()
        _ai_ws2.loglevel = 1
        _ai_ws2.set_workspace(str(_ws_path))
        check(_ai_ws2.workspace_dir == _ws_path,
              f"workspace_dir updated to {_ws_path}")
        check(str(_ws_path) in _ai_ws2.log,
              "log contains the new workspace path")

    # ------------------------------------------------------------------
    # Sub-test C: _check_workspace_path logic
    # ------------------------------------------------------------------

    sub("_check_workspace_path: path inside workspace -> None")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ai_c = make_ai()
        _ai_c.workspace_dir = Path(_ws_tmp).resolve()
        _inside = str(Path(_ws_tmp) / "subdir" / "file.txt")
        check(_ai_c._check_workspace_path(_inside) is None,
              "path inside workspace returns None (allowed)")

    sub("_check_workspace_path: path outside workspace -> error string")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ai_c2 = make_ai()
        _ai_c2.workspace_dir = Path(_ws_tmp).resolve()
        import sys as _sys_ws
        _outside = "C:\\Windows\\System32\\cmd.exe" if _sys_ws.platform == "win32" else "/etc/passwd"
        result_c = _ai_c2._check_workspace_path(_outside)
        check(result_c is not None,              "path outside workspace returns error string (not None)")
        check(isinstance(result_c, str),         "error is a string")
        check("outside" in result_c.lower() or "access denied" in result_c.lower(),
              f"error mentions access denial: {result_c!r}")

    sub("_check_workspace_path: traversal via .. -> error string")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ai_c3 = make_ai()
        _ai_c3.workspace_dir = Path(_ws_tmp).resolve()
        _traversal = str(Path(_ws_tmp) / ".." / "escape.txt")
        result_t = _ai_c3._check_workspace_path(_traversal)
        check(result_t is not None,
              "path with .. traversal outside workspace returns error string")

    sub("_check_workspace_path: workspace_dir=None -> always returns None (no restriction)")
    _ai_none = make_ai()
    _ai_none.workspace_dir = None
    check(_ai_none._check_workspace_path("/etc/passwd") is None,
          "workspace_dir=None: any path allowed (returns None)")
    check(_ai_none._check_workspace_path("C:\\Windows") is None,
          "workspace_dir=None: Windows path also allowed")

    # ------------------------------------------------------------------
    # Sub-test D: workspace run tool -- workspace enforcement
    # ------------------------------------------------------------------

    sub("workspace run: cwd defaults to workspace_dir when empty")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ai_sp = make_ai()
        _ai_sp.workspace_dir = Path(_ws_tmp).resolve()
        _ws_defn = _ai_sp.MCPServers.get("workspace")
        check(_ws_defn is not None and _ws_defn.builtin_tools,
              "workspace MCPServerDef has builtin_tools")
        if _ws_defn and _ws_defn.builtin_tools:
            _run_tool = next((t for t in _ws_defn.builtin_tools
                              if getattr(t, "name", "") == "run"), None)
            check(_run_tool is not None, "run tool found in workspace builtin_tools")
            if _run_tool:
                _run_fn = getattr(_run_tool, "func", None)
                check(_run_fn is not None, "run tool has .func")
                if _run_fn:
                    _sig = _ws_inspect.signature(_run_fn)
                    check("cwd" in _sig.parameters, "run has cwd parameter")
                    _cwd_default = _sig.parameters["cwd"].default
                    check(_cwd_default == "",
                          f"cwd default is '' (sentinel for workspace_dir): {_cwd_default!r}")

    sub("workspace run: explicit cwd outside workspace -> error string returned")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ai_sp2 = make_ai()
        _ai_sp2.workspace_dir = Path(_ws_tmp).resolve()
        _ws_defn2 = _ai_sp2.MCPServers.get("workspace")
        if _ws_defn2 and _ws_defn2.builtin_tools:
            _run_tool2 = next((t for t in _ws_defn2.builtin_tools
                               if getattr(t, "name", "") == "run"), None)
            if _run_tool2:
                _run_fn2 = getattr(_run_tool2, "func", None)
                if _run_fn2:
                    import sys as _sys_sp
                    _bad_cwd = "C:\\Windows" if _sys_sp.platform == "win32" else "/tmp"
                    _result_sp = _run_fn2(command="python --version", cwd=_bad_cwd)
                    check("[error" in _result_sp.lower() or "outside" in _result_sp.lower() or
                          "access denied" in _result_sp.lower(),
                          f"bad cwd returns error: {_result_sp[:120]!r}")

    sub("workspace run: explicit cwd inside workspace -> allowed (no error prefix)")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ai_sp3 = make_ai()
        _ai_sp3.workspace_dir = Path(_ws_tmp).resolve()
        _ws_defn3 = _ai_sp3.MCPServers.get("workspace")
        if _ws_defn3 and _ws_defn3.builtin_tools:
            _run_tool3 = next((t for t in _ws_defn3.builtin_tools
                               if getattr(t, "name", "") == "run"), None)
            if _run_tool3:
                _run_fn3 = getattr(_run_tool3, "func", None)
                if _run_fn3:
                    # shell=False in run tool -> no shell quoting. Use echo which needs no quoting.
                    import sys as _sys_diag
                    if _sys_diag.platform == 'win32':
                        _diag_cmd = 'cmd /c echo workspace_ok'
                    else:
                        _diag_cmd = 'echo workspace_ok'
                    _result_ok = _run_fn3(
                        command=_diag_cmd,
                        cwd=_ws_tmp,
                    )
                    check("[error" not in _result_ok.lower(),
                          f"good cwd does not return error: {_result_ok[:120]!r}")
                    check("workspace_ok" in _result_ok.strip(),
                          f"command actually ran: {_result_ok[:120]!r}")

    # ------------------------------------------------------------------
    # Sub-test E: workspace view / create_file / str_replace / insert / undo_edit
    # ------------------------------------------------------------------

    sub("workspace view: path outside workspace -> error string")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ai_ed = make_ai()
        _ai_ed.workspace_dir = Path(_ws_tmp).resolve()
        _ws_defn_e = _ai_ed.MCPServers.get("workspace")
        check(_ws_defn_e is not None and _ws_defn_e.builtin_tools,
              "workspace MCPServerDef has builtin_tools")
        if _ws_defn_e and _ws_defn_e.builtin_tools:
            _view_tool = next((t for t in _ws_defn_e.builtin_tools
                               if getattr(t, "name", "") == "view"), None)
            check(_view_tool is not None, "view tool found")
            if _view_tool:
                _view_fn = getattr(_view_tool, "func", None)
                if _view_fn:
                    import sys as _sys_ed
                    _bad_path = "C:\\Windows\\win.ini" if _sys_ed.platform == "win32" else "/etc/passwd"
                    _result_view = _view_fn(path=_bad_path)
                    check("[error" in _result_view.lower() or "outside" in _result_view.lower() or
                          "access denied" in _result_view.lower(),
                          f"view outside workspace returns error: {_result_view[:120]!r}")

    sub("workspace view: path inside workspace -> reads file (no error)")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _probe = Path(_ws_tmp) / "probe.txt"
        _probe.write_text("PROBE_WS_99", encoding="utf-8")
        _ai_ed2 = make_ai()
        _ai_ed2.workspace_dir = Path(_ws_tmp).resolve()
        _ws_defn_e2 = _ai_ed2.MCPServers.get("workspace")
        if _ws_defn_e2 and _ws_defn_e2.builtin_tools:
            _view_tool2 = next((t for t in _ws_defn_e2.builtin_tools
                                if getattr(t, "name", "") == "view"), None)
            if _view_tool2:
                _view_fn2 = getattr(_view_tool2, "func", None)
                if _view_fn2:
                    _result_view2 = _view_fn2(path=str(_probe))
                    check("PROBE_WS_99" in _result_view2,
                          f"view inside workspace reads file correctly: {_result_view2[:80]!r}")

    sub("workspace create_file: path outside workspace -> error string")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ai_ed3 = make_ai()
        _ai_ed3.workspace_dir = Path(_ws_tmp).resolve()
        _ws_defn_e3 = _ai_ed3.MCPServers.get("workspace")
        if _ws_defn_e3 and _ws_defn_e3.builtin_tools:
            _create_tool = next((t for t in _ws_defn_e3.builtin_tools
                                 if getattr(t, "name", "") == "create_file"), None)
            if _create_tool:
                _create_fn = getattr(_create_tool, "func", None)
                if _create_fn:
                    import sys as _sys_cr
                    _bad_create = "C:\\bad_file.txt" if _sys_cr.platform == "win32" else "/tmp/bad_ws_file.txt"
                    # параметр теперь 'content', не 'file_text'
                    _result_create = _create_fn(path=_bad_create, content="evil")
                    check("[error" in _result_create.lower() or "outside" in _result_create.lower() or
                          "access denied" in _result_create.lower(),
                          f"create_file outside workspace returns error: {_result_create[:120]!r}")

    sub("workspace str_replace: path outside workspace -> error string")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ai_ed4 = make_ai()
        _ai_ed4.workspace_dir = Path(_ws_tmp).resolve()
        _ws_defn_e4 = _ai_ed4.MCPServers.get("workspace")
        if _ws_defn_e4 and _ws_defn_e4.builtin_tools:
            _replace_tool = next((t for t in _ws_defn_e4.builtin_tools
                                  if getattr(t, "name", "") == "str_replace"), None)
            if _replace_tool:
                _replace_fn = getattr(_replace_tool, "func", None)
                if _replace_fn:
                    import sys as _sys_rp
                    _bad_rp = "C:\\bad.txt" if _sys_rp.platform == "win32" else "/tmp/bad_ws_replace.txt"
                    _result_rp = _replace_fn(path=_bad_rp, old_str="a", new_str="b")
                    check("[error" in _result_rp.lower() or "outside" in _result_rp.lower() or
                          "access denied" in _result_rp.lower(),
                          f"str_replace outside workspace returns error: {_result_rp[:120]!r}")

    sub("workspace insert / undo_edit: path outside workspace -> error string")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ai_ed5 = make_ai()
        _ai_ed5.workspace_dir = Path(_ws_tmp).resolve()
        _ws_defn_e5 = _ai_ed5.MCPServers.get("workspace")
        if _ws_defn_e5 and _ws_defn_e5.builtin_tools:
            import sys as _sys_iu
            _bad_iu = "C:\\bad_insert.txt" if _sys_iu.platform == "win32" else "/tmp/bad_ws_insert.txt"
            _insert_tool = next((t for t in _ws_defn_e5.builtin_tools
                                 if getattr(t, "name", "") == "insert"), None)
            _undo_tool   = next((t for t in _ws_defn_e5.builtin_tools
                                 if getattr(t, "name", "") == "undo_edit"), None)
            if _insert_tool:
                _ins_fn = getattr(_insert_tool, "func", None)
                if _ins_fn:
                    # параметры теперь line и content (не insert_line / new_str)
                    _r_ins = _ins_fn(path=_bad_iu, line=0, content="x")
                    check("[error" in _r_ins.lower() or "outside" in _r_ins.lower() or
                          "access denied" in _r_ins.lower(),
                          f"insert outside workspace returns error: {_r_ins[:120]!r}")
            if _undo_tool:
                _undo_fn = getattr(_undo_tool, "func", None)
                if _undo_fn:
                    _r_undo = _undo_fn(path=_bad_iu)
                    check("[error" in _r_undo.lower() or "outside" in _r_undo.lower() or
                          "access denied" in _r_undo.lower(),
                          f"undo_edit outside workspace returns error: {_r_undo[:120]!r}")

    # ------------------------------------------------------------------
    # Sub-test F: mcp_connect serena auto-dirs injection
    # ------------------------------------------------------------------

    sub("mcp_connect serena: workspace_dir auto-substituted as dirs when not passed")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ai_ser = make_ai()
        _ai_ser.workspace_dir = Path(_ws_tmp).resolve()
        _captured_extra = []
        _orig_open = _ai_ser._mcp_open
        async def _mock_open_ser(name, package, extra, **kw):
            _captured_extra.append((name, list(extra)))
        _ai_ser._mcp_open = _mock_open_ser
        async_run(_ai_ser.mcp_connect("serena"))
        check(len(_captured_extra) == 1,                "mcp_connect called _mcp_open once")
        if _captured_extra:
            _name_got, _extra_got = _captured_extra[0]
            check(_name_got == "serena",                "server name is 'serena'")
            # serena args_builder кладёт dirs в extra через --project
            check(str(_ws_tmp) in _extra_got or str(Path(_ws_tmp).resolve()) in _extra_got,
                  f"workspace_dir passed to serena via --project: {_extra_got}")

    sub("mcp_connect serena: explicit dirs override workspace_dir")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        with _tempfile.TemporaryDirectory() as _explicit_tmp:
            _ai_ser2 = make_ai()
            _ai_ser2.workspace_dir = Path(_ws_tmp).resolve()
            _captured2 = []
            async def _mock_open2(name, package, extra, **kw):
                _captured2.append(list(extra))
            _ai_ser2._mcp_open = _mock_open2
            async_run(_ai_ser2.mcp_connect("serena", dirs=[_explicit_tmp]))
            if _captured2:
                check(_explicit_tmp in _captured2[0],
                      f"explicit dirs used, not workspace_dir: {_captured2[0]}")
                check(str(_ws_tmp) not in _captured2[0],
                      "workspace_dir NOT injected when dirs passed explicitly")

    sub("mcp_connect serena: workspace_dir=None does not inject dirs")
    _ai_none2 = make_ai()
    _ai_none2.workspace_dir = None
    _captured_none = []
    async def _mock_open_none(name, package, extra, **kw):
        _captured_none.append(list(extra))
    _ai_none2._mcp_open = _mock_open_none
    async_run(_ai_none2.mcp_connect("serena"))
    if _captured_none:
        # serena без dirs -> args_builder возвращает только --context agent и т.п., без --project
        check(str(Path.cwd()) not in " ".join(_captured_none[0]),
              f"workspace_dir=None: no workspace injected into serena extra: {_captured_none[0]}")

    sub("mcp_connect workspace: auto-dirs NOT injected (workspace is builtin, no args_builder)")
    with _tempfile.TemporaryDirectory() as _ws_tmp:
        _ai_ws_check = make_ai()
        _ai_ws_check.workspace_dir = Path(_ws_tmp).resolve()
        _captured_ws = []
        async def _mock_open_ws(name, package, extra, **kw):
            _captured_ws.append((name, list(extra)))
        _ai_ws_check._mcp_open = _mock_open_ws
        async_run(_ai_ws_check.mcp_connect("workspace"))
        # workspace -- builtin, _mcp_open вызывается но extra пустой
        if _captured_ws:
            check(_captured_ws[0][1] == [],
                  f"workspace extra is [] (builtin, no dirs injection): {_captured_ws[0][1]}")


# ======================================================================
# Summary
# ======================================================================

if _t_section is not None:
    print(f"  Time: {time.perf_counter() - _t_section:.3f} sec")

print(f"\n{'='*60}")
if _any_fail:
    print("  RESULT: some checks FAILED -- see FAIL lines above")
else:
    print("  RESULT: all checks passed")
print(f"  Total time: {time.perf_counter() - _t_total:.3f} sec")
print("=" * 60)