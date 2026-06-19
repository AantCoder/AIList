"""
Integration tests for MCPMixin: real MCP servers + one LLM call each.
Part 1 of 2: T0 (registry smoke) + T1-T11.

WHAT IS TESTED:
  For each MCP server:
    1. mcp_connect succeeds and tools appear in mcp_tool_names().
    2. One LLM call -- agent uses the tool and returns a meaningful answer.
    3. mcp_disconnect removes the tools.

WHAT IS NOT TESTED:
  - Full coverage of every tool in each server.
  - Answer quality beyond containing an expected keyword/pattern.
  - Error recovery, timeouts, or corrupt inputs.

SERVERS COVERED:
  T1. workspace      -- builtin: runs echo via run tool, checks answer contains probe string;
                       also tests python_run, view, create_file.
  T2. sympy          -- builtin: solves x**2 - 4 = 0, checks roots [-2, 2]
  T3. skills         -- builtin: list_skills with temp dir, checks skill name returned
  T4. git            -- npx: creates temp repo, agent reads git log, checks commit msg
  T5. playwright     -- npx: agent navigates to example.com, checks page text
  T6. memory-plus    -- npx: agent stores entity, searches it, checks probe name returned
  T7. searxng        -- sse: agent searches the web, checks result contains query keyword
  T8. qdrant         -- uvx: agent stores text, retrieves it, checks probe string returned
  T9. piper          -- builtin: unit-tests _piper_model_hf_path + synthesize() API;
                       integration: connect -> agent synthesizes short phrase -> WAV exists

ARCHITECTURE NOTES (ailist.py):
  - MCPMixin.MCPServers  -- реестр серверов, задаётся в подклассе (AIList.__init__)
  - MCPServerDef.args_builder -- callable(**kwargs) -> (extra, env, url_override),
    инкапсулирует специфику одного сервера; живёт как @staticmethod в AIList
  - mcp_connect(name, **kwargs) -- kwargs пробрасываются напрямую в args_builder,
    никаких фиксированных параметров dirs/allowed/blocked/url на уровне MCPMixin

REFACTORING CHANGES (relative to the old test):
  - Removed: T1 subprocess -> replaced by T1 workspace (run tool)
  - Removed: T3 filesystem (npx) -> builtin workspace covers file ops
  - Removed: T4 shell (npx) -> run tool in workspace replaces it
  - Removed: fastskills (uvx) -> replaced by builtin "skills" server (T3)
  - Added:   T1 workspace -- tests run, python_run, view, create_file
  - Added:   T3 skills -- tests list_skills builtin

HOW TO RUN:
  All tests in this file:
      python test_integration_part1.py

  Specific tests only (comma-separated):
      python test_integration_part1.py --only=workspace,sympy
      python test_integration_part1.py --only=skills
      python test_integration_part1.py --only=memory-plus
      python test_integration_part1.py --only=searxng
      python test_integration_part1.py --only=qdrant
      python test_integration_part1.py --only=piper
"""

import os
import sys
import pathlib as _pathlib

# Capture script dir BEFORE any tool calls that may os.chdir().
# workspace run tool calls os.chdir(workspace_dir) in the main process,
# which can affect relative path resolution.
_SCRIPT_DIR = (_pathlib.Path(sys.argv[0]).resolve().parent
               if sys.argv[0] else _pathlib.Path(__file__).resolve().parent)

def _safe_print(s):
    """Print string safely on any console encoding (e.g. cp1251 on Windows)."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode(sys.stdout.encoding or 'utf-8', errors='replace')
               .decode(sys.stdout.encoding or 'utf-8', errors='replace'))

import time
import asyncio

from ailist import AIList, AIListDemo, MCPServerDef

# --------------------------------------------------------------
# Settings
# --------------------------------------------------------------

# Parse --only=a,b,c flag
_only_filter: set[str] = set()
for _arg in sys.argv[1:]:
    if _arg.startswith("--only="):
        _only_filter = set(_arg[len("--only="):].split(","))

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------

_any_fail       = False
_t_total        = time.perf_counter()
_t_section      = None
_collected_logs = []   # логи ai.log из каждого теста, печатаются в Summary
_registered_ais = []   # все ai-объекты созданные через make_ai(), для сбора логов

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
    """Return True if all words are found in text (case-insensitive)."""
    tl = text.lower()
    return all(w.lower() in tl for w in words)

def make_ai() -> AIListDemo:
    """Create a clean AIListDemo instance with no system/user prompts."""
    ai = AIListDemo()
    ai.chat_history = []
    ai.prompts  = {}
    ai.systems  = {}
    ai.prompt   = ""
    ai.system   = ""
    ai.control_context_limit = False
    _registered_ais.append(ai)
    return ai

async def run_async(ai: AIListDemo, prt, label: str) -> str:
    """Run one async LLM call, print the answer, return it."""
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
            _safe_print(f"     {line}")
        return answer
    except Exception as e:
        elapsed = time.perf_counter() - t0
        _safe_print(f"  EXCEPTION ({elapsed:.3f} sec): {e}")
        _any_fail = True
        return ""

def async_run(coro):
    """
    Runs a coroutine in a new event loop.
    Catches all exceptions so a single failing test does not abort the whole file.
    Exceptions from mcp_connect (McpError, ImportError, etc.) are printed and
    recorded as failures -- they happen outside run_async() so need handling here.
    After completion collects ai.log from all ai-objects created during the run.
    """
    global _any_fail
    n_before = len(_registered_ais)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=300))
    except BaseException as e:
        # BaseException вместо Exception: ловим также ExceptionGroup (Python 3.11+),
        # который не наследуется от Exception и иначе не перехватывается.
        msg = f"  FAIL | test raised an unhandled exception: {type(e).__name__}: {e}"
        # Для ExceptionGroup печатаем вложенные исключения -- они содержат реальную причину.
        if isinstance(e, BaseExceptionGroup):
            for i, sub_exc in enumerate(e.exceptions, 1):
                msg += f"\n    sub-exception {i}: {type(sub_exc).__name__}: {sub_exc}"
        print(msg)
        _any_fail = True
    finally:
        loop.close()
        # Собираем логи из ai-объектов созданных во время этого вызова
        for ai in _registered_ais[n_before:]:
            if ai.log.strip():
                _collected_logs.append(ai.log)

def should_run(name: str) -> bool:
    """Return True if this test should run given --only filter."""
    return not _only_filter or name in _only_filter


def _docker_available() -> bool:
    """
    Проверяет что Docker daemon доступен синхронным вызовом `docker info`.
    Возвращает False если Docker не запущен или не установлен.
    Используется для пропуска тестов требующих Docker без FAIL.
    """
    import subprocess as _sp
    try:
        r = _sp.run(["docker", "info"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False

# ======================================================================
# T0. MCPServers registry -- smoke test (всегда запускается)
# ======================================================================

section("T0. MCPServers registry (smoke)")
print("  Checks that MCPServers is populated in AIList instance,")
print("  all expected servers are present, and args_builder is set where needed.")

def _test_registry():
    ai = AIListDemo()

    sub("MCPServers is an instance attribute (not class-level default)")
    check(isinstance(ai.MCPServers, dict),      "MCPServers is a dict")
    check(len(ai.MCPServers) > 0,               "MCPServers is not empty")
    # Каждый экземпляр получает свой собственный словарь, не разделяет с классом
    check(ai.MCPServers is not AIList.MCPServers,
          "ai.MCPServers is instance-own dict, not AIList class default")

    sub("Expected servers are registered")
    expected_servers = [
        "apprise", "piper", "file-converter",
        "workspace", "skills",
        "git", "playwright",
        "memory-plus", "qdrant", "qdrant-db",
        "searxng", "searxng-engine", "sympy", "serena",
    ]
    for name in expected_servers:
        check(name in ai.MCPServers, f"'{name}' in MCPServers")

    sub("Removed servers are NOT in the registry")
    removed_servers = ["filesystem", "shell", "subprocess", "text-editor", "fastskills"]
    for name in removed_servers:
        check(name not in ai.MCPServers, f"'{name}' removed from MCPServers")

    sub("args_builder is set for servers that need it")
    servers_with_builder = ["playwright", "memory-plus", "qdrant", "searxng", "serena"]
    for name in servers_with_builder:
        defn = ai.MCPServers.get(name)
        check(defn is not None and callable(defn.args_builder),
              f"'{name}' has callable args_builder")

    sub("servers without args_builder have args_builder=None")
    servers_without_builder = [
        "git", "sympy", "qdrant-db",
        "apprise", "piper", "file-converter", "workspace", "skills",
    ]
    for name in servers_without_builder:
        defn = ai.MCPServers.get(name)
        check(defn is not None and defn.args_builder is None,
              f"'{name}' has args_builder=None")

    sub("builtin servers have launcher='builtin'")
    builtin_servers = ["apprise", "piper", "file-converter", "workspace", "skills", "sympy"]
    for name in builtin_servers:
        defn = ai.MCPServers.get(name)
        check(defn is not None and defn.launcher == "builtin",
              f"'{name}' launcher == 'builtin'")

    sub("workspace has the expected set of builtin tools")
    ws_defn = ai.MCPServers.get("workspace")
    check(ws_defn is not None and ws_defn.builtin_tools is not None,
          "workspace has builtin_tools")
    if ws_defn and ws_defn.builtin_tools:
        ws_tool_names = {getattr(t, "name", None) for t in ws_defn.builtin_tools}
        # File editing
        for name in ("view", "str_replace", "create_file", "write_file", "insert", "undo_edit"):
            check(name in ws_tool_names, f"workspace tool '{name}' present")
        # File navigation
        for name in ("read_files", "list_dir", "list_dir_tree", "find_file", "file_info", "make_dir", "move_file"):
            check(name in ws_tool_names, f"workspace tool '{name}' present")
        # Search and analysis
        for name in ("grep", "search_for_pattern", "head", "tail", "wc"):
            check(name in ws_tool_names, f"workspace tool '{name}' present")
        # Execution
        for name in ("run", "python_run"):
            check(name in ws_tool_names, f"workspace tool '{name}' present")
        # Old names must be gone
        for old_name in ("editor_view", "editor_str_replace", "editor_create", "editor_insert",
                         "editor_undo", "subprocess_run"):
            check(old_name not in ws_tool_names, f"old tool name '{old_name}' is gone")

    sub("skills server: launcher=builtin, system_prompt_tool='list_skills'")
    sk_defn = ai.MCPServers.get("skills")
    check(sk_defn is not None, "'skills' in MCPServers")
    if sk_defn:
        check(sk_defn.launcher == "builtin",             "skills launcher == 'builtin'")
        check(sk_defn.system_prompt_tool == "list_skills",
              f"skills system_prompt_tool == 'list_skills' (got: {sk_defn.system_prompt_tool!r})")
        check(sk_defn.builtin_tools is not None and len(sk_defn.builtin_tools) == 1,
              f"skills has 1 builtin tool (got: {len(sk_defn.builtin_tools) if sk_defn.builtin_tools else '?'})")
        if sk_defn.builtin_tools:
            t_name = getattr(sk_defn.builtin_tools[0], "name", None)
            check(t_name == "list_skills", f"skills tool name == 'list_skills' (got: {t_name!r})")

    sub("ai.skills_dir: instance attribute exists and defaults to 'skills'")
    check(hasattr(ai, "skills_dir"),           "ai.skills_dir attribute exists")
    check(isinstance(ai.skills_dir, str),      "ai.skills_dir is str (not None)")
    check(ai.skills_dir == "skills",           "ai.skills_dir defaults to 'skills'")

    sub("args_builder returns correct tuple shape")
    ai2 = AIListDemo()

    # playwright: dirs -> PLAYWRIGHT_USER_DATA_DIR в env.
    # allowed=["headless"] НЕ добавляет --headless в extra -- пакет v1.0.12 игнорирует этот флаг.
    # Headless-режим управляется через system_prompt, который инструктирует модель
    # всегда передавать headless=true в каждый вызов playwright_navigate.
    extra, env, url = ai2.MCPServers["playwright"].args_builder(
        dirs=[r"C:\profile"], allowed=["headless"]
    )
    check(env.get("PLAYWRIGHT_USER_DATA_DIR") == r"C:\profile" and url is None,
          "playwright args_builder sets PLAYWRIGHT_USER_DATA_DIR in env")
    check(extra == [],
          "playwright args_builder: extra is always [] (headless via system_prompt, not CLI)")
    # system_prompt должен быть установлен в MCPServerDef
    pw_defn = ai2.MCPServers["playwright"]
    check(pw_defn.system_prompt is not None and "headless" in pw_defn.system_prompt.lower(),
          f"playwright MCPServerDef has system_prompt mentioning headless: {pw_defn.system_prompt!r:.80}")

    # qdrant: allowed -> COLLECTION_NAME in env
    extra, env, url = ai2.MCPServers["qdrant"].args_builder(allowed=["my_col"])
    check(env.get("COLLECTION_NAME") == "my_col" and "EMBEDDING_MODEL" in env,
          "qdrant args_builder sets COLLECTION_NAME and EMBEDDING_MODEL")

    # searxng: url kwarg -> url_override
    extra, env, url = ai2.MCPServers["searxng"].args_builder(url="http://custom:9000/sse")
    check(url == "http://custom:9000/sse",
          "searxng args_builder passes url as url_override")

    # serena: no project -> start-mcp-server + context + no dashboard
    extra, env, url = ai2.MCPServers["serena"].args_builder()
    check("start-mcp-server" in extra,  "serena args_builder includes start-mcp-server")
    check("--context" in extra,         "serena args_builder includes --context")
    check("ide" in extra,               "serena default context is 'ide'")
    check("--open-web-dashboard" in extra and "false" in extra,
          "serena no_dashboard=True by default")
    check("--allow-dangerous-shell-commands" not in extra,
          "serena safe mode by default (no --allow-dangerous-shell-commands)")
    check(url is None,                  "serena url_override is None")

    # serena: with project path
    extra_p, _, _ = ai2.MCPServers["serena"].args_builder(dirs=[r"C:\MyProject"])
    check("--project" in extra_p and r"C:\MyProject" in extra_p,
          "serena args_builder(dirs=[path]) -> --project path")

    # serena: dangerous_shell=True
    extra_d, _, _ = ai2.MCPServers["serena"].args_builder(dangerous_shell=True)
    check("--allow-dangerous-shell-commands" in extra_d,
          "serena dangerous_shell=True -> --allow-dangerous-shell-commands")

    # serena: custom context and modes
    extra_m, _, _ = ai2.MCPServers["serena"].args_builder(
        context="ide", modes=["interactive", "no-onboarding"]
    )
    check("ide" in extra_m,             "serena context='ide' passed")
    check("--mode" in extra_m,          "serena modes -> --mode flags")
    check("no-onboarding" in extra_m,   "serena mode 'no-onboarding' passed")

    # serena: python_version is 3.13
    serena_defn = ai2.MCPServers["serena"]
    check(serena_defn.python_version == "3.13",
          f"serena python_version == '3.13' (got: {serena_defn.python_version!r})")
    check(serena_defn.uvx_command == "serena",
          f"serena uvx_command == 'serena' (got: {serena_defn.uvx_command!r})")

    # serena: system_prompt_tool is set to "initial_instructions"
    check(serena_defn.system_prompt_tool == "initial_instructions",
          f"serena system_prompt_tool == 'initial_instructions' (got: {serena_defn.system_prompt_tool!r})")

_test_registry()


# ======================================================================
# T1. workspace (builtin)
# ======================================================================

if should_run("workspace"):

    section("T1. workspace (builtin)")
    print("  Strategy: connect -> test run (echo), python_run, create_file + view")
    print("  No external process: all tools run directly in Python.")

    import tempfile
    from pathlib import Path

    async def _test_workspace():
        import shutil as _shutil_t1
        tmpdir = tempfile.mkdtemp()
        try:
            ai = make_ai()
            ai.workspace_dir = Path(tmpdir)

            sub("mcp_connect: tools registered")
            await ai.mcp_connect("workspace")
            tool_names = ai.mcp_tool_names()
            check(len(tool_names) > 0,
                  f"workspace provides tools: {tool_names}")
            for expected_tool in ("view", "str_replace", "create_file", "run", "python_run",
                                  "list_dir", "grep", "head", "tail"):
                check(expected_tool in tool_names,
                      f"tool '{expected_tool}' present")
            # Старые имена не должны быть в реестре
            check("subprocess_run" not in tool_names, "old 'subprocess_run' is absent")
            check("editor_view"    not in tool_names, "old 'editor_view' is absent")
            check(len(ai.mcp_list()) == 1, "exactly one server in mcp_list()")

            sub("one LLM call: agent runs echo via run tool")
            RUN_PROBE = "WORKSPACE_RUN_PROBE_77"
            answer = await run_async(
                ai,
                f"Use the run tool to execute the command: echo {RUN_PROBE} "
                f"and tell me exactly what it printed.",
                "workspace run echo probe",
            )
            check(contains(answer, RUN_PROBE), f"answer contains '{RUN_PROBE}'")

            # Сбрасываем историю -- следующий вызов независимый
            ai.chat_history = []

            sub("one LLM call: agent creates file then reads it via view")
            FILE_PROBE = "WS_FILE_CONTENT_88"
            probe_path = str(Path(tmpdir) / "probe.txt")
            answer = await run_async(
                ai,
                f"Use create_file to create the file '{probe_path}' "
                f"with content '{FILE_PROBE}', then use view to read it back "
                f"and show me its contents.",
                "workspace create_file + view",
            )
            check(FILE_PROBE in answer, f"answer contains '{FILE_PROBE}'")
            check(Path(probe_path).exists(), "probe.txt actually exists on disk")

            ai.chat_history = []

            sub("one LLM call: agent runs python_run to compute sum")
            answer = await run_async(
                ai,
                "Use python_run to compute: print(sum(range(1, 11))) "
                "and tell me the result.",
                "workspace python_run sum",
            )
            check(contains(answer, "55"), "answer contains '55' (sum of 1..10)")

            sub("mcp_disconnect: tools removed")
            await ai.mcp_disconnect("workspace")
            check(len(ai.mcp_tool_names()) == 0, "no MCP tools remain after disconnect")

        finally:
            _shutil_t1.rmtree(tmpdir, ignore_errors=True)

    async_run(_test_workspace())


# ======================================================================
# T2. sympy (builtin)
# ======================================================================

if should_run("sympy"):

    section("T2. sympy (builtin)")
    print("  Strategy: connect -> ask agent to solve x**2 - 4 = 0 -> check roots [-2, 2]")
    print("  No external process: builtin tool calls sympy.solve() directly in Python.")

    async def _test_sympy():
        ai = make_ai()

        sub("mcp_connect: tools registered")
        await ai.mcp_connect("sympy")
        tool_names = ai.mcp_tool_names()
        check("sympy_solve" in tool_names,
              f"sympy_solve present in tools: {tool_names}")
        check(len(ai.mcp_list()) == 1,          "exactly one server in mcp_list()")

        sub("one LLM call: agent solves x**2 - 4 = 0")
        answer = await run_async(
            ai,
            "Use the sympy_solve tool to solve the equation x**2 - 4 = 0 and tell me the roots.",
            "sympy solve x**2 - 4 = 0",
        )
        check(contains(answer, "2"),            "answer contains root 2")
        check(contains(answer, "-2"),           "answer contains root -2")

        sub("mcp_disconnect: tools removed")
        await ai.mcp_disconnect("sympy")
        check("sympy_solve" not in ai.mcp_tool_names(),
              "sympy_solve gone after disconnect")

    async_run(_test_sympy())


# ======================================================================
# T3. skills (builtin)
# ======================================================================

if should_run("skills"):

    section("T3. skills (builtin)")
    print("  Strategy: create a temp skills dir with a fake SKILL.md ->")
    print("            set ai.skills_dir -> connect -> check list_skills returns skill name.")

    import tempfile
    from pathlib import Path

    SKILL_NAME = "test-skill-ailist"
    SKILL_DESC = "A test skill for AIList integration testing."

    SKILL_MD = f"""---
name: {SKILL_NAME}
description: {SKILL_DESC}
---

# Test Skill

This is a fake skill used in integration tests.
"""

    async def _test_skills():
        import shutil as _shutil_t3
        tmpdir = tempfile.mkdtemp()
        try:
            # Создаём структуру: tmpdir/test-skill-ailist/SKILL.md
            skill_dir = Path(tmpdir) / SKILL_NAME
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")

            ai = make_ai()
            ai.skills_dir = tmpdir   # абсолютный путь строкой

            sub("ai.skills_dir set correctly")
            check(ai.skills_dir == tmpdir, f"skills_dir == {tmpdir!r}")

            sub("mcp_connect: list_skills tool registered")
            await ai.mcp_connect("skills")
            tool_names = ai.mcp_tool_names()
            check("list_skills" in tool_names,
                  f"list_skills present in tools: {tool_names}")
            check(len(ai.mcp_list()) == 1, "exactly one server in mcp_list()")

            sub("one LLM call: agent lists skills, finds the test skill")
            answer = await run_async(
                ai,
                "Call list_skills and tell me the names of all available skills.",
                "skills list_skills",
            )
            # Проверяем что инструмент реально вызывался.
            tool_calls = [
                tc.get("name", "")
                for msg in ai.chat_history
                for tc in (msg.get("data", {}).get("tool_calls") or [])
            ]
            check("list_skills" in tool_calls,
                  f"list_skills was actually called: {tool_calls}")
            # Проверяем вывод инструмента напрямую, а не ответ модели --
            # LLM может заменить ASCII-дефис на unicode non-breaking hyphen (U+2011)
            # при форматировании в markdown, что сломает contains().
            tool_results_raw = [
                msg.get("data", {}).get("content", "")
                for msg in ai.chat_history
                if msg.get("type") == "tool"
                and msg.get("data", {}).get("name", "") == "list_skills"
            ]
            if not tool_results_raw:
                check(False, "list_skills returned no tool result in chat_history")
            else:
                tool_output = " ".join(
                    r if isinstance(r, str) else str(r)
                    for r in tool_results_raw
                )
                check(SKILL_NAME in tool_output,
                      f"list_skills tool output contains skill name '{SKILL_NAME}'")

            sub("mcp_disconnect: tools removed")
            await ai.mcp_disconnect("skills")
            check("list_skills" not in ai.mcp_tool_names(),
                  "list_skills gone after disconnect")

        finally:
            _shutil_t3.rmtree(tmpdir, ignore_errors=True)

    async_run(_test_skills())


# ======================================================================
# T4. git (npx)
# ======================================================================

if should_run("git"):

    section("T4. git (npx)")
    print("  Requires: npx, @cyanheads/git-mcp-server, git installed in system")
    print("  Strategy: create temp repo with one commit -> connect -> ask agent")
    print("            to show git log -> check answer contains commit message.")

    import tempfile
    from pathlib import Path

    GIT_COMMIT_MSG = "Initial commit AIList test"

    async def _test_git():
        import shutil as _shutil_t4
        tmpdir = tempfile.mkdtemp()
        try:

            # Инициализируем репозиторий с одним коммитом прямо в тесте.
            import subprocess as _sp
            def git(*args):
                return _sp.run(
                    ["git"] + list(args),
                    cwd=tmpdir, capture_output=True, text=True, check=True,
                )
            git("init", "-b", "main")
            git("config", "user.email", "test@ailist.local")
            git("config", "user.name",  "AIList Test")
            probe = Path(tmpdir) / "readme.txt"
            probe.write_text("hello from ailist test", encoding="utf-8")
            git("add", ".")
            git("commit", "-m", GIT_COMMIT_MSG)

            async with make_ai() as ai:

                sub("mcp_connect: tools registered")
                # git не имеет args_builder -- kwargs игнорируются сервером;
                # рабочий каталог задаётся через инструмент git_set_working_dir
                await ai.mcp_connect("git")
                tool_names = ai.mcp_tool_names()
                check(len(tool_names) > 0,
                      f"git server provides tools: {tool_names}")
                lst = ai.mcp_list()
                check(len(lst) == 1,          "one entry in mcp_list()")
                check(lst[0]["name"] == "git", "name == 'git'")

                sub("one LLM call: agent reads git log")
                answer = await run_async(
                    ai,
                    f"Use git_set_working_dir to set the working directory to '{tmpdir}', "
                    f"then use git_log to show the commit log "
                    f"and tell me the commit message of the most recent commit. "
                    f"Do NOT call git_init.",
                    "git log",
                )
                check(GIT_COMMIT_MSG in answer,
                      f"answer contains commit message '{GIT_COMMIT_MSG}'")

                sub("mcp_disconnect: tools removed")
                await ai.mcp_disconnect("git")
                check(len(ai.mcp_tool_names()) == 0, "no MCP tools remain")

        finally:
            _shutil_t4.rmtree(tmpdir, ignore_errors=True)

    async_run(_test_git())


# ======================================================================
# T5. playwright (npx)
# ======================================================================

if should_run("playwright"):

    section("T5. playwright (npx)")
    print("  Requires: npx, @executeautomation/playwright-mcp-server")
    print("  Requires: npx playwright install chromium  (first run only)")
    print("  Strategy: navigate to example.com -> get visible text ->")
    print("            check answer contains 'Example Domain'.")

    async def _test_playwright():
        async with make_ai() as ai:

            sub("mcp_connect: tools registered")
            # allowed=["headless"] -- скрыть окно браузера; обрабатывается _build_playwright
            await ai.mcp_connect("playwright", allowed=["headless"])
            tool_names = ai.mcp_tool_names()
            check(any("navigate" in t.lower() for t in tool_names),
                  f"a navigate tool is present in: {tool_names}")
            lst = ai.mcp_list()
            check(len(lst) == 1,                "one entry in mcp_list()")
            check(lst[0]["name"] == "playwright","name == 'playwright'")

            sub("one LLM call: agent navigates to example.com and reads title")
            answer = await run_async(
                ai,
                "Use playwright_navigate to open https://example.com, "
                "then use playwright_get_visible_text to get the page text "
                "and tell me the main heading of the page.",
                "playwright example.com",
            )
            # Проверяем что инструменты реально вызывались -- защита от ложного позитива:
            # модель знает example.com из обучения и может ответить без браузера.
            tool_calls_in_history = [
                tc.get("name", "")
                for msg in ai.chat_history
                for tc in (msg.get("data", {}).get("tool_calls") or [])
            ]
            check("playwright_navigate" in tool_calls_in_history,
                  f"playwright_navigate was actually called: {tool_calls_in_history}")
            check("playwright_get_visible_text" in tool_calls_in_history,
                  f"playwright_get_visible_text was actually called: {tool_calls_in_history}")

            # Проверяем что ответ инструмента содержит реальный текст страницы.
            tool_results = {
                msg.get("data", {}).get("name", ""): msg.get("data", {}).get("content", "")
                for msg in ai.chat_history
                if msg.get("type") == "tool"
            }
            visible_text = tool_results.get("playwright_get_visible_text", "")
            check(bool(visible_text) and "error" not in visible_text.lower(),
                  "playwright_get_visible_text returned non-empty page text")
            check("Example Domain" in visible_text,
                  "page text contains 'Example Domain' (from tool, not model knowledge)")

            sub("mcp_disconnect: tools removed")
            await ai.mcp_disconnect("playwright")
            check(len(ai.mcp_tool_names()) == 0, "no MCP tools remain")

    async_run(_test_playwright())


# ======================================================================
# T6. memory-plus (npx)
# ======================================================================

if should_run("memory-plus"):

    section("T6. memory-plus (npx)")
    print("  Requires: npx, @modelcontextprotocol/server-memory")
    print("  Strategy: connect with isolated temp .jsonl file ->")
    print("            ask agent to store a probe entity via create_entities ->")
    print("            ask agent to retrieve it via search_nodes ->")
    print("            check answer contains the probe entity name ->")
    print("            verify .jsonl file was actually written to disk ->")
    print("            disconnect.")

    import tempfile
    from pathlib import Path

    MEMORY_PROBE_ENTITY = "AIListTestProbe_99"
    MEMORY_PROBE_OBS    = "observation_ailist_probe_unique"

    async def _test_memory_plus():
        import shutil as _shutil_t7
        tmpdir = tempfile.mkdtemp()
        try:
            memory_file = str(Path(tmpdir) / "memory.jsonl")

            async with make_ai() as ai:

                sub("mcp_connect: tools registered with isolated memory file")
                # dirs[0] -- путь к .jsonl файлу памяти; обрабатывается _build_memory_plus
                await ai.mcp_connect("memory-plus", dirs=[memory_file])
                tool_names = ai.mcp_tool_names()
                check(len(tool_names) > 0,
                      f"memory-plus provides tools: {tool_names}")
                check("create_entities" in tool_names,
                      f"create_entities present in: {tool_names}")
                check("search_nodes" in tool_names,
                      f"search_nodes present in: {tool_names}")
                lst = ai.mcp_list()
                check(len(lst) == 1,                   "one entry in mcp_list()")
                check(lst[0]["name"] == "memory-plus", "name == 'memory-plus'")

                sub("LLM call 1: agent stores probe entity via create_entities")
                answer_store = await run_async(
                    ai,
                    f"Use the create_entities tool to create an entity with "
                    f"name='{MEMORY_PROBE_ENTITY}', entityType='test', "
                    f"observations=['{MEMORY_PROBE_OBS}']. "
                    f"Confirm what you stored.",
                    "memory-plus store entity",
                )
                # Проверяем что инструмент реально вызывался -- защита от ложного позитива.
                tool_calls_store = [
                    tc.get("name", "")
                    for msg in ai.chat_history
                    for tc in (msg.get("data", {}).get("tool_calls") or [])
                ]
                check("create_entities" in tool_calls_store,
                      f"create_entities was actually called: {tool_calls_store}")

                # Сбрасываем историю чата: следующий вызов -- отдельный "сеанс",
                # агент не должен помнить что мы только что сохранили из контекста.
                ai.chat_history = []

                sub("LLM call 2: agent retrieves probe entity via search_nodes")
                answer_search = await run_async(
                    ai,
                    f"Use the search_nodes tool with query='{MEMORY_PROBE_ENTITY}' "
                    f"and tell me the name of the entity you found.",
                    "memory-plus search entity",
                )
                check(MEMORY_PROBE_ENTITY in answer_search,
                      f"answer contains probe entity name '{MEMORY_PROBE_ENTITY}'")

                # Проверяем что search_nodes реально вызывался.
                tool_calls_search = [
                    tc.get("name", "")
                    for msg in ai.chat_history
                    for tc in (msg.get("data", {}).get("tool_calls") or [])
                ]
                check("search_nodes" in tool_calls_search,
                      f"search_nodes was actually called: {tool_calls_search}")

                sub("verify: .jsonl file written to disk by the server")
                file_exists = Path(memory_file).exists()
                check(file_exists,
                      f"memory file exists on disk: {memory_file}")
                if file_exists:
                    raw = Path(memory_file).read_text(encoding="utf-8", errors="replace")
                    check(MEMORY_PROBE_ENTITY in raw,
                          "probe entity name found in raw .jsonl content")

                sub("mcp_disconnect: tools removed")
                await ai.mcp_disconnect("memory-plus")
                check(len(ai.mcp_tool_names()) == 0, "no MCP tools remain")

        finally:
            _shutil_t7.rmtree(tmpdir, ignore_errors=True)

    async_run(_test_memory_plus())


# ======================================================================
# T7. searxng (sse)
# ======================================================================

if should_run("searxng"):

    section("T7. searxng (sse)")
    print("  Requires: Docker Desktop running.")
    print("  Containers are started automatically if not running.")
    print("  Strategy: ensure containers running -> connect via SSE -> agent runs")
    print("            search tool -> check answer contains query keyword ->")
    print("            verify tool was actually called -> disconnect.")

    SEARXNG_SSE_URL   = "http://localhost:8001/sse"
    SEARXNG_QUERY     = "Python programming language"
    SEARXNG_EXPECTED  = "python"

    async def _test_searxng():
        async with make_ai() as ai:

            sub("docker: ensure containers are running")
            for docker_name in ("searxng-engine", "searxng"):
                result = await ai.docker_ensure(docker_name)
                print(f"    '{docker_name}': {result}.")
            check(ai.docker_status("searxng-engine") == "running",
                  "searxng-engine container is running")
            check(ai.docker_status("searxng") == "running",
                  "searxng (mcp) container is running")

            sub("mcp_connect: tools registered via SSE")
            await ai.mcp_connect("searxng", url=SEARXNG_SSE_URL)
            tool_names = ai.mcp_tool_names()
            check(len(tool_names) > 0,
                  f"searxng provides tools: {tool_names}")
            check(any("search" in t.lower() for t in tool_names),
                  f"a search tool is present in: {tool_names}")
            lst = ai.mcp_list()
            check(len(lst) == 1,               "one entry in mcp_list()")
            check(lst[0]["name"] == "searxng", "name == 'searxng'")

            sub(f"one LLM call: agent searches for '{SEARXNG_QUERY}'")
            search_tool = next(t for t in tool_names if "search" in t.lower())
            answer = await run_async(
                ai,
                f"Call the {search_tool} tool exactly like this: "
                f'{search_tool}(query="{SEARXNG_QUERY}") '
                f"-- set only the query field, leave all other fields at their defaults. "
                f"Then summarize the results in 2-3 sentences.",
                f"searxng search: {SEARXNG_QUERY}",
            )
            check(contains(answer, SEARXNG_EXPECTED),
                  f"answer contains '{SEARXNG_EXPECTED}'")

            tool_calls_in_history = [
                tc.get("name", "")
                for msg in ai.chat_history
                for tc in (msg.get("data", {}).get("tool_calls") or [])
            ]
            check(search_tool in tool_calls_in_history,
                  f"{search_tool} was actually called: {tool_calls_in_history}")

            tool_results = [
                msg.get("data", {}).get("content", "")
                for msg in ai.chat_history
                if msg.get("type") == "tool"
                and msg.get("data", {}).get("name", "") == search_tool
            ]
            check(bool(tool_results),
                  "tool returned at least one result message")
            if tool_results:
                good_results = [
                    r if isinstance(r, str) else str(r)
                    for r in tool_results
                    if (r if isinstance(r, str) else str(r)).strip()
                    and "error" not in (r if isinstance(r, str) else str(r)).lower()
                    and "Input validation" not in (r if isinstance(r, str) else str(r))
                ]
                best_preview = good_results[0][:120] if good_results else tool_results[-1][:120]
                check(
                    bool(good_results),
                    f"at least one tool result is non-empty and error-free "
                    f"(best: {best_preview!r})",
                )

            sub("mcp_disconnect: tools removed")
            await ai.mcp_disconnect("searxng")
            check(len(ai.mcp_tool_names()) == 0, "no MCP tools remain")

    if not _docker_available():
        check(False, "Docker daemon is not running -- start Docker Desktop and re-run")
    else:
        async_run(_test_searxng())


# ======================================================================
# T8. qdrant (uvx)
# ======================================================================

if should_run("qdrant"):

    section("T8. qdrant (uvx)")
    print("  Requires: uv installed  (py312 -m pip install uv)")
    print("  Requires: Docker Desktop running (только для БД qdrant-db).")
    print("  Container qdrant-db starts automatically if not running.")
    print("  MCP-сервер запускается локально через uvx -- Docker для него не нужен.")
    print("  NOTE: первый запуск медленный -- uvx скачает mcp-server-qdrant и модель")
    print("        sentence-transformers/all-MiniLM-L6-v2 (~90 MB) с HuggingFace.")
    print("  Strategy: ensure qdrant-db running -> connect via uvx/stdio ->")
    print("            agent stores probe string via qdrant-store ->")
    print("            agent retrieves it via qdrant-find ->")
    print("            check answer contains probe string ->")
    print("            verify tools were actually called -> disconnect.")

    QDRANT_PROBE_TEXT  = "AIListQdrantProbe_unique_42"
    # Уникальное имя коллекции на каждый прогон -- чтобы не накапливать дубли
    # от предыдущих запусков. Коллекция удаляется в cleanup после теста.
    import uuid as _uuid
    QDRANT_COLLECTION  = f"test_ailist_{_uuid.uuid4().hex[:8]}"

    async def _test_qdrant():
        async with make_ai() as ai:

            sub("docker: ensure qdrant-db is running")
            result = await ai.docker_ensure("qdrant-db")
            print(f"    'qdrant-db': {result}.")
            check(ai.docker_status("qdrant-db") == "running",
                  "qdrant-db container is running")

            sub("mcp_connect: tools registered via uvx")
            await ai.mcp_connect("qdrant", allowed=[QDRANT_COLLECTION])
            tool_names = ai.mcp_tool_names()
            check(len(tool_names) > 0,
                  f"qdrant provides tools: {tool_names}")
            check("qdrant-store" in tool_names,
                  f"qdrant-store present in: {tool_names}")
            check("qdrant-find" in tool_names,
                  f"qdrant-find present in: {tool_names}")
            lst = ai.mcp_list()
            check(len(lst) == 1,              "one entry in mcp_list()")
            check(lst[0]["name"] == "qdrant", "name == 'qdrant'")

            sub("LLM call 1: agent stores probe string via qdrant-store")
            answer_store = await run_async(
                ai,
                f"Call qdrant-store with ONLY the 'information' argument set to "
                f"'{QDRANT_PROBE_TEXT}'. Do NOT include metadata. Do not explain.",
                "qdrant store probe",
            )
            tool_calls_store = [
                tc.get("name", "")
                for msg in ai.chat_history
                for tc in (msg.get("data", {}).get("tool_calls") or [])
            ]
            check("qdrant-store" in tool_calls_store,
                  f"qdrant-store was actually called: {tool_calls_store}")

            store_results = [
                msg.get("data", {}).get("content", "")
                for msg in ai.chat_history
                if msg.get("type") == "tool"
                and msg.get("data", {}).get("name", "") == "qdrant-store"
            ]
            if store_results:
                last = store_results[-1] if isinstance(store_results[-1], str) else str(store_results[-1])
                check("error" not in last.lower(),
                      f"qdrant-store result has no error (preview: {last[:120]!r})")

            ai.chat_history = []

            sub("LLM call 2: agent retrieves probe string via qdrant-find")
            answer_find = await run_async(
                ai,
                f"Call qdrant-find RIGHT NOW with query='{QDRANT_PROBE_TEXT}'. "
                f"Do not explain. Just call the tool and show me the raw result.",
                "qdrant find probe",
            )
            check(QDRANT_PROBE_TEXT in answer_find,
                  f"answer contains probe string '{QDRANT_PROBE_TEXT}'")

            tool_calls_find = [
                tc.get("name", "")
                for msg in ai.chat_history
                for tc in (msg.get("data", {}).get("tool_calls") or [])
            ]
            check("qdrant-find" in tool_calls_find,
                  f"qdrant-find was actually called: {tool_calls_find}")

            find_results = [
                msg.get("data", {}).get("content", "")
                for msg in ai.chat_history
                if msg.get("type") == "tool"
                and msg.get("data", {}).get("name", "") == "qdrant-find"
            ]
            check(bool(find_results),
                  "qdrant-find returned at least one result message")
            if find_results:
                good_results = [
                    r if isinstance(r, str) else str(r)
                    for r in find_results
                    if (r if isinstance(r, str) else str(r)).strip()
                    and "error" not in (r if isinstance(r, str) else str(r)).lower()
                ]
                best_preview = good_results[0][:120] if good_results else str(find_results[-1])[:120]
                check(
                    bool(good_results),
                    f"at least one find result is non-empty and error-free "
                    f"(best: {best_preview!r})",
                )

            sub("mcp_disconnect: tools removed")
            await ai.mcp_disconnect("qdrant")
            check(len(ai.mcp_tool_names()) == 0, "no MCP tools remain")

            sub("cleanup: delete test collection from qdrant-db")
            try:
                import urllib.request as _ur
                req = _ur.Request(
                    f"http://localhost:6333/collections/{QDRANT_COLLECTION}",
                    method="DELETE",
                )
                with _ur.urlopen(req, timeout=5) as resp:
                    check(resp.status == 200,
                          f"collection '{QDRANT_COLLECTION}' deleted (status {resp.status})")
            except Exception as _e:
                # Не фейлим тест из-за cleanup -- коллекция была уникальной, мусор не накапливается
                print(f"    cleanup warning: {_e}")

    if not _docker_available():
        check(False, "Docker daemon is not running -- start Docker Desktop and re-run")
    else:
        async_run(_test_qdrant())


# ======================================================================
# T9. piper (builtin TTS)
# ======================================================================

if should_run("piper"):

    section("T9. piper (builtin TTS)")
    print("  Strategy A (unit): PiperTTS._model_hf_path parses model names correctly.")
    print("  Strategy B (unit): synthesize() API shape -- missing piper-tts raises ImportError.")
    print("  Strategy C (integration): connect -> agent synthesizes -> WAV created.")
    print("  NOTE: integration requires 'pip install piper-tts' and internet for first download.")

    from ailist import PiperTTS
    from pathlib import Path as _Path

    # ------------------------------------------------------------------
    # Sub-test A: _model_hf_path unit tests
    # ------------------------------------------------------------------

    sub("PiperTTS._model_hf_path: well-formed model names")
    _cases = [
        ("ru_RU-denis-medium",    "ru/ru_RU/denis/medium/ru_RU-denis-medium"),
        ("en_US-lessac-medium",   "en/en_US/lessac/medium/en_US-lessac-medium"),
        ("en_GB-alan-low",        "en/en_GB/alan/low/en_GB-alan-low"),
        ("de_DE-thorsten-high",   "de/de_DE/thorsten/high/de_DE-thorsten-high"),
    ]
    for model_name, expected_path in _cases:
        try:
            got = PiperTTS._model_hf_path(model_name)
            check(got == expected_path,
                  f"_model_hf_path('{model_name}') == '{expected_path}' (got: '{got}')")
        except Exception as _exc:
            check(False, f"_model_hf_path('{model_name}') raised {type(_exc).__name__}: {_exc}")

    sub("PiperTTS._model_hf_path: malformed name -> ValueError")
    for _bad in ["ru_RU-denis", "just-two", "no-dashes"]:
        try:
            PiperTTS._model_hf_path(_bad)
            check(False, f"_model_hf_path('{_bad}') should raise ValueError")
        except ValueError:
            check(True, f"_model_hf_path('{_bad}') -> ValueError (correct)")
        except Exception as _exc:
            check(False, f"_model_hf_path('{_bad}') unexpected {type(_exc).__name__}: {_exc}")

    # ------------------------------------------------------------------
    # Sub-test B: synthesize() raises ImportError when piper-tts absent
    # ------------------------------------------------------------------

    sub("synthesize() raises ImportError when piper-tts not installed (mock check)")
    try:
        import piper as _piper_pkg  # noqa: F401
        _piper_installed = True
        print("    piper-tts is installed -- skipping ImportError mock check")
        check(True, "piper-tts installed, ImportError check skipped")
    except ImportError:
        _piper_installed = False
        _tmp_piper = PiperTTS(models_dir=_Path("."), output_dir=_Path("."))
        try:
            _tmp_piper.synthesize("test")
            check(False, "synthesize() should raise ImportError when piper-tts absent")
        except ImportError as _e:
            check("piper-tts" in str(_e) or "pip install" in str(_e),
                  f"ImportError message mentions piper-tts: {str(_e)[:120]!r}")
        except Exception as _e:
            check(False, f"synthesize() unexpected {type(_e).__name__}: {_e}")

    # ------------------------------------------------------------------
    # Sub-test C: ai.piper attributes and defaults
    # ------------------------------------------------------------------

    sub("ai.piper: expected attributes and defaults")
    _ai_p = AIListDemo()
    _p = _ai_p.piper
    check(isinstance(_p, PiperTTS),                    "ai.piper is PiperTTS instance")
    check(hasattr(_p, "model"),                        "has .model attribute")
    check(hasattr(_p, "speaker_id"),                   "has .speaker_id attribute")
    check(hasattr(_p, "length_scale"),                 "has .length_scale attribute")
    check(hasattr(_p, "noise_scale"),                  "has .noise_scale attribute")
    check(hasattr(_p, "noise_w"),                      "has .noise_w attribute")
    check(hasattr(_p, "use_gpu"),                      "has .use_gpu attribute")
    check(isinstance(_p.model, str) and len(_p.model) > 0,
          f"piper.model is non-empty string: {_p.model!r}")
    check(_p.use_gpu is False,                         "piper.use_gpu defaults to False (CPU)")
    check(isinstance(_p.models_dir, _Path),            "piper.models_dir is Path")
    check(isinstance(_p.t2v_dir, _Path),               "piper.t2v_dir is Path")

    sub("ai.piper: user can override attributes")
    _ai_p.piper.model = "en_US-lessac-medium"
    _ai_p.piper.length_scale = 0.9
    check(_ai_p.piper.model == "en_US-lessac-medium",  "piper.model overridden correctly")
    check(_ai_p.piper.length_scale == 0.9,             "piper.length_scale overridden correctly")

    # ------------------------------------------------------------------
    # Sub-test D: MCPServers entry + tool names + schema safety
    # ------------------------------------------------------------------

    sub("MCPServers['piper']: launcher=builtin, two tools")
    _defn = AIListDemo().MCPServers.get("piper")
    check(_defn is not None,                           "'piper' present in MCPServers")
    check(_defn is not None and _defn.launcher == "builtin",
          "launcher == 'builtin'")
    check(_defn is not None and _defn.builtin_tools is not None
          and len(_defn.builtin_tools) == 2,
          f"builtin_tools has 2 tools (got: {len(_defn.builtin_tools) if _defn and _defn.builtin_tools else '?'})")
    if _defn and _defn.builtin_tools and len(_defn.builtin_tools) == 2:
        _names = [getattr(t, "name", None) for t in _defn.builtin_tools]
        check("piper_synthesize" in _names, f"piper_synthesize present: {_names}")
        check("piper_set_config" in _names, f"piper_set_config present: {_names}")

    sub("piper tools schema: no Optional/Union types (Ollama Go-template safety)")
    if _defn and _defn.builtin_tools:
        import inspect as _inspect, typing as _typing
        for _t in _defn.builtin_tools:
            _schema = getattr(_t, "args_schema", None)
            _hints  = _typing.get_type_hints(_schema) if _schema else {}
            _bad = [
                f"{name}: {tp}"
                for name, tp in _hints.items()
                if hasattr(tp, "__origin__") and tp.__origin__ is _typing.Union
            ]
            check(len(_bad) == 0,
                  f"{getattr(_t, 'name', '?')}: no Union/Optional (Ollama-safe): {_bad or 'none'}")

    sub("piper_synthesize: only 'text' parameter (ЛЛМ не задаёт путь к файлу)")
    if _defn and _defn.builtin_tools:
        _syn_tool = next((t for t in _defn.builtin_tools
                          if getattr(t, "name", None) == "piper_synthesize"), None)
        if _syn_tool:
            _func = getattr(_syn_tool, "func", None) or getattr(_syn_tool, "coroutine", None)
            if _func:
                _params = list(_inspect.signature(_func).parameters.keys())
                check(_params == ["text"],
                      f"piper_synthesize params == ['text'] (got: {_params})")

    # ------------------------------------------------------------------
    # Sub-test E: integration (только если piper установлен)
    # ------------------------------------------------------------------

    if not _piper_installed:
        print("\n  [integration skipped: piper-tts not installed]")
        print("  Install with: pip install piper-tts")
        print("  Then re-run: python test_integration_part1.py --only=piper")
    else:
        async def _test_piper_integration():
            async with make_ai() as ai:

                sub("mcp_connect: piper_synthesize and piper_set_config registered")
                await ai.mcp_connect("piper")
                tool_names = ai.mcp_tool_names()
                check("piper_synthesize" in tool_names,
                      f"piper_synthesize present: {tool_names}")
                check("piper_set_config" in tool_names,
                      f"piper_set_config present: {tool_names}")
                check(len(ai.mcp_list()) == 1, "exactly one server in mcp_list()")

                sub("LLM call: agent synthesizes short phrase -> WAV created")
                t2v_dir = ai.piper.t2v_dir
                wav_before = set(t2v_dir.glob("t2v_*.wav")) if t2v_dir.exists() else set()

                ai.piper.model = "ru_RU-irina-medium"

                answer = await run_async(
                    ai,
                    "Use the piper_synthesize tool to say 'Тест синтеза речи' "
                    "and tell me the path to the created file.",
                    "piper_synthesize short phrase",
                )

                wav_after = set(t2v_dir.glob("t2v_*.wav")) if t2v_dir.exists() else set()
                new_wavs = wav_after - wav_before
                check(len(new_wavs) > 0,
                      f"at least one new WAV file created in t2v/ (found: {len(new_wavs)})")
                if new_wavs:
                    wav_path = next(iter(new_wavs))
                    check(wav_path.stat().st_size > 1000,
                          f"WAV file is non-trivial (size: {wav_path.stat().st_size} bytes)")
                    print(f"    created: {wav_path}")

                check(contains(answer, "t2v") or contains(answer, ".wav"),
                      f"answer mentions t2v or .wav: {answer[:120]!r}")

                sub("LLM call: piper_set_config changes model")
                # Устанавливаем модель отличную от целевой -- чтобы проверка не была тавтологией.
                # (До этого sub-теста piper.model == "ru_RU-irina-medium" из предыдущего sub-теста.)
                ai.piper.model = "ru_RU-ruslan-medium"
                _model_before = ai.piper.model   # "ru_RU-ruslan-medium"
                _model_target = "ru_RU-irina-medium"
                ai.chat_history = []
                answer_cfg = await run_async(
                    ai,
                    f"Use piper_set_config to change the model to '{_model_target}'. "
                    f"Only call piper_set_config -- do not synthesize anything.",
                    "piper_set_config change model",
                )
                # Проверяем что piper_set_config реально вызывался
                cfg_calls = [
                    tc.get("name", "")
                    for msg in ai.chat_history
                    for tc in (msg.get("data", {}).get("tool_calls") or [])
                ]
                check("piper_set_config" in cfg_calls,
                      f"piper_set_config was actually called: {cfg_calls}")
                # Основная проверка: модель изменилась с отличного значения на целевое
                check(ai.piper.model == _model_target,
                      f"piper.model changed from '{_model_before}' to '{_model_target}' "
                      f"(got: {ai.piper.model!r})")
                # Предупреждение если LLM вызвала лишний piper_synthesize (side effect)
                extra_synth = [c for c in cfg_calls if c == "piper_synthesize"]
                if extra_synth:
                    print(f"    note: LLM made {len(extra_synth)} unexpected piper_synthesize call(s)")

                sub("mcp_disconnect: tools removed")
                await ai.mcp_disconnect("piper")
                check(len(ai.mcp_tool_names()) == 0, "no MCP tools remain after disconnect")

        async_run(_test_piper_integration())


# ======================================================================
# Summary
# ======================================================================

if _t_section is not None:
    print(f"  Time: {time.perf_counter() - _t_section:.3f} sec")

# Лог из библиотеки -- собираем из всех ai-объектов которые оставили лог.
if _collected_logs:
    print(f"\n{'='*60}")
    print("  Log:")
    for entry in _collected_logs:
        _safe_print(str(entry))
    print("=" * 60)

print(f"\n{'='*60}")
if _any_fail:
    print("  RESULT: some checks FAILED - see FAIL lines above")
else:
    print("  RESULT: all checks passed")
print(f"  Total time: {time.perf_counter() - _t_total:.3f} sec")
print("=" * 60)