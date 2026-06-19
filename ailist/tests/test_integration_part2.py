"""
Integration tests for MCPMixin: real MCP servers + one LLM call each.
Part 2 of 2: T0 (registry smoke) + T11 (skills) + T12 (serena).

WHAT IS TESTED:
  T11. skills     -- builtin: подключает реальную папку скиллов (C:\\W\\Python\\School\\skills),
                    list_skills -> ищем известные скиллы по именам; view docx/SKILL.md ->
                    проверяем что содержимое упоминает Word/docx.
  T12. serena     -- uvx: creates temp project with two Python files (mutual references);
                    sub-test A: agent finds cross-file symbol references via find_symbol /
                      find_referencing_symbols and names the referencing function;
                    sub-test B: agent surgically replaces a function body via
                      replace_symbol_body and verifies file content changed correctly.

REFACTORING CHANGES (relative to the old test):
  - T0: обновлён реестр -- убраны filesystem, shell, subprocess, text-editor, fastskills;
        добавлены workspace, skills.
  - T11: fastskills (uvx) -> skills (builtin). Сервер больше не uvx-процесс;
         list_skills реализован как builtin Python-инструмент в AIList.
         Тест использует реальную папку скиллов C:\\W\\Python\\School\\skills
         (ту же что раньше использовал fastskills).
         LLM call 2: путь к docx/SKILL.md берётся из результата list_skills,
         а не хардкодится -- корректно работает при любом уровне вложенности
         (skills/docx/SKILL.md или skills/AnthropicSkills/docx/SKILL.md).
  - T12: serena -- добавлен явный вызов onboarding перед call A,
         чтобы LSP-индекс был готов до запроса find_referencing_symbols.

ARCHITECTURE NOTES (ailist.py):
  - MCPMixin.MCPServers  -- реестр серверов, задаётся в подклассе (AIList.__init__)
  - MCPServerDef.args_builder -- callable(**kwargs) -> (extra, env, url_override),
    инкапсулирует специфику одного сервера; живёт как @staticmethod в AIList
  - mcp_connect(name, **kwargs) -- kwargs пробрасываются напрямую в args_builder,
    никаких фиксированных параметров dirs/allowed/blocked/url на уровне MCPMixin
  - skills_dir -- str атрибут AIList, по умолчанию "skills" (относительно workspace_dir).
    get_skills_dir() резолвит в абсолютный Path. Для теста задаётся абсолютным путём.

HOW TO RUN:
  All tests in this file:
      python test_integration_part2.py

  Specific tests only:
      python test_integration_part2.py --only=skills
      python test_integration_part2.py --only=serena
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
            print(f"     {line}")
        return answer
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  EXCEPTION ({elapsed:.3f} sec): {e}")
        _any_fail = True
        return ""

def async_run(coro):
    """
    Runs a coroutine in a new event loop.
    Catches all exceptions so a single failing test does not abort the whole file.
    After completion collects ai.log from all ai-objects created during the run.
    """
    global _any_fail
    n_before = len(_registered_ais)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=600))
    except BaseException as e:
        msg = f"  FAIL | test raised an unhandled exception: {type(e).__name__}: {e}"
        if isinstance(e, BaseExceptionGroup):
            for i, sub_exc in enumerate(e.exceptions, 1):
                msg += f"\n    sub-exception {i}: {type(sub_exc).__name__}: {sub_exc}"
        print(msg)
        _any_fail = True
    finally:
        loop.close()
        for ai in _registered_ais[n_before:]:
            if ai.log.strip():
                _collected_logs.append(ai.log)

def should_run(name: str) -> bool:
    """Return True if this test should run given --only filter."""
    return not _only_filter or name in _only_filter


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

    sub("args_builder returns correct tuple shape")
    ai2 = AIListDemo()

    # playwright: dirs -> PLAYWRIGHT_USER_DATA_DIR в env; extra всегда []
    # (headless управляется через system_prompt, не через CLI-аргумент)
    extra, env, url = ai2.MCPServers["playwright"].args_builder(
        dirs=[r"C:\profile"], allowed=["headless"]
    )
    check(env.get("PLAYWRIGHT_USER_DATA_DIR") == r"C:\profile" and url is None,
          "playwright args_builder sets PLAYWRIGHT_USER_DATA_DIR in env")
    check(extra == [],
          "playwright args_builder: extra is always [] (headless via system_prompt, not CLI)")
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
    check(serena_defn.system_prompt_tool == "initial_instructions",
          f"serena system_prompt_tool == 'initial_instructions' (got: {serena_defn.system_prompt_tool!r})")

_test_registry()


# ======================================================================
# T11. skills (builtin) -- integration with real skills folder
# ======================================================================

if should_run("skills"):

    section("T11. skills (builtin)")
    print("  Uses the real local skills folder C:\\W\\Python\\School\\skills.")
    print("  No external process -- list_skills is a builtin Python tool in AIList.")
    print("  Strategy:")
    print("    Unit:        ai.skills_dir = absolute path -> get_skills_dir() resolves correctly")
    print("    Integration: connect -> list_skills -> find known skills by name")
    print("                 -> view docx/SKILL.md -> check content mentions Word/docx")
    print("  Note: docx/SKILL.md path is taken from list_skills output,")
    print("        so the test works regardless of nesting level.")

    from pathlib import Path as _Path

    SKILLS_DIR = r"C:\W\Python\School\skills"

    # ------------------------------------------------------------------
    # Sub-test A: unit -- skills_dir resolution
    # ------------------------------------------------------------------

    sub("skills_dir: absolute path resolves correctly via get_skills_dir()")
    _ai_sk = AIListDemo()
    _ai_sk.skills_dir = SKILLS_DIR
    _resolved = _ai_sk.get_skills_dir()
    check(_resolved is not None,
          "get_skills_dir() returns non-None for absolute path")
    check(_resolved == _Path(SKILLS_DIR),
          f"get_skills_dir() == Path(SKILLS_DIR) (got: {_resolved!r})")

    sub("skills_dir: default relative 'skills' resolves relative to workspace_dir")
    _ai_sk2 = AIListDemo()
    # дефолт "skills" -- относительный, резолвится от workspace_dir
    _ws = _ai_sk2.workspace_dir
    _expected_rel = (_ws / "skills") if _ws else None
    _resolved_rel = _ai_sk2.get_skills_dir()
    check(_resolved_rel == _expected_rel,
          f"default 'skills' resolves to workspace_dir/skills (got: {_resolved_rel!r})")

    # ------------------------------------------------------------------
    # Sub-test B: integration -- connect and list real skills
    # ------------------------------------------------------------------

    _skills_path = _Path(SKILLS_DIR)
    if not _skills_path.exists():
        print(f"\n  [skills integration skipped: {SKILLS_DIR!r} does not exist]")
    else:
        async def _test_skills_integration():
            ai = make_ai()
            ai.skills_dir = SKILLS_DIR

            sub("mcp_connect: list_skills tool registered")
            await ai.mcp_connect("skills")
            tool_names = ai.mcp_tool_names()
            check("list_skills" in tool_names,
                  f"list_skills present in tools: {tool_names}")
            check(len(ai.mcp_list()) == 1, "exactly one server in mcp_list()")

            sub("LLM call 1: list available skills via list_skills")
            answer_list = await run_async(
                ai,
                "Call the list_skills tool and tell me all the skill names you found.",
                "skills list_skills",
            )
            # Список актуален на момент последнего ревью теста.
            # При добавлении/переименовании скиллов обновить здесь.
            known = ["docx", "pdf", "pptx", "xlsx", "frontend-design",
                     "skill-creator", "file-reading"]
            found = [s for s in known if s in answer_list.lower()]
            check(len(found) > 0,
                  f"answer mentions known skills (found: {found or 'none'})")

            # Проверяем вызов инструмента и вывод
            tool_calls = [
                tc.get("name", "")
                for msg in ai.chat_history
                for tc in (msg.get("data", {}).get("tool_calls") or [])
            ]
            check("list_skills" in tool_calls,
                  f"list_skills was actually called: {tool_calls}")

            tool_results = [
                msg.get("data", {}).get("content", "")
                for msg in ai.chat_history
                if msg.get("type") == "tool"
                and msg.get("data", {}).get("name", "") == "list_skills"
            ]
            if not tool_results:
                check(False, "list_skills returned no tool result in chat_history")
            else:
                raw = tool_results[-1] if isinstance(tool_results[-1], str) else str(tool_results[-1])
                check("error" not in raw.lower(),
                      f"list_skills result has no error (preview: {raw[:120]!r})")

            # ------------------------------------------------------------------
            # LLM call 2: view docx/SKILL.md -- реальное чтение файла с диска.
            # Путь к docx/SKILL.md берётся из результата list_skills --
            # это корректно работает при любом уровне вложенности
            # (skills/docx/SKILL.md или skills/AnthropicSkills/docx/SKILL.md).
            # ------------------------------------------------------------------
            ai.chat_history = []

            sub("LLM call 2 (integration): view docx/SKILL.md")

            import tempfile as _tempfile
            import shutil as _shutil
            from pathlib import Path as _P

            # Ищем путь к docx/SKILL.md в сыром выводе list_skills.
            # Формат строки: "  path: <абсолютный путь>"
            _docx_path = None
            if tool_results:
                raw_result = tool_results[-1] if isinstance(tool_results[-1], str) else str(tool_results[-1])
                for line in raw_result.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("path:") and "docx" in stripped and "SKILL.md" in stripped:
                        _docx_path = _P(stripped.split("path:", 1)[1].strip())
                        break

            if _docx_path is None or not _docx_path.exists():
                check(False,
                      f"docx SKILL.md not found in list_skills output -- cannot run LLM call 2 "
                      f"(searched in tool result, got path: {_docx_path!r})")
            else:
                # workspace_dir строго изолирован -- копируем SKILL.md в tmpdir.
                # Это предотвращает любые случайные записи в реальную папку скиллов.
                # mkdtemp + ручная очистка вместо TemporaryDirectory: на Windows
                # контекстный менеджер падает с PermissionError если MCP-процесс
                # (workspace-сервер) ещё держит дескрипторы на файлы в папке.
                _skill_tmp = _tempfile.mkdtemp()
                try:
                    _skill_tmp_path = _P(_skill_tmp)
                    _dst = _skill_tmp_path / "docx_SKILL.md"
                    _shutil.copy2(str(_docx_path), str(_dst))

                    ai.workspace_dir = _skill_tmp_path
                    await ai.mcp_connect("workspace")
                    ai.chat_history = []

                    answer_docx = await run_async(
                        ai,
                        f"Use the view tool with path='{str(_dst)}' "
                        "and give me a one-sentence summary of what this skill does.",
                        "skills view docx/SKILL.md",
                    )
                    docx_kw = ["word", "docx", "document", ".docx"]
                    found_kw = [kw for kw in docx_kw if kw in answer_docx.lower()]
                    check(len(found_kw) > 0,
                          f"docx skill summary mentions relevant keywords "
                          f"(found: {found_kw or 'none'}, preview: {answer_docx[:120]!r})")

                    tool_calls2 = [
                        tc.get("name", "")
                        for msg in ai.chat_history
                        for tc in (msg.get("data", {}).get("tool_calls") or [])
                    ]
                    check("view" in tool_calls2,
                          f"view was actually called: {tool_calls2}")

                    view_results = [
                        msg.get("data", {}).get("content", "")
                        for msg in ai.chat_history
                        if msg.get("type") == "tool"
                        and msg.get("data", {}).get("name", "") == "view"
                    ]
                    if view_results:
                        raw = view_results[-1] if isinstance(view_results[-1], str) else str(view_results[-1])
                        check(raw.strip() and "error" not in raw.lower(),
                              f"view result non-empty and error-free (preview: {raw[:120]!r})")

                    await ai.mcp_disconnect("workspace")
                finally:
                    # ignore_errors=True: не падаем если Windows ещё держит файлы.
                    # ОС сама подберёт мусор при следующей перезагрузке.
                    _shutil.rmtree(_skill_tmp, ignore_errors=True)

            sub("mcp_disconnect skills: tools removed")
            await ai.mcp_disconnect("skills")
            check(len(ai.mcp_tool_names()) == 0, "no MCP tools remain after disconnect")

        async_run(_test_skills_integration())


# ======================================================================
# T12. serena (uvx)
# ======================================================================

if should_run("serena"):

    section("T12. serena (uvx)")
    print("  Requires: pip install uv")
    print("  Requires: Python 3.13 available to uv (uv python install 3.13)")
    print("  Strategy A -- cross-file symbol references:")
    print("    Create temp project with two Python files where helper.py defines")
    print("    add_numbers() and main.py calls it. Ask agent to find all places")
    print("    that reference add_numbers and confirm it names main.py / call_math().")
    print("  Strategy B -- surgical symbol replacement:")
    print("    Ask agent to replace the body of add_numbers() with a new implementation")
    print("    and verify the file content changed correctly on disk.")

    import tempfile
    from pathlib import Path
    import subprocess as _sp_serena

    def _serena_preflight() -> str | None:
        """
        Checks prerequisites for the Serena test.
        Returns None if everything is ready, or a human-readable skip message.
        """
        try:
            r = _sp_serena.run(["uv", "--version"], capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                return "uv is not working: " + (r.stderr or r.stdout).strip()
        except FileNotFoundError:
            return "uv is not installed -- run: pip install uv"
        except Exception as e:
            return f"uv check failed: {e}"

        try:
            r = _sp_serena.run(
                ["uv", "python", "list"],
                capture_output=True, text=True, timeout=15,
            )
            if "3.13" not in r.stdout:
                return (
                    "Python 3.13 is not available to uv.\n"
                    "  Fix: uv python install 3.13\n"
                    f"  Currently available: {r.stdout.strip()[:300]}"
                )
        except Exception as e:
            return f"uv python list failed: {e}"

        return None  # all good

    _preflight_msg = _serena_preflight()
    if _preflight_msg:
        print(f"\n  [serena skipped: {_preflight_msg}]")
    else:
        _HELPER_PY = '''\
def add_numbers(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a + b
'''

        _MAIN_PY = '''\
from helper import add_numbers

def call_math() -> int:
    """Call add_numbers and return the result."""
    return add_numbers(10, 20)

if __name__ == "__main__":
    print(call_math())
'''

        async def _test_serena():
            # mkdtemp + ручная очистка вместо TemporaryDirectory: на Windows
            # контекстный менеджер падает с PermissionError если Serena LSP
            # ещё держит дескрипторы на файлы в папке после mcp_disconnect.
            import shutil as _shutil_serena
            tmpdir = tempfile.mkdtemp()
            try:
                proj = Path(tmpdir)
                (proj / "helper.py").write_text(_HELPER_PY, encoding="utf-8")
                (proj / "main.py").write_text(_MAIN_PY, encoding="utf-8")

                async with make_ai() as ai:

                    sub("mcp_connect: serena tools registered")
                    print("  Note: first run may take several minutes (fetching Serena from GitHub).")
                    # Фиксируем workspace_dir в изолированный tmpdir.
                    # Serena не использует workspace_dir (у него свой project path),
                    # но это явно документирует что CWD не является рабочей директорией.
                    ai.workspace_dir = Path(tmpdir)
                    await ai.mcp_connect("serena", dirs=[tmpdir])
                    tool_names = ai.mcp_tool_names()
                    check(len(tool_names) > 0,
                          f"serena provides tools ({len(tool_names)} total)")
                    check(any("find" in t.lower() or "symbol" in t.lower() for t in tool_names),
                          f"at least one symbol/find tool present: {tool_names}")
                    lst = ai.mcp_list()
                    check(len(lst) == 1,             "one entry in mcp_list()")
                    check(lst[0]["name"] == "serena", "name == 'serena'")

                    sub("system_tool_instructions: serena prompt loaded")
                    instr = ai.system_tool_instructions
                    if instr and instr.strip():
                        check(True, f"system_tool_instructions loaded ({len(instr)} chars)")
                    else:
                        print("  INFO | system_tool_instructions is empty -- "
                              "checked: MCP instructions field, list_prompts, initial_instructions tool. "
                              "The LLM relies on tool descriptions only.")

                    # ----------------------------------------------------------
                    # Onboarding: явно запускаем индексацию проекта перед call A.
                    # Без этого LSP-индекс не готов и find_referencing_symbols
                    # возвращает пустой результат.
                    # ----------------------------------------------------------
                    sub("onboarding: indexing project before symbol search")
                    await run_async(
                        ai,
                        f"The project is at {tmpdir}. "
                        "Run the onboarding tool to index this project.",
                        "serena onboarding",
                    )
                    ai.chat_history = []

                    # ----------------------------------------------------------
                    # Sub-test A: cross-file reference discovery
                    # ----------------------------------------------------------
                    sub("LLM call A: find all references to add_numbers across the project")
                    answer_a = await run_async(
                        ai,
                        f"The project is at {tmpdir}. "
                        "Use Serena tools to find all symbols that reference or call "
                        "the function 'add_numbers' defined in helper.py. "
                        "Tell me: which file(s) and which function name(s) call add_numbers?",
                        "serena find_referencing_symbols add_numbers",
                    )
                    check(contains(answer_a, "main"),
                          f"answer mentions main.py: {answer_a[:200]!r}")
                    check(contains(answer_a, "call_math"),
                          f"answer mentions call_math: {answer_a[:200]!r}")

                    tool_calls_a = [
                        tc.get("name", "")
                        for msg in ai.chat_history
                        for tc in (msg.get("data", {}).get("tool_calls") or [])
                    ]
                    check(any("find" in t.lower() or "symbol" in t.lower() or "reference" in t.lower()
                              for t in tool_calls_a),
                          f"agent called a find/symbol/reference tool: {tool_calls_a}")

                    symbol_tool_results = [
                        msg.get("data", {}).get("content", "")
                        for msg in ai.chat_history
                        if msg.get("type") == "tool"
                        and any(kw in msg.get("data", {}).get("name", "")
                                for kw in ("find_symbol", "find_referencing", "get_symbols"))
                    ]
                    if symbol_tool_results:
                        any_lsp_success = any(
                            r and "error" not in str(r).lower()[:50]
                            for r in symbol_tool_results
                        )
                        check(any_lsp_success,
                              "at least one symbol tool succeeded without error (LSP working); "
                              "if FAIL: agent fell back to plain text -- check _mcp_schema_to_pydantic")

                    # ----------------------------------------------------------
                    # Sub-test B: surgical symbol replacement
                    # ----------------------------------------------------------
                    sub("LLM call B: replace body of add_numbers with a new implementation")
                    new_body = "return (a + b) * 2"
                    answer_b = await run_async(  # noqa: F841
                        ai,
                        f"The project is at {tmpdir}. "
                        "Use Serena tools to replace the body of the function 'add_numbers' "
                        f"in helper.py so that it returns: {new_body!r} instead of 'return a + b'. "
                        "Only change the return statement inside the function, nothing else.",
                        "serena replace_symbol_body add_numbers",
                    )

                    helper_content = (proj / "helper.py").read_text(encoding="utf-8")
                    check("(a + b) * 2" in helper_content,
                          f"helper.py contains new body on disk: {helper_content!r}")

                    tool_calls_b = [
                        tc.get("name", "")
                        for msg in ai.chat_history
                        for tc in (msg.get("data", {}).get("tool_calls") or [])
                    ]
                    check(any("replace" in t.lower() or "edit" in t.lower() or "create" in t.lower()
                              for t in tool_calls_b),
                          f"agent called a replace/edit tool: {tool_calls_b}")

                    sub("mcp_disconnect: tools removed")
                    await ai.mcp_disconnect("serena")
                    check(len(ai.mcp_tool_names()) == 0, "no MCP tools remain after disconnect")

            finally:
                # ignore_errors=True: не падаем если Windows ещё держит файлы.
                # ОС сама подберёт мусор при следующей перезагрузке.
                _shutil_serena.rmtree(tmpdir, ignore_errors=True)

        async_run(_test_serena())


# ======================================================================
# Summary
# ======================================================================

if _t_section is not None:
    print(f"  Time: {time.perf_counter() - _t_section:.3f} sec")

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