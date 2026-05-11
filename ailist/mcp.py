import asyncio
import sys as _sys
from pathlib import Path
from langchain_core.tools import StructuredTool
from .base import *

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


# --------------------------------------------------
# MCP -- реестр серверов и универсальный Mixin

import sys as _sys

@dataclass
class MCPServerDef:
    """
    Описание одного MCP-сервера в реестре MCPServers.

    Поля:
        package     -- npm-пакет для npx/uvx (например "@modelcontextprotocol/server-filesystem").
                      Для launcher="builtin" и launcher=None не используется -- передайте "".
        description   -- человекочитаемое описание для документации и логов.
        args_extra    -- список дополнительных аргументов npx после имени пакета;
                        None означает "аргументы отсутствуют или задаются через args_builder".
                        Используйте это поле только для серверов без пользовательских параметров.
        launcher      -- способ запуска сервера:
                          "npx"     -- внешний процесс через stdio/MCP (по умолчанию).
                          "uvx"     -- внешний процесс через stdio/MCP, запускается через
                                      "uv tool run" (не бинарник uvx, который может отсутствовать
                                      в PATH при установке uv через pip).
                                      Требует: pip install uv  или  winget install astral-sh.uv
                          "builtin" -- встроенные Python-инструменты, без npx и MCP-сессии.
                                      Инструменты задаются через поле builtin_tools.
                          "sse"     -- подключение к уже запущенному HTTP/SSE MCP-серверу
                                      по URL. Сервер должен быть запущен заранее (например
                                      через docker_start). URL берётся из поля url или
                                      передаётся явно в mcp_connect(url=...).
                          None      -- запись только для docker_start/docker_stop; mcp_connect
                                      к такому серверу не вызывается. Используется для
                                      вспомогательных контейнеров (БД, поисковик), которые
                                      нужны другому MCP-серверу, но сами MCP не реализуют.
        builtin_tools -- список LangChain-совместимых инструментов (StructuredTool / @tool).
                        Используется только при launcher="builtin". При подключении
                        инструменты регистрируются напрямую без MCP-сессии.
        url           -- дефолтный адрес SSE-сервера (например "http://localhost:8000/sse").
                        Используется только при launcher="sse". Может быть переопределён
                        Может быть переопределён через args_builder (url_override).
        docker        -- параметры для docker_start / docker_stop. Структура словаря:
                          image          -- Docker-образ (обязателен, например "qdrant/qdrant")
                          name           -- имя контейнера (по умолчанию = ключ реестра)
                          ports          -- список строк "host:container" (опционально)
                          volumes        -- список строк "src:dst" (опционально)
                          env            -- dict переменных окружения (опционально)
                          service        -- имя сервиса в docker-compose (по умолчанию = ключ)
                          setup_check    -- shell-команда, выполняемая через docker exec внутри
                                           контейнера; если возвращает код 0 -- настройка уже
                                           применена и setup_commands пропускаются. Опционально.
                          setup_wait_for -- shell-команда для ожидания готовности приложения
                                           внутри контейнера (после открытия порта приложение
                                           может ещё инициализироваться -- например писать конфиги).
                                           _docker_run_setup делает polling каждые 0.5 с пока
                                           команда не вернёт 0. Опционально.
                          wait_http       -- URL для ожидания готовности HTTP-сервера снаружи
                                           контейнера. docker_start делает polling каждые 0.5 с
                                           пока сервер не вернёт любой HTTP-ответ (даже 4xx/5xx --
                                           главное что не connection refused). Используется когда
                                           TCP-порт открывается раньше чем HTTP-сервер готов.
                                           Опционально.
                          setup_commands -- список shell-команд для разовой настройки контейнера,
                                           выполняются через docker exec после первого запуска.
                                           После выполнения контейнер автоматически перезапускается.
                                           Запускаются только если setup_check вернул не 0. Опционально.
        cwd           -- рабочая директория для запуска npx (только для launcher="npx").
                        Приоритет: defn.cwd -> папка ailist.py -> текущая директория.
                        Задавать явно нужно только если npm install выполнен не в папке
                        с ailist.py. Пример: cwd=r"c:\\MyProject"
        args_builder  -- callable(**kwargs) -> tuple[list[str], dict[str, str], str | None]
                        Динамически строит аргументы запуска из произвольных kwargs,
                        переданных пользователем в mcp_connect() / mcp_connects().
                        Возвращает тройку (extra_args, env_vars, url_override):
                          extra_args   -- список дополнительных аргументов командной строки
                          env_vars     -- dict переменных окружения для процесса
                          url_override -- адрес SSE-сервера (None = использовать MCPServerDef.url)
                        Если None -- динамических аргументов нет, используются только args_extra.
                        Вся специфика конкретного сервера сосредоточена здесь;
                        kwargs определяются и документируются самим args_builder.
    """
    package:       str
    description:   str
    args_extra:    list | None = None
    launcher:      str | None  = "npx"      # "npx" | "uvx" | "builtin" | "sse" | None
    builtin_tools: list | None = None
    url:           str | None  = None
    docker:        dict | None = None
    cwd:           str | None  = None
    args_builder:  Callable | None = None
    python_version: str | None = None
    # For launcher="uvx": Python version passed to --python (default "3.12").
    # Override for packages that require a different Python, e.g. Serena needs "3.13".
    system_prompt: str | None = None
    # Optional static text to append to system_tool_instructions when this server is connected.
    # Use for short per-server guidance that does not require a live MCP session.
    # Example: "When using filesystem, prefer read_file over write_file for large files."
    system_prompt_tool: str | None = None
    # Optional name of an MCP tool on this server to call in order to obtain a system prompt.
    # Called automatically after connect, result is appended to system_tool_instructions.
    # Use when the server exposes its own instructions as a tool (e.g. Serena's "initial_instructions",
    # fastskills's "list_skills"). The tool is called with no arguments.
    # Note: the standard MCP instructions field (from InitializeResult) and list_prompts/get_prompt
    # are always tried first automatically; system_prompt_tool is the fallback for servers
    # that expose their instructions only as a callable tool.
    init_timeout: float | None = None
    # Override the MCP session initialization timeout (seconds) for this server.
    # If None -- automatic: uvx servers use 300s on first download, 30s otherwise.
    # Set explicitly for servers with known long startup (e.g. Serena fetches from GitHub).
    uvx_command: str | None = None
    # For launcher="uvx": the executable name to run, used when package is a git/URL source
    # that differs from the command name.
    # If None -- package name is used directly (works for PyPI packages where name == command).
    # If set -- builds: uv tool run --python X --from <package> <uvx_command> <extra...>
    # Example: package="git+https://github.com/oraios/serena", uvx_command="serena"
    exclude_tools: list | None = None
    # Optional list of tool names to exclude after fetching from the server.
    # Applies to all launcher types (npx, uvx, sse, builtin).
    # Use to hide rarely-needed or confusing tools from the model without forking the server.
    # Example: exclude_tools=["git_reflog", "git_worktree", "git_cherry_pick"]
    schema_patches: dict | None = None
    # Optional patches to apply to tool input schemas after fetching from the server.
    # Format: {tool_name: {param_name: {field: value, ...}, ...}}
    # Use to fix misleading defaults in server-provided schemas without forking the server.
    # Example -- change headless default description:
    #   schema_patches={"playwright_navigate": {"headless": {"description": "Run headless (default: true)"}}}
    # Note: only modifies the schema shown to the model (description, default, etc.).
    #       Does not affect the actual runtime behaviour of the tool.




_MCP_TYPE_MAP = {
    "string":  str,
    "integer": int,
    "number":  float,
    "boolean": bool,
    "array":   list,
    "object":  dict,
}


def _mcp_schema_to_pydantic(tool_name: str, schema: dict):
    """
    Строит pydantic-модель из JSON Schema инструмента MCP.
    Нужна чтобы StructuredTool знал имена и типы параметров
    и передавал их корректно при вызове.

    Преобразование типов:
      "string"  -> str,  "integer" -> int,  "number" -> float,
      "boolean" -> bool, "array"   -> List[T], "object" -> dict

    Известные ограничения Ollama (workaround-и помечены ниже):
      Ollama использует шаблонизатор Go для рендеринга схемы tool calling.
      Ряд Python-типов вызывает panic "slice index out of range" при рендеринге:
        - typing.Any         -- возникает когда JSON Schema тип неизвестен
        - Optional[T]        -- Union[T, None] не поддерживается шаблоном
        - List без параметра -- нужен конкретный List[T]
      При работе с провайдерами у которых нет этих ограничений (OpenAI, Anthropic)
      данные workaround-и безвредны -- типы чуть менее точные, но валидные.

    Enum integer workaround:
      Некоторые MCP-серверы (например Serena) объявляют enum-значения как integers
      (например include_kinds: [0, 1, 2] где 0=Function, 1=Class и т.д.).
      Чтобы модель знала что передавать, мы добавляем расшифровку в описание поля,
      а тип оставляем int -- это правильно и для Ollama безопасно.
    """
    from pydantic import create_model, Field
    from typing import List

    properties = (schema or {}).get("properties", {})
    required   = set((schema or {}).get("required", []))

    def _enum_hint(enum_vals: list, title: str = "") -> str:
        """Строит подсказку для enum-значений в описании поля."""
        return f" (allowed values: {enum_vals})"

    def _resolve_type(prop_schema: dict):
        """Определяет Python-тип для одного свойства JSON Schema."""

        # "enum" без "type" -- в MCP все enum строковые.
        # Правильно было бы Literal['a','b','c'], но Ollama его не рендерит.
        # [Ollama workaround] используем str.
        if "enum" in prop_schema and "type" not in prop_schema:
            return str

        # anyOf / oneOf -- берём первый не-null тип.
        # Типичный случай: anyOf: [{type: array, items: {type: integer}}, {type: null}]
        # (fastskills view_range, serena execute_shell_command.cwd и др.)
        # [Ollama workaround] Optional не используем -- берём тип без null-ветки.
        for key in ("anyOf", "oneOf"):
            if key in prop_schema:
                for sub in prop_schema[key]:
                    if sub.get("type") != "null":
                        return _resolve_type(sub)
                return str  # все ветки null -- маловероятно

        json_type = prop_schema.get("type")

        if json_type == "array":
            items = prop_schema.get("items", {})
            item_type = items.get("type")
            # Если items -- integer enum (например Serena's SymbolKind),
            # тип элемента int. Расшифровка добавляется в описание отдельно.
            if item_type == "integer":
                return List[int]
            # [Ollama workaround] items типа "object" или отсутствующий items
            # дали бы List[dict] или List[Any] -- оба ломают шаблон.
            # Используем str как безопасный fallback для элементов.
            item_py = _MCP_TYPE_MAP.get(item_type, str)
            return List[item_py]

        # [Ollama workaround] неизвестный тип -> str вместо Any.
        # Any вызывает panic в шаблонизаторе Go при построении tool schema.
        return _MCP_TYPE_MAP.get(json_type, str)

    def _make_description(prop_name: str, prop_schema: dict) -> str:
        """Строит описание поля, добавляя расшифровку enum если есть."""
        desc = prop_schema.get("description") or prop_schema.get("title") or prop_name
        json_type = prop_schema.get("type")

        # Enum на верхнем уровне поля
        if "enum" in prop_schema:
            desc += _enum_hint(prop_schema["enum"])
            return desc

        # Enum внутри items массива (integer enum -- SymbolKind и подобные)
        if json_type == "array":
            items = prop_schema.get("items", {})
            if "enum" in items:
                desc += _enum_hint(items["enum"])
            elif items.get("type") == "integer" and "oneOf" in prop_schema:
                # oneOf с const/title описывает значения enum
                labels = [
                    f"{entry.get('const')}={entry.get('title', entry.get('const'))}"
                    for entry in prop_schema["oneOf"]
                    if "const" in entry
                ]
                if labels:
                    desc += f" (integer enum: {', '.join(labels)})"

        return desc

    fields = {}
    for prop_name, prop_schema in properties.items():
        py_type = _resolve_type(prop_schema)
        field_desc = _make_description(prop_name, prop_schema)
        field_kwargs = {"description": field_desc} if field_desc else {}

        if prop_name in required:
            fields[prop_name] = (py_type, Field(..., **field_kwargs))
        else:
            # [Ollama workaround] plain тип с дефолтом None вместо Optional[T].
            # Optional[T] = Union[T, None] -- шаблонизатор Go не поддерживает Union.
            # None-значения фильтруются в _mcp_make_caller перед отправкой на сервер.
            fields[prop_name] = (py_type, Field(None, **field_kwargs))

    return create_model(f"{tool_name}_args", **fields)


class MCPMixin:
    """
    Универсальная примесь (mixin) для AIListBase-подклассов.
    Управляет произвольным числом MCP-серверов без дублирования кода.

    Реестр серверов хранится в self.MCPServers (dict[str, MCPServerDef]).
    MCPMixin.MCPServers пуст -- конкретные серверы задаются в подклассе.
    Несколько серверов могут работать одновременно; их инструменты
    суммируются и передаются агенту.

    -- Публичный API ------------------------------------------------------

    Подключение / отключение:
        await ai.mcp_connect("workspace")
        await ai.mcp_connect("playwright", allowed=["headless"])
        await ai.mcp_connects([{"name": "workspace"}, {"name": "git"}, ...])
        await ai.mcp_disconnect("git")
        await ai.mcp_disconnect_all()

    Информация:
        ai.mcp_list()        -> [{"name": str, "description": str, "tools": [str]}]
        ai.mcp_tool_names()  -> [str]  -- плоский список всех активных инструментов

    Docker (для серверов с контейнерами):
        await ai.docker_start("qdrant-db")
        await ai.docker_ensure("searxng")   # запустить / пересоздать при изменении env
        ai.docker_status("qdrant-db")       # "running" | "stopped" | "not found"
        await ai.docker_stop("qdrant-db")

    Поддерживает async with:
        async with AIList() as ai:
            await ai.mcp_connect("workspace")
            ...
    # При выходе из контекста вызывается mcp_disconnect_all().

    -- Как добавить новый сервер ------------------------------------------

    Все изменения -- только в подклассе (AIList):
      1. Написать @staticmethod _build_name(**kwargs) -> (extra, env, url_override)
      2. В __init__ после super(): self.MCPServers["name"] = MCPServerDef(...)

    Тип сервера выбирается полем launcher:

    launcher="npx"  (по умолчанию) -- npm-пакет через stdio:
      args_builder = self._build_name
      package      = "npm-package-name"

    launcher="uvx"  -- Python-пакет через uv tool run:
      args_builder = self._build_name
      package      = "pypi-package-name"
      launcher     = "uvx"

    launcher="builtin"  -- Python-функции напрямую, без npx и MCP-протокола:
      builtin_tools = [my_langchain_tool]
      launcher      = "builtin"
      # args_builder не нужен, kwargs игнорируются

    launcher="sse"  -- уже запущенный HTTP/SSE-сервер:
      url          = "http://localhost:8000/sse"   # дефолт
      launcher     = "sse"
      args_builder = self._build_name  # обычно возвращает url_override

    launcher=None  -- Docker-контейнер без MCP (вспомогательный, например БД):
      launcher = None
      docker   = {"image": "...", "name": "...", "ports": [...]}
      # mcp_connect к этой записи не вызывается
      # используется только docker_start / docker_stop / docker_ensure

    Подробные примеры кода -- в конце файла.
    """
    MCPServers: dict[str, MCPServerDef] = {}
    # Реестр MCP-серверов. Ключ -- короткое имя для mcp_connect(name=...).
    # Задаётся в подклассе (AIList.__init__). MCPMixin оставляет его пустым.

    # -- Внутренние хелперы ----------------------------------------------------

    def _mcp_apply_exclude(self, tools: dict, defn) -> dict:
        """
        Удаляет из словаря tools инструменты перечисленные в defn.exclude_tools.
        Возвращает отфильтрованный словарь (не мутирует исходный).
        Если defn is None или exclude_tools пуст -- возвращает tools без изменений.
        """
        if defn is None or not defn.exclude_tools:
            return tools
        excluded = set(defn.exclude_tools)
        return {k: v for k, v in tools.items() if k not in excluded}

    def _mcp_apply_schema_patches(self, tools: dict, defn) -> dict:
        """
        Применяет патчи к схемам инструментов из defn.schema_patches.
        Патч мержится в properties[param] схемы на уровне JSON Schema.
        Не мутирует исходный словарь tools -- создаёт shallow copy затронутых записей.

        Format: {tool_name: {param_name: {field: value, ...}}}
        Example:
            schema_patches={"playwright_navigate": {"headless": {"description": "..."}}}
        """
        if defn is None or not defn.schema_patches:
            return tools
        result = dict(tools)
        for tool_name, param_patches in defn.schema_patches.items():
            if tool_name not in result:
                continue
            entry = dict(result[tool_name])
            schema = entry.get("schema")
            if schema is None:
                continue
            # Deep-copy only the path we mutate: schema -> properties
            schema = dict(schema)
            props = dict(schema.get("properties") or {})
            for param_name, field_patches in param_patches.items():
                if param_name in props:
                    props[param_name] = {**props[param_name], **field_patches}
                else:
                    props[param_name] = dict(field_patches)
            schema["properties"] = props
            entry["schema"] = schema
            result[tool_name] = entry
        return result

    async def _mcp_open(self, name: str, package: str, extra: list[str], _rebuild: bool = True, url: str | None = None, env: dict | None = None, _retry: bool = True):
        """
        Универсальное ядро подключения: предусловия, открытие сессии, регистрация, rebuild, лог.
        Не содержит никакой логики конкретных серверов -- только механику MCP.
        _rebuild=False используется при батч-подключении (mcp_connects), чтобы отложить
        пересборку агента до завершения всех подключений.

        Для launcher="builtin" пропускает всю stdio/MCP механику и регистрирует
        встроенные Python-инструменты напрямую из MCPServerDef.builtin_tools.
        """
        if not hasattr(self, "_mcp_connections"):
            self._mcp_connections = {}

        if name in self._mcp_connections:
            await self.mcp_disconnect(name, _rebuild=False)

        defn = self.MCPServers.get(name)

        # -- Builtin: встроенные Python-инструменты, без npx и MCP-сессии --------
        if defn is not None and defn.launcher == "builtin":
            raw_tools = defn.builtin_tools or []
            tools = {
                t.name: {
                    "description": t.description or "",
                    "schema":      getattr(t, "args_schema", None),
                    "caller":      None,   # не используется для builtin
                    "builtin_tool": t,     # оригинальный LangChain-инструмент
                }
                for t in raw_tools
            }
            tools = self._mcp_apply_exclude(tools, defn)
            tools = self._mcp_apply_schema_patches(tools, defn)
            self._mcp_connections[name] = {
                "stdio_ctx":    None,
                "session_ctx":  None,
                "session":      None,
                "tools":        tools,
                "server_prompt": "",  # builtin tools have no MCP session to fetch prompts from
            }
            # system_prompt_tool для builtin: вызываем инструмент напрямую (без MCP-сессии).
            # Используется например для "skills" сервера -- чтобы list_skills выполнился
            # при подключении и его вывод попал в system_tool_instructions.
            server_prompt_parts = []
            if defn.system_prompt_tool and defn.system_prompt_tool in tools:
                try:
                    bt = tools[defn.system_prompt_tool]["builtin_tool"]
                    tool_text = bt.invoke({})
                    if isinstance(tool_text, str) and tool_text.strip():
                        server_prompt_parts.append(tool_text.strip())
                except Exception:
                    pass
            if defn.system_prompt:
                server_prompt_parts.append(defn.system_prompt.strip())
            if server_prompt_parts:
                self._mcp_connections[name]["server_prompt"] = "\n\n".join(server_prompt_parts)
            if self.loglevel == 1:
                self.append_log(LOG_MCP_CONNECTED_BUILTIN.format(name=name))
            elif self.loglevel > 1:
                self.append_log(
                    LOG_MCP_CONNECTED_BUILTIN_V.format(
                        name=name,
                        desc=defn.description,
                        tools=list(tools.keys()),
                    )
                )
            if _rebuild:
                self._mcp_rebuild_agent()
            return

        # -- SSE: подключение к уже запущенному HTTP/SSE MCP-серверу ----------------
        if defn is not None and defn.launcher == "sse":
            if not _MCP_AVAILABLE:
                raise ImportError(ERR_MCP_NO_PACKAGE)
            effective_url = url or (defn.url if defn else None)
            if not effective_url:
                raise ValueError(ERR_MCP_SSE_NO_URL.format(name=name))
            sse_ctx = sse_client(effective_url)
            read, write = await sse_ctx.__aenter__()
            session_ctx = None
            try:
                session_ctx = ClientSession(read, write)
                session     = await session_ctx.__aenter__()
                init_result = await session.initialize()
                tools = await self._mcp_fetch_tools(session)
                tools = self._mcp_apply_exclude(tools, defn)
                tools = self._mcp_apply_schema_patches(tools, defn)
                server_prompt = await self._mcp_collect_server_prompt(name, session, init_result, defn)
            except:
                if session_ctx is not None:
                    try:
                        await session_ctx.__aexit__(None, None, None)
                    except Exception:
                        pass
                await sse_ctx.__aexit__(None, None, None)
                raise

            self._mcp_connections[name] = {
                "stdio_ctx":    sse_ctx,
                "session_ctx":  session_ctx,
                "session":      session,
                "tools":        tools,
                "server_prompt": server_prompt,
            }
            if self.loglevel == 1:
                self.append_log(LOG_MCP_CONNECTED_SSE.format(name=name, url=effective_url))
            elif self.loglevel > 1:
                label = defn.description if defn else name
                self.append_log(
                    LOG_MCP_CONNECTED_SSE_V.format(
                        name=name,
                        desc=label,
                        url=effective_url,
                        tools=list(tools.keys()),
                    )
                )
            if _rebuild:
                self._mcp_rebuild_agent()
            return

        # -- NPX / UVX: внешний MCP-сервер через stdio -------------------------------
        if not _MCP_AVAILABLE:
            raise ImportError(ERR_MCP_NO_PACKAGE)

        # cwd для npx/uvx: явный (defn.cwd) -> директория запускаемого скрипта
        npx_cwd = (
            defn.cwd if (defn and defn.cwd)
            else str(Path(_sys.argv[0]).resolve().parent)
        )

        # launcher="uvx" -- запуск через uv tool run (Python-пакеты), аналог npx для JS.
        # Требует установленного uv: pip install uv  или  winget install astral-sh.uv
        # Используем "uv tool run" вместо "uvx": uvx -- отдельный бинарник который может
        # отсутствовать в PATH, тогда как "uv" устанавливается через pip и всегда доступен.
        # --python 3.12 обязателен: без него uv берёт последний системный Python,
        # который может быть несовместим с зависимостями пакета (например pydantic-core
        # не поддерживает Python 3.14, требует компиляции Rust и падает).
        if defn is not None and defn.launcher == "uvx":
            py_ver = defn.python_version or "3.12"
            if defn.uvx_command:
                # Git/URL source: uv tool run --python X --from <package> <command> <extra...>
                uvx_args = ["tool", "run", "--python", py_ver, "--from", package, defn.uvx_command] + extra
            else:
                # PyPI package: uv tool run --python X <package> <extra...>
                uvx_args = ["tool", "run", "--python", py_ver, package] + extra
            if _sys.platform == "win32":
                params = StdioServerParameters(command="cmd", args=["/c", "uv"] + uvx_args, env=env or None, cwd=npx_cwd)
            else:
                params = StdioServerParameters(command="uv", args=uvx_args, env=env or None, cwd=npx_cwd)
        else:
            # launcher="npx" (дефолт)
            if _sys.platform == "win32":
                params = StdioServerParameters(command="cmd", args=["/c", "npx", "-y", package] + extra, env=env or None, cwd=npx_cwd)
            else:
                params = StdioServerParameters(command="npx", args=["-y", package] + extra, env=env or None, cwd=npx_cwd)

        stdio_ctx   = stdio_client(params)
        read, write = await stdio_ctx.__aenter__()
        session_ctx = None
        try:
            session_ctx = ClientSession(read, write)
            session     = await session_ctx.__aenter__()

            # Таймаут инициализации:
            #   uvx: 300 с по умолчанию -- при первом запуске uv скачивает пакет и,
            #        возможно, HuggingFace-модель; при повторных запусках это занимает
            #        секунды, но лишние 270 с ожидания ничего не стоят.
            #        Можно переопределить через MCPServerDef.init_timeout.
            #   npx: 120 с -- npm-пакеты меньше, но первый npm install тоже небыстрый.
            if defn is not None and defn.launcher == "uvx":
                init_timeout = defn.init_timeout if defn.init_timeout is not None else 300.0
            else:
                init_timeout = 120.0
            init_result = await asyncio.wait_for(session.initialize(), timeout=init_timeout)
            tools = await asyncio.wait_for(self._mcp_fetch_tools(session), timeout=30.0)
            server_prompt = await self._mcp_collect_server_prompt(name, session, init_result, defn)
        except Exception as _exc:
            if session_ctx is not None:
                try:
                    await session_ctx.__aexit__(None, None, None)
                except Exception:
                    pass

            # Для uvx: читаем stderr ДО закрытия stdio_ctx -- после __aexit__ pipe закрыт.
            # stdio_client хранит процесс в ._process (внутреннее поле mcp-пакета).
            stderr_text = ""
            if defn is not None and defn.launcher == "uvx":
                try:
                    proc = getattr(stdio_ctx, "_process", None)
                    if proc is not None:
                        # Ждём завершения: при Connection closed процесс может ещё работать.
                        try:
                            await asyncio.wait_for(proc.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            pass
                        if proc.stderr:
                            try:
                                raw = await asyncio.wait_for(proc.stderr.read(), timeout=3.0)
                                stderr_text = raw.decode(errors="replace").strip()
                            except asyncio.TimeoutError:
                                pass
                except Exception:
                    pass

            await stdio_ctx.__aexit__(None, None, None)

            if defn is not None and defn.launcher == "uvx":
                if stderr_text and self.loglevel >= 0:
                    self.append_log(LOG_MCP_UVX_STDERR.format(name=name, stderr=stderr_text))

                # Битый fastembed-кэш: модель скачалась частично (прерванная загрузка).
                # Симптом: NoSuchFile на .onnx внутри fastembed_cache в stderr.
                # Лечение: находим путь к fastembed_cache, удаляем и повторяем один раз.
                if _retry and stderr_text and "fastembed_cache" in stderr_text and "NoSuchFile" in stderr_text:
                    import shutil as _shutil
                    _cache_dir = None
                    for _token in stderr_text.replace("\\", "/").split():
                        if "fastembed_cache" in _token:
                            _p = _pathlib.Path(_token.strip(' \'"([])\\'))
                            while _p != _p.parent:
                                if _p.name == "fastembed_cache":
                                    _cache_dir = _p
                                    break
                                _p = _p.parent
                            if _cache_dir:
                                break
                    if _cache_dir and _cache_dir.exists():
                        if self.loglevel >= 0:
                            self.append_log(f"MCP [{name}] (uvx): corrupt fastembed cache, clearing {_cache_dir} and retrying.")
                        _shutil.rmtree(_cache_dir, ignore_errors=True)
                        await self._mcp_open(name, package, extra, _rebuild=_rebuild, url=url, env=env, _retry=False)
                        return

                if stderr_text:
                    raise RuntimeError(
                        ERR_MCP_UVX_PROCESS_FAILED.format(name=name, stderr=stderr_text)
                    ) from _exc
            raise

        self._mcp_connections[name] = {
            "stdio_ctx":    stdio_ctx,
            "session_ctx":  session_ctx,
            "session":      session,
            "tools":        self._mcp_apply_schema_patches(self._mcp_apply_exclude(tools, defn), defn),
            "server_prompt": server_prompt,
        }

        if self.loglevel == 1:
            self.append_log(LOG_MCP_CONNECTED.format(name=name))
        elif self.loglevel > 1:
            label = defn.description if defn else name
            self.append_log(
                LOG_MCP_CONNECTED_V.format(
                    name=name,
                    desc=label,
                    tools=list(tools.keys()),
                )
            )

        if _rebuild:
            self._mcp_rebuild_agent()

    def _mcp_make_caller(self, session, tool_name: str):
        """Оборачивает MCP tool_call в корутину для StructuredTool."""
        async def call_tool(**kwargs) -> str:
            # Фильтруем "пустые" значения необязательных параметров перед отправкой.
            # MCP-серверы ожидают отсутствия параметра, а не пустого значения.
            #
            # Почему фильтруем здесь, а не в _mcp_schema_to_pydantic:
            #   Pydantic-модель создаётся с дефолтами (None / "" / []) для совместимости
            #   с шаблонизатором Ollama -- он не поддерживает Optional[T].
            #   В результате модель иногда передаёт "" или [] вместо None для
            #   необязательных параметров (особенно с ограничением minLength/minItems).
            #   Фильтрация здесь -- единственное место где это можно исправить надёжно,
            #   без изменения pydantic-схемы и без дополнительных roundtrip-ошибок.
            #
            # Что фильтруется:
            #   None  -- модель не указала параметр
            #   ""    -- пустая строка вместо отсутствующего str-параметра
            #   []    -- пустой список вместо отсутствующего list-параметра
            #
            # Что НЕ фильтруется:
            #   0, False -- валидные значения, могут быть переданы намеренно
            #
            # Дополнительно: коерция float -> int для целочисленных значений.
            # Ollama иногда передаёт целые числа как float (например -1 -> -1.0),
            # что ломает парсинг на стороне сервера (invalid character '.' in JSON).
            def _coerce(v):
                if isinstance(v, float) and v == int(v):
                    return int(v)
                return v

            arguments = {
                k: _coerce(v) for k, v in kwargs.items()
                if v is not None and v != "" and v != []
            }
            result = await session.call_tool(tool_name, arguments=arguments)
            texts = [block.text for block in result.content if hasattr(block, "text")]
            if not texts:
                return "[tool returned non-text content]"
            return "\n".join(texts)
        call_tool.__name__ = tool_name
        return call_tool

    async def _mcp_fetch_tools(self, session) -> dict:
        resp = await session.list_tools()
        return {
            t.name: {
                "description": t.description or "",
                "schema":      t.inputSchema,
                "caller":      self._mcp_make_caller(session, t.name),
            }
            for t in resp.tools
        }

    async def _mcp_collect_server_prompt(self, name: str, session, init_result, defn) -> str:
        """
        Собирает системный промпт для подключённого MCP-сервера в порядке приоритета:

        1. MCP InitializeResult.instructions -- стандартное поле спецификации MCP.
           Большинство современных серверов передают промпт здесь (Serena, fastmcp и др.).
        2. MCP Prompts API (list_prompts + get_prompt) -- альтернативный стандартный путь.
           Используется серверами которые не заполняют instructions.
        3. MCPServerDef.system_prompt_tool -- вызов конкретного инструмента сервера.
           Используется только если пп. 1 и 2 ничего не вернули: если сервер уже передал
           промпт стандартным путём, значит он уже сказал всё что хотел.
           Пример: fastskills "list_skills", Serena "initial_instructions" (fallback).
        4. MCPServerDef.system_prompt -- статичный текст, добавляется всегда поверх всего.
           Пользователь может дополнить любой сервер своими инструкциями.

        Результат -- объединение всех непустых частей через двойной перенос строки.
        """
        dynamic_parts = []

        # 1. InitializeResult.instructions -- стандарт MCP, проверяем первым
        try:
            instructions = getattr(init_result, "instructions", None) or ""
            if instructions.strip():
                dynamic_parts.append(instructions.strip())
        except Exception:
            pass

        # 2. MCP Prompts API -- если instructions пуст, пробуем list_prompts/get_prompt
        if not dynamic_parts:
            try:
                prompts_resp = await session.list_prompts()
                if prompts_resp and prompts_resp.prompts:
                    for prompt_meta in prompts_resp.prompts:
                        try:
                            prompt_resp = await session.get_prompt(prompt_meta.name, arguments={})
                            for msg in (prompt_resp.messages or []):
                                content = msg.content
                                text = content.text if hasattr(content, "text") else (content if isinstance(content, str) else "")
                                if text.strip():
                                    dynamic_parts.append(text.strip())
                        except Exception:
                            pass
            except Exception:
                pass

        # 3. system_prompt_tool -- только если стандартные пути (1 и 2) ничего не вернули.
        # Если сервер уже передал промпт через instructions или list_prompts,
        # дополнительный вызов инструмента не нужен -- сервер уже сказал всё что хотел.
        if not dynamic_parts and defn and defn.system_prompt_tool:
            try:
                result = await session.call_tool(defn.system_prompt_tool, arguments={})
                texts = [b.text for b in (result.content or []) if hasattr(b, "text") and b.text]
                tool_text = "\n".join(texts).strip()
                if tool_text:
                    dynamic_parts.append(tool_text)
            except Exception:
                pass

        if not dynamic_parts and self.loglevel > 1:
            self.append_log(
                f"MCP [{name}]: no system prompt found "
                f"(checked: instructions, list_prompts"
                + (f", tool '{defn.system_prompt_tool}'" if defn and defn.system_prompt_tool else "")
                + ")."
            )

        # 4. Статичный system_prompt из MCPServerDef -- добавляется всегда поверх всего.
        # Пользователь может дополнить любой сервер своими инструкциями независимо
        # от того получили ли мы что-то с сервера.
        all_parts = dynamic_parts[:]
        if defn and defn.system_prompt:
            all_parts.append(defn.system_prompt.strip())

        return "\n\n".join(p for p in all_parts if p)

    def _mcp_rebuild_agent(self):
        """Пересобирает агента из статичных инструментов + всех активных MCP-инструментов.
        Захватывает _lock: нельзя менять агента пока идёт run/run_async,
        и нельзя запустить run/run_async пока агент пересобирается.
        """
        with self._lock:
            static = list(getattr(self, "_static_tools", []))

            # Проверяем конфликты имён инструментов между серверами.
            seen: dict[str, str] = {}  # tool_name -> server_name
            for server_name, conn in self._mcp_connections.items():
                for tool_name in conn["tools"]:
                    if tool_name in seen:
                        raise ValueError(
                            ERR_MCP_TOOL_NAME_CONFLICT.format(
                                tool=tool_name,
                                server1=seen[tool_name],
                                server2=server_name,
                            )
                        )
                    seen[tool_name] = server_name

            mcp_structured = []
            for conn in self._mcp_connections.values():
                for tool_name, info in conn["tools"].items():
                    if "builtin_tool" in info:
                        # Builtin-инструмент -- уже готовый LangChain-объект, берём as-is
                        mcp_structured.append(info["builtin_tool"])
                    else:
                        # NPX MCP-инструмент -- оборачиваем через StructuredTool
                        mcp_structured.append(
                            StructuredTool.from_function(
                                coroutine=info["caller"],
                                name=tool_name,
                                description=info["description"],
                                args_schema=_mcp_schema_to_pydantic(tool_name, info["schema"]),
                            )
                        )
            self._rebuild_agent(static + mcp_structured)

            # Собираем system_tool_instructions из всех подключённых серверов.
            # server_prompt кешируется в conn["server_prompt"] при подключении (_mcp_open).
            # Здесь только читаем кеш -- никаких async вызовов.
            # Сброс при отключении всех серверов -- fragments будет пустым.
            fragments = []
            for server_name, conn in self._mcp_connections.items():
                text = conn.get("server_prompt", "")
                if text and text.strip():
                    fragments.append(text.strip())
            self.system_tool_instructions = "\n\n".join(fragments)

    # -- Публичный API ---------------------------------------------------------

    def _mcp_build_args(self, name: str, **kwargs) -> tuple[str, list[str], dict[str, str], str | None]:
        """
        Диспетчер аргументов: вызывает args_builder сервера (если задан),
        объединяет результат с args_extra из MCPServerDef.
        Возвращает (package, extra_args, env_vars, url_override).
        """
        defn    = self.MCPServers.get(name)
        package = defn.package if defn else name
        extra:        list[str]      = list(defn.args_extra or []) if defn else []
        env:          dict[str, str] = {}
        url_override: str | None     = None

        if defn and defn.args_builder:
            extra_add, env_add, url_override = defn.args_builder(**kwargs)
            extra += extra_add
            env.update(env_add)

        return package, extra, env, url_override

    async def mcp_connect(self, name: str, **kwargs):
        """
        Подключает MCP-сервер из реестра self.MCPServers.
        После подключения инструменты сервера становятся доступны агенту.

        name     -- ключ из MCPServers (например "workspace", "playwright").
        **kwargs -- произвольные аргументы, прозрачно передаются в args_builder
                   данного сервера. Смысл kwargs определяется самим args_builder.
                   Стандартные соглашения встроенных серверов:
                     dirs    -- список директорий или путей
                     allowed -- белый список (коллекции, флаги, команды)
                     blocked -- чёрный список
                     url     -- переопределение адреса SSE-сервера

        Повторный вызов с тем же name переподключает сервер.
        Механика подключения (stdio / SSE / builtin) определяется launcher в MCPServerDef.

        Если workspace_dir задан и для сервера не переданы явные dirs --
        workspace_dir подставляется как dirs[0] по умолчанию для серверов
        serena.
        """
        # Подставляем workspace_dir как dirs по умолчанию для серверов работающих с файлами.
        # Только если: dirs не передан явно И workspace_dir задан.
        _WS_DEFAULT_DIRS_SERVERS = {"serena"}
        ws = getattr(self, "workspace_dir", None)
        if ws is not None and name in _WS_DEFAULT_DIRS_SERVERS and "dirs" not in kwargs:
            kwargs = dict(kwargs, dirs=[str(ws)])

        package, extra, env, url_override = self._mcp_build_args(name, **kwargs)
        effective_url = url_override or (self.MCPServers[name].url if name in self.MCPServers else None)
        await self._mcp_open(name, package, extra, url=effective_url, env=env)

    async def mcp_connects(self, servers: list[dict]):
        """
        Параллельно подключает несколько серверов; агент пересобирается один раз в конце.
        Быстрее последовательных mcp_connect при подключении нескольких npx-серверов.

        servers -- список словарей. Ключ "name" обязателен; остальные ключи
                  передаются как kwargs в args_builder соответствующего сервера:
            [
                {"name": "workspace"},
                {"name": "playwright", "allowed": ["headless"]},
                {"name": "git"},
            ]

        Если хотя бы один сервер не подключился -- все соединения батча закрываются
        и бросается ExceptionGroup. Агент не пересобирается.
        """
        if not servers:
            return

        async def _connect_one(spec: dict):
            name = spec.get("name")
            if not name:
                raise ValueError(ERR_MCP_CONNECTS_NO_NAME)
            kwargs = {k: v for k, v in spec.items() if k != "name"}
            # Подставляем workspace_dir как dirs по умолчанию (аналогично mcp_connect)
            _WS_DEFAULT_DIRS_SERVERS = {"serena"}
            ws = getattr(self, "workspace_dir", None)
            if ws is not None and name in _WS_DEFAULT_DIRS_SERVERS and "dirs" not in kwargs:
                kwargs = dict(kwargs, dirs=[str(ws)])
            package, extra, env, url_override = self._mcp_build_args(name, **kwargs)
            effective_url = url_override or (self.MCPServers[name].url if name in self.MCPServers else None)
            # _rebuild=False -- агент пересобирается один раз после gather
            await self._mcp_open(name, package, extra, _rebuild=False, url=effective_url, env=env)

        results = await asyncio.gather(
            *[_connect_one(s) for s in servers],
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            # Закрываем все соединения которые успели открыться в этом батче,
            # чтобы не оставлять объект в частично подключённом состоянии.
            names_in_batch = {s["name"] for s in servers if "name" in s}
            for name in list(names_in_batch):
                if hasattr(self, "_mcp_connections") and name in self._mcp_connections:
                    try:
                        await self.mcp_disconnect(name, _rebuild=False)
                    except Exception:
                        pass
            raise ExceptionGroup(ERR_MCP_CONNECTS_PARTIAL, errors)

        self._mcp_rebuild_agent()

    async def mcp_disconnect(self, name: str, _rebuild: bool = True):
        """Отключает один MCP-сервер по имени. Агент пересобирается без его инструментов."""
        if not hasattr(self, "_mcp_connections") or name not in self._mcp_connections:
            return
        conn = self._mcp_connections.pop(name)
        # Builtin-соединения не имеют stdio/session контекстов -- пропускаем закрытие.
        if conn.get("session_ctx") is not None or conn.get("stdio_ctx") is not None:
            try:
                await conn["session_ctx"].__aexit__(None, None, None)
            finally:
                await conn["stdio_ctx"].__aexit__(None, None, None)
        if self.loglevel > 0:
            self.append_log(LOG_MCP_DISCONNECTED.format(name=name))

        if _rebuild:
            self._mcp_rebuild_agent()

    async def mcp_disconnect_all(self):
        """Отключает все активные MCP-серверы."""
        names = list(getattr(self, "_mcp_connections", {}).keys())
        for name in names:
            try:
                await self.mcp_disconnect(name, _rebuild=False)
            except Exception:
                pass
        if names:
            self._mcp_rebuild_agent()

    def mcp_list(self) -> list[dict]:
        """
        Возвращает список активных MCP-соединений.
        Каждый элемент: {"name": str, "description": str, "tools": list[str]}
        """
        result = []
        for name, conn in getattr(self, "_mcp_connections", {}).items():
            defn = self.MCPServers.get(name)
            result.append({
                "name":        name,
                "description": defn.description if defn else name,
                "tools":       list(conn["tools"].keys()),
            })
        return result

    def mcp_tool_names(self) -> list[str]:
        """Плоский список имён всех инструментов всех активных MCP-серверов."""
        return [
            tool_name
            for conn in getattr(self, "_mcp_connections", {}).values()
            for tool_name in conn["tools"]
        ]

    # -- Docker-утилиты -------------------------------------------------------

    @staticmethod
    def _docker_find_compose(start_dir: str | None = None) -> str | None:
        """
        Ищет docker-compose.yml / docker-compose.yaml вверх по дереву директорий
        начиная с start_dir (по умолчанию -- текущая рабочая директория).
        Возвращает полный путь к файлу или None если не найден.
        """
        import os
        current = Path(start_dir or os.getcwd()).resolve()
        for candidate in [current, *current.parents]:
            for fname in ("docker-compose.yml", "docker-compose.yaml"):
                p = candidate / fname
                if p.exists():
                    return str(p)
        return None

    @staticmethod
    async def _docker_wait_port(host: str, port: int, timeout: float, http_url: str | None = None) -> None:
        """
        Ждёт пока сервис станет готов к приёму соединений.

        Этапы:
          1. TCP: ждёт пока порт начнёт принимать соединения.
          2. HTTP (если задан http_url): после TCP делает GET на http_url и ждёт
             пока сервер вернёт любой ответ без обрыва соединения.
             Нужно для SSE-серверов, которые занимают порт до завершения инициализации.

        Пробует каждые 0.5 с; бросает RuntimeError если не готово за timeout секунд.
        """
        import socket as _socket
        import urllib.request as _urllib
        import urllib.error as _urlerr
        import time as _time
        deadline = _time.monotonic() + timeout

        # Этап 1: TCP
        while True:
            try:
                with _socket.create_connection((host, port), timeout=0.5):
                    break   # порт открыт -- переходим к HTTP-проверке
            except OSError:
                pass
            if _time.monotonic() >= deadline:
                raise RuntimeError(
                    ERR_DOCKER_PORT_TIMEOUT.format(port=port, timeout=timeout)
                )
            await asyncio.sleep(0.5)

        if not http_url:
            return

        # Этап 2: HTTP -- ждём пока сервер начнёт отвечать (не обрывать соединение)
        while True:
            try:
                req = _urllib.Request(http_url)
                with _urllib.urlopen(req, timeout=1):
                    return   # получили ответ -- сервис готов
            except _urlerr.HTTPError:
                return       # HTTP-ошибка тоже означает что сервер отвечает
            except Exception:
                pass         # RemoteProtocolError, ConnectionReset и т.п. -- ещё не готов
            if _time.monotonic() >= deadline:
                raise RuntimeError(
                    ERR_DOCKER_HTTP_TIMEOUT.format(url=http_url, timeout=timeout)
                )
            await asyncio.sleep(0.5)

    async def _docker_run_setup(self, container_name: str, d: dict, wait: float = 15.0) -> str | None:
        """
        Проверяет нужна ли разовая настройка контейнера и при необходимости выполняет её.

        Алгоритм:
          1. Если setup_check задан -- выполняет docker exec <container> sh -c <setup_check>.
             Код возврата 0 означает "уже настроено" -- возвращаем None, ничего не делаем.
          2. Иначе выполняет каждую команду из setup_commands через docker exec.
          3. Перезапускает контейнер через docker restart и ждёт 3 секунды.

        Возвращает строку с выводом всех команд или None если настройка не потребовалась.
        """
        import subprocess as _sp

        setup_check    = d.get("setup_check")
        setup_wait_for = d.get("setup_wait_for")
        setup_commands = d.get("setup_commands") or []
        if not setup_commands:
            return None

        # Шаг 0: ждём готовности приложения внутри контейнера.
        # Порт может открыться раньше чем приложение запишет конфиги --
        # polling пока setup_wait_for не вернёт 0 (или 30 с таймаут).
        if setup_wait_for:
            import time as _time
            deadline = _time.monotonic() + wait
            if self.loglevel > 0:
                self.append_log(LOG_DOCKER_SETUP_WAITING.format(name=container_name))
            while True:
                result = await asyncio.to_thread(
                    _sp.run,
                    ["docker", "exec", container_name, "sh", "-c", setup_wait_for],
                    capture_output=True,
                )
                if result.returncode == 0:
                    break
                if _time.monotonic() >= deadline:
                    raise RuntimeError(
                        ERR_DOCKER_SETUP_NOT_READY.format(
                            name=container_name, timeout=wait, cmd=setup_wait_for
                        )
                    )
                await asyncio.sleep(0.5)

        # Шаг 1: проверяем нужна ли настройка
        if setup_check:
            result = await asyncio.to_thread(
                _sp.run,
                ["docker", "exec", container_name, "sh", "-c", setup_check],
                capture_output=True,
            )
            if result.returncode == 0:
                # Настройка уже применена -- ничего не делаем
                if self.loglevel > 0:
                    self.append_log(LOG_DOCKER_SETUP_SKIPPED.format(name=container_name))
                return None

        # Шаг 2: выполняем setup_commands
        if self.loglevel > 0:
            self.append_log(LOG_DOCKER_SETUP_APPLYING.format(name=container_name, n=len(setup_commands)))
        out_parts = []
        for cmd_str in setup_commands:
            cmd = ["docker", "exec", container_name, "sh", "-c", cmd_str]
            if self.loglevel > 1:
                self.append_log(LOG_DOCKER_SETUP_CMD.format(name=container_name, cmd=' '.join(cmd)))
            result = await asyncio.to_thread(_sp.run, cmd, capture_output=True, text=True)
            out_parts.append((result.stdout + result.stderr).strip())
            if result.returncode != 0:
                raise RuntimeError(
                    ERR_DOCKER_SETUP_CMD_FAILED.format(
                        name=container_name,
                        code=result.returncode,
                        cmd=cmd_str,
                        out=out_parts[-1],
                    )
                )

        # Шаг 3: перезапускаем контейнер чтобы настройка вступила в силу
        if self.loglevel > 0:
            self.append_log(LOG_DOCKER_SETUP_RESTART.format(name=container_name))
        result = await asyncio.to_thread(
            _sp.run,
            ["docker", "restart", container_name],
            capture_output=True, text=True,
        )
        out_parts.append((result.stdout + result.stderr).strip())
        if result.returncode != 0:
            raise RuntimeError(
                ERR_DOCKER_SETUP_RESTART_FAILED.format(name=container_name, out=out_parts[-1])
            )
        # Ждём пока контейнер снова поднимется после рестарта
        if d.get("ports"):
            host_port_str = d["ports"][0].split(":")[0]
            await MCPMixin._docker_wait_port(
                "127.0.0.1", int(host_port_str), wait,
                http_url=d.get("wait_http"),
            )

        return "\n".join(p for p in out_parts if p)

    @staticmethod
    def _docker_env_matches(container_name: str, required_env: dict) -> bool:
        """
        Проверяет что запущенный контейнер содержит все переменные окружения из required_env
        с правильными значениями. Использует docker inspect -- локальный вызов, ~10 мс.

        Возвращает True если все переменные совпадают, False если хотя бы одна отличается
        или отсутствует (контейнер нужно пересоздать).
        """
        import subprocess as _sp, json as _json
        if not required_env:
            return True
        try:
            result = _sp.run(
                ["docker", "inspect", "--format", "{{json .Config.Env}}", container_name],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                return False
            # Env в Docker -- список строк "KEY=VALUE"
            env_list = _json.loads(result.stdout.strip())
            env_dict = {}
            for item in env_list:
                if "=" in item:
                    k, _, v = item.partition("=")
                    env_dict[k] = v
            return all(env_dict.get(k) == v for k, v in required_env.items())
        except Exception:
            return False

    async def docker_start(self, name: str, compose_file: str | None = None, wait: float = 15.0) -> str:
        """
        Запускает Docker-контейнер(ы) для сервера name и ждёт готовности порта.

        Стратегия:
          1. Если найден docker-compose.yml (рядом со скриптом или выше) --
             выполняет: docker compose -f <file> up -d <service>
             где service = defn.docker["service"] или name.
          2. Если compose-файл не найден -- выполняет docker run -d
             с параметрами из MCPServerDef.docker (image, name, ports, volumes, env).
             Если контейнер уже запущен -- возвращает "already running" без действий.
             Если контейнер остановлен -- выполняет docker start без пересоздания.
             Проверка актуальности env -- задача docker_ensure, не этого метода.
          3. После запуска ждёт пока host-порт из defn.docker["ports"][0] начнёт
             принимать соединения (polling каждые 0.5 с, таймаут = wait секунд).
             Если порт не появился за это время -- бросает RuntimeError.
          4. Если в defn.docker заданы setup_check и setup_commands -- проверяет
             нужна ли разовая настройка (docker exec <container> <setup_check>).
             Если setup_check вернул не 0 -- выполняет setup_commands через docker exec
             и перезапускает контейнер. Повторные вызовы docker_start настройку
             не применяют (setup_check вернёт 0).

        Параметры:
            name         -- ключ из MCPServers.
            compose_file -- явный путь к docker-compose.yml (переопределяет автопоиск).
            wait         -- максимальное время ожидания готовности порта в секундах
                           (по умолчанию 15). Передайте 0 чтобы не ждать.

        Возвращает stdout+stderr команды запуска, "already running" если уже запущен
        (+ вывод setup если выполнялся).
        Выбрасывает RuntimeError если Docker недоступен, defn.docker не задан,
        или контейнер не поднялся за wait секунд.
        """
        import subprocess as _sp
        defn = self.MCPServers.get(name)
        if defn is None or defn.docker is None:
            raise RuntimeError(ERR_DOCKER_NO_DEF.format(name=name))
        d = defn.docker

        # -- Пытаемся через docker-compose ------------------------------------
        compose_path = compose_file or self._docker_find_compose()
        if compose_path:
            service = d.get("service", name)
            cmd = ["docker", "compose", "-f", compose_path, "up", "-d", service]
            if self.loglevel > 0:
                self.append_log(LOG_DOCKER_CMD.format(name=name, cmd=' '.join(cmd)))
            result = await asyncio.to_thread(
                _sp.run, cmd, capture_output=True, text=True
            )
            out = (result.stdout + result.stderr).strip()
            if result.returncode != 0:
                raise RuntimeError(ERR_DOCKER_START_FAILED.format(name=name, out=out))

            # Ждём готовности сервиса -- после compose up контейнер
            # стартует асинхронно, порт открывается через секунду-две.
            if wait > 0 and d.get("ports"):
                host_port_str = d["ports"][0].split(":")[0]
                await self._docker_wait_port(
                    "127.0.0.1", int(host_port_str), wait,
                    http_url=d.get("wait_http"),
                )
            setup_out = await self._docker_run_setup(
                d.get("name", name), d, wait=wait
            )
            if setup_out:
                out = (out + "\n" + setup_out).strip()
            return out

        # -- Fallback: docker run (или docker start если контейнер остановлен) --
        image = d.get("image")
        if not image:
            raise RuntimeError(ERR_DOCKER_NO_IMAGE.format(name=name))
        container_name = d.get("name", name)

        # Если контейнер уже существует -- определяем что делать:
        # running  -> возвращаем "already running" (не запускаем повторно)
        # stopped  -> docker start (без пересоздания)
        # not found -> docker run
        # Проверка актуальности env -- задача docker_ensure, не docker_start.
        existing_status = self.docker_status(name)

        if existing_status == "running":
            if self.loglevel > 0:
                self.append_log(LOG_DOCKER_ALREADY_RUNNING.format(name=name))
            return "already running"

        if existing_status == "stopped":
            if self.loglevel > 0:
                self.append_log(LOG_DOCKER_STARTING_STOPPED.format(name=name))
            result = await asyncio.to_thread(
                _sp.run,
                ["docker", "start", container_name],
                capture_output=True, text=True,
            )
            out = (result.stdout + result.stderr).strip()
            if result.returncode != 0:
                raise RuntimeError(ERR_DOCKER_START_FAILED.format(name=name, out=out))
            if wait > 0 and d.get("ports"):
                host_port_str = d["ports"][0].split(":")[0]
                await self._docker_wait_port(
                    "127.0.0.1", int(host_port_str), wait,
                    http_url=d.get("wait_http"),
                )
            setup_out = await self._docker_run_setup(
                container_name, d, wait=wait
            )
            if setup_out:
                out = (out + "\n" + setup_out).strip()
            return out

        # Если образа нет локально -- скачиваем явно перед docker run.
        # docker run тоже скачивает, но без явного pull мы не знаем сколько это займёт
        # и таймаут _docker_wait_port истечёт раньше чем контейнер поднимется.
        # docker image inspect быстрый (локальный вызов, ~10 мс), не ходит в сеть.
        inspect_result = await asyncio.to_thread(
            _sp.run,
            ["docker", "image", "inspect", image],
            capture_output=True,
        )
        if inspect_result.returncode != 0:
            if self.loglevel > 0:
                self.append_log(LOG_DOCKER_IMAGE_MISSING.format(name=name, image=image))
            pull_result = await asyncio.to_thread(
                _sp.run,
                ["docker", "pull", image],
                capture_output=True, text=True,
            )
            if pull_result.returncode != 0:
                raise RuntimeError(
                    ERR_DOCKER_PULL_FAILED.format(
                        name=name,
                        image=image,
                        out=(pull_result.stdout + pull_result.stderr).strip(),
                    )
                )
            if self.loglevel > 0:
                self.append_log(LOG_DOCKER_IMAGE_PULLED.format(name=name, image=image))

        cmd = ["docker", "run", "-d", "--name", container_name]
        for port in d.get("ports", []):
            cmd += ["-p", port]
        for vol in d.get("volumes", []):
            cmd += ["-v", vol]
        for k, v in (d.get("env") or {}).items():
            cmd += ["-e", f"{k}={v}"]
        # На Linux Docker Engine (без Desktop) host.docker.internal не резолвится
        # автоматически -- добавляем явный маппинг на шлюз хоста.
        # На Windows и macOS Docker Desktop регистрирует это имя самостоятельно,
        # лишний --add-host безвреден.
        if _sys.platform not in ("win32", "darwin"):
            cmd += ["--add-host", "host.docker.internal:host-gateway"]
        cmd.append(image)

        if self.loglevel > 0:
            self.append_log(LOG_DOCKER_CMD.format(name=name, cmd=' '.join(cmd)))
        result = await asyncio.to_thread(
            _sp.run, cmd, capture_output=True, text=True
        )
        out = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            raise RuntimeError(ERR_DOCKER_START_FAILED.format(name=name, out=out))

        # Ждём готовности сервиса -- после docker run контейнер
        # стартует асинхронно, порт открывается через секунду-две.
        if wait > 0 and d.get("ports"):
            host_port_str = d["ports"][0].split(":")[0]
            await self._docker_wait_port(
                "127.0.0.1", int(host_port_str), wait,
                http_url=d.get("wait_http"),
            )

        # Разовая настройка: выполняется только если setup_check вернул не 0.
        # При повторных вызовах docker_start setup_check пройдёт и настройка пропускается.
        setup_out = await self._docker_run_setup(
            d.get("name", name), d, wait=wait
        )
        if setup_out:
            out = (out + "\n" + setup_out).strip()
        return out

    async def docker_stop(self, name: str, compose_file: str | None = None, remove: bool = True) -> str:
        """
        Останавливает Docker-контейнер(ы) для сервера name.

        Стратегия:
          1. Если найден docker-compose.yml -- docker compose -f <file> stop <service>
          2. Иначе -- docker stop <container_name> (и docker rm если remove=True)

        Параметры:
            name         -- ключ из MCPServers.
            compose_file -- явный путь к docker-compose.yml (переопределяет автопоиск).
            remove       -- удалить контейнер после остановки (только для docker stop, по умолчанию True).

        Важно: для связок из двух контейнеров (searxng + searxng-engine)
        нужно остановить оба вызова отдельно, в порядке: сначала MCP-сервер, потом движок.
        Пример:
            await ai.docker_stop("searxng")         # MCP-сервер -- первым
            await ai.docker_stop("searxng-engine")  # сам поисковик -- вторым
            await ai.docker_stop("qdrant-db")       # только БД (MCP-сервер -- uvx-процесс, не контейнер)

        Возвращает строку с stdout+stderr команды.
        """
        import subprocess as _sp
        defn = self.MCPServers.get(name)
        if defn is None or defn.docker is None:
            raise RuntimeError(ERR_DOCKER_NO_DEF_STOP.format(name=name))
        d = defn.docker

        # -- Пытаемся через docker-compose ------------------------------------
        compose_path = compose_file or self._docker_find_compose()
        if compose_path:
            service = d.get("service", name)
            cmd = ["docker", "compose", "-f", compose_path, "stop", service]
            if self.loglevel > 0:
                self.append_log(LOG_DOCKER_STOP_CMD.format(name=name, cmd=' '.join(cmd)))
            result = await asyncio.to_thread(
                _sp.run, cmd, capture_output=True, text=True
            )
            out = (result.stdout + result.stderr).strip()
            if result.returncode != 0:
                raise RuntimeError(ERR_DOCKER_STOP_FAILED.format(name=name, out=out))
            return out

        # -- Fallback: docker stop [+ rm] -------------------------------------
        container_name = d.get("name", name)
        out_parts = []
        for cmd in (
            ["docker", "stop", container_name],
            *([ ["docker", "rm",   container_name] ] if remove else []),
        ):
            if self.loglevel > 0:
                self.append_log(LOG_DOCKER_STOP_CMD.format(name=name, cmd=' '.join(cmd)))
            result = await asyncio.to_thread(
                _sp.run, cmd, capture_output=True, text=True
            )
            out_parts.append((result.stdout + result.stderr).strip())
            # docker rm после stop может вернуть ненулевой код если контейнер уже удалён -- не ошибка
            if result.returncode != 0 and cmd[1] == "stop":
                raise RuntimeError(ERR_DOCKER_STOP_FAILED.format(name=name, out=out_parts[-1]))
        return "\n".join(out_parts)

    def docker_status(self, name: str) -> str:
        """
        Возвращает статус Docker-контейнера для сервера name:
            "running"   -- контейнер запущен
            "stopped"   -- контейнер существует но остановлен
            "not found" -- контейнер не существует
            "unknown"   -- не удалось выполнить docker inspect

        Синхронный вызов, не требует await.
        """
        import subprocess as _sp
        defn = self.MCPServers.get(name)
        if defn is None or defn.docker is None:
            return "not found"
        container_name = defn.docker.get("name", name)
        try:
            result = _sp.run(
                ["docker", "inspect", "--format", "{{.State.Status}}", container_name],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                return "not found"
            status = result.stdout.strip().lower()
            if status == "running":
                return "running"
            return "stopped"
        except Exception:
            return "unknown"


    async def docker_ensure(self, name: str, wait: float = 15.0) -> str:
        """
        Гарантирует что контейнер запущен и его env совпадает с реестром.

        Алгоритм:
          1. Если контейнер не существует -- запускает через docker_start.
          2. Если контейнер запущен -- проверяет env через docker inspect.
             Сравниваются только переменные, заданные в defn.docker["env"].
             Если хотя бы одна переменная отсутствует или имеет неверное значение --
             пересоздаёт контейнер: docker_stop (remove=True) + docker_start.
          3. Если контейнер остановлен -- запускает через docker_start
             (который сделает docker start без пересоздания).

        Возвращает строку с описанием выполненного действия:
            "already running"   -- контейнер работает и env совпадает
            "recreated"         -- пересоздан из-за несовпадения env
            "started"           -- запущен (был остановлен или отсутствовал)

        Синхронный аналог: docker_status -- только для проверки без действий.
        """
        defn = self.MCPServers.get(name)
        if defn is None or defn.docker is None:
            raise RuntimeError(ERR_DOCKER_NO_DEF_ENSURE.format(name=name))
        d = defn.docker
        container_name = d.get("name", name)
        required_env: dict = d.get("env") or {}

        status = self.docker_status(name)

        if status == "running":
            # Проверяем env через _docker_env_matches -- синхронный docker inspect, ~10 мс.
            if not self._docker_env_matches(container_name, required_env):
                if self.loglevel > 0:
                    self.append_log(LOG_DOCKER_ENSURE_RECREATE.format(name=name))
                await self.docker_stop(name, remove=True)
                await self.docker_start(name, wait=wait)
                return "recreated"
            return "already running"

        # stopped или not found -- запускаем (docker_start сам разберётся)
        await self.docker_start(name, wait=wait)
        return "started"

    # -- Playwright: ручной запуск браузера -----------------------------------

    def playwright_open(self, profile_dir: str = "") -> None:
        """
        Открывает Chromium с указанной папкой профиля для ручной работы пользователя.
        Используется когда нужно залогиниться, настроить сессию или начать процесс
        который ЛЛМ затем продолжит через mcp_connect("playwright", dirs=[profile_dir]).

        Браузер запускается в видимом режиме (headed) и остаётся открытым.
        Функция возвращается сразу -- не ждёт закрытия браузера.

        Параметры:
            profile_dir -- путь к папке профиля (сохраняет cookies/сессии).
                          Пустая строка -- временный профиль, данные не сохраняются.

        Требует: npx и @executeautomation/playwright-mcp-server установленные локально
                 в папке, указанной в MCPServerDef.cwd для "playwright".

        Пример:
            ai.playwright_open("C:\\Work\\browser_profile")
            input("Залогиньтесь в браузере, затем нажмите Enter...")
            await ai.mcp_connect("playwright", dirs=["C:\\Work\\browser_profile"])
        """
        import subprocess as _sp
        import sys as _sys

        cmd_args = ["-y", "@executeautomation/playwright-mcp-server"]

        # Рабочая директория: берём cwd из реестра playwright-сервера.
        # Это гарантирует что npx возьмёт локально установленный пакет,
        # браузерные бинарники которого совпадают по версии.
        defn = self.MCPServers.get("playwright")
        # cwd для npx: явный (defn.cwd) -> директория запускаемого скрипта
        npx_cwd = (
            defn.cwd if (defn and defn.cwd)
            else str(Path(_sys.argv[0]).resolve().parent)
        )

        # Строим команду запуска -- только браузер, без MCP-сессии
        if _sys.platform == "win32":
            cmd = ["cmd", "/c", "npx"] + cmd_args
        else:
            cmd = ["npx"] + cmd_args

        import os as _os
        env = _os.environ.copy()
        env["HEADLESS"] = "false"   # всегда видимый для ручной работы
        if profile_dir:
            env["PLAYWRIGHT_USER_DATA_DIR"] = profile_dir

        _sp.Popen(cmd, env=env, cwd=npx_cwd)
        if self.loglevel > 0:
            if profile_dir:
                self.append_log(LOG_PLAYWRIGHT_OPEN.format(profile_dir=profile_dir))
            else:
                self.append_log(LOG_PLAYWRIGHT_OPEN_TEMP)

    # -- async with ------------------------------------------------------------

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.mcp_disconnect_all()