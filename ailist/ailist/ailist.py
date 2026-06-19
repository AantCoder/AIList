from pathlib import Path
from langchain.tools import tool
from .base import *
from .mcp import MCPMixin, MCPServerDef


# --------------------------------------------------
# Builtin-инструменты (launcher="builtin")
# Определяются здесь и передаются в MCPServerDef.builtin_tools.

# --------------------------------------------------
# Режим безопасности subprocess_run
#
# True  (по умолчанию) -- безопасный режим:
#   shell=False + shlex.split() блокируют shell-injection через & ; | && ||
#   Недоступны встроенные команды shell (echo, cd, dir на Windows;
#   echo, ls, source на Unix) -- только внешние бинарники.
#
# False -- доверительный режим:
#   shell=True, команда передаётся строкой напрямую в интерпретатор.
#   Доступны все встроенные команды shell, пайпы, перенаправления.
#   Использовать только если агент работает в контролируемой среде
#   и входные данные не поступают от внешних источников
#   (файлы из интернета, сообщения от пользователей, результаты поиска).
#
# Чтобы переключить: изменить значение ниже.
# --------------------------------------------------
_SUBPROCESS_SAFE_MODE = True

# --------------------------------------------------
# Лимиты размера файлов и рабочей папки workspace
#
# WORKSPACE_FILE_LIMIT_BYTES  -- максимальный размер одного файла при записи через
#                                 write_file / create_file / str_replace / insert.
#                                 <= 0 отключает проверку.
#
# WORKSPACE_DIR_LIMIT_BYTES   -- максимальный суммарный размер всей рабочей папки.
#                                 <= 0 отключает проверку.
#
# Проверка размера папки кэшируется:
#   - пересчёт пропускается если свободное место на диске изменилось менее чем на
#     WORKSPACE_DIR_FREE_DELTA_BYTES (20 МБ) с момента последнего пересчёта;
#   - кэш принудительно сбрасывается через WORKSPACE_DIR_CACHE_TTL_SEC (10 мин).
# --------------------------------------------------
WORKSPACE_FILE_LIMIT_BYTES:      int = 50 * 1024 * 1024   # 50 МБ
WORKSPACE_DIR_LIMIT_BYTES:       int = 1024 * 1024 * 1024 # 1 ГБ
WORKSPACE_DIR_FREE_DELTA_BYTES:  int = 20 * 1024 * 1024   # 20 МБ — порог пропуска пересчёта
WORKSPACE_DIR_CACHE_TTL_SEC:     int = 600                 # 10 минут

# Сообщения об ошибках лимитов (используются внутри workspace-инструментов)
ERR_WS_FILE_TOO_LARGE = (
    "[error: file size {size} bytes exceeds limit {limit} bytes. "
    "Write refused to prevent disk overflow.]"
)
ERR_WS_DIR_TOO_LARGE = (
    "[error: workspace size {size} bytes exceeds limit {limit} bytes. "
    "Write refused to prevent disk overflow.]"
)

@tool
def sympy_solve(expression: str, variable: str = "x") -> str:
    """
    Вычисляет математическое выражение или решает уравнение через SymPy.

    Поведение зависит от формата expression:
      - Уравнение вида "lhs = rhs" (например "x**2 - 4 = 0") -- решается через sympy.solve,
        возвращает список корней.
      - Выражение без "=" (например "diff(sin(x), x)") -- вычисляется через sympy.simplify,
        возвращает упрощённый результат.

    Параметры:
        expression -- математическое выражение или уравнение в синтаксисе SymPy.
        variable   -- переменная для solve/diff (по умолчанию "x").

    Зависимость: pip install sympy
    """
    try:
        import sympy as _sp
        sym = _sp.Symbol(variable)
        local_ns = {variable: sym, **{k: getattr(_sp, k) for k in dir(_sp) if not k.startswith("_")}}
        if "=" in expression:
            lhs_str, rhs_str = expression.split("=", 1)
            lhs = _sp.sympify(lhs_str.strip(), locals=local_ns)
            rhs = _sp.sympify(rhs_str.strip(), locals=local_ns)
            solutions = _sp.solve(lhs - rhs, sym)
            return str(solutions)
        else:
            expr = _sp.sympify(expression.strip(), locals=local_ns)
            return str(_sp.simplify(expr))
    except ImportError:
        return ERR_SYMPY_NOT_INSTALLED
    except Exception as e:
        return ERR_SYMPY_GENERAL.format(e=e)

_BUILTIN_SYMPY      = [sympy_solve]


class AppriseConfig:
    """
    Конфигурация уведомлений через Apprise (pip install apprise).

    Хранит:
        urls      -- список URL дефолтных каналов (отправляются всегда при notify()).
        channels  -- именованные каналы: {"telegram": "tgram://...", "email": "mailto://..."}.
                    ЛЛМ может указать channel= чтобы отправить только в конкретный канал.

    Использование из кода:
        ai.apprise.urls.append("tgram://bot_token/chat_id")
        ai.apprise.channels["work"] = "discord://webhook_id/token"
        ai.notify("Заголовок", "Текст")                  # -> все urls
        ai.notify("Заголовок", "Текст", channel="work")  # -> только work

    URL-синтаксис Apprise: https://github.com/caronc/apprise/wiki
    Примеры:
        Telegram  tgram://{bot_token}/{chat_id}
        Discord   discord://{webhook_id}/{webhook_token}
        Email     mailto://{user}:{password}@gmail.com
        Gotify    gotify://{hostname}/{token}
        Ntfy      ntfy://{hostname}/{topic}
        Slack     slack://{token_a}/{token_b}/{token_c}/#{channel}
    """
    def __init__(self):
        self.urls:     list[str]       = []
        self.channels: dict[str, str]  = {}


class AIList(MCPMixin, AIListBase):
    """
    Конкретная реализация AIListBase + MCPMixin.

    Определяет модель, промпты, логирование и реестр MCP-серверов MCPServers.
    Добавление нового MCP-сервера -- только здесь: @staticmethod _build_* + запись в MCPServers.
    Подробная инструкция по типам серверов -- в docstring класса MCPMixin.
    """

    # -- args_builder-функции ---------------------------------------------
    # Одна функция = один сервер. Каждая инкапсулирует знания о своём сервере:
    # какие kwargs принимать и что из них строить (extra_args, env, url_override).

    @staticmethod
    def _build_playwright(**kwargs):
        """
        dirs[0]    -- путь к папке профиля браузера для сохранения сессий/cookies.
                     Если не передан -- каждый раз создаётся временный профиль.
        allowed[0] -- "headless" чтобы скрыть окно браузера (по умолчанию окно видно).

        Версия 1.0.12: parseArgs() в пакете обрабатывает только --port и --help.
        CLI-аргумент --headless и env-переменная HEADLESS полностью игнорируются.
        Headless-режим управляется только через параметр headless=true в каждом
        вызове инструмента playwright_navigate. Флаг allowed=["headless"] здесь
        только сигнализирует MCPServerDef.system_prompt о необходимости headless --
        сам параметр в extra не передаётся (бесполезен).
        """
        env  = {}
        dirs    = kwargs.get("dirs")
        allowed = kwargs.get("allowed")
        if dirs:
            env["PLAYWRIGHT_USER_DATA_DIR"] = dirs[0]
        # allowed=["headless"] обрабатывается через system_prompt в MCPServerDef,
        # который инструктирует модель передавать headless=true в playwright_navigate.
        # Передавать что-либо в extra или env не нужно -- пакет это не читает.
        return [], env, None

    @staticmethod
    def _build_memory_plus(**kwargs):
        """
        dirs[0] -- путь к файлу памяти .jsonl (опционально).
                   Если не передан -- файл создаётся во временной директории.
        """
        dirs = kwargs.get("dirs")
        env  = {"MEMORY_FILE_PATH": dirs[0]} if dirs else {}
        return [], env, None

    @staticmethod
    def _build_qdrant(**kwargs):
        """
        dirs[0]    -- URL базы данных Qdrant (опционально, по умолчанию localhost:6333).
        allowed[0] -- имя коллекции (опционально, по умолчанию "default").
        EMBEDDING_MODEL захардкожен: sentence-transformers/all-MiniLM-L6-v2.
        При первом запуске uvx скачивает модель с HuggingFace (~90 MB).
        FASTEMBED_CACHE_PATH: явно задаём папку кэша внутри _cache_dir/fastembed.
        Это переносит кэш из Temp (где Windows не создаёт симлинки и очищает при перезагрузке)
        в стабильный AppData, и fastembed копирует файлы напрямую без симлинков.
        """
        from pathlib import Path as _Path
        dirs    = kwargs.get("dirs")
        allowed = kwargs.get("allowed")
        try:
            from platformdirs import user_cache_dir as _ucd
            _fastembed_cache = str(_Path(_ucd("ailist")) / "fastembed")
        except ImportError:
            _fastembed_cache = str(_Path.home() / ".cache" / "ailist" / "fastembed")
        return [], {
            "QDRANT_URL":           dirs[0] if dirs else "http://localhost:6333",
            "COLLECTION_NAME":      allowed[0] if allowed else "default",
            "EMBEDDING_MODEL":      "sentence-transformers/all-MiniLM-L6-v2",
            "FASTEMBED_CACHE_PATH": _fastembed_cache,
        }, None

    @staticmethod
    def _build_searxng(**kwargs):
        """
        url -- адрес SSE-сервера (опционально, переопределяет MCPServerDef.url).
        """
        return [], {}, kwargs.get("url")

    @staticmethod
    def _build_serena(**kwargs):
        """
        Serena MCP server -- semantic code analysis and editing via LSP.
        https://github.com/oraios/serena

        kwargs:
            dirs[0]    -- optional: project path to activate on startup
                         (equivalent to --project <path>).
                         If omitted, the LLM can call activate_project() itself.
            context    -- Serena operation context (default: "ide").
                         "ide" disables Serena's own file ops and shell -- use this
                         when workspace server is connected (they complement each other).
                         "agent" -- full Serena toolset including file ops and shell.
                         "desktop-app" -- same as agent, for GUI clients.
                         "claude-code", "codex" -- optimised for those tools.
            modes      -- list of mode names to enable (default: ["interactive", "editing"]).
                         Specifying any modes overrides Serena's defaults entirely,
                         so include "interactive" and "editing" if you want to keep them.
                         "no-onboarding" -- skip onboarding if project was already indexed.
                         "no-memories"   -- disable memory tools and onboarding.
                         "planning"      -- analysis and planning focus, less direct editing.
                         "one-shot"      -- complete task in a single response.
            dangerous_shell -- allow dangerous shell commands in execute_shell_command
                         (default: False -- safe mode). Only relevant in context="agent".
            no_dashboard -- suppress opening the web dashboard on startup (default: True).

        Recommended usage with workspace server:
            await ai.mcp_connects([
                {"name": "workspace"},
                {"name": "serena", "dirs": [r"C:\\MyProject"], "context": "ide"},
            ])
        Second run (project already indexed):
            {"name": "serena", "dirs": [...], "context": "ide",
             "modes": ["no-onboarding", "interactive", "editing"]}
        """
        project      = (kwargs.get("dirs") or [None])[0]
        context      = kwargs.get("context", "ide")
        modes        = kwargs.get("modes", [])
        dangerous    = kwargs.get("dangerous_shell", False)
        no_dashboard = kwargs.get("no_dashboard", True)

        extra = ["start-mcp-server", "--context", context]

        if project:
            extra += ["--project", project]

        for mode in modes:
            extra += ["--mode", mode]

        if dangerous:
            extra.append("--allow-dangerous-shell-commands")

        if no_dashboard:
            extra += ["--open-web-dashboard", "false"]

        return extra, {}, None

    # -- Apprise: публичный метод и builtin-инструмент --------------------

    def notify(
        self,
        title:   str,
        body:    str,
        channel: str = "",
    ) -> bool:
        """
        Отправляет уведомление через Apprise.

        Параметры:
            title   -- заголовок уведомления.
            body    -- текст уведомления.
            channel -- имя именованного канала из self.apprise.channels.
                      Пустая строка -> отправить на все self.apprise.urls.
                      Если канал не найден -- бросает ValueError.

        Возвращает True если хотя бы одно уведомление доставлено.
        Требует: pip install apprise

        Примеры:
            ai.notify("Готово", "Задача выполнена")
            ai.notify("Ошибка", "Что-то пошло не так", channel="telegram")
        """
        try:
            import apprise as _apprise
        except ImportError:
            raise ImportError(ERR_NOTIFY_NO_APPRISE)

        cfg = self.apprise
        if channel:
            if channel not in cfg.channels:
                raise ValueError(
                    ERR_NOTIFY_CHANNEL_NOT_FOUND.format(
                        channel=channel,
                        available=list(cfg.channels.keys()) or "(none)",
                    )
                )
            urls = [cfg.channels[channel]]
        else:
            urls = list(cfg.urls)

        if not urls:
            raise ValueError(ERR_NOTIFY_NO_URLS)

        ap = _apprise.Apprise()
        for url in urls:
            ap.add(url)
        return ap.notify(title=title, body=body)

    def _make_apprise_tool(self):
        """
        Создаёт builtin LangChain-инструмент notify_send, замкнутый на self.
        Sentinel-строка "" для channel означает "использовать все дефолтные URLs".
        ЛЛМ не должна вызывать этот инструмент без явной просьбы пользователя.
        """
        _self = self

        @tool
        def notify_send(title: str, body: str, channel: str = "") -> str:
            """
            Send a notification via Apprise to the user's configured channels.
            Call this ONLY when the user explicitly asks to send a notification or alert.

            Args:
                title:   Notification title.
                body:    Notification body text.
                channel: Named channel from apprise.channels (e.g. 'telegram', 'email').
                         Leave empty to send to all default URLs (apprise.urls).
            """
            try:
                ok = _self.notify(title=title, body=body, channel=channel)
                if ok:
                    dest = f"channel '{channel}'" if channel else "all default channels"
                    return f"Notification sent to {dest}."
                return "Notification was not delivered (check URLs and credentials)."
            except (ImportError, ValueError) as e:
                return f"[error] {e}"

        return notify_send

    # -- Skills directory --------------------------------------------------

    def get_skills_dir(self) -> Path | None:
        """
        Возвращает абсолютный путь к директории скиллов.

        Если skills_dir -- абсолютный путь, возвращает его напрямую.
        Если относительный -- резолвит относительно workspace_dir.
        None если workspace_dir не задан и skills_dir относительный.

        Директория не создаётся автоматически.
        """
        return self._resolve_dir(self.skills_dir)

    # -- Attachments toggle ------------------------------------------------

    def set_use_attachments(self, enabled: bool) -> None:
        """
        Включает или выключает буфер вложений.

        При enabled=True:  восстанавливает systems[-1] из self._prompt_attachments.
        При enabled=False: удаляет systems[-1] (промпт [ATTACHMENT WORKSPACE POLICY]).

        Идентификация промпта -- по заголовку "[ATTACHMENT WORKSPACE POLICY]"
        в тексте systems[-1], поэтому работает корректно даже если пользователь
        изменил self._prompt_attachments.
        """
        self.use_attachments = enabled
        if enabled:
            self.systems[-1] = {"prompt": self._prompt_attachments}
        else:
            entry = self.systems.get(-1)
            if isinstance(entry, dict) and "[ATTACHMENT WORKSPACE POLICY]" in entry.get("prompt", ""):
                del self.systems[-1]

    # -- Prompt variable substitution --------------------------------------

    @staticmethod
    def _prompt_text(v) -> str:
        """Extract text from a systems/prompts entry for cache snapshot."""
        if isinstance(v, dict):
            return v.get("prompt", "")
        return str(v) if v else ""

    def prepare(self) -> None:
        """
        Overrides AIListBase.prepare() to substitute {WORKSPACE_DIR}, {ATTACHMENTS_DIR},
        {SKILLS_DIR} into a shadow copy of self.systems before passing it to the base class.

        self.systems is never mutated -- the user's templates are preserved as-is.
        The resolved copy is cached and rebuilt only when workspace_dir, attachments_dir,
        skills_dir, system_tool_instructions, or the user-defined systems entries change.
        systems[0] is excluded from the cache key -- it is computed, not user-defined.
        """
        ws = str(self.workspace_dir) if self.workspace_dir else "(not set)"
        ad_path = self._resolve_dir(self.attachments_dir)
        ad = str(ad_path) if ad_path is not None else "(not set)"
        sd_path = self.get_skills_dir()
        sd = str(sd_path) if sd_path is not None else "(not set)"

        # Создаём папку вложений при необходимости -- непосредственно перед run.
        if self.use_attachments and ad_path is not None:
            ad_path.mkdir(parents=True, exist_ok=True)

        # Cache key: variable values + snapshot of user-defined systems entries.
        # Key 0 is excluded -- it is written by super().prepare(), not the user.
        # system_tool_instructions is included because it changes on mcp_connect/disconnect.
        systems_snapshot = tuple(
            (k, self._prompt_text(v))
            for k, v in sorted(self.systems.items())
            if k != 0
        )
        cache_key = (ws, ad, sd, self.system_tool_instructions, systems_snapshot)

        current_key      = getattr(self, "_prompt_vars_cache_key", None)
        current_resolved = getattr(self, "_prompt_vars_resolved", {})

        if cache_key != current_key:
            replacements = {
                "{WORKSPACE_DIR}":   ws,
                "{ATTACHMENTS_DIR}": ad,
                "{SKILLS_DIR}":      sd,
            }
            resolved = {}
            for k, v in self.systems.items():
                if k == 0:
                    continue  # systems[0] is computed by super().prepare(), skip here
                if isinstance(v, dict) and "prompt" in v:
                    text = v["prompt"]
                    for placeholder, value in replacements.items():
                        text = text.replace(placeholder, value)
                    resolved[k] = {**v, "prompt": text}
                else:
                    resolved[k] = v
            self._prompt_vars_cache_key = cache_key
            self._prompt_vars_resolved  = resolved
            current_resolved = resolved

        # Run super().prepare() with the resolved copy so it writes systems[0]
        # (combined self.system + self.system_tool_instructions) into current_resolved.
        # Then sync systems[0] back to self.systems so compile_combine can read it.
        saved = self.systems
        self.systems = current_resolved
        try:
            super().prepare()
        finally:
            self.systems = saved
        if 0 in current_resolved:
            self.systems[0] = current_resolved[0]
        elif 0 in self.systems:
            del self.systems[0]

    def compile_combine(self, prt) -> list:
        """
        Overrides AIListBase.compile_combine() to pass the variable-substituted
        copy of self.systems to compile_section for the entire compilation.

        self.prepare() is called explicitly first to ensure _prompt_vars_resolved
        is built before the swap -- this matters when compile_combine is called
        directly (e.g. from get_systemprompt_log) before any run_async.
        The second prepare() call inside super().compile_combine() hits the cache
        and is effectively free.
        """
        self.prepare()
        current_resolved = getattr(self, "_prompt_vars_resolved", self.systems)
        saved = self.systems
        self.systems = current_resolved
        try:
            return super().compile_combine(prt)
        finally:
            self.systems = saved

    def _check_workspace_write(self, content_len: int = 0) -> str | None:
        """
        Проверяет лимиты перед записью файла:
        1. Размер записываемого контента (content_len) против WORKSPACE_FILE_LIMIT_BYTES.
        2. Суммарный размер папки workspace против WORKSPACE_DIR_LIMIT_BYTES.

        Размер папки кэшируется:
          - пропускает пересчёт если free disk изменился < WORKSPACE_DIR_FREE_DELTA_BYTES
            с момента прошлого измерения И не истёк TTL кэша.
          - сбрасывает кэш через WORKSPACE_DIR_CACHE_TTL_SEC секунд.

        Возвращает None если всё в порядке, или строку ошибки если лимит превышен.
        Если workspace_dir не задан — всегда None.
        """
        import time as _time
        import shutil as _shutil_ws
        import os as _os_ws

        # -- Лимит одного файла --
        if WORKSPACE_FILE_LIMIT_BYTES > 0 and content_len > WORKSPACE_FILE_LIMIT_BYTES:
            return ERR_WS_FILE_TOO_LARGE.format(
                size=content_len, limit=WORKSPACE_FILE_LIMIT_BYTES
            )

        # -- Лимит папки --
        if WORKSPACE_DIR_LIMIT_BYTES <= 0 or self.workspace_dir is None:
            return None

        now = _time.monotonic()
        ttl_expired = (now - self._ws_size_cache_time) > WORKSPACE_DIR_CACHE_TTL_SEC

        # Текущее свободное место на диске
        try:
            free_now = _shutil_ws.disk_usage(self.workspace_dir).free
        except Exception:
            return None  # не можем проверить — пропускаем

        free_changed = abs(free_now - self._ws_free_cache) >= WORKSPACE_DIR_FREE_DELTA_BYTES

        if ttl_expired or free_changed or self._ws_free_cache < 0:
            # Пересчитываем размер папки
            total = 0
            try:
                for entry in _os_ws.scandir(self.workspace_dir):
                    try:
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat().st_size
                        elif entry.is_dir(follow_symlinks=False):
                            for root, _dirs, files in _os_ws.walk(entry.path):
                                for f in files:
                                    try:
                                        total += _os_ws.path.getsize(_os_ws.path.join(root, f))
                                    except OSError:
                                        pass
                    except OSError:
                        pass
            except Exception:
                return None

            self._ws_size_cache = total
            self._ws_free_cache = free_now
            self._ws_size_cache_time = now

        if self._ws_size_cache > WORKSPACE_DIR_LIMIT_BYTES:
            return ERR_WS_DIR_TOO_LARGE.format(
                size=self._ws_size_cache, limit=WORKSPACE_DIR_LIMIT_BYTES
            )

        return None

    def _make_workspace_tools(self, exclude: set[str] | None = None, exclude_patterns: list[str] | None = None):
        """
        Creates all built-in workspace tools bound to this instance.

        File editing: view, str_replace, create_file, write_file, insert, undo_edit.
        File navigation: list_dir, dir_tree, find_file, file_info, make_dir, move_file, read_files.
        Search and analysis: grep, search_for_pattern, head, tail, wc.
        Execution: run, python_run.

        Backups (one level per file) are stored in self._workspace_backups.
        All paths are checked against workspace_dir.
        Enable with: await ai.mcp_connect("workspace")
        """
        import re as _re
        import os as _os
        import shutil as _shutil
        import fnmatch as _fnmatch
        import io as _io
        import threading as _threading
        import csv as _csv
        import math as _math
        import itertools as _itertools
        import collections as _collections
        import glob as _glob
        import datetime as _datetime
        import hashlib as _hashlib
        import json as _json

        _backups = self._workspace_backups
        _self = self

        def _check(path: str) -> str | None:
            return _self._check_workspace_path(path)

        def _resolve(path: str) -> str:
            """
            Абсолютный путь для файловых I/O-операций, НЕ зависящий от текущей
            рабочей директории процесса (os.getcwd()). Вызывать после успешной
            проверки _check(path). Для сообщений/логов продолжаем использовать
            исходный (возможно относительный) `path`, как и раньше.
            """
            return str(_self._resolve_workspace_path(path))

        def _check_write(content: str) -> str | None:
            return _self._check_workspace_write(len(content.encode("utf-8")))

        def _backup(path: str):
            if path not in _backups:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        _backups[path] = f.read()
                except Exception:
                    pass

        # -- File reading --------------------------------------------------

        @tool
        def view(path: str, start_line: int = 0, end_line: int = -1) -> str:
            """
            Read a file with line numbers. Optionally read only a specific line range.

            Always call view() before str_replace to confirm the exact text to replace.
            For large files, use start_line/end_line to read only the relevant section.

            Args:
                path:       Path to the file.
                start_line: First line to show (1-indexed). 0 means from the beginning.
                end_line:   Last line to show (inclusive). -1 means until end of file.

            Returns file contents with line numbers (format: "N\\tline").

            Examples:
                view("app.py")                    # read the whole file
                view("app.py", start_line=10, end_line=50)  # read lines 10 to 50 only
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                abs_path = _resolve(path)
                if not Path(abs_path).exists():
                    return ERR_EDITOR_FILE_NOT_FOUND.format(path=path)
                with open(abs_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                total_lines = len(lines)
                if start_line > 0 or end_line != -1:
                    s = max(0, start_line - 1)
                    e = end_line if end_line != -1 else len(lines)
                    lines = lines[s:e]
                    base = start_line if start_line > 0 else 1
                else:
                    base = 1
                result = "".join(f"{base + i}\t{line}" for i, line in enumerate(lines))
                # Мягкое предупреждение для больших файлов (только при чтении всего файла)
                if (start_line == 0 and end_line == -1 and total_lines > 300):
                    result += (
                        f"\n[NOTE: file has {total_lines} lines. "
                        f"Use view(path, start_line=N, end_line=M) to read specific sections, "
                        f"or grep/head/tail to navigate large files.]"
                    )
                return result
            except Exception as exc:
                return ERR_EDITOR_VIEW.format(e=exc)

        @tool
        def read_files(paths: list) -> str:
            """
            Read multiple files in a single call. More efficient than calling view() repeatedly.

            Returns each file's contents prefixed with a header line.
            Errors for individual files are reported inline without stopping the rest.

            Args:
                paths: List of file paths to read (list of strings).
                       Example: ["src/main.py", "src/utils.py"]
            """
            parts = []
            for path in paths:
                ws_err = _check(path)
                if ws_err:
                    parts.append(f"--- File: {path} ---\n{ws_err}")
                    continue
                try:
                    abs_path = _resolve(path)
                    with open(abs_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    parts.append(f"--- File: {path} ---\n{content}")
                except Exception as exc:
                    parts.append(f"--- File: {path} ---\n" + ERR_WS_READ_FAILED.format(e=exc))
            return "\n\n".join(parts) if parts else "[no files]"

        # -- File writing --------------------------------------------------

        @tool
        def str_replace(path: str, old_str: str, new_str: str) -> str:
            """
            Replace an exact unique string in a file with new text.

            old_str must appear exactly once in the file -- use view() first to confirm.
            Saves a backup for undo_edit().

            Args:
                path:    Path to the file.
                old_str: Text to find and replace (must be unique in the file).
                new_str: Replacement text.
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                abs_path = _resolve(path)
                if not Path(abs_path).exists():
                    return ERR_EDITOR_FILE_NOT_FOUND.format(path=path)
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                count = content.count(old_str)
                if count == 0:
                    return ERR_EDITOR_TEXT_NOT_FOUND.format(path=path)
                if count > 1:
                    return ERR_EDITOR_MULTIPLE_MATCHES.format(count=count, path=path)
                new_content = content.replace(old_str, new_str, 1)
                ws_err = _check_write(new_content)
                if ws_err:
                    return ws_err
                _backup(abs_path)
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return MSG_EDITOR_REPLACED
            except Exception as exc:
                return ERR_EDITOR_REPLACE.format(e=exc)

        @tool
        def create_file(path: str, content: str) -> str:
            """
            Create a new file with the given content.

            Fails if the file already exists -- use write_file() to overwrite.
            Creates parent directories automatically.

            Args:
                path:    Path for the new file.
                content: File content.
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                abs_path = _resolve(path)
                p = Path(abs_path)
                if p.exists():
                    return ERR_EDITOR_FILE_EXISTS.format(path=path)
                ws_err = _check_write(content)
                if ws_err:
                    return ws_err
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return MSG_EDITOR_CREATED.format(path=path)
            except Exception as exc:
                return ERR_EDITOR_CREATE.format(e=exc)

        @tool
        def write_file(path: str, content: str) -> str:
            """
            Write content to a file, creating or overwriting it.

            Creates parent directories automatically.
            Saves a backup of the previous version for undo_edit().
            Use create_file() if you want an error when the file already exists.

            Args:
                path:    Path to the file.
                content: Content to write.
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                ws_err = _check_write(content)
                if ws_err:
                    return ws_err
                abs_path = _resolve(path)
                p = Path(abs_path)
                if p.exists():
                    _backup(abs_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return MSG_WS_FILE_WRITTEN.format(path=path)
            except Exception as exc:
                return ERR_WS_WRITE_FAILED.format(e=exc)

        @tool
        def insert(path: str, line: int, content: str) -> str:
            """
            Insert text after a specific line in a file.

            Always call view() first to find the correct line number.
            Saves a backup for undo_edit().

            Args:
                path:    Path to the file.
                line:    Insert AFTER this line number.
                         Use 0 to insert at the very beginning of the file.
                         Example: line=5 inserts after line 5 (before line 6).
                content: Text to insert. A newline is appended automatically if missing.
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                abs_path = _resolve(path)
                if not Path(abs_path).exists():
                    return ERR_EDITOR_FILE_NOT_FOUND.format(path=path)
                with open(abs_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if line > len(lines):
                    return ERR_EDITOR_LINE_EXCEEDS.format(line=line, length=len(lines))
                text = content if content.endswith("\n") else content + "\n"
                new_content = "".join(lines[:line]) + text + "".join(lines[line:])
                ws_err = _check_write(new_content)
                if ws_err:
                    return ws_err
                _backup(abs_path)
                lines.insert(line, text)
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                return MSG_EDITOR_INSERTED.format(line=line)
            except Exception as exc:
                return ERR_EDITOR_INSERT.format(e=exc)

        @tool
        def undo_edit(path: str) -> str:
            """
            Restore a file to its state before the last str_replace, write_file, or insert.

            Only one level of undo is available per file.

            Args:
                path: Path to the file to restore.
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                abs_path = _resolve(path)
                if abs_path not in _backups:
                    return ERR_EDITOR_NO_BACKUP.format(path=path)
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(_backups.pop(abs_path))
                return MSG_EDITOR_RESTORED.format(path=path)
            except Exception as exc:
                return ERR_EDITOR_UNDO.format(e=exc)

        # -- Directory and file operations ---------------------------------

        @tool
        def list_dir(path: str) -> str:
            """
            List the immediate contents of a directory (files and subdirectories, one level only).

            Use list_dir_tree() to see the full recursive structure.

            Args:
                path: Directory path to list.
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                p = Path(_resolve(path))
                if not p.exists() or not p.is_dir():
                    return ERR_WS_DIR_NOT_FOUND.format(path=path)
                lines = []
                for item in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                    tag = "[dir] " if item.is_dir() else "      "
                    lines.append(f"{tag}{item}")
                return "\n".join(lines) if lines else "(empty)"
            except Exception as exc:
                return ERR_WS_READ_FAILED.format(e=exc)

        @tool
        def list_dir_tree(path: str) -> str:
            """
            List ALL files and subdirectories inside a directory, recursively (full tree).

            Use list_dir() to see only the immediate contents (one level).

            Args:
                path: Root directory path to traverse.
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                p = Path(_resolve(path))
                if not p.exists() or not p.is_dir():
                    return ERR_WS_DIR_NOT_FOUND.format(path=path)
                # Директории которые не нужны ни модели ни пользователю
                SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
                lines = []
                for root, dirs, files in _os.walk(str(p)):
                    # Мутируем dirs на месте — os.walk не будет заходить в исключённые папки
                    dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                    root_path = Path(root)
                    for d in dirs:
                        lines.append(f"[dir] {root_path / d}")
                    for f in sorted(files):
                        lines.append(f"      {root_path / f}")
                return "\n".join(lines) if lines else "(empty)"
            except Exception as exc:
                return ERR_WS_READ_FAILED.format(e=exc)

        @tool
        def find_file(pattern: str, root: str = ".") -> str:
            """
            Find files matching a name pattern, searching recursively from root.

            Args:
                pattern: Glob pattern for the filename (e.g. "*.py", "config.*", "test_*").
                root:    Directory to search from. Defaults to current directory.

            Returns a list of matching file paths, one per line.
            """
            try:
                ws_err = _check(root)
                if ws_err:
                    return ws_err
                p = Path(_resolve(root))
                if not p.exists():
                    return ERR_WS_DIR_NOT_FOUND.format(path=root)
                matches = sorted(str(m.resolve()) for m in p.rglob(pattern))
                return "\n".join(matches) if matches else "(no files found)"
            except Exception as exc:
                return ERR_WS_READ_FAILED.format(e=exc)

        @tool
        def file_info(path: str) -> str:
            """
            Get file metadata: size in bytes, last modified time, file/dir type.

            Use before reading a large file to check its size.
            Use to verify that a file or directory exists.

            Args:
                path: Path to the file or directory.
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                p = Path(_resolve(path))
                if not p.exists():
                    return ERR_EDITOR_FILE_NOT_FOUND.format(path=path)
                st = p.stat()
                kind = "dir" if p.is_dir() else "file"
                ext = p.suffix if p.is_file() else ""
                mtime = _datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
                return (
                    f"type: {kind}\n"
                    f"size: {st.st_size} bytes\n"
                    f"modified: {mtime}\n"
                    + (f"extension: {ext}\n" if ext else "")
                )
            except Exception as exc:
                return ERR_WS_STAT_FAILED.format(e=exc)

        @tool
        def make_dir(path: str) -> str:
            """
            Create a directory, including all missing parent directories.

            Args:
                path: Directory path to create.
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                Path(_resolve(path)).mkdir(parents=True, exist_ok=True)
                return MSG_WS_DIR_CREATED.format(path=path)
            except Exception as exc:
                return ERR_WS_MKDIR_FAILED.format(e=exc)

        @tool
        def move_file(src: str, dst: str) -> str:
            """
            Move or rename a file or directory.

            Args:
                src: Source path.
                dst: Destination path.
            """
            try:
                ws_err = _check(src)
                if ws_err:
                    return ws_err
                ws_err = _check(dst)
                if ws_err:
                    return ws_err
                _shutil.move(_resolve(src), _resolve(dst))
                return MSG_WS_MOVED.format(src=src, dst=dst)
            except Exception as exc:
                return ERR_WS_MOVE_FAILED.format(e=exc)

        # -- Search and analysis -------------------------------------------

        @tool
        def grep(path: str, pattern: str, context_lines: int = 0) -> str:
            """
            Search within ONE FILE for lines matching a pattern. Returns line numbers and context.

            Use this before str_replace to find the exact location of text to edit.
            Use search_for_pattern() when you need to search across multiple files.

            Args:
                path:          File to search (must be a file, not a directory).
                pattern:       Regular expression (re.search semantics).
                context_lines: Lines of context before and after each match (default 0).
                               Example: context_lines=2 shows 2 lines before and after.

            Returns matching lines with line numbers in format "filepath:N> line".
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                abs_path = _resolve(path)
                if not Path(abs_path).exists():
                    return ERR_EDITOR_FILE_NOT_FOUND.format(path=path)
                with open(abs_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                rx = _re.compile(pattern)
                match_indices = [i for i, l in enumerate(lines) if rx.search(l)]
                if not match_indices:
                    return "(no matches)"
                shown = set()
                for mi in match_indices:
                    for i in range(max(0, mi - context_lines), min(len(lines), mi + context_lines + 1)):
                        shown.add(i)
                result = []
                prev = None
                for i in sorted(shown):
                    if prev is not None and i > prev + 1:
                        result.append("--")
                    marker = ">" if i in set(match_indices) else " "
                    result.append(f"{path}:{i+1}{marker} {lines[i]}", )
                    prev = i
                return "".join(result)
            except Exception as exc:
                return ERR_WS_GREP_FAILED.format(e=exc)

        @tool
        def search_for_pattern(pattern: str, path: str = ".") -> str:
            """
            Search ACROSS ALL FILES in a directory for lines matching a pattern.

            Use this to find where a function, class, or variable is defined or used in a project.
            Use grep() when you already know the file and want line numbers with context.

            Args:
                pattern: Regular expression to search for.
                path:    Directory to search recursively, or a single file.
                         Defaults to current directory (searches the whole workspace).

            Returns matching lines with file paths and line numbers: "file:N: line".
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                p = Path(_resolve(path))
                if not p.exists():
                    return ERR_EDITOR_FILE_NOT_FOUND.format(path=path)
                rx = _re.compile(pattern)
                results = []

                # Бинарные расширения — пропускаем содержимое, выдаём заглушку при совпадении имени
                _BINARY_EXT = {
                    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
                    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
                    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
                    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
                    ".mp3", ".mp4", ".wav", ".ogg", ".avi", ".mov",
                    ".woff", ".woff2", ".ttf", ".eot",
                    ".pyc", ".pyd", ".pyo",
                }
                # Максимальный размер файла для поиска (1 МБ)
                _MAX_FILE_BYTES = 1 * 1024 * 1024
                # Максимальное количество строк результата
                _MAX_RESULT_LINES = 200

                def _search_file(fp):
                    fp = Path(fp)
                    # Пропускаем файлы из списка исключений
                    if exclude and fp.name in exclude:
                        return
                    if exclude_patterns and any(_fnmatch.fnmatch(fp.name, p) for p in exclude_patterns):
                        return
                    # Пропускаем бинарные по расширению
                    if fp.suffix.lower() in _BINARY_EXT:
                        if rx.search(fp.name):
                            results.append(f"{fp}: [binary file, content skipped]")
                        return
                    # Пропускаем слишком большие файлы
                    try:
                        if fp.stat().st_size > _MAX_FILE_BYTES:
                            results.append(f"{fp}: [file too large ({fp.stat().st_size} bytes), skipped]")
                            return
                    except OSError:
                        return
                    try:
                        with open(fp, "r", encoding="utf-8", errors="replace") as f:
                            for i, line in enumerate(f, 1):
                                if len(results) >= _MAX_RESULT_LINES:
                                    results.append(f"[... truncated: more than {_MAX_RESULT_LINES} matches ...]")
                                    return
                                if rx.search(line):
                                    results.append(f"{fp}:{i}: {line.rstrip()}")
                    except Exception:
                        pass

                if p.is_file():
                    _search_file(p)
                else:
                    for fp in sorted(p.rglob("*")):
                        if len(results) >= _MAX_RESULT_LINES:
                            results.append(f"[... truncated: more than {_MAX_RESULT_LINES} matches ...]")
                            break
                        if fp.is_file():
                            _search_file(fp)
                return "\n".join(results) if results else "(no matches)"
            except Exception as exc:
                return ERR_WS_GREP_FAILED.format(e=exc)

        @tool
        def head(path: str, n: int = 20) -> str:
            """
            Show the first N lines of a file with line numbers.

            Args:
                path: Path to the file.
                n:    Number of lines to show (default 20).
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                abs_path = _resolve(path)
                if not Path(abs_path).exists():
                    return ERR_EDITOR_FILE_NOT_FOUND.format(path=path)
                with open(abs_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()[:n]
                return "".join(f"{i+1}\t{l}" for i, l in enumerate(lines))
            except Exception as exc:
                return ERR_EDITOR_VIEW.format(e=exc)

        @tool
        def tail(path: str, n: int = 20) -> str:
            """
            Show the last N lines of a file with line numbers.

            Args:
                path: Path to the file.
                n:    Number of lines to show (default 20).
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                abs_path = _resolve(path)
                if not Path(abs_path).exists():
                    return ERR_EDITOR_FILE_NOT_FOUND.format(path=path)
                with open(abs_path, "r", encoding="utf-8") as f:
                    all_lines = f.readlines()
                lines = all_lines[-n:]
                base = max(1, len(all_lines) - n + 1)
                return "".join(f"{base + i}\t{l}" for i, l in enumerate(lines))
            except Exception as exc:
                return ERR_EDITOR_VIEW.format(e=exc)

        @tool
        def wc(path: str) -> str:
            """
            Count the number of lines, words, and characters in a file (like Unix wc).

            Use this to quickly check file size in lines before deciding whether to
            read the whole file or use a line range with view().

            Args:
                path: Path to the file.
            """
            try:
                ws_err = _check(path)
                if ws_err:
                    return ws_err
                abs_path = _resolve(path)
                if not Path(abs_path).exists():
                    return ERR_EDITOR_FILE_NOT_FOUND.format(path=path)
                with open(abs_path, "r", encoding="utf-8") as f:
                    text = f.read()
                lines = text.count("\n")
                words = len(text.split())
                chars = len(text)
                return f"lines: {lines}, words: {words}, chars: {chars}"
            except Exception as exc:
                return ERR_EDITOR_VIEW.format(e=exc)

        # -- Execution -----------------------------------------------------

        @tool
        def run(command: str, cwd: str = "") -> str:
            """
            Run an external command and return its output (stdout + stderr combined).

            Safe mode (default): shell=False, no shell builtins, no pipes.
            Only external binaries in PATH work: python, git, pytest, ruff, mypy, etc.
            For pipes, redirects, or complex multi-step logic use python_run() instead.

            Args:
                command: Command with arguments.
                         Examples: "git log --oneline -5"
                                   "python -m pytest tests/test_foo.py -v"
                                   "ruff check src/"
                cwd:     Working directory (str). Leave empty to use workspace_dir.

            Returns combined stdout + stderr. Non-zero exit code appended as [exit code: N].
            """
            import subprocess as _sp
            import shlex as _shlex
            import sys as _sys

            effective_cwd: str | None
            if cwd:
                err = _self._check_workspace_path(cwd)
                if err:
                    return err
                effective_cwd = str(_self._resolve_workspace_path(cwd))
            elif _self.workspace_dir is not None:
                effective_cwd = str(_self.workspace_dir)
            else:
                effective_cwd = None

            try:
                if _SUBPROCESS_SAFE_MODE:
                    use_posix = _sys.platform != "win32"
                    try:
                        args = _shlex.split(command, posix=use_posix)
                    except ValueError as e:
                        return ERR_SUBPROCESS_PARSE.format(e=e)
                    if not args:
                        return ERR_SUBPROCESS_EMPTY
                    result = _sp.run(
                        args,
                        shell=False,
                        capture_output=True,
                        text=True,
                        cwd=effective_cwd,
                        timeout=30,
                    )
                else:
                    result = _sp.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        cwd=effective_cwd,
                        timeout=30,
                    )
                out = result.stdout or ""
                err = result.stderr or ""
                combined = (out + err).strip()
                if result.returncode != 0:
                    combined += f"\n[exit code: {result.returncode}]"
                return combined or "[no output]"
            except _sp.TimeoutExpired:
                return ERR_SUBPROCESS_TIMEOUT.format(timeout=30)
            except FileNotFoundError:
                return ERR_SUBPROCESS_NOT_FOUND.format(
                    cmd=command.split()[0] if command.split() else command
                )
            except Exception as e:
                return ERR_SUBPROCESS_GENERAL.format(e=e)

        @tool
        def python_run(code: str) -> str:
            """
            Execute Python code and capture printed output.

            Use for: file analysis (counting lines, finding patterns), JSON/CSV processing,
            multi-step calculations, anything that needs loops or data structures.
            For running external programs use run() instead.

            Available names: re, json, os, pathlib, Path, shutil, csv, math,
            itertools, collections, fnmatch, glob, datetime, hashlib.
            WORKSPACE (str) contains the workspace directory path.

            Use print() to produce output -- return values are ignored.
            Do not use os.system() -- use run() for external commands.
            All file paths must be inside WORKSPACE.

            Examples:
                # Find all function definitions in a file:
                lines = open(WORKSPACE + "/app.py").readlines()
                print(f"Total lines: {len(lines)}")
                for i, l in enumerate(lines, 1):
                    if l.strip().startswith("def "):
                        print(f"  {i}: {l.rstrip()}")

                # Read a JSON config and print a field:
                import json
                data = json.load(open(WORKSPACE + "/config.json"))
                print(data.get("version", "not found"))
            """
            import pathlib as _pathlib

            namespace = {
                "WORKSPACE": str(_self.workspace_dir) if _self.workspace_dir else "",
                "re":          _re,
                "json":        _json,
                "os":          _os,
                "pathlib":     _pathlib,
                "Path":        Path,
                "shutil":      _shutil,
                "csv":         _csv,
                "math":        _math,
                "itertools":   _itertools,
                "collections": _collections,
                "fnmatch":     _fnmatch,
                "glob":        _glob,
                "datetime":    _datetime,
                "hashlib":     _hashlib,
            }

            stdout_capture = _io.StringIO()
            result_holder = [None]
            exc_holder    = [None]

            def _run():
                import contextlib as _contextlib
                with _contextlib.redirect_stdout(stdout_capture):
                    try:
                        exec(code, namespace)  # noqa: S102
                    except Exception as e:
                        exc_holder[0] = e

            t = _threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=30)

            output = stdout_capture.getvalue()
            if t.is_alive():
                return ERR_WS_PYTHON_TIMEOUT.format(timeout=30)
            if exc_holder[0] is not None:
                import traceback as _tb
                return ERR_WS_PYTHON_ERROR.format(e=_tb.format_exc())
            return output if output else "[no output]"

        return [
            view, read_files,
            str_replace, create_file, write_file, insert, undo_edit,
            list_dir, list_dir_tree, find_file, file_info, make_dir, move_file,
            grep, search_for_pattern, head, tail, wc,
            run, python_run,
        ]

    def _make_skills_tools(self):
        """
        Creates the list_skills builtin tool, bound to this instance.
        Reads SKILL.md files from ai.skills_dir.
        Source adapted from FastSkills (MIT license, github: nj19257/FastSkills).
        Enable with: await ai.mcp_connect("skills")
        """
        import re as _re
        _FRONTMATTER_RE = _re.compile(r"^---\s*\n(.*?)\n---", _re.DOTALL)
        _YAML_NAME_RE   = _re.compile(r"^name:\s*(.+)", _re.MULTILINE)
        _YAML_DESC_RE   = _re.compile(r"^description:\s*(.+(?:\n\s+.+)*)", _re.MULTILINE)
        _self = self

        def _parse_frontmatter(text: str) -> tuple:
            fm_match = _FRONTMATTER_RE.match(text)
            if not fm_match:
                return "", ""
            frontmatter = fm_match.group(1)
            name_match = _YAML_NAME_RE.search(frontmatter)
            name = name_match.group(1).strip().strip("'\"") if name_match else ""
            desc_match = _YAML_DESC_RE.search(frontmatter)
            if desc_match:
                raw = desc_match.group(1)
                desc = " ".join(line.strip() for line in raw.splitlines())
                desc = desc.strip().strip("'\"")
            else:
                desc = ""
            return name, desc

        def _add_skill(out: list, skill_dir, skill_md, parse_fn) -> None:
            """Appends one skill entry to out, parsing frontmatter from skill_md."""
            try:
                text = skill_md.read_text(encoding="utf-8")
                name, desc = parse_fn(text)
            except Exception:
                name, desc = "", ""
            out.append({
                "name":        name or skill_dir.name,
                "description": desc,
                "dir":         str(skill_dir),
                "path":        str(skill_md),
            })

        @tool
        def list_skills() -> str:
            """
            List available agent skills with name, description, directory, and SKILL.md path.

            Workflow for using a skill:
              1. If you don't already know which skills are available, call list_skills() first.
                 Skip this step if the skill list is already in the conversation or system prompt.
              2. Call view(path=<SKILL.md path>) to read the skill's instructions.
                 Do not skip this step — the instructions tell you exactly what to do.
              3. Follow the instructions in SKILL.md exactly before doing anything else.

            The "dir" field is the skill directory — pass it as cwd to run() if needed.
            """
            skills_path = _self.get_skills_dir()
            if skills_path is None or not skills_path.exists() or not skills_path.is_dir():
                return ERR_WS_SKILLS_DIR_NOT_FOUND.format(path=_self.skills_dir)

            skills = []
            try:
                # Два уровня: skills/X/SKILL.md и skills/X/Y/SKILL.md.
                # Если нашли SKILL.md на уровне X — глубже не смотрим.
                for lvl1 in sorted(skills_path.iterdir(), key=lambda p: p.name.lower()):
                    if not lvl1.is_dir():
                        continue
                    if (lvl1 / "SKILL.md").exists():
                        # уровень 1 — скилл найден, внутрь не заходим
                        _add_skill(skills, lvl1, lvl1 / "SKILL.md", _parse_frontmatter)
                    else:
                        # уровень 1 — контейнер (например public/, examples/), смотрим внутрь
                        for lvl2 in sorted(lvl1.iterdir(), key=lambda p: p.name.lower()):
                            if not lvl2.is_dir():
                                continue
                            if (lvl2 / "SKILL.md").exists():
                                _add_skill(skills, lvl2, lvl2 / "SKILL.md", _parse_frontmatter)
                            # глубже уровня 2 не идём
            except PermissionError:
                return ERR_WS_SKILLS_DIR_NOT_FOUND.format(path=_self.skills_dir)

            if not skills:
                return f"(no skills found in {_self.skills_dir})"

            lines = [f"Found {len(skills)} skill(s) in {_self.skills_dir}:\n"]
            for s in skills:
                lines.append(f"- {s['name']}")
                if s["description"]:
                    lines.append(f"  description: {s['description']}")
                lines.append(f"  dir:  {s['dir']}")
                lines.append(f"  path: {s['path']}")
            lines.append(
                "\nTo use a skill: call view(path=<path>) to read its SKILL.md, "
                "then follow the instructions."
            )
            return "\n".join(lines)

        return [list_skills]



    # -- Инициализация -----------------------------------------------------

    def __init__(self, modelName: str, context_limit: int, provider, tools: list, context_schema=None,
                 base_url=None, api_key=None):
        # _static_tools нужен MCPMixin для пересборки агента
        self._static_tools = list(tools)

        # _workspace_backups нужен до _make_workspace_tools внутри MCPServers
        self._workspace_backups: dict[str, str] = {}

        # skills_dir -- имя поддиректории внутри workspace_dir со скиллами агента.
        # Полный путь возвращает get_skills_dir().
        # Менять: ai.skills_dir = "skills"  или  ai.skills_dir = r"C:\absolute\path"
        # Абсолютный путь используется как есть; относительный -- от workspace_dir.
        self.skills_dir: str = "skills"

        super().__init__(modelName, context_limit, provider, tools, context_schema, base_url, api_key)

        # -- Policy prompts ------------------------------------------------
        # Negative keys keep these blocks before any user-added systems[N>=0].
        # Use {WORKSPACE_DIR}, {ATTACHMENTS_DIR}, {SKILLS_DIR} anywhere in these
        # strings -- they are substituted automatically before each run.
        # To disable a block: delete or comment out the corresponding assignment.
        # To enable/disable the attachments policy at runtime: use set_use_attachments().

        self._prompt_verification = """\
[VERIFICATION POLICY]
Before answering any question that involves a library, API, version number, tool behavior, or compatibility claim -- web search or check first. Do not rely on training data for these topics.
After forming your answer, check: does it contain any claim you have not confirmed? If yes -- mark it explicitly as [UNVERIFIED].
If search results contradict your answer -- use the search result, not your assumption."""

        self._prompt_minimal_change = """\
[MINIMAL CHANGE POLICY]
When modifying files, code, text, or any structured content:
1. Use surgical edit tools (str_replace) rather than rewriting entire files. Never rewrite a whole file when only a fragment needs changing.
2. Every character outside the intended change scope must remain byte-for-byte identical: whitespace, indentation, line endings, comments, variable names, imports not involved in the fix.
3. Never create a new file as a substitute for editing an existing one.
4. Do not add unrequested boilerplate, comments, or "improvements" unless explicitly asked.
5. Never embed references to the conversation in file content -- no phrases like "Last change", "Option N selected", "Fixed problem", "As requested", or any other text that refers to the dialogue rather than the subject matter."""

        self._prompt_workspace = """\
[WORKSPACE BOUNDARY POLICY]
WORKSPACE_DIR:   {WORKSPACE_DIR}
ATTACHMENTS_DIR: {ATTACHMENTS_DIR}
SKILLS_DIR:      {SKILLS_DIR}

ABSOLUTE RESTRICTIONS:
1. Never read, write, move, copy, delete, or reference any file whose resolved absolute path is outside WORKSPACE_DIR. This includes paths using ../, symlinks, or environment variables that resolve outside WORKSPACE_DIR.
2. Before using any path: resolve it to absolute form mentally. If it is not inside WORKSPACE_DIR -- refuse and report instead.
PERMITTED EXCEPTIONS (no file access):
- Querying installed packages: pip list, importlib.util.find_spec
- Reading the current Python version or platform
- Calling MCP tools whose path arguments point inside WORKSPACE_DIR"""

        self._prompt_attachments = """\
[ATTACHMENT WORKSPACE POLICY]
ATTACHMENTS_DIR: {ATTACHMENTS_DIR}
The directory ATTACHMENTS_DIR is the exchange buffer for this session: all files you read, create, or modify go here by default.
Rules:
1. When a task involves modifying an attached file, edit it in place in ATTACHMENTS_DIR using a surgical tool (str_replace / patch).
2. Place all created or generated files in ATTACHMENTS_DIR unless a different location is explicitly specified.
3. Never delete files from ATTACHMENTS_DIR unless explicitly instructed by the user."""

        self.systems[-4] = {"prompt": self._prompt_verification}
        self.systems[-3] = {"prompt": self._prompt_minimal_change}
        self.systems[-2] = {"prompt": self._prompt_workspace}
        self.systems[-1] = {"prompt": self._prompt_attachments}

        # -- Cache for prompt variable substitution ------------------------
        # prepare() substitutes {WORKSPACE_DIR} and {SKILLS_DIR} into a shadow
        # copy of self.systems before passing it to super().prepare().
        # The cache is invalidated when workspace_dir, skills_dir, or systems change.
        self._prompt_vars_cache_key: tuple = ()
        self._prompt_vars_resolved:  dict  = {}

        # -- Workspace size cache -----------------------------------------
        # Кэшируем суммарный размер папки чтобы не пересчитывать на каждую запись.
        # Обновляется только если свободное место на диске изменилось > delta или истёк TTL.
        self._ws_size_cache:      int   = 0    # последний измеренный размер папки, байт
        self._ws_free_cache:      int   = -1   # последнее измеренное free disk, байт (-1 = не измерялось)
        self._ws_size_cache_time: float = 0.0  # время последнего пересчёта (time.monotonic)

        # -- Piper TTS -----------------------------------------------------
        # Объект для синтеза речи. Настройки меняются через атрибуты:
        #   ai.piper.model        = "ru_RU-irina-medium"
        #   ai.piper.length_scale = 0.9
        # Инструменты для ЛЛМ: piper_synthesize и piper_set_config --
        # доступны после await ai.mcp_connect("piper").
        _piper_output = (self.workspace_dir or Path.cwd()) / "t2v"
        self.piper = PiperTTS(
            models_dir = self._cache_dir / "piper",
            output_dir = _piper_output,
        )

        # -- Apprise: уведомления ------------------------------------------
        # Настройка каналов из кода:
        #   ai.apprise.urls.append("tgram://bot_token/chat_id")
        #   ai.apprise.channels["work"] = "discord://webhook_id/token"
        # Отправка из кода:
        #   ai.notify("Заголовок", "Текст")               # -> все urls
        #   ai.notify("Заголовок", "Текст", channel="work") # -> конкретный канал
        # Инструмент для ЛЛМ: notify_send -- доступен после mcp_connect("apprise").
        self.apprise = AppriseConfig()

        # -- Реестр MCP-серверов ------------------------------------------
        # Добавить новый сервер = одна запись здесь.
        self.MCPServers = {
            "apprise": MCPServerDef(
                # Отправка уведомлений через Apprise (pip install apprise).
                # Поддерживает 100+ сервисов: Telegram, Discord, Email, Slack, Gotify, ntfy и др.
                # Каналы настраиваются через ai.apprise:
                #   ai.apprise.urls.append("tgram://token/chat_id")   # дефолтные
                #   ai.apprise.channels["work"] = "discord://..."      # именованные
                # Инструмент ЛЛМ: notify_send(title, body, channel="")
                package       = "",
                description   = "Push notifications via Apprise: Telegram, Discord, Email and 100+ services.",
                launcher      = "builtin",
                builtin_tools = [self._make_apprise_tool()],
            ),
            "piper": MCPServerDef(
                # Синтез речи через локальный piper-tts (pip install piper-tts).
                # Модели (.onnx) скачиваются автоматически с HuggingFace (MIT, без регистрации)
                # в папку piper_modules/ рядом с ailist.py при первом вызове.
                # Выходные WAV-файлы сохраняются в t2v/t2v_<timestamp>.wav.
                # Настройки задаются через ai.piper (model, length_scale и др.)
                # Инструменты ЛЛМ: piper_synthesize (генерация), piper_set_config (настройки).
                package       = "",
                description   = "Text-to-speech (TTS) via Piper: text -> WAV file. Models downloaded automatically.",
                launcher      = "builtin",
                builtin_tools = self.piper.as_tools(),
            ),
            "file-converter": MCPServerDef(
                package       = "",
                description   = "File conversion: PDF, Office (docx/xlsx/pptx), HTML, audio (Whisper), video.",
                launcher      = "builtin",
                builtin_tools = self._converter.as_tools(),
            ),
            "workspace": MCPServerDef(
                # Builtin workspace tools: file read/edit, search, directory navigation,
                # run external commands, execute Python code.
                # All paths are restricted to workspace_dir.
                # Tools: view, str_replace, create_file, write_file, insert, undo_edit,
                #        read_files, list_dir, list_dir_tree, find_file, file_info,
                #        make_dir, move_file, grep, search_for_pattern, head, tail, wc,
                #        run, python_run.
                package       = "",
                description   = (
                    "Workspace tools: read/edit files, search, run commands and Python code, "
                    "navigate directories. All paths must be within the workspace."
                ),
                launcher      = "builtin",
                builtin_tools_factory = self._make_workspace_tools,
                system_prompt = (
                    "FILE EDITING WORKFLOW:\n"
                    "1. LOCATE -- use grep(file, pattern) to find the exact line.\n"
                    "2. CONFIRM -- use view(file, start_line, end_line) to see the exact text.\n"
                    "3. EDIT -- choose the right tool:\n"
                    "   str_replace(file, old_str, new_str) -- change part of an existing file.\n"
                    "   write_file(file, content)           -- replace the ENTIRE file content.\n"
                    "   create_file(file, content)          -- create a NEW file (fails if exists).\n"
                    "4. VERIFY -- use view() again to confirm the change is correct.\n"
                    "\n"
                    "SEARCH:\n"
                    "  grep(file, pattern)              -- search inside ONE file, get line numbers.\n"
                    "  search_for_pattern(pattern, dir) -- search ACROSS ALL files in a directory.\n"
                    "\n"
                    "PYTHON:\n"
                    "  python_run(code) -- use instead of multiple tool calls when you need loops,\n"
                    "  data processing, JSON parsing, or counting lines/words across files.\n"
                    "  WORKSPACE variable inside python_run contains the workspace path (str).\n"
                    "  Use print() to return output. Example:\n"
                    "    lines = open(WORKSPACE+'/app.py').readlines()\n"
                    "    print(f'Lines: {len(lines)}')\n"
                    "\n"
                    "RULES:\n"
                    "  - Never read a whole file when grep or view with a range is enough.\n"
                    "  - Never use run() for file operations -- use the file tools above.\n"
                    "  - list_dir() shows one level. list_dir_tree() shows everything recursively."
                ),
            ),
            "skills": MCPServerDef(
                # Builtin skills browser: lists available agent skills from ai.skills_dir.
                # Source: FastSkills (MIT, github: nj19257/FastSkills).
                # Default skills_dir = "skills" (relative to workspace_dir).
                # Override: ai.skills_dir = "my_skills"  or  ai.skills_dir = r"C:\absolute\path"
                # Full resolved path is returned by ai.get_skills_dir().
                # Calls list_skills on connect and adds the result to system_tool_instructions.
                # Usage: await ai.mcp_connect("skills")
                package            = "",
                description        = "Agent skills: list available skills with paths. Set ai.skills_dir before connecting.",
                launcher           = "builtin",
                builtin_tools      = self._make_skills_tools(),
                system_prompt_tool = "list_skills",
            ),
            "git": MCPServerDef(
                package      = "@cyanheads/git-mcp-server",
                description  = "Git operations: status, log, diff, commit, branch, push, pull.",
                exclude_tools = [
                    # Retain only the 8 tools a weak LLM needs daily.
                    # Everything below is excluded: advanced/rare operations that cause
                    # confusion without adding value for typical coding workflows.
                    "git_stash",
                    "git_tag",
                    "git_merge",
                    "git_rebase",
                    "git_fetch",
                    "git_reset",
                    "git_reflog",
                    "git_remote",
                    "git_show",
                    "git_cherry_pick",
                    "git_clean",
                    "git_worktree",
                    "git_changelog_analyze",
                    "git_wrapup_instructions",
                    "git_blame",
                ],
            ),
            "playwright": MCPServerDef(
                # Версия 1.0.12: parseArgs() обрабатывает только --port и --help.
                # --headless и HEADLESS env полностью игнорируются на уровне запуска сервера.
                # Headless-режим управляется исключительно через параметр headless=true
                # в каждом вызове playwright_navigate.
                # system_prompt обеспечивает что модель всегда передаёт headless=true
                # когда пользователь подключил сервер с allowed=["headless"].
                package      = "@executeautomation/playwright-mcp-server",
                description  = "Browser automation via Playwright: navigation, clicks, screenshots, scraping.",
                args_builder = self._build_playwright,
                system_prompt = (
                    "playwright_navigate: ALWAYS set headless=true. "
                    "NEVER open a visible browser window unless the user explicitly requests it. "
                    "Example: playwright_navigate(url='...', headless=true)"
                ),
                exclude_tools = [
                    # Codegen session tools -- for test recording, not needed in agent workflows.
                    "start_codegen_session",
                    "end_codegen_session",
                    "get_codegen_session",
                    "clear_codegen_session",
                ],
                schema_patches = {
                    "playwright_navigate": {
                        "headless": {
                            "description": "Run browser in headless mode (default: true). "
                                           "Always set to true unless the user explicitly asks for a visible browser.",
                        },
                    },
                },
            ),
            "memory-plus": MCPServerDef(
                package      = "@modelcontextprotocol/server-memory",
                description  = "Persistent agent memory: knowledge graph stored in a JSONL file.",
                args_builder = self._build_memory_plus,
            ),
            "qdrant": MCPServerDef(
                # "uv tool run" запускает mcp-server-qdrant как stdio-процесс.
                # Установка uv: pip install uv  (или winget install astral-sh.uv)
                # Сервер предоставляет два инструмента: qdrant-store и qdrant-find.
                # При первом подключении uv скачивает модель sentence-transformers (~90 MB).
                package      = "mcp-server-qdrant",
                description  = "Qdrant vector database: semantic storage and retrieval using embeddings.",
                launcher     = "uvx",
                args_builder = self._build_qdrant,
            ),
            "qdrant-db": MCPServerDef(
                package     = "",     # не используется: запись только для docker_start/docker_stop
                description = "Qdrant database container (required for mcp-server-qdrant).",
                launcher    = None,
                url         = None,
                docker      = {
                    "image":   "qdrant/qdrant",
                    "name":    "qdrant",
                    "ports":   ["6333:6333"],
                    "volumes": ["qdrant_storage:/qdrant/storage"],
                },
            ),
            "searxng": MCPServerDef(
                package      = "",     # sse: npx не используется
                description  = "SearXNG metasearch: privacy-friendly web search.",
                launcher     = "sse",
                url          = "http://localhost:8001/sse",
                args_builder = self._build_searxng,
                system_prompt = (
                    "web_search(query: str, time_range: 'day'|'week'|'month'|'year'|null=null, "
                    "result_format: 'text'|'json'='text'). "
                    "Only query is required. Never pass categories or any other fields."
                ),
                docker       = {
                    "image":   "ghcr.io/sacode/searxng-simple-mcp:latest",
                    "name":    "mcp-server-searxng",
                    "ports":   ["8001:8000"],
                    "wait_http": "http://localhost:8001/sse",
                    "env": {
                        "TRANSPORT_PROTOCOL":      "sse",
                        "FASTMCP_HOST":            "0.0.0.0",
                        "FASTMCP_PORT":            "8000",
                        "SEARXNG_MCP_SEARXNG_URL": "http://host.docker.internal:8080",
                    },
                },
            ),
            "searxng-engine": MCPServerDef(
                package     = "",     # не используется: запись только для docker_start/docker_stop
                description = "SearXNG search engine container (required for mcp-server-searxng).",
                launcher    = None,
                url         = None,
                docker      = {
                    "image":   "searxng/searxng",
                    "name":    "searxng",
                    "ports":   ["8080:8080"],
                    "volumes": ["searxng_config:/etc/searxng"],
                    "setup_wait_for": "test -f /etc/searxng/settings.yml",
                    "setup_check":    "grep -qP '^    - json' /etc/searxng/settings.yml",
                    "setup_commands": [
                        (
                            "python3 -c \"import re, pathlib; "
                            "p = pathlib.Path('/etc/searxng/settings.yml'); "
                            "t = p.read_text(); "
                            "t = re.sub(r'(  formats:\\n)((?:    - \\S+\\n)*)', "
                            "'  formats:\\n    - html\\n    - json\\n', t); "
                            "p.write_text(t)\""
                        ),
                    ],
                },
            ),
            "sympy": MCPServerDef(
                package       = "",   # builtin: npx не используется
                description   = "Symbolic math and equation solving via SymPy (built-in tool).",
                launcher      = "builtin",
                builtin_tools = _BUILTIN_SYMPY,
            ),
            "serena": MCPServerDef(
                # Semantic code analysis and editing via LSP (Language Server Protocol).
                # Unique capabilities not available in workspace server:
                #   - find_symbol / find_referencing_symbols -- LSP-aware search by symbol,
                #     understands scope, imports, overloading (not just text matching).
                #   - replace_symbol_body -- replaces a function/class body using AST boundaries,
                #     no need to know the exact text.
                #   - insert_after/before_symbol -- insert code relative to a symbol.
                #   - rename_symbol -- project-wide rename with scope awareness (true refactoring).
                #   - safe_delete_symbol -- delete only if symbol has no usages.
                #   - get_symbols_overview -- structural overview of a file (classes, methods).
                #   - memory tools -- persistent per-project markdown memory across sessions.
                #   - onboarding -- automatic project structure analysis on first connect.
                #
                # Default context="ide": Serena disables its own file ops and shell,
                # complementing workspace server without tool name conflicts.
                # Switch to context="agent" for standalone Serena without workspace.
                #
                # Supports 30+ languages. Language servers downloaded automatically on first use.
                # No additional pip/npm installs needed for Python (uses Pyright built-in).
                #
                # Parameters for mcp_connect(): see _build_serena docstring.
                #
                # Requires: pip install uv
                # Runs via: uvx -p 3.13 --from git+https://github.com/oraios/serena serena
                package      = "git+https://github.com/oraios/serena",
                description  = (
                    "Serena: LSP-based semantic code analysis. "
                    "Symbol search/rename/replace/delete with scope awareness, "
                    "project-wide refactoring, persistent memory. "
                    "Use context='ide' (default) alongside workspace server."
                ),
                launcher     = "uvx",
                python_version = "3.13",
                uvx_command  = "serena",  # --from git+... serena start-mcp-server ...
                init_timeout = 600.0,   # first run: uv fetches from GitHub + may install Python 3.13
                args_builder = self._build_serena,
                system_prompt_tool = "initial_instructions",
                # Serena exposes its full "Instructions Manual" via the initial_instructions tool.
                # This is the recommended fallback for clients that don't read MCP instructions field.
                # Note: in context="ide" this tool is available; in context="agent" instructions
                # are delivered via InitializeResult.instructions instead (standard MCP path).
            ),
        }

# --------------------------------------------------
# Определяем context_limit для ollama так:
#  cmd /k ollama show gpt-oss:20bgpu
#  Из результата берём минимум из num_ctx и context length (хотя должно быть num_ctx <= context length)
#  num_ctx как правило указан в Modelfile 
#  Можно уменьшить на некоторый коэф для запаса на ответ

# --------------------------------------------------
# ----------------------------------------------------------
# Примеры использования MCP
# ----------------------------------------------------------
#
# -- Рекомендованные конфигурации ---------------------------
#
# A. Файловая работа / скрипты (~20 инструментов):
#       await ai.mcp_connect("workspace")
#
# B. Программирование без LSP (~28 инструментов):
#       await ai.mcp_connects([
#           {"name": "workspace"},
#           {"name": "git"},            # git_status/log/diff/add/commit/push/pull/branch
#       ])
#
# C. Программирование с LSP-рефакторингом (~27 инструментов):
#       await ai.mcp_connects([
#           {"name": "workspace"},
#           {"name": "serena", "dirs": [r"C:\MyProject"]},  # context="ide" по умолчанию
#       ])
#
# D. Полный кодинг (LSP + git, ~35 инструментов):
#       await ai.mcp_connects([
#           {"name": "workspace"},
#           {"name": "serena", "dirs": [r"C:\MyProject"]},
#           {"name": "git"},
#       ])
#
# -- Несовместимые комбинации --------------------------------
#
# НЕЛЬЗЯ: serena(agent) + workspace -- конфликт имён инструментов.
#   serena всегда подключать в context="ide" (дефолт) рядом с workspace.
#   context="agent" -- только если workspace НЕ подключён.
#
# НЕЛЬЗЯ: searxng без docker_ensure("searxng-engine") и docker_ensure("searxng").
# НЕЛЬЗЯ: qdrant без docker_ensure("qdrant-db").
#
# ОСТОРОЖНО: слишком много серверов одновременно -- слабая модель теряется.
#   Рекомендуемый предел: ~35 инструментов суммарно.
#
# -- Базовый паттерн -- async with гарантирует mcp_disconnect_all при выходе:
#   async with AIListDemo() as ai:
#       await ai.mcp_connect("workspace")
#       print(await ai.run_async("Покажи файлы"))
#
# Несколько серверов одновременно (параллельный старт):
#   async with AIListDemo() as ai:
#       await ai.mcp_connects([
#           {"name": "workspace"},
#           {"name": "playwright",  "allowed": ["headless"]},
#           {"name": "memory-plus", "dirs": [r"C:\Work\memory.jsonl"]},
#           {"name": "sympy"},
#           {"name": "qdrant",      "allowed": ["my_collection"]},
#           {"name": "searxng",     "url": "http://localhost:8001/sse"},
#       ])
#       print(ai.mcp_list())         # активные соединения
#       print(ai.mcp_tool_names())   # все инструменты
#
# Со скиллами (задать skills_dir до подключения):
#   async with AIListDemo() as ai:
#       ai.skills_dir = Path(r"C:\Work\skills")
#       await ai.mcp_connects([
#           {"name": "workspace"},
#           {"name": "skills"},
#       ])
#       print(await ai.run_async("Что умеешь?"))
#
# С Serena (LSP-анализ кода) -- context="ide" отключает файловые инструменты Serena,
# workspace берёт на себя файловые операции, Serena даёт LSP и память:
#   async with AIListDemo() as ai:
#       await ai.mcp_connects([
#           {"name": "workspace"},
#           {"name": "serena", "dirs": [r"C:\MyProject"]},  # context="ide" by default
#       ])
#       # Повторный запуск (проект уже проиндексирован):
#       #   {"name": "serena", "dirs": [...],
#       #    "modes": ["no-onboarding", "interactive", "editing"]}
#
# Playwright -- профиль браузера и режим headless:
#   await ai.mcp_connect("playwright", dirs=[r"C:\Work\profile"], allowed=["headless"])
#
# Ручной логин в браузере перед подключением агента:
#   ai.playwright_open(r"C:\Work\profile")
#   input("Залогиньтесь, затем нажмите Enter...")
#   await ai.mcp_connect("playwright", dirs=[r"C:\Work\profile"])
#
# Qdrant -- uvx-сервер + Docker-контейнер для БД:
#   async with AIListDemo() as ai:
#       await ai.docker_ensure("qdrant-db")              # запустить если не запущен
#       await ai.mcp_connect("qdrant", allowed=["my_col"])  # коллекция my_col
#       print(await ai.run_async("Запомни: Python -- мой любимый язык"))
#
# SearXNG -- SSE-сервер в Docker:
#   async with AIListDemo() as ai:
#       await ai.docker_ensure("searxng-engine")
#       await ai.docker_ensure("searxng")
#       await ai.mcp_connect("searxng")
#       print(await ai.run_async("Найди последние новости про Python 3.14"))
#
# Docker-утилиты:
#   ai.docker_status("qdrant-db")   # "running" | "stopped" | "not found"
#   await ai.docker_start("qdrant-db")
#   await ai.docker_stop("qdrant-db")
#   await ai.docker_ensure("searxng")  # создаёт/перезапускает если env изменился
#
# Добавление нового сервера -- см. инструкцию в docstring класса MCPMixin.