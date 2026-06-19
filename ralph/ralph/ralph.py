"""
ralph.py — Ральф-цикл для AIList
=====================================
Запуск:
    python ralph.py                     # интерактивный ввод задачи
    python ralph.py "сделай todo-app"   # задача из аргумента
    python ralph.py --resume            # продолжить прерванный сеанс (prd.json уже есть)

Файлы в рабочей папке проекта (--workdir, по умолчанию ./ralph_project/):
    PRD.md      — подробное описание проекта
    prd.json    — список user-stories (задач для цикла)
    progress.txt — лог итераций (дописывается, не затирается)

Адаптировано под слабую локальную модель:
    — PRD.md и prd.json генерируются короткими отдельными запросами
    — Каждая итерация цикла выдаёт одну user-story
    — В каждый промпт добавляется только необходимый контекст
    — Лимит итераций настраивается через --max-iter (по умолчанию 50)
"""

import argparse
from dataclasses import dataclass
import os
import asyncio
import json
import re
import sys
import datetime
from datetime import datetime as _dt
from pathlib import Path

from langchain_core.tools import StructuredTool

# ─── Подключение AIList ──────────────────────────────────────────────────────────
# Убедитесь что пакет ailist доступен в PYTHONPATH.
# Если запускаете из папки рядом с пакетом — достаточно sys.path ниже.
#sys.path.insert(0, str(Path(__file__).resolve().parent))
from ailist import AIList, Provider  


# ══════════════════════════════════════════════════════════════════════════════
# Константы
# ══════════════════════════════════════════════════════════════════════════════

# Интервал запуска QA-аудитора (каждые N итераций и после последней)
QA_INTERVAL: int = 4

# Ветка git для работы (генерируется из названия проекта)
# Формат: ralph/<kebab-from-project-name>


# ══════════════════════════════════════════════════════════════════════════════
# Логгер в файл
# ══════════════════════════════════════════════════════════════════════════════

class FileLogger:
    """
    Пишет в файл ralph.log всё, что выводится в консоль (через .banner/.log),
    плюс подробный лог AIList (ai.log при loglevel=2) и рассуждения модели
    (ai.last_thinking) — только для итераций цикла.

    Консольный вывод не затрагивается.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.write_text(
            f"# Ralph log\nНачало: {_dt.now().isoformat()}\n\n",
            encoding="utf-8",
        )

    def write(self, text: str) -> None:
        try:
            with self._path.open("a", encoding="utf-8", errors="replace") as f:
                f.write(text)
        except Exception as e:
            # Последний рубеж: лог писать некуда, но не падаем
            print(f"[FileLogger] write error: {e}", flush=True)

    def banner(self, text: str) -> None:
        line = "═" * 60
        self.write(f"\n{line}\n  {text}\n{line}\n")

    def log(self, text: str) -> None:
        ts = _dt.now().strftime("%H:%M:%S")
        self.write(f"[{ts}] {text}\n")

    def ai_log(self, ai) -> None:
        """Дампит ai.log (накопленный AIList-лог уровня 2) и сбрасывает его."""
        content = ai.log
        ai.log = ""   # сбрасываем ДО записи — чтобы не потерять сброс при ошибке записи
        if content:
            self.write("\n--- AIList log ---\n")
            self.write(content.lstrip("\n"))
            self.write("\n--- end AIList log ---\n")

    def ai_thinking(self, ai) -> None:
        """Записывает ai.last_thinking в файл лога, если есть.
        При log_thinking=True в AIListBase last_thinking содержит рассуждения
        из ВСЕХ AI-шагов раунда, объединённые через разделитель.
        """
        if ai.last_thinking:
            self.write("\n--- Thinking ---\n")
            self.write(ai.last_thinking)
            self.write("\n--- end Thinking ---\n")


# ══════════════════════════════════════════════════════════════════════════════
# Git-утилиты
# ══════════════════════════════════════════════════════════════════════════════

import subprocess as _sp

def _git(workdir: Path, *args) -> tuple[int, str]:
    """Запускает git-команду в workdir. Возвращает (returncode, stdout+stderr)."""
    result = _sp.run(
        ["git", "-C", str(workdir)] + list(args),
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def _to_kebab(s: str) -> str:
    """Преобразует строку в kebab-case для имени ветки."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9а-яёa-z\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:40] or "project"


def git_setup(workdir: Path, branch_name: str, flog: "FileLogger") -> None:
    """
    Инициализирует git-репозиторий в workdir если его нет,
    создаёт .gitignore, делает начальный коммит и переключается на ветку.
    """
    # Инициализируем если нет .git
    if not (workdir / ".git").exists():
        _git(workdir, "init")
        _git(workdir, "config", "user.email", "ralph@localhost")
        _git(workdir, "config", "user.name", "Ralph Agent")
        flog.log("git init выполнен")

    # .gitignore
    gitignore = workdir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "ralph.log\n*.tmp\n*.zip\n*.tar\n*.tar.gz\n*.gz\n",
            encoding="utf-8",
        )
        flog.log(".gitignore создан")

    # Начальный коммит если нет ни одного
    rc, _ = _git(workdir, "rev-parse", "HEAD")
    if rc != 0:
        _git(workdir, "add", "-A")
        _git(workdir, "commit", "-m", "chore: initial ralph setup")
        flog.log("Начальный коммит создан")

    # Переключаемся на ветку (создаём если нет)
    rc, _ = _git(workdir, "rev-parse", "--verify", branch_name)
    if rc == 0:
        _git(workdir, "checkout", branch_name)
    else:
        _git(workdir, "checkout", "-b", branch_name)
    flog.log(f"Ветка: {branch_name}")


def git_commit(workdir: Path, story_id: str, story_title: str, flog: "FileLogger") -> None:
    """Коммитит все изменения после успешной итерации."""
    _git(workdir, "add", "-A")
    msg = f"feat: [{story_id}] {story_title}"
    rc, out = _git(workdir, "commit", "-m", msg)
    if rc == 0:
        flog.log(f"git commit: {msg}")
    else:
        flog.log(f"git commit skipped (nothing to commit): {out[:100]}")


def git_rollback(workdir: Path, flog: "FileLogger") -> None:
    """Откатывает все незакоммиченные изменения к HEAD.
    -fdx: удаляет неотслеживаемые файлы включая игнорируемые (*.zip, *.tmp и т.п.)
    """
    _git(workdir, "reset", "--hard", "HEAD")
    _git(workdir, "clean", "-fdx")
    flog.log("git rollback: откат к HEAD выполнен")


def git_current_hash(workdir: Path) -> str:
    """Возвращает хэш текущего коммита (сокращённый)."""
    _, out = _git(workdir, "rev-parse", "--short", "HEAD")
    return out.strip() or "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE.md — накопленные паттерны проекта
# ══════════════════════════════════════════════════════════════════════════════

CLAUDE_MD_TEMPLATE = """\
# Паттерны и память проекта
<!-- Этот файл ведётся автоматически. Не редактируй вручную разделы с маркерами. -->

## Паттерны и соглашения
<!-- PATTERNS_START -->

<!-- PATTERNS_END -->

## Известные проблемы и антипаттерны
<!-- ISSUES_START -->

<!-- ISSUES_END -->
"""


def claude_md_init(workdir: Path) -> None:
    """Создаёт CLAUDE.md с пустыми разделами если файла нет."""
    path = workdir / "CLAUDE.md"
    if not path.exists():
        path.write_text(CLAUDE_MD_TEMPLATE, encoding="utf-8")


def _claude_md_append_section(workdir: Path, text: str, start_marker: str, end_marker: str) -> str:
    """Добавляет text в секцию между маркерами. Возвращает статус."""
    path = workdir / "CLAUDE.md"
    if not path.exists():
        claude_md_init(workdir)
    content = path.read_text(encoding="utf-8")
    if start_marker not in content or end_marker not in content:
        return "ERROR: маркеры не найдены в CLAUDE.md"
    entry = f"- {text.strip()}"
    # Вставляем перед закрывающим маркером
    content = content.replace(end_marker, f"{entry}\n{end_marker}", 1)
    path.write_text(content, encoding="utf-8")
    return "OK"


# Инструменты для агентов — две отдельные функции с одним аргументом

def make_claude_tools(workdir: Path) -> list:
    """Создаёт инструменты для записи в CLAUDE.md."""

    def add_pattern(text: str) -> str:
        """
        Добавляет паттерн или соглашение проекта в CLAUDE.md.
        Используй только для реально переиспользуемых наблюдений:
        как устроены импорты, формат конфигов, соглашения по именованию.
        НЕ пиши story-специфичные детали.

        text -- одна строка описания паттерна
        """
        return _claude_md_append_section(
            workdir, text, "<!-- PATTERNS_START -->", "<!-- PATTERNS_END -->"
        )

    def add_known_issue(text: str) -> str:
        """
        Добавляет известную проблему или антипаттерн в CLAUDE.md.
        Используй чтобы предупредить следующие итерации о том, что НЕ работает.
        Формат: 'Не делай X потому что Y. Вместо этого делай Z.'

        text -- одна строка описания проблемы и рекомендации
        """
        return _claude_md_append_section(
            workdir, text, "<!-- ISSUES_START -->", "<!-- ISSUES_END -->"
        )

    return [
        StructuredTool.from_function(add_pattern,     name="add_pattern"),
        StructuredTool.from_function(add_known_issue, name="add_known_issue"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Инструменты Ральф-цикла
# ══════════════════════════════════════════════════════════════════════════════

def make_ralph_tools(prd_path: Path) -> list:
    """
    Создаёт инструменты специфичные для Ральф-цикла.

    prd_path -- абсолютный путь к prd.json. Захватывается в замыкании,
                прогресс-файл progress.txt берётся из того же каталога.

    Использование:
        ralph_tools = make_ralph_tools(workdir / "prd.json")
        async with make_worker(tools=ralph_tools) as ai:
            ...

    Инструменты:
        prd_next()                              -- следующая невыполненная story
        prd_done(story_id, notes="")            -- пометить story выполненной
        prd_status()                            -- сводка прогресса
        progress_append(story_id, done, ...)    -- добавить запись в progress.txt
    """

    def _load() -> dict:
        """Читает prd.json с автоопределением кодировки."""
        for enc in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return json.loads(prd_path.read_text(encoding=enc))
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Не удалось прочитать {prd_path}: неизвестная кодировка")

    def _save(data: dict) -> None:
        """Записывает prd.json атомарно через временный файл."""
        tmp = prd_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        tmp.replace(prd_path)

    def prd_next() -> str:
        """
        Возвращает следующую невыполненную user-story (наименьший priority где passes=false)
        в виде JSON-строки. Поля: id, title, description, acceptanceCriteria, priority.
        Если все выполнены — возвращает {"done": true}.
        """
        data = _load()
        pending = [s for s in data["userStories"] if not s.get("passes", False)]
        if not pending:
            return json.dumps({"done": True}, ensure_ascii=False)
        story = min(pending, key=lambda s: s["priority"])
        return json.dumps({
            "id":                 story["id"],
            "title":              story["title"],
            "description":        story.get("description", ""),
            "acceptanceCriteria": story.get("acceptanceCriteria", []),
            "priority":           story["priority"],
        }, ensure_ascii=False, indent=2)

    def prd_done(story_id: str, notes: str = "") -> str:
        """
        Атомарно выставляет passes=true для story с указанным id.
        Опционально записывает notes (строка с наблюдениями или описанием проблемы).
        Возвращает подтверждение или сообщение об ошибке.
        """
        data = _load()
        for story in data["userStories"]:
            if story["id"] == story_id:
                story["passes"] = True
                if notes:
                    story["notes"] = notes
                _save(data)
                return f"OK: {story_id} помечена как выполненная."
        return f"ERROR: story '{story_id}' не найдена в prd.json."

    def prd_status() -> str:
        """
        Возвращает краткую сводку прогресса: сколько done/total
        и список оставшихся id с приоритетом и названием.
        """
        data = _load()
        stories = data.get("userStories", [])
        total   = len(stories)
        done    = sum(1 for s in stories if s.get("passes", False))
        pending = [
            f'{s["id"]} (priority {s["priority"]}): {s["title"]}'
            for s in stories if not s.get("passes", False)
        ]
        lines = [f"Выполнено: {done}/{total}"]
        if pending:
            lines.append("Осталось:")
            lines.extend(f"  - {p}" for p in pending)
        else:
            lines.append("Все задачи выполнены.")
        return "\n".join(lines)

    def progress_append(story_id: str, done: str, files: str = "", notes: str = "") -> str:
        """
        Дозаписывает запись в progress.txt в стандартном формате с автоматическим timestamp.
        Никогда не перезаписывает файл целиком — только дописывает в конец.

        story_id -- id story (например "US-001")
        done     -- что было сделано (краткое описание)
        files    -- изменённые файлы через запятую (опционально)
        notes    -- наблюдения или замечания (опционально)
        """
        progress_path = prd_path.parent / "progress.txt"
        ts = datetime.datetime.now().isoformat(timespec="seconds")

        # Берём title из prd.json чтобы заголовок записи был полным
        try:
            data  = _load()
            title = next(
                (s["title"] for s in data["userStories"] if s["id"] == story_id),
                story_id,
            )
        except Exception:
            title = story_id

        lines = [
            "---",
            f"[{ts}] {story_id} - {title}",
            f"Что сделано: {done}",
        ]
        if files:
            lines.append(f"Файлы изменены: {files}")
        if notes:
            lines.append(f"Наблюдения: {notes}")
        lines.append("---\n")

        with progress_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return "OK: запись добавлена в progress.txt."

    def project_context(n: int = 5) -> str:
        """
        Возвращает карту проекта: дерево файлов рабочей папки + содержимое
        последних N изменённых файлов (не считая служебных: prd.json, progress.txt,
        ralph.log, PRD.md).

        Используй в начале итерации вместо ручного list_dir_tree + нескольких view.
        Параметр n (по умолчанию 5) — сколько последних файлов показать.
        Каждый файл обрезается до 60 строк — для ориентира, не для полного чтения.

        Возвращает текст с разделами:
          ## Project tree
          ## Recent files (last N modified)
        """
        workdir = prd_path.parent

        # --- дерево ---
        def _tree(root: Path, prefix: str = "") -> list[str]:
            lines = []
            try:
                entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
            except PermissionError:
                return lines
            for i, entry in enumerate(entries):
                connector = "└── " if i == len(entries) - 1 else "├── "
                lines.append(prefix + connector + entry.name)
                if entry.is_dir():
                    extension = "    " if i == len(entries) - 1 else "│   "
                    lines.extend(_tree(entry, prefix + extension))
            return lines

        tree_lines = _tree(workdir)
        tree_str   = "\n".join(tree_lines) if tree_lines else "(пусто)"

        # --- последние N изменённых файлов ---
        SKIP_NAMES = {"prd.json", "progress.txt", "ralph.log", "PRD.md",
                      "prd.json.tmp"}
        SKIP_SUFFIXES = {".log", ".tmp"}
        SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
        MAX_LINES = 60

        all_files = [
            p for p in workdir.rglob("*")
            if p.is_file()
            and p.name not in SKIP_NAMES
            and p.suffix not in SKIP_SUFFIXES
            and not any(part in SKIP_DIRS for part in p.parts)
        ]
        all_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        recent = all_files[:n]

        file_sections = []
        for fp in recent:
            rel = fp.relative_to(workdir)
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                file_sections.append(f"### {rel}\n(не удалось прочитать: {e})")
                continue
            file_lines = text.splitlines()
            truncated  = len(file_lines) > MAX_LINES
            shown      = file_lines[:MAX_LINES]
            body       = "\n".join(shown)
            suffix_note = f"\n... (показано {MAX_LINES}/{len(file_lines)} строк)" if truncated else ""
            file_sections.append(f"### {rel}\n```\n{body}{suffix_note}\n```")

        files_str = "\n\n".join(file_sections) if file_sections else "(файлов нет)"

        return (
            f"## Project tree\n{tree_str}\n\n"
            f"## Recent files (last {n} modified)\n{files_str}"
        )

    # prd_done намеренно не включён в инструменты агентов —
    # статус задачи меняем сами из Python после подтверждения проверяющего шустрика.
    # _prd_done и _load/_save доступны через замыкание в ralph_step().
    return (
        [
            StructuredTool.from_function(prd_next,        name="prd_next"),
            StructuredTool.from_function(prd_status,      name="prd_status"),
            StructuredTool.from_function(progress_append, name="progress_append"),
            StructuredTool.from_function(project_context, name="project_context"),
        ],
        _load,
        _save,
        prd_done,
        prd_next,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Агент Ральфа
# ══════════════════════════════════════════════════════════════════════════════

class RalphAgent(AIList):
    """
    Агент для Ральф-цикла. Наследует AIList.
    Создавать через методы RalphFactory (или её подкласса).
    """

    def __init__(self, model_name: str, context_limit: int, provider,
                 temperature: float, thinking: str | None,
                 tools=None, context_schema=None):
        super().__init__(
            modelName      = model_name,
            context_limit  = context_limit,
            provider       = provider,
            tools          = tools or [],
            context_schema = context_schema,
        )
        self._ralph_temperature = temperature
        self._ralph_thinking    = thinking

    def make_run_config(self) -> dict:
        """Строит config для run_async из параметров профиля."""
        config: dict = {"configurable": {"temperature": self._ralph_temperature}}
        return self.apply_thinking_mode(config, thinking=self._ralph_thinking)


class RalphFactory:
    """
    Фабрика агентов Ральф-цикла. Создаёт трёх агентов с разными профилями.
    Переопределите нужные методы в подклассе, чтобы изменить модель,
    температуру или thinking для конкретной роли — см. ralph_demo.py.
    """

    _MODEL_NAME:    str   = "ollama:gpt-oss:20b"
    _CONTEXT_LIMIT: int   = int(32768 * 0.8)
    _PROVIDER             = Provider.TEXT_REASONING

    def hustler(self, **kwargs) -> RalphAgent:
        """Шустрик: детерминированный, без размышлений. Быстро и предсказуемо."""
        return RalphAgent(self._MODEL_NAME, self._CONTEXT_LIMIT, self._PROVIDER,
                          temperature=0.0, thinking=None, **kwargs)

    def worker(self, **kwargs) -> RalphAgent:
        """Работяга: сбалансированный профиль для основного цикла."""
        return RalphAgent(self._MODEL_NAME, self._CONTEXT_LIMIT, self._PROVIDER,
                          temperature=0.7, thinking="medium", **kwargs)

    def thinker(self, **kwargs) -> RalphAgent:
        """Умник: высокая температура и глубокие размышления. Для сложных задач."""
        return RalphAgent(self._MODEL_NAME, self._CONTEXT_LIMIT, self._PROVIDER,
                          temperature=1.1, thinking="max", **kwargs)



# ══════════════════════════════════════════════════════════════════════════════
# Промпты (все тексты сосредоточены здесь, не в скиллах)
# ══════════════════════════════════════════════════════════════════════════════

# --- Фаза 1: генерация PRD.md ─────────────────────────────────────────────────
PROMPT_PRD_SYSTEM = """\
Ты — технический аналитик. Твоя задача — написать подробный Product Requirements Document (PRD.md).

Пиши ТОЛЬКО документ, без вступлений и пояснений.

Используй Markdown. Структура:
# [Название проекта]
## Цель
## Технический стек (предложи сам, если не указан)
## Функциональные требования
### [Функция 1]
- описание
- детали реализации
### [Функция 2]
...
## Нефункциональные требования
## Ограничения
## Критерии готовности (Definition of Done)

Помни, что основная цель сделать реалистичный результат, который возможно сделать, предсказуемый, без оверинженеринга, как принято в сообществе и именно то что просит пользователь.
"""

PROMPT_PRD_USER = """\
Задача от пользователя:
{user_task}

Напиши полный PRD.md для этой задачи.
"""

# --- Фаза 2: генерация prd.json ───────────────────────────────────────────────
PROMPT_JSON_SYSTEM = """\
Ты — технический менеджер проекта. Конвертируй PRD.md в список user-stories формата JSON.

ПРАВИЛА:
1. Каждая story — это ОДНА небольшая задача, выполнимая за один сеанс агента (один контекстный окон).
   Большие задачи ОБЯЗАТЕЛЬНО разбивай на несколько маленьких.
2. Порядок: от фундамента к надстройке (база данных → API → UI → тесты).
3. В acceptanceCriteria перечисли конкретные проверяемые условия.
4. priority: 1 (высший) … N (низший). Нумеруй последовательно.
5. passes: всегда false при генерации.

Отвечай ТОЛЬКО валидным JSON, без пояснений, без markdown-блоков.
Формат:
{
  "project": "Название",
  "description": "Краткое описание из PRD",
  "userStories": [
    {
      "id": "US-001",
      "title": "Название истории",
      "description": "Как [роль], я хочу [действие], чтобы [результат].",
      "acceptanceCriteria": ["Критерий 1", "Критерий 2"],
      "priority": 1,
      "passes": false,
      "notes": ""
    }
  ]
}
"""

PROMPT_JSON_USER = """\
PRD.md:
{prep_text}

Сгенерируй prd.json.
"""

# --- Фаза 3: один шаг Ральф-цикла ─────────────────────────────────────────────
PROMPT_RALPH_SYSTEM = """\
Ты — автономный программирующий агент. У тебя есть доступ к инструментам файловой системы
и специальным инструментам управления задачами проекта.

ИНСТРУМЕНТЫ УПРАВЛЕНИЯ ЗАДАЧАМИ (используй их, не python_run):
  prd_next()                                   -- получить следующую невыполненную story (JSON)
  prd_done(story_id, notes="")                 -- пометить story выполненной (атомарно)
  prd_status()                                 -- сводка прогресса: done/total и список остатка
  progress_append(story_id, done, files, notes) -- добавить запись в progress.txt

ТВОЙ АЛГОРИТМ ЗА ОДНУ ИТЕРАЦИЮ:
1. Вызови prd_next() — получишь текущую задачу с id, title, description, acceptanceCriteria.
   Если вернулось {"done": true} — все задачи выполнены, выведи <RALPH_COMPLETE> и стоп.
2. Реализуй ТОЛЬКО эту одну story. Не трогай другие.
3. После реализации:
   a. Вызови prd_done(story_id) — выставит passes=true.
   b. Вызови progress_append(story_id, done="...", files="...", notes="...") — добавит запись в progress.txt, с разделами Что сделано, Файлы изменены и Наблюдения.
4. Выведи РОВНО ОДИН сигнал — никогда оба одновременно:
   Если остались незавершённые задачи — выведи: <RALPH_CONTINUE>
   Если все выполнены              — выведи: <RALPH_COMPLETE>

ВАЖНО:
- Работай только с файлами внутри рабочей папки проекта.
- Если что-то не получается — вызови prd_done(story_id, notes="описание проблемы") для записи проблемы в notes в prd.json,
  и напиши <RALPH_STUCK> в конце.
- Не пиши длинных объяснений. Только действия и краткий итог.
"""

PROMPT_RALPH_USER = """\
Рабочая папка проекта: {workdir}
Файлы проекта находятся в этой папке.

Выполни одну итерацию Ральф-цикла.
"""

# ---------------------------------------------------------------------------
# Подфаза A: шустрик выбирает задачу
# ---------------------------------------------------------------------------
PROMPT_SELECT_SYSTEM = """Ты - планировщик задач. Тебе дан список user-stories проекта в формате JSON.
Твоя единственная задача: выбрать ОДНУ story для выполнения на следующей итерации.

Правила выбора:
- Выбирай story с наименьшим значением priority среди тех, где passes=false.
- Если все passes=true - верни: {"done": true}

Отвечай ТОЛЬКО валидным JSON, без пояснений, без markdown-блоков.
Формат ответа: {"story_id": "US-NNN"}
"""

PROMPT_SELECT_USER = """Выбери следующую задачу для выполнения.
"""

# ---------------------------------------------------------------------------
# Подфаза B: работяга выполняет задачу
# ---------------------------------------------------------------------------
PROMPT_WORKER_SYSTEM = """Ты - автономный программирующий агент. У тебя есть доступ к инструментам файловой системы.

ИНСТРУМЕНТЫ:
  progress_append(story_id, done, files, notes) -- добавить запись в progress.txt после выполнения
  add_pattern(text)      -- записать паттерн в CLAUDE.md (только реально важный)
  add_known_issue(text)  -- записать антипаттерн в CLAUDE.md (только если нашёл проблему)

ТВОЯ ЗАДАЧА:
Тебе передана конкретная user-story. Реализуй её. Не трогай другие задачи.

НЕ меняй поле passes в prd.json - это сделает внешняя система после проверки.

ПОСЛЕ РЕАЛИЗАЦИИ:
1. Вызови progress_append(story_id, done="...", files="...", notes="...").
2. В самом конце ответа выведи РОВНО ОДИН из двух тегов:
   <STORY_OK>   - если задача реализована и критерии выполнены
   <STORY_FAIL> - если что-то не получилось. После тега добавь:
   <ISSUE_ADVICE>совет как обойти проблему в одной строке</ISSUE_ADVICE>
   <FAIL_LOG>что пробовали и что не сработало в 2-3 предложениях</FAIL_LOG>

Не пиши длинных объяснений. Только действия и краткий итог.
"""

PROMPT_WORKER_USER = """Рабочая папка проекта: {workdir}

Задача передана в системный промпт.
Реализуй её.
"""

# ---------------------------------------------------------------------------
# Подфаза C: шустрик проверяет выполнение
# ---------------------------------------------------------------------------
PROMPT_CHECK_SYSTEM = """Ты - инспектор качества. У тебя есть доступ к инструментам файловой системы (только чтение).

ТВОЯ ЗАДАЧА:
Проверить, выполнена ли user-story согласно её acceptanceCriteria.
Читай файлы проекта, ищи нужные элементы, проверяй каждый критерий.

ФОРМАТ ОТВЕТА:
- Если ВСЕ критерии выполнены: напиши только <CHECK_PASS>
- Если хотя бы один не выполнен: опиши какой именно критерий не выполнен и почему,
  затем выведи <CHECK_FAIL>

Проверяй реально, читая файлы — не доверяй словам предыдущего агента.
"""

PROMPT_CHECK_USER = """Рабочая папка проекта: {workdir}

Задача для проверки передана в системный промпт.
Проверь каждый критерий по файлам проекта.
"""

# ---------------------------------------------------------------------------
# Подфаза D: умник исправляет после провала
# ---------------------------------------------------------------------------
PROMPT_FIXER_SYSTEM = """Ты - старший программирующий агент. У тебя есть доступ к инструментам файловой системы.

ИНСТРУМЕНТЫ:
  progress_append(story_id, done, files, notes) -- добавить запись в progress.txt
  add_pattern(text)      -- записать паттерн в CLAUDE.md (только реально важный)
  add_known_issue(text)  -- записать антипаттерн в CLAUDE.md (только если нашёл проблему)

ТВОЯ ЗАДАЧА:
До тебя работяга и проверяющий пытались выполнить user-story — история их работы
передана выше. Проверка не прошла. Исправь проблему и доделай задачу.

Задача и PRD переданы в системном промпте.
НЕ меняй поле passes в prd.json.

ПОСЛЕ ИСПРАВЛЕНИЯ:
1. Вызови progress_append(story_id, done="...", files="...", notes="...").
2. В конце выведи РОВНО ОДИН тег:
   <STORY_OK>   - если задача теперь выполнена
   <STORY_FAIL> - если не удалось исправить. После тега добавь:
   <ISSUE_ADVICE>совет как обойти проблему в одной строке</ISSUE_ADVICE>
   <FAIL_LOG>что пробовали и что не сработало в 2-3 предложениях</FAIL_LOG>

Не пиши длинных объяснений. Только действия и итог.
"""

PROMPT_FIXER_USER = """Рабочая папка проекта: {workdir}

Выше в истории — работа работяги и проверяющего.

Результат проверки (что именно не прошло):
{check_verdict}

Исправь задачу.
"""



# ── Аудитор E1: шустрик проверяет файлы ─────────────────────────────────────
PROMPT_AUDIT_TECH_SYSTEM = """\
Ты — технический аудитор. У тебя есть доступ к инструментам файловой системы (только чтение).

ТВОЯ ЗАДАЧА:
Проверить файлы которые упоминаются в последних итерациях progress.txt.
Тебе будет передан фрагмент progress.txt с последними записями.

ЧТО ПРОВЕРЯТЬ:
1. Файлы существуют (используй file_info или view).
2. JSON-файлы парсятся без ошибок (используй python_run: json.loads(open(f).read())).
3. HTML-файлы: теги закрыты, нет очевидных синтаксических проблем.
4. Нет broken references: если в HTML подключён styles.css — он существует.
5. Размер файлов разумный (не 0 байт, не пустые).

ФОРМАТ ОТВЕТА:
Для каждого файла: статус (OK / ПРОБЛЕМА) и краткое описание.
В конце — раздел ## Резюме: общий вывод в 3-5 предложениях что нашёл.
Не пиши длинных объяснений. Только факты.
"""

PROMPT_AUDIT_TECH_USER = """\
Рабочая папка: {workdir}

Последние записи из progress.txt для анализа:
{progress_tail}

Проверь файлы упомянутые в этих записях.
"""

# ── Аудитор E2: умник оценивает проект целиком ───────────────────────────────
PROMPT_AUDIT_SMART_SYSTEM = """\
Ты — старший технический аудитор. У тебя есть доступ к инструментам файловой системы.

ТВОЯ ЗАДАЧА:
Оценить соответствие текущего состояния проекта выполненным задачам.
Технический аудит уже сделан — его резюме передано тебе.
Твоя задача — смысловой уровень.

ЧТО ПРОВЕРЯТЬ:
1. Пройди по stories с passes=true в prd.json. Действительно ли они выполнены?
2. Нет ли регрессий: не сломали ли поздние итерации то, что сделали ранние?
3. Связность проекта: всё ли подключено, нет ли висящих заглушек?
4. Общее качество на данном этапе разработки.

ИНСТРУМЕНТЫ:
  add_known_issue(text) -- записать антипаттерн в CLAUDE.md (только если нашёл реальную проблему)
  prd_status()          -- текущий статус задач

ВОЗМОЖНЫЕ ДЕЙСТВИЯ:
- Исправить мелкие несоответствия самостоятельно (файловые инструменты доступны).
- Пометить story как невыполненную: напиши <REOPEN: US-NNN> чтобы система вернула её в очередь.
- Критическая остановка: если проект невозможно продолжать — напиши:
  <CRITICAL_STOP>
  КРАТКАЯ ПРИЧИНА: одна строка
  ПОДРОБНО: подробное объяснение и рекомендации
  </CRITICAL_STOP>

Не трогай prd.json, progress.txt, CLAUDE.md напрямую — только через инструменты.
Пиши кратко. В конце напиши ## Итог аудита: общий вывод.
"""

PROMPT_AUDIT_SMART_USER = """\
Рабочая папка: {workdir}

Резюме технического аудита:
{tech_summary}

Проведи смысловой аудит проекта.
"""

# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# RalphProfile — датакласс профиля
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RalphProfile:
    name:          str    # идентификатор: "build", "research", ...
    description:   str    # описание — когда применять
    has_prep:      bool   # True = запускать фазу подготовительного документа
    prep_doc_name: str    # имя файла: "PRD.md", "brief.md", ""
    prep_system:   str    # системный промпт фазы подготовки
    prep_user:     str    # user-промпт фазы подготовки, {user_task}
    json_system:   str    # системный промпт декомпозиции
    json_user:     str    # user-промпт декомпозиции: {prep_text} или {user_task}
    cycle_system:  str    # системный промпт рабочего цикла
    mcp_servers:   list   # список имён MCP-серверов
    output_file:   str    # имя итогового файла ("result.md", "" если нет)
    banner_prep:   str    # текст баннера фазы подготовки
    banner_json:   str    # текст баннера фазы декомпозиции
    init_dirs:     list   # папки которые нужно создать перед циклом


# ══════════════════════════════════════════════════════════════════════════════
# Промпты регулировщика (Фаза 1)
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_DISPATCH_SYSTEM = """\
Ты — диспетчер задач. Твоя единственная задача — проанализировать запрос
пользователя и выбрать подходящий профиль выполнения.

ДОСТУПНЫЕ ПРОФИЛИ:

"build" — разработка программного обеспечения. Создание приложений, скриптов,
  утилит, конфигурационных файлов, документации к коду. Результат — набор
  файлов в рабочей папке проекта.
  Примеры: "сделай TODO-приложение", "напиши скрипт для парсинга CSV",
  "создай REST API на FastAPI", "сгенерируй Dockerfile для проекта".

"research" — исследование списка объектов. Задача содержит перечень сущностей
  (книги, компании, URL, статьи, люди, продукты) которые нужно изучить,
  сравнить или оценить по каким-либо критериям. Результат — отчёт.
  Примеры: "изучи эти 20 книг и скажи о чём каждая", "сравни 10 конкурентов",
  "найди информацию о каждой из этих компаний", "проанализируй эти статьи".

"data" — обработка данных. Задача предполагает трансформацию, нормализацию,
  фильтрацию или агрегацию структурированных данных. Входные данные уже есть
  (файл, текст, список). Результат — преобразованные данные в новом формате.
  Примеры: "конвертируй этот CSV в JSON", "нормализуй эти адреса",
  "объедини эти два файла по полю id", "вычисли статистику по этим данным".

"content" — генерация текстового контента. Создание серии текстовых материалов
  по заданной теме, формату и стилю. Результат — набор текстовых файлов.
  Примеры: "напиши 10 описаний товаров", "создай серию постов для блога",
  "сгенерируй шаблоны email для разных сценариев", "напиши FAQ для продукта".

"simple" — простая однородная задача не требующая предварительного планирования.
  Задача ясна из описания и не предполагает сложной декомпозиции.
  Примеры: "переименуй все файлы в папке по такому правилу",
  "создай файл с такой структурой", "сделай резервную копию этих файлов".

ПРАВИЛО ВЫБОРА:
- Если задача явно про написание кода или создание программного продукта — "build".
- Если есть явный список объектов для изучения/исследования — "research".
- Если есть входные данные которые нужно преобразовать — "data".
- Если нужно создать несколько похожих текстовых материалов — "content".
- Если задача простая и однозначная — "simple".
- При сомнении между "build" и другим профилем — выбирай "build".

Отвечай ТОЛЬКО валидным JSON без пояснений и без markdown-блоков:
{"profile": "название профиля", "reason": "одно предложение почему выбран этот профиль"}
"""

PROMPT_DISPATCH_USER = """\
Задача пользователя:
{user_task}

Выбери профиль.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Промпты профиля build
# ══════════════════════════════════════════════════════════════════════════════
# Используем уже существующие PROMPT_PRD_SYSTEM, PROMPT_PRD_USER,
# PROMPT_JSON_SYSTEM, PROMPT_JSON_USER, PROMPT_RALPH_SYSTEM

# ══════════════════════════════════════════════════════════════════════════════
# Промпты профиля research
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_RESEARCH_BRIEF_SYSTEM = """\
Ты — аналитик. Твоя задача — написать краткий Research Brief на основе
запроса пользователя.

Пиши ТОЛЬКО документ, без вступлений и пояснений. Используй Markdown.

Структура:
# [Название исследования]
## Цель
Одно предложение — что именно нужно узнать или выяснить.
## Объекты исследования
Полный список объектов из запроса пользователя. Каждый объект — отдельная
строка. Не добавляй объекты которых нет в запросе, не пропускай ни одного.
## Критерии оценки
Что именно нужно выяснить о каждом объекте. Конкретные вопросы или параметры.
## Формат результата
Описание итогового файла result.md: структура каждой записи, поля, формат.
## Источники
Рекомендуемые источники информации (общие рекомендации, не конкретные URL).
"""

PROMPT_RESEARCH_BRIEF_USER = """\
Запрос пользователя:
{user_task}

Напиши Research Brief.
"""

PROMPT_RESEARCH_JSON_SYSTEM = """\
Ты — планировщик задач. Преобразуй Research Brief в список задач формата JSON.

ПРАВИЛА:
1. Каждая story — исследование ОДНОГО объекта из раздела "Объекты исследования".
   Один объект — одна story. Не объединяй несколько объектов в одну story.
2. Последняя story (с наибольшим priority) всегда: "Собрать итоговый отчёт".
   Она агрегирует результаты всех предыдущих story в result.md.
3. В description каждой story укажи полное название объекта как в Brief.
4. В acceptanceCriteria — список строк (массив): конкретный файл с результатом
   (research/item_001.md) и что в нём должно быть.
5. passes: всегда false при генерации.

Отвечай ТОЛЬКО валидным JSON без пояснений и без markdown-блоков.
Формат:
{
  "project": "Название исследования",
  "description": "Краткое описание из Brief",
  "userStories": [
    {
      "id": "US-001",
      "title": "Название объекта исследования",
      "description": "Полное название объекта и что нужно выяснить.",
      "acceptanceCriteria": ["Файл research/item_001.md создан", "Содержит критерий 1", "Содержит критерий 2"],
      "priority": 1,
      "passes": false,
      "notes": ""
    }
  ]
}
"""

PROMPT_RESEARCH_JSON_USER = """\
Research Brief:
{prep_text}

Сгенерируй prd.json.
"""

PROMPT_RESEARCH_SYSTEM = """\
Ты — автономный агент-исследователь. У тебя есть доступ к инструментам
файловой системы, поиска в интернете и управления задачами проекта.

ИНСТРУМЕНТЫ УПРАВЛЕНИЯ ЗАДАЧАМИ (используй их, не python_run):
  project_context(n=5)                          -- структура проекта и последние файлы
  prd_next()                                    -- получить следующую невыполненную story (JSON)
  prd_done(story_id, notes="")                  -- пометить story выполненной (атомарно)
  prd_status()                                  -- сводка прогресса
  progress_append(story_id, done, files, notes) -- добавить запись в progress.txt

ТВОЙ АЛГОРИТМ ЗА ОДНУ ИТЕРАЦИЮ:
1. Вызови prd_next(). Если {"done": true} — выведи <RALPH_COMPLETE> и стоп.
2. Вызови project_context() чтобы понять что уже собрано.
3. Если это story с исследованием объекта:
   a. Используй инструменты поиска чтобы найти информацию об объекте.
   b. Оцени объект по критериям из acceptanceCriteria.
   c. Сохрани результат в отдельный файл в папке research/ (путь в acceptanceCriteria).
   d. Формат файла: краткий Markdown с заголовком объекта и ответами на все критерии.
4. Если это финальная story (агрегация):
   a. Прочитай все файлы из папки research/.
   b. Собери итоговый отчёт в {output_file} согласно формату из Research Brief.
5. После выполнения:
   a. Вызови prd_done(story_id).
   b. Вызови progress_append(story_id, done="...", files="...", notes="...").
6. Выведи РОВНО ОДИН сигнал:
   Если остались незавершённые задачи — выведи: <RALPH_CONTINUE>
   Если все выполнены              — выведи: <RALPH_COMPLETE>

ВАЖНО:
- Работай только с файлами внутри рабочей папки проекта.
- Если не удалось найти информацию — запиши "Информация не найдена" в файл
  результата и вызови prd_done с notes об этом.
- Не пиши длинных объяснений. Только действия и краткий итог.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Промпты профиля data
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_DATA_SPEC_SYSTEM = """\
Ты — дата-инженер. Твоя задача — написать краткую Data Spec на основе
запроса пользователя.

Пиши ТОЛЬКО документ, без вступлений и пояснений. Используй Markdown.

Структура:
# [Название задачи обработки данных]
## Цель
Одно предложение — что именно нужно сделать с данными.
## Входные данные
Описание источника: имя файла (если указан), формат, структура, объём (если известен).
## Выходные данные
Формат выходного файла, структура, имя файла output (предложи если не указано).
## Правила трансформации
Конкретные правила преобразования.
## Валидация
Как проверить что результат корректен.
## Граничные случаи
Что делать с пустыми значениями, дубликатами, некорректными данными.
"""

PROMPT_DATA_SPEC_USER = """\
Запрос пользователя:
{user_task}

Напиши Data Spec.
"""

PROMPT_DATA_JSON_SYSTEM = """\
Ты — планировщик задач. Преобразуй Data Spec в список задач формата JSON.

ПРАВИЛА:
1. Декомпозируй трансформацию на атомарные шаги: чтение, очистка, нормализация,
   агрегация, запись, валидация.
2. Порядок: чтение → очистка/нормализация → трансформация → агрегация → запись → валидация.
3. Последняя story — финальная валидация результата против Spec.
4. В acceptanceCriteria — список строк (массив) с конкретными проверками для каждого шага.
5. passes: всегда false при генерации.

Отвечай ТОЛЬКО валидным JSON без пояснений и без markdown-блоков.
Формат:
{
  "project": "Название задачи обработки данных",
  "description": "Краткое описание из Spec",
  "userStories": [
    {
      "id": "US-001",
      "title": "Название шага",
      "description": "Что именно нужно сделать на этом шаге.",
      "acceptanceCriteria": ["Конкретная проверка 1", "Конкретная проверка 2"],
      "priority": 1,
      "passes": false,
      "notes": ""
    }
  ]
}
"""

PROMPT_DATA_JSON_USER = """\
Data Spec:
{prep_text}

Сгенерируй prd.json.
"""

PROMPT_DATA_SYSTEM = """\
Ты — автономный агент обработки данных. У тебя есть доступ к инструментам
файловой системы и управления задачами проекта.

ИНСТРУМЕНТЫ УПРАВЛЕНИЯ ЗАДАЧАМИ (используй их, не python_run):
  project_context(n=5)                          -- структура проекта и последние файлы
  prd_next()                                    -- получить следующую невыполненную story (JSON)
  prd_done(story_id, notes="")                  -- пометить story выполненной (атомарно)
  prd_status()                                  -- сводка прогресса
  progress_append(story_id, done, files, notes) -- добавить запись в progress.txt

ТВОЙ АЛГОРИТМ ЗА ОДНУ ИТЕРАЦИЮ:
1. Вызови prd_next(). Если {"done": true} — выведи <RALPH_COMPLETE> и стоп.
2. Вызови project_context() чтобы понять текущее состояние данных.
3. Реализуй шаг трансформации через python_run:
   - Читай файлы с encoding='utf-8'. При ошибке пробуй 'cp1251'.
   - Пиши промежуточные результаты в отдельные файлы (не перезаписывай входные).
   - Выводи через print() статистику: количество записей, ошибок, первые 3 строки.
4. Итоговый результат пиши в {output_file}.
5. После выполнения:
   a. Вызови prd_done(story_id).
   b. Вызови progress_append(story_id, done="...", files="...", notes="...").
6. Выведи РОВНО ОДИН сигнал:
   Если остались незавершённые задачи — выведи: <RALPH_CONTINUE>
   Если все выполнены              — выведи: <RALPH_COMPLETE>

ВАЖНО:
- Не удаляй входные файлы — только читай их.
- Не пиши длинных объяснений. Только действия, статистика, краткий итог.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Промпты профиля content
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_CONTENT_PLAN_SYSTEM = """\
Ты — контент-стратег. Твоя задача — написать краткий Content Plan на основе
запроса пользователя.

Пиши ТОЛЬКО документ, без вступлений и пояснений. Используй Markdown.

Структура:
# [Название контент-проекта]
## Цель и аудитория
Для кого создаётся контент и зачем.
## Единицы контента
Полный список материалов для создания. Каждая единица — отдельная строка
с порядковым номером и кратким описанием темы/назначения.
## Требования к формату
Тип файла, структура каждого материала, объём, язык.
## Тон и стиль
## Ключевые слова или темы
## Пример
Краткий пример одной единицы или её структуры.
"""

PROMPT_CONTENT_PLAN_USER = """\
Запрос пользователя:
{user_task}

Напиши Content Plan.
"""

PROMPT_CONTENT_JSON_SYSTEM = """\
Ты — планировщик задач. Преобразуй Content Plan в список задач формата JSON.

ПРАВИЛА:
1. Каждая story — создание ОДНОЙ единицы контента. Одна единица — одна story.
2. Последняя story: "Проверить и скомпилировать" — финальный просмотр и index.md.
3. В title — конкретная тема единицы, не просто "Создать текст №3".
4. В acceptanceCriteria — список строк (массив): путь к файлу (content/item_001.md),
   объём, ключевые разделы которые должны присутствовать.
5. В description для каждой story продублируй ключевые требования из Content Plan.
6. passes: всегда false при генерации.

Отвечай ТОЛЬКО валидным JSON без пояснений и без markdown-блоков.
Формат:
{
  "project": "Название контент-проекта",
  "description": "Краткое описание из Content Plan",
  "userStories": [
    {
      "id": "US-001",
      "title": "Название единицы контента",
      "description": "Тема, ключевые требования к стилю и содержанию.",
      "acceptanceCriteria": ["Файл content/item_001.md создан", "Объём 300-500 слов", "Присутствуют разделы: Заголовок, Краткое описание, Ключевые персонажи/конфликт"],
      "priority": 1,
      "passes": false,
      "notes": ""
    }
  ]
}
"""

PROMPT_CONTENT_JSON_USER = """\
Content Plan:
{prep_text}

Сгенерируй prd.json.
"""

PROMPT_CONTENT_SYSTEM = """\
Ты — автономный контент-агент. У тебя есть доступ к инструментам
файловой системы и управления задачами проекта.

ИНСТРУМЕНТЫ УПРАВЛЕНИЯ ЗАДАЧАМИ (используй их, не python_run):
  project_context(n=3)                          -- структура проекта и последние файлы
  prd_next()                                    -- получить следующую невыполненную story (JSON)
  prd_done(story_id, notes="")                  -- пометить story выполненной (атомарно)
  prd_status()                                  -- сводка прогресса
  progress_append(story_id, done, files, notes) -- добавить запись в progress.txt

ТВОЙ АЛГОРИТМ ЗА ОДНУ ИТЕРАЦИЮ:
1. Вызови prd_next(). Если {"done": true} — выведи <RALPH_COMPLETE> и стоп.
2. Вызови project_context(n=2) чтобы увидеть уже созданные материалы.
   Используй их для поддержания единого стиля и тона.
3. Создай единицу контента согласно description и acceptanceCriteria story.
   Соблюдай формат, структуру и объём. Создавай папку content/ если нет (make_dir).
4. Если это финальная story (компиляция):
   - Прочитай все созданные файлы из папки content/.
   - Создай {output_file} со списком всех материалов и кратким содержанием каждого.
5. После выполнения:
   a. Вызови prd_done(story_id).
   b. Вызови progress_append(story_id, done="...", files="...", notes="...").
6. Выведи РОВНО ОДИН сигнал:
   Если остались незавершённые задачи — выведи: <RALPH_CONTINUE>
   Если все выполнены              — выведи: <RALPH_COMPLETE>

ВАЖНО:
- Работай только с файлами внутри рабочей папки проекта.
- Не пиши вступлений типа "Конечно, я создам...". Только сам контент в файл.
- Не пиши длинных объяснений. Только краткий итог что создано.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Промпты профиля simple
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_SIMPLE_JSON_SYSTEM = """\
Ты — планировщик задач. Разбей задачу пользователя на минимальный список
атомарных шагов формата JSON.

ПРАВИЛА:
1. Каждый шаг — одно конкретное действие с конкретным результатом.
2. Шагов столько сколько реально нужно. Если задача за один шаг — одна story.
3. В acceptanceCriteria — список строк (массив) с конкретными проверяемыми результатами.
4. Порядок: от первого действия к последнему.
5. passes: всегда false при генерации.

Отвечай ТОЛЬКО валидным JSON без пояснений и без markdown-блоков.
Формат:
{
  "project": "Название задачи",
  "description": "Краткое описание",
  "userStories": [
    {
      "id": "US-001",
      "title": "Название шага",
      "description": "Что именно нужно сделать.",
      "acceptanceCriteria": ["Конкретный проверяемый результат"],
      "priority": 1,
      "passes": false,
      "notes": ""
    }
  ]
}
"""

PROMPT_SIMPLE_JSON_USER = """\
Задача пользователя:
{user_task}

Сгенерируй prd.json.
"""

PROMPT_SIMPLE_SYSTEM = """\
Ты — автономный агент. У тебя есть доступ к инструментам файловой системы
и управления задачами проекта.

ИНСТРУМЕНТЫ УПРАВЛЕНИЯ ЗАДАЧАМИ (используй их, не python_run):
  project_context(n=5)                          -- структура проекта и последние файлы
  prd_next()                                    -- получить следующую невыполненную story (JSON)
  prd_done(story_id, notes="")                  -- пометить story выполненной (атомарно)
  prd_status()                                  -- сводка прогресса
  progress_append(story_id, done, files, notes) -- добавить запись в progress.txt

ТВОЙ АЛГОРИТМ ЗА ОДНУ ИТЕРАЦИЮ:
1. Вызови prd_next(). Если {"done": true} — выведи <RALPH_COMPLETE> и стоп.
2. Вызови project_context() чтобы понять текущее состояние.
3. Выполни задачу. Используй подходящие инструменты.
4. После выполнения:
   a. Вызови prd_done(story_id).
   b. Вызови progress_append(story_id, done="...", files="...", notes="...").
5. Выведи РОВНО ОДИН сигнал:
   Если остались незавершённые задачи — выведи: <RALPH_CONTINUE>
   Если все выполнены              — выведи: <RALPH_COMPLETE>

ВАЖНО:
- Работай только с файлами внутри рабочей папки проекта.
- Если что-то не получается — вызови prd_done(story_id, notes="описание проблемы")
  и напиши <RALPH_STUCK> в конце.
- Не пиши длинных объяснений. Только действия и краткий итог.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Реестр профилей
# ══════════════════════════════════════════════════════════════════════════════

PROFILES: dict = {
    "build": RalphProfile(
        name="build",
        description="Разработка программного обеспечения",
        has_prep=True,
        prep_doc_name="PRD.md",
        prep_system=PROMPT_PRD_SYSTEM,
        prep_user=PROMPT_PRD_USER,
        json_system=PROMPT_JSON_SYSTEM,
        json_user=PROMPT_JSON_USER,
        cycle_system=PROMPT_RALPH_SYSTEM,
        mcp_servers=["workspace"],
        output_file="",
        banner_prep="Фаза 2: Генерация PRD.md",
        banner_json="Фаза 3: Генерация prd.json",
        init_dirs=[],
    ),
    "research": RalphProfile(
        name="research",
        description="Исследование списка объектов",
        has_prep=True,
        prep_doc_name="brief.md",
        prep_system=PROMPT_RESEARCH_BRIEF_SYSTEM,
        prep_user=PROMPT_RESEARCH_BRIEF_USER,
        json_system=PROMPT_RESEARCH_JSON_SYSTEM,
        json_user=PROMPT_RESEARCH_JSON_USER,
        cycle_system=PROMPT_RESEARCH_SYSTEM,
        mcp_servers=["workspace", "searxng"],
        output_file="result.md",
        banner_prep="Фаза 2: Генерация Research Brief",
        banner_json="Фаза 3: Декомпозиция по объектам",
        init_dirs=["research"],
    ),
    "data": RalphProfile(
        name="data",
        description="Обработка и трансформация данных",
        has_prep=True,
        prep_doc_name="data_spec.md",
        prep_system=PROMPT_DATA_SPEC_SYSTEM,
        prep_user=PROMPT_DATA_SPEC_USER,
        json_system=PROMPT_DATA_JSON_SYSTEM,
        json_user=PROMPT_DATA_JSON_USER,
        cycle_system=PROMPT_DATA_SYSTEM,
        mcp_servers=["workspace"],
        output_file="output.csv",
        banner_prep="Фаза 2: Генерация Data Spec",
        banner_json="Фаза 3: Декомпозиция трансформации",
        init_dirs=[],
    ),
    "content": RalphProfile(
        name="content",
        description="Генерация текстового контента",
        has_prep=True,
        prep_doc_name="content_plan.md",
        prep_system=PROMPT_CONTENT_PLAN_SYSTEM,
        prep_user=PROMPT_CONTENT_PLAN_USER,
        json_system=PROMPT_CONTENT_JSON_SYSTEM,
        json_user=PROMPT_CONTENT_JSON_USER,
        cycle_system=PROMPT_CONTENT_SYSTEM,
        mcp_servers=["workspace"],
        output_file="index.md",
        banner_prep="Фаза 2: Генерация Content Plan",
        banner_json="Фаза 3: Декомпозиция контента",
        init_dirs=["content"],
    ),
    "simple": RalphProfile(
        name="simple",
        description="Простая однородная задача",
        has_prep=False,
        prep_doc_name="",
        prep_system="",
        prep_user="",
        json_system=PROMPT_SIMPLE_JSON_SYSTEM,
        json_user=PROMPT_SIMPLE_JSON_USER,
        cycle_system=PROMPT_SIMPLE_SYSTEM,
        mcp_servers=["workspace"],
        output_file="",
        banner_prep="",
        banner_json="Фаза 2: Декомпозиция задачи",
        init_dirs=[],
    ),
}

# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════════════

def read_json(path: Path) -> dict:
    """
    Читает JSON-файл с автоопределением кодировки.
    Пробует UTF-8, затем UTF-8-BOM, затем cp1251 (Windows-1251).
    Бросает ValueError если файл не является валидным JSON.
    """
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            text = path.read_text(encoding=enc)
            return json.loads(text)
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as e:
            raise ValueError(f"Невалидный JSON в {path}: {e}") from e
    raise ValueError(f"Не удалось прочитать {path}: неизвестная кодировка")


def banner(text: str) -> None:
    line = "═" * 60
    print(f"\n{line}\n  {text}\n{line}")


def log(text: str) -> None:
    ts = _dt.now().strftime("%H:%M:%S")
    print(f"[{ts}] {text}")


def all_done(prd: dict) -> bool:
    return all(s.get("passes", False) for s in prd.get("userStories", []))


def count_remaining(prd: dict) -> int:
    return sum(1 for s in prd.get("userStories", []) if not s.get("passes", False))


def extract_signal(text: str) -> str:
    """
    Извлечь сигнал из ответа агента.

    Если в ответе присутствуют оба тега одновременно (<RALPH_COMPLETE> и
    <RALPH_CONTINUE>) — возвращает CONFLICT. Caller должен доверять
    remaining_after вместо сигнала.
    """
    upper    = text.upper()
    has_done = "<RALPH_COMPLETE>" in upper
    has_cont = "<RALPH_CONTINUE>" in upper
    has_stuck = "<RALPH_STUCK>" in upper

    if has_done and has_cont:
        return "CONFLICT"
    if has_done:
        return "COMPLETE"
    if has_stuck:
        return "STUCK"
    if has_cont:
        return "CONTINUE"
    return "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
# Основная логика
# ══════════════════════════════════════════════════════════════════════════════


async def generate_prep_doc(
    ai: RalphAgent,
    user_task: str,
    workdir: Path,
    flog: "FileLogger",
    profile: "RalphProfile",
) -> str:
    """Фаза подготовительного документа (PRD.md, brief.md, data_spec.md и т.д.)."""
    banner(profile.banner_prep)
    flog.banner(profile.banner_prep)

    ai.systems[1] = {"prompt": profile.prep_system}
    ai.chat_history.clear()

    prep_text = await ai.run_async(profile.prep_user.format(user_task=user_task))

    msg = f"{profile.prep_doc_name} сгенерирован ({len(prep_text)} символов)"
    log(msg); flog.log(msg)

    doc_path = workdir / profile.prep_doc_name
    doc_path.write_text(prep_text, encoding="utf-8")
    msg = f"Сохранён: {doc_path}"
    log(msg); flog.log(msg)

    return prep_text


async def generate_prd_json(
    ai: RalphAgent,
    source_text: str,
    workdir: Path,
    flog: "FileLogger",
    profile: "RalphProfile",
    user_task: str = "",
) -> dict:
    """
    Фаза декомпозиции: генерирует prd.json.

    source_text -- текст подготовительного документа (если has_prep=True)
                   или исходная задача пользователя (если has_prep=False, профиль simple).
    user_task   -- оригинальная задача; нужна для профилей без prep (simple).
    """
    banner(profile.banner_json)
    flog.banner(profile.banner_json)

    ai.systems[1] = {"prompt": profile.json_system}
    ai.chat_history.clear()

    # Профиль может использовать {prep_text} или {user_task} — форматируем оба
    try:
        prt = profile.json_user.format(prep_text=source_text, user_task=user_task or source_text)
    except KeyError:
        prt = profile.json_user.format(user_task=user_task or source_text)

    raw = await ai.run_async(prt)

    # Извлекаем JSON из ответа модели.
    # Ищем позицию первой '{' и пробуем декодировать с неё нарастающим окном.
    # Это надёжнее чем жадный regex \{[\s\S]+\} который при наличии текста
    # после закрывающей скобки захватывает лишнее и ломает json.loads.
    prd_json = None
    start = raw.find('{')
    if start != -1:
        # Пробуем взять фрагмент от start до каждой '}' справа налево
        # (то есть от самого длинного совпадения к самому короткому).
        # json.loads сам скажет где JSON заканчивается через JSONDecodeError.
        # Быстрый путь: decoder.raw_decode останавливается ровно на конце объекта.
        try:
            prd_json, _ = json.JSONDecoder().raw_decode(raw, start)
        except json.JSONDecodeError:
            pass

    if prd_json is None:
        raise ValueError(f"Модель не вернула валидный JSON.\nОтвет:\n{raw[:500]}")
    stories = prd_json.get("userStories", [])
    msg = f"prd.json сгенерирован: {len(stories)} user-stories"
    log(msg); flog.log(msg)

    json_path = workdir / "prd.json"
    json_path.write_text(
        json.dumps(prd_json, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    msg = f"Сохранён: {json_path}"
    log(msg); flog.log(msg)

    # Создаём пустой progress.txt если не существует
    progress_path = workdir / "progress.txt"
    if not progress_path.exists():
        progress_path.write_text(
            f"# Progress Log\nПроект: {prd_json.get('project', '???')}\n"
            f"Начало: {_dt.now().isoformat()}\n\n",
            encoding="utf-8"
        )

    return prd_json



# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции для подфаз
# ══════════════════════════════════════════════════════════════════════════════

def _make_context_prompts(
    workdir: Path,
    story_json: str | None = None,
    include_prd: bool = False,
) -> dict:
    """
    Строит контекстный словарь prompts для агента.

    prompts[10] — PRD.md        (только если include_prd=True)
    prompts[20] — progress.txt
    prompts[40] — story JSON    (если передан; без дублирования в prt)

    include_prd=True только для умника-исправителя.
    Остальным агентам PRD.md не нужен — он увеличивает контекст без пользы.
    """
    p: dict = {}

    if include_prd:
        prd_md = workdir / "PRD.md"
        if prd_md.exists():
            p[10] = {"text_file": str(prd_md)}

    progress = workdir / "progress.txt"
    if progress.exists():
        p[20] = {"text_file": str(progress)}

    if story_json:
        p[40] = {"prompt": f"Текущая задача:\n{story_json}"}

    # CLAUDE.md — паттерны и антипаттерны проекта (если существует и непустой)
    claude_md = workdir / "CLAUDE.md"
    if claude_md.exists():
        content = claude_md.read_text(encoding="utf-8", errors="replace").strip()
        # Передаём только если есть что-то кроме шаблона (маркеры не пустые)
        if "<!-- PATTERNS_START -->" not in content or content != CLAUDE_MD_TEMPLATE.strip():
            p[15] = {"text_file": str(claude_md)}

    return p


def _strip_thinking(chat_history: list) -> list:
    """
    Возвращает копию chat_history с удалёнными блоками рассуждений.

    Для строкового content — убирает всё между <thinking>...</thinking>.
    Для list-content (Anthropic) — выбрасывает блоки type="thinking".
    additional_kwargs["reasoning_content"] обнуляется.
    Исходная история не мутируется.
    """
    import copy, re as _re
    _THINK_RE = _re.compile(r"<thinking>.*?</thinking>", _re.DOTALL | _re.IGNORECASE)

    result = []
    for msg in chat_history:
        msg = copy.deepcopy(msg)
        data = msg.get("data", {})

        # Строковый content
        content = data.get("content", "")
        if isinstance(content, str):
            data["content"] = _THINK_RE.sub("", content).strip()

        # List-content (Anthropic/Gemini blocks)
        elif isinstance(content, list):
            data["content"] = [
                b for b in content
                if not (isinstance(b, dict) and b.get("type") == "thinking")
            ]

        # additional_kwargs reasoning_content (Ollama)
        ak = data.get("additional_kwargs")
        if isinstance(ak, dict) and "reasoning_content" in ak:
            ak["reasoning_content"] = ""

        msg["data"] = data
        result.append(msg)
    return result


def _pick_next_story(prd: dict) -> dict | None:
    """
    Выбирает следующую невыполненную story.

    Среди всех pending-задач берём минимальный priority.
    Если несколько задач имеют одинаковый минимальный priority —
    выбираем случайную из них (чтобы не зависать на одной при повторах).
    Возвращает dict story или None если все выполнены.
    """
    import random
    pending = [s for s in prd.get("userStories", []) if not s.get("passes", False)]
    if not pending:
        return None
    min_priority = min(s["priority"] for s in pending)
    candidates = [s for s in pending if s["priority"] == min_priority]
    return random.choice(candidates)


def _deprioritize(story_id: str, _load, _save) -> None:
    """
    Сдвигает приоритет неудачной story вправо через дробное число,
    вставляя её между текущей позицией и следующей story.

    Алгоритм:
      Берём priority неудачницы (P) и наименьший priority среди остальных
      незавершённых задач с priority > P (назовём его Q).
      Новый priority = (P + Q) / 2.
      Если Q не существует (неудачница уже последняя) — P + 1.
    """
    data = _load()
    stories = data["userStories"]

    target = next((s for s in stories if s["id"] == story_id), None)
    if target is None:
        return

    current_p = target["priority"]

    # Следующий приоритет среди незавершённых (строго больше текущего)
    others = [
        s["priority"] for s in stories
        if s["id"] != story_id and not s.get("passes", False)
        and s["priority"] > current_p
    ]
    next_p = min(others) if others else current_p + 1

    target["priority"] = (current_p + next_p) / 2
    _save(data)


# ══════════════════════════════════════════════════════════════════════════════
# Утилиты отката и памяти о неудачах
# ══════════════════════════════════════════════════════════════════════════════

def _parse_fail_memory(answer: str) -> tuple[str, str]:
    """
    Извлекает из ответа умника два поля после <STORY_FAIL>:
      <ISSUE_ADVICE>текст</ISSUE_ADVICE>   — совет для CLAUDE.md
      <FAIL_LOG>текст</FAIL_LOG>           — хронология для progress.txt

    Возвращает (issue_advice, fail_log). Пустые строки если теги не найдены.
    """
    def _extract(tag: str, text: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    return _extract("ISSUE_ADVICE", answer), _extract("FAIL_LOG", answer)


def _record_fail(
    workdir: Path,
    story_id: str,
    story_title: str,
    issue_advice: str,
    fail_log: str,
    commit_hash: str,
    flog: "FileLogger",
) -> None:
    """
    После отката записывает информацию о неудаче:
    - issue_advice → CLAUDE.md ## Известные проблемы (через _claude_md_append_section)
    - fail_log     → progress.txt с маркером [ОТМЕНЕНО]
    """
    ts = _dt.now().isoformat(timespec="seconds")

    # CLAUDE.md — только если есть совет
    if issue_advice:
        _claude_md_append_section(
            workdir, f"[{story_id}] {issue_advice}",
            "<!-- ISSUES_START -->", "<!-- ISSUES_END -->"
        )
        flog.log(f"[FAIL] Совет записан в CLAUDE.md")

    # progress.txt — хронологическая запись об отмене
    progress_path = workdir / "progress.txt"
    fail_entry_lines = [
        "---",
        f"[{ts}] ❌ ОТМЕНЕНО: {story_id} - {story_title}",
        f"Откат к коммиту: {commit_hash}",
    ]
    if fail_log:
        fail_entry_lines.append(f"Что пробовали и что не сработало: {fail_log}")
    fail_entry_lines.append("---\n")

    with progress_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(fail_entry_lines))
    flog.log(f"[FAIL] Запись об отмене добавлена в progress.txt")


# ══════════════════════════════════════════════════════════════════════════════
# Подфазы одной итерации
# ══════════════════════════════════════════════════════════════════════════════

AGENT_TIMEOUT = 300  # 5 минут на один вызов агента

async def _run_with_timeout(
    ai: "RalphAgent",
    prt: str,
    config: dict,
    flog: "FileLogger",
    label: str,
) -> str | None:
    """
    Запускает ai.run_async с таймаутом AGENT_TIMEOUT секунд.
    При таймауте или исключении делает одну повторную попытку.
    Возвращает ответ агента или None при двойном сбое.
    """
    for attempt in (1, 2):
        try:
            result = await asyncio.wait_for(
                ai.run_async(prt, config=config),
                timeout=AGENT_TIMEOUT,
            )
            return result
        except asyncio.TimeoutError:
            msg = f"{label} таймаут {AGENT_TIMEOUT}s (попытка {attempt}/2)"
            flog.log(f"WARNING: {msg}")
            if attempt == 2:
                return None
        except Exception as e:
            msg = f"{label} ошибка: {e} (попытка {attempt}/2)"
            flog.log(f"ERROR: {msg}")
            if attempt == 2:
                return None
        await asyncio.sleep(3)
    return None


async def _subphase_work(
    worker: "RalphAgent",
    workdir: Path,
    story: dict,
    flog: "FileLogger",
    iteration: int,
) -> tuple[bool, str, list]:
    """Подфаза B: работяга выполняет задачу. Возвращает (success, answer, chat_history)."""
    story_json = json.dumps(story, ensure_ascii=False, indent=2)

    worker.systems[1] = {"prompt": PROMPT_WORKER_SYSTEM}
    # story передаётся в prompts[40], а не в prt — чтобы не дублировать
    worker.prompts = _make_context_prompts(workdir, story_json, include_prd=False)
    worker.chat_history.clear()
    worker.log = ""

    flog.write(f"\n=== Итерация {iteration}: [B] работяга выполняет {story['id']} ===\n")
    worker.get_systemprompt_log()
    flog.ai_log(worker)

    prt = PROMPT_WORKER_USER.format(workdir=str(workdir))
    answer = await _run_with_timeout(worker, prt, worker.make_run_config(),
                                      flog, f"[B] iter={iteration}")

    flog.write(f"\n=== Итерация {iteration}: [B] лог ===\n")
    flog.ai_log(worker)

    if answer is None:
        return False, "Агент не вернул ответ", []

    success = "<STORY_OK>" in answer.upper()
    return success, answer, list(worker.chat_history)


async def _subphase_check(
    hustler: "RalphAgent",
    workdir: Path,
    story: dict,
    flog: "FileLogger",
    iteration: int,
) -> tuple[bool, str]:
    """
    Подфаза C: шустрик проверяет выполнение по acceptanceCriteria.
    Возвращает (passed, answer). answer содержит причину при CHECK_FAIL.
    """
    story_json = json.dumps(story, ensure_ascii=False, indent=2)

    hustler.systems[1] = {"prompt": PROMPT_CHECK_SYSTEM}
    hustler.prompts = _make_context_prompts(workdir, story_json, include_prd=False)
    hustler.chat_history.clear()
    hustler.log = ""

    flog.write(f"\n=== Итерация {iteration}: [C] шустрик проверяет {story['id']} ===\n")
    hustler.get_systemprompt_log()
    flog.ai_log(hustler)

    prt = PROMPT_CHECK_USER.format(workdir=str(workdir))
    answer = await _run_with_timeout(hustler, prt, hustler.make_run_config(),
                                      flog, f"[C] iter={iteration}")

    flog.write(f"\n=== Итерация {iteration}: [C] лог ===\n")
    flog.ai_log(hustler)

    if answer is None:
        return False, "Проверяющий не вернул ответ"
    passed = "<CHECK_PASS>" in answer.upper()
    return passed, answer


async def _subphase_fix(
    thinker: "RalphAgent",
    workdir: Path,
    story: dict,
    worker_history: list,
    check_answer: str,
    check_history: list,
    flog: "FileLogger",
    iteration: int,
) -> tuple[bool, str]:
    """
    Подфаза D: умник исправляет задачу после провала.

    История работяги и проверяющего (без рассуждений) передаётся в chat_history.
    Задача и PRD — в prompts. В prt — только вердикт проверяющего.
    Возвращает (success, answer).
    """
    story_json = json.dumps(story, ensure_ascii=False, indent=2)
    combined_history = _strip_thinking(worker_history + check_history)
    check_verdict = check_answer[:800] if check_answer else "(проверка не проводилась)"

    thinker.systems[1] = {"prompt": PROMPT_FIXER_SYSTEM}
    thinker.prompts = _make_context_prompts(workdir, story_json, include_prd=True)
    thinker.chat_history = combined_history
    thinker.log = ""

    flog.write(f"\n=== Итерация {iteration}: [D] умник исправляет {story['id']} ===\n")
    thinker.get_systemprompt_log()
    flog.ai_log(thinker)

    prt = PROMPT_FIXER_USER.format(workdir=str(workdir), check_verdict=check_verdict)
    answer = await _run_with_timeout(thinker, prt, thinker.make_run_config(),
                                      flog, f"[D] iter={iteration}")

    flog.write(f"\n=== Итерация {iteration}: [D] лог ===\n")
    flog.ai_log(thinker)

    if answer is None:
        return False, ""
    return "<STORY_OK>" in answer.upper(), answer


# ══════════════════════════════════════════════════════════════════════════════
# Аудитор (подфазы E1 и E2)
# ══════════════════════════════════════════════════════════════════════════════

def _progress_tail(workdir: Path, n_entries: int = 8) -> str:
    """Возвращает последние n_entries записей из progress.txt (по разделителю ---)."""
    path = workdir / "progress.txt"
    if not path.exists():
        return "(progress.txt не найден)"
    content = path.read_text(encoding="utf-8", errors="replace")
    # Делим по блокам между ---
    blocks = re.split(r"(?m)^---$", content)
    # Берём последние n*2 кусков (каждая запись это 2 блока: заголовок + тело)
    tail_blocks = blocks[-(n_entries * 2):]
    return "---".join(tail_blocks).strip() or content[-3000:]


async def _subphase_audit(
    hustler: "RalphAgent",
    thinker: "RalphAgent",
    workdir: Path,
    flog: "FileLogger",
    iteration: int,
    _prd_done_fn,
    _load_fn,
    _save_fn,
) -> str | None:
    """
    Подфаза E: двухэтапный аудит.
    E1 — шустрик: технический аудит файлов из последних итераций.
    E2 — умник: смысловой аудит, исправление, reopening stories.

    Возвращает None в норме, или строку с причиной критической остановки.
    """
    flog.write(f"\n=== Итерация {iteration}: [E] аудит ===\n")
    log(f"[E] Запуск аудитора"); flog.log("[E] Запуск аудитора")

    # ── E1: шустрик — технический аудит ─────────────────────────────────────
    progress_tail = _progress_tail(workdir, n_entries=QA_INTERVAL + 1)

    hustler.systems[1] = {"prompt": PROMPT_AUDIT_TECH_SYSTEM}
    hustler.prompts = _make_context_prompts(workdir, include_prd=False)
    hustler.chat_history.clear()
    hustler.log = ""

    flog.write(f"\n=== Итерация {iteration}: [E1] технический аудит ===\n")
    hustler.get_systemprompt_log()
    flog.ai_log(hustler)

    prt_e1 = PROMPT_AUDIT_TECH_USER.format(
        workdir=str(workdir),
        progress_tail=progress_tail,
    )
    tech_answer = await _run_with_timeout(
        hustler, prt_e1, hustler.make_run_config(), flog, f"[E1] iter={iteration}"
    )

    flog.write(f"\n=== Итерация {iteration}: [E1] лог ===\n")
    flog.ai_log(hustler)

    # Извлекаем резюме из ответа E1
    tech_summary = "(E1 не вернул ответ)"
    if tech_answer:
        m = re.search(r"## Резюме(.*?)$", tech_answer, re.DOTALL)
        tech_summary = m.group(1).strip() if m else tech_answer[-1000:]

    log(f"[E1] технический аудит завершён"); flog.log("[E1] завершён")

    # ── E2: умник — смысловой аудит ──────────────────────────────────────────
    thinker.systems[1] = {"prompt": PROMPT_AUDIT_SMART_SYSTEM}
    thinker.prompts = _make_context_prompts(workdir, include_prd=True)
    thinker.chat_history.clear()
    thinker.log = ""

    flog.write(f"\n=== Итерация {iteration}: [E2] смысловой аудит ===\n")
    thinker.get_systemprompt_log()
    flog.ai_log(thinker)

    prt_e2 = PROMPT_AUDIT_SMART_USER.format(
        workdir=str(workdir),
        tech_summary=tech_summary,
    )
    smart_answer = await _run_with_timeout(
        thinker, prt_e2, thinker.make_run_config(), flog, f"[E2] iter={iteration}"
    )

    flog.write(f"\n=== Итерация {iteration}: [E2] лог ===\n")
    flog.ai_log(thinker)

    if not smart_answer:
        log("[E2] умник не вернул ответ"); flog.log("[E2] нет ответа")
        return None

    # Обрабатываем REOPEN
    for m in re.finditer(r"<REOPEN:\s*(US-\d+)>", smart_answer, re.IGNORECASE):
        sid = m.group(1).strip()
        data = _load_fn()
        for s in data["userStories"]:
            if s["id"] == sid:
                s["passes"] = False
                s["notes"] = f"Reopened by auditor at iteration {iteration}"
        _save_fn(data)
        msg = f"[E2] Story {sid} возвращена в очередь"
        log(msg); flog.log(msg)

    # Проверяем критическую остановку
    crit_match = re.search(
        r"<CRITICAL_STOP>(.*?)</CRITICAL_STOP>", smart_answer, re.DOTALL | re.IGNORECASE
    )
    if crit_match:
        crit_text = crit_match.group(1).strip()
        brief_m = re.search(r"КРАТКАЯ ПРИЧИНА:\s*(.+)", crit_text)
        detail_m = re.search(r"ПОДРОБНО:\s*(.+)", crit_text, re.DOTALL)
        brief  = brief_m.group(1).strip() if brief_m else crit_text[:100]
        detail = detail_m.group(1).strip() if detail_m else crit_text

        # Записываем в progress.txt
        ts = _dt.now().isoformat(timespec="seconds")
        progress_path = workdir / "progress.txt"
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(f"\n---\n[{ts}] 🛑 КРИТИЧЕСКАЯ ОСТАНОВКА\n")
            f.write(f"Краткая причина: {brief}\n")
            f.write(f"Подробно: {detail}\n---\n")

        log(f"🛑 КРИТИЧЕСКАЯ ОСТАНОВКА: {brief}")
        flog.log(f"CRITICAL_STOP: {brief}")
        return f"КРИТИЧЕСКАЯ ОСТАНОВКА: {brief}\n\n{detail}"

    log("[E2] аудит завершён"); flog.log("[E2] завершён")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Основной цикл
# ══════════════════════════════════════════════════════════════════════════════


async def ralph_loop(
    hustler: "RalphAgent",
    worker: "RalphAgent",
    thinker: "RalphAgent",
    workdir: Path,
    max_iter: int,
    flog: "FileLogger",
    _load,
    _save,
    _prd_done,
    branch_name: str,
    profile: "RalphProfile",
) -> None:
    """Ральф-цикл с подфазами A→B→C→D и аудитором E."""
    phase_num = 4  # Фаза 4 т.к. фазы 1=dispatch, 2=prep, 3=json
    banner(f"Фаза {phase_num}: Ральф-цикл [{profile.name}]")
    flog.banner(f"Фаза {phase_num}: Ральф-цикл [{profile.name}]")

    for ai in (hustler, worker, thinker):
        ai.loglevel = 2
        ai.log_thinking = True

    # Системный промпт рабочего цикла из профиля
    worker.systems[1] = {"prompt": profile.cycle_system}

    json_path = workdir / "prd.json"

    for iteration in range(1, max_iter + 1):
        prd = read_json(json_path)
        remaining = count_remaining(prd)

        if remaining == 0:
            msg = "Все задачи выполнены."
            log(f"✅ {msg}"); flog.log(msg)
            break

        msg = f"Итерация {iteration}/{max_iter} | Осталось задач: {remaining}"
        log(msg); flog.log(msg)

        # ── A: выбор задачи ──────────────────────────────────────────────────
        prd = read_json(json_path)
        story = _pick_next_story(prd)
        if story is None:
            log("🎉 Все задачи выполнены."); flog.log("Все задачи выполнены.")
            break
        story_id = story["id"]
        msg = f"Задача: [{story_id}] {story['title']}"
        log(msg); flog.log(msg)

        # Запоминаем хэш текущего коммита для возможного отката
        commit_hash = git_current_hash(workdir)

        # ── B: работяга выполняет ────────────────────────────────────────────
        try:
            work_ok, work_answer, work_history = await _subphase_work(
                worker, workdir, story, flog, iteration
            )
        except Exception as e:
            msg = f"[B] ошибка работяги: {e}"
            log(f"⚠️  {msg}"); flog.log(f"ERROR: {msg}")
            work_ok, work_answer, work_history = False, str(e), []

        log(f"[B] работяга: {'OK' if work_ok else 'FAIL'}")
        flog.log(f"[B] работяга: {'OK' if work_ok else 'FAIL'}")

        final_ok = False
        final_answer = work_answer  # ответ последнего агента (для FAIL-парсинга)

        if not work_ok:
            msg = "[B→D] работяга не справился, передаём умнику"
            log(f"⚠️  {msg}"); flog.log(f"WARNING: {msg}")
            try:
                fix_ok, fix_answer = await _subphase_fix(
                    thinker, workdir, story,
                    work_history, work_answer, [],
                    flog, iteration,
                )
            except Exception as e:
                msg = f"[D] ошибка умника: {e}"
                log(f"⚠️  {msg}"); flog.log(f"ERROR: {msg}")
                fix_ok, fix_answer = False, str(e)
            log(f"[D] умник: {'OK' if fix_ok else 'FAIL'}")
            flog.log(f"[D] умник: {'OK' if fix_ok else 'FAIL'}")
            final_ok = fix_ok
            final_answer = fix_answer
        else:
            # ── C: шустрик проверяет ─────────────────────────────────────────
            try:
                check_ok, check_answer = await _subphase_check(
                    hustler, workdir, story, flog, iteration
                )
                check_history = list(hustler.chat_history)
            except Exception as e:
                msg = f"[C] ошибка проверки: {e}"
                log(f"⚠️  {msg}"); flog.log(f"ERROR: {msg}")
                check_ok, check_answer, check_history = False, str(e), []

            log(f"[C] проверка: {'PASS' if check_ok else 'FAIL'}")
            flog.log(f"[C] проверка: {'PASS' if check_ok else 'FAIL'}")

            if not check_ok:
                msg = "[C→D] проверка не прошла, передаём умнику"
                log(f"⚠️  {msg}"); flog.log(f"WARNING: {msg}")
                try:
                    fix_ok, fix_answer = await _subphase_fix(
                        thinker, workdir, story,
                        work_history, check_answer, check_history,
                        flog, iteration,
                    )
                except Exception as e:
                    msg = f"[D] ошибка умника: {e}"
                    log(f"⚠️  {msg}"); flog.log(f"ERROR: {msg}")
                    fix_ok, fix_answer = False, str(e)
                log(f"[D] умник: {'OK' if fix_ok else 'FAIL'}")
                flog.log(f"[D] умник: {'OK' if fix_ok else 'FAIL'}")
                final_ok = fix_ok
                final_answer = fix_answer
            else:
                final_ok = True

        # ── Финальное решение по задаче ──────────────────────────────────────
        if final_ok:
            _prd_done(story_id)
            git_commit(workdir, story_id, story["title"], flog)
            msg = f"✅ [{story_id}] выполнена, закоммичена."
            log(msg); flog.log(msg)
        else:
            # Парсим память о неудаче ДО отката
            issue_advice, fail_log = _parse_fail_memory(final_answer)
            # Откат к состоянию до итерации
            git_rollback(workdir, flog)
            # После отката дописываем в progress.txt и CLAUDE.md
            _record_fail(workdir, story_id, story["title"],
                         issue_advice, fail_log, commit_hash, flog)
            # Коммитим только служебные файлы (progress.txt, CLAUDE.md)
            git_commit(workdir, story_id, f"[FAIL] {story['title']}", flog)
            # Сдвигаем приоритет чтобы не зависать
            _deprioritize(story_id, _load, _save)
            msg = f"❌ [{story_id}] не выполнена — откат, приоритет сдвинут."
            log(msg); flog.log(msg)

        # ── E: аудитор (каждые QA_INTERVAL итераций) ─────────────────────────
        is_last = (iteration == max_iter) or (count_remaining(read_json(json_path)) == 0)
        if iteration % QA_INTERVAL == 0 or is_last:
            stop_reason = await _subphase_audit(
                hustler, thinker, workdir, flog, iteration,
                _prd_done, _load, _save,
            )
            if stop_reason:
                banner("🛑 КРИТИЧЕСКАЯ ОСТАНОВКА")
                print(stop_reason)
                return

        if iteration == max_iter:
            msg = f"Достигнут лимит итераций ({max_iter})."
            log(f"⏹  {msg}"); flog.log(msg)

    # Финальная сводка
    try:
        prd = read_json(json_path)
    except Exception as e:
        log(f"⚠️  Не удалось прочитать финальный prd.json: {e}")
        return

    total = len(prd.get("userStories", []))
    done  = total - count_remaining(prd)
    summary = f"Итог: {done}/{total} задач выполнено"
    banner(summary)
    flog.banner(summary)

    for s in prd.get("userStories", []):
        mark_con = "✅" if s.get("passes") else "❌"
        mark_log = "[OK]" if s.get("passes") else "[--]"
        line = f"[{s['id']}] {s['title']}"
        print(f"  {mark_con} {line}")
        flog.write(f"  {mark_log} {line}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════════════════


async def dispatch(
    ai: "RalphAgent",
    user_task: str,
    flog: "FileLogger",
) -> "RalphProfile":
    """
    Фаза 1: Регулировщик.
    Один вызов модели без инструментов, temperature=0.0.
    Возвращает RalphProfile. При невалидном ответе автоматически перезапускается.
    При неизвестном профиле — fallback на "build" с предупреждением.
    """
    banner("Фаза 1: Определение профиля")
    flog.banner("Фаза 1: Определение профиля")

    ai.systems[1] = {"prompt": PROMPT_DISPATCH_SYSTEM}
    ai.chat_history.clear()

    prt = PROMPT_DISPATCH_USER.format(user_task=user_task)
    config = {"configurable": {"temperature": 0.0}}

    for attempt in range(1, 4):  # до 3 попыток
        raw = await ai.run_async(prt, config=config)

        if raw:
            m = re.search(r'\{[\s\S]+?\}', raw)
            if m:
                try:
                    result = json.loads(m.group())
                    profile_name = result.get("profile", "").strip().lower()
                    reason = result.get("reason", "")

                    if profile_name in PROFILES:
                        profile = PROFILES[profile_name]
                        msg = f'Профиль: {profile_name} ("{reason}")'
                        log(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}")
                        flog.log(msg)
                        return profile
                    else:
                        msg = f"Неизвестный профиль '{profile_name}', fallback → build"
                        log(f"⚠️  {msg}"); flog.log(f"WARNING: {msg}")
                        return PROFILES["build"]
                except json.JSONDecodeError:
                    pass

        msg = f"[dispatch] невалидный ответ, попытка {attempt}/3"
        log(f"⚠️  {msg}"); flog.log(f"WARNING: {msg}")
        ai.chat_history.clear()
        await asyncio.sleep(2)

    # После 3 неудач — fallback на build
    msg = "dispatch не смог определить профиль после 3 попыток, fallback → build"
    log(f"⚠️  {msg}"); flog.log(f"WARNING: {msg}")
    return PROFILES["build"]


async def main(factory: RalphFactory = None) -> None:
    parser = argparse.ArgumentParser(description="Ральф-цикл: автономный агент на AIList")
    parser.add_argument("task", nargs="?", default="",
        help="Задача в свободном тексте (если не указана — спросим интерактивно)")
    parser.add_argument("--workdir", default="./ralph_project",
        help="Рабочая папка (по умолчанию: ./ralph_project)")
    parser.add_argument("--max-iter", type=int, default=50,
        help="Максимальное число итераций (по умолчанию: 50)")
    parser.add_argument("--resume", action="store_true",
        help="Пропустить фазы 1-3, продолжить с существующим prd.json")
    parser.add_argument("--profile", default="",
        help="Принудительный выбор профиля: build|research|data|content|simple")
    args = parser.parse_args()

    factory = factory or RalphFactory()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    flog = FileLogger(workdir / "ralph.log")
    msg = f"Рабочая папка: {workdir}"
    log(msg); flog.log(msg)

    json_path   = workdir / "prd.json"
    ralph_tools, _load, _save, _prd_done, _ = make_ralph_tools(json_path)
    claude_tools = make_claude_tools(workdir)
    all_tools = ralph_tools + claude_tools

    async with factory.hustler(tools=all_tools) as hustler, \
               factory.worker(tools=all_tools)  as worker, \
               factory.thinker(tools=all_tools) as thinker:

        # workspace подключаем сразу — нужен всем профилям на всех фазах
        # exclude/exclude_patterns: эти файлы будут невидимы для search_for_pattern
        _SEARCH_EXCLUDE = {"ralph.log", "progress.txt"}
        for ai in (hustler, worker, thinker):
            await ai.mcp_connect("workspace",
                                 exclude=_SEARCH_EXCLUDE,
                                 exclude_patterns=["*.log"])

        msg = f"Инструменты: {', '.join(worker.mcp_tool_names())}"
        log(msg); flog.log(msg)

        for ai in (hustler, worker, thinker):
            os.chdir(workdir)
            ai.set_workspace(workdir)
            ai.set_use_attachments(False)

        claude_md_init(workdir)

        if args.resume:
            if not json_path.exists():
                print(f"Ошибка: {json_path} не найден. Запустите без --resume.")
                return
            msg = "Режим --resume: пропускаем фазы 1-3."
            log(msg); flog.log(msg)
            prd_data = read_json(json_path)
            branch_name = prd_data.get("branchName", "ralph/resume")

            # Читаем сохранённый профиль
            profile_path = workdir / "profile.json"
            if profile_path.exists():
                try:
                    saved = json.loads(profile_path.read_text(encoding="utf-8"))
                    profile = PROFILES.get(saved.get("profile", ""), PROFILES["build"])
                except Exception:
                    profile = PROFILES["build"]
            else:
                profile = PROFILES["build"]
            msg = f"Профиль (из profile.json): {profile.name}"
            log(msg); flog.log(msg)

        else:
            task = args.task.strip()
            if not task:
                print("\nОпишите задачу (Enter дважды для завершения):")
                task_lines = []
                try:
                    while True:
                        line = input()
                        if line == "" and task_lines and task_lines[-1] == "":
                            break
                        task_lines.append(line)
                except EOFError:
                    pass
                task = "\n".join(task_lines).strip()

            if not task:
                print("Задача не задана. Выход.")
                return

            flog.log(f"Задача пользователя: {task}")

            # ── Фаза 1: регулировщик ──────────────────────────────────────────
            if args.profile:
                if args.profile not in PROFILES:
                    print(f"Неизвестный профиль: {args.profile}. Доступны: {list(PROFILES)}")
                    return
                profile = PROFILES[args.profile]
                msg = f"Профиль задан вручную: {profile.name}"
                log(msg); flog.log(msg)
            else:
                profile = await dispatch(hustler, task, flog)

            # Сохраняем профиль для --resume
            (workdir / "profile.json").write_text(
                json.dumps({"profile": profile.name}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            # ── Фаза 2: подготовительный документ (если нужен) ───────────────
            if profile.has_prep:
                prep_text = await generate_prep_doc(worker, task, workdir, flog, profile)
            else:
                prep_text = task  # simple: JSON генерируется прямо из задачи

            # ── Фаза 3: декомпозиция ──────────────────────────────────────────
            prd_data = await generate_prd_json(
                worker, prep_text, workdir, flog, profile, user_task=task
            )

            # Подключаем профиле-специфичные MCP-серверы (кроме workspace — уже есть)
            extra_servers = [s for s in profile.mcp_servers if s != "workspace"]
            for srv in extra_servers:
                try:
                    for ai in (hustler, worker, thinker):
                        await ai.mcp_connect(srv)
                    msg = f"MCP подключён: {srv}"
                    log(msg); flog.log(msg)
                except Exception as e:
                    msg = f"MCP {srv} недоступен: {e} — продолжаем без него"
                    log(f"⚠️  {msg}"); flog.log(f"WARNING: {msg}")

            # Создаём нужные папки профиля
            for d in profile.init_dirs:
                (workdir / d).mkdir(exist_ok=True)

            # Генерируем имя ветки
            project_name = prd_data.get("project", task)
            branch_name = "ralph/" + _to_kebab(project_name)
            prd_data["branchName"] = branch_name
            json_path.write_text(
                json.dumps(prd_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            flog.log(f"branchName: {branch_name}")

        git_setup(workdir, branch_name, flog)

        await ralph_loop(
            hustler, worker, thinker,
            workdir, args.max_iter, flog,
            _load, _save, _prd_done,
            branch_name,
            profile,
        )


if __name__ == "__main__":
    asyncio.run(main())