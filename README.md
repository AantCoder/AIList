# AIList projects

Two Python packages for building and running LLM agents — from single-model scripts to autonomous multi-iteration coding loops.

---

## [AIList](./ailist/README.md) — LangChain agent framework

Multi-provider LLM agent engine with MCP server integration, file attachments, and persistent history.

**Why AIList instead of raw LangChain:**
- Single API across Ollama, Claude, GPT, Gemini, Qwen — swap models without changing call-site code
- MCP servers connect at runtime in one line; the agent is automatically rebuilt with new tools
- Files just work — PDFs, images, Office, audio, video are auto-converted to whatever format the model supports
- Context management built in — auto-summarization, token estimates before sending, branchable history
- Thinking mode (Claude, o-series, Qwen reasoning) via a single `apply_thinking_mode()` call
- Subclass once per project, configure everything in `__init__`, keep call-site code minimal

```python
async with MyAgent() as ai:
    await ai.mcp_connect("workspace")
    result = await ai.run_async("Refactor the auth module.")
```

→ [Full docs and examples](./ailist/README.md)

---

## [Ralph](./ralph/README.md) — Autonomous iterative coding agent

Autonomous agent that decomposes a task into user stories and executes them one by one in a loop, built on AIList.

**Why Ralph:**
- Handles weak local models — one story per iteration keeps the context small and focused
- Three-agent pipeline per story: worker builds, checker verifies, thinker fixes on failure
- Periodic auditor catches regressions and can reopen completed stories
- Full git integration — branch per project, commit per story, rollback on failure
- Five task profiles (build / research / data / content / simple) with different decomposition strategies
- Extend by subclassing `RalphFactory` — swap models per role without touching core logic

```bash
python ralph_demo.py "Build an offline HTML calculator page"
```

→ [Full docs](./ralph/README.md)

---

## Structure

```
AIList/
├── ailist/          # AIList package (LLM engine + MCP + file attachments)
└── ralph/           # Ralph package (autonomous agent built on AIList)
```

## License

MIT



---



# AIList проекты

Два пакета Python для создания и запуска агентов LLM — от сценариев с одной моделью до автономных многоитерационных циклов кодирования.

---

## [AIList](./ailist/README.md) — структура агента LangChain

Механизм агента LLM с несколькими поставщиками с интеграцией сервера MCP, вложениями файлов и постоянной историей.

**Почему AIList вместо сырого LangChain:**
— Единый API для Ollama, Claude, GPT, Gemini, Qwen — меняйте модели без изменения кода сайта вызова
— Серверы MCP подключаются во время выполнения в одну строку; агент автоматически перестраивается с помощью новых инструментов
- Файлы просто работают - PDF-файлы, изображения, Office, аудио, видео автоматически конвертируются в любой формат, поддерживаемый моделью
- Встроенное управление контекстом - автоматическое суммирование, оценки токенов перед отправкой, разветвляемая история
- Режим мышления (Клод, o-series, рассуждения Квен) с помощью одного вызова `apply_thinking_mode()`
- Подкласс один раз для каждого проекта, настройте все в `__init__`, сохраняйте минимальный код места вызова
```python
асинхронно с MyAgent() как ai:
await ai.mcp_connect("workspace")
result = await ai.run_async("Рефакторинг модуля аутентификации.")
```

→ [Полная документация и примеры](./ailist/README.md)

---

## [Ralph](./ralph/README.md) — Автономный агент итеративного кодирования

Автономный агент, разлагающий задачу на пользовательские истории и выполняющий их одну за другой в цикле, построенный на AIList.

**Почему Ральф:**
- Обрабатывает слабые локальные модели — одна история на итерацию сохраняет контекст небольшим и целенаправленным
- Конвейер из трех агентов на историю: рабочий строит, проверяет, мыслитель исправляет сбой
- Периодический аудитор выявляет регрессии и может повторно открывать завершенные истории
- Полная интеграция с git — ветвление для каждого проекта, фиксация для каждой истории, откат в случае сбоя
- Пять профилей задач (программирование/исследование/данные/контент/множества простых задач) с различными стратегиями декомпозиции
- Расширение путем создания подкласса `RalphFactory` — меняйте модели для каждой роли, не затрагивая базовую логику
```bash
python ralph_demo.py "Создание автономной страницы HTML-калькулятора"
```

→ [Полная документация](./ralph/README.md)

---

## Структура

```
AIList/
├── ailist/ # Пакет AIList (движок LLM + MCP + вложения файлов)
└── ralph/ # Пакет Ralph (автономный агент, построенный на AIList)
```

## Лицензия

MIT