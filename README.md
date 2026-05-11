# AIList

A LangChain-based Python wrapper for running LLM agents with multi-provider support, multi-agent orchestration, file attachments, and MCP server integration.

## What it does

- **Orchestrate multiple agents in plain Python** — branch conversations, roll back history, run agents in parallel, hand off between models; no graph framework needed
- **Talk to any LLM** — Ollama (local), Anthropic Claude, OpenAI GPT, Google Gemini, and others through a single unified API
- **Connect MCP servers** — extend the agent with tools: file editing, browser automation, Git, vector memory, web search, TTS, notifications, and more. Register any server in a few lines — plain Python tools, stdio MCP, HTTP/SSE MCP, or Docker-hosted MCP. Per-server system prompts, tool filtering, and schema patches are supported out of the box
- **Attach files** — just pass the file and don't think about model compatibility; PDFs, images, Office documents, audio, and video are each delivered in the best format the model supports, and automatically converted to text or transcribed via Whisper when it doesn't
- **Persistent chat history** — maintained automatically across calls; warns or auto-summarizes when the context window fills up; since history is just a Python list, you can branch it, merge histories from multiple agents, or surgically remove any entry

---

## ⚡ Quick start

### Any provider, same API

```python
from ailist import AIListBase, Provider

# Local model via Ollama
ai = AIListBase("ollama:llama3.2", context_limit=8192, provider=Provider.LLAMA)

# Anthropic Claude
ai = AIListBase("anthropic:claude-opus-4-6", context_limit=180000, provider=Provider.ANTHROPIC)

# OpenAI
ai = AIListBase("openai:gpt-4o", context_limit=128000, provider=Provider.OPENAI)

ai.system = "You are a helpful assistant."
print(ai.run("Explain quantum entanglement in simple terms."))
```

History is kept automatically across calls. Clear it when you need a fresh start:

```python
ai.run("My name is Alex.")
print(ai.run("What is my name?"))  # → "Your name is Alex."

ai.chat_history = []
```

### Attach files

Pass any file to a single message or make it a permanent part of every prompt:

```python
# One message
response = ai.run({
    0: {"prompt": "Summarize this document and extract all dates."},
    1: {"file": "contract.docx"},
})

# Permanent — attached to every subsequent message
ai.prompts[1] = {"file": "report.pdf"}
response = ai.run("What are the key risks mentioned?")
```

### Async

```python
import asyncio
response = asyncio.run(ai.run_async("Hello!"))
```

---

## 🔌 MCP servers

Connect external tools at runtime — the agent is automatically rebuilt with the new capabilities.

```python
import asyncio
from pathlib import Path
from ailist import AIListDemo

async def main():
    async with AIListDemo() as ai:
        ai.workspace_dir = Path(r"C:\MyProject")  # set working directory before connecting

        await ai.mcp_connects([
            {"name": "workspace"},
            {"name": "git"},
            {"name": "serena", "dirs": [r"C:\MyProject"]},  # LSP analysis, ide mode by default
        ])

        response = await ai.run_async("List all Python files and show the git status.")
        print(response)

asyncio.run(main())
```

> **`workspace_dir`** scopes all file tools to a single directory. Set it before calling `mcp_connect`. Default is `Path.cwd()`.

### Available servers

| Name | What it does | Requires |
|---|---|---|
| `workspace` | Read/write/search files, run commands and Python code — scoped to `workspace_dir` | — |
| `skills` | Browse agent skill files from a local directory | — |
| `git` | Git operations (status, log, diff, commit, push, pull, branch) | Node.js |
| `playwright` | Control a browser (navigate, click, screenshot, scrape) | Node.js |
| `memory-plus` | Persistent knowledge graph in a JSONL file | Node.js |
| `qdrant` | Semantic vector storage and search | uv, Docker |
| `searxng` | Privacy-friendly web search | Docker |
| `serena` | LSP-based semantic code analysis: find/rename/replace symbols across 30+ languages, persistent project memory | uv |
| `sympy` | Symbolic math and equation solving | `sympy` |
| `piper` | Text-to-speech, generates WAV files | `piper-tts` |
| `apprise` | Push notifications (Telegram, Discord, Email…) | `apprise` |
| `file-converter` | Convert files to text (PDF, Office, audio, video) | — |

### Adding your own server

Register any server by adding an `MCPServerDef` entry in your `AIList` subclass. Four launcher types are supported:

```python
from ailist import AIList, MCPServerDef, Provider

class MyAgent(AIList):
    def __init__(self):
        super().__init__("openai:gpt-4o", int(128000 * 0.8), Provider.OPENAI)

        # Plain Python tools — no MCP protocol, just @tool functions
        self.MCPServers["mytools"] = MCPServerDef(
            package="", launcher="builtin",
            builtin_tools=[my_tool_a, my_tool_b],
            description="My custom tools",
        )

        # stdio MCP server via npx
        self.MCPServers["mytool"] = MCPServerDef(
            package="@myorg/my-mcp-server",
            description="My npx MCP server",
        )

        # HTTP/SSE MCP server already running elsewhere
        self.MCPServers["mysse"] = MCPServerDef(
            package="", launcher="sse",
            url="http://localhost:8000/sse",
            description="My SSE MCP server",
        )

        # Docker-hosted MCP (SSE inside container)
        self.MCPServers["myservice"] = MCPServerDef(
            package="", launcher="sse",
            url="http://localhost:9000/sse",
            docker={"image": "myorg/myservice", "name": "myservice", "ports": ["9000:9000"]},
            description="My Docker MCP server",
        )
```

Per-server customisation after connecting:

```python
async with MyAgent() as ai:
    await ai.mcp_connect("mytool",
        exclude_tools=["tool_i_dont_need"],          # hide specific tools from the model
        schema_patches={"some_tool": {"param": {"description": "clearer description"}}},
    )
    # add a system prompt hint for this server's tools
    ai.system_tool_instructions += "\nWhen using mytool, prefer X over Y."
```

### workspace + serena — recommended pairing for code projects

`workspace` handles all file operations (~20 tools: read, edit, search, run commands). `serena` adds LSP-level code intelligence on top: find symbols by name across the codebase, rename with scope awareness, replace a function body without knowing its exact text, persistent per-project memory.

`serena` defaults to `context="ide"` which disables its own file tools — no conflicts:

```python
ai.workspace_dir = Path(r"C:\MyProject")
await ai.mcp_connects([
    {"name": "workspace"},
    {"name": "serena", "dirs": [r"C:\MyProject"]},
])

# Second run — project already indexed, skip onboarding:
# {"name": "serena", "dirs": [...], "modes": ["no-onboarding", "interactive", "editing"]}
```

Approximate tool counts by combination:

| Combination | Tools |
|---|---|
| `workspace` only | ~20 |
| `workspace` + `git` | ~28 |
| `workspace` + `serena` | ~27 |
| `workspace` + `serena` + `git` | ~35 |

> **Note:** `serena(context="agent")` and `workspace` together cause tool name conflicts — always use the default `context="ide"` when both are connected.

### Notifications when a long task finishes

```python
async with AIListDemo() as ai:
    await ai.mcp_connect("apprise")
    ai.apprise.urls.append("tgram://YOUR_BOT_TOKEN/YOUR_CHAT_ID")

    await ai.run_async(
        "Analyze all CSV files in C:\\Data, write a summary report, "
        "and notify me when done."
    )
```

Supports Telegram, Discord, email, Slack, and [100+ other services](https://github.com/caronc/apprise/wiki).

---

## 🤖 Multi-agent patterns

Each `AIList` instance is a plain Python object — `chat_history` is just a list. No graph framework, no special primitives. Branch, merge, hand off, run in parallel — it's all regular Python.

### Branch and rollback

Save history before a risky step; restore if the result is unsatisfactory:

```python
ai = AIListBase("ollama:llama3.2", context_limit=8192, provider=Provider.LLAMA)
ai.run("Analyze the project structure and suggest a refactoring plan.")

checkpoint = list(ai.chat_history)
result_a = ai.run("Apply the plan — rename modules and update imports.")
if "error" in result_a.lower():
    ai.chat_history = checkpoint
    result_b = ai.run("Apply only the safe parts of the plan, skip renames.")
```

### Parallel agents

Run multiple agents simultaneously, each with its own context:

```python
async def main():
    ai_docs  = AIListBase("ollama:llama3.2", context_limit=8192, provider=Provider.LLAMA)
    ai_code  = AIListBase("ollama:llama3.2", context_limit=8192, provider=Provider.LLAMA)
    ai_tests = AIListBase("ollama:llama3.2", context_limit=8192, provider=Provider.LLAMA)

    docs, code, tests = await asyncio.gather(
        ai_docs.run_async("Write documentation for this module."),
        ai_code.run_async("Refactor this module for readability."),
        ai_tests.run_async("Write unit tests for this module."),
    )
```

### Hand off between models

Draft with a fast local model, polish with a powerful one:

```python
draft_ai = AIListBase("ollama:llama3.2",           context_limit=8192,   provider=Provider.LLAMA)
final_ai = AIListBase("anthropic:claude-opus-4-6", context_limit=180000, provider=Provider.ANTHROPIC)

draft_ai.run("Write a first draft of the executive summary based on these notes.")
draft_ai.run("Expand the risks section.")
draft_ai.run("Add a conclusion.")

# Hand the full conversation to Claude for the final pass
final_ai.chat_history = list(draft_ai.chat_history)
result = final_ai.run("Polish the entire document: fix tone, tighten language, ensure consistency.")
```

### Router + specialists

A lightweight model decides which specialist handles the request:

```python
async def main():
    router = AIListBase("ollama:llama3.2", context_limit=8192, provider=Provider.LLAMA)
    router.system = "Reply with exactly one word: CODE, DATA, or SEARCH."
    decision = await router.run_async(user_query)

    async with AIListDemo() as ai:
        ai.workspace_dir = Path(r"C:\MyProject")
        if "CODE" in decision:
            await ai.mcp_connects([{"name": "workspace"}, {"name": "git"}])
        elif "DATA" in decision:
            await ai.mcp_connect("qdrant")
        elif "SEARCH" in decision:
            await ai.mcp_connect("searxng")
        result = await ai.run_async(user_query)
```

### Merge histories from parallel agents

Combine what multiple agents learned into a single context:

```python
async def main():
    ai_a = AIListBase("ollama:llama3.2", context_limit=8192, provider=Provider.LLAMA)
    ai_b = AIListBase("ollama:llama3.2", context_limit=8192, provider=Provider.LLAMA)

    await asyncio.gather(
        ai_a.run_async("Research the market opportunity for this product idea."),
        ai_b.run_async("Research the main technical risks for this product idea."),
    )

    final = AIListBase("anthropic:claude-opus-4-6", context_limit=180000, provider=Provider.ANTHROPIC)
    final.chat_history = ai_a.chat_history + ai_b.chat_history
    result = await final.run_async("Based on everything above, write an investment memo.")
```

---

## 📎 File attachments

### Supported formats

**Native (passed as-is when the provider supports it):** PNG, JPG, JPEG, WEBP, GIF, BMP, TIFF, PDF, MP3, WAV, OGG, M4A, FLAC, AAC, MP4, MOV, AVI, WEBM.

**Auto-converted to text:** DOCX, DOC, XLSX, XLS, PPTX, HTML, and all plain text formats (TXT, MD, PY, JS, TS, GO, RS, CPP, C, JAVA, JSON, CSV, YAML, SQL, and more).

### Audio and video transcription

Audio and video files are automatically transcribed via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and passed to the model as text — even when the provider has no native audio support. Video also extracts evenly spaced frames.

```python
response = ai.run({
    0: {"prompt": "Who is speaking and what are the main points?"},
    1: {"file": "meeting.mp3"},
})
```

Transcription runs on CPU by default. Enable GPU acceleration:

```python
ai.transcript_on_cuda = True
ai._converter = FileConverter(use_cuda=True)
```

---

## 💬 Prompts, system messages, and context structure

Every call to `run()` assembles the model's context in this order:

```
[systems]  →  [prompts]  →  [chat_history]  →  [current run() input]
```

**`systems`** becomes the system message. **`prompts`** are prepended to every human message as permanent context. **`chat_history`** is the conversation so far — each `run()` appends the new exchange to it automatically, so the model remembers everything until you clear it. The current `run()` input lands at the very end.

This structure matters for quality: models read the beginning (systems + prompts) and the end (current input) most reliably, and tend to under-weight the middle (history). Use `prompts` for reference material that should always be in focus — a project overview, a style guide, a PDF the model should consult on every question — rather than burying it in history.

`system` / `prompt` are shorthands for `systems[0]` / `prompts[0]`. The dictionary form accepts any integer key — entries are sent in ascending key order, which lets you compose a system prompt from independent parts without string concatenation:

```python
ai.system = "You are a code reviewer."                        # systems[0]
ai.systems[1] = {"prompt": policy_text, "tag": "policy"}     # wrapped in <policy>...</policy>
ai.systems[2] = {"file": "conventions.md"}                   # injected after policy

ai.prompt = "Focus on security issues."                       # prompts[0]
ai.prompts[1] = {"file": "codebase_overview.md"}             # always in context, always fresh
```

MCP tool instructions are appended to the system message automatically and never interfere with your entries.

---

## 📊 Context and history management

```python
ai.context_limit = 8192          # max tokens sent to the model — set to model's context window
                                  # minus expected response length to leave room for the answer
                                  # (passed to constructor; can be changed at any time)
ai.auto_summarize_history = True  # compress old history when context_limit is approached (default: False)
ai.auto_summarize_history_keep_last = 4  # keep last N exchanges verbatim when summarizing (default: 1)
ai.control_context_limit = True   # raise an error instead of overflowing silently (default: True)

# Manual summarization
ai.summarize_history(keep_last=2)

# Token estimate before sending (no API call)
_, estimated = ai.estimate_tokens_withfactor("My next question")
print(estimated, "/", ai.context_limit)  # e.g. 3241 / 8192

# Clear history
ai.chat_history = []
```

### Token metrics

All token counts are available after every `run()` / `run_async()` — useful for cost tracking and capacity planning:

```python
# How full is the context window?
print(ai.input_tokens, "/", ai.context_limit)

# Actual tokens billed this call (includes all intermediate tool calls)
print("Billed input: ", ai.full_input_tokens)
print("Billed output:", ai.full_output_tokens)

# Cumulative totals since object creation
print("Total input: ", ai.full_input_tokens_total)
print("Total output:", ai.full_output_tokens_total)
```

`input_tokens` reflects how much of the context window your request occupies. `full_input_tokens` sums all API round-trips in a single `run()` call, including intermediate tool calls — use it for cost estimation. Cumulative totals let you track spend across a long session without external instrumentation.

---

## 🧠 Thinking mode

Supported on Anthropic Claude, OpenAI o-series, and Qwen/DeepSeek reasoning models. Each provider exposes reasoning through a different API parameter — `apply_thinking_mode` handles all the differences internally, so you just pick a level:

```python
config = ai.apply_thinking_mode(thinking="high")
response = await ai.run_async("Solve this step by step.", config=config)
print(ai.last_thinking)   # the model's internal reasoning
print(response)           # the final answer
```

Available levels:

| Level | Description |
|---|---|
| `off` | Thinking disabled (or minimum, if the provider doesn't support full off) |
| `low` | Minimal reasoning budget |
| `medium` | Moderate reasoning budget |
| `high` | Extended reasoning — good default for hard tasks |
| `max` | Maximum budget the provider supports |

---

## 🔍 Logging

```python
ai.loglevel = 0   # silent (default in AIList)
ai.loglevel = 1   # timing and token stats
ai.loglevel = 2   # + message text per round (default in AIListDemo)
ai.loglevel = 3   # + full JSON history

print(ai.log)          # accumulated log
print(ai.last_message) # last log entry

ai.get_systemprompt_log()  # snapshot current system prompt to log (useful after mcp_connect)
```

---

## 🏗️ Architecture

```
AIListBase          — core engine: history, files, providers, token counting
    └── AIList      — adds MCP servers, workspace tools, policy prompts, variable substitution
            └── AIListDemo   — ready-to-run config for development (local Ollama model, loglevel=2)
```

Use `AIListBase` for scripts that don't need MCP. Use `AIList` as the base for your own production subclass. Use `AIListDemo` for quick experiments — and as a reference for how to structure your own subclass.

The recommended pattern is to subclass `AIList` once per project, baking in your model, workspace, MCP servers, and any prompt adjustments. This keeps all agent configuration in one place and makes call-site code minimal:

```python
from pathlib import Path
from ailist import AIList, Provider

class MyAgent(AIList):
    def __init__(self, tools=None):
        super().__init__(
            modelName     = "openai:gpt-4o",
            context_limit = int(128000 * 0.8),
            provider      = Provider.OPENAI,
            tools         = tools or [],
        )
        self.loglevel = 1
        self.workspace_dir = Path(r"C:\MyProject")

        # Adjust or disable built-in policy prompts:
        # self.systems[-3] = {}   # disable MINIMAL CHANGE POLICY
        # self.systems[-2] = {}   # disable VERIFICATION POLICY
        # self._prompt_workspace = "Custom workspace instructions..."

# Usage is then just:
async with MyAgent() as ai:
    await ai.mcp_connects([{"name": "workspace"}, {"name": "git"}])
    result = await ai.run_async("Refactor the auth module.")
```

### Policy prompts and workspace

`AIList` automatically injects workspace boundary and behaviour policies into the system prompt. Variables are resolved before each `run()`:

```python
from pathlib import Path

ai.workspace_dir = Path(r"C:\MyProject")   # → {WORKSPACE_DIR} in prompts
ai.attachments_dir = "attachments"         # → {ATTACHMENTS_DIR} = workspace/attachments
ai.skills_dir = "skills"                   # → {SKILLS_DIR} = workspace/skills

# Policies are stored as editable strings:
ai._prompt_workspace       # [WORKSPACE BOUNDARY POLICY]
ai._prompt_minimal_change  # [MINIMAL CHANGE POLICY]
ai._prompt_verification    # [VERIFICATION POLICY]
ai._prompt_attachments     # [ATTACHMENT WORKSPACE POLICY]

# Toggle attachment policy at runtime:
ai.set_use_attachments(True)
```

---

## 📋 Supported providers

| Model / family | Constant | Vision | Audio | PDF | Thinking |
|---|---|---|---|---|---|
| Anthropic Claude | `Provider.ANTHROPIC` | ✓ | — | ✓ | ✓ |
| OpenAI GPT / o-series | `Provider.OPENAI` | ✓ | ✓ | — | ✓ |
| Google Gemini | `Provider.GEMINI` | ✓ | ✓ | ✓ | ✓ |
| Qwen / DeepSeek | `Provider.QWEN` | ✓ | — | — | ✓ |
| Llama 3.x | `Provider.LLAMA` | ✓ | — | — | — |
| Mistral / Mixtral | `Provider.MISTRAL` | — | — | — | — |
| Text + thinking, no vision | `Provider.TEXT_REASONING` | — | — | — | ✓ |
| Text only | `Provider.TEXT_ONLY` | — | — | — | — |

For a model not listed above, pass a `ProviderCaps` instance directly instead of a `Provider` constant. `ProviderCaps` is a dataclass — set only the flags your model actually supports:

```python
from ailist import AIListBase, ProviderCaps

my_caps = ProviderCaps(
    supports_content_blocks = True,
    supports_binary         = True,
    image_format            = "image_url",
    pdf_format              = None,       # no native PDF → extract text via pypdf
    supports_audio          = False,
    supports_video          = False,
)

ai = AIListBase("ollama:my-custom-model", context_limit=32768, provider=my_caps)
```

---

## 🗂️ Skills

Skills are Markdown files that teach the agent how to perform a specific task — which tools to call, in what order, what to watch out for. They are one of the most powerful ways to improve agent reliability without touching the model or the code.

Any skill file from the community works as-is: drop it into your skills directory and the agent can read it on demand via the `skills` MCP server.

```python
async with MyAgent() as ai:
    ai.skills_dir = Path(r"C:\MyProject\skills")
    await ai.mcp_connects([
        {"name": "workspace"},
        {"name": "skills"},   # exposes list_skills / get_skill tools to the agent
    ])

    # The agent will look up the relevant skill automatically,
    # or you can load one explicitly:
    result = await ai.run_async(
        "Read the skill 'write_tests' and follow it to write tests for auth.py."
    )
```

Skills can also be pre-loaded into the system prompt as a permanent prompt block, so the agent always has them in context without a tool call:

```python
skill_text = Path(r"C:\MyProject\skills\write_tests.md").read_text()
ai.prompts[10] = {"prompt": skill_text, "tag": "skill_write_tests"}
```

---

# Installation

Requirements:
- Python 3.11 or 3.12

```cmd
pip install langchain langchain-core pydantic platformdirs
pip install langchain-ollama langchain-anthropic langchain-openai langchain-google-genai
pip install mcp tiktoken
pip install pypdf python-docx openpyxl python-pptx
pip install "beautifulsoup4[lxml]"
pip install faster-whisper opencv-python moviepy
pip install Pillow xhtml2pdf markdown
pip install piper-tts apprise sympy
pip install fastmcp
pip install uv huggingface-hub

uv tool install --python 3.12 mcp-server-qdrant
uv python install 3.13  # needed for Serena

npx -y @cyanheads/git-mcp-server --help
npx -y @executeautomation/playwright-mcp-server --help
npx -y @modelcontextprotocol/server-memory --help
```

* Node.js (for git/playwright/memory): https://nodejs.org/
* Docker (for Qdrant/SearXNG): https://www.docker.com/

Optional - pre-download, the project will download itself the first time you use it:

```cmd
python -c "from faster_whisper import WhisperModel; WhisperModel('turbo')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

| What | Where | When | Size |
|---|---|---|---|
| Whisper `turbo` (audio/video transcription) | `~/.cache/ailist/whisper` | First audio or video file attached | ~800 MB |
| Piper voice model (TTS) | `~/.cache/ailist/piper` | First `piper_synthesize` call | ~50–100 MB per voice |
| Serena language servers (LSP) | managed by uv | First symbol operation in a new language | varies |
| Qdrant embedding model | managed by uv | First `await ai.mcp_connect("qdrant")` | ~90 MB |

> All models are downloaded once and shared across projects. Override the cache location with the `AILIST_CACHE_DIR` environment variable or by setting `ai._cache_dir = Path("/my/cache")` before first use.