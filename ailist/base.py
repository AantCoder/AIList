import base64
import threading
from pathlib import Path
import asyncio
from datetime import datetime
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import messages_to_dict, messages_from_dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from dataclasses import dataclass
from typing import Callable, Any
import json
from .file_converter import FileConverter
from .piper_tts import PiperTTS

try:
    import tiktoken as _tiktoken
    _TIKTOKEN_ENC = _tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TIKTOKEN_ENC = None

# --------------------------------------------------
# Localization

# --- Errors (developer-facing, raised as exceptions) ---
ERR_UNKNOWN_EXT_FILE  = "AIList: unknown file extension '{ext}'. Specify filetype explicitly."
ERR_UNKNOWN_EXT_URL   = "AIList: unknown URL extension '{ext}'. Specify filetype explicitly."
ERR_UNKNOWN_BLOCK     = "AIList: unknown block format: {block}"
ERR_BINARY_TEXT_ONLY  = "AIList: binary file '{name}' is not available for TEXT_ONLY provider. Pass content via text_files already converted."
ERR_OUT_CONTEXT_LIMIT = "AIList: request cancelled: request tokens {tokens} exceed limit {limit}"
ERR_PDF_NO_LIB        = "AIList: failed to extract text from PDF '{name}': install pypdf (pip install pypdf)."
ERR_OFFICE_NO_LIB     = "AIList: failed to extract text from '{name}': install the required library (python-docx / openpyxl / python-pptx)."
ERR_HTML_NO_LIB       = "AIList: failed to extract text from HTML '{name}': install beautifulsoup4 (pip install beautifulsoup4 lxml)."
ERR_AUDIO_NO_LIB      = "AIList: failed to transcribe '{name}': install faster-whisper (pip install faster-whisper)."
ERR_VIDEO_NO_CV2      = "AIList: failed to extract frames from '{name}': install opencv-python (pip install opencv-python)."
ERR_VIDEO_CANT_OPEN   = "AIList: failed to open video file '{name}': file is corrupted or format is not supported."
ERR_VIDEO_NO_MOVIEPY  = "AIList: failed to extract audio from '{name}': install moviepy (pip install moviepy)."
ERR_DOC_NO_LIBREOFFICE     = "AIList: install LibreOffice to read .doc files or convert the file to .docx. File: {name}"
ERR_DOC_LIBREOFFICE_FAILED = "AIList: LibreOffice did not produce a .docx when converting '{name}'. File may be corrupted or format is not supported."

ERR_CONTENT_BLOCKS = (
    "AIList: provider does not support content block arrays (supports_content_blocks=False), "
    "but a block of type '{btype}' was encountered which cannot be passed as a string. "
    "Only text blocks and images (image_url) are allowed. "
    "For document/audio/video use a provider with supports_content_blocks=True."
)

# --- File stubs sent to the LLM when native delivery is not possible ---
FILE_BINARY_UNAVAILABLE  = "[binary file '{name}' is not available -- pass content via text_files]"
FILE_AUDIO_UNSUPPORTED   = "[audio file '{name}' ({ext}) is not supported by this provider -- pass content via text_files]"
FILE_VIDEO_UNSUPPORTED   = "[video file '{name}' ({ext}) is not supported by this provider -- pass content via text_files]"
FILE_DOC_UNSUPPORTED     = "[file '{name}' ({ext}) is not natively supported by this provider -- pass content via text_files]"
FILE_URL_STUB            = "[URL: {url}]"
FILE_PDF_URL_STUB        = "[PDF URL: {url}]"

# --- File labels prepended to content blocks sent to the LLM ---
FILE_LABEL               = "--- File: {name} ---"
FILE_PDF_TEXT_LABEL      = "--- File: {name} (extracted text from PDF) ---"
FILE_AUDIO_TRANSCRIPT    = "--- File: {name} (audio transcription) ---"
FILE_VIDEO_TRANSCRIPT    = "--- File: {name} (audio transcription from video) ---"
FILE_VIDEO_FRAMES_LABEL  = "--- File: {name} (frames from video) ---"
FILE_IMAGE_DESCRIPTION   = "--- File: {name} (image description via LLaVA) ---"

# --- Office conversion labels sent to the LLM ---
OFFICE_SHEET_LABEL       = "[Sheet: {title}]"
OFFICE_SLIDE_LABEL       = "[Slide {num}]"

# --- XML tag wrapper applied to prompt blocks ---
PROMPT_TAG_WRAP          = "<{tag}>\n{text}\n</{tag}>"

# --- Whisper word-level timestamp line format ---
AUDIO_WORD_TIMESTAMP     = "[{start:.2f}s -> {end:.2f}s] {word}"

# --- Token / performance stats written to self.log ---
MSG_TOKENS = "Context: {context}. Tokens: in {input}, out {output}. Full tokens: in {full_input}, out {full_output}. Total: in {input_total}, out {output_total}. Execution {time_prepare} sec, api {time_run} sec"
MSG_LOAD_MODEL           = "Load model: {sec:.1f} sec"
MSG_PREFILL              = "Prefill: {sec:.2f} sec"
MSG_DECODE               = "Decode: {sec:.1f} sec"
MSG_PREFILL_SPEED        = "Prefill speed: {speed:.2f} tokens/sec"
MSG_DECODE_SPEED         = "Decode speed: {speed:.2f} tokens/sec"
MSG_GENERAL_SPEED        = "General speed: {speed:.2f} tokens/sec"
MSG_GENERATION_BREAK_LENGTH  = "Generation break by token limit"
MSG_GENERATION_BREAK_TIMEOUT = "Generation break by timeout"
MSG_GENERATION_BREAK         = "Generation break"

# --- postrun / history log formatting ---
MSG_RUN_UNKNOWN_ERROR    = "Run unknown error"
LOG_PREV_MESSAGES        = "...{n} messages...\n"
LOG_END                  = "END>"
LOG_HISTORY_ATTACHMENTS  = "Attachments: {n}"
LOG_HISTORY_TOOL_CALLS   = "Tool calls: {calls}"
LOG_THINKING_START       = "<thinking>"
LOG_THINKING_END         = "</thinking>"
LOG_SET_SYSTEM_PROMPT    = "Set system prompt:\n{text}"

# --- Agent rebuild log ---
LOG_AGENT_REBUILT        = "Agent rebuilt. Model: {model}. Tools: {tools}"

# --- MCP connection log messages ---
LOG_MCP_CONNECTED_BUILTIN   = "MCP [{name}] connected (builtin)."
LOG_MCP_CONNECTED_BUILTIN_V = "MCP [{name}] connected (builtin, {desc}). Tools: {tools}"
LOG_MCP_CONNECTED_SSE       = "MCP [{name}] connected (sse: {url})."
LOG_MCP_CONNECTED_SSE_V     = "MCP [{name}] connected (sse, {desc}, {url}). Tools: {tools}"
LOG_MCP_CONNECTED           = "MCP [{name}] connected."
LOG_MCP_CONNECTED_V         = "MCP [{name}] connected ({desc}). Tools: {tools}"
LOG_MCP_DISCONNECTED        = "MCP [{name}] disconnected."
LOG_MCP_UVX_STDERR          = "MCP [{name}] (uvx) stderr:\n{stderr}"

# --- MCP error messages ---
ERR_MCP_NO_PACKAGE          = "MCPMixin requires the mcp package. Install: pip install mcp"
ERR_MCP_SSE_NO_URL          = "MCP [{name}]: launcher='sse' requires a URL. Pass url= in mcp_connect() kwargs or set url in MCPServerDef."
ERR_MCP_UVX_PROCESS_FAILED  = "MCP [{name}] (uvx): process exited with error:\n{stderr}"
ERR_MCP_TOOL_NAME_CONFLICT  = "MCP tool name conflict: '{tool}' is declared in both '{server1}' and '{server2}'. Rename the tool on one of the servers."
ERR_MCP_CONNECTS_PARTIAL    = "mcp_connects: some servers failed to connect"
ERR_MCP_CONNECTS_NO_NAME    = "mcp_connects: each entry must contain the 'name' key"

# --- Docker log messages ---
LOG_DOCKER_ALREADY_RUNNING  = "docker_start [{name}]: container already running."
LOG_DOCKER_STARTING_STOPPED = "docker_start [{name}]: container stopped, running docker start."
LOG_DOCKER_IMAGE_MISSING    = "docker_start [{name}]: image {image!r} not found locally, pulling..."
LOG_DOCKER_IMAGE_PULLED     = "docker_start [{name}]: image {image!r} pulled."
LOG_DOCKER_CMD              = "docker_start [{name}]: {cmd}"
LOG_DOCKER_STOP_CMD         = "docker_stop [{name}]: {cmd}"
LOG_DOCKER_ENSURE_RECREATE  = "docker_ensure [{name}]: env changed, recreating container."
LOG_DOCKER_SETUP_WAITING    = "docker_setup [{name}]: waiting for application readiness..."
LOG_DOCKER_SETUP_SKIPPED    = "docker_setup [{name}]: setup_check passed, setup skipped."
LOG_DOCKER_SETUP_APPLYING   = "docker_setup [{name}]: applying one-time setup ({n} commands)."
LOG_DOCKER_SETUP_CMD        = "docker_setup [{name}]: {cmd}"
LOG_DOCKER_SETUP_RESTART    = "docker_setup [{name}]: restarting container."

# --- Playwright log messages ---
LOG_PLAYWRIGHT_OPEN         = "playwright_open: browser started with profile '{profile_dir}'"
LOG_PLAYWRIGHT_OPEN_TEMP    = "playwright_open: browser started (temporary profile)"

# --- notify error messages ---
ERR_NOTIFY_NO_APPRISE       = "notify: apprise is not installed. Install: pip install apprise"
ERR_NOTIFY_CHANNEL_NOT_FOUND = "notify: channel '{channel}' not found in apprise.channels. Available: {available}"
ERR_NOTIFY_NO_URLS          = "notify: no URLs configured. Add one: ai.apprise.urls.append('tgram://...') or ai.apprise.channels['name'] = 'discord://...'"

# --- args_builder error messages ---
ERR_FASTSKILLS_NO_DIRS      = "fastskills: specify the skills folder path via dirs=[path]. Example: await ai.mcp_connect('fastskills', dirs=[r'C:\\W\\skills'])"

# --- workspace tool messages ---
ERR_EDITOR_FILE_NOT_FOUND   = "Error: file not found: {path}"
ERR_EDITOR_FILE_EXISTS      = "Error: file already exists: {path}"
ERR_EDITOR_TEXT_NOT_FOUND   = "Error: text not found in {path}"
ERR_EDITOR_MULTIPLE_MATCHES = "Error: {count} matches found in {path} -- old_str must be unique"
ERR_EDITOR_LINE_EXCEEDS     = "Error: line number {line} exceeds file length ({length})"
ERR_EDITOR_NO_BACKUP        = "Error: no backup found for {path}"
ERR_EDITOR_VIEW             = "Error viewing file: {e}"
ERR_EDITOR_CREATE           = "Error creating file: {e}"
ERR_EDITOR_REPLACE          = "Error replacing text: {e}"
ERR_EDITOR_INSERT           = "Error inserting text: {e}"
ERR_EDITOR_UNDO             = "Error undoing edit: {e}"
MSG_EDITOR_CREATED          = "Successfully created file: {path}"
MSG_EDITOR_REPLACED         = "Successfully replaced text at exactly one location."
MSG_EDITOR_INSERTED         = "Successfully inserted text at line {line}."
MSG_EDITOR_RESTORED         = "Successfully restored {path} to previous state."
ERR_WS_DIR_NOT_FOUND        = "Error: directory not found: {path}"
ERR_WS_MOVE_FAILED          = "Error moving file: {e}"
ERR_WS_MKDIR_FAILED         = "Error creating directory: {e}"
ERR_WS_READ_FAILED          = "Error reading file: {e}"
ERR_WS_WRITE_FAILED         = "Error writing file: {e}"
ERR_WS_STAT_FAILED          = "Error getting file info: {e}"
ERR_WS_GREP_FAILED          = "Error searching: {e}"
ERR_WS_PYTHON_ERROR         = "Python error:\n{e}"
ERR_WS_PYTHON_TIMEOUT       = "[error: execution timed out after {timeout}s]"
ERR_WS_SKILLS_DIR_NOT_FOUND = "[error: skills directory not found: {path}]"
MSG_WS_MOVED                = "Moved: {src} -> {dst}"
MSG_WS_DIR_CREATED          = "Created directory: {path}"
MSG_WS_FILE_WRITTEN         = "Written: {path}"

# --- Docker error messages ---
ERR_DOCKER_NO_DEF           = "docker_start: server '{name}' not found in registry or has no docker parameters."
ERR_DOCKER_NO_IMAGE         = "docker_start [{name}]: 'image' field is not set in docker parameters."
ERR_DOCKER_PULL_FAILED      = "docker_start [{name}]: failed to pull image {image!r}:\n{out}"
ERR_DOCKER_START_FAILED     = "docker_start [{name}] failed:\n{out}"
ERR_DOCKER_STOP_FAILED      = "docker_stop [{name}] failed:\n{out}"
ERR_DOCKER_NO_DEF_STOP      = "docker_stop: server '{name}' not found in registry or has no docker parameters."
ERR_DOCKER_NO_DEF_ENSURE    = "docker_ensure: server '{name}' not found in registry or has no docker parameters."
ERR_DOCKER_PORT_TIMEOUT     = "docker_start: port {port} did not open within {timeout:.0f}s. Check that the container started: docker ps"
ERR_DOCKER_HTTP_TIMEOUT     = "docker_start: service at {url} did not respond within {timeout:.0f}s."
ERR_DOCKER_SETUP_NOT_READY  = "docker_setup [{name}]: application not ready within {timeout:.0f}s. Command: {cmd}"
ERR_DOCKER_SETUP_CMD_FAILED = "docker_setup [{name}]: command failed (exit code {code}):\n{cmd}\n{out}"
ERR_DOCKER_SETUP_RESTART_FAILED = "docker_setup [{name}]: docker restart failed:\n{out}"

# --- workspace boundary messages ---
ERR_WORKSPACE_VIOLATION     = "[error: path '{path}' is outside the workspace directory '{workspace}'. Access denied.]"
LOG_WORKSPACE_SET           = "Workspace set to: {path}"
LOG_ATTACHMENTS_COPIED      = "Attachments: copied '{src}' -> '{dst}'"
LOG_ATTACHMENTS_SKIPPED_SIZE = "Attachments: skipped '{name}' -- file size {size} bytes exceeds limit {limit} bytes"

# --- subprocess_run tool messages ---
ERR_SUBPROCESS_PARSE        = "[error: failed to parse command: {e}]"
ERR_SUBPROCESS_EMPTY        = "[error: empty command]"
ERR_SUBPROCESS_NOT_FOUND    = "[error: command not found: '{cmd}']"
ERR_SUBPROCESS_TIMEOUT      = "[timeout after {timeout}s]"
ERR_SUBPROCESS_GENERAL      = "[error: {e}]"

# --- sympy_solve tool messages ---
ERR_SYMPY_NOT_INSTALLED     = "[error: sympy is not installed. Install: pip install sympy]"
ERR_SYMPY_GENERAL           = "[error: {e}]"

# --- Summarization prompts sent to the LLM ---
SUMMARIZE_BOUNDARY    = "Above are system instructions and permanent prompts.\nBelow is our conversation."
SUMMARIZE_REQUEST     = (
    "Above is the history of our conversation. "
    "Write a brief summary: what we discussed, what decisions were made, "
    "and what important facts and agreements were mentioned. "
    "Only the essence of the conversation, write concisely and without unnecessary words. "
    "Do not retell system instructions or permanent prompts."
)
SUMMARIZE_LABEL       = "[Summary of previous conversation]"

# --------------------------------------------------
# Implementation

@dataclass
class ProviderCaps:
    """
    Набор возможностей конкретного провайдера/модели.
    Вся логика _file_to_langchain и compile_combine опирается только на эти флаги,
    не проверяя имя провайдера напрямую.

    supports_binary   -- можно ли передавать файлы как base64 (изображения, PDF, аудио, видео)
    supports_system   -- есть ли роль SystemMessage (system role)
    image_format      -- формат блока для изображений:
                          "image_url"  -- {"type": "image_url", "image_url": {"url": "data:...;base64,..."}}
                          "media"      -- {"type": "media", "data": ..., "mime_type": ...}   (Gemini)
    video_format      -- формат блока для видео:
                          "video"            -- {"type": "video", "source": {"type": "base64", ...}}  (Anthropic)
                          "video_url"        -- {"type": "video_url", "video_url": {"url": "data:...;base64,..."}}  (Qwen native API)
                          "video_as_image_url" -- {"type": "image_url", "image_url": {"url": "data:video/...;base64,..."}}
                                               временный обход для Ollama/LangChain, которые не пропускают video_url;
                                               заменить на "video_url" когда появится нативная поддержка
    supports_document -- можно ли передавать PDF/docx и прочие документы как base64
    supports_audio    -- можно ли передавать аудио как base64
    supports_video    -- можно ли передавать видео как base64
    supports_url_doc  -- можно ли передавать document по URL (без скачивания)
    supports_system_files -- принимает ли SystemMessage бинарные вложения (True у Anthropic)
    supports_content_blocks -- принимает ли провайдер content сообщения как массив блоков
                              {"type": "text"/"image_url"/"document"/...}, а не как строку.
                              False -- провайдер ожидает строку; LangChain склеивает текстовые
                              блоки автоматически, но разделители (SEPARATOR) теряются,
                              а бинарные вложения между текстами передать невозможно.
                              Инвариант: supports_binary=True всегда подразумевает
                              supports_content_blocks=True. Обратное неверно -- теоретически
                              возможен провайдер принимающий массив текстовых блоков,
                              но не поддерживающий бинарные вложения.
                              Актуально и для HumanMessage, и для SystemMessage
                              (если supports_system=True).
    """
    supports_binary:   bool = False
    supports_system:   bool = True
    image_format:      str  = "image_url"   # "image_url" | "media"
    video_format:      str  = "video"        # "video" | "video_url" | "video_as_image_url"
    audio_format:      str  = "audio"        # "audio" (Anthropic) | "media" (Gemini) | "input_audio" (OpenAI) | None -> Whisper-fallback
    pdf_format:        str  = "document"     # "file" (Anthropic новый формат) | "document" (base64) | None -> текст через pypdf
    supports_document: bool = False
    supports_audio:    bool = False
    supports_video:    bool = False
    supports_url_doc:  bool = False
    supports_system_files: bool = False
    # True -- провайдер принимает бинарные вложения прямо в SystemMessage (напр. Anthropic)
    # False -- бинарные блоки из systems нужно переносить в prompts
    supports_content_blocks: bool = False
    # True -- content может быть массивом блоков {"type":...}, всегда при supports_binary=True, и обычно если модель поддерживает картнки 
    # False -- провайдер ожидает строку (см. описание выше)
    local_token_count: bool = False
    # True  -- get_num_tokens_from_messages() работает локально (без сети), результат точный.
    #          Сейчас только OpenAI: использует tiktoken локально.
    # False -- локального токенизатора нет; вызов может уйти в сеть (Anthropic)
    #          или выбросить NotImplementedError (Ollama, большинство остальных).
    #          estimate_tokens() при allow_api=False сразу использует символьный fallback.
    thinking_mode_config: dict | None = None
    # Конфигурация режима думания (extended thinking / reasoning).
    # None -- провайдер не поддерживает thinking mode (TEXT_ONLY, TEXT_ONLY_WITH_SYSTEM).
    # Структура словаря:
    #   "param":  str   -- имя параметра, который ставится напрямую в configurable
    #   "off":    any   -- значение для уровня 'off'/False (None = не добавлять параметр = выключить)
    #   "low":    any   -- значение для уровня 'low'
    #   "medium": any   -- значение для уровня 'medium'
    #   "high":   any   -- значение для уровня 'high'
    #   "max":    any   -- значение для уровня 'max' (максимальный бюджет/усилие)
    # Применяется через AIListBase.apply_thinking_mode().


class Provider:
    """
    Готовые профили провайдеров. Каждый -- экземпляр ProviderCaps.
    Можно передавать напрямую экземпляр ProviderCaps для нестандартных моделей.
    """
    ANTHROPIC = ProviderCaps(
        supports_content_blocks = True,
        supports_binary         = True,
        supports_system         = True,
        image_format            = "image_url",
        audio_format            = None,        # нет нативной поддержки -> Whisper-fallback
        pdf_format              = "file",         # {"type":"file","file":{"filename":...,"file_data":"data:application/pdf;base64,..."}}
        supports_document       = True,
        supports_audio          = False,
        supports_video          = False,
        supports_url_doc        = True,
        supports_system_files   = True,
        thinking_mode_config    = {
            "param":  "thinking",
            "off":    None,                                                        # не добавлять параметр = выключить
            "low":    {"type": "enabled", "budget_tokens": 1024},                  # минимальный бюджет
            "medium": {"type": "enabled", "budget_tokens": 2048},
            "high":   {"type": "enabled", "budget_tokens": 4096},
            "max":    {"type": "enabled", "budget_tokens": 8192},                  # максимальный бюджет
        },
    )
    OPENAI = ProviderCaps(  #GPT/o-серия
        supports_content_blocks = True,
        supports_binary         = True,
        supports_system         = True,
        image_format            = "image_url",
        audio_format            = "input_audio", # {"type":"input_audio","input_audio":{"data":...,"format":"mp3"}}
        pdf_format              = None,           # нет нативной поддержки -> текст через pypdf
        supports_document       = False,
        supports_audio          = False,          # True -- только для gpt-4o-audio-preview; WAV и MP3
        supports_video          = False,
        supports_url_doc        = False,
        local_token_count       = True,
        thinking_mode_config    = {
            "param":  "reasoning_effort",  # реальное имя параметра LangChain для OpenAI
            "off":    "low",               # нет полного отключения -> минимальный reasoning
            "low":    "low",
            "medium": "medium",
            "high":   "high",
            "max":    "high",              # нет нативного max -> fallback на high
        },
    )
    TEXT_REASONING = ProviderCaps(
        # Text-only model with thinking mode support (no vision, no binary attachments).
        # Examples: gpt-oss, DeepSeek-R1, older Qwen without vision, similar local reasoning models.
        supports_content_blocks = True,
        supports_binary         = False,
        supports_system         = True,
        audio_format            = None,    # нет поддержки -> Whisper-fallback
        pdf_format              = None,    # нет поддержки -> текст через pypdf
        supports_document       = False,
        supports_audio          = False,
        supports_video          = False,   # текстовая модель
        supports_url_doc        = False,
        supports_system_files   = False,
        local_token_count       = False,
        thinking_mode_config    = {
            "param":  "reasoning",
            "off":    "low",         # полного отключения нет -> минимальный уровень
            "low":    "low",
            "medium": "medium",
            "high":   "high",
            "max":    "high",        # три уровня, выше high нет
        },
    )
    GEMINI = ProviderCaps(
        supports_content_blocks = True,
        supports_binary         = True,
        supports_system         = True,
        image_format            = "media",
        audio_format            = "media",       # {"type":"media","data":...,"mime_type":...}
        pdf_format              = "document",    # base64 document
        supports_document       = True,
        supports_audio          = True,
        supports_video          = True,
        supports_url_doc        = True,
        thinking_mode_config    = {
            "param":  "thinking_level",
            "off":    None,       # нет полного off -> не добавлять параметр
            "low":    "low",
            "medium": "medium",
            "high":   "high",
            "max":    "high",     # нет нативного max -> fallback на high
        },
    )
    QWEN = ProviderCaps(
        # Also suitable for DeepSeek models running via Ollama (same capability set).
        supports_content_blocks = True,
        supports_binary         = True,
        supports_system         = True,
        image_format            = "image_url",
        video_format            = "video_url",  # revert "video_as_image_url" to "video_url" when Ollama/LangChain support video_url natively
        audio_format            = None,          # нет нативной поддержки -> Whisper-fallback
        pdf_format              = None,          # нет нативной поддержки -> текст через pypdf
        supports_document       = False,
        supports_audio          = False,
        supports_video          = True,
        supports_url_doc        = True,
        supports_system_files   = False,
        local_token_count       = False,
        thinking_mode_config    = {
            "param":  "reasoning",  # ChatOllama: reasoning=True -> additional_kwargs["reasoning_content"]
            "off":    False,
            "low":    False,        # нет градаций -> low = off
            "medium": True,
            "high":   True,
            "max":    True,         # нет градаций -> max = high
        },
    )
    LLAMA = ProviderCaps(
        # Meta Llama 3.x via Ollama. Supports vision (image_url) and content blocks,
        # but no native audio, video, or PDF -- these fall back to Whisper / pypdf.
        # No thinking mode support.
        supports_content_blocks = True,
        supports_binary         = True,
        supports_system         = True,
        image_format            = "image_url",
        audio_format            = None,          # no native support -> Whisper-fallback
        pdf_format              = None,          # no native support -> text via pypdf
        supports_document       = False,
        supports_audio          = False,
        supports_video          = False,
        supports_url_doc        = False,
        supports_system_files   = False,
        local_token_count       = False,
        thinking_mode_config    = None,          # no thinking mode
    )
    MISTRAL = ProviderCaps(
        # Mistral / Mixtral / Codestral via Ollama or Mistral API.
        # Text-only: no vision, no audio, no PDF. Supports system role and content blocks.
        # No thinking mode support.
        supports_content_blocks = True,
        supports_binary         = False,         # text-only, no binary attachments
        supports_system         = True,
        image_format            = "image_url",
        audio_format            = None,          # no native support -> Whisper-fallback
        pdf_format              = None,          # no native support -> text via pypdf
        supports_document       = False,
        supports_audio          = False,
        supports_video          = False,
        supports_url_doc        = False,
        supports_system_files   = False,
        local_token_count       = False,
        thinking_mode_config    = None,          # no thinking mode
    )
    TEXT_ONLY_WITH_SYSTEM = ProviderCaps(
        supports_content_blocks = False,
        supports_binary         = False,
        supports_system         = True,
        image_format            = "image_url",
        audio_format            = None,
        pdf_format              = None,
    )
    TEXT_ONLY = ProviderCaps(
        supports_content_blocks = False,
        supports_binary         = False,
        supports_system         = False,
        image_format            = "image_url",
        audio_format            = None,
        pdf_format              = None,
    )

SEPARATOR = "\n\n---\n\n"

# Результат compile_block: блоки данных + флаг встраивания.
# is_inline=True  -> текстовое содержимое, объединяется с другими текстами через SEPARATOR
# is_inline=False -> вложение (файл, url), ставится в file_blocks отдельным блоком(ами)
@dataclass
class CompiledBlock:
    blocks:    list        # список LangChain-совместимых dict-блоков
    is_inline: bool        # способ встраивания в compile_section. Внимание is_inline=True, при количестве blocks>1, блоки будут объеденены со стандартным SEPARATOR (указывает ЛЛМ на отсутствие связи между блоками)
    metadata:  dict = None # метаданные блока, пробрасываются в итоговое сообщение через compile_section
    
# --------------------------------------------------   
    
# Определяет как трактовать файл по расширению
EXT_TYPE = {
    # Текст -- читаем как строку, вставляем в text-блок
    "txt": "text", "md": "text", "py": "text", "json": "text", "sql": "text",
    "csv": "text", "xml": "text", "yaml": "text", "yml": "text",
    "js": "text", "ts": "text", "css": "text", "sh": "text",
    "bat": "text", "log": "text", "ini": "text", "toml": "text", "rst": "text",
    "svg": "text", "jsx": "text", "tsx": "text", "vue": "text", "go": "text",
    "rs": "text", "cpp": "text", "c": "text", "h": "text", "java": "text",
    "rb": "text", "php": "text", "kt": "text", "swift": "text", "r": "text",
    # Изображения -- base64
    "png": "image_url", "jpg": "image_url", "jpeg": "image_url",
    "gif": "image_url", "webp": "image_url", "bmp": "image_url",
    "tiff": "image_url", "tif": "image_url",
    # PDF -- отдельный тип: нативный или через извлечение текста в зависимости от провайдера
    "pdf": "pdf",
    # Office-документы -- конвертируем в текст (нет нативной поддержки ни у кого из провайдеров)
    "docx": "office", "doc": "office",
    "xls": "office", "xlsx": "office",
    "pptx": "office",
    # HTML -- конвертируем в текст через BeautifulSoup
    "html": "html", "htm": "html",
    # Аудио -- нативно или через Whisper
    "mp3": "audio", "wav": "audio", "ogg": "audio", "m4a": "audio",
    "flac": "audio", "aac": "audio",
    # Видео -- нативно или через кадры+аудио
    "mp4": "video", "mov": "video", "avi": "video", "webm": "video",
}

# MIME-типы для base64-файлов
MIME_MAP = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
    "tiff": "image/tiff", "tif": "image/tiff",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg", "m4a": "audio/mp4",
    "flac": "audio/flac", "aac": "audio/aac",
    "mp4": "video/mp4", "mov": "video/quicktime", "avi": "video/x-msvideo", "webm": "video/webm",
}

class AIListBase:
    """
    Класс для запуска запросов ЛЛМ через LangChain
    Принимает блоки в постоянные промты в prompts, systems - это словари, с числовыми ключами для сортировки.
    Аргумент prt в run или run_async  - передается последним как активный вопрос к ЛЛМ, может быть строкой промпта, или таким же словарём как prompts или systems
    
    Блоки для словарей prompts, systems и prt могут быть:
    { "type":... }                          -- готовый LangChain-блок, без дополнительной обработки
    { "prompt":"content" }                   -- текстовое содержимое
    { "prompt":"content", "tag":"name" }     -- текст, обёрнутый в XML-тег <name>...</name>
    { "url":"URL" }                         -- вложение по URL без скачивания. Поддержка зависит от провайдера (см. ProviderCaps.supports_url_doc)
    { "url":"URL", "filetype":"pdf" }       -- явное указание типа если расширение отсутствует в URL
    { "file":"filename" }                   -- внедрение локального файл отдельной записью (не в тексте промта); тип определяется по расширению
                                              Текстовые файлы (py, md, json...) -- встраиваются как текст, но не сливаются с промтом.
                                              Бинарные (png, pdf...) -- передаются как base64; провайдер должен поддерживать
    { "file":"filename", "filetype":"png" } -- явное указание типа файла
    { "text_file":"filename" }              -- читает файл как текст автоматически (utf-8, errors=replace) и встраивает его в текст промта после него.
                                              Файл будет обрамлён разделителем, и будет указано имя файла.
    { "text_file":"filename", "content":... } -- предконвертированный текст (например Excel->markdown), файл не читается, а берётся сразу содержимое content
    { ... "metadata": {} }                  -- метаданные будут переданы в LangChain-блок

    Порядок блоков внутри словарей (prompts/systems/prt) определяется числовыми ключами (меньший -- раньше).
    prompt и system -- частные случаи для удобства, записываются в prompts[0] и systems[0], и могут быть просто текстом, а не блоком
    Перед передачей бинарных блоков рекомендуется проверить поддержку через can_accept(block).
    """

    def __init__(self, modelName, context_limit, provider: Provider = Provider.TEXT_ONLY, tools = [], context_schema=None):
        """
        Параметры:
            modelName      -- строка модели в формате LangChain init_chat_model,
                             например "ollama:qwen3:8b", "anthropic:claude-opus-4-5", "openai:gpt-4o".
            context_limit  -- максимальное число токенов контекста модели.
                             Рекомендуется брать из настроек модели и уменьшать на запас под ответ
                             (например int(num_ctx * 0.8)). Используется для контроля лимита
                             перед запросом и автосуммаризации истории.
            provider       -- профиль возможностей провайдера (экземпляр ProviderCaps).
                             Используйте готовые профили из класса Provider:
                             Provider.ANTHROPIC, Provider.OPENAI, Provider.GEMINI,
                             Provider.QWEN, Provider.TEXT_REASONING, Provider.TEXT_ONLY.
                             Можно передать собственный ProviderCaps для нестандартных моделей.
            tools          -- список LangChain-инструментов (@tool) доступных агенту.
                             Передайте [] если инструменты не нужны.
            context_schema -- схема контекста для агента (опционально, передаётся в create_agent).
        """
        # Единый замок для run/run_async и rebuild агента (_mcp_rebuild_agent).
        # threading.Lock выбран намеренно: он работает поперёк потоков и event loop-ов,
        # что необходимо для run() -- он запускает run_async() в отдельном потоке
        # через concurrent.futures. asyncio.Lock привязан к конкретному loop
        # и не защитил бы от одновременного вызова run() и run_async() из разных потоков.
        # В async-коде захватывается через asyncio.to_thread(self._lock.acquire)
        # -- это не блокирует event loop, а переносит blocking-acquire в threadpool.
        self._lock = threading.Lock()
        self.context_limit = context_limit
        self.provider = provider
        self.chat_history = []
        self.drop_history = []
        self.agent = {}
        self.debug_run_result = {}
    
        self.prompts = {}
        self.prompt = ""
        self.systems = {}
        self.system = ""
        self.system_tool_instructions = ""
        # Автоматически собирается при подключении MCP-серверов (_mcp_open):
        # читается из InitializeResult.instructions, list_prompts/get_prompt,
        # вызова system_prompt_tool и статичного system_prompt из MCPServerDef.
        # при каждой пересборке агента (_mcp_rebuild_agent).
        # Объединяется с self.system в systems[0] -- см. prepare().
        # systems[0] зарезервирован -- всё остальное пространство systems в вашем распоряжении.

        # -- Cache directory -----------------------------------------------
        # Папка для кэша библиотеки: модели Whisper, модели Piper и т.д.
        # По умолчанию -- стандартный пользовательский кэш ОС (через platformdirs).
        # Переменная окружения AILIST_CACHE_DIR переопределяет путь глобально.
        # Можно задать явно: ai._cache_dir = Path("/my/cache")
        # Менять нужно до первого использования Whisper или TTS.
        import os as _os
        _cache_env = _os.environ.get("AILIST_CACHE_DIR")
        if _cache_env:
            self._cache_dir: Path = Path(_cache_env)
        else:
            try:
                from platformdirs import user_cache_dir as _user_cache_dir
                self._cache_dir = Path(_user_cache_dir("ailist"))
            except ImportError:
                # platformdirs не установлен -- fallback на ~/.cache/ailist
                self._cache_dir = Path.home() / ".cache" / "ailist"

        # -- Workspace directory -------------------------------------------
        # Рабочая директория -- все инструменты ограничены ею.
        # Менять через set_workspace(path). None -- без ограничений.
        self.workspace_dir: Path | None = Path.cwd()

        # -- Attachments buffer --------------------------------------------
        # use_attachments=True -- при обработке каждого file/text_file блока
        #   копировать исходный файл в get_attachments_dir() перед конвертацией.
        # attachments_dir -- имя поддиректории внутри workspace_dir.
        # attachments_max_file_size -- файлы крупнее этого порога не копируются (байт).
        self.use_attachments: bool = False
        self.attachments_dir: str  = "attachments"
        self.attachments_max_file_size: int = 20 * 1024 * 1024  # 20 МБ

        self.transcript_on_cuda = False             # Включите, если есть поддержка и если не мешает занимаемая видеопамять (займёт примерно 3-4 ГБ VRAM)
        self._converter = FileConverter(            # Конвертер файлов; use_cuda синхронизирован с transcript_on_cuda.
            use_cuda=self.transcript_on_cuda,       # После смены transcript_on_cuda пересоздайте: self._converter = FileConverter(use_cuda=True)
        )                                           # Whisper-модели кэшируются в стандартный ~/.cache/huggingface/hub/
        self.control_context_limit = True           # Отмена запроса при выходе контекста за лимит context_limit
        self.auto_summarize_history = False         # Попытаться сжать историю summarize_history до отмены запроса при выходе контекста за лимит context_limit
        self.auto_summarize_history_keep_last = 1   # Аргумент keep_last в summarize_history при автозапуске
        self.input_tokens:        int = 0  # токены входящего контекста нашего запроса (input_tokens первого AI-сообщения раунда); отражает сколько контекста занято
        self.output_tokens:       int = 0  # токены финального ответа модели (output_tokens последнего AI-сообщения раунда); отражает размер конечного ответа
        self.full_input_tokens:   int = 0  # суммарные входящие токены по всем AI-вызовам раунда (включая промежуточные при использовании инструментов); для оценки стоимости API
        self.full_output_tokens:  int = 0  # суммарные исходящие токены по всем AI-вызовам раунда (включая промежуточные ответы с tool_call); для оценки стоимости API
        self.input_tokens_total:  int = 0  # накопитель input_tokens за всё время работы объекта
        self.output_tokens_total: int = 0  # накопитель output_tokens за всё время работы объекта
        self.full_input_tokens_total:  int = 0  # накопитель full_input_tokens за всё время работы объекта
        self.full_output_tokens_total: int = 0  # накопитель full_output_tokens за всё время работы объекта
        self.loglevel = 0  # Уровень логирования в self.log:
        #   0 -- логирование выключено
        #   1 -- ошибки, время выполнения и статистика токенов
        #   2 -- то же + текст сообщений раунда (human/ai/tool)
        #   3 -- то же + полный JSON истории (весь контекст без статичных prompts/systems)
        self.last_message: str = ""
        self.last_thinking: str | None = None  # рассуждения модели из последнего запроса (None если недоступны или thinking mode не использовался)
        self.log_thinking: bool = False  # True -- собирать thinking из ВСЕХ AI-сообщений раунда
        #          и вставлять в history_tostr обёрнутым в <thinking>...</thinking>
        self.log:     str = ""
        self.runstart = datetime.now()
        self.runinvoke = datetime.now()
        self.runfinish = datetime.now()
        self.last_load_sec = 0
        self.last_prefill_sec = 0
        self.last_decode_sec = 0
        self.last_general_sec = 0
        self.last_prefill_speed = 0 # скорость загрузки контекста токенов/сек по данным из модели, если доступна
        self.last_decode_speed = 0 # скорость генерации ответа токенов/сек по данным из модели, если доступна
        self.last_general_speed = 0 # крайне примерная скорость из общего времени запроса / сумма входящих и исходящих токенов

        # Кэш токенов статичных промптов (systems + prompts).
        # Сбрасывается автоматически при изменении словарей (через _dict_snapshot),
        # или явно через update().
        self._tokens_systems:               int = 0  # последний подсчёт токенов для systems
        self._tokens_prompts:                int = 0  # последний подсчёт токенов для prompts
        self._tokens_chat_history:          int = 0  # последний подсчёт токенов для chat_history
        self._tokens_prt:                   int = 0  # последний подсчёт токенов для аргумента prt в estimate_tokens_prt
        self._tokens_snapshot_systems:      int = 0  # snapshot systems на момент последнего подсчёта
        self._tokens_snapshot_prompts:       int = 0  # snapshot prompts на момент последнего подсчёта
        self._tokens_snapshot_chat_history: int = 0  # snapshot chat_history на момент последнего подсчёта
        self._last_chat_history_len: int = 0 # служебное значение - длинна истории до запроса
        self._token_correction_factor = 1   # коэф. корректировки оценочных токенов к реальным из ответа. Автоматически вычесляется после запроса

        self._model_name = modelName
        self._model = init_chat_model(
            modelName,
            configurable_fields="any",
        )
        self._context_schema = context_schema
        self._rebuild_agent(tools)

    def _rebuild_agent(self, tools: list):
        """
        Пересоздаёт self.agent с заданным списком инструментов.
        Единственное место где вызывается create_agent -- гарантирует что
        _context_schema и прочие параметры агента всегда применяются.
        """
        agent_kwargs = dict(
            model=self._model,
            tools=tools,
        )
        if self._context_schema is not None:
            agent_kwargs["context_schema"] = self._context_schema
        self.agent = create_agent(**agent_kwargs)
        if self.loglevel > 1:
            self.append_log(
                LOG_AGENT_REBUILT.format(
                    model=self._model_name,
                    tools=[t.name if hasattr(t, 'name') else str(t) for t in tools],
                )
            )

    # -------------------------------------------------------------------------
    # Thinking mode
    # -------------------------------------------------------------------------

    def apply_thinking_mode(
        self,
        config: dict | None = None,
        thinking: str | bool | None = None,
    ) -> dict:
        """
        Добавляет параметр thinking mode в config["configurable"] в зависимости
        от текущего провайдера (self.provider).

        Параметры:
            config   -- существующий config (можно None или {}).
            thinking -- None  (ничего не делать),
                       False (уровень 'off'),
                       True  (уровень 'high'),
                       str: 'off', 'low', 'medium', 'high', 'max'.

        Если conf["off"] равно None -- параметр не добавляется в configurable совсем
        (провайдер интерпретирует отсутствие параметра как выключение thinking mode).

        Возвращает тот же объект config (мутирует in-place).
        """
        if thinking is None:
            return config or {}
        if config is None:
            config = {}
        if not isinstance(config.get("configurable"), dict):
            config["configurable"] = {}
        cfg = config["configurable"]

        caps = self.provider
        # Провайдер не поддерживает thinking mode
        if caps.thinking_mode_config is None:
            return config

        conf = caps.thinking_mode_config

        # Булево -> строковый уровень
        if isinstance(thinking, bool):
            level = "high" if thinking else "off"
        else:
            level = thinking.strip().lower()

        # Получаем значение; неизвестный уровень -> не добавляем параметр совсем
        if level in conf:
            value = conf[level]
        else:
            value = None

        # None = выключить: не добавляем параметр совсем
        if value is None:
            return config

        param = conf["param"]
        cfg[param] = value
        return config

    def extract_thinking(self, msg) -> tuple[str | None, str]:
        """
        Извлекает рассуждения и финальный ответ из AI-сообщения.

        Поддерживаемые провайдеры:
          Ollama (Qwen, gpt-oss и др.) -- рассуждения в additional_kwargs["reasoning_content"],
                                         content уже чистый без тегов.
          Anthropic                    -- content является списком блоков;
                                         блок {"type":"thinking"} содержит рассуждения,
                                         блок {"type":"text"} содержит финальный ответ.
          Gemini                       -- блок {"type":"thinking"} в content (аналогично Anthropic)
                                         или поле thought в response_metadata candidates.
          Прочие                       -- рассуждения недоступны, возвращает (None, content).

        Возвращает: (thinking: str | None, answer: str)
          thinking -- текст рассуждений или None если недоступны / не было thinking mode
          answer   -- финальный ответ модели (чистый текст без тегов рассуждений)
        """
        content = msg.content
        additional = getattr(msg, "additional_kwargs", {}) or {}
        meta = getattr(msg, "response_metadata", {}) or {}

        # --- Ollama: reasoning_content в additional_kwargs ---
        # ChatOllama с reasoning=True кладёт рассуждения сюда,
        # а content уже содержит только финальный ответ.
        if "reasoning_content" in additional:
            thinking = additional["reasoning_content"] or None
            answer = content if isinstance(content, str) else ""
            return thinking, answer

        # --- Anthropic / Gemini: content как список блоков ---
        if isinstance(content, list):
            thinking_parts = []
            answer_parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "thinking":
                    thinking_parts.append(block.get("thinking") or block.get("text") or "")
                elif btype in ("text", "thought"):
                    # "thought"=True в Gemini-блоках означает рассуждение
                    if block.get("thought"):
                        thinking_parts.append(block.get("text") or "")
                    else:
                        answer_parts.append(block.get("text") or "")
            # Gemini fallback: candidates в response_metadata
            if not thinking_parts:
                for candidate in meta.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if part.get("thought"):
                            thinking_parts.append(part.get("text") or "")
            thinking = "\n".join(thinking_parts).strip() or None
            answer   = "\n".join(answer_parts).strip()
            return thinking, answer

        # --- Прочие: рассуждения недоступны ---
        return None, content if isinstance(content, str) else ""

    # -------------------------------------------------------------------------
    # Вспомогательные функции формирования блоков
    # -------------------------------------------------------------------------

    def _make_text_block(self, text):
        """Оборачивает строку в стандартный LangChain text-блок."""
        return {"type": "text", "text": text}

    def _make_prompt_text(self, block):
        """Извлекает текст из блока, оборачивает в XML-тег если задан."""
        text = block["prompt"]
        if "tag" in block:
            tag = block["tag"]
            text = PROMPT_TAG_WRAP.format(tag=tag, text=text)
        return text

    def _is_nontext_block(self, block: dict) -> bool:
        """
        Возвращает True если блок является не-текстовым вложением
        (file или url с не-текстовым расширением).
        Такие блоки нельзя передать в SystemMessage -- их нужно перебрасывать в prompts.
        """
        if "file" in block:
            ext = block.get("filetype") or Path(block["file"]).name.rsplit(".", 1)[-1].lower()
            return EXT_TYPE.get(ext) != "text"
        if "url" in block:
            ext = block.get("filetype") or block["url"].rsplit(".", 1)[-1].lower()
            return EXT_TYPE.get(ext) != "text"
        if "type" in block:
            # Готовый LangChain-блок -- если не text, считаем вложением
            return block.get("type") != "text"
        # text_file, prompt -- всегда текст
        return False

    # -------------------------------------------------------------------------
    # Workspace и attachments
    # -------------------------------------------------------------------------

    def _check_workspace_path(self, path: str) -> str | None:
        """
        Проверяет что path находится внутри self.workspace_dir.
        Возвращает None если всё в порядке, или строку с ошибкой если нарушение.
        Если workspace_dir не задан (None) -- всегда возвращает None (без ограничений).
        """
        if self.workspace_dir is None:
            return None
        try:
            resolved = Path(path).resolve()
            if not resolved.is_relative_to(self.workspace_dir):
                return ERR_WORKSPACE_VIOLATION.format(
                    path=path, workspace=self.workspace_dir
                )
        except Exception:
            return ERR_WORKSPACE_VIOLATION.format(
                path=path, workspace=self.workspace_dir
            )
        return None

    def set_workspace(self, path: str) -> None:
        """
        Устанавливает рабочую директорию workspace_dir.
        Путь резолвится в абсолютный. Записывает в лог.
        """
        self.workspace_dir = Path(path).resolve()
        self.append_log(LOG_WORKSPACE_SET.format(path=self.workspace_dir))

    def get_attachments_dir(self) -> Path | None:
        """
        Возвращает абсолютный путь к буферной директории вложений.

        Если attachments_dir -- абсолютный путь, возвращает его напрямую.
        Если относительный -- резолвит относительно workspace_dir.
        None если workspace_dir не задан и attachments_dir относительный.
        None если use_attachments=False.

        Директория не создаётся здесь -- только при реальном использовании
        (в _copy_to_attachments и prepare()).
        """
        if not self.use_attachments:
            return None
        return self._resolve_dir(self.attachments_dir)

    def _resolve_dir(self, dir_attr: str) -> Path | None:
        """
        Резолвит строку директории в абсолютный Path.
        Абсолютный путь (Unix или Windows) возвращается как есть;
        относительный -- от workspace_dir.
        None если workspace_dir не задан и путь относительный.
        Используется внутри get_attachments_dir, get_skills_dir и prepare().
        """
        from pathlib import PureWindowsPath
        p = Path(dir_attr)
        if p.is_absolute() or PureWindowsPath(dir_attr).is_absolute():
            return Path(dir_attr)
        if self.workspace_dir is None:
            return None
        return self.workspace_dir / dir_attr

    def _copy_to_attachments(self, src_path: str) -> None:
        """
        Копирует файл src_path в get_attachments_dir().

        Логика именования при коллизии имён:
          Если файл с таким именем уже существует И его resolved-путь совпадает
          с src_path -- пропускаем (файл уже в буфере).
          Если файл с таким именем уже существует И путь другой -- переименовываем:
            "name.ext"  ->  "name (folder1 subfolder2).ext"
          где folder1/subfolder2 -- части пути src_path, усечённые до разумной длины.

        Файлы крупнее attachments_max_file_size не копируются.
        Молча пропускает если use_attachments=False или workspace_dir=None.
        """
        dst_dir = self.get_attachments_dir()  # None если use_attachments=False или нет workspace
        if dst_dir is None:
            return

        src = Path(src_path).resolve()

        # Если файл уже лежит в буфере -- не копируем
        try:
            if src.is_relative_to(dst_dir):
                return
        except Exception:
            pass

        if not src.exists():
            return

        # Проверка размера
        try:
            size = src.stat().st_size
        except Exception:
            return
        if size > self.attachments_max_file_size:
            if self.loglevel > 0:
                self.append_log(LOG_ATTACHMENTS_SKIPPED_SIZE.format(
                    name=src.name, size=size, limit=self.attachments_max_file_size
                ))
            return

        # Создаём директорию непосредственно перед копированием
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        # Определяем имя файла в буфере
        stem = src.stem
        suffix = src.suffix
        dst = dst_dir / src.name

        if dst.exists() and dst.resolve() != src:
            # Коллизия: другой файл с тем же именем.
            # Строим суффикс из частей пути: берём до 3 частей родительского пути,
            # заменяем разделители пробелами, обрезаем до 60 символов.
            parts = list(src.parent.parts)
            # Убираем корень диска/слеш и слишком длинные части
            path_tag = " ".join(p for p in parts if p not in ("\\", "/", ""))
            if len(path_tag) > 60:
                path_tag = path_tag[-60:]
            new_name = f"{stem} ({path_tag}){suffix}"
            dst = dst_dir / new_name

        import shutil as _shutil
        try:
            _shutil.copy2(str(src), str(dst))
            if self.loglevel > 1:
                self.append_log(LOG_ATTACHMENTS_COPIED.format(src=src, dst=dst))
        except Exception:
            pass  # копирование не должно ломать основной поток

    # -------------------------------------------------------------------------
    # Конвертеры -- делегируют в self._converter (FileConverter).
    # Сигнатуры сохранены для полной обратной совместимости.
    # Логика, зависимости и Whisper-кэш живут в file_converter.py.
    # -------------------------------------------------------------------------

    def _pdf_to_text(self, path) -> str:
        return self._converter.pdf_to_text(path)

    def _office_to_text(self, path, ext) -> str:
        return self._converter.office_to_text(path, ext)

    def _html_to_text(self, path) -> str:
        return self._converter.html_to_text(path)

    def _audio_to_transcript(
        self,
        path,
        *,
        model_size:       str          = "turbo",
        language:         str | None   = None,
        word_timestamps:  bool         = False,
        vad_filter:       bool         = True,
        initial_prompt:   str | None   = None,
    ) -> str:
        return self._converter.audio_to_transcript(
            path,
            model_size=model_size,
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt,
        )

    def _video_extract_frames_b64(self, path, num_frames: int = 5) -> list[str]:
        return self._converter.video_extract_frames_b64(path, num_frames=num_frames)

    def _video_extract_audio_transcript(self, path) -> str | None:
        return self._converter.video_extract_audio_transcript(path)

    def _image_to_description(self, path) -> str:
        """Описывает изображение текстом через LLaVA (Ollama). Для не-vision моделей."""
        return self._converter.describe_image(path)


    # -------------------------------------------------------------------------
    # Форматирование файлов в LangChain-блоки
    # -------------------------------------------------------------------------

    def _file_to_langchain(self, path, ext):
        """
        Читает локальный файл и возвращает список LangChain-совместимых dict-блоков.
        Возвращает list[dict] -- один или несколько блоков (например кадры видео + текст).

        Первым блоком всегда идёт текстовая метка с именем файла и способом доставки:
          FILE_LABEL              -- текст/html/office/pdf нативно/аудио нативно/видео нативно/изображение
          FILE_AUDIO_TRANSCRIPT   -- аудио транскрибировано через Whisper
          FILE_VIDEO_TRANSCRIPT   -- видео: аудиодорожка транскрибирована (+ затем кадры)
          FILE_VIDEO_FRAMES_LABEL -- видео: кадры (следует после FILE_VIDEO_TRANSCRIPT если есть)

        Логика выбора формата:
          text    -- читается как UTF-8 строка -> text-блок
          html    -- конвертируется в текст через BeautifulSoup -> text-блок
          office  -- конвертируется в текст через python-docx/openpyxl/python-pptx -> text-блок
          pdf     -- нативно (Anthropic: "file"-блок; Gemini/document: base64) или текст через pypdf
          image   -- base64 в формате провайдера (image_url / media)
          audio   -- нативно (Anthropic: "audio"; Gemini: "media"; OpenAI: "input_audio")
                    или транскрипция через Whisper -> text-блок
          video   -- нативно (Anthropic/Gemini) или кадры + транскрипция аудио

        Выбрасывает исключение если расширение неизвестно или файл не найден.
        """
        kind = EXT_TYPE.get(ext)
        if kind is None:
            raise ValueError(ERR_UNKNOWN_EXT_FILE.format(ext=ext))

        name = Path(path).name
        caps = self.provider  # ProviderCaps
        label = self._make_text_block(FILE_LABEL.format(name=name))

        # --- Текст: читаем как строку ---
        if kind == "text":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return [label, self._make_text_block(content)]

        # --- HTML: конвертируем в текст ---
        if kind == "html":
            text = self._html_to_text(path)
            return [label, self._make_text_block(text)]

        # --- Office: конвертируем в текст ---
        if kind == "office":
            text = self._office_to_text(path, ext)
            return [label, self._make_text_block(text)]

        # --- PDF ---
        # Обрабатываем до проверки supports_binary: при pdf_format=None результат
        # всегда текстовый (через pypdf), бинарный файл модели не передаётся.
        if kind == "pdf":
            pdf_fmt = caps.pdf_format
            if pdf_fmt is None:
                # Нет нативной поддержки -- извлекаем текст, supports_binary не нужен
                text = self._pdf_to_text(path)
                pdf_label = self._make_text_block(FILE_PDF_TEXT_LABEL.format(name=name))
                return [pdf_label, self._make_text_block(text)]
            # Нативный формат требует binary -- проверяем поддержку
            if not caps.supports_binary:
                return [self._make_text_block(FILE_BINARY_UNAVAILABLE.format(name=name))]
            data = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
            mime = MIME_MAP.get(ext, "application/octet-stream")
            if pdf_fmt == "file":
                # Anthropic новый формат (предпочтительный): модель видит PDF нативно
                return [label, {"type": "file", "file": {
                    "filename": name,
                    "file_data": f"data:{mime};base64,{data}",
                }}]
            # pdf_fmt == "document": Gemini и старый Anthropic
            return [label, {"type": "document", "source": {
                "type": "base64", "media_type": mime, "data": data,
            }}]

        # --- Аудио ---
        # audio_format=None означает отсутствие нативной поддержки -> Whisper-транскрипция.
        # Конвертация не требует supports_binary, поэтому обрабатываем до этой проверки.
        if kind == "audio":
            audio_fmt = caps.audio_format
            if audio_fmt is None:
                # Нет нативной поддержки -- транскрибируем через Whisper
                transcript = self._audio_to_transcript(path)
                audio_label = self._make_text_block(FILE_AUDIO_TRANSCRIPT.format(name=name))
                return [audio_label, self._make_text_block(transcript)]
            # Нативный формат требует binary
            if not caps.supports_binary:
                return [self._make_text_block(FILE_BINARY_UNAVAILABLE.format(name=name))]
            data = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
            mime = MIME_MAP.get(ext, "application/octet-stream")
            if audio_fmt == "audio":
                # Anthropic
                return [label, {"type": "audio", "source": {
                    "type": "base64", "media_type": mime, "data": data,
                }}]
            if audio_fmt == "media":
                # Gemini
                return [label, {"type": "media", "data": data, "mime_type": mime}]
            if audio_fmt == "input_audio":
                # OpenAI gpt-4o-audio: поддерживает только WAV и MP3
                fmt = ext if ext in ("wav", "mp3") else "mp3"
                return [label, {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}]
            # Неизвестный audio_format -- заглушка
            return [self._make_text_block(FILE_AUDIO_UNSUPPORTED.format(name=name, ext=ext))]

        # --- Видео ---
        # supports_video=False означает отсутствие нативной поддержки -> кадры + транскрипция.
        # Конвертация не требует supports_binary, поэтому обрабатываем до этой проверки.
        if kind == "video":
            if not caps.supports_video:
                # Нет нативной поддержки -- разбираем на кадры + транскрипция аудио
                blocks = []

                # Транскрипция аудио (если есть moviepy + faster-whisper).
                # None  -- дорожки нет, пропускаем без ошибки.
                # str   -- транскрипт (может быть пустым, если речи нет: тишина/музыка).
                #         Пустая строка тоже прикладывается как текстовый блок -- явный сигнал модели.
                # RuntimeError -- библиотека не установлена или ошибка транскрипции -> пробрасываем.
                transcript = self._video_extract_audio_transcript(path)
                if transcript is not None:
                    blocks.append(self._make_text_block(FILE_VIDEO_TRANSCRIPT.format(name=name)))
                    blocks.append(self._make_text_block(transcript))

                # Кадры (если есть opencv)
                frames_b64 = self._video_extract_frames_b64(path)
                if frames_b64:
                    blocks.append(self._make_text_block(FILE_VIDEO_FRAMES_LABEL.format(name=name)))
                    for frame_b64 in frames_b64:
                        if caps.image_format == "media":
                            blocks.append({"type": "media", "data": frame_b64, "mime_type": "image/jpeg"})
                        else:
                            blocks.append({"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_b64}",
                            }})

                if not blocks:
                    return [self._make_text_block(FILE_VIDEO_UNSUPPORTED.format(name=name, ext=ext))]
                return blocks

            # Нативный формат требует binary
            if not caps.supports_binary:
                return [self._make_text_block(FILE_BINARY_UNAVAILABLE.format(name=name))]
            data = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
            mime = MIME_MAP.get(ext, "application/octet-stream")
            if caps.video_format == "video_url":
                return [label, {"type": "video_url", "video_url": {
                    "url": f"data:{mime};base64,{data}",
                }}]
            if caps.video_format == "video_as_image_url":
                # Временный обход: LangChain/Ollama не пропускают тип video_url,
                # поэтому передаём как image_url с video MIME-типом в data URI.
                # Большинство мультимодальных моделей читают MIME из URI и обрабатывают верно.
                # TODO: заменить на "video_url" ветку когда Ollama/LangChain добавят поддержку.
                return [label, {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{data}",
                }}]
            # Anthropic / Gemini нативный формат
            return [label, {"type": "video", "source": {
                "type": "base64", "media_type": mime, "data": data,
            }}]

        # --- Изображение ---
        if kind == "image_url":
            if not caps.supports_binary:
                # Не-vision модель: описываем изображение текстом через LLaVA (Ollama).
                # Если Ollama недоступна -- возвращаем заглушку, не роняем запрос.
                try:
                    description = self._image_to_description(path)
                    img_label = self._make_text_block(FILE_IMAGE_DESCRIPTION.format(name=name))
                    return [img_label, self._make_text_block(description)]
                except Exception:
                    return [self._make_text_block(FILE_BINARY_UNAVAILABLE.format(name=name))]

            data = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
            mime = MIME_MAP.get(ext, "application/octet-stream")
            if caps.image_format == "media":
                return [label, {"type": "media", "data": data, "mime_type": mime}]
            return [label, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}]

        return [self._make_text_block(FILE_BINARY_UNAVAILABLE.format(name=name))]

    def _url_to_langchain(self, url, ext):
        """
        Формирует LangChain-блок(и) для URL без скачивания файла.
        Возвращает list[dict].
        Поддержка зависит от флагов ProviderCaps (self.provider).
        """
        kind = EXT_TYPE.get(ext)
        if kind is None:
            raise ValueError(ERR_UNKNOWN_EXT_URL.format(ext=ext))

        caps = self.provider  # ProviderCaps
        name = url.rstrip("/").rsplit("/", 1)[-1] or url
        label = self._make_text_block(FILE_LABEL.format(name=name))

        if not caps.supports_binary:
            return [self._make_text_block(FILE_URL_STUB.format(url=url))]

        mime = MIME_MAP.get(ext, "application/octet-stream")

        if kind == "image_url":
            if caps.image_format == "media":
                # Gemini: передаём URL как file_uri в media-блоке
                return [label, {"type": "media", "mime_type": mime, "file_uri": url}]
            return [label, {"type": "image_url", "image_url": {"url": url}}]

        if kind == "pdf":
            if caps.pdf_format == "file":
                return [label, {"type": "file", "file": {"file_data": url}}]
            if caps.supports_url_doc:
                return [label, {"type": "document", "source": {"type": "url", "url": url}}]
            return [self._make_text_block(FILE_PDF_URL_STUB.format(url=url))]

        # audio/video/office по URL -- только если провайдер поддерживает document-URL
        if caps.supports_url_doc:
            return [label, {"type": "document", "source": {"type": "url", "url": url}}]

        return [self._make_text_block(FILE_URL_STUB.format(url=url))]

    # -------------------------------------------------------------------------
    # Компиляция блоков
    # -------------------------------------------------------------------------

    def compile_block(self, block) -> CompiledBlock:
        """
        Принимает один элемент словаря prompts/systems/prt.
        Возвращает CompiledBlock(blocks, is_inline) -- список LangChain-блоков
        и флаг того, как compile_section должен их встраивать:

          is_inline=True  -- текст (prompt/text_file), объединяется с другими текстами через SEPARATOR
          is_inline=False -- вложение (file/url), ставится в file_blocks после всего текста

        Метка файла формируется внутри _file_to_langchain / _url_to_langchain в зависимости
        от типа файла и способа доставки (FILE_LABEL, FILE_AUDIO_TRANSCRIPT, FILE_VIDEO_TRANSCRIPT,
        FILE_VIDEO_FRAMES_LABEL). compile_block меток не добавляет.

        Вся логика форматирования инкапсулирована в _file_to_langchain и _url_to_langchain.
        Выбрасывает исключение при ошибках чтения файлов или неизвестных типах.
        """
        meta = block.get("metadata") or None

        # Готовые LangChain-данные -- встраиваем как вложение
        if "type" in block:
            return CompiledBlock(blocks=[block], is_inline=False, metadata=meta)

        # Текстовый блок -- инлайн
        if "prompt" in block:
            return CompiledBlock(
                blocks=[self._make_text_block(self._make_prompt_text(block))],
                is_inline=True,
                metadata=meta,
            )

        # Локальный файл: метка имени + данные, всегда как вложение (is_inline=False).
        # Все файлы идут после текстового промпта -- пользователь может ссылаться на них по имени.
        # Метка формируется внутри _file_to_langchain в зависимости от типа файла и способа доставки.
        if "file" in block:
            name  = Path(block["file"]).name
            ext   = block.get("filetype") or name.rsplit(".", 1)[-1].lower()
            self._copy_to_attachments(block["file"])
            data_blocks = self._file_to_langchain(block["file"], ext)
            return CompiledBlock(blocks=data_blocks, is_inline=False, metadata=meta)
            
        # URL: метка имени + данные, вложение.
        # Метка формируется внутри _url_to_langchain.
        if "url" in block:
            url  = block["url"]
            ext  = block.get("filetype") or url.rsplit(".", 1)[-1].lower()
            data_blocks = self._url_to_langchain(url, ext)
            return CompiledBlock(blocks=data_blocks, is_inline=False, metadata=meta)

        # Предконвертированный текстовый файл (например Excel->markdown).
        # Формат: {"text_file": "filename.xlsx", "content": "...текст..."}
        #      или {"text_file": "config.json"} -- content отсутствует, файл читается автоматически
        # Метка встроена в текст, встраивается инлайн вместе с другими текстами.
        if "text_file" in block:
            filename = block["text_file"]
            if "content" in block:
                text = block["content"]
            else:
                self._copy_to_attachments(filename)
                with open(filename, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            label    = FILE_LABEL.format(name=Path(filename).name)
            combined = f"{label}\n\n{text}" if text else label
            return CompiledBlock(blocks=[self._make_text_block(combined)], is_inline=True, metadata=meta)

        raise ValueError(ERR_UNKNOWN_BLOCK.format(block=block))

    def compile_section(self, section_dict, message_class):
        """
        Принимает весь словарь (prompts, systems или prt как словарь).
        Итерирует блоки в порядке sorted(keys).

        Использует CompiledBlock.is_inline для маршрутизации:
          is_inline=True  -> текст идёт в text_parts, объединяется через SEPARATOR в конце
          is_inline=False -> блоки идут в file_blocks, сохраняют порядок после текста

        Итоговый content: [объединённый текст] + [file_blocks...]
        Сначала текстовый промт (инструкция/вопрос), потом вложения.

        Если supports_content_blocks=False -- провайдер ожидает строку, а не массив блоков.
        В этом случае все текстовые блоки склеиваются через SEPARATOR в одну строку.
        Встреча любого не-текстового блока при этом вызывает исключение -- такая комбинация
        невозможна по инварианту (supports_binary=True -> supports_content_blocks=True).

        Возвращает message_class объект или None если всё пустое.
        """
        if not section_dict:
            return None

        caps = self.provider  # ProviderCaps

        text_parts    = []   # инлайн-тексты -- объединяются через SEPARATOR в конце
        file_blocks   = []   # вложения с метками -- стоят после текста
        merged_meta   = {}   # metadata из всех блоков; приоритет у блоков с меньшим ключом

        for key in sorted(section_dict):
            compiled = self.compile_block(section_dict[key])
            # Мержим metadata: ключи из блоков с меньшим индексом не перезаписываются
            if compiled.metadata:
                for k, v in compiled.metadata.items():
                    if k not in merged_meta:
                        merged_meta[k] = v
            if compiled.is_inline:
                for b in compiled.blocks:
                    if b["type"] == "text" and b["text"].strip():
                        text_parts.append(b["text"])
            else:
                # При !supports_content_blocks допустимы только текст и изображения.
                # Изображения остаются как блоки (модель может принимать их даже без
                # поддержки массива текстовых блоков). Всё остальное -- ошибка.
                if not caps.supports_content_blocks:
                    bad = [
                        b for b in compiled.blocks
                        if b.get("type") not in ("text", "image_url")
                    ]
                    if bad:
                        raise ValueError(ERR_CONTENT_BLOCKS.format(btype=bad[0].get("type")))
                file_blocks.extend(compiled.blocks)

        # Сначала объединённый текст, потом вложения
        content = []
        if text_parts:
            combined_text = SEPARATOR.join(text_parts)
            content.append({"type": "text", "text": combined_text})
        content.extend(file_blocks)

        if not content:
            return None

        # Kwargs для передачи metadata -- только если есть что передавать
        meta_kwargs = {"metadata": merged_meta} if merged_meta else {}

        # Если провайдер ожидает строку -- склеиваем текстовые блоки в одну строку.
        # Изображения при этом допустимы и остаются как блоки рядом со склеенным текстом.
        if not caps.supports_content_blocks:
            text_combined = SEPARATOR.join(
                b["text"] for b in content if b.get("type") == "text"
            )
            images = [b for b in content if b.get("type") == "image_url"]
            if not images:
                # Только текст -- передаём строкой
                return message_class(content=text_combined, **meta_kwargs)
            # Текст + изображения -- передаём массивом [текст, ...изображения]
            mixed = []
            if text_combined:
                mixed.append({"type": "text", "text": text_combined})
            mixed.extend(images)
            return message_class(content=mixed, **meta_kwargs)

        # Если единственный блок -- передаём content как строку (не массив),
        # это универсальнее и безопаснее для любых провайдеров.
        if len(content) == 1 and content[0]["type"] == "text":
            return message_class(content=content[0]["text"], **meta_kwargs)

        return message_class(content=content, **meta_kwargs)

    # -------------------------------------------------------------------------
    # Основные функции
    # -------------------------------------------------------------------------
    
    def prepare(self):
        # systems[0] -- зарезервирован: объединение self.system и self.system_tool_instructions.
        # Весь остальной диапазон ключей systems в распоряжении пользователя.
        system_parts = [p for p in (self.system, self.system_tool_instructions) if p and p.strip()]
        combined_system = "\n\n".join(system_parts)
        systems0 = {"prompt": combined_system} if isinstance(combined_system, str) else combined_system
        if not 0 in self.systems or self.systems[0] != systems0:
            self.systems[0] = systems0

        # prompts[0] -- место для self.prompt (простой строки или одного блока)
        prompts0 = {"prompt": self.prompt} if isinstance(self.prompt, str) else self.prompt
        if not 0 in self.prompts or self.prompts[0] != prompts0:
            self.prompts[0] = prompts0

        if not self.chat_history:
            self._tokens_snapshot_chat_history = 0
        self._last_chat_history_len = len(self.chat_history)

    def compile(self, prt) -> list:
        return self.compile_convert(self.compile_combine(prt))
    
    def compile_combine(self, prt) -> list:
        self.prepare()
        
        caps = self.provider  # ProviderCaps

        if caps.supports_system:
            # Провайдер поддерживает SystemMessage -- обычный путь.
            if caps.supports_system_files:
                # Провайдер принимает файлы в SystemMessage -- переброс не нужен,
                # компилируем systems целиком включая бинарные блоки.
                system_msg = self.compile_section(self.systems, SystemMessage)
                prompt_msg  = self.compile_section(self.prompts,  HumanMessage)
            else:
                # Но SystemMessage не может содержать бинарные вложения --
                # не-текстовые блоки из systems переносим в prompts (не мутируя self.systems).
                systems_text = {}   # только текстовые блоки -- остаются в SystemMessage
                systems_files = {}  # не-текстовые -- перебрасываются в prompts
                for k, v in self.systems.items():
                    if self._is_nontext_block(v):
                        systems_files[k] = v
                    else:
                        systems_text[k] = v
                        
                system_msg = self.compile_section(systems_text, SystemMessage)

                # Файлы из systems добавляем в prompts с ключами (k - 10000),
                # чтобы они гарантированно встали раньше всех остальных prompts.
                # Используем временный словарь -- self.prompts не меняем.
                prompts_with_sysfiles = {}
                for k, v in systems_files.items():
                    prompts_with_sysfiles[k - 10000] = v
                prompts_with_sysfiles.update(self.prompts)
                prompt_msg = self.compile_section(prompts_with_sysfiles, HumanMessage)
        else:
            # Провайдер не поддерживает system role --
            # все системные инструкции объединяем с prompts как HumanMessage.
            # Используем временный словарь -- self.systems не меняем.
            merged = {}
            for k, v in self.systems.items():
                merged[k - 10000] = v   # сдвигаем далеко влево чтобы не смешивались с prompts
            merged.update(self.prompts)
            system_msg = None
            prompt_msg  = self.compile_section(merged, HumanMessage)

        # prt -- строка или словарь как prompts
        if isinstance(prt, str):
            human_msg = HumanMessage(content=prt)
        else:
            human_msg = self.compile_section(prt, HumanMessage)

        s = messages_to_dict([system_msg]) if system_msg else []
        p = messages_to_dict([prompt_msg]) if prompt_msg else []
        h = messages_to_dict([human_msg]) if human_msg else []

        # Статичные промпты (помечены для удаления из истории после ответа)
        static = self.messages_add_mark(s) + self.messages_add_mark(p)

        # Если статичные промпты заканчиваются на human -- вставляем fake_ai
        # на границе между статичными промптами и историей чата.
        # Помечаем его тоже, чтобы он не сохранился в chat_history.
        if static and static[-1].get("type") == "human":
            fake_ai = messages_to_dict([AIMessage(content="OK.")])
            self.messages_add_mark(fake_ai)
            static = static + fake_ai

        before_h = static + self.chat_history

        # Если история чата заканчивается на human (не должно, но на всякий случай)
        # -- ещё один барьер перед финальным запросом пользователя.
        # НЕ помечаем -- пусть сохранится в истории, иначе следующий запрос
        # снова создаст два human подряд.
        if before_h and before_h[-1].get("type") == "human":
            fake_ai = messages_to_dict([AIMessage(content="OK.")])
            before_h = before_h + fake_ai

        result = before_h + h

        return result

    def compile_convert(self, messages: list) -> list:
        return messages_from_dict(messages)
        
    async def run_async(self, prt, config = None, context = None):
        await asyncio.to_thread(self._lock.acquire)
        try:
            self.runstart = datetime.now()
            self.prerun(prt)
            messages = self.compile(prt)
            if config is None: config = {}
            self.runinvoke = datetime.now()
            result = await self.agent.ainvoke({"messages": messages}, config=config, context=context)
            self.runfinish = datetime.now()
            return self.postrun(result)
        finally:
            self._lock.release()
    
    def run(self, prt, config = None, context = None):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, self.run_async(prt, config, context))
            return future.result()
    
    def prerun(self, prt):
        # Восстанавливаем рабочую директорию в workspace_dir перед каждым запуском.
        # Это гарантирует что относительные пути в инструментах всегда начинаются
        # от корня workspace, даже если предыдущий вызов инструмента сменил cwd.
        if self.workspace_dir is not None:
            try:
                import os as _os_prerun
                _os_prerun.chdir(self.workspace_dir)
            except OSError:
                pass

        _, tokens = self.estimate_tokens_withfactor(prt)
        #print("tokens=" + str(tokens) + " factor=" + str(self._token_correction_factor))
        if self.auto_summarize_history and tokens > self.context_limit:
            self.summarize_history(self.auto_summarize_history_keep_last)
            _, tokens = self.estimate_tokens_withfactor(prt)
            #print("summarize_history() tokens=" + str(tokens) + " factor=" + str(self._token_correction_factor))
            #print(self.log)
            #print(json.dumps(self.drop_history, ensure_ascii=False, indent=2))
            #print(json.dumps(self.chat_history, ensure_ascii=False, indent=2))
        
        if self.control_context_limit and tokens > self.context_limit:
            raise ValueError(ERR_OUT_CONTEXT_LIMIT.format(tokens=tokens, limit=self.context_limit))
        
    def postrun(self, result):
        history = result["messages"]
        serialized = messages_to_dict(history)
        self.debug_run_result = result
        self.chat_history = self.messages_delete_mark(serialized)
        self._tokens_snapshot_chat_history = 0

        if not self.chat_history:        
            self.last_message = MSG_RUN_UNKNOWN_ERROR
            self.append_log(self.last_message)
            return None

        # Извлекаем рассуждения из AI-сообщений раунда (до сериализации,
        # пока additional_kwargs ещё доступны на объекте LangChain).
        # Если log_thinking=True -- собираем из ВСЕХ AI-сообщений раунда,
        # объединяя через разделитель; иначе берём только из последнего.
        self.last_thinking = None
        if self.log_thinking:
            thinking_parts = []
            for msg in history:
                if isinstance(msg, AIMessage):
                    t, _ = self.extract_thinking(msg)
                    if t and t.strip():
                        thinking_parts.append(t.strip())
            if thinking_parts:
                self.last_thinking = "\n\n---\n\n".join(thinking_parts)
        else:
            for msg in reversed(history):
                if isinstance(msg, AIMessage):
                    thinking, _ = self.extract_thinking(msg)
                    self.last_thinking = thinking
                    break

        last = self.chat_history[-1]
        
        # Собираем usage_metadata из новых AI-сообщений раунда (от _last_chat_history_len до конца).
        # При использовании инструментов агент генерирует цепочку:
        #   AI(tool_call) -> ToolMessage -> ... -> AI(финальный)
        #
        # input_tokens  -- берём из ПЕРВОГО AI-сообщения: это наш реальный контекст до инструментов,
        #                 отражает сколько места занято в контексте нашим запросом.
        # output_tokens -- берём из ПОСЛЕДНЕГО AI-сообщения: это финальный ответ пользователю.
        # full_input_tokens  -- сумма input_tokens всех AI-сообщений раунда: полная входящая нагрузка
        #                      на модель (каждый промежуточный вызов тарифицируется отдельно).
        # full_output_tokens -- сумма output_tokens всех AI-сообщений раунда: полная исходящая нагрузка
        #                      (включает промежуточные ответы с tool_call и thinking-токены если есть).
        self.input_tokens       = 0
        self.output_tokens      = 0
        self.full_input_tokens  = 0
        self.full_output_tokens = 0
        response_metadata       = None
        new_messages = self.chat_history[self._last_chat_history_len:]
        first_ai_done = False
        for msg in new_messages:
            if msg.get("type") == "ai":
                usage = msg.get("data", {}).get("usage_metadata", None)
                if usage:
                    if not first_ai_done:
                        self.input_tokens = usage.get("input_tokens", 0)
                        first_ai_done = True
                    self.output_tokens      = usage.get("output_tokens", 0)  # перезаписываем -- нужен последний
                    self.full_input_tokens  += usage.get("input_tokens", 0)
                    self.full_output_tokens += usage.get("output_tokens", 0)
                # response_metadata берём из последнего AI-сообщения
                rm = msg.get("data", {}).get("response_metadata", None)
                if rm:
                    response_metadata = rm
        self.input_tokens_total       += self.input_tokens
        self.output_tokens_total      += self.output_tokens
        self.full_input_tokens_total  += self.full_input_tokens
        self.full_output_tokens_total += self.full_output_tokens

        time_prepare=round((self.runfinish-self.runstart).total_seconds(), 3)
        time_run=round((self.runfinish-self.runinvoke).total_seconds(), 3)
        self.last_load_sec = 0
        self.last_general_sec = time_run
        self.last_prefill_sec = time_run #грубо пишем общее время и туда и туда, точно определяем в parse_response_metadata
        self.last_decode_sec = time_run #грубо пишем общее время и туда и туда, точно определяем в parse_response_metadata
        self.last_prefill_speed = 0 #вычисляется в parse_response_metadata
        self.last_decode_speed = 0

        message = MSG_TOKENS.format(
            context=str(self.input_tokens) + "/" + str(self.context_limit),
            input=self.input_tokens,
            output=self.output_tokens,
            full_input=self.full_input_tokens,
            full_output=self.full_output_tokens,
            input_total=self.input_tokens_total,
            output_total=self.output_tokens_total,
            time_prepare=str(time_prepare),
            time_run=str(time_run)
        )
        if response_metadata:
            message_rm = self.parse_response_metadata(response_metadata);
            if isinstance(message_rm, str):
                if message_rm.strip(): 
                    message += "\n" + message_rm + "\n" #строку в конце можно убрать, она для отделения от истории
        self.last_message = message
        
        if self.loglevel > 1: 
            message += ("\n" + 
                (LOG_PREV_MESSAGES.format(n=self._last_chat_history_len) if self._last_chat_history_len > 0 else "") +
                self.history_tostr(self.chat_history[self._last_chat_history_len:]) + 
                "\n" + LOG_END)
        if self.loglevel > 2: 
            message += "\n" + json.dumps(serialized, ensure_ascii=False, indent=2)
        if self.loglevel > 1: 
            message += "\n"
        self.append_log(message)
        
        # Обновляем _token_correction_factor -- отношение реальных входящих токенов к нашей оценке.
        # Используем self.input_tokens (первое AI-сообщение раунда) -- он отражает реальный размер
        # контекста до инструментов, что соответствует тому что оценивает estimate_tokens_prt.
        tokens_before_run = self._tokens_systems + self._tokens_prompts + self._tokens_chat_history + self._tokens_prt
        if tokens_before_run > 0 and self.input_tokens > 0:
            self._token_correction_factor = self.input_tokens / tokens_before_run
        else:
            self._token_correction_factor = 1
        if self._token_correction_factor > 10:
            self._token_correction_factor = 10
        if self._token_correction_factor < 0.1:
            self._token_correction_factor = 0.1
        
        if last.get("type") != "ai":
            return None
        return last.get("data", {}).get("content", "")

    def append_log(self, message: str):
        """Добавляет запись в лог: перенос строки, временная метка, сообщение."""
        if self.loglevel == 0: 
            return
        now = datetime.now()
        ts = now.strftime("%Y.%m.%d %H:%M:%S") + f":{now.microsecond // 1000:03d}"
        self.log += f"\n[{ts}] {message}"
    
    def parse_response_metadata(self, response_metadata):
        """
        Парсим структуру ответа response_metadata.
        Должно учитываться, что у разных поставщиков совершенно разные подходы.
        На незнакомых поставщиках метод может не получить данные, но не должен выдавать ошибок.
        
        Также здесь вычесляется скорость на основании уже заполненых self.input_tokens и self.output_tokens
        """
        
        if not response_metadata or not isinstance(response_metadata, dict):
            return ""
        
        lines = []
        
        # Load model - self.last_load_sec - Загрузка модели
        if "load_duration" in response_metadata:
            load_sec = response_metadata["load_duration"] / 1_000_000_000  # наносекунды -> секунды
            self.last_load_sec = load_sec
            lines.append(MSG_LOAD_MODEL.format(sec=load_sec))
        
        # Prefill - self.last_prefill_sec - загрузка контекста
        if "prompt_eval_duration" in response_metadata:
            prefill_sec = response_metadata["prompt_eval_duration"] / 1_000_000_000
            self.last_prefill_sec = prefill_sec
            lines.append(MSG_PREFILL.format(sec=prefill_sec))
        
        # Decode - self.last_decode_sec - генерация ответа
        if "eval_duration" in response_metadata:
            decode_sec = response_metadata["eval_duration"] / 1_000_000_000
            self.last_decode_sec = decode_sec
            lines.append(MSG_DECODE.format(sec=decode_sec))
        
        # Вычисляем скорость.
        # prefill -- загрузка контекста: используем input_tokens (первый вызов раунда, до инструментов).
        # decode  -- генерация ответа: используем output_tokens (финальный ответ, последний вызов).
        # general -- общая нагрузка: full_input + full_output (все вызовы раунда включая инструменты).
        if self.last_prefill_sec > 0.1:
            self.last_prefill_speed = self.input_tokens / self.last_prefill_sec
        else:
            self.last_prefill_speed = 0
        if self.last_decode_sec > 0.1:
            self.last_decode_speed = self.output_tokens / self.last_decode_sec
        else:
            self.last_decode_speed = 0
        if self.last_general_sec > 0.1:
            self.last_general_speed = (self.full_input_tokens + self.full_output_tokens) / self.last_general_sec
        else:
            self.last_general_speed = 0
        
        if self.last_prefill_speed > 0:
            lines.append(MSG_PREFILL_SPEED.format(speed=self.last_prefill_speed))
        if self.last_decode_speed > 0:
            lines.append(MSG_DECODE_SPEED.format(speed=self.last_decode_speed))
        if self.last_general_speed > 0:
            lines.append(MSG_GENERAL_SPEED.format(speed=self.last_general_speed))
        
        # Признаки завершения (неудачного)
        
        if "done_reason" in response_metadata:
            done_reason = response_metadata["done_reason"]
            if done_reason == "length":
                lines.append(MSG_GENERATION_BREAK_LENGTH)
            elif done_reason == "timeout":
                lines.append(MSG_GENERATION_BREAK_TIMEOUT)
            elif done_reason != "stop":
                lines.append(MSG_GENERATION_BREAK)
        elif "done" in response_metadata:
            if not response_metadata["done"]:
                lines.append(MSG_GENERATION_BREAK)
        
        return "\n".join(lines)
        
    def get_systemprompt_log(self):
        """
        Записывает в лог системные промпты и постоянные prompts точно так,
        как они будут переданы модели -- в формате history_tostr (system>, human>).

        Вызывает compile("") который внутри вызывает prepare() с подстановкой
        переменных и объединением system + system_tool_instructions.
        Актуально после любого изменения: prompt, prompts, system, systems,
        а также после mcp_connect/mcp_disconnect (меняет system_tool_instructions).
        """
        compiled = self.compile("")
        message = self.history_tostr(messages_to_dict(compiled)) + "\n" + LOG_END + "\n"
        message = message.replace("human> \n" + LOG_END + "\n", LOG_END + "\n")
        self.append_log(LOG_SET_SYSTEM_PROMPT.format(text=message))
        return message
    
    def messages_add_mark(self, messages: list, value = '1') -> list:
    # метка 1 - это часть неизменного системного и пользовательского промта в начале
    # метка 2 - это саммори для сокращение истории, удаляется при накопительном логировании удалённой истории в drop_history
        for item in messages:
            data = item.get('data', {})
            if 'metadata' not in data:
                data['metadata'] = {}
            data['metadata']['ail_s'] = value
        return messages

    def messages_delete_mark(self, messages: list, value = '1') -> list:
        return [
            msg for msg in messages
            if str(msg.get('data', {}).get('metadata', {}).get('ail_s', '')) != value
        ]

    # -------------------------------------------------------------------------
    # Дополнительные функции 
    # -------------------------------------------------------------------------

    def can_accept(self, block: dict) -> bool:
        """
        Сообщает, будет ли блок передан модели НАТИВНО -- без промежуточной конвертации.

        True  -- файл отправляется модели as-is (base64, url, готовый блок).
        False -- файл будет преобразован перед отправкой: PDF->текст, docx->текст,
                аудио->транскрипция, видео->кадры+текст, html->текст и т.д.

        Заметка: False НЕ означает что файл не будет обработан. Все форматы
        у которых есть путь конвертации передаются прозрачно -- просто не нативно.
        Единственный случай когда файл не может быть обработан вообще --
        изображение при supports_binary=False (нет ни нативной передачи, ни конвертации).

        Типы блоков:
          { "prompt": ... }     -- True (текст, нативно)
          { "text_file": ... } -- True (уже предобработан)
          { "type": ... }      -- True (готовый LangChain-блок)
          { "file": ... }      -- зависит от kind и caps
          { "url": ... }       -- зависит от kind и caps
        """
        caps = self.provider  # ProviderCaps

        if "prompt" in block or "text_file" in block or "type" in block:
            return True

        if "file" in block:
            name = Path(block["file"]).name
            ext  = block.get("filetype") or name.rsplit(".", 1)[-1].lower()
            kind = EXT_TYPE.get(ext)
            if kind is None:
                return False
            # Конвертируемые форматы -- всегда False (конвертация прозрачна, но не нативно)
            if kind in ("text", "html", "office"):
                return False
            if kind == "pdf":
                return caps.pdf_format is not None   # None -> pypdf-конвертация
            if kind == "audio":
                return caps.audio_format is not None # None -> Whisper-транскрипция
            if kind == "video":
                return caps.supports_video           # False -> кадры+транскрипция
            if kind == "image_url":
                return caps.supports_binary          # нет конвертации -- только binary
            return False

        if "url" in block:
            url = block["url"]
            ext = block.get("filetype") or url.rsplit(".", 1)[-1].lower()
            kind = EXT_TYPE.get(ext)
            if kind is None:
                return False
            if kind in ("text", "html", "office"):
                return False
            if kind == "image_url":
                return caps.supports_binary
            if kind == "pdf":
                return caps.pdf_format is not None or caps.supports_url_doc
            return caps.supports_url_doc

        return False

    def _count_text_tokens(self, text: str) -> int:
        """
        Считает токены для текстовой строки.
        Если tiktoken доступен -- использует кодировку cl100k_base (GPT-4),
        результат точный для OpenAI и приближённый (~5-15% погрешность) для остальных.
        Если tiktoken недоступен -- символьный fallback: len(text) // 4.
        """
        if _TIKTOKEN_ENC is not None:
            #print("================================= " + str(len(_TIKTOKEN_ENC.encode(text))) + " vs " + str((len(text.encode("utf-8")) + 3) // 4))
            return len(_TIKTOKEN_ENC.encode(text))
            
        return (len(text.encode("utf-8")) + 3) // 4

    @staticmethod
    def _dict_snapshot(obj):
        """
        Быстрый хэш состояния объекта.
        Для словаря: по id его и его элементов.
        Для списка: по id его и его элементов.
        Ловит замену самого объекта и замену любого элемента.
        Не ловит мутацию внутри элемента -- для таких случаев есть update().
        """
        if isinstance(obj, dict):
            return hash((id(obj), tuple((k, id(v)) for k, v in sorted(obj.items()))))
        elif isinstance(obj, list):
            return hash((id(obj), tuple(id(item) for item in obj)))
        else:
            return hash(id(obj))
        
    def update(self):
        """
        Принудительно сбрасывает кэш токенов статичных промтов.
        Запускайте после изменения любого из объектов:
        prompt
        prompts
        system
        systems
        chat_history
        """
        self._tokens_snapshot_systems = 0
        self._tokens_snapshot_prompts = 0
        self._tokens_snapshot_chat_history = 0
        self._last_chat_history_len = len(self.chat_history)
        
    def estimate_tokens_withfactor(self, prt = ""):
        """
        Выдаёт оценку аналогично estimate_tokens_prt, но с коэффициентом коррекции
        
        Результат: (токены для prt, токены для systems, prompts, chat_history и prt)
        """
        self.estimate_tokens_prt(prt)
        return int(self._token_correction_factor * self._tokens_prt), int(self._token_correction_factor * (self._tokens_systems + self._tokens_prompts + self._tokens_chat_history + self._tokens_prt))

    def estimate_tokens_prt(self, prt, allow_api: bool = False) -> tuple[int, bool]:
        """
        Оценивает токены для systems, prompts, chat_history и текущего запроса prt.

        systems, prompts и chat_history кэшируются: пересчитываются только если словари изменились
        (определяется через _dict_snapshot по id элементов).
        prt считается всегда -- он меняется при каждом вызове.

        Возвращает (tokens_prt, is_exact).
          is_exact=True  -- результат точный (API или локальный tiktoken у OpenAI).
          is_exact=False -- приближённая оценка (tiktoken cl100k_base или символьный fallback).
        """
        # Подготовка так же как это делает compile_combine перед компиляцией.
        self.prepare()

        # --- Кэш для systems ---
        snapshot_systems = self._dict_snapshot(self.systems)
        if snapshot_systems != self._tokens_snapshot_systems:
            system_msg = self.compile_section(self.systems, SystemMessage)
            self._tokens_systems, is_exact_s  = self.estimate_tokens([system_msg] if system_msg else [], allow_api)
            self._tokens_snapshot_systems = snapshot_systems
        else:
            is_exact_s = self.provider.local_token_count or allow_api

        # --- Кэш для prompts ---
        snapshot_prompts = self._dict_snapshot(self.prompts)
        if snapshot_prompts != self._tokens_snapshot_prompts:
            prompt_msg = self.compile_section(self.prompts, HumanMessage)
            self._tokens_prompts, is_exact_p  = self.estimate_tokens([prompt_msg]  if prompt_msg  else [], allow_api)
            self._tokens_snapshot_prompts = snapshot_prompts
        else:
            is_exact_p = self.provider.local_token_count or allow_api

        # --- Кэш для chat_history ---
        snapshot_chat_history = self._dict_snapshot(self.chat_history)
        if snapshot_chat_history != self._tokens_snapshot_chat_history:
            history_msgs = self.compile_convert(self.chat_history)
            self._tokens_chat_history, is_exact_ch  = self.estimate_tokens(history_msgs  if history_msgs  else [], allow_api)
            self._tokens_snapshot_chat_history = snapshot_chat_history
        else:
            is_exact_ch = self.provider.local_token_count or allow_api
    
        # --- prt считаем всегда ---
        if isinstance(prt, str):
            human_msg = HumanMessage(content=prt)
        else:
            human_msg = self.compile_section(prt, HumanMessage)
        self._tokens_prt, is_exact_h = self.estimate_tokens([human_msg] if human_msg else [], allow_api)

        #is_exact = is_exact_s and is_exact_p and is_exact_ch and is_exact_h
        return self._tokens_prt, is_exact_h

    def estimate_tokens(self, messages: list, allow_api: bool = False) -> tuple[int, bool]:
        """
        Оценивает количество токенов в списке LangChain-сообщений.

        messages   -- список LangChain-объектов (результат compile() или compile_convert()).
        allow_api  -- разрешить сетевой запрос для точного подсчёта.
                     False: только локально; при отсутствии локального токенизатора --
                            символьный fallback (~/4 символа на токен).
                     True:  пробуем get_num_tokens_from_messages(); для провайдеров
                            без локального токенизатора (Anthropic и др.) это может
                            уйти в сеть -- используй осознанно.

        Возвращает (n_tokens, is_exact):
          is_exact=True  -- результат точный (локальный tiktoken или API).
          is_exact=False -- символьная оценка, погрешность ~20-30%.
        """
        # Локальный точный подсчёт -- только для провайдеров у которых он гарантированно
        # не уходит в сеть (сейчас только OpenAI/tiktoken).
        if self.provider.local_token_count:
            try:
                n = self.agent.get_graph().nodes  # достучаться до модели внутри агента
                # Агент оборачивает модель -- берём модель напрямую через bound
                model = self.agent.nodes["agent"].bound  # type: ignore
                res = model.get_num_tokens_from_messages(messages), True
                return res
            except Exception:
                pass  # не удалось добраться до модели -- идём дальше

        # API-подсчёт -- только если явно разрешён и провайдер не локальный
        # (локальный уже обработан выше и не попадёт сюда)
        if allow_api and not self.provider.local_token_count:
            try:
                model = self.agent.nodes["agent"].bound  # type: ignore
                res = model.get_num_tokens_from_messages(messages), True
                return res
            except Exception:
                pass  # провайдер не поддерживает -- fallback

        # Символьный fallback: перебираем блоки, для каждого типа своя оценка.
        total_tokens = 0
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else ""
            if isinstance(content, str):
                total_tokens += self._count_text_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")

                    if btype == "text":
                        # Текст: tiktoken если доступен, иначе символы / 4
                        total_tokens += self._count_text_tokens(block.get("text", ""))

                    elif btype == "image_url":
                        # Изображение: (width * height) / 750 -- формула Anthropic,
                        # используем как универсальную оценку когда точный подсчёт недоступен.
                        # Пробуем декодировать base64 и получить размеры через PIL.
                        try:
                            import PIL.Image, io
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                b64 = url.split(",", 1)[1]
                                img = PIL.Image.open(io.BytesIO(base64.b64decode(b64)))
                                w, h = img.size
                                total_tokens += (w * h) // 750
                            else:
                                # URL без base64 -- размер неизвестен, грубая оценка
                                total_tokens += 1000
                        except Exception:
                            # PIL недоступен или декодирование не удалось -- грубая оценка
                            total_tokens += 1000

                    elif btype == "media":
                        # Gemini-формат изображения
                        try:
                            import PIL.Image, io
                            data = block.get("data", "")
                            img = PIL.Image.open(io.BytesIO(base64.b64decode(data)))
                            w, h = img.size
                            total_tokens += (w * h) // 750
                        except Exception:
                            total_tokens += 1000

                    elif btype in ("document", "audio", "video"):
                        # Бинарный файл в base64: длина_base64_строки / 1.33 / 4
                        # /1.33 -- убираем раздувание base64, получаем исходные байты
                        # /4    -- грубый перевод байт в токены
                        src = block.get("source", {})
                        b64 = src.get("data", "") if isinstance(src, dict) else ""
                        total_tokens += int(len(b64) / 1.33 / 4)

        return total_tokens, False

    def summarize_history(self, keep_last: int = 0):
        """
        Сжимает начало chat_history, оставляя keep_last последних записей нетронутыми.

        keep_last -- минимальное количество записей с конца которые гарантированно остаются.
        Реальная граница сдвигается назад (т.е. keep_last только увеличивается) до тех пор
        пока первая запись to_keep не окажется HumanMessage -- чтобы не разрывать связки
        вызовов инструментов (ai -> tool -> ai).

        Возвращает True если суммаризация выполнена, False если не было нужды.
        """
        if not self.chat_history:
            return False

        # --- Определяем границу cut: индекс первой записи to_keep ---
        cut = len(self.chat_history) - keep_last

        # keep_last >= len истории -- нечего сжимать
        if cut <= 0:
            return False

        if keep_last > 0:
            # Сдвигаем cut назад пока to_keep не начнётся с HumanMessage --
            # чтобы не разрывать связки вызовов инструментов (ai -> tool -> ai).
            # Граница только растёт -- keep_last не нарушается.
            while cut > 0 and self.chat_history[cut].get("type") != "human":
                cut -= 1

            # human не нашёлся -- сжимать некуда
            if cut == 0:
                return False

        to_compress = self.chat_history[:cut]
        to_keep     = self.chat_history[cut:]

        # --- Promts с явной границей в конце ---
        # float('inf') гарантирует что граница встанет последней при sorted()
        # Граница добавляется только если systems или prompts дают реальный контент --
        # проверяем через compile_section, а не по наличию ключей в словаре
        # (словари могут быть непустыми но состоять из пустых строк).
        system_msg = self.compile_section(self.systems,  SystemMessage)
        prompt_msg  = self.compile_section(self.prompts,   HumanMessage)

        prompts_with_boundary = dict(self.prompts)
        if system_msg or prompt_msg:
            prompts_with_boundary[float('inf')] = {"prompt": SUMMARIZE_BOUNDARY}

        prompt_msg = self.compile_section(prompts_with_boundary, HumanMessage)

        s = messages_to_dict([system_msg]) if system_msg else []
        p = messages_to_dict([prompt_msg])  if prompt_msg  else []
        self.messages_add_mark(s)
        self.messages_add_mark(p)

        summarize_request = HumanMessage(content=SUMMARIZE_REQUEST)
        summarize_request_d = messages_to_dict([summarize_request])
        self.messages_add_mark(summarize_request_d)

        messages_for_llm = self.compile_convert(
            s + p + to_compress + summarize_request_d
        )

        result = self.agent.invoke({"messages": messages_for_llm})

        summary_text = None
        usage_metadata    = None
        response_metadata = None
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                summary_text      = msg.content.strip()
                usage_metadata    = getattr(msg, "usage_metadata",    None)
                response_metadata = getattr(msg, "response_metadata", None)
                break

        if not summary_text:
            return False

        # Учитываем токены суммаризации в общих счётчиках.
        # Суммаризация -- отдельный запрос к модели, его стоимость не должна теряться.
        # Добавляем в full_* (полная нагрузка на API), но не в input_*/output_* --
        # те отражают только токены основного пользовательского запроса.
        if usage_metadata:
            s_input  = usage_metadata.get("input_tokens",  0)
            s_output = usage_metadata.get("output_tokens", 0)
            self.full_input_tokens_total  += s_input
            self.full_output_tokens_total += s_output

        # Пара human+ai чтобы не ломать чередование типов в chat_history.
        # Метаданные переносим из ответа модели -- фиксирует стоимость суммаризации.
        summary_ai = AIMessage(content=summary_text)
        if usage_metadata:
            summary_ai.usage_metadata    = usage_metadata
        if response_metadata:
            summary_ai.response_metadata = response_metadata

        summary_pair = messages_to_dict([
            HumanMessage(content=SUMMARIZE_LABEL),
            summary_ai,
        ])

        self.drop_history = self.drop_history + self.messages_delete_mark(list(to_compress), '2')
        self.chat_history = self.messages_add_mark(summary_pair, '2') + list(to_keep)
        return True

    def history_tostr(self, history: list) -> str:
        """
        Преобразует history в текстовый диалог.
        
        Формат вывода:
        human> текст
        Attachments: 2
        aI>
        Tool calls: tool_func
        tool> текст
        aI> текст        
        """
        if not history:
            return ""
        
        lines = []
        
        for msg in history:
            msg_type = msg.get("type", "")#.upper()
            data = msg.get("data", {})
            content = data.get("content", "")
                            
            # Форматируем основное сообщение
            if isinstance(content, str):
                if self.log_thinking and msg_type == "ai":
                    # Для строкового контента thinking живёт в additional_kwargs
                    reasoning = (data.get("additional_kwargs") or {}).get("reasoning_content")
                    if reasoning and reasoning.strip():
                        # ai> <thinking> идёт на одной строке с меткой сообщения
                        lines.append(f"{msg_type}> {LOG_THINKING_START}")
                        lines.append(reasoning.strip())
                        lines.append(LOG_THINKING_END)
                        # content (финальный ответ) после закрывающего тега не дублируем меткой
                        if content.strip():
                            lines.append(content)
                    else:
                        lines.append(f"{msg_type}> {content}")
                else:
                    lines.append(f"{msg_type}> {content}")
            elif isinstance(content, list):
                # Для составных сообщений (с файлами, изображениями)
                text_parts = []
                thinking_parts = []
                other = 0
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "thinking":
                        thinking_parts.append(
                            block.get("thinking") or block.get("text") or ""
                        )
                    else:
                        other += 1
                if self.log_thinking and msg_type == "ai" and thinking_parts:
                    # ai> <thinking> идёт на одной строке с меткой сообщения
                    lines.append(f"{msg_type}> {LOG_THINKING_START}")
                    lines.append("\n".join(thinking_parts).strip())
                    lines.append(LOG_THINKING_END)
                    if text_parts:
                        lines.append("\n".join(text_parts))
                    # метка msg_type уже выведена выше — не дублируем
                elif text_parts:
                    combined_text = "\n".join(text_parts)
                    lines.append(f"{msg_type}> {combined_text}")
                else:                
                    lines.append(f"{msg_type}>")    
                if other>0:
                    lines.append(LOG_HISTORY_ATTACHMENTS.format(n=other))
                    
            calls = []
            tool_calls = data.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    name_tc = tool_call.get("name", "")
                    args = tool_call.get("args", {}) or tool_call.get("input", {}) or {}
                    if args:
                        def _trunc(v, limit=1000):
                            if isinstance(v, str):
                                return repr(v if len(v) <= limit else v[:limit] + "...")
                            s = str(v)
                            return s if len(s) <= limit else s[:limit] + "..."
                        args_str = ", ".join(f"{k}={_trunc(v)}" for k, v in args.items())
                        calls.append(f"{name_tc}({args_str})")
                    else:
                        calls.append(name_tc)
            callsstr = ", ".join(calls)
            if callsstr:
                lines.append(LOG_HISTORY_TOOL_CALLS.format(calls=callsstr))
            
        return "\n".join(lines)