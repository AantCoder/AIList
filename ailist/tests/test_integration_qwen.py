"""
Integration tests for AIList with the Qwen provider.

WHAT IS TESTED:
  - The model actually receives and reads each attachment type.
  - Plain text formats (txt, py, json, csv): content is passed as-is,
    the model can see specific keywords and numbers from the files.
  - HTML: tags stripped by BeautifulSoup, model sees plain text only.
  - Office formats (.docx, .xlsx): text conversion works, model reads the data.
  - PDF: text conversion via pypdf works, model reads the data.
  - Image: native image_url block, model describes the content.
  - Video: native video_url (Qwen), model describes what is happening.
  - Audio: Qwen has no native audio support -> transparently transcribed via faster-whisper
    (local, CPU-only, no API key required). Request 6 runs this if faster-whisper is installed.

WHAT IS NOT TESTED:
  - Answer quality - only that the answer is non-empty and contains keywords.
  - Accuracy of image/video recognition - subjective, verified manually.
  - Behaviour with corrupt files or empty context.

REQUIRED FILES IN ITEST_FILES_DIR:
  itest_text.txt   - two lines: "Keyword: ALPHA" and "Number: 4821"
  itest_code.py    - function secret() returns "BETA", constant MAGIC = 9374
  itest_data.json  - {"codename": "GAMMA", "value": 5510}
  itest_table.csv  - header name,score; row DELTA,7753
  itest_page.html  - <body> contains EPSILON and 6621, <script> tag with junk
  itest_doc.docx   - paragraphs "Word: ZETA" and "Value: 3307"
  itest_table.xlsx - cell A1=ETA, B1=8864
  itest_doc.pdf    - text "Label: THETA" and "Number: 2293"
  itest_photo.jpg  - any JPEG photo with a clearly visible object
  itest_video.mp4  - short video 5-15 sec with some action
  itest_audio.mp3  - short audio clip with clearly spoken content (for Request 6)

REQUEST STRATEGY (5+1 requests total):
  Request 1 - txt + py + json + csv  (4 files, one question per file)
  Request 2 - html + docx + xlsx + pdf  (4 files, all converted to text)
  Request 3 - jpg  (native image_url)
  Request 4 - mp4  (native video_url)
  Request 5 - compile_combine smoke: system prompt + history + new question
  Request 6 - mp3  (faster-whisper local transcription, CPU-only)
"""

import os
import sys

def _safe_print(s):
    """Print string safely on any console encoding (e.g. cp1251 on Windows)."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode(sys.stdout.encoding or 'utf-8', errors='replace')
               .decode(sys.stdout.encoding or 'utf-8', errors='replace'))

import time
from pathlib import Path
from dataclasses import replace

# --------------------------------------------------------------
# Settings - adjust to your environment
# --------------------------------------------------------------

# Path to test fixture files.
# Set AILIST_TEST_FILES_DIR env var, or pass as first argument: python test_integration_qwen.py /path/to/files
ITEST_FILES_DIR = (
    sys.argv[1] if len(sys.argv) > 1 else
    os.environ.get("AILIST_TEST_FILES_DIR", "test_files")
)

from ailist import AIList, AIListBase, Provider

# --------------------------------------------------------------
# AIList with Qwen - do not change the model, only switch provider
# --------------------------------------------------------------


# We disable native video support, because Ollama does not support her yet
custom_provider = replace(Provider.QWEN, supports_video=False, video_format=None)

class AIListTest(AIListBase):
    def __init__(self):
        #super().__init__("ollama:gpt-oss:20bgpu", int(16384*0.8), Provider.GPT_OSS)
        super().__init__("ollama:qwen3.5:35b-a3b", int(65536*0.8), custom_provider)

ai = AIListTest()
ai.loglevel = 2          # time and token stats only, no full history dump
ai.control_context_limit = False  # skip token limit check (no local tokenizer for Qwen)

# Disable global prompts to keep context clean and avoid influencing answers
ai.prompts  = {}
ai.systems = {}
ai.prompt   = ""
ai.system  = ""

def fp(name: str) -> str:
    return str(Path(ITEST_FILES_DIR) / name)

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------

_any_fail  = False
_t_total   = time.perf_counter()  # общий таймер всего прогона
_t_section = None                  # таймер текущей секции

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

def run(prt, label: str) -> str:
    """Run a request, print the answer, return answer string."""
    t0 = time.perf_counter()
    print(f"\n  >> Request: {label}")
    try:
        config = ai.apply_thinking_mode(thinking=False)
        answer = ai.run(prt, config=config)
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

def contains(text: str, *words) -> bool:
    """Return True if all words are found in text (case-insensitive)."""
    tl = text.lower()
    return all(w.lower() in tl for w in words)


# --------------------------------------------------------------
# Request 1 - Plain text formats: txt, py, json, csv
# One request, four files, four questions in one prompt.
# Checks that the model sees specific keywords from each file.
# --------------------------------------------------------------

section("Request 1 - txt / py / json / csv")
print("  Expected: model names ALPHA/4821, BETA/9374, GAMMA/5510, DELTA/7753")

ai.chat_history = []

prt = {
    0: {"file": fp("itest_text.txt")},
    1: {"file": fp("itest_code.py")},
    2: {"file": fp("itest_data.json")},
    3: {"file": fp("itest_table.csv")},
    4: {"prompt": (
        "Answer four questions about the attached files. "
        "Be brief, facts only.\n"
        "1. What is the keyword and number in the .txt file?\n"
        "2. What does the secret() function return and what is the value of MAGIC in the .py file?\n"
        "3. What are the values of codename and value in the JSON file?\n"
        "4. What is the name and score in the first data row of the CSV file?"
    )},
}

ans1 = run(prt, "txt+py+json+csv")

check(contains(ans1, "alpha"),  "txt: keyword ALPHA")
check(contains(ans1, "4821"),   "txt: number 4821")
check(contains(ans1, "beta"),   "py: secret() -> BETA")
check(contains(ans1, "9374"),   "py: MAGIC = 9374")
check(contains(ans1, "gamma"),  "json: codename=GAMMA")
check(contains(ans1, "5510"),   "json: value=5510")
check(contains(ans1, "delta"),  "csv: name=DELTA")
check(contains(ans1, "7753"),   "csv: score=7753")


# --------------------------------------------------------------
# Request 2 - Converted formats: html, docx, xlsx, pdf
# All converted to text before sending - Qwen has no native support.
# Checks that conversion worked and the model reads the data.
# --------------------------------------------------------------

section("Request 2 - html / docx / xlsx / pdf")
print("  Expected: model names EPSILON/6621, ZETA/3307, ETA/8864, THETA/2293")
print("  Note: all files converted to text before sending (Qwen has no native support)")

ai.chat_history = []

prt = {
    0: {"file": fp("itest_page.html")},
    1: {"file": fp("itest_doc.docx")},
    2: {"file": fp("itest_table.xlsx")},
    3: {"file": fp("itest_doc.pdf")},
    4: {"prompt": (
        "Answer four questions about the attached files. "
        "Be brief, facts only.\n"
        "1. What word and number appear in the HTML page?\n"
        "2. What word follows 'Word:' and what number follows 'Value:' in the Word document?\n"
        "3. What is in cells A1 and B1 of the Excel table?\n"
        "4. What keyword and number are written in the PDF document?"
    )},
}

ans2 = run(prt, "html+docx+xlsx+pdf")

check(contains(ans2, "epsilon"),  "html: EPSILON")
check(contains(ans2, "6621"),     "html: 6621")
check(contains(ans2, "zeta"),     "docx: ZETA")
check(contains(ans2, "3307"),     "docx: 3307")
check(contains(ans2, "eta"),      "xlsx: ETA")
check(contains(ans2, "8864"),     "xlsx: 8864")
check(contains(ans2, "theta"),    "pdf: THETA")
check(contains(ans2, "2293"),     "pdf: 2293")


# --------------------------------------------------------------
# Request 3 - Image (native image_url)
# Qwen sees the image directly. Checks that the model responds
# with something meaningful about the content.
# Result verified manually (subjective).
# --------------------------------------------------------------

section("Request 3 - jpg (native image_url)")
print("  Expected: non-empty answer describing the photo content")
print("  Check: answer is non-empty and contains the words 'red' and 'car'")

ai.chat_history = []

prt = {
    0: {"file": fp("itest_photo.jpg")},
    1: {"prompt": "Briefly describe what is shown in this image. 1-2 sentences."},
}

ans3 = run(prt, "jpg")

check(len(ans3.strip()) > 10, "answer is non-empty (>10 chars)")
check(contains(ans3, "red"),  "description contains 'red'")
check(contains(ans3, "car"),  "description contains 'car'")


# --------------------------------------------------------------
# Request 4 - Video (native video_url for Qwen)
# Qwen supports video_url. Checks that the block is formed
# correctly and the model answers a question about the video.
# --------------------------------------------------------------

section("Request 4 - mp4 (native video_url)")
print("  Expected: non-empty answer describing what happens in the video")
print("  Check: auto - answer is non-empty; manual - description matches the video")
print("  Note: Qwen receives video_url with data:video/mp4;base64,... - heavy on context!")
print("        Use a short file (<=15 sec, ideally <=5 MB)")

ai.chat_history = []

prt = {
    0: {"file": fp("itest_video.mp4")},
    1: {"prompt": "Briefly describe what is happening in this video. 1-2 sentences."},
}

ans4 = run(prt, "mp4")

check(len(ans4.strip()) > 10, "answer is non-empty (>10 chars)")
print("  MANUAL CHECK: read the answer above and confirm it matches the video")


# --------------------------------------------------------------
# Request 5 - Smoke test: compile_combine (system prompt + history)
# Verifies the full context assembly chain:
# system -> prompt -> chat_history -> new question.
# Uses plain text only to isolate this level from attachment logic.
# --------------------------------------------------------------

section("Request 5 - Smoke: system + prompts + chat_history")
print("  Expected: model follows system prompt and remembers previous answer")

ai.chat_history = []
ai.system = "You are a concise assistant. Start every reply with the '#' character."
ai.prompt  = ""

# First turn - create history
ans5a = run("Remember the number 1337. Reply with just: 'Remembered'.", "smoke turn 1")
check(ans5a.strip().startswith("#"), "system prompt: answer starts with '#'")

# Second turn - verify history is used
ans5b = run("What number did I ask you to remember?", "smoke turn 2 (history)")
check(contains(ans5b, "1337"), "chat_history: model remembers 1337")
check(ans5b.strip().startswith("#"), "system prompt persists in turn 2")

# Reset state
ai.system = ""
ai.chat_history = []


# --------------------------------------------------------------
# Request 6 - Audio (faster-whisper local transcription)
# Qwen has no native audio support (audio_format=None) -> the file is
# transparently transcribed via faster-whisper (CPU, no GPU, no API key)
# before being sent as text.
# Skipped if faster-whisper is not installed.
# The audio file should contain clearly spoken content so the model
# can answer a factual question about what was said.
# --------------------------------------------------------------

section("Request 6 - mp3 (faster-whisper local transcription, CPU-only)")
print("  Expected: model reads the transcript and answers about spoken content")
print("  Note: transcription happens inside _file_to_langchain before the LLM request")
print("  Note: no API key required, model downloaded automatically on first run")

ai.chat_history = []

try:
    from faster_whisper import WhisperModel as _check_fw  # noqa: F401
    _fw_available = True
except ImportError:
    _fw_available = False

if not _fw_available:
    print("  SKIP - faster-whisper not installed (pip install faster-whisper)")
else:
    prt = {
        0: {"file": fp("itest_audio.mp3")},
        1: {"prompt": (
            "The attached file contains a transcription of an audio recording. "
            "What is the main topic or content of what was said? "
            "Answer in 1-2 sentences."
        )},
    }
    ans6 = run(prt, "mp3 (faster-whisper)")
    check(len(ans6.strip()) > 10, "answer is non-empty (>10 chars)")
    print("  MANUAL CHECK: confirm the answer matches what was spoken in the audio")

# Закрываем таймер последней секции (Request 6) через section() невозможно --
# Summary не является секцией. Закрываем вручную и сбрасываем чтобы не повторить.
if _t_section is not None:
    print(f"  Time: {time.perf_counter() - _t_section:.3f} sec")
    _t_section = None

# --------------------------------------------------------------
# Summary
# --------------------------------------------------------------

print(f"\n{'='*60}")
print("  Log:")
_safe_print(ai.log)
print("=" * 60)
if _any_fail:
    print("  RESULT: some checks FAILED - see FAIL lines above")
else:
    print("  RESULT: all automatic checks passed")
print(f"  Total time: {time.perf_counter() - _t_total:.3f} sec")
print("=" * 60)