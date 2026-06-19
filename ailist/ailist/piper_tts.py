"""
piper_tts.py -- локальный синтез речи через piper-tts.

Установка:
    pip install piper-tts huggingface_hub

Использование (из кода):
    from piper_tts import PiperTTS
    from pathlib import Path

    piper = PiperTTS(
        models_dir=Path.home() / ".cache/ailist/piper",
        output_dir=Path("t2v"),
    )
    piper.model = "ru_RU-irina-medium"
    path = piper.synthesize("Привет!")           # -> t2v/t2v_<timestamp>.wav
    path = piper.synthesize("Текст", output_file="out.wav")

Использование через MCP-инструменты (для ЛЛМ):
    tools = piper.as_tools()
    # tools[0] -- piper_synthesize(text)         -> генерация WAV
    # tools[1] -- piper_set_config(...)           -> смена настроек
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from langchain.tools import tool


class PiperTTS:
    """
    Синтез речи через локальный piper-tts.

    Все настройки хранятся как атрибуты экземпляра; пользователь меняет их напрямую.
    ЛЛМ может менять настройки через инструмент piper_set_config -- изменения
    остаются в силе на всё время жизни объекта (навсегда в рамках сессии).

    Атрибуты (дефолты):
        model        (str)        -- имя голосовой модели, например 'ru_RU-irina-medium'
        speaker_id   (int|None)   -- спикер для multi-speaker моделей (None -> дефолт модели)
        length_scale (float|None) -- скорость речи: <1.0 быстрее, >1.0 медленнее
        noise_scale  (float|None) -- вариативность интонации 0.0-1.0
        noise_w      (float|None) -- вариативность ударений 0.0-1.0
        use_gpu      (bool)       -- True = GPU (onnxruntime-gpu), False = CPU

    Пути:
        models_dir -- папка с .onnx/.onnx.json файлами моделей (передаётся явно).
        t2v_dir    -- папка для выходных WAV-файлов (передаётся явно как output_dir).
        Обе папки создаются при первом использовании.
    """

    def __init__(
        self,
        models_dir:  Path,
        output_dir:  Path,
        model:       str        = "ru_RU-ruslan-medium",
        speaker_id:  int | None = None,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w:     float | None = None,
        use_gpu:     bool       = False,
    ):
        self.models_dir: Path = models_dir
        self.t2v_dir:    Path = output_dir

        self.model:        str        = model
        self.speaker_id:   int | None = speaker_id
        self.length_scale: float | None = length_scale
        self.noise_scale:  float | None = noise_scale
        self.noise_w:      float | None = noise_w
        self.use_gpu:      bool       = use_gpu

    # ------------------------------------------------------------------
    # Внутренние хелперы
    # ------------------------------------------------------------------

    @staticmethod
    def _model_hf_path(model_name: str) -> str:
        """
        Строит путь к файлу модели на HuggingFace из имени вида "ru_RU-denis-medium".

        Формат пути в репозитории rhasspy/piper-voices:
            {lang_family}/{lang_code}/{speaker}/{quality}/{model_name}
        Пример: ru/ru_RU/denis/medium/ru_RU-denis-medium

        Имя модели однозначно задаёт все компоненты -- парсинг без внешнего каталога.
        """
        parts = model_name.split("-", 2)
        if len(parts) != 3:
            raise ValueError(
                f"piper: неверный формат имени модели '{model_name}'. "
                "Ожидается: {lang_code}-{speaker}-{quality}, "
                "например 'ru_RU-irina-medium' или 'en_US-lessac-medium'."
            )
        lang_code, speaker, quality = parts
        lang_family = lang_code.split("_")[0]   # ru_RU -> ru
        return f"{lang_family}/{lang_code}/{speaker}/{quality}/{model_name}"

    def _ensure_model(self, model_name: str) -> Path:
        """
        Гарантирует наличие .onnx и .onnx.json файлов для модели в models_dir.
        Скачивает с HuggingFace (rhasspy/piper-voices, MIT, без регистрации) если нужно.
        Возвращает путь к .onnx файлу.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "piper: для автоматического скачивания моделей установите: "
                "pip install huggingface_hub"
            )

        self.models_dir.mkdir(parents=True, exist_ok=True)
        hf_base = self._model_hf_path(model_name)

        # hf_hub_download кладёт файл по полному суб-пути внутри local_dir:
        #   models_dir/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx
        # Проверяем и возвращаем именно этот путь, не переписываем в плоское имя.
        # Переименование ломало кэш: hf_hub_download не видел перемещённый
        # файл и скачивал его заново при каждом запуске.
        onnx_path = self.models_dir / f"{hf_base}.onnx"
        json_path = self.models_dir / f"{hf_base}.onnx.json"

        for local_path, hf_filename in (
            (onnx_path, f"{hf_base}.onnx"),
            (json_path, f"{hf_base}.onnx.json"),
        ):
            if not local_path.exists():
                hf_hub_download(
                    repo_id   = "rhasspy/piper-voices",
                    filename  = hf_filename,
                    local_dir = str(self.models_dir),
                )

        return onnx_path

    def _run_synthesis(self, text: str, output_file: str | None = None) -> str:
        """
        Ядро синтеза: скачивает модель если нужно, создаёт WAV, возвращает путь.
        Параметры синтеза берутся из атрибутов self.
        output_file=None -> t2v/t2v_<timestamp>.wav
        """
        try:
            from piper.voice import PiperVoice
            from piper import SynthesisConfig
        except ImportError:
            raise ImportError(
                "piper: установите piper-tts: pip install piper-tts"
            )
        import wave

        onnx_path = self._ensure_model(self.model)

        if output_file:
            out_path = Path(output_file)
        else:
            self.t2v_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = self.t2v_dir / f"t2v_{ts}.wav"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # use_cuda: требует onnxruntime-gpu; False -> CPU
        voice = PiperVoice.load(str(onnx_path), use_cuda=self.use_gpu)

        # SynthesisConfig: передаём только явно заданные параметры.
        # Незаданные (None) оставляем на дефолты модели.
        # noise_w_scale -- актуальное имя в piper-tts >= 1.x (устаревшее: noise_w).
        syn_kwargs: dict = {}
        if self.length_scale is not None: syn_kwargs["length_scale"]  = self.length_scale
        if self.noise_scale  is not None: syn_kwargs["noise_scale"]   = self.noise_scale
        if self.noise_w      is not None: syn_kwargs["noise_w_scale"] = self.noise_w
        syn_config = SynthesisConfig(**syn_kwargs) if syn_kwargs else None

        call_kwargs: dict = {}
        if self.speaker_id is not None: call_kwargs["speaker_id"] = self.speaker_id
        if syn_config      is not None: call_kwargs["syn_config"] = syn_config

        with wave.open(str(out_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file, **call_kwargs)

        return str(out_path)

    # ------------------------------------------------------------------
    # Публичный API (для пользователя из кода)
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text:         str,
        output_file:  str | None   = None,
        model:        str | None   = None,
        speaker_id:   int | None   = None,
        length_scale: float | None = None,
        noise_scale:  float | None = None,
        noise_w:      float | None = None,
        use_gpu:      bool | None  = None,
    ) -> str:
        """
        Синтезирует речь из text и сохраняет WAV-файл.

        Параметры, переданные явно, перекрывают текущие настройки объекта
        только для этого вызова -- дефолты не меняются.

        Параметры:
            text          -- текст для озвучивания
            output_file   -- путь к WAV-файлу (None -> t2v/t2v_<timestamp>.wav)
            model         -- имя модели (None -> self.model)
            speaker_id    -- спикер для multi-speaker моделей (None -> self.speaker_id)
            length_scale  -- скорость речи (None -> self.length_scale)
            noise_scale   -- вариативность интонации (None -> self.noise_scale)
            noise_w       -- вариативность ударений (None -> self.noise_w)
            use_gpu       -- GPU-инференс (None -> self.use_gpu)

        Возвращает путь к созданному WAV-файлу.

        Примеры:
            piper.synthesize("Привет!")
            piper.synthesize("Hello!", model="en_US-lessac-medium", length_scale=0.9)
            piper.synthesize("Текст", output_file=r"C:\\output\\speech.wav")
        """
        # Временно подменяем атрибуты если переданы явные значения,
        # восстанавливаем после синтеза -- чтобы не менять дефолты объекта.
        saved = {}
        overrides = {
            "model":        model,
            "speaker_id":   speaker_id,
            "length_scale": length_scale,
            "noise_scale":  noise_scale,
            "noise_w":      noise_w,
            "use_gpu":      use_gpu,
        }
        for attr, val in overrides.items():
            if val is not None:
                saved[attr] = getattr(self, attr)
                setattr(self, attr, val)
        try:
            return self._run_synthesis(text, output_file)
        finally:
            for attr, val in saved.items():
                setattr(self, attr, val)

    # ------------------------------------------------------------------
    # MCP builtin-инструменты (для ЛЛМ)
    # ------------------------------------------------------------------

    def as_tools(self) -> list:
        """
        Возвращает два LangChain builtin-инструмента для ЛЛМ:
            piper_synthesize  -- генерация WAV из текста
            piper_set_config  -- смена настроек синтеза
        Инструменты замкнуты на self -- видят актуальные настройки.

        Sentinel-значения вместо Optional[T] во всех сигнатурах:
            str   -> ""    (пустая строка = "не задано")
            int   -> -1    (отрицательный = "не задано")
            float -> -1.0
            bool  без None (Ollama поддерживает bool напрямую)
        Причина: Go-шаблонизатор Ollama падает с "slice index out of range"
        на Optional[T] = Union[T, None]. Тот же workaround в _mcp_schema_to_pydantic.
        """
        _self = self
        return [_make_piper_synthesize(_self), _make_piper_set_config(_self)]


# ------------------------------------------------------------------
# Фабрики инструментов -- вынесены из as_tools() чтобы @tool-декоратор
# применялся на уровне модуля, а не пересоздавался при каждом вызове.
# Замыкание на _self передаётся явным аргументом.
# ------------------------------------------------------------------

def _make_piper_synthesize(piper: PiperTTS):
    @tool
    def piper_synthesize(text: str) -> str:
        """
        Synthesize speech from text using Piper TTS.
        Saves the result as a WAV file and returns its path.

        Uses the current voice settings (model, speed, etc.).
        To change settings before synthesizing, call piper_set_config first.

        Args:
            text: The text to synthesize into speech.
        """
        return piper._run_synthesis(text)
    return piper_synthesize


def _make_piper_set_config(piper: PiperTTS) -> object:
    @tool
    def piper_set_config(
        model:        str   = "",
        speaker_id:   int   = -1,
        length_scale: float = -1.0,
        noise_scale:  float = -1.0,
        noise_w:      float = -1.0,
        use_gpu:      bool  = False,
    ) -> str:
        """
        Change Piper TTS voice settings. Call this ONLY when the user explicitly
        asks to change the voice, speed, or other synthesis parameters.
        Do NOT call this before every synthesis -- settings persist for the entire session.

        Changes are permanent for this session: once set, all subsequent
        piper_synthesize calls will use the new values until changed again.

        Only pass the parameters you actually want to change; leave others at their
        sentinel defaults (empty string / -1 / -1.0 / False) to keep current values.

        Args:
            model:        Voice model name, e.g. 'ru_RU-irina-medium' or 'en_US-lessac-medium'.
                          Leave empty to keep the current model.
            speaker_id:   Speaker index for multi-speaker models. Use -1 to keep current.
            length_scale: Speaking rate -- below 1.0 is faster, above 1.0 is slower.
                          Use -1.0 to keep current.
            noise_scale:  Intonation variability 0.0-1.0. Use -1.0 to keep current.
            noise_w:      Stress variability 0.0-1.0. Use -1.0 to keep current.
            use_gpu:      True to use GPU (requires onnxruntime-gpu), False for CPU.
        """
        changed = []

        if model:
            piper.model = model
            changed.append(f"model='{model}'")

        # speaker_id: -1 = не менять; 0 и выше -- валидные значения
        if speaker_id >= 0:
            piper.speaker_id = speaker_id
            changed.append(f"speaker_id={speaker_id}")

        # float-параметры: -1.0 = не менять; 0.0 и выше -- валидные
        if length_scale >= 0.0:
            piper.length_scale = length_scale
            changed.append(f"length_scale={length_scale}")
        if noise_scale >= 0.0:
            piper.noise_scale = noise_scale
            changed.append(f"noise_scale={noise_scale}")
        if noise_w >= 0.0:
            piper.noise_w = noise_w
            changed.append(f"noise_w={noise_w}")

        # use_gpu: False -- это и дефолт sentinel, и валидное "выключить GPU".
        # Чтобы отличить одно от другого, меняем только если True.
        if use_gpu:
            piper.use_gpu = True
            changed.append("use_gpu=True")

        if not changed:
            return "No settings changed (all values were at sentinel defaults)."
        return "Piper config updated: " + ", ".join(changed)

    return piper_set_config