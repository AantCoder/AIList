from langchain.tools import tool
from ailist import AIList, Provider

# --------------------------------------------------
# Individual part


@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def wait_install() -> str:
    """Используй этот метод, чтобы немного подождать установки, если какой-то инструмент попросил подождать. 
    После этого метода проверь тот инструмент ещё раз. Если он опять попросит подождать, то ты можешь запускать это ожидание снова, пока он не установиться, но не более 10 раз."""
    import time
    time.sleep(30)
    return "Short wait complete."

DEFAULT_TOOLS = [get_weather_for_location, wait_install]


# --------------------------------------------------
# AIListDemo -- готовая к запуску конфигурация для разработки и тестирования.
# Наследует AIList и предустанавливает конкретную модель, loglevel и базовые инструменты.
# Для продакшн-проектов создайте свой подкласс AIList с нужной моделью и инструментами.

class AIListDemo(AIList):
    """
    Готовая к запуску конфигурация AIList для разработки и тестирования.

    Предустанавливает:
      - Модель: ollama:gpt-oss:20bgpu (TEXT_REASONING, context 16384 * 0.8)
      - loglevel = 2 (полный лог: токены + история раунда)
      - DEFAULT_TOOLS: get_weather_for_location, wait_install

    Для продакшн-проектов создайте собственный подкласс AIList:

        class MyAgent(AIList):
            def __init__(self, tools=None, context_schema=None):
                super().__init__(
                    modelName      = "openai:gpt-4o",
                    context_limit  = int(128000 * 0.8),
                    provider       = Provider.OPENAI,
                    tools          = tools or [],
                    context_schema = context_schema,
                )
                self.loglevel = 1
                # ... свои промпты и серверы

    Использование:
        async with AIListDemo() as ai:
            await ai.mcp_connect("workspace")
            print(await ai.run_async("Привет!"))
    """

    def __init__(self, tools=None, context_schema=None):
        if tools is None:
            tools = list(DEFAULT_TOOLS)

        #super().__init__("ollama:qwen3.5:35b-a3b", int(65536*0.8), Provider.QWEN, tools, context_schema)
        #super().__init__("ollama:gpt-oss:20bgpu", int(16384*0.8), Provider.TEXT_REASONING, tools, context_schema)
        #super().__init__("ollama:qwen", int(16384*0.8), Provider.QWEN, tools, context_schema)

        super().__init__(
            modelName      = "ollama:gpt-oss:20bgpu",
            context_limit  = int(16384 * 0.8),
            provider       = Provider.TEXT_REASONING,
            tools          = tools,
            context_schema = context_schema,
        )
        self.loglevel = 2