"""
ralph_demo.py — конфигурация агентов для Ральф-цикла.
Аналог demo.py для AIList: только настройки, вся логика в ralph.py.

Запуск: python ralph_demo.py  (аргументы те же, что у ralph.py)
"""

import asyncio
from ailist import Provider

# Временно до нормальной установки эти 3 строки:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ralph import RalphAgent, RalphFactory, main


class RalphDemoFactory(RalphFactory):
    """Конкретная конфигурация агентов. Переопределите нужные методы под свой проект."""

    def hustler(self, **kwargs) -> RalphAgent:
        return RalphAgent("ollama:gpt-oss:20bgpu", int(32768 * 0.8), Provider.TEXT_REASONING,
                          temperature=0.0, thinking=None, **kwargs)

    def worker(self, **kwargs) -> RalphAgent:
        return RalphAgent("ollama:gpt-oss:20bgpu", int(32768 * 0.8), Provider.TEXT_REASONING,
                          temperature=0.7, thinking="medium", **kwargs)

    def thinker(self, **kwargs) -> RalphAgent:
        return RalphAgent("ollama:gpt-oss:20bgpu", int(32768 * 0.8), Provider.TEXT_REASONING,
                          temperature=1.1, thinking="max", **kwargs)


if __name__ == "__main__":
    asyncio.run(main(factory=RalphDemoFactory()))