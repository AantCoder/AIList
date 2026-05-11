rem === If the alist project is not installed, then before running the tests you need to install:
cd ..
uv run --python 3.11 -m hatchling
uv run --python 3.11 -m editables
uv run --python 3.11 -m -e .

rem === Or if you have py312.cmd:
rem cd ..
rem py312 -m pip install hatchling
rem py312 -m pip install editables
rem py312 -m pip install -e .

rem === Example of running a test with UTF-8 encoding
rem cmd /k chcp 65001 & py312 test_integration_part1.py



