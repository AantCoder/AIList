rd /S /Q ralph_project
call py312 ralph_demo.py --max-iter 20 "Дай 10 оригенальных сюжета для космической фантастики. Результат запиши в текстовые файлы"
@echo.
@cd ralph_project
git log --oneline --graph --all  && git log --stat --format="%%H %%s" -10 
@cd ..
pause