rd /S /Q ralph_project
call py312 ralph_demo.py --max-iter 30 "Создай оффлайн html страницу с красивым калькулятором"
@echo.
@cd ralph_project
git log --oneline --graph --all  && git log --stat --format="%%H %%s" -10 
@cd ..
pause