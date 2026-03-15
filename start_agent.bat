@echo off
title Pokemon Agent
cd /d "%~dp0"
call venv\Scripts\activate.bat
python -u agent.py
pause
