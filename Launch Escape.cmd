@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m flybrain.escape_repair
if errorlevel 1 pause
