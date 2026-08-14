@echo off
cd /d "%~dp0"
echo Starting Voice RAG Backend on port 7860...
"%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --port 7860
pause
