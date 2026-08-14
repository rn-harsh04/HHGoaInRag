$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Write-Host "Starting Voice RAG Backend on port 7860..." -ForegroundColor Cyan
& "$ScriptDir\.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --port 7860
