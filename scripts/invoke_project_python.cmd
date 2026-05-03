@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo Missing .venv Python. Run scripts\bootstrap_windows.ps1 first.
  exit /b 1
)

"%VENV_PY%" %*
exit /b %ERRORLEVEL%
