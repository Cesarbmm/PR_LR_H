@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
set "PATH=%REPO_ROOT%\.venv\Scripts;%PATH%"

echo Project venv is first in PATH:
where python
echo.
echo Use: scripts\invoke_project_python.cmd -c "import torch; print(torch.cuda.is_available())"
echo Use: scripts\open_godot_viewer.cmd -Play
echo.
cmd /k
