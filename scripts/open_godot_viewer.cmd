@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0open_godot_viewer.ps1" %*
exit /b %ERRORLEVEL%
