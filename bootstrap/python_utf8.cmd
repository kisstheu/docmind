@echo off
setlocal

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set "REPO_ROOT=%~dp0.."
set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Python venv not found: "%PYTHON_EXE%" 1>&2
    exit /b 1
)

"%PYTHON_EXE%" %*
exit /b %ERRORLEVEL%
