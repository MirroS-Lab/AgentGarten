@echo off
rem Windows entry point for scripts\run_normalcrafter.sh (bash/ssh/scp underneath -- runs through Git Bash).
rem Usage: scripts\run_normalcrafter.bat VIDEO_PATH [--max-res 864] [--host user@host] [--remote-dir PATH]
setlocal
where bash >nul 2>&1
if errorlevel 1 (
  echo ERROR: bash not found on PATH. Install Git for Windows ^(provides Git Bash^) first.
  exit /b 1
)
bash "%~dp0run_normalcrafter.sh" %*
exit /b %ERRORLEVEL%
