@echo off
REM synapse-login — run the Synapse device-flow login directly, no LLM in the loop.
REM Claude Code puts a plugin's bin\ on PATH, so this is just `synapse-login` in any terminal.
REM Locates the login script relative to itself; config comes from settings.json (no env vars,
REM no %CLAUDE_PLUGIN_ROOT%). Run it in a terminal for live output (prints a device code, polls).
setlocal
set "SCRIPT=%~dp0..\scripts\synapse_login.py"
where python >nul 2>nul && (
  python "%SCRIPT%" %*
  exit /b %errorlevel%
)
where python3 >nul 2>nul && (
  python3 "%SCRIPT%" %*
  exit /b %errorlevel%
)
echo synapse-login: need python or python3 on PATH 1>&2
exit /b 1
