@echo off
REM synapse-private — toggle private mode ("off the record") for a Claude Code session.
REM Claude Code puts a plugin's bin\ on PATH, so this is just `synapse-private on <session-id>`
REM in any terminal (or `! synapse-private on <session-id>` inside a session). Locates the
REM toggle relative to itself; config comes from settings.json (no env vars, no
REM %CLAUDE_PLUGIN_ROOT%). Exits nonzero if either write fails — only exit 0 means the
REM session is genuinely off the record.
setlocal
set "SCRIPT=%~dp0..\scripts\private_mode.py"
where python >nul 2>nul && (
  python "%SCRIPT%" %*
  exit /b %errorlevel%
)
where python3 >nul 2>nul && (
  python3 "%SCRIPT%" %*
  exit /b %errorlevel%
)
echo synapse-private: need python or python3 on PATH 1>&2
exit /b 1
