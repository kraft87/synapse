@echo off
REM synapse-devices — list / mint / revoke Synapse device credentials.
REM Claude Code puts a plugin's bin\ on PATH, so this is just `synapse-devices list`
REM in any terminal (or `! synapse-devices list` inside a session). Locates the CLI relative
REM to itself; config comes from settings.json (no env vars, no %CLAUDE_PLUGIN_ROOT%).
REM Requires a FULL-TRUST device token — the shared machine token is refused by design.
setlocal
set "SCRIPT=%~dp0..\scripts\device_admin.py"
where python >nul 2>nul && (
  python "%SCRIPT%" %*
  exit /b %errorlevel%
)
where python3 >nul 2>nul && (
  python3 "%SCRIPT%" %*
  exit /b %errorlevel%
)
echo synapse-devices: need python or python3 on PATH 1>&2
exit /b 1
