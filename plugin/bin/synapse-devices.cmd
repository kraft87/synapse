@echo off
REM synapse-devices - list / mint / revoke Synapse device credentials.
REM Claude Code puts a plugin's bin\ on PATH, so this is just `synapse-devices list`
REM in any terminal (or `! synapse-devices list` inside a session). Locates the CLI relative
REM to itself; config comes from settings.json (no env vars, no %CLAUDE_PLUGIN_ROOT%).
REM Requires a FULL-TRUST device token - the shared machine token is refused by design.
setlocal
set "SCRIPT=%~dp0..\scripts\device_admin.py"
REM Each call and its exit on a line of its own: cmd expands %errorlevel% in a
REM parenthesised block before the block runs.
where python >nul 2>nul || goto try3
python "%SCRIPT%" %*
exit /b %errorlevel%
:try3
where python3 >nul 2>nul || goto none
python3 "%SCRIPT%" %*
exit /b %errorlevel%
:none
echo synapse-devices: need python or python3 on PATH 1>&2
exit /b 1
