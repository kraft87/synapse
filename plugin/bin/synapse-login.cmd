@echo off
REM synapse-login - `synapse login`: sign in, enroll this machine, wire its harnesses.
REM Claude Code puts a plugin's bin\ on PATH inside sessions. Locates the CLI relative to
REM itself; config comes from ~/.synapse/config.json (no env vars, no %CLAUDE_PLUGIN_ROOT%).
setlocal
set "SCRIPT=%~dp0..\scripts\synapse_cli.py"
REM Each call and its exit on a line of its own: cmd expands %errorlevel% in a
REM parenthesised block before the block runs.
where python >nul 2>nul || goto try3
python "%SCRIPT%" login %*
exit /b %errorlevel%
:try3
where python3 >nul 2>nul || goto none
python3 "%SCRIPT%" login %*
exit /b %errorlevel%
:none
echo synapse-login: need python or python3 on PATH 1>&2
exit /b 1
