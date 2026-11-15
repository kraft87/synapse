@echo off
REM synapse-private - toggle private mode ("off the record") for a Claude Code session.
REM Claude Code puts a plugin's bin\ on PATH, so this is just `synapse-private on <session-id>`
REM in any terminal (or `! synapse-private on <session-id>` inside a session). Locates the
REM toggle relative to itself; config comes from settings.json (no env vars, no
REM %CLAUDE_PLUGIN_ROOT%). Exits nonzero if either write fails - only exit 0 means the
REM session is genuinely off the record.
setlocal
set "SCRIPT=%~dp0..\scripts\private_mode.py"
REM synapse-python.cmd finds a Python 3.11 or newer (never the Microsoft Store
REM stand-in) and runs the script once. The call and its exit on lines of their own.
call "%~dp0synapse-python.cmd" "%SCRIPT%" %*
exit /b %errorlevel%
