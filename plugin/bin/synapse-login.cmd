@echo off
REM synapse-login - `synapse login`: sign in, enroll this machine, wire its harnesses.
REM Claude Code puts a plugin's bin\ on PATH inside sessions. Locates the CLI relative to
REM itself; config comes from ~/.synapse/config.json (no env vars, no %CLAUDE_PLUGIN_ROOT%).
setlocal
set "SCRIPT=%~dp0..\scripts\synapse_cli.py"
REM synapse-python.cmd finds a Python 3.11 or newer (never the Microsoft Store
REM stand-in) and runs the script once. The call and its exit on lines of their own.
call "%~dp0synapse-python.cmd" "%SCRIPT%" login %*
exit /b %errorlevel%
