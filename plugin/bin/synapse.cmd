@echo off
REM synapse - the Synapse client CLI (login, config, install, update, skill retire).
REM Claude Code puts a plugin's bin\ on PATH inside sessions; outside them the managed
REM client's shim (%USERPROFILE%\.synapse\bin\synapse.cmd) is the one on PATH.
setlocal
set "SCRIPT=%~dp0..\scripts\synapse_cli.py"
REM synapse-python.cmd finds a Python 3.11 or newer (never the Microsoft Store
REM stand-in) and runs the script once. The call and its exit on lines of their own.
call "%~dp0synapse-python.cmd" "%SCRIPT%" %*
exit /b %errorlevel%
