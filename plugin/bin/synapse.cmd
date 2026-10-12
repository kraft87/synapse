@echo off
REM synapse - the Synapse client CLI (login, config, install, update, skill retire).
REM Claude Code puts a plugin's bin\ on PATH inside sessions; outside them the managed
REM client's shim (%USERPROFILE%\.synapse\bin\synapse.cmd) is the one on PATH.
setlocal
set "SCRIPT=%~dp0..\scripts\synapse_cli.py"
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
echo synapse: need python or python3 on PATH 1>&2
exit /b 1
