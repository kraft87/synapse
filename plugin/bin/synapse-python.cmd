@echo off
REM synapse-python: run a Synapse client script under Python 3.11 or newer, exactly once.
REM The cmd twin of synapse-python: Claude Code runs the MCP headers helper through cmd on
REM Windows, and PowerShell finds this file for `synapse-python` too. Same lookup: the first
REM of python3, python and py -3 that runs Python 3.11 or newer, so the Microsoft Store
REM stand-in ("Python was not found") and older Pythons are skipped. The script runs once;
REM its output and exit status are its own. ASCII only: cmd reads a batch file in the
REM console code page.
setlocal
REM Never a python.exe from the current directory (the repository a session opened).
set "NoDefaultCurrentDirectoryInExePath=1"
set "SYNAPSE_PY="
call :probe python3
if not defined SYNAPSE_PY call :probe python
if not defined SYNAPSE_PY call :probe py -3
if not defined SYNAPSE_PY goto none
REM The call and its exit on lines of their own: cmd expands %errorlevel% in a
REM parenthesised block before the block runs.
"%SYNAPSE_PY%" %*
exit /b %errorlevel%

:probe
for /f "usebackq delims=" %%P in (`%* -c "import sys; sys.version_info >= (3, 11) and sys.stdout.write(sys.executable)" 2^>nul ^<nul`) do set "SYNAPSE_PY=%%P"
exit /b 0

:none
echo Synapse: no Python 3.11 or newer found (tried python3, python and py -3). Install Python from python.org, then open a new terminal. 1>&2
exit /b 1
