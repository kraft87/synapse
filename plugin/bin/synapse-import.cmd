@echo off
REM synapse-import - bulk-import existing Claude Code transcripts into Synapse, no LLM in the loop.
REM Claude Code puts a plugin's bin\ on PATH, so this is just `synapse-import` in any terminal
REM (or `! synapse-import` inside a session). Locates the import script relative to itself;
REM config comes from settings.json (no env vars, no %CLAUDE_PLUGIN_ROOT%). Prints a summary
REM and asks for confirmation first; the server dedups by span_id, so re-running is always safe.
setlocal
set "SCRIPT=%~dp0..\scripts\import_history.py"
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
echo synapse-import: need python or python3 on PATH 1>&2
exit /b 1
