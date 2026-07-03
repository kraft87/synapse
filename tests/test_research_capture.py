"""The research-archive capture must match exactly the three source families,
verbatim from tool results, and nothing else."""

from __future__ import annotations

from ingestion.research_capture import _first_author, classify_command


def test_classifies_bird_thread():
    kind, sid, url = classify_command("bird thread https://x.com/foo/status/123456789012 --json")
    assert (kind, sid) == ("x_thread", "thread:x:123456789012")
    assert "123456789012" in url
    # bare numeric id form
    kind, sid, _ = classify_command("bird thread 987654321098 --json 2>/dev/null")
    assert (kind, sid) == ("x_thread", "thread:x:987654321098")


def test_classifies_bsky_thread():
    kind, sid, _ = classify_command(
        '"$BSKY" thread https://bsky.app/profile/druce.ai/post/3mki3hqjyj22d --json'
    )
    assert (kind, sid) == ("bsky_thread", "thread:bsky:3mki3hqjyj22d")
    kind, sid, _ = classify_command(
        "bsky thread at://did:plc:abc/app.bsky.feed.post/3kxyz123 --json"
    )
    assert (kind, sid) == ("bsky_thread", "thread:bsky:3kxyz123")


def test_classifies_yt_transcript_cleaning_step_only():
    # The python cleaning step (has the WEBVTT marker + /tmp/yt_<id> glob) matches...
    cmd = "python3 -c \"import re, glob; f = glob.glob('/tmp/yt_dQw4w9WgXcQ*.vtt'); re.sub(r'WEBVTT', '', '')\""
    kind, sid, url = classify_command(cmd)
    assert (kind, sid) == ("yt_transcript", "transcript:dQw4w9WgXcQ")
    assert "dQw4w9WgXcQ" in url
    # ...but the yt-dlp download itself (empty stdout) does not.
    assert classify_command('yt-dlp --write-auto-sub -o "/tmp/yt_%(id)s" <url>') is None
    # nor does a stray ls of the cache dir
    assert classify_command("ls /tmp/yt_dQw4w9WgXcQ.vtt") is None


def test_ignores_everything_else():
    assert classify_command("git status") is None
    assert classify_command("bird search 'memory systems' -n 15 --json") is None
    assert classify_command("bsky search 'zep' --json") is None
    assert classify_command("echo bird thread is a concept") is None  # no id-shaped arg


def test_first_author_best_effort():
    assert _first_author("x_thread", '[{"author": {"username": "steipete"}, "text": "hi"}]') == (
        "@steipete"
    )
    assert _first_author("x_thread", "=== THREAD: formatted by an agent ===") is None
    assert _first_author("yt_transcript", "anything") is None
