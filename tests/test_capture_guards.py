"""ingestion.capture.guards: the URLs and tools server-side capture never stores."""

from __future__ import annotations

import pytest

from ingestion.capture.guards import is_public_url, tool_exclusion


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/a",
        "http://docs.python.org/3/library/",
        "https://sub.domain.co.uk:8443/x?y=1#z",
        "https://8.8.8.8/dns",
        "https://[2606:4700:4700::1111]/",
        "https://xn--bcher-kva.example.org/",
        "https://localhost.example.com/",  # a public name that merely starts like one
    ],
)
def test_public_urls_pass(url):
    assert is_public_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:3000/",
        "http://LOCALHOST/",
        "http://127.0.0.1/",
        "http://127.1.2.3:8080/",
        "http://10.0.0.5/",
        "http://172.16.4.4/",
        "http://192.168.1.1/admin",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata, link-local
        "http://100.64.0.1/",  # carrier-grade NAT, the Tailscale range
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[fd00::1]/",
        "http://[fe80::1]/",
        "http://[::ffff:10.0.0.1]/",  # v4 private address in a v6 wrapper
        "http://intranet/wiki",  # single-label host
        "http://printer.local/",
        "http://nas.lan:5000/",
        "http://wiki.home/",
        "http://db.internal/",
        "http://box.tailnet-name.ts.net/",
        "http://thing.localdomain/",
        "http://router.home.arpa/",
        "http://2130706433/",  # 127.0.0.1 as one decimal number
        "http://0x7f.0.0.1/",
        "https://user:secret@example.com/",  # credentials in the URL
        "ftp://example.com/file",
        "file:///etc/passwd",
        "example.com/no-scheme",
        "https:///nohost",
        "",
    ],
)
def test_non_public_urls_are_refused(url):
    assert not is_public_url(url)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("mcp__Synapse__recall", "own_tool"),
        ("mcp__Synapse__fetch", "own_tool"),
        ("mcp__plugin_synapse_synapse__recall_full_turns", "own_tool"),
        ("mcp__memory__recall_feedback", "own_tool"),  # a server renamed beyond recognition
        ("Task", "subagent"),
        ("Agent", "subagent"),
        ("mcp__playwright__browser_navigate", "browser"),
        ("mcp__puppeteer__puppeteer_navigate", "browser"),
        ("mcp__claude-in-chrome__navigate", "browser"),
        ("mcp__Claude_Browser__open", "browser"),
        ("mcp__claude_ai_Gmail__search_threads", "denied_family"),
        ("mcp__Google_Calendar__list_events", "denied_family"),
        ("mcp__Google_Drive__search_files", "denied_family"),
        ("mcp__slack__search_messages", "denied_family"),
        ("mcp__github__search_code", "denied_family"),
        ("mcp__plugin_linear_linear__list_issues", "denied_family"),
        ("mcp__notion__notion-search", "denied_family"),
        ("Bash", "shell"),
        ("PowerShell", "shell"),
        ("shell", "shell"),
        ("exec_command", "shell"),
        # Research tools, under any connector name
        ("WebFetch", None),
        ("WebSearch", None),
        ("mcp__claude_ai_firecrawl__firecrawl_scrape", None),
        ("mcp__firecrawl__firecrawl_search", None),
        ("mcp__Exa_search__web_search_exa", None),
        ("mcp__fetch__fetch", None),  # the reference fetch server, not Synapse's fetch
        ("mcp__brave__brave_web_search", None),
        ("mcp__tavily__tavily-search", None),
    ],
)
def test_tool_exclusion(name, expected):
    assert tool_exclusion(name, {}) == expected


def test_a_tool_that_takes_a_command_is_a_shell_whatever_it_is_called():
    assert tool_exclusion("mcp__ops__run", {"command": "curl https://example.com"}) == "shell"
    assert tool_exclusion("mcp__ops__run", {"url": "https://example.com"}) is None
