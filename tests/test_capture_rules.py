"""ingestion.capture.ruleset and the shipped rules: sources the shapes cannot see, taught by
files. No database: the rules decide, ``decide`` reports what they would store."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ingestion.capture import ToolCall, decide
from ingestion.capture import ruleset as ruleset_mod
from ingestion.capture.detect import Capture, Skip
from ingestion.capture.ruleset import RuleError, active_rules, load_rules
from ingestion.web_artifacts import ArchiveEntry
from ingestion.web_extractors import WebScrape

FIXTURES = Path(__file__).parent / "fixtures" / "capture_rules"
PAGE = "# A Real Article\n\n" + "The measured latency was low and the design held up well. " * 12


def call(name: str, tool_input: dict, text: str, **kw) -> ToolCall:
    fields = dict(
        tool_use_id="toolu_1",
        tool_name=name,
        tool_input=tool_input,
        result_text=text,
        is_error=False,
        session_id="s1",
        project="proj",
        timestamp=None,
    )
    return ToolCall(**{**fields, **kw})


def bash(command: str, text: str) -> ToolCall:
    return call("Bash", {"command": command}, text)


def archived(c: ToolCall) -> tuple[Capture, ArchiveEntry]:
    d = decide(c)
    assert isinstance(d, Capture) and isinstance(d.parsed, ArchiveEntry), f"not archived: {d}"
    assert d.destination == "archive"
    return d, d.parsed


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text)
    return path


@pytest.fixture(autouse=True)
def _no_operator_dir(monkeypatch):
    monkeypatch.delenv("SYNAPSE_CAPTURE_DIR", raising=False)
    load_rules.cache_clear()
    yield
    load_rules.cache_clear()


@pytest.fixture
def operator(tmp_path, monkeypatch) -> Path:
    """An operator rules directory, active for the test."""
    monkeypatch.setenv("SYNAPSE_CAPTURE_DIR", str(tmp_path))
    return tmp_path


SEARCHABLE_RULE = """
destination = "searchable"
{extra}
[match]
tool = ["mcp__acme__*", "Fetch"]
[fields]
url = "{{input.link}}"
title = "{{json.page.title}}"
body = "{{json.page.markdown}}"
"""


def searchable(extra: str = "", tool: str = "mcp__acme__*") -> str:
    return SEARCHABLE_RULE.format(extra=extra).replace("mcp__acme__*", tool)


# ---------- the shipped rules, against fixtures ----------


class TestShippedRules:
    def test_the_shipped_set_is_the_three_threads_and_the_deny_list(self):
        rules = active_rules()
        assert [r.name for r in rules.rules] == ["bsky_thread", "x_thread", "yt_transcript"]
        assert {r.destination for r in rules.rules} == {"archive"}
        assert {"gmail", "slack", "github"} <= rules.deny_words

    @pytest.mark.parametrize(
        "command",
        [
            "bird thread https://x.com/steipete/status/1790000000000000001 --json",
            "bird thread 1790000000000000001 --json 2>/dev/null",
            "cd /tmp && bird thread 'https://twitter.com/x/status/1790000000000000001?s=20' --json",
        ],
    )
    def test_an_x_thread_is_archived_verbatim(self, command):
        text = (FIXTURES / "x_thread.json").read_text()
        c, entry = archived(bash(command, text))
        assert c.source == "x_thread" and not c.kg_extract
        assert (entry.kind, entry.source_id) == ("x_thread", "thread:x:1790000000000000001")
        assert entry.url == "https://x.com/i/status/1790000000000000001"
        assert entry.author == "@steipete"
        assert entry.content == text.strip()

    def test_a_bluesky_thread_keeps_its_page_url(self):
        text = (FIXTURES / "bsky_thread.json").read_text()
        url = "https://bsky.app/profile/druce.ai/post/3mki3hqjyj22d"
        for command in (f'"$BSKY" thread {url} --json', f"bsky thread '{url}' --json"):
            _, entry = archived(bash(command, text))
            assert (entry.kind, entry.source_id) == ("bsky_thread", "thread:bsky:3mki3hqjyj22d")
            assert entry.url == url
            assert entry.author == "@druce.ai"

    def test_a_bluesky_at_uri_links_to_a_search_for_the_post(self):
        text = (FIXTURES / "bsky_thread.json").read_text()
        command = "bsky thread at://did:plc:abc/app.bsky.feed.post/3kxyz123 --json"
        _, entry = archived(bash(command, text))
        assert entry.source_id == "thread:bsky:3kxyz123"
        assert entry.url == "https://bsky.app/search?q=3kxyz123"

    def test_a_youtube_transcript_is_the_cleaning_step_not_the_download(self):
        text = (FIXTURES / "yt_transcript.txt").read_text()
        cleaning = (
            "python3 -c \"import re, glob; f = glob.glob('/tmp/yt_dQw4w9WgXcQ*.vtt'); "
            "re.sub(r'WEBVTT', '', open(f[0]).read())\""
        )
        _, entry = archived(bash(cleaning, text))
        assert (entry.kind, entry.source_id) == ("yt_transcript", "transcript:dQw4w9WgXcQ")
        assert entry.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert entry.author is None
        # the yt-dlp download prints nothing worth keeping, and an ls is no transcript
        for other in (
            'yt-dlp --write-auto-sub -o "/tmp/yt_%(id)s" URL',
            "ls /tmp/yt_dQw4w9WgXcQ.vtt",
        ):
            d = decide(bash(other, text))
            assert isinstance(d, Skip) and d.reason == "shell"

    @pytest.mark.parametrize(
        "command",
        [
            "git status",
            "bird search 'memory systems' -n 15 --json",
            "bsky search 'zep' --json",
            "echo bird thread is a concept",
        ],
    )
    def test_other_commands_stay_uncaptured(self, command):
        d = decide(bash(command, (FIXTURES / "x_thread.json").read_text()))
        assert isinstance(d, Skip) and d.reason == "shell"

    def test_a_short_result_is_not_kept(self):
        d = decide(bash("bird thread 1790000000000000001 --json", '{"error": "rate limited"}'))
        assert isinstance(d, Skip) and d.reason == "too_short"

    def test_a_failed_command_is_not_kept(self):
        d = decide(
            call(
                "Bash",
                {"command": "bird thread 1790000000000000001"},
                (FIXTURES / "x_thread.json").read_text(),
                is_error=True,
            )
        )
        assert isinstance(d, Skip)

    def test_codex_shell_commands_match_too(self):
        text = (FIXTURES / "x_thread.json").read_text()
        for tool_input in (
            {"cmd": "bird thread 1790000000000000001 --json"},  # exec_command
            {"command": ["bash", "-lc", "bird thread 1790000000000000001 --json"]},  # argv
        ):
            c = call("exec_command", tool_input, text)
            assert isinstance(decide(c), Capture)


# ---------- what a rule is held to ----------


class TestRulesAreHeldToTheBasics:
    def test_a_rule_cannot_reach_synapse_s_own_tools(self, operator):
        write(operator, "mine.toml", searchable(tool="mcp__*__recall"))
        out = json.dumps({"page": {"title": "t", "markdown": PAGE}})
        d = decide(call("mcp__Synapse__recall", {"link": "https://example.com/a"}, out))
        assert isinstance(d, Skip) and d.reason == "own_tool"

    def test_a_rule_stores_only_public_urls(self, operator):
        write(operator, "mine.toml", searchable())
        out = json.dumps({"page": {"title": "t", "markdown": PAGE}})
        d = decide(call("mcp__acme__read", {"link": "http://intranet.local/wiki"}, out))
        assert isinstance(d, Skip) and d.reason == "non_public_url"

    def test_an_archive_url_must_be_public_too(self, operator):
        write(operator, "mine.toml", ARCHIVE_RULE.replace("https://", "http://localhost/"))
        d = decide(bash("dump thread 4242", "x" * 300))
        assert isinstance(d, Skip) and d.reason == "non_public_url"

    def test_a_rule_whose_fields_do_not_fill_leaves_the_call_to_the_next_stage(self, operator):
        write(operator, "mine.toml", searchable())
        # No "link" in the input: the rule has nothing to say, the shapes take it from here
        out = json.dumps({"page": {"title": "t", "markdown": PAGE}})
        d = decide(call("mcp__acme__read", {"url": "https://example.com/a"}, out))
        assert isinstance(d, Capture) and d.source == "generic"

    def test_a_rule_overrides_the_generic_detector_and_the_deny_list(self, operator):
        write(operator, "mine.toml", searchable(tool="mcp__gmail__*"))
        out = json.dumps({"page": {"title": "T", "markdown": PAGE}})
        c = call("mcp__gmail__read", {"link": "https://example.com/a"}, out)
        d = decide(c)
        assert isinstance(d, Capture) and d.source == "mine"
        # the generic detector alone would have refused the connector
        write(operator, "mine.toml", "enabled = false\n")
        load_rules.cache_clear()
        denied = decide(call("mcp__gmail__read", {"url": "https://example.com/a"}, PAGE))
        assert isinstance(denied, Skip) and denied.reason == "denied_family"


ARCHIVE_RULE = """
destination = "archive"
[match]
command = 'dump thread (?P<n>\\d+)'
[fields]
url = "https://example.org/t/{n}"
source_id = "dump:{n}"
body = "{output}"
"""


# ---------- operator rules ----------


class TestOperatorRules:
    def test_a_searchable_rule_stores_a_page_and_the_graph_only_if_it_says_so(self, operator):
        write(operator, "docs_site.toml", searchable())
        out = json.dumps({"page": {"title": "From JSON", "markdown": PAGE}})
        c = call("mcp__acme__read", {"link": "https://example.com/doc"}, out)
        d = decide(c)
        assert isinstance(d, Capture) and d.destination == "searchable" and not d.kg_extract
        assert d.source == "docs_site"
        assert isinstance(d.parsed, WebScrape)
        assert (d.parsed.url, d.parsed.title) == ("https://example.com/doc", "From JSON")
        assert d.parsed.content_markdown.startswith("# A Real Article")

        write(operator, "docs_site.toml", searchable("kg_extract = true"))
        load_rules.cache_clear()
        trusted = decide(c)
        assert isinstance(trusted, Capture) and trusted.kg_extract

    def test_a_rule_body_is_held_to_the_quality_floors(self, operator):
        write(operator, "docs_site.toml", searchable())
        short = json.dumps({"page": {"title": "t", "markdown": "tiny page"}})
        d = decide(call("mcp__acme__read", {"link": "https://example.com/a"}, short))
        assert isinstance(d, Skip) and d.reason == "too_short"
        wall = json.dumps({"page": {"title": "t", "markdown": "Please log in to continue"}})
        d = decide(call("mcp__acme__read", {"link": "https://example.com/a"}, wall))
        assert isinstance(d, Skip) and d.reason == "blocked_page"

    def test_min_chars_is_the_rule_s_to_set(self, operator):
        write(operator, "docs_site.toml", searchable("min_chars = 5"))
        out = json.dumps({"page": {"title": "t", "markdown": "a short page"}})
        assert isinstance(
            decide(call("mcp__acme__read", {"link": "https://example.com/a"}, out)), Capture
        )

    def test_an_operator_rule_overrides_the_shipped_rule_of_the_same_name(self, operator):
        write(
            operator,
            "x_thread.toml",
            ARCHIVE_RULE.replace("dump:{n}", "mine:{n}").replace("dump thread", "bird thread"),
        )
        _, entry = archived(bash("bird thread 4242", "x" * 300))
        assert entry.source_id == "mine:4242"
        assert entry.kind == "x_thread"  # the kind is the rule's name
        assert [r.name for r in active_rules().rules] == [
            "bsky_thread",
            "x_thread",
            "yt_transcript",
        ]

    def test_enabled_false_turns_a_shipped_rule_off(self, operator):
        write(operator, "x_thread.toml", "enabled = false\n")
        d = decide(
            bash("bird thread 1790000000000000001", (FIXTURES / "x_thread.json").read_text())
        )
        assert isinstance(d, Skip) and d.reason == "shell"
        assert [r.name for r in active_rules().rules] == ["bsky_thread", "yt_transcript"]

    def test_the_first_matching_rule_in_name_order_decides(self, operator):
        write(operator, "a_first.toml", ARCHIVE_RULE.replace("dump:{n}", "first:{n}"))
        write(operator, "b_second.toml", ARCHIVE_RULE.replace("dump:{n}", "second:{n}"))
        _, entry = archived(bash("dump thread 7", "x" * 300))
        assert entry.source_id == "first:7"

    def test_an_input_field_a_kind_and_alternatives(self, operator):
        write(
            operator,
            "pages.toml",
            """
            destination = "archive"
            kind = "wiki"
            [match]
            tool = "mcp__wiki__get"
            [fields]
            url = "{input.page_url}"
            source_id = "wiki:{input.page_id}"
            title = ["{json.title}", "{json.name}"]
            body = "{output}"
            """,
        )
        text = json.dumps({"name": "The page", "text": "y" * 300})
        c = call("mcp__wiki__get", {"page_url": "https://wiki.example.com/p", "page_id": 12}, text)
        _, entry = archived(c)
        assert (entry.kind, entry.source_id, entry.title) == ("wiki", "wiki:12", "The page")

    def test_the_operator_deny_list_replaces_the_shipped_one(self, operator):
        write(operator, "deny.toml", 'words = ["acme"]\n')
        assert active_rules().deny_words == {"acme"}
        d = decide(call("mcp__acme__fetch", {"url": "https://example.com/a"}, PAGE))
        assert isinstance(d, Skip) and d.reason == "denied_family"
        # gmail is no longer denied: this operator's copy of the list is theirs
        assert isinstance(
            decide(call("mcp__gmail__fetch", {"url": "https://example.com/a"}, PAGE)), Capture
        )

    def test_an_operator_dir_that_is_not_a_directory_is_an_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SYNAPSE_CAPTURE_DIR", str(tmp_path / "missing"))
        with pytest.raises(RuleError, match=r"SYNAPSE_CAPTURE_DIR=.*missing is not a directory"):
            active_rules()


# ---------- hooks ----------


HOOK = """
def extract(tool_input, output):
    if "ticket" not in tool_input:
        return None
    return {"url": "https://tickets.example.com/" + tool_input["ticket"],
            "source_id": "ticket:" + tool_input["ticket"], "body": output.upper()}
"""


class TestHooks:
    def rule(self, operator, hook: str = "tickets.py:extract") -> None:
        write(operator, "tickets.py", HOOK)
        write(
            operator,
            "tickets.toml",
            f'destination = "archive"\nhook = "{hook}"\n[match]\ntool = "mcp__desk__*"\n',
        )

    def test_a_hook_supplies_the_fields_a_rule_cannot_express(self, operator):
        self.rule(operator)
        c = call("mcp__desk__open", {"ticket": "T-9"}, "the whole ticket thread " * 20)
        _, entry = archived(c)
        assert entry.source_id == "ticket:T-9"
        assert entry.content.startswith("THE WHOLE TICKET THREAD")

    def test_a_hook_that_returns_none_declines_the_call(self, operator):
        self.rule(operator)
        d = decide(call("mcp__desk__open", {"other": 1}, "the whole ticket thread " * 20))
        assert isinstance(d, Skip) and d.reason == "no_source"

    def test_a_hook_that_returns_unknown_fields_is_an_error_naming_the_rule_file(self, operator):
        self.rule(operator)
        write(operator, "tickets.py", HOOK.replace('"body"', '"text"'))
        load_rules.cache_clear()
        with pytest.raises(RuleError, match=r"tickets\.toml: hook returned"):
            decide(call("mcp__desk__open", {"ticket": "T-9"}, "the whole ticket thread " * 20))

    @pytest.mark.parametrize(
        "spec,problem",
        [
            ("tickets.py", "expected 'file.py:function'"),
            ("../tickets.py:extract", "the file next to the rule"),
            ("tickets.py:extract()", "the file next to the rule"),
            ("missing.py:extract", "does not exist"),
            ("tickets.py:nothing", "has no function 'nothing'"),
        ],
    )
    def test_a_bad_hook_names_the_rule_and_the_problem(self, operator, spec, problem):
        write(operator, "tickets.py", HOOK)
        write(
            operator,
            "tickets.toml",
            f'destination = "archive"\nhook = "{spec}"\n[match]\ntool = "x"\n',
        )
        with pytest.raises(RuleError, match=rf"tickets\.toml: hook: .*{problem}"):
            active_rules()

    def test_a_hook_that_fails_to_import_is_an_error(self, operator):
        write(operator, "tickets.py", "raise RuntimeError('boom')\n")
        write(
            operator,
            "tickets.toml",
            'destination = "archive"\nhook = "tickets.py:extract"\n[match]\ntool = "x"\n',
        )
        with pytest.raises(RuleError, match=r"tickets\.toml: hook: importing tickets\.py failed"):
            active_rules()

    def test_a_rule_cannot_have_fields_and_a_hook(self, operator):
        write(operator, "tickets.py", HOOK)
        text = 'hook = "tickets.py:extract"\n' + ARCHIVE_RULE
        write(operator, "tickets.toml", text)
        with pytest.raises(
            RuleError, match=r"tickets\.toml: hook: a rule has \[fields\] or a hook"
        ):
            active_rules()


# ---------- validation: the file and the field are named ----------


@pytest.mark.parametrize(
    "text,where",
    [
        ("this is = not toml [", "file"),
        ('enabled = "no"\n', "enabled: expected true or false"),
        ('destination = "elsewhere"\n', "destination: must be one of searchable, archive"),
        ('destination = "archive"\n', "[match]: missing"),
        ('destination = "archive"\n[match]\n', "[match]: needs a tool glob or a command regex"),
        ('destination = "archive"\n[match]\ntool = 3\n', "[match] tool: expected a string"),
        (
            'destination = "archive"\n[match]\ncommand = "("\n[fields]\n',
            "[match] command: not a regex",
        ),
        (
            'destination = "archive"\n[match]\ntool = "x"\n',
            "[fields]: missing: say where the values come from, or name a hook",
        ),
        (
            'destination = "archive"\n[match]\ntool = "x"\n[fields]\nurl = "u"\nbody = "{output}"\n',
            "[fields] source_id: required by a archive rule",
        ),
        (
            'destination = "searchable"\n[match]\ntool = "x"\n[fields]\n'
            'url = "u"\nbody = "{output}"\nsource_id = "s"\n',
            "[fields] source_id: not used by a searchable rule",
        ),
        (
            'destination = "searchable"\n[match]\ntool = "x"\n[fields]\n'
            'url = "{nope}"\nbody = "{output}"\n',
            "[fields] url: {nope} is not a group",
        ),
        (
            'destination = "searchable"\n[match]\ntool = "x"\n[fields]\n'
            'url = "{input.}"\nbody = "{output}"\n',
            "[fields] url: {input.} names no path",
        ),
        (
            'destination = "searchable"\nkind = "k"\n[match]\ntool = "x"\n[fields]\n'
            'url = "u"\nbody = "{output}"\n',
            "kind: only an archive rule has a kind",
        ),
        (
            'destination = "archive"\nkg_extract = true\n[match]\ntool = "x"\n[fields]\n'
            'url = "u"\nsource_id = "s"\nbody = "{output}"\n',
            "kg_extract: only a searchable rule",
        ),
        (
            'destination = "archive"\nmin_chars = 0\n[match]\ntool = "x"\n',
            "min_chars: expected a positive integer",
        ),
        ('destination = "archive"\nbogus = 1\n', "rule: unknown key 'bogus'"),
        (
            'destination = "archive"\n[match]\ncommand = "(?P<output>x)"\n[fields]\n',
            "[match] command: group name 'output' is reserved",
        ),
    ],
)
def test_a_bad_rule_names_the_file_and_the_field(operator, text, where):
    write(operator, "broken.toml", text)
    with pytest.raises(RuleError, match=rf"broken\.toml: {re.escape(where)}"):
        active_rules()


def test_a_bad_deny_list_is_named_too(operator):
    write(operator, "deny.toml", "words = []\n")
    with pytest.raises(RuleError, match=r"deny\.toml: words: expected a string or a list"):
        active_rules()


def test_the_error_is_a_value_error_so_the_server_can_refuse_to_start():
    assert issubclass(RuleError, ValueError)


def test_check_config_loads_the_rules_and_reads_the_switch(monkeypatch, operator):
    from ingestion.capture import check_config

    check_config()
    write(operator, "broken.toml", 'destination = "nowhere"\n')
    load_rules.cache_clear()
    with pytest.raises(RuleError, match=r"broken\.toml"):
        check_config()
    (operator / "broken.toml").unlink()
    load_rules.cache_clear()
    monkeypatch.setenv("SYNAPSE_WEB_CAPTURE", "maybe")
    with pytest.raises(ValueError, match="SYNAPSE_WEB_CAPTURE"):
        check_config()


def test_the_shipped_directory_is_where_the_package_says(tmp_path):
    assert ruleset_mod.SHIPPED_DIR.name == "rules"
    assert (ruleset_mod.SHIPPED_DIR / "deny.toml").is_file()
