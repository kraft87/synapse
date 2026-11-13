"""The Hermes names the client copies, read from a real Hermes checkout.

``harnesses/hermes/messages.py`` copies Hermes's failed-turn notices and their
``display_kind``; ``harnesses/hermes/sessions.py`` copies the shape of a gateway local
session (its chat id, its route, the state keys of its creation receipt), the
surfaces Hermes stores as an owner's own chat, and the session store's columns it
reads; ``harnesses/hermes/lineage.py`` copies the ``model_config`` keys under which a
branch, a reset and a compressed continuation name the session they came from;
``harnesses/hermes/provider.py`` reads how a session switch is announced (``reset``, and
``reason="resume"``) to tell a conversation that goes on under a new id from another
conversation, and, for a session a new agent is built for, takes the same from its row
(``lineage.continues``). A Hermes that renames any of them would turn a failed turn back into an
uploaded reply, the owner's ``hermes chat`` back into an inert session, a copy of a Bot
Chat back into the owner's chat once its parent is deleted, or a compressed private
conversation back into an uploaded one, without a unit test noticing: these checks ask
the checkout itself.

Skipped unless ``SYNAPSE_HERMES_CHECKOUT`` names a Hermes checkout with its virtualenv
at ``.venv`` (as tests/test_hermes_integration.py). A driver runs under the checkout's
interpreter with a temp HERMES_HOME and HOME, imports the modules and prints the names;
the checkout is only read and the ``hermes`` program is never started.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from typing import Any

import pytest

from tests import test_hermes_integration as base
from tests.helpers.client_core import use_core

use_core()
from synapse_client.harnesses.hermes import lineage as hl  # noqa: E402
from synapse_client.harnesses.hermes import messages as hm  # noqa: E402
from synapse_client.harnesses.hermes import sessions as hs  # noqa: E402

pytestmark = pytest.mark.skipif(
    not base._CHECKOUT, reason="set SYNAPSE_HERMES_CHECKOUT to a Hermes checkout to run"
)

_DRIVER = textwrap.dedent(
    """
    import inspect, json, os, re, sqlite3
    from pathlib import Path

    import hermes_state_mutation_branch as branch
    import hermes_state_mutation_compress as compress
    import hermes_state_mutation_reset as reset
    from agent import turn_failure_copy as failures
    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key
    from gateway.session_policy import STORED_AS_LOCAL, SURFACES
    from hermes_state_keys import local_identity
    from hermes_state_common import SCHEMA_SQL, _non_continuation_child_sql
    from hermes_state_local import API_BINDING_PREFIX, POLICY_PREFIX

    store = sqlite3.connect(":memory:")
    store.executescript(SCHEMA_SQL)
    columns = {table: [row[1] for row in store.execute("PRAGMA table_info(%s)" % table)]
               for table in ("sessions", "state_meta")}
    notices = [failures.FAILED_TURN_NOTICE, failures.PARTIAL_FAILED_TURN_NOTICE,
               failures.AUTOMATION_INTERRUPTED_NOTICE]
    owner = "uid:%d" % os.getuid()
    chat = local_identity(os.environ["HERMES_HOME"], owner, "req-1")
    local = SessionSource(platform=Platform.LOCAL, chat_id=chat, chat_type="dm", user_id=owner)
    origins = [key for key, module in (("_branched_from", branch), ("_reset_from", reset),
                                       ("_compressed_from", compress))
               if "model_config={'%s': target}" % key in inspect.getsource(module)]
    def says(path, text):
        return " ".join(Path(path).read_text(encoding="utf-8").split()).count(text)

    switches = {
        "compression": says("agent/conversation_compression.py",
                            'reset=False, reason="compression" )'),
        "branch": says("hermes_cli/cli_commands_mixin.py",
                       'parent_session_id=parent_session_id, reason="branch")'),
        "resume": says("hermes_cli/cli_commands_mixin.py",
                       'parent_session_id=old_session_id, reason="resume")'),
        "resume_and_branch_keep": says("hermes_cli/cli_session_switch.py",
                                       'reset=False, reason=reason)'),
        "new_session": says("hermes_cli/cli_session_mixin.py",
                            'reset=True, reason="new_session")'),
    }
    from agent.memory_provider import MemoryProvider
    def takes(name):
        return list(inspect.signature(getattr(MemoryProvider, name)).parameters)

    named = {
        "turn_start": says("agent/turn_context.py",
                           "prefetch_all(_query, session_id=agent.session_id)"),
        "prefetch": takes("prefetch"),
        "sync_turn_names_one": "session_id" in takes("sync_turn"),
        "on_session_end": takes("on_session_end"),
        "on_pre_compress": takes("on_pre_compress"),
    }
    def in_order(path, *texts):
        text = " ".join(Path(path).read_text(encoding="utf-8").split())
        at = [text.rfind(one) for one in texts]
        return -1 not in at and at == sorted(at)

    turn_starts = {
        "hook": takes("on_turn_start"),
        "says_what_came_before": says("agent/turn_context.py",
                                      "previous_message=previous_message, )"),
        "before_the_prefetch": in_order("agent/turn_context.py",
                                        "agent._memory_manager.on_turn_start(",
                                        "agent._memory_manager.prefetch_all("),
        "after_the_compaction": in_order("agent/turn_context.py",
                                         "agent._user_turn_count += 1",
                                         "run_turn_start_compaction(",
                                         "else _memory_turn_start_and_prefetch("),
    }
    forks = _non_continuation_child_sql("c.", "?")
    print("RESULT " + json.dumps({
        "fork_keys": sorted(set(re.findall(r"[$][.](_[a-z_]+)", forks))),
        "fork_keys_name_the_parent": forks.count("!= ?"),
        "tui_branch_row": says(
            "tui_gateway/methods_session.py",
            'db.create_session(new_key, source=source, model=model, '
            'model_config={"_branched_from": parent_key}, parent_session_id=parent_key,'),
        "failed_kind": failures.FAILED_TURN_DISPLAY_KIND,
        "failed_notices": notices,
        "untyped": [failures.untyped_failed_turn_display_kind("assistant", n) for n in notices],
        "chat": chat,
        "route": build_session_key(local),
        "policy_prefix": POLICY_PREFIX,
        "api_prefix": API_BINDING_PREFIX,
        "surfaces": sorted(SURFACES),
        "stored_as_local": sorted(STORED_AS_LOCAL),
        "columns": columns,
        "origins": origins,
        "switches": switches,
        "named": named,
        "turn_starts": turn_starts,
    }))
    """
)


@pytest.fixture(scope="module")
def names(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    checkout, python = base._checkout()
    root = tmp_path_factory.mktemp("checkout-names")
    (root / "hermes").mkdir()
    (root / "home").mkdir()
    guard = root / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(base._GUARD, encoding="utf-8")
    driver = root / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    done = subprocess.run(
        [str(python), str(driver)],
        cwd=str(checkout),
        env={
            "HOME": str(root / "home"),
            "HERMES_HOME": str(root / "hermes"),
            "PATH": os.defpath,
            "PYTHONPATH": os.pathsep.join((str(guard), str(checkout))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(root),
            "LANG": "C.UTF-8",
            "SYNAPSE_TEST_NETWORK_LOG": str(root / "network.log"),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")]
    assert done.returncode == 0 and lines, f"driver failed:\n{done.stdout}\n{done.stderr}"
    result: dict[str, Any] = json.loads(lines[-1][len("RESULT ") :])
    return result


def test_this_hermes_failed_turn_notice_is_never_a_reply(names: dict[str, Any]) -> None:
    assert names["failed_kind"] == hm._FAILED_TURN_KIND
    assert sorted(names["failed_notices"]) == sorted(hm._FAILED_TURN_NOTICES)
    assert names["untyped"] == [hm._FAILED_TURN_KIND] * len(names["failed_notices"])
    for notice in names["failed_notices"]:
        assert hm.to_records([{"role": "assistant", "content": notice}]) == []
        typed = {"role": "assistant", "content": notice, "display_kind": names["failed_kind"]}
        assert hm.to_records([typed]) == []


def test_this_hermes_local_session_has_the_shape_the_rule_knows(names: dict[str, Any]) -> None:
    assert hs._LOCAL_CHAT.fullmatch(names["chat"])
    route = hs._LOCAL_ROUTE.fullmatch(names["route"])
    assert route is not None and route.group(1) == names["chat"], names["route"]
    assert (names["policy_prefix"], names["api_prefix"]) == (hs._POLICY_KEY, hs._API_KEY)
    assert hs.OWNER_SOURCES <= set(names["surfaces"])
    assert set(names["stored_as_local"]) - {"tool"} <= hs.OWNER_SOURCES


def test_this_hermes_store_has_the_columns_the_rule_reads(names: dict[str, Any]) -> None:
    """A column the rule selects and the store lacks reads as "the session store could
    not be read": every session INERT, the owner's own included."""
    sessions = {"id", "title", "source", "chat_id", "model_config", "parent_session_id"}
    assert sessions <= set(names["columns"]["sessions"])
    assert {"key", "value"} <= set(names["columns"]["state_meta"])


def test_this_hermes_names_a_copys_origin_the_way_the_rule_reads_it(names: dict[str, Any]) -> None:
    """Hermes clears ``parent_session_id`` when a parent is deleted; these keys are what
    still says a session is a copy or continuation of it. A tripwire for a rename in the
    three gateway writers, by their source text: it does not run a delete, and it does
    not cover the in-process writers."""
    assert names["origins"] == list(hl._ORIGINS)


def test_this_hermes_knows_no_kind_of_child_the_rule_has_not_decided_on(
    names: dict[str, Any],
) -> None:
    """``lineage.continues`` decides from a row whether the session holds its parent's
    turns, and so whether it is private with it: a branch and a compressed continuation
    do, a reset and a subagent do not. That is not Hermes's rule, which asks whether a
    child is the same session going on and turns a branch away. What is read here is
    Hermes's list of the keys that mark a child as something else, each counted only
    when it names the parent: every key on it is one ``continues`` has decided on, and a
    new key there is a new kind of child someone has to decide on. Hermes's list also
    turns away a child whose ``source`` is ``tool``, which ``continues`` does not read."""
    assert names["fork_keys"] == sorted({"_branched_from", *hl._OWN_CONVERSATION})
    assert names["fork_keys_name_the_parent"] >= 3


def test_this_hermes_writes_a_tui_branchs_row_with_the_link_the_rule_reads(
    names: dict[str, Any],
) -> None:
    """The TUI and Desktop build a new agent for a branch and announce no switch: the
    row, with the parent link and ``_branched_from``, is all the provider has to know
    the session holds a private parent's turns. When the row is written is not read:
    the provider looks at it again at each upload until it names a parent. By the
    source text of the one call that writes it."""
    assert names["tui_branch_row"] == 1


def test_this_hermes_announces_a_session_switch_the_way_the_provider_reads_it(
    names: dict[str, Any],
) -> None:
    """The provider keeps a conversation private across a new session id unless the
    switch is a reset (``/new``) or says ``reason="resume"`` (another conversation). A
    compression that came to pass ``reset=True`` would upload a private conversation
    under its new id; a resume under another reason would make the conversation resumed
    private for no reason of its own. By the source text of the call sites."""
    assert names["switches"] == {
        "compression": 2,  # the child session, and the notice after either mode
        "branch": 1,
        "resume": 1,
        "resume_and_branch_keep": 1,
        "new_session": 1,
    }


def test_this_hermes_names_the_session_where_the_provider_reads_it(names: dict[str, Any]) -> None:
    """The flush hooks carry no session id, so a flush goes under the id Hermes named
    last (harnesses/hermes/live.py): at the start of each turn (``prefetch``, on a
    thread Hermes waits for, with the agent's session) and at ``sync_turn``. A Hermes
    that stopped naming it as the turn starts would leave a flush, after a move it did
    not announce, under the session left; one whose flush hooks came to carry an id
    should be read there instead. By the call site's source text and the base class's
    signatures."""
    assert names["named"] == {
        "turn_start": 1,
        "prefetch": ["self", "query", "session_id"],
        "sync_turn_names_one": True,
        "on_session_end": ["self", "messages"],
        "on_pre_compress": ["self", "messages"],
    }


def test_this_hermes_says_when_a_turn_starts_the_way_the_provider_reads_it(
    names: dict[str, Any],
) -> None:
    """A session that holds turns from before uploads a turn only when the provider saw
    it begin on the record (``on_turn_start``, turn_upload.TurnUploader.begin). That
    reads three things off Hermes. The hook comes after the turn-start compaction: a
    switch the compaction makes has moved the provider by then, so the turn is counted
    under the id it runs in. It comes before the turn's ``prefetch``. And it carries the
    user message before this one (``previous_message``), which tells a session Hermes
    said was new that it holds copies; without it an ACP fork is left to the count of
    its turns alone. By the source text of the call sites and the base class's
    signature."""
    assert names["turn_starts"] == {
        "hook": ["self", "turn_number", "message", "kwargs"],
        "says_what_came_before": 1,
        "before_the_prefetch": True,
        "after_the_compaction": True,
    }
