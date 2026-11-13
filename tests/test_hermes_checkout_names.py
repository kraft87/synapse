"""The Hermes names the client copies, read from a real Hermes checkout.

``harnesses/hermes/messages.py`` copies Hermes's failed-turn notices and their
``display_kind``; ``harnesses/hermes/sessions.py`` copies the shape of a gateway local
session (its chat id, its route, the state keys of its creation receipt), the
surfaces Hermes stores as an owner's own chat, and the session store's columns it
reads. A Hermes that renames any of them would
turn a failed turn back into an uploaded reply, or the owner's ``hermes chat`` back into
an inert session, without a unit test noticing: these checks ask the checkout itself.

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
from synapse_client.harnesses.hermes import messages as hm  # noqa: E402
from synapse_client.harnesses.hermes import sessions as hs  # noqa: E402

pytestmark = pytest.mark.skipif(
    not base._CHECKOUT, reason="set SYNAPSE_HERMES_CHECKOUT to a Hermes checkout to run"
)

_DRIVER = textwrap.dedent(
    """
    import json, os, sqlite3

    from agent import turn_failure_copy as failures
    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key
    from gateway.session_policy import STORED_AS_LOCAL, SURFACES
    from hermes_state_keys import local_identity
    from hermes_state_common import SCHEMA_SQL
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
    print("RESULT " + json.dumps({
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
