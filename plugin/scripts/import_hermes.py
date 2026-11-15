#!/usr/bin/env python3
# mypy: ignore-errors
"""synapse import hermes: import past Hermes sessions into Synapse.

    synapse import hermes                     # interactive: optional date range, summary, y/N
    synapse import hermes --yes               # scripted: no prompts, import everything
    synapse import hermes --hermes-home PATH  # another home (default $HERMES_HOME, ~/.hermes)
    synapse import hermes --dry-run           # the summary and one line per session, nothing sent
    synapse import hermes --no-hermes-scrub   # send the stored text even when Hermes cannot scrub it

Reads ``<hermes_home>/state.db`` read-only and sends the finished turns of the sessions
the live memory provider would have uploaded, to the same ``/ingest`` route in the same
``messages`` format (synapse_client/harnesses/hermes/history.py has the rules). It reads
through the provider's own reader and records what it sends in the provider's ledger
(``<hermes_home>/synapse/shipped``): a row the provider or an earlier import shipped is
not sent again, and the provider does not send again what the import sent. A turn sent
twice anyway maps to the same span id, so the server keeps one copy: interrupting and
re-running is safe.

Hermes's store holds the rows unscrubbed; the live provider gets them from Hermes with
secrets masked. So the import has Hermes scrub what it read, with Hermes's own code run
by Hermes's interpreter (harnesses/hermes/scrub.py), before anything is sent. When
Hermes's scrub cannot be run, nothing is sent: ``--no-hermes-scrub`` sends the stored
text as clients up to 0.24.22 did, which leaves secrets to the server's own redaction
at ingest. Private sessions, and sessions holding turns of a private one, are skipped:
by the marker in the private folder and by the server's row.

Each imported turn runs KG extraction on the server's LLM, so the interactive run asks
for an optional date range (by each session's last activity) and for a confirmation
before anything is sent. The settings (URL, token, private folder) are the provider's:
~/.synapse/config.json and the environment.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synapse_client import __version__, private
from synapse_client.api import Client
from synapse_client.harnesses.hermes import history, scrub
from synapse_client.harnesses.hermes.install import default_home
from synapse_client.settings import Settings
from synapse_client.shipped import Ledger

#: Seconds per ingest request, and per private-mode lookup.
TIMEOUT = 60.0
LOOKUP_TIMEOUT = 15.0
PROGRAM = "synapse import hermes"


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _ask_date(label: str) -> float | None:
    for _ in range(3):
        try:
            raw = input(f"  {label} (YYYY-MM-DD, Enter = no limit): ").strip()
        except EOFError:
            return None
        if not raw:
            return None
        try:
            return datetime.strptime(raw, "%Y-%m-%d").timestamp()
        except ValueError:
            print("  Not a date. Use YYYY-MM-DD, e.g. 2026-05-01.")
    print("  Three invalid entries. Continuing without this limit.")
    return None


def in_range(sessions, since: float | None, until: float | None):
    """The sessions whose last activity falls in [since, until), either bound optional."""
    return [
        s
        for s in sessions
        if (since is None or s.last_at >= since) and (until is None or s.last_at < until)
    ]


def prompt_date_range(sessions):
    """Ask for an optional date range (by each session's last activity) and filter.
    Enter at both prompts, or no terminal, keeps every session."""
    if not sessions:
        return sessions
    first = min(s.last_at for s in sessions)
    last = max(s.last_at for s in sessions)
    print(f"Sessions span {_day(first)} to {_day(last)} (by last activity).")
    print("Limit the import by date? Press Enter at both prompts to import everything.")
    since = _ask_date("from")
    until = _ask_date("up to and including")
    if until is not None:
        until += 86400.0  # the end date is inclusive
    if since is None and until is None:
        return sessions
    kept = in_range(sessions, since, until)
    print(f"  {len(kept)} of {len(sessions)} session(s) in range.")
    print()
    return kept


def private_lookup(settings: Settings, client: Client):
    """``is_private(session_id)``: True, False, or None when it cannot be told. The
    marker in the private folder first, then the server's row, once per id."""
    known: dict[str, bool | None] = {}

    def is_private(session_id: str) -> bool | None:
        if session_id in known:
            return known[session_id]
        state = private.state(settings, session_id)
        if state == private.PRIVATE:
            answer: bool | None = True
        elif state == private.UNKNOWN:
            answer = None
        else:
            try:
                answer = bool(client.private_get(session_id, timeout=LOOKUP_TIMEOUT).get("private"))
            except Exception:
                answer = None
        known[session_id] = answer
        return answer

    return is_private


def _short(session_id: str) -> str:
    return session_id if len(session_id) <= 24 else session_id[:22] + ".."


def ledger_of(settings: Settings) -> Ledger:
    """The ledger the live provider keeps (under the Synapse home)."""
    return Ledger(os.path.join(str(settings.home), "shipped", "hermes"))


def scrubbed(chosen, home: str, python: str | None):
    """``{session id: [(payload, new turns), ...]}`` for ``chosen``, every payload's
    records scrubbed by Hermes (scrub.rows), or None when Hermes could not scrub them."""
    plans = {s.id: history.batches(s) for s in chosen}
    keyed = {
        f"{sid}\x00{i}": payload["records"]
        for sid, plan in plans.items()
        for i, (payload, _) in enumerate(plan)
    }
    done = scrub.rows(keyed, python, home)
    if done is None:
        return None
    for sid, plan in plans.items():
        for i, (payload, _) in enumerate(plan):
            payload["records"] = done[f"{sid}\x00{i}"]
    return plans


def send(client: Client, session, ledger: Ledger | None = None, plan=None) -> int:
    """POST one session's turns (``plan``: its scrubbed bodies, ``scrubbed``; else as
    read); returns how many the server took as new. Each body the server took is
    recorded in ``ledger``. RAISES private.PrivateSession, before any request of that
    batch, for a session gone private since, and what a failed request raises."""
    ingested = 0
    for payload, new_turns in plan if plan is not None else history.batches(session):
        reply = client.ingest(payload, timeout=TIMEOUT)
        if ledger is not None:
            try:
                history.record(ledger, session.id, new_turns)
            except OSError as err:
                print(
                    f"  could not record what was sent ({type(err).__name__}); the provider "
                    "may send it again, which the server drops",
                    file=sys.stderr,
                )
        try:
            ingested += int(reply.get("ingested", 0))
        except (AttributeError, TypeError, ValueError):
            pass
    return ingested


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Import past Hermes sessions from its state.db into Synapse "
        "(the live provider's rules; span_id dedup makes re-runs safe).",
    )
    ap.add_argument(
        "--hermes-home",
        metavar="PATH",
        help="Hermes home (default $HERMES_HOME, else ~/.hermes)",
    )
    ap.add_argument("--yes", "-y", action="store_true", help="skip the prompts")
    ap.add_argument(
        "--dry-run", action="store_true", help="show what would be imported; send nothing"
    )
    ap.add_argument(
        "--no-hermes-scrub",
        action="store_true",
        help="send the stored text when Hermes's secret scrub cannot be run "
        "(the server's own redaction still applies)",
    )
    args = ap.parse_args(argv)

    home = os.path.expanduser(args.hermes_home) if args.hermes_home else str(default_home())
    try:
        ledger = ledger_of(Settings.load())
        sessions = history.read(home, ledger=ledger)
    except FileNotFoundError:
        print(f"{PROGRAM}: no session store at {os.path.join(home, 'state.db')}", file=sys.stderr)
        return 1
    except history.SchemaError as err:
        print(f"{PROGRAM}: {err} Nothing was sent.", file=sys.stderr)
        return 1
    except sqlite3.Error as err:
        print(f"{PROGRAM}: could not read state.db ({err}). Nothing was sent.", file=sys.stderr)
        return 1
    except OSError as err:
        print(
            f"{PROGRAM}: could not read the marks under {home}/synapse/inert "
            f"({type(err).__name__}). Nothing was sent.",
            file=sys.stderr,
        )
        return 1
    if not sessions:
        print(f"No sessions in {home}/state.db. Nothing to import.")
        return 0

    if not args.yes and not args.dry_run:
        sessions = prompt_date_range(sessions)
        if not sessions:
            print("No sessions in that date range. Nothing to import.")
            return 0

    settings = Settings.load()
    client = Client(settings, f"synapse-client/{__version__} hermes-import")
    is_private = private_lookup(settings, client)
    chosen, skipped = [], Counter()
    verdicts = []
    for s in sessions:
        reason = s.reason or history.private_reason(s, is_private)
        verdicts.append((s, reason))
        if reason:
            skipped[reason] += 1
        else:
            chosen.append(s)

    turns = sum(len(s.pairs) for s in chosen)
    copies = sum(s.copies for s in chosen)
    token = "configured" if settings.token else "none"
    print(f"Hermes home {home}: {len(sessions)} session(s)")
    print(f"  to import        {len(chosen)} session(s), {turns} finished turn(s)")
    if copies:
        print(f"  left out         {copies} turn(s) copied from an earlier session")
    if skipped:
        print(f"  skipped          {sum(skipped.values())} session(s):")
        for reason, n in skipped.most_common():
            print(f"    {n:>4}  {reason}")
    print(f"  target           {settings.ingest_url} (auth token: {token})")
    print()
    if args.dry_run:
        for s, reason in verdicts:
            what = reason or f"{len(s.pairs)} turn(s)"
            print(f"  {_day(s.last_at)}  {_short(s.id):<24}  {what}")
        return 0
    if not chosen:
        print("Nothing to import.")
        return 0
    print("Importing runs KG extraction on the server's configured LLM for every NEW")
    print("turn. That consumes subscription usage or API credits, roughly in proportion")
    print("to the turn count above. Turns already in Synapse (uploaded by the provider or")
    print("an earlier import) are skipped by the server, so interrupting with Ctrl-C and")
    print("re-running is safe: the import resumes where it left off.")
    print()
    if not args.yes:
        try:
            answer = input("Proceed? [y/N] ")
        except EOFError:
            print("No interactive terminal for the confirmation. Re-run with --yes.")
            return 1
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted. Nothing was sent.")
            return 1

    try:
        plans = scrubbed(chosen, home, scrub.interpreter())
        if plans is None and not args.no_hermes_scrub:
            print(
                f"{PROGRAM}: Hermes's secret scrub could not be run (the hermes program, or "
                "its agent package, was not found). Nothing was sent. Re-run with "
                "--no-hermes-scrub to send the stored text; the server still redacts "
                "credentials it recognises.",
                file=sys.stderr,
            )
            return 1
        return _ship(client, chosen, ledger, plans)
    except KeyboardInterrupt:
        print("\nInterrupted. Turns already sent are kept (dedup by span_id): re-run to resume.")
        return 130


def _ship(client: Client, chosen, ledger: Ledger | None = None, plans=None) -> int:
    ok = failed = refused = ingested_total = 0
    for i, s in enumerate(chosen, 1):
        label = f"[{i}/{len(chosen)}] {_short(s.id)}"
        try:
            n = send(client, s, ledger, plans.get(s.id) if plans else None)
        except private.PrivateSession:
            refused += 1
            print(f"{label} -> skipped (private mode)", flush=True)
            continue
        except Exception as err:
            failed += 1
            print(f"{label} -> FAILED ({type(err).__name__}: {str(err)[:120]})", flush=True)
            continue
        ok += 1
        ingested_total += n
        print(f"{label} -> ingested {n}", flush=True)
    print()
    print(f"Done: {ingested_total} new turn(s) ingested from {ok}/{len(chosen)} session(s).")
    if refused:
        print(f"{refused} session(s) went private during the import and were not sent.")
    if failed:
        print(f"{failed} session(s) failed. Re-run to retry; turns already in are skipped.")
    return 1 if failed and ok == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
