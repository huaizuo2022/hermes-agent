"""Runtime sqlite3 upgrade: alias stdlib sqlite3 to pysqlite3 when newer.

The production host's Python 3.12 links the OS SQLite 3.26.0, which lacks
the FTS5 trigram tokenizer (needs >= 3.34); hermes_state then disables
full-text session search with a "SQLite FTS5 unavailable" warning.
pysqlite3-binary (pinned in the ``web`` extra, linux only) ships a
self-contained modern SQLite.  Aliasing the stdlib module upgrades every
process that imports hermes_cli — web server, cron tick, scripts — without
touching any ``import sqlite3`` site.

The shim stays inert when pysqlite3 is missing or not newer than the
stdlib, so the same code runs fine on macOS/Windows or after the OS Python
is upgraded.
"""

from __future__ import annotations

import sys


def install() -> None:
    try:
        import pysqlite3
    except ImportError:
        return

    import sqlite3 as stdlib_sqlite3

    def _version_tuple(value: str) -> tuple[int, ...]:
        return tuple(int(p) for p in value.split(".") if p.isdigit())

    if _version_tuple(pysqlite3.sqlite_version) <= _version_tuple(
        stdlib_sqlite3.sqlite_version
    ):
        return

    # pysqlite3's package layout mirrors sqlite3's (dbapi2, dump, exceptions).
    sys.modules["sqlite3"] = pysqlite3
    for sub in ("dbapi2", "dump", "exceptions"):
        candidate = getattr(pysqlite3, sub, None)
        if candidate is not None:
            sys.modules[f"sqlite3.{sub}"] = candidate


# Guard: run at most once per process.  If sqlite3 is already aliased to
# pysqlite3 (site-packages zz_pysqlite3_shim.py, or a re-import), skip.
_current = sys.modules.get("sqlite3")
installed = _current is not None and getattr(_current, "__name__", "") == "pysqlite3"
if not installed:
    install()
