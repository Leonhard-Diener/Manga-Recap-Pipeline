"""Transactional, manga-scoped history. Current reviews are replayed on every run."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from .matching import stable_id


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def save_revision(database: Path, result: dict, reviews: dict, *, run_id: str, manifests=()) -> None:
    """Commit evidence, decisions and membership together; retain revoked history."""
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as db:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            raise ValueError(f"Unsupported identity database version {version}.")
        db.execute("PRAGMA foreign_keys = ON")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS revisions (
                id TEXT PRIMARY KEY, result_json TEXT NOT NULL, reviews_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, evidence_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS revision_decisions (
                revision TEXT NOT NULL REFERENCES revisions(id),
                decision TEXT NOT NULL REFERENCES decisions(id), PRIMARY KEY (revision, decision));
            CREATE TABLE IF NOT EXISTS records (
                revision TEXT NOT NULL REFERENCES revisions(id), kind TEXT NOT NULL,
                id TEXT NOT NULL, data_json TEXT NOT NULL, PRIMARY KEY (revision, kind, id));
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, revision TEXT NOT NULL REFERENCES revisions(id),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            PRAGMA user_version = 1;
        """)
        with db:
            owner = db.execute("SELECT value FROM metadata WHERE key='manga'").fetchone()
            if owner and owner[0] != result["manga"]:
                raise ValueError("Identity database belongs to a different manga.")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('manga', ?)", (result["manga"],))
            revision = result["revision"]
            db.execute("INSERT OR IGNORE INTO revisions(id,result_json,reviews_json) VALUES (?,?,?)",
                       (revision, _json(result), _json(reviews)))
            for kind in ("assignments", "different", "names"):
                for decision in reviews.get(kind, []):
                    identity = stable_id("decision_", kind, decision)
                    db.execute("INSERT OR IGNORE INTO decisions VALUES (?,?,?)", (identity, kind, _json(decision)))
                    db.execute("INSERT OR IGNORE INTO revision_decisions VALUES (?,?)", (revision, identity))
            def record(kind, identity, data):
                db.execute("INSERT OR IGNORE INTO records VALUES (?,?,?,?)",
                           (revision, kind, identity, _json(data)))
            for row in result["characters"]:
                record("instance", row["instance_id"], row)
            for row in result["identities"]:
                record("entity", row["entity_id"], row)
            for row in result["visual_groups"]:
                record("visual_group", row["group_id"], row)
            for row in result["transformation_proposals"]:
                record("claim", row["event_id"], row)
            for manifest in manifests:
                for page in manifest.get("pages", []):
                    identity = page.get("page_id") or stable_id("page_", page["page"], page.get("source_hash"))
                    record("page", identity, page)
                    for panel in page.get("panels", []):
                        record("panel", panel.get("panel_id") or stable_id("panel_", identity, panel["box"]), panel)
            db.execute("INSERT INTO runs(id,revision) VALUES (?,?)", (run_id, revision))
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('current_revision', ?)", (revision,))


def load_revision(database: Path, revision: str | None = None) -> dict:
    """Read a frozen resolution and its review input without changing current state."""
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        if revision is None:
            current = db.execute("SELECT value FROM metadata WHERE key='current_revision'").fetchone()
            revision = current[0] if current else None
        row = db.execute("SELECT result_json,reviews_json FROM revisions WHERE id=?", (revision,)).fetchone()
        if row is None:
            raise KeyError(revision)
        return {"result": json.loads(row[0]), "reviews": json.loads(row[1])}
