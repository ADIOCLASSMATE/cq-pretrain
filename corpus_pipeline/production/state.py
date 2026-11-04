"""Disk-backed immutable input spool and resumable bounded batch ledger."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from .common import canonical, digest


class State:
    def __init__(self, path, identity):
        self.db = sqlite3.connect(path, timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records(ordinal INTEGER PRIMARY KEY, sample_id TEXT UNIQUE NOT NULL,
                payload TEXT NOT NULL, sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS asset_origins(sha256 TEXT NOT NULL, sample_id TEXT NOT NULL,
                origin TEXT NOT NULL, rejected INTEGER NOT NULL, PRIMARY KEY(sha256,sample_id));
            CREATE INDEX IF NOT EXISTS negative_assets ON asset_origins(sha256,rejected);
            CREATE TABLE IF NOT EXISTS source_protections(sha256 TEXT NOT NULL, sample_id TEXT NOT NULL,
                image_index INTEGER NOT NULL, payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                PRIMARY KEY(sha256,sample_id,image_index));
            CREATE TABLE IF NOT EXISTS image_screening_outcomes(sha256 TEXT PRIMARY KEY,
                decision TEXT NOT NULL, payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS batch_results(batch_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS imports(record_id TEXT NOT NULL, candidate_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
        """)
        existing = self.get("identity")
        if existing is not None and existing != identity:
            raise ValueError("Run configuration/code/input identity changed; choose a new output directory")
        self.set("identity", identity)

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (key, canonical(value)))

    def ingest(self, records):
        if self.get("ingested"):
            return self.db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        # Incomplete ingestion is reconstructed; no processing/publication starts
        # until every selected source association has been indexed.
        with self.db:
            self.db.execute("DELETE FROM records")
            self.db.execute("DELETE FROM asset_origins")
            self.db.execute("DELETE FROM source_protections")
            self.db.execute("DELETE FROM image_screening_outcomes")
        count = 0
        try:
            self.db.execute("BEGIN")
            for count, record in enumerate(records, 1):
                sid = record["original_sample_id"]
                self.db.execute("INSERT INTO records VALUES(?,?,?,?)", (count - 1, sid, canonical(record), digest(record)))
                provenance = record["normalized"]["provenance"]["images"]
                if len(provenance) != len(record["source_image_refs"]):
                    raise ValueError("image provenance count mismatch")
                for image_index, (ref, evidence) in enumerate(zip(record["source_image_refs"], provenance)):
                    origin = evidence["origin"]
                    negative = origin == "ai_generated" or (origin == "ai_edited" and evidence.get("edit_source_origin") not in {"natural", "unknown", "mixed", None})
                    self.db.execute("""INSERT INTO asset_origins VALUES(?,?,?,?)
                        ON CONFLICT(sha256,sample_id) DO UPDATE SET
                        origin=CASE WHEN excluded.rejected>asset_origins.rejected THEN excluded.origin ELSE asset_origins.origin END,
                        rejected=MAX(asset_origins.rejected,excluded.rejected)""", (ref["sha256"], sid, origin, int(negative)))
                    verified_edit = (origin == "ai_edited"
                                     and evidence.get("policy_exception") == "ai_edit_from_natural_source"
                                     and evidence.get("edit_source_origin") == "natural"
                                     and bool(evidence.get("edit_source_evidence")))
                    if origin in {"programmatic", "rendered"} or verified_edit:
                        protection = {"asset_id": ref["sha256"], "record_id": sid, "image_index": image_index,
                                      "source_record_sha256": record.get("source_record_sha256"),
                                      "collection": record.get("collection"), "source": record.get("source"),
                                      "provenance": deepcopy(evidence)}
                        self.db.execute("INSERT INTO source_protections VALUES(?,?,?,?,?)",
                                        (ref["sha256"], sid, image_index, canonical(protection), digest(protection)))
                if count % 1000 == 0:
                    self.db.commit()
                    self.db.execute("BEGIN")
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self.set("ingested", True)
        self.set("input_records", count)
        return count

    def batches(self, batch_size):
        cursor = self.db.execute("SELECT payload,sha256 FROM records ORDER BY ordinal")
        while rows := cursor.fetchmany(batch_size):
            result = []
            for payload, sha in rows:
                record = json.loads(payload)
                if digest(record) != sha:
                    raise ValueError("Input spool checksum mismatch")
                result.append(record)
            yield result

    def rejected_asset(self, sha):
        return self.db.execute("SELECT 1 FROM asset_origins WHERE sha256=? AND rejected=1 LIMIT 1", (sha,)).fetchone() is not None

    def protected_asset(self, sha):
        """Return exact-image source protection from the complete selected input.

        Explicit negative source evidence always takes precedence. The source
        records and normalized provenance remain available without reclassifying
        any unknown/mixed record that refers to the same bytes.
        """
        if self.rejected_asset(sha):
            return None
        sources = []
        for payload, expected in self.db.execute(
                "SELECT payload,payload_sha256 FROM source_protections WHERE sha256=? ORDER BY sample_id,image_index", (sha,)):
            value = json.loads(payload)
            if digest(value) != expected or value["asset_id"] != sha:
                raise ValueError("Source protection checksum mismatch")
            sources.append(value)
        return ({"asset_id": sha, "evidence_scope": "normalized_source_policy_exact_image",
                 "sources": sources} if sources else None)

    def _screening_entry(self, sha):
        row = self.db.execute("SELECT decision,payload,payload_sha256 FROM image_screening_outcomes WHERE sha256=?", (sha,)).fetchone()
        if row is None:
            return None
        decision, payload, expected = row
        entry = json.loads(payload)
        if (digest(entry) != expected or entry["effective"]["asset_id"] != sha
                or entry["effective"]["decision"] != decision):
            raise ValueError("Image screening state checksum or binding mismatch")
        return entry

    def save_screening_outcome(self, sha, outcome):
        """Persist a caller-validated screening decision separately from origin.

        The run identity binds the screening policy, models and thresholds.
        Any differing decisions for the same SHA conflict, including a pending
        versus an admit/reject; conflict remains pending until a separate explicit
        resolution workflow. Repeated identical receipts are idempotent. All
        distinct same-decision evidence is retained in ``evidence_history``.
        This persistence method does not validate model receipts or certify origin.
        """
        if (not isinstance(sha, str) or len(sha) != 64
                or any(char not in "0123456789abcdef" for char in sha)):
            raise ValueError("Invalid screening asset SHA256")
        if not isinstance(outcome, dict) or outcome.get("decision") not in {"admit", "reject", "pending"}:
            raise ValueError("Invalid screening outcome decision")
        if outcome.get("asset_id", sha) != sha:
            raise ValueError("Screening outcome asset binding mismatch")
        if self.db.execute("SELECT 1 FROM asset_origins WHERE sha256=? LIMIT 1", (sha,)).fetchone() is None:
            raise ValueError("Screening outcome asset is outside selected input")
        value = deepcopy(outcome)
        value["asset_id"] = sha
        with self.db:
            previous = self._screening_entry(sha)
            if previous is not None and value == previous["effective"]:
                return deepcopy(previous["effective"])
            observations = {digest(item): item for item in previous["observations"]} if previous else {}
            observations[digest(value)] = value
            history = [observations[key] for key in sorted(observations)]
            decisions = {item["decision"] for item in history}
            if len(decisions) > 1:
                effective = {"asset_id": sha, "decision": "pending",
                             "reason": "image_screening_outcome_conflict",
                             "reason_codes": ["image_screening_outcome_conflict"],
                             "evidence_scope": "screening_not_provenance_certificate",
                             "source_certification": False, "outcomes": history}
            else:
                effective = deepcopy(history[0])
                if len(history) > 1:
                    effective["evidence_history"] = history
            entry = {"effective": effective, "observations": history}
            self.db.execute("INSERT OR REPLACE INTO image_screening_outcomes VALUES(?,?,?,?)",
                            (sha, effective["decision"], canonical(entry), digest(entry)))
        return deepcopy(effective)

    def screening_outcome(self, sha):
        """Return the complete bound screening outcome, or None when unprocessed."""
        entry = self._screening_entry(sha)
        return deepcopy(entry["effective"]) if entry else None

    def screening_rejections(self):
        """Return sorted rejected SHA256s for asset revocation, validating receipts."""
        result = []
        for (sha,) in self.db.execute("SELECT sha256 FROM image_screening_outcomes WHERE decision='reject' ORDER BY sha256"):
            if self.screening_outcome(sha)["decision"] == "reject":
                result.append(sha)
        return result

    def import_candidates(self, rows):
        from .staged_publish import _validate_candidate
        with self.db:
            for row in rows:
                _validate_candidate(row)
                sid, cid = row["record_id"], row["candidate_id"]
                record_row = self.db.execute("SELECT payload FROM records WHERE sample_id=?", (sid,)).fetchone()
                if record_row is None:
                    # A shared catalog may cover more than this selected input.
                    continue
                record = json.loads(record_row[0])
                if (row["source_record_sha256"] != record["source_record_sha256"]
                        or row["asset_ids"] != [ref["sha256"] for ref in record["source_image_refs"]]
                        or row["native_task"] != record["normalized"]["category"]
                        or row["collection"] != record.get("collection", "dcvlm")):
                    raise ValueError("Imported candidate does not bind the original selected record")
                existing = self.db.execute("SELECT payload FROM imports WHERE candidate_id=?", (cid,)).fetchone()
                if existing and existing[0] != canonical(row):
                    raise ValueError("Candidate identity collision")
                self.db.execute("INSERT OR IGNORE INTO imports VALUES(?,?,?)", (sid, cid, canonical(row)))

    def candidates(self, sid):
        return [json.loads(r[0]) for r in self.db.execute("SELECT payload FROM imports WHERE record_id=? ORDER BY candidate_id", (sid,))]

    def save_batch(self, batch_id, result):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO batch_results VALUES(?,?)", (batch_id, canonical(result)))

    def close(self):
        self.db.close()
