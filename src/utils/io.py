"""Small, dependency-free artifact and checkpoint helpers."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_key(row):
    return str(row["query_id"]), str(row["doc_id"])


def read_manifest(path):
    rows, seen, datasets, thresholds = [], set(), set(), set()
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("dataset", "query_id", "doc_id", "query_text", "doc_text"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Manifest line {line_number}: missing/non-string {field}")
            grade, threshold = row.get("true_relevance"), row.get("relevance_threshold")
            if type(grade) is not int or grade < 0 or type(threshold) is not int or threshold < 1:
                raise ValueError(f"Manifest line {line_number}: invalid grade/threshold")
            if row.get("true_binary") != int(grade >= threshold):
                raise ValueError(f"Manifest line {line_number}: inconsistent binary gold label")
            if pair_key(row) in seen:
                raise ValueError(f"Duplicate manifest pair {pair_key(row)}")
            seen.add(pair_key(row))
            datasets.add(row["dataset"])
            thresholds.add(threshold)
            rows.append(row)
    if not rows or len(datasets) != 1 or len(thresholds) != 1:
        raise ValueError("Manifest must contain one nonempty collection and relevance definition")
    return rows


def read_predictions(path, relevance_threshold=2, deduplicate_last=False):
    rows = {}
    duplicates = 0
    with open(path, encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            key = pair_key(raw)
            if not all(key):
                raise ValueError("Prediction IDs must be nonempty")
            grade = int(raw["true_relevance"])
            probability = float(raw["prob_relevant"])
            label = int(raw["predicted_label"])
            if grade < 0 or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid grade/probability for {key}")
            if label not in (0, 1) or (probability != 0.5 and label != int(probability > 0.5)):
                raise ValueError(f"Inconsistent prediction for {key}")
            gold = int(grade >= relevance_threshold)
            if raw.get("true_binary") not in (None, "", str(gold)):
                raise ValueError(f"Inconsistent gold label for {key}")
            if key in rows:
                duplicates += 1
                if not deduplicate_last:
                    raise ValueError(f"Duplicate prediction {key}; legacy diagnostics require --deduplicate-last")
            rows[key] = {**raw, "true_relevance": grade, "true_binary": gold,
                         "prob_relevant": probability, "predicted_label": label,
                         "confidence": max(probability, 1 - probability)}
    if not rows:
        raise ValueError("No predictions found")
    return list(rows.values()), duplicates


def write_csv(path, rows, fields=None):
    import io
    rows = list(rows)
    if fields is None:
        if not rows:
            raise ValueError("Explicit CSV fields required for an empty table")
        fields = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


class PredictionStore:
    """One transaction per batch; a killed job cannot leave half a CSV row."""

    def __init__(self, path, experiment):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, value TEXT)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS predictions "
                                "(query_id TEXT, doc_id TEXT, value TEXT, PRIMARY KEY(query_id, doc_id))")
        encoded = json.dumps(experiment, sort_keys=True, allow_nan=False)
        existing = self.connection.execute("SELECT value FROM metadata WHERE id=1").fetchone()
        if existing and existing[0] != encoded:
            self.connection.close()
            raise ValueError("Experiment changed. Use a new output directory; checkpoint cannot be reused.")
        if not existing:
            with self.connection:
                self.connection.execute("INSERT INTO metadata VALUES (1, ?)", (encoded,))

    def rows(self):
        return [json.loads(row[0]) for row in self.connection.execute(
            "SELECT value FROM predictions ORDER BY query_id, doc_id")]

    def add_batch(self, rows):
        with self.connection:
            self.connection.executemany("INSERT INTO predictions VALUES (?, ?, ?)", [
                (*pair_key(row), json.dumps(row, allow_nan=False)) for row in rows])

    def backup(self, path):
        """Publish a consistent SQLite snapshot without copying a live database file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        os.close(fd)
        try:
            destination = sqlite3.connect(temporary)
            try:
                self.connection.backup(destination)
            finally:
                destination.close()
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def close(self):
        self.connection.close()
