""" Shared helpers for writing JSONL metadata files """
from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write a list of dict rows to a JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {len(rows):>7} rows -> {path}")
