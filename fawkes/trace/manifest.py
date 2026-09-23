"""SHA-256 manifest tooling (the rlearn rule: hash everything, copy nothing)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest16(value: str) -> str:
    """A short digest for embedding in artifacts (Phoenix bundle style)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def catalog_entry(path: str | Path) -> dict:
    path = Path(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_manifest(paths, out_path: str | Path, extra: dict | None = None) -> dict:
    """Write a JSON manifest of the given files plus any extra metadata."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "files": [catalog_entry(p) for p in paths]}
    if extra:
        manifest.update(extra)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
