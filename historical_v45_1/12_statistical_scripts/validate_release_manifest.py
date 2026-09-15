#!/usr/bin/env python3
"""Validate the submission reproducibility package against its manifest.

The manifest covers every file in the package.  Its own entry is marked as
self-referential and therefore intentionally has no byte count or digest.
Every other file must match both fields exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    manifest_path = package / "release_manifest_submission.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {entry["path"]: entry for entry in manifest["entries"]}
    actual = {
        path.relative_to(package).as_posix(): path
        for path in package.rglob("*")
        if path.is_file()
    }

    errors: list[str] = []
    missing = sorted(set(entries) - set(actual))
    unindexed = sorted(set(actual) - set(entries))
    if missing:
        errors.append(f"missing files: {missing}")
    if unindexed:
        errors.append(f"unindexed files: {unindexed}")

    for relative, path in sorted(actual.items()):
        entry = entries.get(relative)
        if not entry:
            continue
        if entry.get("self_referential"):
            if relative != "release_manifest_submission.json":
                errors.append(f"unexpected self-referential entry: {relative}")
            if entry.get("sha256") is not None or entry.get("bytes") is not None:
                errors.append("manifest self-entry must use null sha256 and bytes")
            continue
        if path.stat().st_size != entry.get("bytes"):
            errors.append(f"byte mismatch: {relative}")
        if sha256_file(path) != entry.get("sha256"):
            errors.append(f"sha256 mismatch: {relative}")

    status = "PASS" if not errors else "FAIL"
    print(json.dumps({
        "status": status,
        "package": str(package),
        "actual_file_count": len(actual),
        "manifest_entry_count": len(entries),
        "hashed_file_count": sum(not e.get("self_referential", False) for e in entries.values()),
        "errors": errors,
    }, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
