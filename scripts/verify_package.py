"""Verify every published file and reject unindexed additions."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]
expected = json.loads((ROOT/'SHA256SUMS.json').read_text(encoding='utf-8'))
actual = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in ROOT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.json'
          and '.git' not in p.relative_to(ROOT).parts}
assert actual == expected, 'Integrity mismatch, missing file or unindexed extra file'
print(f'PASS: {len(actual)} indexed files; hashes match; no extra files')
