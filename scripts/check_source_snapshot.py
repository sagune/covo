#!/usr/bin/env python
"""Verify source-only snapshot hashes and Python syntax, without importing ML code."""
import ast
import hashlib
import json
from pathlib import Path
import sys


def main():
    snapshot = Path(__file__).resolve().parents[1] / 'server_snapshot'
    manifest = json.loads((snapshot / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    failures = []
    checked = 0
    for record in manifest['files']:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            failures.append({'path': str(relative), 'error': 'unsafe path'})
            continue
        path = snapshot / relative
        if not path.is_file():
            failures.append({'path': str(relative), 'error': 'missing'})
            continue
        blob = path.read_bytes()
        if len(blob) != record['bytes'] or hashlib.sha256(blob).hexdigest() != record['sha256']:
            failures.append({'path': str(relative), 'error': 'hash/size mismatch'})
        if path.suffix == '.py':
            checked += 1
            try:
                ast.parse(blob.decode('utf-8-sig'), filename=str(relative))
            except (SyntaxError, UnicodeError) as error:
                failures.append({'path': str(relative), 'error': type(error).__name__,
                                 'line': getattr(error, 'lineno', None)})
    print(json.dumps({'status': 'PASS' if not failures else 'FAIL',
                      'manifest_files': len(manifest['files']), 'python_files': checked,
                      'failures': failures}, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
