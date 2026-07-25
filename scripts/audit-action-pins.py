#!/usr/bin/env python3
"""Audit GitHub Actions workflow/composite action refs.

Fails (non-zero exit) if any external `uses:` reference is not pinned to a
full 40-character commit SHA. Local refs (`./...`) and Docker refs
(`docker://...`) are ignored. Run from the repository root:

    python3 scripts/audit-action-pins.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is optional, not a project dependency
    yaml = None

WORKFLOWS_DIR = Path(".github/workflows")
USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def find_uses_refs(text: str) -> list[str]:
    """Extract `uses:` values, including within commented-out YAML blocks."""
    refs = []
    for line in text.splitlines():
        stripped = line.lstrip("# ").strip()
        match = USES_RE.match(stripped) or USES_RE.match(line)
        if match:
            refs.append(match.group(1))
    return refs


def is_pinned(ref: str) -> bool:
    if ref.startswith("./") or ref.startswith("docker://"):
        return True
    if "@" not in ref:
        return False
    _, _, version = ref.rpartition("@")
    return bool(SHA_RE.match(version))


def validate_yaml(path: Path) -> str | None:
    if yaml is None:
        return None
    try:
        yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        return str(exc)
    return None


def main() -> int:
    if not WORKFLOWS_DIR.is_dir():
        print(f"No workflows directory found at {WORKFLOWS_DIR}", file=sys.stderr)
        return 1

    unpinned: list[tuple[Path, str]] = []
    yaml_errors: list[tuple[Path, str]] = []

    for path in sorted(WORKFLOWS_DIR.glob("*.y*ml")):
        error = validate_yaml(path)
        if error:
            yaml_errors.append((path, error))
            continue
        for ref in find_uses_refs(path.read_text()):
            if not is_pinned(ref):
                unpinned.append((path, ref))

    if yaml_errors:
        print("YAML validation errors:")
        for path, error in yaml_errors:
            print(f"  {path}: {error}")

    if unpinned:
        print("Unpinned/mutable external action refs found:")
        for path, ref in unpinned:
            print(f"  {path}: {ref}")

    if yaml_errors or unpinned:
        return 1

    suffix = "" if yaml is not None else " (YAML syntax check skipped: pyyaml not installed)"
    print(f"OK: all external action refs are pinned to full commit SHAs and YAML is valid.{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
