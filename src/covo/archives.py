"""Archive helpers for bundled small data artifacts."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Dict


def unpack_zip_member(zip_path: str | Path, output_path: str | Path, *, member: str = "", force: bool = False) -> Dict[str, object]:
    archive_path = Path(zip_path)
    target_path = Path(output_path)
    if target_path.exists() and not force:
        return {
            "input": str(archive_path),
            "output": str(target_path),
            "member": member,
            "status": "exists",
            "bytes": target_path.stat().st_size,
        }

    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        selected = member
        if not selected:
            if len(names) == 1:
                selected = names[0]
            elif target_path.name in names:
                selected = target_path.name
            else:
                raise ValueError(f"Archive has multiple files; pass --member. Members: {names}")
        if selected not in names:
            raise ValueError(f"Archive member not found: {selected}")
        if target_path.parent:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(selected) as reader, target_path.open("wb") as writer:
            shutil.copyfileobj(reader, writer)

    return {
        "input": str(archive_path),
        "output": str(target_path),
        "member": selected,
        "status": "written",
        "bytes": target_path.stat().st_size,
    }
