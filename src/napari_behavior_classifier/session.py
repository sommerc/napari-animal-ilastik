"""Save/restore GUI session state: which files were open and their annotations.

Features and trained models are deliberately NOT persisted here - they're
cheap to recompute from the .h5 files themselves, which keeps session files
small and avoids staleness if the feature/filter code changes between
sessions. Class colors are saved so the palette stays consistent on reload.
"""

from __future__ import annotations

import json
from pathlib import Path

from .annotation.store import LabelStore


def save_session(
    path: str | Path,
    h5_paths: list[str],
    store: LabelStore,
    class_colors: dict[str, str],
) -> None:
    data = {
        "h5_paths": h5_paths,
        "class_colors": class_colors,
        "labels": [
            {"source_file": source_file, "individual": individual, "frame": frame, "class": class_name}
            for (source_file, individual, frame), class_name in store.labels.items()
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_session(path: str | Path) -> tuple[list[str], LabelStore, dict[str, str]]:
    data = json.loads(Path(path).read_text())

    store = LabelStore()
    for row in data.get("labels", []):
        store.labels[(row["source_file"], row["individual"], row["frame"])] = row["class"]

    return data.get("h5_paths", []), store, data.get("class_colors", {})
