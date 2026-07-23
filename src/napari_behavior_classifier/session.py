"""Save/restore GUI session state: which files were open and their annotations.

Raw/filtered feature *values* are deliberately NOT persisted here - they're
cheap to recompute from the .h5 files themselves, which keeps session files
small and avoids staleness if the feature/filter code changes between
sessions. Class colors are saved so the palette stays consistent on reload.
Which feature groups/scales are *selected* (see `features.selection`) IS
persisted, since that's a user choice, not a derived value.

A trained model, if present, IS persisted - as a sibling `.joblib` file next
to the session .json, referenced by filename only (`model_path` below) so
the pair stays portable if the two files are moved together.
"""

from __future__ import annotations

import json
from pathlib import Path

from .annotation.store import LabelStore
from .features.selection import FeatureSelection


def save_session(
    path: str | Path,
    h5_paths: list[str],
    store: LabelStore,
    class_colors: dict[str, str],
    model_path: str | None = None,
    feature_selection: FeatureSelection | None = None,
) -> None:
    data = {
        "h5_paths": h5_paths,
        "class_colors": class_colors,
        "model_path": model_path,
        "feature_selection": feature_selection.to_dict() if feature_selection is not None else None,
        "labels": [
            {"source_file": source_file, "individual": individual, "frame": frame, "class": class_name}
            for (source_file, individual, frame), class_name in store.labels.items()
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_session(
    path: str | Path,
) -> tuple[list[str], LabelStore, dict[str, str], str | None, FeatureSelection | None]:
    data = json.loads(Path(path).read_text())

    store = LabelStore()
    for row in data.get("labels", []):
        store.set(row["source_file"], row["individual"], row["frame"], row["class"])

    raw_selection = data.get("feature_selection")
    feature_selection = FeatureSelection.from_dict(raw_selection) if raw_selection is not None else None

    return data.get("h5_paths", []), store, data.get("class_colors", {}), data.get("model_path"), feature_selection
