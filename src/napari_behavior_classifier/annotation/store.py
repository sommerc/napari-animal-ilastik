"""Sparse per-(source_file, individual, frame) class label storage.

Source file identity is part of the key (not just individual+frame) because
frame numbers restart at 0 in every recording - without it, labeling frame
100 in one file would silently collide with frame 100 in another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class LabelStore:
    labels: dict[tuple[str, str, int], str] = field(default_factory=dict)
    _listeners: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def add_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    @staticmethod
    def _normalize(source_file: str) -> str:
        """Canonicalize path strings (slash direction etc.) so a key set via one
        representation (e.g. forward slashes in a hand-built or cross-platform
        session.json) is still found under another (e.g. `str(Path(...))`'s
        OS-native form, which is what the widget and `api.py` both use)."""
        return str(Path(source_file))

    def set(self, source_file: str, individual: str, frame: int, class_name: str) -> None:
        self.labels[(self._normalize(source_file), individual, frame)] = class_name
        self._notify()

    def clear(self, source_file: str, individual: str, frame: int) -> None:
        if self.labels.pop((self._normalize(source_file), individual, frame), None) is not None:
            self._notify()

    def set_range(
        self, source_file: str, individual: str, start_frame: int, end_frame: int, class_name: str
    ) -> None:
        """Label every frame in [start_frame, end_frame] (inclusive, order-independent) as one bout."""
        source_file = self._normalize(source_file)
        lo, hi = sorted((start_frame, end_frame))
        for frame in range(lo, hi + 1):
            self.labels[(source_file, individual, frame)] = class_name
        self._notify()

    def clear_range(self, source_file: str, individual: str, start_frame: int, end_frame: int) -> None:
        source_file = self._normalize(source_file)
        lo, hi = sorted((start_frame, end_frame))
        changed = False
        for frame in range(lo, hi + 1):
            if self.labels.pop((source_file, individual, frame), None) is not None:
                changed = True
        if changed:
            self._notify()

    def rename_class(self, old_name: str, new_name: str) -> None:
        """Relabel every frame currently tagged `old_name` as `new_name` (keys unchanged)."""
        changed = False
        for key, cls in self.labels.items():
            if cls == old_name:
                self.labels[key] = new_name
                changed = True
        if changed:
            self._notify()

    def get(self, source_file: str, individual: str, frame: int) -> str | None:
        return self.labels.get((self._normalize(source_file), individual, frame))

    def labeled_frames(self, source_file: str, individual: str) -> list[int]:
        source_file = self._normalize(source_file)
        return sorted(f for (s, i, f) in self.labels if s == source_file and i == individual)

    def source_files(self) -> list[str]:
        return sorted({s for (s, _i, _f) in self.labels})

    def to_dense_array(self, source_file: str, individual: str, n_frames: int) -> np.ndarray:
        """(n_frames,) object array of class names, None where unlabeled."""
        source_file = self._normalize(source_file)
        arr = np.full(n_frames, None, dtype=object)
        for (s, ind, frame), cls in self.labels.items():
            if s == source_file and ind == individual:
                arr[frame] = cls
        return arr

    def to_dataframe(self):
        import pandas as pd

        rows = [(s, i, f, c) for (s, i, f), c in self.labels.items()]
        return pd.DataFrame(rows, columns=["source_file", "individual", "frame", "class"])

    def save(self, path: str | Path) -> None:
        self.to_dataframe().to_csv(path, index=False)

    @classmethod
    def load(cls, path: str | Path) -> "LabelStore":
        import pandas as pd

        df = pd.read_csv(path)
        store = cls()
        # "class" is a reserved keyword, so itertuples can't use attribute access for it;
        # plain unnamed tuples sidestep that entirely.
        for source_file, individual, frame, class_name in df[
            ["source_file", "individual", "frame", "class"]
        ].itertuples(index=False, name=None):
            store.labels[(cls._normalize(source_file), individual, int(frame))] = class_name
        return store

    def _notify(self) -> None:
        for callback in self._listeners:
            callback()
