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

    def set(self, source_file: str, individual: str, frame: int, class_name: str) -> None:
        self.labels[(source_file, individual, frame)] = class_name
        self._notify()

    def clear(self, source_file: str, individual: str, frame: int) -> None:
        if self.labels.pop((source_file, individual, frame), None) is not None:
            self._notify()

    def get(self, source_file: str, individual: str, frame: int) -> str | None:
        return self.labels.get((source_file, individual, frame))

    def labeled_frames(self, source_file: str, individual: str) -> list[int]:
        return sorted(f for (s, i, f) in self.labels if s == source_file and i == individual)

    def source_files(self) -> list[str]:
        return sorted({s for (s, _i, _f) in self.labels})

    def to_dense_array(self, source_file: str, individual: str, n_frames: int) -> np.ndarray:
        """(n_frames,) object array of class names, None where unlabeled."""
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
            store.labels[(source_file, individual, int(frame))] = class_name
        return store

    def _notify(self) -> None:
        for callback in self._listeners:
            callback()
