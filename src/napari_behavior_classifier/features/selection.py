"""Which (feature group, filter-bank scale) cells are enabled - the model behind
the "Select features" matrix dialog (Y axis: feature group name, from
`kinematics.FEATURE_GROUPS`; X axis: `RAW_SCALE` plus one column per filter-bank
sigma from `filterbank.DEFAULT_SIGMAS`).

Checking (group, RAW_SCALE) includes that group's raw, unfiltered channels.
Checking (group, scale_label(sigma)) includes that group's channels filtered at
that sigma - all three filter kinds together (smooth/rate/variability), since the
dialog is a 2D group x scale grid, not a 3D group x scale x filter-kind one.

Persisted verbatim both in the saved session (so reopening it remembers the
choice) and inside the saved model bundle (`classifier.train.SavedModel`) - a
model's own feature_selection is what predicting later must replay, since the
live session selection may have changed since that model was trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import filterbank, kinematics

RAW_SCALE = "raw"


def scale_label(sigma: float) -> str:
    """Matches the suffix `filterbank.apply_filter_bank` already puts on filtered
    feature names (`..._smooth_s3`), so a column's key and its features' name
    suffix agree without a separate lookup table."""
    return f"s{sigma:g}"


@dataclass
class FeatureSelection:
    enabled: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def all_enabled(
        cls, group_names: list[str], sigmas: tuple[float, ...] = filterbank.DEFAULT_SIGMAS
    ) -> FeatureSelection:
        """Every group, raw + every scale - matches the plugin's behavior before
        this selection existed, so it's the right default for old sessions/models
        that predate it."""
        scales = {RAW_SCALE, *(scale_label(s) for s in sigmas)}
        return cls(enabled={name: set(scales) for name in group_names})

    def is_enabled(self, group_name: str, scale: str) -> bool:
        return scale in self.enabled.get(group_name, set())

    def copy(self) -> FeatureSelection:
        return FeatureSelection(enabled={name: set(scales) for name, scales in self.enabled.items()})

    def cache_key(self) -> tuple:
        """Hashable snapshot, for keying a cache of already-computed features by
        which selection produced them."""
        return tuple(sorted((name, tuple(sorted(scales))) for name, scales in self.enabled.items()))

    def to_dict(self) -> dict[str, list[str]]:
        return {name: sorted(scales) for name, scales in self.enabled.items()}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> FeatureSelection:
        return cls(enabled={name: set(scales) for name, scales in data.items()})


def compute_selected_features(
    groups: list[kinematics.FeatureGroup],
    selection: FeatureSelection,
    sigmas: tuple[float, ...] = filterbank.DEFAULT_SIGMAS,
) -> tuple[np.ndarray, list[str]]:
    """Raw and/or filtered features per `selection`, concatenated in group order
    (raw channels before that same group's filtered ones, matching
    `filterbank.with_filter_bank`'s existing raw-then-filtered convention)."""
    rows: list[np.ndarray] = []
    names: list[str] = []

    for group in groups:
        if selection.is_enabled(group.name, RAW_SCALE):
            rows.append(group.features)
            names.extend(group.names)

        selected_sigmas = tuple(s for s in sigmas if selection.is_enabled(group.name, scale_label(s)))
        if selected_sigmas:
            filtered, filtered_names = filterbank.apply_filter_bank(group.features, group.names, selected_sigmas)
            rows.append(filtered)
            names.extend(filtered_names)

    if not rows:
        n_frames = groups[0].features.shape[1] if groups else 0
        return np.zeros((0, n_frames), dtype=np.float32), []

    return np.concatenate(rows, axis=0), names
