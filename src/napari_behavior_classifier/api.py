"""Programmatic (non-GUI) API: train across multiple .h5 files, predict on new ones.

Mirrors what the interactive widget does, with zero napari/Qt dependency,
for headless/scripted workflows. Each .h5 file's annotations are expected
to live in a sidecar CSV produced by `annotation.store.LabelStore.save()`
(the same "Save annotations" button the widget exposes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from .annotation.store import LabelStore
from .classifier import train as train_module
from .features import filterbank, kinematics
from .io.h5_reader import Skeleton, check_consistent_skeleton, extract_skeleton, load_tracks
from .session import load_session


def compute_all_features(h5_path: str | Path, individual: str) -> tuple[np.ndarray, list[str]]:
    """Raw + filter-bank features for one file/individual, ready for train/predict."""
    ds = load_tracks(h5_path)
    raw_features, raw_names = kinematics.compute_features(ds, individual)
    return filterbank.with_filter_bank(raw_features, raw_names)


def _load_features_checking_skeleton(
    h5_path: str | Path, individual: str, reference_skeleton: Skeleton | None
) -> tuple[np.ndarray, Skeleton]:
    """Load one file's combined features, verifying its skeleton matches every other
    pooled file's (see `io.h5_reader.check_consistent_skeleton` for why this matters).
    `reference_skeleton` is None for the first file in a pool - it just gets recorded.
    """
    h5_path_str = str(Path(h5_path))
    ds = load_tracks(h5_path_str)
    skeleton = extract_skeleton(ds)
    if reference_skeleton is not None:
        check_consistent_skeleton(reference_skeleton, skeleton, context=h5_path_str)

    raw_features, raw_names = kinematics.compute_features(ds, individual)
    features, _names = filterbank.with_filter_bank(raw_features, raw_names)
    return features, skeleton


def train_from_files(
    h5_paths: list[str | Path],
    annotation_paths: list[str | Path],
    individual: str,
) -> Pipeline:
    """Train one RandomForest pipeline on labeled frames pooled across multiple .h5 files.

    `annotation_paths[i]` is the CSV saved by `LabelStore.save()` for `h5_paths[i]`.
    Frame indices are offset per file internally so labels from different
    files never collide in the pooled training set.
    """
    if len(h5_paths) != len(annotation_paths):
        raise ValueError("h5_paths and annotation_paths must be the same length")

    all_features = []
    all_labels: dict[int, str] = {}
    offset = 0
    reference_skeleton: Skeleton | None = None

    for h5_path, annotation_path in zip(h5_paths, annotation_paths):
        h5_path_str = str(Path(h5_path))
        features, reference_skeleton = _load_features_checking_skeleton(h5_path, individual, reference_skeleton)
        store = LabelStore.load(annotation_path)
        for frame in store.labeled_frames(h5_path_str, individual):
            all_labels[offset + frame] = store.get(h5_path_str, individual, frame)
        all_features.append(features)
        offset += features.shape[1]

    combined = np.concatenate(all_features, axis=1)
    return train_module.train(combined, all_labels)


def train_from_session(session_path: str | Path, individual: str) -> Pipeline:
    """Train one pipeline pooling every file + annotation in a session saved via `session.save_session`."""
    h5_paths, store, _class_colors = load_session(session_path)

    all_features = []
    all_labels: dict[int, str] = {}
    offset = 0
    reference_skeleton: Skeleton | None = None

    for h5_path in h5_paths:
        h5_path_str = str(Path(h5_path))
        features, reference_skeleton = _load_features_checking_skeleton(h5_path, individual, reference_skeleton)
        for frame in store.labeled_frames(h5_path_str, individual):
            all_labels[offset + frame] = store.get(h5_path_str, individual, frame)
        all_features.append(features)
        offset += features.shape[1]

    combined = np.concatenate(all_features, axis=1)
    return train_module.train(combined, all_labels)


def predict_file(pipeline: Pipeline, h5_path: str | Path, individual: str) -> np.ndarray:
    """Predict a class label per frame for one file/individual with an already-trained pipeline."""
    features, _names = compute_all_features(h5_path, individual)
    return train_module.predict(pipeline, features)


save_pipeline = train_module.save_pipeline
load_pipeline = train_module.load_pipeline
