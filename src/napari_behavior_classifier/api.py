"""Programmatic (non-GUI) API: train across multiple .slp files, predict on new ones.

Mirrors what the interactive widget does, with zero napari/Qt dependency,
for headless/scripted workflows. Each .slp file's annotations are expected
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
from .io.slp_reader import load_tracks
from .session import load_session


def compute_all_features(slp_path: str | Path, individual: str) -> tuple[np.ndarray, list[str]]:
    """Raw + filter-bank features for one file/individual, ready for train/predict."""
    ds = load_tracks(slp_path)
    raw_features, raw_names = kinematics.compute_features(ds, individual)
    return filterbank.with_filter_bank(raw_features, raw_names)


def train_from_files(
    slp_paths: list[str | Path],
    annotation_paths: list[str | Path],
    individual: str,
) -> Pipeline:
    """Train one RandomForest pipeline on labeled frames pooled across multiple .slp files.

    `annotation_paths[i]` is the CSV saved by `LabelStore.save()` for `slp_paths[i]`.
    Frame indices are offset per file internally so labels from different
    files never collide in the pooled training set.
    """
    if len(slp_paths) != len(annotation_paths):
        raise ValueError("slp_paths and annotation_paths must be the same length")

    all_features = []
    all_labels: dict[int, str] = {}
    offset = 0

    for slp_path, annotation_path in zip(slp_paths, annotation_paths):
        slp_path_str = str(Path(slp_path))
        features, _names = compute_all_features(slp_path, individual)
        store = LabelStore.load(annotation_path)
        for frame in store.labeled_frames(slp_path_str, individual):
            all_labels[offset + frame] = store.get(slp_path_str, individual, frame)
        all_features.append(features)
        offset += features.shape[1]

    combined = np.concatenate(all_features, axis=1)
    return train_module.train(combined, all_labels)


def train_from_session(session_path: str | Path, individual: str) -> Pipeline:
    """Train one pipeline pooling every file + annotation in a session saved via `session.save_session`."""
    slp_paths, store, _class_colors = load_session(session_path)

    all_features = []
    all_labels: dict[int, str] = {}
    offset = 0

    for slp_path in slp_paths:
        slp_path_str = str(Path(slp_path))
        features, _names = compute_all_features(slp_path, individual)
        for frame in store.labeled_frames(slp_path_str, individual):
            all_labels[offset + frame] = store.get(slp_path_str, individual, frame)
        all_features.append(features)
        offset += features.shape[1]

    combined = np.concatenate(all_features, axis=1)
    return train_module.train(combined, all_labels)


def predict_file(pipeline: Pipeline, slp_path: str | Path, individual: str) -> np.ndarray:
    """Predict a class label per frame for one file/individual with an already-trained pipeline."""
    features, _names = compute_all_features(slp_path, individual)
    return train_module.predict(pipeline, features)


save_pipeline = train_module.save_pipeline
load_pipeline = train_module.load_pipeline
