"""Programmatic (non-GUI) API: train across multiple .h5 files, predict on new ones.

Mirrors what the interactive widget does, with zero napari/Qt dependency,
for headless/scripted workflows. Each .h5 file's annotations are expected
to live in a sidecar CSV produced by `annotation.store.LabelStore.save()`
(the same "Save annotations" button the widget exposes).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.pipeline import Pipeline

from .annotation.store import LabelStore
from .classifier import train as train_module
from .features import filterbank, kinematics
from .io.h5_reader import Skeleton, check_consistent_skeleton, extract_skeleton, load_tracks
from .session import load_session


@dataclass
class TrainResult:
    """A pooled-training run: the fitted pipeline plus everything needed to report on it
    (mirrors what the widget's `_TrainingSummaryDialog` shows)."""

    pipeline: Pipeline
    oob_report: train_module.OOBReport | None
    class_counts: Counter
    individuals: list[str]
    files: list[str]
    n_features: int


def compute_all_features(h5_path: str | Path, individual: str) -> tuple[np.ndarray, list[str]]:
    """Raw + filter-bank features for one file/individual, ready for train/predict."""
    ds = load_tracks(h5_path)
    raw_features, raw_names = kinematics.compute_features(ds, individual)
    return filterbank.with_filter_bank(raw_features, raw_names)


def _load_ds_checking_skeleton(
    h5_path: str | Path, reference_skeleton: Skeleton | None
) -> tuple[xr.Dataset, Skeleton]:
    """Load one file's dataset, verifying its skeleton matches every other pooled
    file's (see `io.h5_reader.check_consistent_skeleton` for why this matters).
    `reference_skeleton` is None for the first file in a pool - it just gets recorded.
    """
    h5_path_str = str(Path(h5_path))
    ds = load_tracks(h5_path_str)
    skeleton = extract_skeleton(ds)
    if reference_skeleton is not None:
        check_consistent_skeleton(reference_skeleton, skeleton, context=h5_path_str)
    return ds, skeleton


def _pool_labeled_individual(
    ds: xr.Dataset,
    store: LabelStore,
    h5_path_str: str,
    individual: str,
    offset: int,
    all_features: list,
    all_labels: dict,
    contributing_individuals: set,
    contributing_files: set,
) -> int:
    """Append this (file, individual)'s features/labels to the pooled training set,
    if it has any labeled frames at all; returns the new offset."""
    labeled_frames = store.labeled_frames(h5_path_str, individual)
    if not labeled_frames:
        return offset
    raw_features, raw_names = kinematics.compute_features(ds, individual)
    features, _names = filterbank.with_filter_bank(raw_features, raw_names)
    for frame in labeled_frames:
        all_labels[offset + frame] = store.get(h5_path_str, individual, frame)
    all_features.append(features)
    contributing_individuals.add(individual)
    contributing_files.add(h5_path_str)
    return offset + features.shape[1]


def _build_train_result(all_features: list, all_labels: dict, contributing_individuals: set, contributing_files: set) -> TrainResult:
    if len(set(all_labels.values())) < 2:
        raise ValueError("Need labeled frames from at least 2 classes, across any animals/files")

    combined = np.concatenate(all_features, axis=1)
    pipeline = train_module.train(combined, all_labels)
    report = train_module.oob_summary(pipeline, combined, all_labels)
    class_counts = Counter(all_labels.values())

    return TrainResult(
        pipeline=pipeline,
        oob_report=report,
        class_counts=class_counts,
        individuals=sorted(contributing_individuals),
        files=sorted(Path(f).name for f in contributing_files),
        n_features=combined.shape[0],
    )


def train_from_files(h5_paths: list[str | Path], annotation_paths: list[str | Path]) -> TrainResult:
    """Train one RandomForest pipeline pooling labeled frames across every tracked
    individual and every .h5 file (classes are monadic: a label like "grooming" means
    the same thing regardless of which animal it's attached to, so all animals'
    labeled frames feed one shared classifier).

    `annotation_paths[i]` is the CSV saved by `LabelStore.save()` for `h5_paths[i]`.
    Frame indices are offset per (file, individual) internally so labels never collide
    in the pooled training set.
    """
    if len(h5_paths) != len(annotation_paths):
        raise ValueError("h5_paths and annotation_paths must be the same length")

    all_features = []
    all_labels: dict[int, str] = {}
    offset = 0
    reference_skeleton: Skeleton | None = None
    contributing_individuals: set[str] = set()
    contributing_files: set[str] = set()

    for h5_path, annotation_path in zip(h5_paths, annotation_paths):
        h5_path_str = str(Path(h5_path))
        ds, reference_skeleton = _load_ds_checking_skeleton(h5_path, reference_skeleton)
        store = LabelStore.load(annotation_path)
        for individual in [str(v) for v in ds.coords["individuals"].values]:
            offset = _pool_labeled_individual(
                ds, store, h5_path_str, individual, offset, all_features, all_labels,
                contributing_individuals, contributing_files,
            )

    return _build_train_result(all_features, all_labels, contributing_individuals, contributing_files)


def train_from_session(session_path: str | Path) -> TrainResult:
    """Train one pipeline pooling every file + individual + annotation in a session
    saved via `session.save_session` (see `train_from_files` for the pooling rationale)."""
    h5_paths, store, _class_colors, _model_path = load_session(session_path)

    all_features = []
    all_labels: dict[int, str] = {}
    offset = 0
    reference_skeleton: Skeleton | None = None
    contributing_individuals: set[str] = set()
    contributing_files: set[str] = set()

    for h5_path in h5_paths:
        h5_path_str = str(Path(h5_path))
        ds, reference_skeleton = _load_ds_checking_skeleton(h5_path, reference_skeleton)
        for individual in [str(v) for v in ds.coords["individuals"].values]:
            offset = _pool_labeled_individual(
                ds, store, h5_path_str, individual, offset, all_features, all_labels,
                contributing_individuals, contributing_files,
            )

    return _build_train_result(all_features, all_labels, contributing_individuals, contributing_files)


def predict_file(pipeline: Pipeline, h5_path: str | Path, individual: str) -> np.ndarray:
    """Predict a class label per frame for one file/individual with an already-trained pipeline."""
    features, _names = compute_all_features(h5_path, individual)
    return train_module.predict(pipeline, features)


def predict_file_all_individuals(pipeline: Pipeline, h5_path: str | Path) -> dict[str, np.ndarray]:
    """Predict a class label per frame for every tracked individual in one file."""
    ds = load_tracks(str(Path(h5_path)))
    predictions = {}
    for individual in [str(v) for v in ds.coords["individuals"].values]:
        raw_features, raw_names = kinematics.compute_features(ds, individual)
        features, _names = filterbank.with_filter_bank(raw_features, raw_names)
        predictions[individual] = train_module.predict(pipeline, features)
    return predictions


save_pipeline = train_module.save_pipeline
load_pipeline = train_module.load_pipeline
