"""On-demand RandomForest training/prediction over labeled frames.

Not live yet (phase 3): the caller triggers train + predict explicitly.
Kept as a plain function pair (not a class) since there's no state to hold
between calls beyond the fitted sklearn Pipeline the caller already owns.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline() -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, random_state=0),
    )


def train(features: np.ndarray, labels_by_frame: dict[int, str]) -> Pipeline:
    """features: (n_features, n_frames). Fits only on the given labeled frames."""
    if not labels_by_frame:
        raise ValueError("no labeled frames to train on")

    frames = sorted(labels_by_frame)
    x = features[:, frames].T  # (n_labeled, n_features)
    y = [labels_by_frame[f] for f in frames]

    pipeline = build_pipeline()
    pipeline.fit(x, y)
    return pipeline


def predict(pipeline: Pipeline, features: np.ndarray) -> np.ndarray:
    """features: (n_features, n_frames) -> (n_frames,) predicted class labels."""
    return pipeline.predict(features.T)


def save_pipeline(pipeline: Pipeline, path: str | Path) -> None:
    joblib.dump(pipeline, path)


def load_pipeline(path: str | Path) -> Pipeline:
    return joblib.load(path)
