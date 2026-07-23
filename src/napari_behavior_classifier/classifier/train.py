"""On-demand RandomForest training/prediction over labeled frames.

Not live yet (phase 3): the caller triggers train + predict explicitly.
Kept as a plain function pair (not a class) since there's no state to hold
between calls beyond the fitted sklearn Pipeline the caller already owns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline() -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        # n_jobs=-1: parallelize both fit and predict (incl. OOB scoring) across all
        # available CPU cores - sklearn's default (n_jobs=None) is single-threaded.
        RandomForestClassifier(n_estimators=100, random_state=0, oob_score=True, n_jobs=-1),
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


@dataclass
class OOBReport:
    """Out-of-bag evaluation, structured for a per-class table (metrics + confusion matrix)."""

    accuracy: float
    classes: list[str]
    precision: np.ndarray  # (n_classes,)
    recall: np.ndarray  # (n_classes,)
    f1: np.ndarray  # (n_classes,)
    support: np.ndarray  # (n_classes,) - OOB-valid frame count per class, may be < annotated count
    confusion: np.ndarray  # (n_classes, n_classes), rows=true class, cols=predicted class, same order as `classes`


def oob_summary(pipeline: Pipeline, features: np.ndarray, labels_by_frame: dict[int, str]) -> OOBReport | None:
    """Out-of-bag accuracy, per-class precision/recall/F1, and a confusion matrix.

    All come "for free" from the RandomForest's own bagging (each tree
    predicts only the labeled frames it didn't train on), giving an honest
    generalization estimate without needing a separate held-out split.
    Returns None if too few labeled frames per class leaves no sample with a
    usable out-of-bag prediction.
    """
    frames = sorted(labels_by_frame)
    y = np.array([labels_by_frame[f] for f in frames])

    rf = pipeline.named_steps["randomforestclassifier"]
    oob_proba = rf.oob_decision_function_  # (n_labeled, n_classes); rows can be all-NaN
    valid = ~np.isnan(oob_proba).any(axis=1)
    if not valid.any():
        return None

    y_pred = rf.classes_[np.argmax(oob_proba[valid], axis=1)]
    y_true = y[valid]
    classes = list(rf.classes_)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    confusion = confusion_matrix(y_true, y_pred, labels=classes)

    return OOBReport(
        accuracy=rf.oob_score_,
        classes=classes,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        confusion=confusion,
    )


def save_pipeline(pipeline: Pipeline, path: str | Path) -> None:
    joblib.dump(pipeline, path)


def load_pipeline(path: str | Path) -> Pipeline:
    return joblib.load(path)
