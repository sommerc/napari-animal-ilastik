"""Build napari layer data for a single current frame.

Overlays are recomputed per frame (not pre-baked for the whole video) so
memory stays flat regardless of video length, and the same refresh path
will later double as the live-prediction overlay update.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def frame_points(ds: xr.Dataset, frame: int) -> tuple[np.ndarray, dict]:
    """(y, x) points for all individuals/keypoints at `frame`, napari Points-ready."""
    pos = ds["position"].isel(time=frame).values  # (individuals, keypoints, 2) xy
    conf = ds["confidence"].isel(time=frame).values  # (individuals, keypoints)

    individuals = ds.coords["individuals"].values
    keypoints = ds.coords["keypoints"].values

    rows, individual_labels, keypoint_labels, confidences = [], [], [], []
    for i, individual in enumerate(individuals):
        for k, keypoint in enumerate(keypoints):
            x, y = pos[i, k]
            if np.isnan(x) or np.isnan(y):
                continue
            rows.append((y, x))
            individual_labels.append(individual)
            keypoint_labels.append(keypoint)
            confidences.append(conf[i, k])

    data = np.array(rows, dtype=np.float32) if rows else np.empty((0, 2), dtype=np.float32)
    properties = dict(
        individual=np.array(individual_labels),
        keypoint=np.array(keypoint_labels),
        confidence=np.array(confidences, dtype=np.float32),
    )
    return data, properties


def frame_skeleton_lines(ds: xr.Dataset, frame: int) -> list[np.ndarray]:
    """One (2, 2) [y, x] line per visible skeleton edge, per individual, at `frame`."""
    pos = ds["position"].isel(time=frame).values  # (individuals, keypoints, 2) xy
    edges = ds.attrs["skeleton_edges"]

    lines = []
    for i in range(pos.shape[0]):
        for src, dst in edges:
            x0, y0 = pos[i, src]
            x1, y1 = pos[i, dst]
            if np.isnan([x0, y0, x1, y1]).any():
                continue
            lines.append(np.array([[y0, x0], [y1, x1]], dtype=np.float32))
    return lines


def frame_bboxes(ds: xr.Dataset, frame: int, padding: float = 5.0) -> list[tuple[str, np.ndarray]]:
    """(individual, (4, 2) [y, x] rectangle) per individual with >=1 visible keypoint, at `frame`."""
    pos = ds["position"].isel(time=frame).values  # (individuals, keypoints, 2) xy
    individuals = ds.coords["individuals"].values

    boxes = []
    for i in range(pos.shape[0]):
        xy = pos[i]
        if np.all(np.isnan(xy)):
            continue
        x_min, y_min = np.nanmin(xy, axis=0) - padding
        x_max, y_max = np.nanmax(xy, axis=0) + padding
        box = np.array([[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]], dtype=np.float32)
        boxes.append((str(individuals[i]), box))
    return boxes
