"""Per-individual kinematic features derived from bodypart coordinates.

Distances follow skeleton edges (bone lengths); segment angles are computed
at any node with exactly two neighbors (a chain joint, e.g. along a tail),
since a 3+-way junction has no single well-defined bend. Speeds are simple
frame-to-frame displacements, not smoothed - smoothing/derivative filters
belong to the upcoming ilastik-style filter bank, not here.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _node_adjacency(edges: list[tuple[int, int]], n_nodes: int) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def compute_features(ds: xr.Dataset, individual: str) -> tuple[np.ndarray, list[str]]:
    """Return (n_features, n_frames) float32 array and matching feature names."""
    pos = ds["position"].sel(individuals=individual).values  # (time, keypoints, 2)
    node_names = list(ds.coords["keypoints"].values)
    edges = ds.attrs["skeleton_edges"]
    adjacency = _node_adjacency(edges, len(node_names))

    rows: list[np.ndarray] = []
    names: list[str] = []

    for a, b in edges:
        dist = np.linalg.norm(pos[:, a] - pos[:, b], axis=-1)
        rows.append(dist)
        names.append(f"dist_{node_names[a]}_{node_names[b]}")

    for node, neighbors in adjacency.items():
        if len(neighbors) != 2:
            continue
        prev_node, next_node = neighbors
        incoming = pos[:, node] - pos[:, prev_node]
        outgoing = pos[:, next_node] - pos[:, node]
        cos_theta = np.sum(incoming * outgoing, axis=-1) / (
            np.linalg.norm(incoming, axis=-1) * np.linalg.norm(outgoing, axis=-1)
        )
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        rows.append(angle)
        names.append(f"segment_angle_{node_names[node]}")

    centroid = np.nanmean(pos, axis=1)  # (time, 2)
    rows.append(_speed(centroid))
    names.append("speed_centroid")

    for k, name in enumerate(node_names):
        rows.append(_speed(pos[:, k]))
        names.append(f"speed_{name}")

    features = np.stack(rows, axis=0).astype(np.float32)
    return features, names


def _speed(xy: np.ndarray) -> np.ndarray:
    """Frame-to-frame displacement magnitude, same length as input (first frame = 0)."""
    diffs = np.diff(xy, axis=0, prepend=xy[:1])
    return np.linalg.norm(diffs, axis=-1)
