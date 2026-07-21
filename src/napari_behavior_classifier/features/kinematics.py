"""Per-individual kinematic features derived from bodypart coordinates.

Organized into named, independently computable feature groups (distances,
angles, speed, ...) registered in FEATURE_GROUPS below, so adding a new kind
of feature later is one self-contained `(ds, individual) -> (features,
names)` function plus one registry entry - not a change scattered through a
single long function. The grouping is also what lets the feature-map
visualization draw a labeled, gapped section per group instead of one
undifferentiated stack of rows.

Angles are ordered by walking the skeleton graph (see `discover_chains()`)
rather than raw node index, so a chain like a tail or a limb (root to tip)
lands as a contiguous, anatomically-ordered run of rows. That matters for
the feature-map heatmap: a periodic behavior (a tail beat) shows up as a
visible traveling wave down the chain instead of being scrambled by
whatever order the skeleton happened to be authored in. Chains are walked
starting only from degree-1 endpoint nodes, so anything with no endpoint to
walk from - a cycle (e.g. two eye nodes and the head forming a triangle), or
a direct branch-to-branch edge with no chain nodes in between - has no
natural walk direction and instead keeps the skeleton's original node-index
order, appended after every discovered chain (`_chain_ordered_angle_triples`).
Segment angles are computed at any node with exactly two neighbors (a chain
joint), since a 3+-way junction has no single well-defined bend. Distances
are deliberately *not* chain-reordered - a bone length doesn't carry a "walk
direction" the way an angle does, so they're just listed in the skeleton's
own edge order (see `compute_distances`). Speeds are simple frame-to-frame
displacements, not smoothed - smoothing/derivative filters belong to the
filter bank, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import xarray as xr


def _node_adjacency(edges: list[tuple[int, int]], n_nodes: int) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def discover_chains(edges: list[tuple[int, int]], n_nodes: int) -> list[list[int]]:
    """Find the skeleton's open chains: walk inward from every degree-1 endpoint node,
    through degree-2 nodes, until the first node with degree != 2 is reached (a branch
    point, or another endpoint), then reverse the walk - so a chain reads root-to-tip,
    e.g. Heart_Center -> Hand, not the other way around.

    Starting only from endpoints (never from branch points) means a cycle - e.g. two
    eye nodes and the head forming a triangle - is never returned as a chain: a cycle
    has no degree-1 node to start a walk from, and thus no natural direction to walk
    in either. The same reasoning excludes a direct branch-to-branch edge with no
    chain nodes in between (this frog skeleton's Heart_Center-Tail_Stem body
    connector, for instance) - it isn't reachable from any endpoint walk. Callers
    (`_chain_ordered_edges`/`_chain_ordered_angle_triples`) fall back to the
    skeleton's original edge-list/node-index order for anything left uncovered here.

    Deterministic and depends only on the skeleton's own edges/node count - the same
    skeleton always yields the same chains in the same order, which is what keeps
    `compute_distances`/`compute_angles` row order stable across every file sharing
    that skeleton (see `io.h5_reader.check_consistent_skeleton` for the other half of
    that guarantee - that files actually do share the same skeleton in the first place).
    """
    adjacency = _node_adjacency(edges, n_nodes)
    degree = {node: len(neighbors) for node, neighbors in adjacency.items()}
    endpoints = [node for node in range(n_nodes) if degree[node] == 1]

    visited_edges: set[tuple[int, int]] = set()
    chains: list[list[int]] = []

    for start in endpoints:
        first_step = adjacency[start][0]
        if _edge_key(start, first_step) in visited_edges:
            continue  # the far end of a chain whose other tip is also an endpoint
        chain = [start, first_step]
        visited_edges.add(_edge_key(start, first_step))
        previous, current = start, first_step
        while degree[current] == 2:
            next_node = next(n for n in adjacency[current] if n != previous)
            visited_edges.add(_edge_key(current, next_node))
            chain.append(next_node)
            previous, current = current, next_node
        chains.append(list(reversed(chain)))

    return chains


def _speed(xy: np.ndarray) -> np.ndarray:
    """Frame-to-frame displacement magnitude, same length as input (first frame = 0)."""
    diffs = np.diff(xy, axis=0, prepend=xy[:1])
    return np.linalg.norm(diffs, axis=-1)


def compute_distances(ds: xr.Dataset, individual: str) -> tuple[np.ndarray, list[str]]:
    """Skeleton-edge bone lengths, one per skeleton edge in the skeleton's own order.

    Unlike angles, distances aren't chain-reordered: a bone length doesn't carry the
    same "which direction did we walk" meaning an angle does, so there's no benefit to
    imposing anatomical order here - every edge, junction-to-junction or not, cycle or
    not, is just listed as the skeleton itself lists it.
    """
    pos = ds["position"].sel(individuals=individual).values  # (time, keypoints, 2)
    node_names = list(ds.coords["keypoints"].values)
    edges = ds.attrs["skeleton_edges"]

    rows: list[np.ndarray] = []
    names: list[str] = []
    for a, b in edges:
        dist = np.linalg.norm(pos[:, a] - pos[:, b], axis=-1)
        rows.append(dist)
        names.append(f"dist_{node_names[a]}_{node_names[b]}")

    return np.stack(rows, axis=0).astype(np.float32), names


def compute_angles(ds: xr.Dataset, individual: str) -> tuple[np.ndarray, list[str]]:
    """Signed bend angle at each degree-2 chain node, in [-180, 180] degrees
    (0 = colinear/straight, +-90 = perpendicular, +-180 = folded straight back).

    Signed (not just magnitude) so a left bend and a right bend are distinguishable,
    not just "how much" bend - the sign is which side `outgoing` turns toward relative
    to `incoming`, via the standard 2D cross-product trick (rotate `incoming` 90 deg to
    get a vector perpendicular to it, then the sign of its dot product with `outgoing`
    tells which side of `incoming` that `outgoing` falls on). In chain-walk order
    (open chains only - see module docstring).
    """
    pos = ds["position"].sel(individuals=individual).values  # (time, keypoints, 2)
    node_names = list(ds.coords["keypoints"].values)
    edges = ds.attrs["skeleton_edges"]

    rows: list[np.ndarray] = []
    names: list[str] = []
    for prev_node, node, next_node in _chain_ordered_angle_triples(edges, len(node_names)):
        incoming = pos[:, node] - pos[:, prev_node]
        outgoing = pos[:, next_node] - pos[:, node]
        cos_theta = np.sum(incoming * outgoing, axis=-1) / (
            np.linalg.norm(incoming, axis=-1) * np.linalg.norm(outgoing, axis=-1)
        )
        magnitude = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        incoming_perp = np.stack([-incoming[..., 1], incoming[..., 0]], axis=-1)
        sign = -np.sign(np.sum(incoming_perp * outgoing, axis=-1))

        rows.append(magnitude * sign)
        names.append(f"segment_angle_{node_names[node]}")

    return np.stack(rows, axis=0).astype(np.float32), names


def _chain_ordered_angle_triples(edges: list[tuple[int, int]], n_nodes: int) -> list[tuple[int, int, int]]:
    """(prev, node, next) for every degree-2 node, chains first (in walk order), then
    any node no chain reaches - those keep the original node-index order/neighbors
    instead, since there's no chain walk direction to inherit."""
    chains = discover_chains(edges, n_nodes)

    triples = [
        (prev_node, node, next_node)
        for chain in chains
        for prev_node, node, next_node in zip(chain, chain[1:], chain[2:])
    ]
    covered_nodes = {node for _prev, node, _next in triples}

    adjacency = _node_adjacency(edges, n_nodes)
    for node in range(n_nodes):
        if len(adjacency[node]) == 2 and node not in covered_nodes:
            prev_node, next_node = adjacency[node]
            triples.append((prev_node, node, next_node))

    return triples


def compute_speeds(ds: xr.Dataset, individual: str) -> tuple[np.ndarray, list[str]]:
    """Frame-to-frame displacement magnitude of the centroid and of each individual node."""
    pos = ds["position"].sel(individuals=individual).values  # (time, keypoints, 2)
    node_names = list(ds.coords["keypoints"].values)

    rows: list[np.ndarray] = []
    names: list[str] = []

    centroid = np.nanmean(pos, axis=1)  # (time, 2)
    rows.append(_speed(centroid))
    names.append("speed_centroid")

    for k, name in enumerate(node_names):
        rows.append(_speed(pos[:, k]))
        names.append(f"speed_{name}")

    return np.stack(rows, axis=0).astype(np.float32), names


FeatureComputer = Callable[[xr.Dataset, str], tuple[np.ndarray, list[str]]]

# Display order matches computation order. To add a new feature kind: write a
# `(ds, individual) -> (features, names)` function above and append one entry
# here - nothing else in this module needs to change.
FEATURE_GROUPS: list[tuple[str, FeatureComputer]] = [
    ("Distances", compute_distances),
    ("Angles", compute_angles),
    ("Speed", compute_speeds),
]


@dataclass
class FeatureGroup:
    name: str
    features: np.ndarray  # (n_features_in_group, n_frames) float32
    names: list[str]


def compute_feature_groups(ds: xr.Dataset, individual: str) -> list[FeatureGroup]:
    """Every registered feature group, computed and kept separate (e.g. for labeled display)."""
    return [FeatureGroup(name, *compute(ds, individual)) for name, compute in FEATURE_GROUPS]


def flatten_feature_groups(groups: list[FeatureGroup]) -> tuple[np.ndarray, list[str]]:
    """Concatenate feature groups back into the flat (features, names) shape training/filtering expect."""
    features = np.concatenate([g.features for g in groups], axis=0)
    names = [name for g in groups for name in g.names]
    return features, names


def compute_features(ds: xr.Dataset, individual: str) -> tuple[np.ndarray, list[str]]:
    """Return (n_features, n_frames) float32 array and matching feature names, all groups concatenated."""
    return flatten_feature_groups(compute_feature_groups(ds, individual))
