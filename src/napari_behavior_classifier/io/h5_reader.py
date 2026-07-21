"""Read SLEAP analysis .h5 files into a labeled xarray.Dataset.

SLEAP's "analysis" HDF5 export already stores per-frame bodypart tracks as a
flat (individuals, space, keypoints, time) ndarray plus skeleton metadata, so
loading it is a handful of vectorized h5py reads - no per-frame Instance
object reconstruction like parsing a `.slp` predictions file requires. That
makes it both faster to load and free of the intermittent native-code
crashes occasionally seen loading large `.slp` files via sleap_io.

Dataset layout follows the same (time, individuals, keypoints, space)
convention as the `movement` package, so downstream code and any future
interop with that ecosystem stay compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

ANALYSIS_SUFFIX = ".predictions.analysis.h5"


@dataclass(frozen=True)
class Skeleton:
    """A skeleton's node names + edges - the part of a dataset that must be identical
    across every file pooled into one session, since feature row order (kinematics'
    `discover_chains`-based ordering) depends on both staying fixed. Two files from the
    same SLEAP project always produce an equal `Skeleton`; two files from *different*
    projects (or a differently-edited skeleton) generally won't.
    """

    node_names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]


def extract_skeleton(ds: xr.Dataset) -> Skeleton:
    return Skeleton(
        node_names=tuple(str(n) for n in ds.coords["keypoints"].values),
        edges=tuple(tuple(edge) for edge in ds.attrs["skeleton_edges"]),
    )


def check_consistent_skeleton(reference: Skeleton, candidate: Skeleton, *, context: str = "") -> None:
    """Raise if `candidate` isn't the exact same skeleton as `reference`.

    Pooling frames across files (training/prediction, or just labeling the feature-map
    rows) silently assumes every open file shares one skeleton definition - a mismatch
    here would misalign feature rows across files with no other visible symptom, so
    this is checked eagerly (at file-open time) rather than left to be discovered as a
    confusing downstream training/prediction result.
    """
    if candidate != reference:
        where = f" ({context})" if context else ""
        raise ValueError(
            f"skeleton mismatch{where}: expected {len(reference.node_names)} nodes "
            f"{reference.node_names}, got {len(candidate.node_names)} nodes {candidate.node_names}. "
            "Files opened together must come from the same SLEAP project/skeleton."
        )


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _resolve_video_path(h5_path: Path, stored_video_path: str) -> Path:
    """Find the source video next to the .h5 file.

    The `video_path` stored inside the file records wherever SLEAP originally
    ran (often a different machine's path entirely), so it's unusable as-is.
    The video is expected to sit alongside the analysis file instead; when
    the file follows SLEAP's own `<video>.predictions.analysis.h5` naming,
    stripping that suffix recovers the video's own filename exactly.
    """
    name = h5_path.name
    if name.endswith(ANALYSIS_SUFFIX):
        candidate = h5_path.with_name(name[: -len(ANALYSIS_SUFFIX)])
    else:
        candidate = h5_path.with_name(Path(stored_video_path).name)
    if not candidate.exists():
        raise FileNotFoundError(
            f"could not find source video for {h5_path!s} - expected it at {candidate!s}"
        )
    return candidate


def load_tracks(h5_path: str | Path) -> xr.Dataset:
    """Load a SLEAP analysis .h5 file straight into a (time, individuals, keypoints, space) Dataset."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        tracks = f["tracks"][:]  # (individuals, space, keypoints, time)
        point_scores = f["point_scores"][:]  # (individuals, keypoints, time)
        node_names = [_decode(n) for n in f["node_names"][:]]
        edges = [tuple(edge) for edge in f["edge_inds"][:].tolist()]
        individual_names = [_decode(n) for n in f["track_names"][:]]
        stored_video_path = _decode(f["video_path"][()])

    position = np.transpose(tracks, (3, 0, 2, 1)).astype(np.float32)
    confidence = np.transpose(point_scores, (2, 0, 1)).astype(np.float32)

    video_path = _resolve_video_path(h5_path, stored_video_path)

    return xr.Dataset(
        data_vars=dict(
            position=(("time", "individuals", "keypoints", "space"), position),
            confidence=(("time", "individuals", "keypoints"), confidence),
        ),
        coords=dict(
            time=np.arange(position.shape[0]),
            individuals=individual_names,
            keypoints=node_names,
            space=["x", "y"],
        ),
        attrs=dict(
            source_h5=str(h5_path),
            video_path=str(video_path),
            skeleton_edges=edges,
        ),
    )
