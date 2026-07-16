"""Read SLEAP .slp prediction files into a labeled xarray.Dataset.

Dataset layout follows the same (time, individuals, keypoints, space)
convention as the `movement` package, so downstream code and any future
interop with that ecosystem stay compatible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sleap_io as sio
import xarray as xr


def load_labels(slp_path: str | Path, max_attempts: int = 5) -> sio.Labels:
    """Load a .slp file, keeping its attached video handle open for frame reads.

    sleap-io 0.9.1 has an intermittent, upstream read bug on large files: it
    occasionally segfaults or raises a spurious AttributeError/SystemError
    while parsing labeled frames (observed ~1 in 3 loads on a 216k-frame
    file, unrelated to file location or any code here). Retrying is a
    pragmatic workaround until that's fixed upstream.
    """
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            labels = sio.load_slp(str(slp_path))
            if labels.video is None or labels.video.shape is None:
                # Another shape this same upstream flakiness takes: load_slp returns
                # without raising, but the video backend silently failed to open.
                last_error = RuntimeError("loaded Labels has no usable video (video.shape is None)")
                continue
            return labels
        except Exception as e:  # noqa: BLE001 - retrying a known-flaky upstream call
            last_error = e
    raise RuntimeError(
        f"sleap_io.load_slp failed {max_attempts} times in a row for {slp_path!s} "
        "(see https://github.com/talmolab/sleap-io for known issues)"
    ) from last_error


def labels_to_tracks(labels: sio.Labels, source_slp: str | Path = "") -> xr.Dataset:
    """Convert an already-loaded sleap_io.Labels into a tracks Dataset."""
    data = labels.numpy(return_confidence=True)  # (time, individuals, keypoints, 3)
    position = data[..., :2].astype(np.float32)
    confidence = data[..., 2].astype(np.float32)

    skeleton = labels.skeleton
    node_names = [n.name for n in skeleton.nodes]
    node_index = {name: i for i, name in enumerate(node_names)}
    edges = [(node_index[e.source.name], node_index[e.destination.name]) for e in skeleton.edges]

    individual_names = [t.name for t in labels.tracks] or ["individual_0"]

    video = labels.video
    n_frames, height, width, n_channels = video.shape

    ds = xr.Dataset(
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
            source_slp=str(source_slp),
            video_path=str(video.filename),
            fps=float(getattr(video, "fps", 0.0) or 0.0),
            frame_shape=(height, width, n_channels),
            skeleton_edges=edges,
        ),
    )
    return ds


def load_tracks(slp_path: str | Path) -> xr.Dataset:
    """Load a .slp file straight into a (time, individuals, keypoints, space) Dataset."""
    slp_path = Path(slp_path)
    return labels_to_tracks(load_labels(slp_path), source_slp=slp_path)
