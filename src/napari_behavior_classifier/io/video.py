"""Lazy, frame-indexed video access for napari's Image layer.

Wraps sleap_io's Video (which already does on-demand frame decoding) and
squeezes its trailing singleton channel axis so the array looks like a
plain (T, H, W) stack to napari. This is the only place sleap_io is still
used now that pose data comes from the analysis .h5 (see `h5_reader.py`) -
here it's just opening the source video for frame reads, not parsing any
prediction file.

Benchmarked on a 216k-frame video (frame-navigation-style reads, see project
notes): opencv ~5ms/frame, pyav ~9ms/frame, sleap_io's own "FFMPEG" backend
(the default when opencv/av aren't installed) ~350ms/frame - 70x slower,
enough to make frame-by-frame scrubbing feel frozen. opencv is set as the
default plugin below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sleap_io as sio

sio.set_default_video_plugin("opencv")


def load_video_frames(video_path: str | Path) -> "LazyVideoFrames":
    """Open a video file for on-demand frame reads."""
    return LazyVideoFrames(sio.load_video(str(video_path)))


class LazyVideoFrames:
    def __init__(self, video) -> None:
        n_frames, height, width, n_channels = video.shape
        if n_channels != 1:
            raise ValueError(f"expected single-channel (grayscale) video, got {n_channels} channels")
        self._video = video
        self.shape = (n_frames, height, width)
        self.dtype = video[0].dtype
        self.ndim = 3

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        frame_key, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())

        if isinstance(frame_key, slice):
            indices = range(*frame_key.indices(len(self)))
            arr = np.stack([self._video[i][..., 0] for i in indices], axis=0)
        else:
            arr = self._video[frame_key][..., 0]

        return arr[rest] if rest else arr
