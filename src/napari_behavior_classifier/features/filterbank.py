"""Ilastik-style multi-scale temporal filter bank over kinematic feature channels.

Each filter runs along time, independently per feature channel - the direct
analogue of how ilastik's pixel filters run per color channel and never mix
channels together. A true 2D filter across the feature axis would blend
unrelated units (a distance with an angle), so "2D" here means the same
per-channel philosophy, just with time as the one spatial axis. Scales
(sigma) are in frames, not seconds, since fps varies by recording.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

DEFAULT_SIGMAS = (1.0, 3.0, 9.0, 27.0)

# the three per-scale transforms produced for every channel (see apply_filter_bank);
# named here so callers (e.g. the timeline "feature view" selector) can enumerate them
FILTER_KINDS = ("smooth", "rate", "variability")


def apply_single_filter(features: np.ndarray, kind: str, sigma: float) -> np.ndarray:
    """One filter-bank transform applied per channel along time, shape-preserving:
    (n_features, n_frames) -> (n_features, n_frames). Used to *view* the effect of a
    single filter (kind in FILTER_KINDS) at one scale, not to build a training matrix."""
    if kind not in FILTER_KINDS:
        raise ValueError(f"unknown filter kind {kind!r}; expected one of {FILTER_KINDS}")
    out = np.empty_like(features, dtype=np.float32)
    for i in range(features.shape[0]):
        filled = _fill_nans(features[i])
        if kind == "smooth":
            out[i] = gaussian_filter1d(filled, sigma=sigma)
        elif kind == "rate":
            out[i] = gaussian_filter1d(filled, sigma=sigma, order=1)
        else:  # "variability"
            out[i] = _local_std(filled, sigma)
    return out


def with_filter_bank(
    features: np.ndarray,
    feature_names: list[str],
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
) -> tuple[np.ndarray, list[str]]:
    """Concatenate raw features with their multi-scale filter responses."""
    filtered, filtered_names = apply_filter_bank(features, feature_names, sigmas)
    combined = np.concatenate([features, filtered], axis=0)
    return combined, feature_names + filtered_names


def apply_filter_bank(
    features: np.ndarray,
    feature_names: list[str],
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS,
) -> tuple[np.ndarray, list[str]]:
    """features: (n_features, n_frames) -> (n_features * len(sigmas) * 3, n_frames).

    Per channel and scale: a smoothed value (Gaussian), a rate of change
    (Gaussian derivative - captures oscillation during an active bout), and
    a local variability (Gaussian-windowed std - captures erratic/spasm-like
    signal that a plain smoothed value or rate would average away).
    """
    rows: list[np.ndarray] = []
    names: list[str] = []

    for i, name in enumerate(feature_names):
        filled = _fill_nans(features[i])

        for sigma in sigmas:
            rows.append(gaussian_filter1d(filled, sigma=sigma))
            names.append(f"{name}_smooth_s{sigma:g}")

            rows.append(gaussian_filter1d(filled, sigma=sigma, order=1))
            names.append(f"{name}_rate_s{sigma:g}")

            rows.append(_local_std(filled, sigma))
            names.append(f"{name}_variability_s{sigma:g}")

    return np.stack(rows, axis=0).astype(np.float32), names


def _fill_nans(channel: np.ndarray) -> np.ndarray:
    """Linearly interpolate gaps; holds the nearest valid value past the edges."""
    valid = ~np.isnan(channel)
    if not valid.any():
        return np.zeros_like(channel)
    idx = np.arange(len(channel))
    return np.interp(idx, idx[valid], channel[valid])


def _local_std(filled: np.ndarray, sigma: float) -> np.ndarray:
    mean = gaussian_filter1d(filled, sigma=sigma)
    mean_sq = gaussian_filter1d(filled**2, sigma=sigma)
    variance = np.clip(mean_sq - mean**2, 0.0, None)
    return np.sqrt(variance)
