"""Pure array-building functions for the timeline widget.

Kept free of Qt so the binning/normalization logic is testable without a
running application - the widget itself just paints whatever these return.

Every binning function takes a (view_start, view_end) frame range rather
than always the full [0, n_frames) - that's what makes time-axis zooming
possible: the widget just narrows the range it asks for, the bin edges
still divide evenly across the same pixel width.
"""

from __future__ import annotations

import numpy as np
from napari.utils import Colormap
from napari.utils.colormaps import AVAILABLE_COLORMAPS

DEFAULT_COLORMAP = "gray"
DEFAULT_DIVERGING_COLORMAP = "coolwarm"

# A hand-picked approximation of matplotlib's "coolwarm" diverging colormap
# (blue -> near-white -> red) - not pixel-identical (matplotlib isn't a
# dependency here), but the same shape. Diverging colormaps suit features
# that vary around a meaningful baseline rather than from a true zero - e.g.
# a segment angle deviating from "straight" - where a sequential map like
# gray/viridis hides which side of the baseline a value falls on.
_CUSTOM_COLORMAPS: dict[str, Colormap] = {
    "coolwarm": Colormap(
        [
            (0.230, 0.299, 0.754, 1.0),
            (0.552, 0.690, 0.988, 1.0),
            (0.865, 0.865, 0.865, 1.0),
            (0.957, 0.647, 0.511, 1.0),
            (0.706, 0.016, 0.150, 1.0),
        ],
        display_name="coolwarm",
    ),
}

# Sequential maps first, then our custom diverging one, then napari's own
# diverging ("PiYG") and cyclic ("twilight"/"twilight_shifted" - also a
# reasonable fit for angle-like data) maps.
CURATED_COLORMAPS = (
    [name for name in ("gray", "viridis", "magma", "inferno", "plasma", "turbo", "fire", "ice") if name in AVAILABLE_COLORMAPS]
    + list(_CUSTOM_COLORMAPS)
    + [name for name in ("PiYG", "twilight", "twilight_shifted") if name in AVAILABLE_COLORMAPS]
)


def get_colormap(name: str) -> Colormap:
    if name in _CUSTOM_COLORMAPS:
        return _CUSTOM_COLORMAPS[name]
    return AVAILABLE_COLORMAPS.get(name, AVAILABLE_COLORMAPS[DEFAULT_COLORMAP])


def _view_bin_edges(view_start: int, view_end: int, width: int, n_frames: int) -> np.ndarray | None:
    """Bin edges for `width` bins over [view_start, view_end), clamped to valid array indices."""
    if width <= 0 or view_end <= view_start or n_frames == 0:
        return None
    edges = np.linspace(view_start, view_end, width + 1)
    return np.clip(edges, 0, n_frames).astype(int)


def build_prediction_strip(
    predictions: np.ndarray | None,
    view_start: int,
    view_end: int,
    width: int,
    class_colors: dict[str, str],
    default_color: str = "#555555",
) -> np.ndarray:
    """(width, 3) uint8 RGB row: one majority-vote class color per pixel bin."""
    default_rgb = _hex_to_rgb(default_color)
    strip = np.tile(np.array(default_rgb, dtype=np.uint8), (width, 1))
    if predictions is None:
        return strip

    bin_edges = _view_bin_edges(view_start, view_end, width, len(predictions))
    if bin_edges is None:
        return strip

    for x in range(width):
        f0, f1 = bin_edges[x], max(bin_edges[x + 1], bin_edges[x] + 1)
        chunk = predictions[f0:f1]
        chunk = chunk[chunk != None]  # noqa: E711 - object array, no `is not None` broadcast
        if len(chunk) == 0:
            continue
        values, counts = np.unique(chunk, return_counts=True)
        cls = values[np.argmax(counts)]
        strip[x] = _hex_to_rgb(class_colors.get(cls, default_color))
    return strip


def build_feature_heatmap(
    features: np.ndarray,
    view_start: int,
    view_end: int,
    width: int,
    vmin: np.ndarray | None = None,
    vmax: np.ndarray | None = None,
) -> np.ndarray:
    """(n_features, width) float in [0, 1]: row-normalized, column-binned to the view range.

    vmin/vmax are per-feature-row normalization bounds - (n_features,) arrays.
    Defaulting to None (computed from the full signal's 2nd/98th percentile)
    keeps color meaning stable while merely zooming; passing explicit bounds
    (from `compute_percentile_range`) is how "recalculate contrast" rescales
    to just the currently visible range instead.
    """
    n_features, n_frames = features.shape
    if n_frames == 0:
        return np.zeros((n_features, width), dtype=np.float32)

    row_mean = np.nanmean(features, axis=1, keepdims=True)
    row_mean = np.nan_to_num(row_mean, nan=0.0)
    filled = np.where(np.isnan(features), row_mean, features)

    if vmin is None or vmax is None:
        lo = np.percentile(filled, 2, axis=1, keepdims=True)
        hi = np.percentile(filled, 98, axis=1, keepdims=True)
    else:
        lo = np.asarray(vmin).reshape(-1, 1)
        hi = np.asarray(vmax).reshape(-1, 1)
    span = np.where(hi > lo, hi - lo, 1.0)
    normed = np.clip((filled - lo) / span, 0.0, 1.0)

    binned = np.zeros((n_features, width), dtype=np.float32)
    bin_edges = _view_bin_edges(view_start, view_end, width, n_frames)
    if bin_edges is None:
        return binned

    for x in range(width):
        f0, f1 = bin_edges[x], max(bin_edges[x + 1], bin_edges[x] + 1)
        binned[:, x] = normed[:, f0:f1].mean(axis=1)
    return binned


def compute_percentile_range(
    features: np.ndarray, view_start: int, view_end: int, low: float = 2.0, high: float = 98.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature-row (low, high) percentile over [view_start, view_end) - for "recalculate contrast"."""
    n_features, n_frames = features.shape
    f0 = max(int(view_start), 0)
    f1 = min(int(view_end), n_frames)
    if f1 <= f0:
        f0, f1 = 0, n_frames
    chunk = features[:, f0:f1]
    lo = np.nanpercentile(chunk, low, axis=1)
    hi = np.nanpercentile(chunk, high, axis=1)
    return lo, hi


def build_undetected_mask(features: np.ndarray, view_start: int, view_end: int, width: int, threshold: float = 0.5) -> np.ndarray:
    """(width,) bool: True where most frames in that pixel bin have no detection at all.

    A frame with zero detected keypoints has NaN in every raw feature channel
    (every distance/angle/speed derived from an all-missing position is
    itself NaN), so "all features NaN" is a reliable per-frame no-detection
    signal without needing the raw position array here too.
    """
    n_features, n_frames = features.shape
    mask = np.zeros(width, dtype=bool)
    if n_features == 0:
        return mask

    bin_edges = _view_bin_edges(view_start, view_end, width, n_frames)
    if bin_edges is None:
        return mask

    undetected = np.all(np.isnan(features), axis=0)  # (n_frames,)
    for x in range(width):
        f0, f1 = bin_edges[x], max(bin_edges[x + 1], bin_edges[x] + 1)
        mask[x] = undetected[f0:f1].mean() > threshold
    return mask


def colorize_heatmap(binned: np.ndarray, colormap_name: str = DEFAULT_COLORMAP) -> np.ndarray:
    """binned: (n_features, width) in [0, 1] -> (n_features, width, 3) uint8 RGB."""
    cmap = get_colormap(colormap_name)
    rgba = cmap.map(binned.ravel())  # (n_features * width, 4) float in [0, 1]
    rgb = (rgba[:, :3] * 255).astype(np.uint8)
    return rgb.reshape(binned.shape[0], binned.shape[1], 3)


def blank_columns(rgb: np.ndarray, column_mask: np.ndarray, background_color: tuple[int, int, int]) -> np.ndarray:
    """Overwrite masked columns with background_color across every row. rgb: (H, width, 3) or (width, 3)."""
    rgb = rgb.copy()
    rgb[..., column_mask, :] = background_color
    return rgb


def group_pixel_spans(
    group_sizes: list[tuple[str, int]], top: int, height: int, gap_px: int
) -> list[tuple[str, int, int]]:
    """Split a `height`-pixel vertical band into one sub-span per (name, n_features) group,
    proportional to each group's row count, with `gap_px` of background between groups.

    Cumulative rounding (rather than rounding each group's height independently)
    keeps spans gapless/driftless: the last group's bottom always lands exactly
    on `top + height` minus the trailing gaps.
    """
    if not group_sizes or height <= 0:
        return []

    total_features = sum(n for _, n in group_sizes)
    if total_features <= 0:
        return []

    total_gap = gap_px * (len(group_sizes) - 1)
    available = max(height - total_gap, 0)

    cumulative = 0
    boundaries = [0]
    for _, n in group_sizes:
        cumulative += n
        boundaries.append(round(cumulative / total_features * available))

    spans = []
    y = top
    for i, (name, _n) in enumerate(group_sizes):
        span_height = boundaries[i + 1] - boundaries[i]
        spans.append((name, y, y + span_height))
        y += span_height + gap_px
    return spans


def frame_for_x(x: float, width: int, view_start: int, view_end: int) -> int:
    if width <= 0 or view_end <= view_start:
        return view_start
    frame = view_start + x / width * (view_end - view_start)
    return int(np.clip(frame, view_start, view_end - 1))


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
