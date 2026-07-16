"""Wire a loaded SLEAP project into a napari Viewer.

Parsing a .slp file (load_project_data) is kept separate from building its
napari layers (build_layers) so a multi-file session can cache the parsed
data per file - expensive, and occasionally flaky upstream in sleap-io - and
cheaply tear down/rebuild just the layers when switching which file is
currently displayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import napari
import numpy as np
import xarray as xr

from ..io.slp_reader import labels_to_tracks, load_labels
from ..io.video import LazyVideoFrames
from . import layers

DEFAULT_BOX_COLOR = "cyan"

GetBoxColor = Callable[[str, int], str]


@dataclass
class ProjectData:
    """Parsed .slp contents: dataset + lazy video. Cheap to keep resident once loaded."""

    ds: xr.Dataset
    video: LazyVideoFrames


@dataclass
class ProjectLayers:
    """The napari layers currently displaying one ProjectData. Torn down on file switch."""

    image_layer: napari.layers.Image
    points_layer: napari.layers.Points
    skeleton_layer: napari.layers.Shapes
    bbox_layer: napari.layers.Shapes
    refresh: Callable[[], None]


def load_project_data(slp_path: str | Path) -> ProjectData:
    """Parse a .slp file - the expensive (and occasionally flaky-upstream) step."""
    slp_path = Path(slp_path)
    labels = load_labels(slp_path)
    ds = labels_to_tracks(labels, source_slp=slp_path)
    video = LazyVideoFrames(labels.video)
    return ProjectData(ds=ds, video=video)


def build_layers(
    data: ProjectData,
    viewer: napari.Viewer,
    get_box_color: GetBoxColor | None = None,
) -> ProjectLayers:
    """Add fresh napari layers for already-parsed ProjectData. Cheap - safe to call on every file switch."""
    ds = data.ds
    get_box_color = get_box_color or (lambda individual, frame: DEFAULT_BOX_COLOR)

    image_layer = viewer.add_image(data.video, name=Path(ds.attrs["video_path"]).name, colormap="gray")

    points_layer = viewer.add_points(
        np.empty((0, 2), dtype=np.float32),
        name="bodyparts",
        size=6,
        face_color="white",
    )
    skeleton_layer = viewer.add_shapes(
        [],
        shape_type="line",
        name="skeleton",
        edge_color="yellow",
        edge_width=2,
    )
    bbox_layer = viewer.add_shapes(
        [],
        shape_type="rectangle",
        name="bounding box",
        edge_color=DEFAULT_BOX_COLOR,
        face_color="transparent",
        edge_width=2,
    )

    def refresh(event=None) -> None:
        frame = int(viewer.dims.current_step[0])

        point_data, properties = layers.frame_points(ds, frame)
        points_layer.data = point_data
        points_layer.properties = properties

        skeleton_lines = layers.frame_skeleton_lines(ds, frame)
        skeleton_layer.data = skeleton_lines
        if skeleton_lines:
            skeleton_layer.shape_type = ["line"] * len(skeleton_lines)

        boxes = layers.frame_bboxes(ds, frame)
        bbox_layer.data = [box for _, box in boxes]
        if boxes:
            bbox_layer.shape_type = ["rectangle"] * len(boxes)
            bbox_layer.edge_color = [get_box_color(individual, frame) for individual, _ in boxes]

    viewer.dims.events.current_step.connect(refresh)
    refresh()

    return ProjectLayers(
        image_layer=image_layer,
        points_layer=points_layer,
        skeleton_layer=skeleton_layer,
        bbox_layer=bbox_layer,
        refresh=refresh,
    )


def remove_layers(viewer: napari.Viewer, project_layers: ProjectLayers) -> None:
    """Tear down one file's layers (and its frame-change subscription) when switching away from it."""
    viewer.dims.events.current_step.disconnect(project_layers.refresh)
    for layer in (
        project_layers.image_layer,
        project_layers.points_layer,
        project_layers.skeleton_layer,
        project_layers.bbox_layer,
    ):
        if layer in viewer.layers:
            viewer.layers.remove(layer)
