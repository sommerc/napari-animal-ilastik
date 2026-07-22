"""Wire a loaded SLEAP project into a napari Viewer.

Parsing a .h5 file (load_project_data) is kept separate from building its
napari layers (build_layers) so a multi-file session can cache the parsed
data per file - the .h5 load is cheap, but the video open and layer wiring
are worth keeping around across frame changes - and cheaply tear down/rebuild
just the layers when switching which file is currently displayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import napari
import numpy as np
import xarray as xr

from ..io.h5_reader import load_tracks
from ..io.video import LazyVideoFrames, load_video_frames
from . import layers

DEFAULT_BOX_COLOR = "cyan"

# A manual annotation gets a thicker box outline than a model prediction, since
# both otherwise share the same per-class color palette and would be
# indistinguishable at a glance. The individual currently selected in the dropdown
# gets a further width bonus on top of either - real "bold", via the box outline
# itself, rather than text styling (napari's Shapes/Points TextManager has no
# font-weight field and its `size` is a single value for the whole layer, so per-item
# bold text isn't possible - but edge_width is a genuine per-item property).
ANNOTATION_EDGE_WIDTH = 4.0
PREDICTION_EDGE_WIDTH = 1.5
ACTIVE_BOX_EXTRA_WIDTH = 3.0

BOX_LABEL_FONT_SIZE = 10
BOX_LABEL_COLOR = "white"
BOX_LABEL_TRANSLATION = [-8, 0]

GetBoxStyle = Callable[[str, int], tuple[str, float]]
GetActiveIndividual = Callable[[], str]


@dataclass
class ProjectData:
    """Parsed .h5 contents: dataset + lazy video. Cheap to keep resident once loaded."""

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


def load_project_data(h5_path: str | Path) -> ProjectData:
    """Parse a SLEAP analysis .h5 file and open its source video."""
    ds = load_tracks(h5_path)
    video = load_video_frames(ds.attrs["video_path"])
    return ProjectData(ds=ds, video=video)


def build_layers(
    data: ProjectData,
    viewer: napari.Viewer,
    get_box_style: GetBoxStyle | None = None,
    get_active_individual: GetActiveIndividual | None = None,
) -> ProjectLayers:
    """Add fresh napari layers for already-parsed ProjectData. Cheap - safe to call on every file switch."""
    ds = data.ds
    get_box_style = get_box_style or (lambda individual, frame: (DEFAULT_BOX_COLOR, PREDICTION_EDGE_WIDTH))
    get_active_individual = get_active_individual or (lambda: "")

    image_layer = viewer.add_image(data.video, name=Path(ds.attrs["video_path"]).name, colormap="gray")

    points_layer = viewer.add_points(
        np.empty((0, 2), dtype=np.float32),
        name="bodyparts",
        size=3,
        face_color="white",
        border_width=0,
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
        edge_width=PREDICTION_EDGE_WIDTH,
        properties={"individual": np.array([], dtype=object)},
        text={
            "string": "{individual}",
            "size": BOX_LABEL_FONT_SIZE,
            "color": BOX_LABEL_COLOR,
            "anchor": "upper_left",
            "translation": BOX_LABEL_TRANSLATION,
        },
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
            bbox_layer.properties = {"individual": np.array([individual for individual, _ in boxes], dtype=object)}
            active_individual = get_active_individual()
            styles = [get_box_style(individual, frame) for individual, _ in boxes]
            bbox_layer.edge_color = [color for color, _ in styles]
            bbox_layer.edge_width = [
                width + (ACTIVE_BOX_EXTRA_WIDTH if individual == active_individual else 0)
                for (individual, _box), (_color, width) in zip(boxes, styles)
            ]

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
