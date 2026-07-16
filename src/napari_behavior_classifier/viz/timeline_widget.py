"""Clickable timeline: annotation bar + prediction bar + feature heatmap, vs frame.

Docked below the movie (not the side panel) since it's a wide horizontal
strip meant to span the viewer, not fit in a narrow column.
"""

from __future__ import annotations

import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QImage, QMouseEvent, QPainter, QPixmap, QResizeEvent, QWheelEvent
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import timeline

ANNOTATION_BAR_HEIGHT = 14
PREDICTION_BAR_HEIGHT = 14
BAR_GAP = 2
HEATMAP_MARGIN = 6
BACKGROUND_COLOR = (30, 30, 30)
MIN_VISIBLE_FRAMES = 20
ZOOM_STEP = 1.25


class TimelineWidget(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.n_frames = 0
        self.view_start = 0
        self.view_end = 0
        self.predictions: np.ndarray | None = None
        self.annotations: np.ndarray | None = None
        self.features: np.ndarray | None = None
        self.class_colors: dict[str, str] = {}
        self.colormap_name = timeline.DEFAULT_COLORMAP
        self.heatmap_vmin: np.ndarray | None = None
        self.heatmap_vmax: np.ndarray | None = None
        self._cached_pixmap: QPixmap | None = None

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer_layout)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Feature map colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(timeline.CURATED_COLORMAPS)
        self.colormap_combo.setCurrentText(self.colormap_name)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        controls_row.addWidget(self.colormap_combo)

        self.reset_zoom_button = QPushButton("Reset zoom")
        self.reset_zoom_button.clicked.connect(self._on_reset_zoom_clicked)
        controls_row.addWidget(self.reset_zoom_button)

        self.recalc_contrast_button = QPushButton("Recalculate feature map contrast")
        self.recalc_contrast_button.clicked.connect(self._on_recalc_contrast_clicked)
        controls_row.addWidget(self.recalc_contrast_button)

        controls_row.addStretch()
        controls_row.addWidget(QLabel("Ctrl+scroll: zoom  |  Ctrl+drag: pan"))
        outer_layout.addLayout(controls_row)

        self.canvas = _TimelineCanvas(self)
        outer_layout.addWidget(self.canvas)

        self.setMinimumHeight(180)
        self.viewer.dims.events.current_step.connect(lambda event: self.canvas.update())

    def set_data(
        self,
        n_frames: int,
        predictions: np.ndarray | None,
        annotations: np.ndarray | None,
        features: np.ndarray | None,
        class_colors: dict[str, str],
    ) -> None:
        if n_frames != self.n_frames:
            # a genuinely different file - don't carry over a zoom/contrast setting sized for the old one
            self.view_start = 0
            self.view_end = n_frames
            self.heatmap_vmin = None
            self.heatmap_vmax = None
        self.n_frames = n_frames
        self.predictions = predictions
        self.annotations = annotations
        self.features = features
        self.class_colors = class_colors
        self.canvas.invalidate()

    def _on_colormap_changed(self, name: str) -> None:
        self.colormap_name = name
        self.canvas.invalidate()

    def _on_reset_zoom_clicked(self) -> None:
        self.view_start = 0
        self.view_end = self.n_frames
        self.canvas.invalidate()

    def _on_recalc_contrast_clicked(self) -> None:
        if self.features is None:
            return
        self.heatmap_vmin, self.heatmap_vmax = timeline.compute_percentile_range(
            self.features, self.view_start, self.view_end
        )
        self.canvas.invalidate()


class _TimelineCanvas(QWidget):
    """The actual painted/clickable surface; kept separate from the colormap combo row above it."""

    def __init__(self, owner: TimelineWidget) -> None:
        super().__init__()
        self.owner = owner
        self._cached_pixmap: QPixmap | None = None
        self._panning = False
        self._pan_start_x = 0.0
        self._pan_start_view = (0, 0)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def invalidate(self) -> None:
        self._cached_pixmap = None
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._cached_pixmap = None
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        if self._cached_pixmap is None:
            self._cached_pixmap = self._render()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cached_pixmap)

        view_start, view_end = self.owner.view_start, self.owner.view_end
        if self.owner.n_frames > 0 and view_end > view_start:
            frame = int(self.owner.viewer.dims.current_step[0])
            if view_start <= frame < view_end:
                x = int((frame - view_start) / (view_end - view_start) * self.width())
                painter.setPen(QColor("red"))
                painter.drawLine(x, 0, x, self.height())
        painter.end()

    def _render(self) -> QPixmap:
        width = max(self.width(), 1)
        height = max(self.height(), 1)

        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[:] = BACKGROUND_COLOR

        n_frames = self.owner.n_frames
        view_start, view_end = self.owner.view_start, self.owner.view_end
        features = self.owner.features
        class_colors = self.owner.class_colors

        if n_frames > 0 and view_end > view_start:
            annotation_strip = timeline.build_prediction_strip(
                self.owner.annotations, view_start, view_end, width, class_colors
            )
            rgb[:ANNOTATION_BAR_HEIGHT] = annotation_strip[None, :, :]

            pred_top = ANNOTATION_BAR_HEIGHT + BAR_GAP
            prediction_strip = timeline.build_prediction_strip(
                self.owner.predictions, view_start, view_end, width, class_colors
            )

            undetected_mask = None
            if features is not None and features.shape[0] > 0:
                undetected_mask = timeline.build_undetected_mask(features, view_start, view_end, width)
                prediction_strip = timeline.blank_columns(prediction_strip, undetected_mask, BACKGROUND_COLOR)
            rgb[pred_top : pred_top + PREDICTION_BAR_HEIGHT] = prediction_strip[None, :, :]

            heat_top = pred_top + PREDICTION_BAR_HEIGHT + HEATMAP_MARGIN
            heat_height = height - heat_top
            if features is not None and heat_height > 0 and features.shape[0] > 0:
                binned = timeline.build_feature_heatmap(
                    features, view_start, view_end, width, self.owner.heatmap_vmin, self.owner.heatmap_vmax
                )
                heat_rgb = timeline.colorize_heatmap(binned, self.owner.colormap_name)
                if undetected_mask is not None:
                    heat_rgb = timeline.blank_columns(heat_rgb, undetected_mask, BACKGROUND_COLOR)
                n_features = heat_rgb.shape[0]
                row_idx = (np.arange(heat_height) * n_features // heat_height).clip(0, n_features - 1)
                rgb[heat_top : heat_top + heat_height] = heat_rgb[row_idx]

        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image.copy())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._panning = True
            self._pan_start_x = event.position().x()
            self._pan_start_view = (self.owner.view_start, self.owner.view_end)
            event.accept()
            return
        self._seek_to_x(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning and (event.buttons() & Qt.MouseButton.LeftButton):
            self._update_pan(event.position().x())
            event.accept()
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._seek_to_x(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            self._panning = False
            event.accept()

    def _update_pan(self, current_x: float) -> None:
        owner = self.owner
        start_view_start, start_view_end = self._pan_start_view
        view_width = start_view_end - start_view_start
        delta_frames = (current_x - self._pan_start_x) / max(self.width(), 1) * view_width

        new_start = start_view_start - delta_frames
        new_start = max(0, min(new_start, owner.n_frames - view_width))

        owner.view_start = int(round(new_start))
        owner.view_end = int(round(new_start + view_width))
        self.invalidate()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            super().wheelEvent(event)
            return

        owner = self.owner
        if owner.n_frames == 0:
            event.accept()
            return

        view_start, view_end = owner.view_start, owner.view_end
        view_width = view_end - view_start
        cursor_frame = timeline.frame_for_x(event.position().x(), self.width(), view_start, view_end)

        zoom_in = event.angleDelta().y() > 0
        new_width = view_width / ZOOM_STEP if zoom_in else view_width * ZOOM_STEP
        new_width = max(MIN_VISIBLE_FRAMES, min(new_width, owner.n_frames))

        # keep the frame under the cursor stationary while the window resizes around it
        fraction = (cursor_frame - view_start) / view_width if view_width > 0 else 0.5
        new_start = cursor_frame - fraction * new_width
        new_start = max(0, min(new_start, owner.n_frames - new_width))

        owner.view_start = int(round(new_start))
        owner.view_end = int(round(new_start + new_width))
        self.invalidate()
        event.accept()

    def _seek_to_x(self, x: float) -> None:
        frame = timeline.frame_for_x(x, self.width(), self.owner.view_start, self.owner.view_end)
        self.owner.viewer.dims.set_current_step(0, frame)
