"""Clickable timeline: annotation bar + prediction bar + feature heatmap, vs frame.

Docked below the movie (not the side panel) since it's a wide horizontal
strip meant to span the viewer, not fit in a narrow column.
"""

from __future__ import annotations

import napari
import numpy as np
from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QResizeEvent, QWheelEvent
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from . import timeline

ANNOTATION_BAR_HEIGHT = 14
PREDICTION_BAR_HEIGHT = 14
BAR_GAP = 2
HEATMAP_MARGIN = 6
GROUP_GAP_PX = 6
LABEL_WIDTH = 170
LABEL_FONT_SIZE = 8
TEXT_LABEL_HEIGHT = 16
CONTROL_ROW_HEIGHT = 20
RESET_BUTTON_WIDTH = 44
# below this span height, a group's colormap/contrast controls are hidden rather
# than squeezed illegibly small
MIN_GROUP_CONTROL_HEIGHT = TEXT_LABEL_HEIGHT + CONTROL_ROW_HEIGHT + 4
BACKGROUND_COLOR = (30, 30, 30)
MIN_VISIBLE_FRAMES = 20
ZOOM_STEP = 1.25
# Current-frame marker: a black outline behind a white core reads against light,
# dark, and colored regions alike (unlike a single solid color, which disappears
# against a colormap using that same hue) - the same halo trick as subtitle text.
CURRENT_FRAME_OUTLINE_WIDTH = 4
CURRENT_FRAME_LINE_WIDTH = 2
# translucent so the annotation/prediction bars and heatmap stay legible underneath
SELECTION_FILL_COLOR = (255, 255, 0, 60)

# Distances/speed are plain sequential magnitudes; angles vary around a meaningful
# baseline ("straight") rather than from a true zero, so a diverging colormap reads
# better there by default. Any other/future group falls back to DEFAULT_COLORMAP.
_DEFAULT_GROUP_COLORMAP = {
    "Distances": "viridis",
    "Angles": timeline.DEFAULT_DIVERGING_COLORMAP,
    "Angular velocity": "PiYG",  # diverging like Angles, but visually distinct from it
    "Speed": "plasma",
}


class TimelineWidget(QWidget):
    # emitted whenever `selection` is set or cleared, so listeners (the side-panel
    # label display) can stay in sync without polling every mouse move
    selection_changed = Signal()

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.n_frames = 0
        self.view_start = 0
        self.view_end = 0
        self.predictions: np.ndarray | None = None
        self.annotations: np.ndarray | None = None
        self.features: np.ndarray | None = None
        self.feature_group_sizes: list[tuple[str, int]] | None = None
        self.class_colors: dict[str, str] = {}
        # (lo, hi) inclusive frame range picked via Shift+drag on the timeline, for
        # one-shot bout labeling; None when nothing is selected
        self.selection: tuple[int, int] | None = None
        # all three keyed by feature-group name
        self.group_colormap: dict[str, str] = {}
        self.group_vmin: dict[str, np.ndarray | None] = {}
        self.group_vmax: dict[str, np.ndarray | None] = {}

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer_layout)

        controls_row = QHBoxLayout()
        self.reset_zoom_button = QPushButton("Reset zoom")
        self.reset_zoom_button.clicked.connect(self._on_reset_zoom_clicked)
        controls_row.addWidget(self.reset_zoom_button)

        controls_row.addStretch()
        controls_row.addWidget(QLabel("Ctrl+scroll: zoom  |  Ctrl+drag: pan  |  Shift+drag: select range"))
        # stretch 0: the controls row keeps its natural (small) height even as the dock grows
        outer_layout.addLayout(controls_row, 0)

        self.canvas = _TimelineCanvas(self)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # stretch 1: the feature map canvas absorbs all extra space when the dock is resized
        outer_layout.addWidget(self.canvas, 1)

        self.setMinimumHeight(320)
        self.viewer.dims.events.current_step.connect(lambda event: self.canvas.update())

    def set_data(
        self,
        n_frames: int,
        predictions: np.ndarray | None,
        annotations: np.ndarray | None,
        features: np.ndarray | None,
        class_colors: dict[str, str],
        feature_group_sizes: list[tuple[str, int]] | None = None,
    ) -> None:
        if n_frames != self.n_frames:
            # a genuinely different file - don't carry over a zoom/contrast setting sized
            # for the old one. Colormap *choice* is left alone: it's a property of what
            # kind of feature a group holds, not of which file is currently open.
            self.view_start = 0
            self.view_end = n_frames
            self.group_vmin = {}
            self.group_vmax = {}
            self.selection = None
        for name, _count in feature_group_sizes or []:
            self.group_colormap.setdefault(name, _DEFAULT_GROUP_COLORMAP.get(name, timeline.DEFAULT_COLORMAP))

        self.n_frames = n_frames
        self.predictions = predictions
        self.annotations = annotations
        self.features = features
        self.feature_group_sizes = feature_group_sizes
        self.class_colors = class_colors
        self.canvas.sync_group_controls()
        self.canvas.invalidate()

    def group_feature_slice(self, group_name: str) -> np.ndarray | None:
        """This group's rows within `self.features`, or None if not currently known."""
        if self.features is None or not self.feature_group_sizes:
            return None
        offset = 0
        for name, count in self.feature_group_sizes:
            if name == group_name:
                return self.features[offset : offset + count]
            offset += count
        return None

    def set_selection(self, start_frame: int, end_frame: int) -> None:
        if self.n_frames == 0:
            return
        lo, hi = sorted((start_frame, end_frame))
        lo = max(0, min(lo, self.n_frames - 1))
        hi = max(0, min(hi, self.n_frames - 1))
        self.selection = (lo, hi)
        self.selection_changed.emit()
        self.canvas.update()

    def clear_selection(self) -> None:
        if self.selection is not None:
            self.selection = None
            self.selection_changed.emit()
            self.canvas.update()

    def _on_reset_zoom_clicked(self) -> None:
        self.view_start = 0
        self.view_end = self.n_frames
        self.canvas.invalidate()

    def _on_group_colormap_changed(self, group_name: str, colormap_name: str) -> None:
        self.group_colormap[group_name] = colormap_name
        self.canvas.invalidate()

    def _on_group_reset_contrast_clicked(self, group_name: str) -> None:
        group_features = self.group_feature_slice(group_name)
        if group_features is None:
            return
        self.group_vmin[group_name], self.group_vmax[group_name] = timeline.compute_percentile_range(
            group_features, self.view_start, self.view_end
        )
        self.canvas.invalidate()


class _TimelineCanvas(QWidget):
    """The actual painted/clickable surface; kept separate from the reset-zoom row above it.

    Per-group colormap/contrast controls are real child widgets (QComboBox +
    QPushButton) positioned over the left label margin - everything else (bars,
    heatmap, text labels) is painted directly into one numpy-built pixmap for speed.
    """

    def __init__(self, owner: TimelineWidget) -> None:
        super().__init__()
        self.owner = owner
        self._cached_pixmap: QPixmap | None = None
        self._panning = False
        self._pan_start_x = 0.0
        self._pan_start_view = (0, 0)
        self._selecting = False
        self._selection_anchor_frame = 0
        self._group_controls: dict[str, tuple[QComboBox, QPushButton]] = {}
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def invalidate(self) -> None:
        self._cached_pixmap = None
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._cached_pixmap = None
        self._reposition_group_controls()
        super().resizeEvent(event)

    # -- per-group colormap/contrast controls --

    def sync_group_controls(self) -> None:
        """Create/remove combo+button pairs to match the current set of feature groups."""
        owner = self.owner
        current_names = [name for name, _count in owner.feature_group_sizes or []]
        current_set = set(current_names)

        for stale_name in set(self._group_controls) - current_set:
            combo, button = self._group_controls.pop(stale_name)
            combo.deleteLater()
            button.deleteLater()

        for name in current_set - set(self._group_controls):
            combo = QComboBox(self)
            combo.addItems(timeline.CURATED_COLORMAPS)
            combo.setCurrentText(owner.group_colormap.get(name, timeline.DEFAULT_COLORMAP))
            combo.currentTextChanged.connect(lambda text, n=name: owner._on_group_colormap_changed(n, text))
            button = QPushButton("Reset", self)
            button.setToolTip(f"Recalculate {name} contrast to the currently visible range")
            button.clicked.connect(lambda checked=False, n=name: owner._on_group_reset_contrast_clicked(n))
            self._group_controls[name] = (combo, button)

        self._reposition_group_controls()

    def _reposition_group_controls(self) -> None:
        for name, y0, y1 in self._group_spans():
            combo, button = self._group_controls.get(name, (None, None))
            if combo is None:
                continue
            span_height = y1 - y0
            visible = span_height >= MIN_GROUP_CONTROL_HEIGHT
            combo.setVisible(visible)
            button.setVisible(visible)
            if not visible:
                continue
            controls_y = y0 + TEXT_LABEL_HEIGHT
            combo_width = LABEL_WIDTH - 8 - RESET_BUTTON_WIDTH - 2
            combo.setGeometry(4, controls_y, combo_width, CONTROL_ROW_HEIGHT)
            button.setGeometry(4 + combo_width + 2, controls_y, RESET_BUTTON_WIDTH, CONTROL_ROW_HEIGHT)

    def _heat_region(self) -> tuple[int, int]:
        """(heat_top, heat_height) - shared by rendering, labeling, and control layout."""
        height = max(self.height(), 1)
        pred_top = ANNOTATION_BAR_HEIGHT + BAR_GAP
        heat_top = pred_top + PREDICTION_BAR_HEIGHT + HEATMAP_MARGIN
        return heat_top, max(height - heat_top, 0)

    def _group_spans(self) -> list[tuple[str, int, int]]:
        heat_top, heat_height = self._heat_region()
        group_sizes = self.owner.feature_group_sizes or []
        return timeline.group_pixel_spans(group_sizes, heat_top, heat_height, GROUP_GAP_PX)

    # -- painting --

    def paintEvent(self, event) -> None:
        if self._cached_pixmap is None:
            self._cached_pixmap = self._render()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cached_pixmap)

        view_start, view_end = self.owner.view_start, self.owner.view_end
        data_width = max(self.width() - LABEL_WIDTH, 1)
        if self.owner.n_frames > 0 and view_end > view_start and self.owner.selection is not None:
            sel_lo, sel_hi = self.owner.selection
            if sel_hi >= view_start and sel_lo < view_end:
                view_width = view_end - view_start
                x0 = LABEL_WIDTH + (max(sel_lo, view_start) - view_start) / view_width * data_width
                x1 = LABEL_WIDTH + (min(sel_hi, view_end - 1) + 1 - view_start) / view_width * data_width
                painter.fillRect(QRectF(x0, 0, x1 - x0, self.height()), QColor(*SELECTION_FILL_COLOR))

        if self.owner.n_frames > 0 and view_end > view_start:
            frame = int(self.owner.viewer.dims.current_step[0])
            if view_start <= frame < view_end:
                # +0.5 centers the marker within the frame's own pixel bin, not its left edge
                fraction = (frame - view_start + 0.5) / (view_end - view_start)
                x = int(round(LABEL_WIDTH + fraction * data_width))
                painter.setPen(QPen(QColor("black"), CURRENT_FRAME_OUTLINE_WIDTH))
                painter.drawLine(x, 0, x, self.height())
                painter.setPen(QPen(QColor("white"), CURRENT_FRAME_LINE_WIDTH))
                painter.drawLine(x, 0, x, self.height())

        self._draw_row_labels(painter)
        painter.end()

    def _draw_row_labels(self, painter: QPainter) -> None:
        """Left-margin labels for the annotation/prediction bars and each feature group."""
        pred_top = ANNOTATION_BAR_HEIGHT + BAR_GAP

        font = painter.font()
        font.setPointSize(LABEL_FONT_SIZE)
        painter.setFont(font)
        painter.setPen(QColor("white"))

        self._draw_label(painter, "Annotations", 0, ANNOTATION_BAR_HEIGHT)
        self._draw_label(painter, "Predictions", pred_top, pred_top + PREDICTION_BAR_HEIGHT)

        for name, y0, y1 in self._group_spans():
            # leave room below the text for the colormap/contrast controls, if they fit
            text_bottom = y0 + TEXT_LABEL_HEIGHT if (y1 - y0) >= MIN_GROUP_CONTROL_HEIGHT else y1
            self._draw_label(painter, name, y0, text_bottom)

    @staticmethod
    def _draw_label(painter: QPainter, text: str, y0: int, y1: int) -> None:
        if y1 <= y0:
            return
        rect = QRectF(4, y0, LABEL_WIDTH - 8, y1 - y0)
        painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

    def _render(self) -> QPixmap:
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        data_width = max(width - LABEL_WIDTH, 1)

        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[:] = BACKGROUND_COLOR

        n_frames = self.owner.n_frames
        view_start, view_end = self.owner.view_start, self.owner.view_end
        features = self.owner.features
        class_colors = self.owner.class_colors

        if n_frames > 0 and view_end > view_start:
            annotation_strip = timeline.build_prediction_strip(
                self.owner.annotations, view_start, view_end, data_width, class_colors
            )
            rgb[:ANNOTATION_BAR_HEIGHT, LABEL_WIDTH:] = annotation_strip[None, :, :]

            pred_top = ANNOTATION_BAR_HEIGHT + BAR_GAP
            prediction_strip = timeline.build_prediction_strip(
                self.owner.predictions, view_start, view_end, data_width, class_colors
            )

            undetected_mask = None
            if features is not None and features.shape[0] > 0:
                undetected_mask = timeline.build_undetected_mask(features, view_start, view_end, data_width)
                prediction_strip = timeline.blank_columns(prediction_strip, undetected_mask, BACKGROUND_COLOR)
            rgb[pred_top : pred_top + PREDICTION_BAR_HEIGHT, LABEL_WIDTH:] = prediction_strip[None, :, :]

            heat_top, heat_height = self._heat_region()
            if features is not None and heat_height > 0 and features.shape[0] > 0:
                n_features = features.shape[0]
                group_sizes = self.owner.feature_group_sizes
                if not group_sizes or sum(n for _, n in group_sizes) != n_features:
                    group_sizes = [("", n_features)]

                spans = timeline.group_pixel_spans(group_sizes, heat_top, heat_height, GROUP_GAP_PX)
                row_start = 0
                for (_label, y0, y1), (group_name, count) in zip(spans, group_sizes):
                    group_h = y1 - y0
                    if group_h > 0 and count > 0:
                        group_features = features[row_start : row_start + count]
                        colormap_name = self.owner.group_colormap.get(group_name, timeline.DEFAULT_COLORMAP)
                        vmin = self.owner.group_vmin.get(group_name)
                        vmax = self.owner.group_vmax.get(group_name)
                        binned = timeline.build_feature_heatmap(
                            group_features, view_start, view_end, data_width, vmin, vmax
                        )
                        heat_rgb = timeline.colorize_heatmap(binned, colormap_name)
                        if undetected_mask is not None:
                            heat_rgb = timeline.blank_columns(heat_rgb, undetected_mask, BACKGROUND_COLOR)
                        row_idx = (np.arange(group_h) * count // group_h).clip(0, count - 1)
                        rgb[y0:y1, LABEL_WIDTH:] = heat_rgb[row_idx]
                    row_start += count

        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image.copy())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.position().x() < LABEL_WIDTH:
            return  # clicks on the row-label margin don't seek/pan/select
        if event.button() == Qt.MouseButton.LeftButton and (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._selecting = True
            self._selection_anchor_frame = self._frame_at_x(event.position().x())
            self.owner.set_selection(self._selection_anchor_frame, self._selection_anchor_frame)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._panning = True
            self._pan_start_x = event.position().x()
            self._pan_start_view = (self.owner.view_start, self.owner.view_end)
            event.accept()
            return
        self.owner.clear_selection()
        self._seek_to_x(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._selecting and (event.buttons() & Qt.MouseButton.LeftButton):
            self.owner.set_selection(self._selection_anchor_frame, self._frame_at_x(event.position().x()))
            event.accept()
            return
        if self._panning and (event.buttons() & Qt.MouseButton.LeftButton):
            self._update_pan(event.position().x())
            event.accept()
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._seek_to_x(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._selecting:
            self._selecting = False
            event.accept()
            return
        if self._panning:
            self._panning = False
            event.accept()

    def _frame_at_x(self, x: float) -> int:
        data_width = max(self.width() - LABEL_WIDTH, 1)
        return timeline.frame_for_x(x - LABEL_WIDTH, data_width, self.owner.view_start, self.owner.view_end)

    def _update_pan(self, current_x: float) -> None:
        owner = self.owner
        start_view_start, start_view_end = self._pan_start_view
        view_width = start_view_end - start_view_start
        data_width = max(self.width() - LABEL_WIDTH, 1)
        delta_frames = (current_x - self._pan_start_x) / data_width * view_width

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
        data_width = max(self.width() - LABEL_WIDTH, 1)
        cursor_x = max(event.position().x() - LABEL_WIDTH, 0)
        cursor_frame = timeline.frame_for_x(cursor_x, data_width, view_start, view_end)

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
        self.owner.viewer.dims.set_current_step(0, self._frame_at_x(x))
