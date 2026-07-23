"""Ilastik-style "paint" matrix for picking which feature groups/scales train on.

Rows are feature group names (`kinematics.FEATURE_GROUPS`), columns are `RAW_SCALE`
plus one filter-bank sigma each. Click a cell to toggle it; drag across other cells
to paint them to that same new state in one stroke (matching ilastik's own feature
selection dialog). Clicking a row or column header toggles that whole row/column.
"""

from __future__ import annotations

from qtpy.QtCore import QPointF, QRect, QSize, Qt, Signal
from qtpy.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget

from ..features import filterbank, kinematics
from ..features.selection import RAW_SCALE, FeatureSelection, scale_label

_CELL_SIZE = 32
_ROW_LABEL_WIDTH = 170
_COL_LABEL_HEIGHT = 28
_GRID_COLOR = QColor("#808080")
_CELL_BG_COLOR = QColor(128, 128, 128, 30)
_CHECK_COLOR = QColor("#4363d8")
_CHECK_WIDTH = 2.5
_HEADER_BG_COLOR = QColor(128, 128, 128, 50)


class FeatureMatrixWidget(QWidget):
    """A row_labels x col_labels grid of paintable on/off cells."""

    selection_changed = Signal()

    def __init__(
        self,
        row_labels: list[str],
        col_labels: list[str],
        selected: set[tuple[int, int]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._row_labels = row_labels
        self._col_labels = col_labels
        self._selected = set(selected)
        self._paint_value: bool | None = None
        self._painted_this_stroke: set[tuple[int, int]] = set()
        self.setFixedSize(self._grid_size())

    def _grid_size(self) -> QSize:
        width = _ROW_LABEL_WIDTH + _CELL_SIZE * len(self._col_labels)
        height = _COL_LABEL_HEIGHT + _CELL_SIZE * len(self._row_labels)
        return QSize(width, height)

    def is_selected(self, row: int, col: int) -> bool:
        return (row, col) in self._selected

    def selected_cells(self) -> set[tuple[int, int]]:
        return set(self._selected)

    def _cell_rect(self, row: int, col: int) -> QRect:
        return QRect(
            _ROW_LABEL_WIDTH + col * _CELL_SIZE, _COL_LABEL_HEIGHT + row * _CELL_SIZE, _CELL_SIZE, _CELL_SIZE
        )

    def _cell_at(self, x: int, y: int) -> tuple[int, int] | None:
        if x < _ROW_LABEL_WIDTH or y < _COL_LABEL_HEIGHT:
            return None
        row, col = (y - _COL_LABEL_HEIGHT) // _CELL_SIZE, (x - _ROW_LABEL_WIDTH) // _CELL_SIZE
        if 0 <= row < len(self._row_labels) and 0 <= col < len(self._col_labels):
            return int(row), int(col)
        return None

    def _row_header_at(self, x: int, y: int) -> int | None:
        if x >= _ROW_LABEL_WIDTH or y < _COL_LABEL_HEIGHT:
            return None
        row = (y - _COL_LABEL_HEIGHT) // _CELL_SIZE
        return int(row) if 0 <= row < len(self._row_labels) else None

    def _col_header_at(self, x: int, y: int) -> int | None:
        if y >= _COL_LABEL_HEIGHT or x < _ROW_LABEL_WIDTH:
            return None
        col = (x - _ROW_LABEL_WIDTH) // _CELL_SIZE
        return int(col) if 0 <= col < len(self._col_labels) else None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        row_header = self._row_header_at(pos.x(), pos.y())
        if row_header is not None:
            self._toggle_line([(row_header, c) for c in range(len(self._col_labels))])
            return
        col_header = self._col_header_at(pos.x(), pos.y())
        if col_header is not None:
            self._toggle_line([(r, col_header) for r in range(len(self._row_labels))])
            return

        cell = self._cell_at(pos.x(), pos.y())
        if cell is None:
            return
        self._paint_value = not self.is_selected(*cell)
        self._painted_this_stroke = set()
        self._apply(*cell)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._paint_value is None:
            return
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        cell = self._cell_at(pos.x(), pos.y())
        if cell is None or cell in self._painted_this_stroke:
            return
        self._apply(*cell)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._paint_value = None
        self._painted_this_stroke.clear()

    def _apply(self, row: int, col: int) -> None:
        if self._paint_value:
            self._selected.add((row, col))
        else:
            self._selected.discard((row, col))
        self._painted_this_stroke.add((row, col))
        self.update()
        self.selection_changed.emit()

    def _toggle_line(self, cells: list[tuple[int, int]]) -> None:
        """A header click turns its whole row/column on if any cell in it is off,
        otherwise off - the usual "select all" checkbox-header convention."""
        turn_on = not all(self.is_selected(*c) for c in cells)
        for c in cells:
            if turn_on:
                self._selected.add(c)
            else:
                self._selected.discard(c)
        self.update()
        self.selection_changed.emit()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            painter.fillRect(QRect(0, 0, _ROW_LABEL_WIDTH, self.height()), _HEADER_BG_COLOR)
            painter.fillRect(QRect(0, 0, self.width(), _COL_LABEL_HEIGHT), _HEADER_BG_COLOR)

            for r, label in enumerate(self._row_labels):
                rect = QRect(0, _COL_LABEL_HEIGHT + r * _CELL_SIZE, _ROW_LABEL_WIDTH - 4, _CELL_SIZE)
                painter.drawText(rect.adjusted(6, 0, -4, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), label)

            for c, label in enumerate(self._col_labels):
                rect = QRect(_ROW_LABEL_WIDTH + c * _CELL_SIZE, 0, _CELL_SIZE, _COL_LABEL_HEIGHT)
                painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)

            for r in range(len(self._row_labels)):
                for c in range(len(self._col_labels)):
                    rect = self._cell_rect(r, c)
                    painter.fillRect(rect, _CELL_BG_COLOR)
                    painter.setPen(_GRID_COLOR)
                    painter.drawRect(rect)
                    if self.is_selected(r, c):
                        self._draw_check(painter, rect)
        finally:
            painter.end()

    @staticmethod
    def _draw_check(painter: QPainter, rect: QRect) -> None:
        """A small checkmark centered in the cell - the only mark that a cell is on
        (no full fill), so the grid reads as a checklist rather than a heatmap."""
        painter.save()
        pen = QPen(_CHECK_COLOR)
        pen.setWidthF(_CHECK_WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        # two-segment tick within the middle ~40% of the cell
        x, y, s = rect.x(), rect.y(), rect.width()
        low = QPointF(x + 0.34 * s, y + 0.52 * s)
        elbow = QPointF(x + 0.45 * s, y + 0.66 * s)
        high = QPointF(x + 0.68 * s, y + 0.34 * s)
        painter.drawPolyline([low, elbow, high])
        painter.restore()


class SelectFeaturesDialog(QDialog):
    """Wraps `FeatureMatrixWidget` with OK/Cancel; call `.exec()`, then read
    `.selection` if it returned `QDialog.DialogCode.Accepted`."""

    def __init__(
        self,
        parent: QWidget | None,
        initial: FeatureSelection,
        sigmas: tuple[float, ...] = filterbank.DEFAULT_SIGMAS,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select features")

        self._row_labels = [name for name, _ in kinematics.FEATURE_GROUPS]
        self._sigmas = sigmas
        self._col_keys = [RAW_SCALE] + [scale_label(s) for s in sigmas]
        col_labels = ["raw"] + [f"σ={s:g}" for s in sigmas]  # unicode sigma

        selected = {
            (r, c)
            for r, group_name in enumerate(self._row_labels)
            for c, key in enumerate(self._col_keys)
            if initial.is_enabled(group_name, key)
        }
        self._matrix = FeatureMatrixWidget(self._row_labels, col_labels, selected, self)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._matrix)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selection(self) -> FeatureSelection:
        enabled: dict[str, set[str]] = {name: set() for name in self._row_labels}
        for row, col in self._matrix.selected_cells():
            enabled[self._row_labels[row]].add(self._col_keys[col])
        return FeatureSelection(enabled=enabled)
