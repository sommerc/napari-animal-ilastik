"""Main napari dock widget: open one or more SLEAP projects, annotate across all of them."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .annotation.store import LabelStore
from .classifier import train as train_module
from .features import filterbank, kinematics
from .io.h5_reader import Skeleton, check_consistent_skeleton, extract_skeleton
from .session import load_session, save_session
from .viz.timeline_widget import TimelineWidget
from .viz.viewer import (
    ANNOTATION_EDGE_WIDTH,
    DEFAULT_BOX_COLOR,
    PREDICTION_EDGE_WIDTH,
    ProjectData,
    ProjectLayers,
    build_layers,
    load_project_data,
    remove_layers,
)

_CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
]


@dataclass
class OpenFile:
    h5_path: str
    data: ProjectData
    layers: ProjectLayers | None = None


class _TrainingSummaryDialog(QDialog):
    """Read-only, copyable report shown right after training: a header plus a single
    table combining per-class annotation counts, out-of-bag precision/recall/F1, and
    the out-of-bag confusion matrix (one "-> {class}" column per predicted class)."""

    def __init__(
        self,
        parent: QWidget,
        individuals: list[str],
        file_names: list[str],
        class_counts: Counter,
        n_features: int,
        report: train_module.OOBReport | None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training summary")

        layout = QVBoxLayout()
        self.setLayout(layout)

        n_frames = sum(class_counts.values())
        accuracy_line = (
            f"Out-of-bag accuracy: {report.accuracy:.1%}"
            if report is not None
            else "Out-of-bag accuracy: n/a (not enough labeled frames per class)"
        )
        header_lines = [
            f"Individuals (pooled): {', '.join(individuals)}",
            f"Files: {', '.join(file_names)}",
            f"Labeled frames: {n_frames}  |  Features: {n_features}",
            accuracy_line,
        ]
        header_label = QLabel("\n".join(header_lines))
        layout.addWidget(header_label)

        classes = report.classes if report is not None else sorted(class_counts)
        headers = ["Class", "Annotated", "Precision", "Recall", "F1"] + [f"-> {c}" for c in classes]
        table = QTableWidget(len(classes), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)

        for row, cls in enumerate(classes):
            values = [cls, str(class_counts.get(cls, 0))]
            if report is not None:
                values += [f"{report.precision[row]:.2f}", f"{report.recall[row]:.2f}", f"{report.f1[row]:.2f}"]
                values += [str(n) for n in report.confusion[row]]
            else:
                values += ["n/a"] * (len(headers) - 2)
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, col, item)

        table.resizeColumnsToContents()
        table.setMinimumHeight(120)
        layout.addWidget(table)

        button_row = QHBoxLayout()
        copy_button = QPushButton("Copy to clipboard")
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(self._as_text(header_lines, headers, table)))
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_row.addWidget(copy_button)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    @staticmethod
    def _as_text(header_lines: list[str], headers: list[str], table: QTableWidget) -> str:
        """Tab-separated so the table half pastes cleanly into a spreadsheet."""
        rows = ["\t".join(headers)]
        for row in range(table.rowCount()):
            rows.append("\t".join(table.item(row, col).text() for col in range(table.columnCount())))
        return "\n".join(header_lines) + "\n\n" + "\n".join(rows)


class BehaviorClassifierWidget(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.open_files: dict[str, OpenFile] = {}
        self.active_h5_path: str | None = None
        self._last_frame: dict[str, int] = {}  # keyed by h5_path, restored when switching back
        self._reference_skeleton: Skeleton | None = None  # set from the first opened file
        self.store = LabelStore()
        self._shortcuts: list[QShortcut] = []
        # fixed (not rebuilt by _rebind_hotkeys, unlike the 1-9 class hotkeys): "0" always
        # clears whatever's under the current selection/frame, regardless of class list
        self._clear_shortcut = QShortcut(QKeySequence("0"), self.viewer.window._qt_window)
        self._clear_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._clear_shortcut.activated.connect(self._on_clear_label)
        self._class_colors: dict[str, str] = {}
        self._raw_features_cache: dict[tuple[str, str], list[kinematics.FeatureGroup]] = {}
        self._combined_features_cache: dict[tuple[str, str], tuple[np.ndarray, list[str]]] = {}
        # one shared model: classes are monadic (a class label means the same thing
        # regardless of which tracked animal it's attached to), so training pools
        # labeled frames across every individual and every open file into one pipeline
        self._monadic_pipeline: object | None = None
        self._predictions: dict[tuple[str, str], np.ndarray] = {}  # keyed by (source_file, individual)

        self.timeline_widget = TimelineWidget(viewer)
        self.timeline_widget.selection_changed.connect(self._refresh_current_label)
        self.viewer.window.add_dock_widget(self.timeline_widget, area="bottom", name="Timeline")

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.open_button = QPushButton("Open SLEAP analysis (.h5)...")
        self.open_button.clicked.connect(self._on_open_clicked)
        layout.addWidget(self.open_button)

        layout.addWidget(QLabel("Open files (click to switch):"))
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self._on_file_selection_changed)
        layout.addWidget(self.file_list)

        self.remove_file_button = QPushButton("Remove selected file")
        self.remove_file_button.clicked.connect(self._on_remove_file_clicked)
        layout.addWidget(self.remove_file_button)

        session_io_row = QHBoxLayout()
        self.save_session_button = QPushButton("Save session...")
        self.save_session_button.clicked.connect(self._on_save_session_clicked)
        self.load_session_button = QPushButton("Load session...")
        self.load_session_button.clicked.connect(self._on_load_session_clicked)
        session_io_row.addWidget(self.save_session_button)
        session_io_row.addWidget(self.load_session_button)
        layout.addLayout(session_io_row)

        indiv_row = QHBoxLayout()
        indiv_row.addWidget(QLabel("Individual:"))
        self.individual_combo = QComboBox()
        self.individual_combo.currentIndexChanged.connect(self._on_individual_changed)
        indiv_row.addWidget(self.individual_combo)
        layout.addLayout(indiv_row)

        layout.addWidget(QLabel(
            "Classes (1-9: label current frame, or Shift+drag a range on the timeline first; 0: clear):"
        ))
        self.class_list = QListWidget()
        layout.addWidget(self.class_list)

        add_row = QHBoxLayout()
        self.new_class_edit = QLineEdit()
        self.new_class_edit.setPlaceholderText("new class name")
        self.new_class_edit.returnPressed.connect(self._on_add_class)
        self.add_class_button = QPushButton("Add")
        self.add_class_button.clicked.connect(self._on_add_class)
        add_row.addWidget(self.new_class_edit)
        add_row.addWidget(self.add_class_button)
        layout.addLayout(add_row)

        self.remove_class_button = QPushButton("Remove selected class")
        self.remove_class_button.clicked.connect(self._on_remove_class)
        layout.addWidget(self.remove_class_button)

        self.current_label_display = QLabel("Frame: - | Label: -")
        layout.addWidget(self.current_label_display)

        self.clear_label_button = QPushButton("Clear label(s) on this frame/selection")
        self.clear_label_button.clicked.connect(self._on_clear_label)
        layout.addWidget(self.clear_label_button)

        annotations_io_row = QHBoxLayout()
        self.save_annotations_button = QPushButton("Save annotations...")
        self.save_annotations_button.clicked.connect(self._on_save_annotations_clicked)
        self.load_annotations_button = QPushButton("Load annotations...")
        self.load_annotations_button.clicked.connect(self._on_load_annotations_clicked)
        annotations_io_row.addWidget(self.save_annotations_button)
        annotations_io_row.addWidget(self.load_annotations_button)
        layout.addLayout(annotations_io_row)

        self.train_predict_button = QPushButton("Train + Predict")
        self.train_predict_button.clicked.connect(self._on_train_and_predict_clicked)
        layout.addWidget(self.train_predict_button)

        self.status_label = QLabel("Not trained yet")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        model_io_row = QHBoxLayout()
        self.export_model_button = QPushButton("Export model...")
        self.export_model_button.clicked.connect(self._on_export_model_clicked)
        self.export_predictions_button = QPushButton("Export predictions (.csv)...")
        self.export_predictions_button.clicked.connect(self._on_export_predictions_clicked)
        model_io_row.addWidget(self.export_model_button)
        model_io_row.addWidget(self.export_predictions_button)
        layout.addLayout(model_io_row)

        self.viewer.dims.events.current_step.connect(self._refresh_current_label)

    # -- opening / switching files --

    @property
    def active_layers(self) -> ProjectLayers | None:
        if self.active_h5_path is None:
            return None
        return self.open_files[self.active_h5_path].layers

    def _on_open_clicked(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Open SLEAP analysis", "", "SLEAP analysis files (*.h5)")
        for path in paths:
            self.open_file(path)

    def open_file(self, h5_path: str | Path) -> None:
        h5_path = str(Path(h5_path))
        if h5_path not in self.open_files:
            data = load_project_data(h5_path)
            skeleton = extract_skeleton(data.ds)
            if self._reference_skeleton is None:
                self._reference_skeleton = skeleton
            else:
                try:
                    check_consistent_skeleton(self._reference_skeleton, skeleton, context=Path(h5_path).name)
                except ValueError as e:
                    QMessageBox.warning(self, "Skeleton mismatch", str(e))
                    self.status_label.setText(f"Skeleton mismatch: {Path(h5_path).name} not added (see dialog)")
                    return
            self.open_files[h5_path] = OpenFile(h5_path=h5_path, data=data)
            item = QListWidgetItem(Path(h5_path).name)
            item.setData(Qt.ItemDataRole.UserRole, h5_path)
            self.file_list.addItem(item)
        self._select_file_in_list(h5_path)

    def _select_file_in_list(self, h5_path: str) -> None:
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == h5_path:
                self.file_list.setCurrentItem(item)
                return

    def _on_remove_file_clicked(self) -> None:
        row = self.file_list.currentRow()
        if row < 0:
            return
        h5_path = self.file_list.item(row).data(Qt.ItemDataRole.UserRole)

        if h5_path == self.active_h5_path:
            layers = self.open_files[h5_path].layers
            if layers is not None:
                remove_layers(self.viewer, layers)
            self.active_h5_path = None

        self.open_files.pop(h5_path, None)
        self._last_frame.pop(h5_path, None)
        self._raw_features_cache = {k: v for k, v in self._raw_features_cache.items() if k[0] != h5_path}
        self._combined_features_cache = {k: v for k, v in self._combined_features_cache.items() if k[0] != h5_path}
        self._predictions = {k: v for k, v in self._predictions.items() if k[0] != h5_path}

        # removing the current row auto-selects a neighboring file (if any), which
        # fires _on_file_selection_changed -> _switch_to_file for us
        self.file_list.takeItem(row)

        if not self.open_files:
            self._reference_skeleton = None  # no files left - a future open() may set any skeleton as reference
            self.individual_combo.clear()
            self.current_label_display.setText("Frame: - | Label: -")
            self.timeline_widget.set_data(0, None, None, None, {})

        self.status_label.setText(f"Removed {Path(h5_path).name}")

    def _on_file_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        if current is None:
            return
        self._switch_to_file(current.data(Qt.ItemDataRole.UserRole))

    def _switch_to_file(self, h5_path: str) -> None:
        if h5_path == self.active_h5_path:
            return
        self.timeline_widget.clear_selection()
        if self.active_h5_path is not None:
            self._last_frame[self.active_h5_path] = int(self.viewer.dims.current_step[0])
            old = self.open_files[self.active_h5_path]
            if old.layers is not None:
                remove_layers(self.viewer, old.layers)
                old.layers = None

        open_file = self.open_files[h5_path]
        open_file.layers = build_layers(
            open_file.data,
            self.viewer,
            get_box_style=lambda individual, frame, sp=h5_path: self._style_for_individual(sp, individual, frame),
            get_active_individual=lambda: self.individual_combo.currentText(),
        )
        self.active_h5_path = h5_path

        n_frames = open_file.data.ds.sizes["time"]
        frame = min(self._last_frame.get(h5_path, 0), n_frames - 1)
        self.viewer.dims.set_current_step(0, frame)

        previous_individual = self.individual_combo.currentText()
        self.individual_combo.blockSignals(True)
        self.individual_combo.clear()
        self.individual_combo.addItems([str(v) for v in open_file.data.ds.coords["individuals"].values])
        index = self.individual_combo.findText(previous_individual)
        if index >= 0:
            self.individual_combo.setCurrentIndex(index)
        self.individual_combo.blockSignals(False)

        self._refresh_current_label()
        self._refresh_timeline()

    def _on_individual_changed(self, index: int) -> None:
        self._refresh_current_label()
        self._refresh_timeline()
        if self.active_layers is not None:
            self.active_layers.refresh()

    # -- classes --

    def _on_add_class(self) -> None:
        name = self.new_class_edit.text().strip()
        if not name or name in self._class_names():
            return
        color = _CLASS_COLORS[len(self._class_colors) % len(_CLASS_COLORS)]
        self._class_colors[name] = color
        item = QListWidgetItem(f"{self.class_list.count() + 1}. {name}")
        item.setBackground(QColor(color))
        item.setData(Qt.ItemDataRole.UserRole, name)
        self.class_list.addItem(item)
        self.new_class_edit.clear()
        self._rebind_hotkeys()
        self._refresh_timeline()

    def _on_remove_class(self) -> None:
        row = self.class_list.currentRow()
        if row >= 0:
            self.class_list.takeItem(row)
            self._renumber_classes()
            self._rebind_hotkeys()
            self._refresh_timeline()

    def _renumber_classes(self) -> None:
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)
            item.setText(f"{i + 1}. {name}")

    def _class_names(self) -> list[str]:
        return [self.class_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.class_list.count())]

    def _rebind_hotkeys(self) -> None:
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self._shortcuts.clear()

        main_window = self.viewer.window._qt_window
        for i, name in enumerate(self._class_names()[:9]):
            shortcut = QShortcut(QKeySequence(str(i + 1)), main_window)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(self._make_hotkey_callback(name))
            self._shortcuts.append(shortcut)

    def _make_hotkey_callback(self, class_name: str):
        def _callback():
            self._assign_label(class_name)

        return _callback

    # -- labeling --

    def _assign_label(self, class_name: str) -> None:
        if self.active_h5_path is None:
            return
        individual = self.individual_combo.currentText()
        selection = self.timeline_widget.selection
        if selection is not None:
            start_frame, end_frame = selection
            self.store.set_range(self.active_h5_path, individual, start_frame, end_frame, class_name)
            self.timeline_widget.clear_selection()
        else:
            frame = int(self.viewer.dims.current_step[0])
            self.store.set(self.active_h5_path, individual, frame, class_name)
        self.active_layers.refresh()
        self._refresh_current_label()
        self._refresh_timeline()

    def _on_clear_label(self) -> None:
        if self.active_h5_path is None:
            return
        individual = self.individual_combo.currentText()
        selection = self.timeline_widget.selection
        if selection is not None:
            start_frame, end_frame = selection
            self.store.clear_range(self.active_h5_path, individual, start_frame, end_frame)
            self.timeline_widget.clear_selection()
        else:
            frame = int(self.viewer.dims.current_step[0])
            self.store.clear(self.active_h5_path, individual, frame)
        self.active_layers.refresh()
        self._refresh_current_label()
        self._refresh_timeline()

    def _refresh_current_label(self, event=None) -> None:
        if self.active_h5_path is None:
            return
        individual = self.individual_combo.currentText()
        frame = int(self.viewer.dims.current_step[0])
        label = self.store.get(self.active_h5_path, individual, frame) or "-"
        selection = self.timeline_widget.selection
        if selection is not None:
            start_frame, end_frame = selection
            self.current_label_display.setText(
                f"Frame: {frame} | Label: {label} | Selection: {start_frame}-{end_frame} "
                f"({end_frame - start_frame + 1} frames)"
            )
        else:
            self.current_label_display.setText(f"Frame: {frame} | Label: {label}")

    def _style_for_individual(self, source_file: str, individual: str, frame: int) -> tuple[str, float]:
        """(edge_color, edge_width) for this individual's bounding box - a manual annotation
        renders bolder than a model prediction so the two are distinguishable at a glance."""
        label = self.store.get(source_file, individual, frame)
        if label:
            return self._class_colors.get(label, DEFAULT_BOX_COLOR), ANNOTATION_EDGE_WIDTH
        predictions = self._predictions.get((source_file, individual))
        if predictions is not None:
            return self._class_colors.get(predictions[frame], DEFAULT_BOX_COLOR), PREDICTION_EDGE_WIDTH
        return DEFAULT_BOX_COLOR, PREDICTION_EDGE_WIDTH

    def _on_save_annotations_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save annotations", "", "CSV files (*.csv)")
        if path:
            self.store.save(path)

    def _on_load_annotations_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load annotations", "", "CSV files (*.csv)")
        if not path:
            return
        loaded = LabelStore.load(path)
        self.store.labels.update(loaded.labels)
        self._refresh_current_label()
        self._refresh_timeline()

    # -- features / training --

    def _get_raw_feature_groups(self, source_file: str, individual: str) -> list[kinematics.FeatureGroup]:
        key = (source_file, individual)
        if key not in self._raw_features_cache:
            ds = self.open_files[source_file].data.ds
            self._raw_features_cache[key] = kinematics.compute_feature_groups(ds, individual)
        return self._raw_features_cache[key]

    def _get_combined_features(self, source_file: str, individual: str) -> tuple[np.ndarray, list[str]]:
        key = (source_file, individual)
        if key not in self._combined_features_cache:
            raw_features, raw_names = kinematics.flatten_feature_groups(self._get_raw_feature_groups(source_file, individual))
            self._combined_features_cache[key] = filterbank.with_filter_bank(raw_features, raw_names)
        return self._combined_features_cache[key]

    def _on_train_and_predict_clicked(self) -> None:
        if self._on_train_clicked():
            self._on_predict_clicked()

    def _on_train_clicked(self) -> bool:
        """Fits the shared pooled model. Returns whether training actually happened,
        so the combined Train+Predict button can skip predicting on failure."""
        if not self.open_files:
            return False

        all_features = []
        all_labels: dict[int, str] = {}
        offset = 0
        contributing_individuals: set[str] = set()
        contributing_files: set[str] = set()

        for source_file, open_file in self.open_files.items():
            individuals = [str(v) for v in open_file.data.ds.coords["individuals"].values]
            for individual in individuals:
                labeled_frames = self.store.labeled_frames(source_file, individual)
                if not labeled_frames:
                    continue
                features, _names = self._get_combined_features(source_file, individual)
                for frame in labeled_frames:
                    all_labels[offset + frame] = self.store.get(source_file, individual, frame)
                all_features.append(features)
                contributing_individuals.add(individual)
                contributing_files.add(source_file)
                offset += features.shape[1]

        if len(set(all_labels.values())) < 2:
            self.status_label.setText("Need labeled frames from at least 2 classes, across any animals/files")
            return False

        combined = np.concatenate(all_features, axis=1)
        pipeline = train_module.train(combined, all_labels)
        self._monadic_pipeline = pipeline
        report = train_module.oob_summary(pipeline, combined, all_labels)

        self.status_label.setText(
            f"Trained on {len(all_labels)} frames pooled across {len(contributing_individuals)} "
            f"individual(s) in {len(contributing_files)} file(s) "
            f"({len(set(all_labels.values()))} classes, {combined.shape[0]} features)"
            + (f" | OOB accuracy: {report.accuracy:.0%}" if report is not None else "")
        )
        class_counts = Counter(all_labels.values())
        file_names = [Path(f).name for f in contributing_files]
        _TrainingSummaryDialog(
            self, sorted(contributing_individuals), file_names, class_counts, combined.shape[0], report
        ).exec()
        return True

    def _predict_all_open_files(self, pipeline) -> int:
        """Predict every individual across every open file with the shared monadic model."""
        n_predicted = 0
        for source_file, open_file in self.open_files.items():
            individuals = [str(v) for v in open_file.data.ds.coords["individuals"].values]
            for individual in individuals:
                features, _names = self._get_combined_features(source_file, individual)
                self._predictions[(source_file, individual)] = train_module.predict(pipeline, features)
                n_predicted += 1
        return n_predicted

    def _on_predict_clicked(self) -> None:
        if self._monadic_pipeline is None:
            self.status_label.setText("No trained (or loaded) model yet")
            return

        n_predicted = self._predict_all_open_files(self._monadic_pipeline)
        self.status_label.setText(f"Predicted on {n_predicted} animal track(s)")

        if self.active_layers is not None:
            self.active_layers.refresh()
        self._refresh_timeline()

    def _on_export_model_clicked(self) -> None:
        """Export-only: the model is meant for later batch processing (e.g. via `api.py`
        or another tool), not for loading back into this widget."""
        if self._monadic_pipeline is None:
            self.status_label.setText("Train a model before exporting it")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export model", "", "Model files (*.joblib)")
        if path:
            train_module.save_pipeline(self._monadic_pipeline, path)
            self.status_label.setText(f"Exported model to {Path(path).name}")

    def _on_export_predictions_clicked(self) -> None:
        if not self._predictions:
            self.status_label.setText("No predictions yet - run Train + Predict first")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export predictions", "", "CSV files (*.csv)")
        if not path:
            return

        rows = [
            (Path(source_file).name, individual, frame, class_name)
            for (source_file, individual), predictions in self._predictions.items()
            for frame, class_name in enumerate(predictions)
        ]
        pd.DataFrame(rows, columns=["File", "Animal_ID", "Frame", "Class"]).to_csv(path, index=False)
        self.status_label.setText(f"Exported {len(rows)} predicted frames to {Path(path).name}")

    # -- session (which files + annotations) --

    def _on_save_session_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save session", "", "Session files (*.json)")
        if path:
            save_session(path, list(self.open_files.keys()), self.store, self._class_colors)

    def _on_load_session_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load session", "", "Session files (*.json)")
        if not path:
            return
        h5_paths, store, class_colors = load_session(path)
        self.store = store

        self.class_list.clear()
        self._class_colors = {}
        for name, color in class_colors.items():
            self._class_colors[name] = color
            item = QListWidgetItem(f"{self.class_list.count() + 1}. {name}")
            item.setBackground(QColor(color))
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.class_list.addItem(item)
        self._rebind_hotkeys()

        for h5_path in h5_paths:
            self.open_file(h5_path)

        self._refresh_current_label()
        self._refresh_timeline()

    # -- timeline --

    def _refresh_timeline(self) -> None:
        if self.active_h5_path is None:
            return
        individual = self.individual_combo.currentText()
        if not individual:
            return
        source_file = self.active_h5_path
        n_frames = self.open_files[source_file].data.ds.sizes["time"]
        groups = self._get_raw_feature_groups(source_file, individual)
        raw_features, _ = kinematics.flatten_feature_groups(groups)
        self.timeline_widget.set_data(
            n_frames=n_frames,
            predictions=self._predictions.get((source_file, individual)),
            annotations=self.store.to_dense_array(source_file, individual, n_frames),
            features=raw_features,
            feature_group_sizes=[(g.name, g.features.shape[0]) for g in groups],
            class_colors=dict(self._class_colors),
        )
