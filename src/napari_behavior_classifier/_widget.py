"""Main napari dock widget: open one or more SLEAP projects, annotate across all of them."""

from dataclasses import dataclass
from pathlib import Path

import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .annotation.store import LabelStore
from .classifier import train as train_module
from .features import filterbank, kinematics
from .session import load_session, save_session
from .viz.timeline_widget import TimelineWidget
from .viz.viewer import DEFAULT_BOX_COLOR, ProjectData, ProjectLayers, build_layers, load_project_data, remove_layers

_CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
]


@dataclass
class OpenFile:
    slp_path: str
    data: ProjectData
    layers: ProjectLayers | None = None


class BehaviorClassifierWidget(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.open_files: dict[str, OpenFile] = {}
        self.active_slp_path: str | None = None
        self.store = LabelStore()
        self._shortcuts: list[QShortcut] = []
        self._class_colors: dict[str, str] = {}
        self._raw_features_cache: dict[tuple[str, str], tuple[np.ndarray, list[str]]] = {}
        self._combined_features_cache: dict[tuple[str, str], tuple[np.ndarray, list[str]]] = {}
        self._pipelines: dict[str, object] = {}  # keyed by individual name, pooled across files
        self._predictions: dict[tuple[str, str], np.ndarray] = {}  # keyed by (source_file, individual)

        self.timeline_widget = TimelineWidget(viewer)
        self.viewer.window.add_dock_widget(self.timeline_widget, area="bottom", name="Timeline")

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.open_button = QPushButton("Open SLEAP project (.slp)...")
        self.open_button.clicked.connect(self._on_open_clicked)
        layout.addWidget(self.open_button)

        layout.addWidget(QLabel("Open files (click to switch):"))
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self._on_file_selection_changed)
        layout.addWidget(self.file_list)

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

        layout.addWidget(QLabel("Classes (press 1-9 to label current frame):"))
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

        self.clear_label_button = QPushButton("Clear label on this frame")
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

        self.train_button = QPushButton("Train on all open files and predict")
        self.train_button.clicked.connect(self._on_train_clicked)
        layout.addWidget(self.train_button)

        self.train_status_label = QLabel("Not trained yet")
        layout.addWidget(self.train_status_label)

        model_io_row = QHBoxLayout()
        self.save_model_button = QPushButton("Save model...")
        self.save_model_button.clicked.connect(self._on_save_model_clicked)
        self.load_model_button = QPushButton("Load model...")
        self.load_model_button.clicked.connect(self._on_load_model_clicked)
        model_io_row.addWidget(self.save_model_button)
        model_io_row.addWidget(self.load_model_button)
        layout.addLayout(model_io_row)

        self.viewer.dims.events.current_step.connect(self._refresh_current_label)

    # -- opening / switching files --

    @property
    def active_layers(self) -> ProjectLayers | None:
        if self.active_slp_path is None:
            return None
        return self.open_files[self.active_slp_path].layers

    def _on_open_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open SLEAP predictions", "", "SLEAP files (*.slp)")
        if path:
            self.open_file(path)

    def open_file(self, slp_path: str | Path) -> None:
        slp_path = str(Path(slp_path))
        if slp_path not in self.open_files:
            data = load_project_data(slp_path)
            self.open_files[slp_path] = OpenFile(slp_path=slp_path, data=data)
            item = QListWidgetItem(Path(slp_path).name)
            item.setData(Qt.ItemDataRole.UserRole, slp_path)
            self.file_list.addItem(item)
        self._select_file_in_list(slp_path)

    def _select_file_in_list(self, slp_path: str) -> None:
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == slp_path:
                self.file_list.setCurrentItem(item)
                return

    def _on_file_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        if current is None:
            return
        self._switch_to_file(current.data(Qt.ItemDataRole.UserRole))

    def _switch_to_file(self, slp_path: str) -> None:
        if slp_path == self.active_slp_path:
            return
        if self.active_slp_path is not None:
            old = self.open_files[self.active_slp_path]
            if old.layers is not None:
                remove_layers(self.viewer, old.layers)
                old.layers = None

        open_file = self.open_files[slp_path]
        open_file.layers = build_layers(
            open_file.data,
            self.viewer,
            get_box_color=lambda individual, frame, sp=slp_path: self._color_for_individual(sp, individual, frame),
        )
        self.active_slp_path = slp_path

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
        if self.active_slp_path is None:
            return
        individual = self.individual_combo.currentText()
        frame = int(self.viewer.dims.current_step[0])
        self.store.set(self.active_slp_path, individual, frame, class_name)
        self.active_layers.refresh()
        self._refresh_current_label()
        self._refresh_timeline()

    def _on_clear_label(self) -> None:
        if self.active_slp_path is None:
            return
        individual = self.individual_combo.currentText()
        frame = int(self.viewer.dims.current_step[0])
        self.store.clear(self.active_slp_path, individual, frame)
        self.active_layers.refresh()
        self._refresh_current_label()
        self._refresh_timeline()

    def _refresh_current_label(self, event=None) -> None:
        if self.active_slp_path is None:
            return
        individual = self.individual_combo.currentText()
        frame = int(self.viewer.dims.current_step[0])
        label = self.store.get(self.active_slp_path, individual, frame) or "-"
        self.current_label_display.setText(f"Frame: {frame} | Label: {label}")

    def _color_for_individual(self, source_file: str, individual: str, frame: int) -> str:
        label = self.store.get(source_file, individual, frame)
        if label:
            return self._class_colors.get(label, DEFAULT_BOX_COLOR)
        predictions = self._predictions.get((source_file, individual))
        if predictions is not None:
            return self._class_colors.get(predictions[frame], DEFAULT_BOX_COLOR)
        return DEFAULT_BOX_COLOR

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

    def _get_raw_features(self, source_file: str, individual: str) -> tuple[np.ndarray, list[str]]:
        key = (source_file, individual)
        if key not in self._raw_features_cache:
            ds = self.open_files[source_file].data.ds
            self._raw_features_cache[key] = kinematics.compute_features(ds, individual)
        return self._raw_features_cache[key]

    def _get_combined_features(self, source_file: str, individual: str) -> tuple[np.ndarray, list[str]]:
        key = (source_file, individual)
        if key not in self._combined_features_cache:
            raw_features, raw_names = self._get_raw_features(source_file, individual)
            self._combined_features_cache[key] = filterbank.with_filter_bank(raw_features, raw_names)
        return self._combined_features_cache[key]

    def _on_train_clicked(self) -> None:
        if not self.open_files:
            return
        individual = self.individual_combo.currentText()
        if not individual:
            return

        all_features = []
        all_labels: dict[int, str] = {}
        file_spans: dict[str, tuple[int, int]] = {}
        offset = 0

        for source_file, open_file in self.open_files.items():
            individuals = [str(v) for v in open_file.data.ds.coords["individuals"].values]
            if individual not in individuals:
                continue
            features, _names = self._get_combined_features(source_file, individual)
            for frame in self.store.labeled_frames(source_file, individual):
                all_labels[offset + frame] = self.store.get(source_file, individual, frame)
            all_features.append(features)
            file_spans[source_file] = (offset, features.shape[1])
            offset += features.shape[1]

        if len(set(all_labels.values())) < 2:
            self.train_status_label.setText("Need labeled frames from at least 2 classes, across any open files")
            return

        combined = np.concatenate(all_features, axis=1)
        pipeline = train_module.train(combined, all_labels)
        self._pipelines[individual] = pipeline

        combined_predictions = train_module.predict(pipeline, combined)
        for source_file, (start, length) in file_spans.items():
            self._predictions[(source_file, individual)] = combined_predictions[start : start + length]

        frames = sorted(all_labels)
        train_accuracy = pipeline.score(combined[:, frames].T, [all_labels[f] for f in frames])
        self.train_status_label.setText(
            f"Trained on {len(all_labels)} frames across {len(file_spans)} file(s) "
            f"({len(set(all_labels.values()))} classes, {combined.shape[0]} features) | "
            f"train accuracy: {train_accuracy:.0%}"
        )

        if self.active_layers is not None:
            self.active_layers.refresh()
        self._refresh_timeline()

    def _on_save_model_clicked(self) -> None:
        individual = self.individual_combo.currentText()
        pipeline = self._pipelines.get(individual)
        if pipeline is None:
            self.train_status_label.setText("Train a model before saving it")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save model", "", "Model files (*.joblib)")
        if path:
            train_module.save_pipeline(pipeline, path)

    def _on_load_model_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load model", "", "Model files (*.joblib)")
        if not path or not self.open_files:
            return
        individual = self.individual_combo.currentText()
        pipeline = train_module.load_pipeline(path)
        self._pipelines[individual] = pipeline

        for source_file, open_file in self.open_files.items():
            individuals = [str(v) for v in open_file.data.ds.coords["individuals"].values]
            if individual not in individuals:
                continue
            features, _names = self._get_combined_features(source_file, individual)
            self._predictions[(source_file, individual)] = train_module.predict(pipeline, features)

        self.train_status_label.setText(f"Loaded model from {Path(path).name}")
        if self.active_layers is not None:
            self.active_layers.refresh()
        self._refresh_timeline()

    # -- session (which files + annotations) --

    def _on_save_session_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save session", "", "Session files (*.json)")
        if path:
            save_session(path, list(self.open_files.keys()), self.store, self._class_colors)

    def _on_load_session_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load session", "", "Session files (*.json)")
        if not path:
            return
        slp_paths, store, class_colors = load_session(path)
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

        for slp_path in slp_paths:
            self.open_file(slp_path)

        self._refresh_current_label()
        self._refresh_timeline()

    # -- timeline --

    def _refresh_timeline(self) -> None:
        if self.active_slp_path is None:
            return
        individual = self.individual_combo.currentText()
        if not individual:
            return
        source_file = self.active_slp_path
        n_frames = self.open_files[source_file].data.ds.sizes["time"]
        raw_features, _ = self._get_raw_features(source_file, individual)
        self.timeline_widget.set_data(
            n_frames=n_frames,
            predictions=self._predictions.get((source_file, individual)),
            annotations=self.store.to_dense_array(source_file, individual, n_frames),
            features=raw_features,
            class_colors=dict(self._class_colors),
        )
