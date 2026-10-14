"""CrackQuant GUI - English interface."""
import os, sys, csv, gc
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (
    QMainWindow, QGraphicsView, QGraphicsScene,
    QMenuBar, QAction, QToolBar, QStatusBar, QDockWidget,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QGroupBox, QFileDialog, QMessageBox, QDoubleSpinBox,
    QSplitter, QTabWidget, QFormLayout, QAbstractItemView,
    QComboBox, QCheckBox, QGraphicsRectItem, QDialog, QDialogButtonBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QBrush, QPen


class HoverSortHeader(QHeaderView):
    """Show the next sort direction on hover; sort only after a header click."""
    sort_requested = pyqtSignal(int, object)

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setSectionsClickable(True)
        self.setMouseTracking(True)
        self.setSortIndicatorShown(False)
        self._hover_section = -1
        self._sorted_section = -1
        self._sorted_order = Qt.AscendingOrder

    def _next_order(self, section):
        if section == self._sorted_section:
            return Qt.DescendingOrder if self._sorted_order == Qt.AscendingOrder else Qt.AscendingOrder
        return Qt.AscendingOrder

    def mouseMoveEvent(self, event):
        section = self.logicalIndexAt(event.pos())
        if section != self._hover_section:
            self._hover_section = section
            self.viewport().update()
        self.setToolTip("Click a column header to sort; click again to reverse the order.")
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hover_section = -1
        self.viewport().update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        section = self.logicalIndexAt(event.pos())
        if event.button() == Qt.LeftButton and section >= 0:
            order = self._next_order(section)
            self._sorted_section = section
            self._sorted_order = order
            self.sort_requested.emit(section, self._sorted_order)
            self.viewport().update()
            event.accept()
            return
        super().mousePressEvent(event)

    def paintSection(self, painter, rect, logical_index):
        super().paintSection(painter, rect, logical_index)
        if logical_index != self._hover_section:
            return
        painter.save()
        painter.setPen(QColor("#555"))
        arrow = "▲" if self._next_order(logical_index) == Qt.AscendingOrder else "▼"
        painter.drawText(rect.adjusted(4, 0, -7, 0), Qt.AlignRight | Qt.AlignVCenter, arrow)
        painter.restore()


class NumericTableWidgetItem(QTableWidgetItem):
    """Sort result columns by numeric value."""
    def __init__(self, text, value):
        super().__init__(text)
        self._numeric_value = float(value)

    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return self._numeric_value < other._numeric_value
        return super().__lt__(other)

# Lazy imports to reduce startup memory
_predict = None; _load_model = None; _DEVICE = None; _quantify = None

def _get_predict():
    global _predict, _load_model, _DEVICE
    if _predict is None:
        from inference import predict as p, load_model as lm, DEVICE as d
        _predict = p; _load_model = lm; _DEVICE = d
    return _predict, _load_model, _DEVICE

def _get_quantify():
    global _quantify
    if _quantify is None:
        from quantify import quantify as q
        _quantify = q
    return _quantify


class SegThread(QThread):
    done = pyqtSignal(object, object)
    err = pyqtSignal(str)
    msg = pyqtSignal(str)
    def __init__(self, path, thresh):
        super().__init__(); self.path = path; self.thresh = thresh
    def run(self):
        try:
            self.msg.emit("Loading model...")
            predict_fn, _, _ = _get_predict()
            bin_mask, prob = predict_fn(self.path, self.thresh)
            self.done.emit(bin_mask, prob)
        except Exception as e:
            self.err.emit(str(e))


class QuantThread(QThread):
    done = pyqtSignal(list, np.ndarray, np.ndarray)
    err = pyqtSignal(str)
    msg = pyqtSignal(str)
    def __init__(self, binary, pixel_resolution_um=1.0):
        super().__init__(); self.binary = binary; self.pixel_resolution_um = pixel_resolution_um
    def run(self):
        try:
            self.msg.emit("Extracting skeleton...")
            q = _get_quantify()
            res, skel, ov = q(self.binary, self.pixel_resolution_um)
            self.done.emit(res, skel, ov)
        except Exception as e:
            self.err.emit(str(e))


class BatchThread(QThread):
    """Process batch images sequentially."""
    item_done = pyqtSignal(int, object, object, object, object, object)
    err = pyqtSignal(str)
    msg = pyqtSignal(str)
    done = pyqtSignal(int)

    def __init__(self, paths, threshold, pixel_resolution_um=1.0):
        super().__init__()
        self.paths = paths
        self.threshold = threshold
        self.pixel_resolution_um = pixel_resolution_um

    def run(self):
        try:
            predict_fn, _, _ = _get_predict()
            quantify_fn = _get_quantify()
            for index, path in enumerate(self.paths):
                self.msg.emit(f"Batch processing {index + 1}/{len(self.paths)}: {os.path.basename(path)}")
                binary, prob = predict_fn(path, self.threshold)
                results, skeleton, overlay = quantify_fn(binary, self.pixel_resolution_um)
                self.item_done.emit(index, binary, prob, results, skeleton, overlay)
                gc.collect()
            self.done.emit(len(self.paths))
        except Exception as e:
            self.err.emit(str(e))


class ZoomableView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._z = 0

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 0.85
        self._z += 1 if f > 1 else -1
        if self._z > 0: self.scale(f, f)
        else: self._z = 0; self.fit_view()

    def fit_view(self):
        self._z = 0
        if self.scene() and self.scene().items():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.fit_view)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._z == 0:
            QTimer.singleShot(0, self.fit_view)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CrackQuant - Coal CT Crack Recognition and Quantification")
        self.resize(1500, 900)
        self.img_path = None
        self.orig_arr = None
        self.bin_mask = None
        self.prob_map = None
        self.quant_rst = None
        self.skel_arr = None
        self.over_arr = None
        # Each batch item stores its own image, segmentation and quantification results.
        self.batch_items = []
        self.current_index = -1
        self._updating_batch_combo = False
        self._batch_active = False

        self._setup_menu()
        self._setup_toolbar()
        self._setup_central()
        self._setup_dock()
        self.statusBar().showMessage("Ready - open an image to begin")

    # ======== Menu ========
    def _setup_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("File(&F)")
        fm.addAction("Open Image...\tCtrl+O", self._open)
        fm.addAction("Import Images...\tCtrl+Shift+O", self._open_batch)
        fm.addSeparator()
        fm.addAction("Save Results...\tCtrl+S", self._save)
        fm.addAction("Export Batch Results...", self._export_batch)
        fm.addSeparator()
        fm.addAction("Exit\tAlt+F4", self.close)
        vm = mb.addMenu("View(&V)")
        vm.addAction("Fit to Window\tCtrl+F", self._fit_all)
        hm = mb.addMenu("Help(&H)")
        hm.addAction("About", self._about)

    # ======== Toolbar ========
    def _setup_toolbar(self):
        tb = self.addToolBar("Main Toolbar"); tb.setMovable(False)
        a = tb.addAction("Open Image"); a.triggered.connect(self._open)
        a = tb.addAction("Import Images"); a.triggered.connect(self._open_batch)
        a = tb.addAction("◀ Previous"); a.triggered.connect(self._previous_image); self._act_prev = a
        a = tb.addAction("Next ▶"); a.triggered.connect(self._next_image); self._act_next = a
        tb.addSeparator()
        a = tb.addAction("Segment Cracks"); a.triggered.connect(self._segment); self._act_seg = a
        tb.addSeparator()
        a = tb.addAction("Quantify Cracks"); a.triggered.connect(self._quantify); self._act_quant = a
        tb.addSeparator()
        a = tb.addAction("Save Results"); a.triggered.connect(self._save); self._act_save = a
        a = tb.addAction("Export Batch"); a.triggered.connect(self._export_batch); self._act_export_batch = a
        self._act_seg.setEnabled(False); self._act_quant.setEnabled(False)
        self._act_save.setEnabled(False)
        self._act_prev.setEnabled(False); self._act_next.setEnabled(False)
        self._act_export_batch.setEnabled(False)

    # ======== Central views ========
    def _setup_central(self):
        cw = QWidget(); self.setCentralWidget(cw)
        lo = QVBoxLayout(cw); lo.setContentsMargins(2,2,2,2)
        self._tabs = QTabWidget()

        # Tab 0: original image | probability map | binary segmentation
        s0 = QSplitter(Qt.Horizontal)
        self._gv_orig = ZoomableView(); self._gv_orig.setScene(QGraphicsScene())
        self._gv_prob = ZoomableView(); self._gv_prob.setScene(QGraphicsScene())
        self._gv_seg  = ZoomableView(); self._gv_seg.setScene(QGraphicsScene())
        for gv,tt in [(self._gv_orig,"Original"),(self._gv_prob,"Probability Map"),(self._gv_seg,"Binary Segmentation")]:
            lb = QLabel(tt); lb.setStyleSheet("color:#aaa;font-size:10px;padding:2px;background:#1a1a2e")
        s0.addWidget(self._gv_orig); s0.addWidget(self._gv_prob); s0.addWidget(self._gv_seg)
        self._tabs.addTab(s0, "Step 1 - Crack Segmentation")

        # Tab 1: skeleton
        self._gv_skel = ZoomableView(); self._gv_skel.setScene(QGraphicsScene())
        self._tabs.addTab(self._gv_skel, "Step 2 - Skeleton")

        # Tab 2: quantification
        self._gv_quant = ZoomableView(); self._gv_quant.setScene(QGraphicsScene())
        self._tabs.addTab(self._gv_quant, "Quantification Results")
        self._tabs.currentChanged.connect(self._fit_current_tab)

        lo.addWidget(self._tabs)

    # ======== Right panel ========
    def _setup_dock(self):
        dock = QDockWidget("Control Panel", self); dock.setAllowedAreas(Qt.RightDockWidgetArea)
        dock.setMinimumWidth(360)
        panel = QWidget(); dock.setWidget(panel)
        pl = QVBoxLayout(panel)
        pl.setSpacing(9)

        # Batch tasks and image navigation
        gb = QGroupBox("Image Tasks")
        lb = QVBoxLayout()
        row = QHBoxLayout()
        self._btn_open_batch = QPushButton("Import Images"); self._btn_open_batch.clicked.connect(self._open_batch)
        self._btn_prev = QPushButton("◀ Previous"); self._btn_prev.clicked.connect(self._previous_image)
        self._btn_next = QPushButton("Next ▶"); self._btn_next.clicked.connect(self._next_image)
        row.addWidget(self._btn_open_batch); row.addWidget(self._btn_prev); row.addWidget(self._btn_next)
        lb.addLayout(row)
        row2 = QHBoxLayout()
        self._btn_remove_item = QPushButton("Remove Current"); self._btn_remove_item.clicked.connect(self._remove_current_item)
        self._btn_clear_cache = QPushButton("Clear Cache"); self._btn_clear_cache.clicked.connect(self._clear_batch_cache)
        row2.addWidget(self._btn_remove_item); row2.addWidget(self._btn_clear_cache)
        lb.addLayout(row2)
        self._combo_batch = QComboBox(); self._combo_batch.setMinimumHeight(30)
        self._combo_batch.currentIndexChanged.connect(self._select_batch_item)
        self._combo_batch.setEnabled(False)
        lb.addWidget(self._combo_batch)
        self._lbl_batch = QLabel("Current task: no image imported")
        self._lbl_batch.setWordWrap(True)
        lb.addWidget(self._lbl_batch)
        gb.setLayout(lb); pl.addWidget(gb)

        # Step 1
        g1 = QGroupBox("Step 1 - BCM-UNet Segmentation")
        l1 = QVBoxLayout()
        fl = QFormLayout()
        self._spin_th = QDoubleSpinBox(); self._spin_th.setRange(0.1,0.9)
        self._spin_th.setValue(0.5); self._spin_th.setSingleStep(0.05)
        self._spin_th.setToolTip("Probability threshold for binary conversion (value > threshold = crack)")
        fl.addRow("Binary Threshold:", self._spin_th)
        self._check_scale = QCheckBox("Enable physical scale conversion (optional)")
        self._spin_scale = QDoubleSpinBox(); self._spin_scale.setRange(0.000001, 1000000.0)
        self._spin_scale.setDecimals(6); self._spin_scale.setValue(1.0); self._spin_scale.setSingleStep(0.1)
        self._spin_scale.setSuffix(" μm / px"); self._spin_scale.setEnabled(False)
        self._check_scale.toggled.connect(self._spin_scale.setEnabled)
        fl.addRow(self._check_scale)
        fl.addRow("Pixel Resolution:", self._spin_scale)
        l1.addLayout(fl)
        self._btn_seg = QPushButton("▶ Segment Current Image (BCM-UNet)"); self._btn_seg.clicked.connect(self._segment)
        l1.addWidget(self._btn_seg)
        self._btn_batch_run = QPushButton("▶ Process and Quantify All Images"); self._btn_batch_run.clicked.connect(self._run_batch)
        self._btn_batch_run.setEnabled(False); l1.addWidget(self._btn_batch_run)
        self._prog1 = QProgressBar(); self._prog1.setVisible(False); l1.addWidget(self._prog1)
        l1.addWidget(QLabel("Pipeline: original → model inference → probability map → thresholding"))
        g1.setLayout(l1); pl.addWidget(g1)

        # Step 2
        g2 = QGroupBox("Step 2 - Crack Quantification")
        l2 = QVBoxLayout()
        self._btn_quant = QPushButton("▶ Quantify Cracks"); self._btn_quant.clicked.connect(self._quantify)
        l2.addWidget(self._btn_quant)
        self._prog2 = QProgressBar(); self._prog2.setVisible(False); l2.addWidget(self._prog2)
        self._lbl_cnt = QLabel("Crack count: —")
        self._lbl_cnt.setStyleSheet("font-size:15px;font-weight:bold;color:#FF6600;")
        l2.addWidget(self._lbl_cnt)
        l2.addWidget(QLabel("Pipeline: binary mask → skeletonization + junction repair → branch separation → length / width / inclination"))
        g2.setLayout(l2); pl.addWidget(g2)

        # Results table
        g3 = QGroupBox("Quantification Results")
        l3 = QVBoxLayout()
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Crack ID", "Length (px)", "Mean Width (px)", "Inclination (deg)"])
        result_header = HoverSortHeader(Qt.Horizontal, self._table)
        result_header.sort_requested.connect(self._sort_results_table)
        self._table.setHorizontalHeader(result_header)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSortingEnabled(False)
        self._table.horizontalHeader().setToolTip("Click a column header to sort; hover to preview the next order.")
        self._table.verticalHeader().setDefaultSectionSize(28)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        l3.addWidget(self._table)
        g3.setLayout(l3); pl.addWidget(g3)

        pl.addStretch()
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        # Global font styling applies to the interface, not image annotations.
        self.setStyleSheet("""
            QMainWindow, QWidget { font-family: 'Microsoft YaHei'; font-size: 12pt; }
            QGroupBox { font-size: 12pt; font-weight: bold; margin-top: 10px; padding-top: 8px; }
            QPushButton { font-size: 12pt; min-height: 32px; padding: 3px 8px; }
            QTableWidget { font-size: 11pt; gridline-color: #d0d0d0; }
            QHeaderView::section { font-size: 11pt; font-weight: bold; padding: 6px; }
            QComboBox, QDoubleSpinBox { font-size: 11pt; min-height: 26px; }
            QLabel { font-size: 11pt; }
            QTabBar::tab { font-size: 11pt; padding: 7px 12px; }
        """)

    # ======== Actions ========
    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Crack Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)")
        if not path: return
        # A single image also enters the task list so navigation and export use one data model.
        self._append_batch_paths([path])

    def _open_batch(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Import Crack Images", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)")
        if not paths: return
        self._append_batch_paths(paths)

    def _set_batch_paths(self, paths):
        self.batch_items = [{
            'path': p, 'orig': None, 'binary': None, 'prob': None,
            'results': None, 'skeleton': None, 'overlay': None,
        } for p in paths]
        self._updating_batch_combo = True
        self._combo_batch.clear()
        for i, p in enumerate(paths):
            self._combo_batch.addItem(f"{i+1:03d}  {os.path.basename(p)}")
        self._updating_batch_combo = False
        self._combo_batch.setEnabled(bool(paths))
        self._btn_batch_run.setEnabled(bool(paths))
        self._act_export_batch.setEnabled(False)
        self._select_item(0)

    def _append_batch_paths(self, paths):
        """Append images without clearing existing results; duplicate paths are ignored."""
        existing = {os.path.normcase(os.path.abspath(item['path'])) for item in self.batch_items}
        new_paths = []
        for path in paths:
            key = os.path.normcase(os.path.abspath(path))
            if key not in existing:
                existing.add(key); new_paths.append(path)
        if not new_paths:
            return
        old_index = self.current_index if self.current_index >= 0 else 0
        self.batch_items.extend({
            'path': p, 'orig': None, 'binary': None, 'prob': None,
            'results': None, 'skeleton': None, 'overlay': None,
        } for p in new_paths)
        self._updating_batch_combo = True
        self._combo_batch.clear()
        for i, item in enumerate(self.batch_items):
            self._combo_batch.addItem(f"{i+1:03d}  {os.path.basename(item['path'])}")
        self._updating_batch_combo = False
        self._combo_batch.setEnabled(True)
        self._select_item(min(old_index, len(self.batch_items)-1))
        self._btn_batch_run.setEnabled(True)

    def _remove_current_item(self):
        if self.current_index < 0 or self.current_index >= len(self.batch_items):
            return
        item = self.batch_items[self.current_index]
        answer = QMessageBox.question(self, "Remove Current", f"Remove this image and its processing cache?\n{os.path.basename(item['path'])}",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        del self.batch_items[self.current_index]
        if not self.batch_items:
            self.current_index = -1
            self._combo_batch.clear(); self._combo_batch.setEnabled(False)
            self._clear_view(self._gv_orig); self._clear_view(self._gv_prob); self._clear_view(self._gv_seg)
            self._clear_view(self._gv_skel); self._clear_view(self._gv_quant)
            self._table.setRowCount(0); self._lbl_batch.setText("Current task: no image imported")
            self._btn_batch_run.setEnabled(False); self._act_save.setEnabled(False); self._act_export_batch.setEnabled(False)
            return
        self._updating_batch_combo = True
        self._combo_batch.clear()
        for i, entry in enumerate(self.batch_items):
            self._combo_batch.addItem(f"{i+1:03d}  {os.path.basename(entry['path'])}")
        self._updating_batch_combo = False
        self._select_item(min(self.current_index, len(self.batch_items)-1))

    def _clear_batch_cache(self):
        if not self.batch_items:
            return
        answer = QMessageBox.question(self, "Clear Cache", "Remove all images and processing results from the current task?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.batch_items = []; self.current_index = -1
            self._combo_batch.clear(); self._combo_batch.setEnabled(False)
            for gv in [self._gv_orig, self._gv_prob, self._gv_seg, self._gv_skel, self._gv_quant]: self._clear_view(gv)
            self._table.setRowCount(0); self._lbl_batch.setText("Current task: no image imported")
            self._btn_batch_run.setEnabled(False); self._act_save.setEnabled(False); self._act_export_batch.setEnabled(False)

    def _select_batch_item(self, index):
        if self._updating_batch_combo or index < 0: return
        self._select_item(index)

    def _select_item(self, index):
        if index < 0 or index >= len(self.batch_items): return
        self.current_index = index
        item = self.batch_items[index]
        self.img_path = item['path']
        if item['orig'] is None:
            try:
                item['orig'] = np.array(Image.open(item['path']).convert('L'))
            except Exception as e:
                QMessageBox.critical(self, "Open Failed", f"Unable to read image:\n{item['path']}\n\n{e}")
                return
        self.orig_arr = item['orig']; self.bin_mask = item['binary']; self.prob_map = item['prob']
        self.quant_rst = item['results']; self.skel_arr = item['skeleton']; self.over_arr = item['overlay']
        self._show_gray(self._gv_orig, self.orig_arr, "Original Image")
        if self.prob_map is not None:
            self._show_heatmap(self._gv_prob, self.prob_map, "BCM-UNet Probability Map")
        else:
            self._clear_view(self._gv_prob)
        if self.bin_mask is not None:
            self._show_gray(self._gv_seg, self.bin_mask, "Binary Crack Segmentation")
        else:
            self._clear_view(self._gv_seg)
        if self.skel_arr is not None:
            self._show_gray(self._gv_skel, self.skel_arr, "Crack Skeleton")
        else:
            self._clear_view(self._gv_skel)
        self._fill_results_table()
        if self.over_arr is not None:
            self._show_overlay()
        else:
            self._clear_view(self._gv_quant)
        done = "Complete" if self.quant_rst is not None else "Pending"
        self._lbl_batch.setText(f"Current image: {index+1}/{len(self.batch_items)}  |  {done}\n{os.path.basename(item['path'])}")
        idle = not self._batch_active
        self._act_seg.setEnabled(idle); self._btn_seg.setEnabled(idle)
        self._btn_batch_run.setEnabled(idle)
        has_binary = self.bin_mask is not None
        self._act_quant.setEnabled(idle and has_binary); self._btn_quant.setEnabled(idle and has_binary)
        self._act_save.setEnabled(self.quant_rst is not None)
        can_nav = len(self.batch_items) > 1
        self._act_prev.setEnabled(can_nav and index > 0); self._btn_prev.setEnabled(can_nav and index > 0)
        self._act_next.setEnabled(can_nav and index < len(self.batch_items)-1); self._btn_next.setEnabled(can_nav and index < len(self.batch_items)-1)
        self.statusBar().showMessage(f"Loaded image {index+1}/{len(self.batch_items)}: {os.path.basename(item['path'])}")

    def _previous_image(self):
        if self.current_index > 0:
            self._combo_batch.setCurrentIndex(self.current_index - 1)

    def _next_image(self):
        if self.current_index + 1 < len(self.batch_items):
            self._combo_batch.setCurrentIndex(self.current_index + 1)

    def _clear_view(self, gv):
        gv.setScene(QGraphicsScene())

    def _segment(self):
        if not self.img_path: return
        self._act_seg.setEnabled(False); self._btn_seg.setEnabled(False)
        self._prog1.setVisible(True); self._prog1.setRange(0,0)
        self._th = SegThread(self.img_path, self._spin_th.value())
        self._th.done.connect(self._on_seg); self._th.err.connect(self._on_err)
        self._th.msg.connect(lambda s: self.statusBar().showMessage(s))
        self._th.start()

    def _on_seg(self, binary, prob):
        self.bin_mask = binary; self.prob_map = prob
        if self.current_index >= 0:
            item = self.batch_items[self.current_index]
            item['binary'] = binary; item['prob'] = prob
            item['results'] = None; item['skeleton'] = None; item['overlay'] = None
        self._prog1.setVisible(False)
        self._act_seg.setEnabled(True); self._btn_seg.setEnabled(True)
        self._act_quant.setEnabled(True); self._btn_quant.setEnabled(True)
        n = np.sum(binary)
        # Display probability map and binary segmentation.
        self._show_heatmap(self._gv_prob, prob, "BCM-UNet Probability Map")
        self._show_gray(self._gv_seg, binary, f"Binary Segmentation ({n} crack pixels)")
        self.statusBar().showMessage(f"Inference complete: {n} crack pixels (threshold={self._spin_th.value()})")
        gc.collect()

    def _quantify(self):
        if self.bin_mask is None: return
        self._act_quant.setEnabled(False); self._btn_quant.setEnabled(False)
        self._prog2.setVisible(True); self._prog2.setRange(0,0)
        self._qth = QuantThread(self.bin_mask, self._pixel_resolution())
        self._qth.done.connect(self._on_quant); self._qth.err.connect(self._on_err)
        self._qth.msg.connect(lambda s: self.statusBar().showMessage(s))
        self._qth.start()

    def _on_quant(self, results, skeleton, overlay):
        self.quant_rst = results; self.skel_arr = skeleton; self.over_arr = overlay
        if self.current_index >= 0:
            item = self.batch_items[self.current_index]
            item['results'] = results; item['skeleton'] = skeleton; item['overlay'] = overlay
        self._prog2.setVisible(False)
        self._act_quant.setEnabled(True); self._btn_quant.setEnabled(True)
        self._act_save.setEnabled(True)
        n = len(results)
        self._lbl_cnt.setText(f"Crack count: {n}")
        self._fill_results_table()
        self._show_gray(self._gv_skel, skeleton, f"Skeleton - {n} cracks")
        self._show_overlay()
        self.statusBar().showMessage(f"Quantification complete: {n} cracks")

    def _pixel_resolution(self):
        return self._spin_scale.value() if self._check_scale.isChecked() else 1.0

    def _fill_results_table(self):
        results = self.quant_rst or []
        self._table.setRowCount(len(results))
        for i, r in enumerate(results):
            vals = [
                (str(r['id']), r['id']),
                (f"{r['length_px']:.1f}", r['length_px']),
                (f"{r['width_px']:.1f}", r['width_px']),
                (f"{r['angle_deg']:.1f}", r['angle_deg']),
            ]
            for j, (text, value) in enumerate(vals):
                item = NumericTableWidgetItem(text, value); item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(i, j, item)
        self._lbl_cnt.setText(f"Crack count: {len(results)}" if self.quant_rst is not None else "Crack count: —")

    def _sort_results_table(self, column, order):
        if self._table.rowCount() > 1:
            self._table.sortItems(column, order)

    def _fit_current_tab(self, index):
        views = [None, self._gv_skel, self._gv_quant]
        if 0 <= index < len(views) and views[index] is not None:
            QTimer.singleShot(0, views[index].fit_view)

    def _run_batch(self):
        if not self.batch_items: return
        unfinished = [item['path'] for item in self.batch_items if item['results'] is None]
        if not unfinished:
            QMessageBox.information(self, "Nothing to Process", "All images in the current batch are already quantified.")
            return
        answer = QMessageBox.question(
            self, "Batch Processing", f"The application will segment and quantify {len(unfinished)} pending images in sequence.\n"
            "You may view the current image during processing, but do not start another operation. Start now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer != QMessageBox.Yes: return
        self._batch_active = True
        self._set_processing(False)
        self._prog1.setVisible(True); self._prog1.setRange(0, len(unfinished)); self._prog1.setValue(0)
        self._batch_done_count = 0
        self._batch_paths = unfinished
        self._bth = BatchThread(unfinished, self._spin_th.value(), self._pixel_resolution())
        self._bth.item_done.connect(self._on_batch_item)
        self._bth.msg.connect(self.statusBar().showMessage)
        self._bth.err.connect(self._on_batch_err)
        self._bth.done.connect(self._on_batch_done)
        self._bth.start()

    def _on_batch_item(self, order, binary, prob, results, skeleton, overlay):
        path = self._batch_paths[order]
        item_index = next((i for i, item in enumerate(self.batch_items) if item['path'] == path), -1)
        if item_index >= 0:
            item = self.batch_items[item_index]
            item['binary'] = binary; item['prob'] = prob; item['results'] = results
            item['skeleton'] = skeleton; item['overlay'] = overlay
        self._batch_done_count += 1
        self._prog1.setValue(self._batch_done_count)
        self._act_export_batch.setEnabled(True)
        if item_index == self.current_index:
            self._select_item(item_index)

    def _on_batch_done(self, count):
        self._prog1.setVisible(False)
        self._batch_active = False
        self._set_processing(True)
        self._act_export_batch.setEnabled(any(item['results'] is not None for item in self.batch_items))
        QMessageBox.information(self, "Batch Processing Complete", f"Segmentation and quantification finished for {count} images.\nUse the Image Tasks list to review results or choose Export Batch.")

    def _on_batch_err(self, msg):
        self._prog1.setVisible(False)
        self._batch_active = False
        self._set_processing(True)
        QMessageBox.critical(self, "Batch Processing Error", msg)

    def _set_processing(self, enabled):
        self._btn_seg.setEnabled(enabled); self._act_seg.setEnabled(enabled)
        self._btn_quant.setEnabled(enabled and self.bin_mask is not None); self._act_quant.setEnabled(enabled and self.bin_mask is not None)
        self._btn_batch_run.setEnabled(enabled and bool(self.batch_items))
        self._btn_open_batch.setEnabled(enabled); self._combo_batch.setEnabled(enabled and bool(self.batch_items))

    def _export_options(self, title="Select Export Content"):
        dialog = QDialog(self); dialog.setWindowTitle(title); dialog.setMinimumWidth(360)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select the result types to export:"))
        checks = {}
        choices = [
            ('original', 'Original Image'), ('probability', 'Probability Map'),
            ('binary', 'Binary Crack Segmentation'), ('skeleton', 'Crack Skeleton'),
            ('overlay', 'Annotated Quantification Image (ID, length, width, inclination)'),
            ('individual', 'Individual Crack Images (one image per crack)'),
            ('csv', 'Quantification CSV Table'),
        ]
        for key, label in choices:
            cb = QCheckBox(label); cb.setChecked(True); layout.addWidget(cb); checks[key] = cb
        note = QLabel("Length and width are exported in px by default. Enable physical scale conversion to add μm fields to the CSV.")
        note.setWordWrap(True); layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return None
        return {key: cb.isChecked() for key, cb in checks.items()}

    # Candidate fonts for the annotated export images, in order of preference.
    _EXPORT_FONTS = (
        'DejaVuSans.ttf',
        'arial.ttf',
        'segoeui.ttf',
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/segoeui.ttf',
    )

    def _font_for_export(self, size=16):
        """Return a TrueType font for annotation, falling back to the default."""
        for name in self._EXPORT_FONTS:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _result_label(result):
        # Two-line labels reduce the chance of annotations leaving a narrow canvas.
        return (f"#{result['id']}  L:{result['length_px']:.0f}px\n"
                f"W:{result['width_px']:.1f}px  {result['angle_deg']:.0f}°")

    def _label_font_and_size(self, draw, label, canvas_width, canvas_height, preferred_size):
        """Reduce the font size as needed for the label to fit the canvas."""
        size = preferred_size
        while size > 7:
            font = self._font_for_export(size)
            bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=1)
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if width <= max(1, canvas_width - 12) and height <= max(1, canvas_height - 8):
                return font, width, height
            size -= 1
        font = self._font_for_export(7)
        bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=1)
        return font, bbox[2] - bbox[0], bbox[3] - bbox[1]

    @staticmethod
    def _clamp_label_position(x, y, width, height, canvas_width, canvas_height, pad=4):
        """Clamp annotation boxes to the image boundary."""
        max_x = max(pad, canvas_width - width - pad)
        max_y = max(pad, canvas_height - height - pad)
        return max(pad, min(int(x), max_x)), max(pad, min(int(y), max_y))

    def _annotated_array(self, item):
        """Draw quantification labels directly into exported images."""
        if item.get('overlay') is None:
            return None
        arr = np.asarray(item['overlay']).copy().astype(np.uint8)
        image = Image.fromarray(arr, mode='RGB'); draw = ImageDraw.Draw(image)
        colors = [(230,57,70),(46,125,50),(25,118,210),(245,124,0),(123,31,162)]
        canvas_h, canvas_w = arr.shape[:2]
        occupied = []
        offsets = [(10,-32),(10,12),(-190,-32),(-190,12),(10,-62),(-190,40)]
        for i, result in enumerate(item.get('results') or []):
            path = result.get('path') or []
            if not path: continue
            mid = path[len(path)//2]; base_x, base_y = int(mid[1]), int(mid[0])
            label = self._result_label(result)
            font, tw, th = self._label_font_and_size(
                draw, label, canvas_w, canvas_h, max(12, min(24, int(max(arr.shape[:2]) / 70)))
            )
            candidate_pos = None
            for dx, dy in offsets:
                x, y = self._clamp_label_position(base_x + dx, base_y + dy, tw, th, canvas_w, canvas_h)
                box = (x-4, y-3, x+tw+4, y+th+3)
                if not any(box[0] < old[2] and box[2] > old[0] and box[1] < old[3] and box[3] > old[1] for old in occupied):
                    candidate_pos = (x, y, box); occupied.append(box); break
            if candidate_pos is None:
                x, y = self._clamp_label_position(
                    base_x + offsets[i % len(offsets)][0], base_y + offsets[i % len(offsets)][1],
                    tw, th, canvas_w, canvas_h
                )
                candidate_pos = (x, y, (x-4,y-3,x+tw+4,y+th+3))
            x, y, box = candidate_pos
            draw.rectangle(box, fill=(0,0,0), outline=None)
            draw.multiline_text((x,y), label, font=font, fill=colors[i % len(colors)], spacing=1)
        return np.asarray(image)

    def _individual_array(self, item, result):
        """Generate an image highlighting one crack and draw its measurement label."""
        shape = np.asarray(item['overlay']).shape
        base = np.full(shape, 18, dtype=np.uint8)
        image = Image.fromarray(base, mode='RGB'); draw = ImageDraw.Draw(image)
        path = result.get('path') or []
        color = (245,124,0)
        if path:
            draw.line([(int(x),int(y)) for y,x in path], fill=color, width=max(2, int(max(shape[:2])/350)))
            mid = path[len(path)//2]; label = self._result_label(result)
            canvas_h, canvas_w = shape[:2]
            font, tw, th = self._label_font_and_size(
                draw, label, canvas_w, canvas_h, max(14, min(26, int(max(shape[:2])/60)))
            )
            x, y = self._clamp_label_position(int(mid[1])+10, int(mid[0])-32, tw, th, canvas_w, canvas_h)
            draw.rectangle((x-4,y-3,x+tw+4,y+th+3), fill=(0,0,0))
            draw.multiline_text((x,y), label, font=font, fill=color, spacing=1)
        return np.asarray(image)

    def _write_one_result(self, directory, item, prefix, options):
        os.makedirs(directory, exist_ok=True)
        include_physical = self._check_scale.isChecked()
        csv_p = None
        if options.get('original') and item.get('orig') is not None:
            Image.fromarray(item['orig'].astype(np.uint8)).save(os.path.join(directory, f'{prefix}_original.png'))
        if options.get('probability') and item.get('prob') is not None:
            Image.fromarray((np.clip(item['prob'],0,1)*255).astype(np.uint8)).save(os.path.join(directory, f'{prefix}_probability.png'))
        if options.get('binary') and item.get('binary') is not None:
            Image.fromarray(item['binary'].astype(np.uint8)*255).save(os.path.join(directory, f'{prefix}_binary.png'))
        if options.get('skeleton') and item.get('skeleton') is not None:
            Image.fromarray(item['skeleton'].astype(np.uint8)*255).save(os.path.join(directory, f'{prefix}_skeleton.png'))
        if options.get('overlay'):
            annotated = self._annotated_array(item)
            if annotated is not None: Image.fromarray(annotated).save(os.path.join(directory, f'{prefix}_overlay_annotated.png'))
        if options.get('individual'):
            for i, result in enumerate(item.get('results') or [], 1):
                Image.fromarray(self._individual_array(item, result)).save(os.path.join(directory, f'{prefix}_crack_{i:03d}.png'))
        if options.get('csv'):
            csv_p = os.path.join(directory, f'{prefix}_crack_results.csv')
            with open(csv_p, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f); header = ['Crack ID', 'Length_px', 'Mean_Width_px', 'Inclination_deg']
                if include_physical: header += ['Length_um', 'Mean_Width_um']
                w.writerow(header)
                for r in item.get('results') or []:
                    row = [r['id'], f"{r['length_px']:.1f}", f"{r['width_px']:.1f}", f"{r['angle_deg']:.1f}"]
                    if include_physical: row += [f"{r['length_um']:.3f}", f"{r['width_um']:.3f}"]
                    w.writerow(row)
        return csv_p

    def _save(self):
        if not self.quant_rst: return
        options = self._export_options("Select Current Image Export Content")
        if options is None: return
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not d: return
        prefix = os.path.splitext(os.path.basename(self.img_path))[0]
        self._write_one_result(d, self.batch_items[self.current_index], prefix, options)
        QMessageBox.information(self, "Save Complete", f"Results saved to:\n{d}")

    def _export_batch(self):
        completed = [item for item in self.batch_items if item['results'] is not None]
        if not completed:
            QMessageBox.warning(self, "No Results", "Run quantification for the current image or process a batch first.")
            return
        d = QFileDialog.getExistingDirectory(self, "Select Batch Output Directory")
        if not d: return
        options = self._export_options("Select Batch Export Content")
        if options is None: return
        all_rows = []
        include_physical = self._check_scale.isChecked() and options.get('csv', True)
        for i, item in enumerate(completed, 1):
            stem = os.path.splitext(os.path.basename(item['path']))[0]
            safe_prefix = f"{i:03d}_{stem}"
            subdir = os.path.join(d, safe_prefix)
            self._write_one_result(subdir, item, safe_prefix, options)
            if not options.get('csv'):
                continue
            for r in item['results']:
                row = [os.path.basename(item['path']), r['id'], r['length_px'], r['width_px'], r['angle_deg']]
                if include_physical: row += [r['length_um'], r['width_um']]
                all_rows.append(row)
        summary = os.path.join(d, 'batch_summary.csv')
        with open(summary, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            header = ['Image File', 'Crack ID', 'Length_px', 'Mean_Width_px', 'Inclination_deg']
            if include_physical: header += ['Length_um', 'Mean_Width_um']
            w.writerow(header); w.writerows(all_rows)
        QMessageBox.information(self, "Batch Export Complete", f"Results exported for {len(completed)} images.\nSummary file:\n{summary}")

    def _on_err(self, msg):
        self._prog1.setVisible(False); self._prog2.setVisible(False)
        self._act_seg.setEnabled(True); self._btn_seg.setEnabled(True)
        self._act_quant.setEnabled(True); self._btn_quant.setEnabled(True)
        self._btn_batch_run.setEnabled(bool(self.batch_items))
        QMessageBox.critical(self, "Error", msg)

    def _fit_all(self):
        for gv in [self._gv_orig,self._gv_prob,self._gv_seg,self._gv_skel,self._gv_quant]:
            if gv.scene() and gv.scene().items(): gv.fit_view()

    def _about(self):
        QMessageBox.about(self, "About CrackQuant v1.0",
            "Coal CT crack recognition and quantification\n\n"
            "Model: BCM-UNet (boundary guidance + continuity enhancement + multi-scale fusion)\n"
            "Skeletonization: thinning with junction line-fitting repair\n"
            "Quantification: direction-based branch separation with length, width and inclination")

    def _show_gray(self, gv, data, title=""):
        h, w = data.shape
        # Display thumbnail is capped at 768 px.
        MAX_DISP = 768
        if max(w,h) > MAX_DISP:
            scale = MAX_DISP / max(w,h)
            from PIL import Image as PILImg
            arr = np.array(PILImg.fromarray(data.astype(np.uint8) if data.dtype!=bool else np.where(data,255,0).astype(np.uint8))
                          .resize((int(w*scale), int(h*scale)), PILImg.NEAREST))
        else:
            arr = data.astype(np.uint8) if data.dtype!=bool else np.where(data,255,0).astype(np.uint8)
        h2, w2 = arr.shape
        qimg = QImage(arr.data, w2, h2, w2, QImage.Format_Grayscale8).copy()
        gv.setScene(QGraphicsScene()); gv.scene().addPixmap(QPixmap.fromImage(qimg))
        if title:
            t = gv.scene().addText(title, QFont("Microsoft YaHei", 12)); t.setDefaultTextColor(QColor("#666"))
        gv.fit_view()

    def _show_heatmap(self, gv, data, title=""):
        h, w = data.shape
        MAX_DISP = 768
        if max(w,h) > MAX_DISP:
            scale = MAX_DISP / max(w,h)
            from PIL import Image as PILImg
            data = np.array(PILImg.fromarray((np.clip(data,0,1)*255).astype(np.uint8))
                           .resize((int(w*scale), int(h*scale)), PILImg.BILINEAR)) / 255.0
            h, w = data.shape
        v = np.clip(data, 0, 1)
        r = np.clip((v-0.5)*4, 0, 1) + np.clip((v-0.75)*4, 0, 1)
        g = np.clip(v*4, 0, 1) - np.clip((v-0.25)*4, 0, 1) + np.clip((v-0.5)*4, 0, 1) - np.clip((v-0.75)*4, 0, 1)
        b = np.clip(1-v*4, 0, 1) - np.clip((v-0.25)*4, 0, 1)
        r = np.clip(r, 0, 1); g = np.clip(g, 0, 1); b = np.clip(b, 0, 1)
        rgb = (np.stack([r,g,b], axis=-1) * 255).astype(np.uint8).copy()
        qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888).copy()
        gv.setScene(QGraphicsScene()); gv.scene().addPixmap(QPixmap.fromImage(qimg))
        if title:
            t = gv.scene().addText(title, QFont("Microsoft YaHei", 12)); t.setDefaultTextColor(QColor("#666"))
        gv.fit_view()

    def _show_overlay(self):
        if self.over_arr is None: return
        h, w, _ = self.over_arr.shape
        annotated = self._annotated_array({**(self.batch_items[self.current_index] if self.current_index >= 0 else {}),
                                           'overlay': self.over_arr, 'results': self.quant_rst})
        if annotated is None:
            annotated = self.over_arr
        qimg = QImage(annotated.data, w, h, w*3, QImage.Format_RGB888)
        scene = QGraphicsScene(); scene.addPixmap(QPixmap.fromImage(qimg.copy()))
        self._gv_quant.setScene(scene); self._gv_quant.fit_view()
