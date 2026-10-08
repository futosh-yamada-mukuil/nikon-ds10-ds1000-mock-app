"""Desktop file workflow based on the supplied normal-mode video."""

from datetime import datetime
from html import escape
import json
from pathlib import Path
import sys

# NumPy's macOS Accelerate initialization must happen on the main thread.
# Importing it for the first time from a Qt worker caused a native bus error
# in the verified NumPy 1.26.4 / macOS environment.
import numpy
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QTimer, Signal, QStandardPaths, QUrl
from PySide6.QtGui import QPixmap, QPainter, QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGraphicsScene,
    QGraphicsView, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .domain import InferenceSettings
from .downloads import CsvDownload
from .media import MediaSource
from .reporting import CANDIDATE_COLOR, export_bundle, render_result
from .workers import AnalysisWorker, ModelWorker


def project_directory():
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        if sys.platform == "darwin" and executable.parent.parent.name == "Contents" and executable.parent.parent.parent.suffix == ".app":
            # Keep local settings outside the signed application bundle.
            return executable.parent.parent.parent.parent
        return executable.parent
    return Path(__file__).resolve().parent.parent


class Preview(QGraphicsView):
    scale_changed = Signal(float)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = self.scene().addPixmap(QPixmap())
        self.setBackgroundBrush(Qt.black)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        # Hidden scrollbars avoid fitInView triggering recursive resize events.
        # ScrollHandDrag still pans the image when it is magnified.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.auto_fit = True

    def set_image(self, image, *, reset_view=False):
        self.item.setPixmap(QPixmap.fromImage(ImageQt(image)))
        self.scene().setSceneRect(self.item.boundingRect())
        if reset_view:
            self.auto_fit = True
        if self.auto_fit:
            self.fit()

    def fitted_scale(self):
        bounds = self.item.boundingRect()
        if bounds.isEmpty():
            return None
        return min(max(1, self.viewport().width() - 4) / bounds.width(),
                   max(1, self.viewport().height() - 4) / bounds.height())

    def fit(self):
        self.auto_fit = True
        if self.item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.item.boundingRect(), Qt.KeepAspectRatio)
        self.scale_changed.emit(self.transform().m11() * 100)

    def zoom(self, factor, *, mouse_anchor=False):
        minimum = self.fitted_scale()
        if minimum is None or factor <= 0:
            return
        target = max(minimum, min(minimum * 32, self.transform().m11() * factor))
        if target <= minimum * 1.00001:
            self.fit()
            return
        self.auto_fit = False
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse if mouse_anchor else QGraphicsView.AnchorViewCenter)
        self.scale(target / self.transform().m11(), target / self.transform().m11())
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale_changed.emit(self.transform().m11() * 100)

    def zoom_in(self):
        self.zoom(1.25)

    def zoom_out(self):
        self.zoom(1 / 1.25)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        minimum = self.fitted_scale()
        if self.auto_fit or (minimum is not None and self.transform().m11() < minimum):
            self.fit()

    def wheelEvent(self, event):
        if not event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            super().wheelEvent(event)
            return
        # Horizontal/zero-delta events never change the zoom. With a deliberate
        # modifier, support both mouse wheels and vertical pixel-only trackpads.
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            self.zoom(1.2 if delta > 0 else 1 / 1.2, mouse_anchor=True)
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self, config=None, device="auto", autoload=True):
        super().__init__()
        self.setWindowTitle("精子細胞検出・分類アプリ | DS10 / DS1000 · モック")
        self.resize(1500, 940)
        self.setMinimumSize(1000, 700)
        self.base_dir = project_directory()
        self.config_path = config or self.base_dir / "config/models.local.json"
        self.device = device
        self.engine = self.loader = self.worker = self.media = None
        self.source_path = self.original_image = self.result = None
        self.frame_index = 0
        self.history = []
        self.run_settings = None
        self.run_metadata = None
        self.paused = self.close_pending = False
        self.downloads = []
        self.export_directory = Path(QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)) / "NikonMockApp"
        self.download_opener = QDesktopServices.openUrl
        self._build_ui()
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.advance_preview)
        self._read_config()
        self.set_busy(False)
        if autoload and self.detector_path.text() and self.classifier_path.text():
            QTimer.singleShot(0, self.load_models)

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f6f8; color: #213044; }
            QGroupBox { border: 1px solid #d4dce5; border-radius: 5px;
                margin-top: 12px; padding: 12px 8px 7px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; }
            QPushButton { background: white; border: 1px solid #c9d3df;
                border-radius: 4px; padding: 7px 9px; min-height: 18px; }
            QPushButton:hover { background: #e9f1fa; }
            QPushButton:disabled { color: #9aabba; background: #ecf0f3; }
            QPushButton#primary { background: #1769aa; color: white; font-weight: 600; }
            QPushButton#primary:disabled { background: #9ab5cd; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: white;
                border: 1px solid #ccd5df; border-radius: 3px; padding: 4px; }
            QTableWidget, QPlainTextEdit { background: white; border: 1px solid #d5dde6; }
            QHeaderView::section { background: #eaf0f6; padding: 5px; border: 0; }
            QProgressBar { border: 1px solid #cdd7e1; text-align: center; min-height: 17px; }
            QProgressBar::chunk { background: #267bbe; }
        """)
        split = QSplitter(Qt.Horizontal)
        self.setCentralWidget(split)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        left = QVBoxLayout(panel)
        title = QLabel("設定パネル")
        title.setStyleSheet("font-size: 20px; font-weight: 700; padding: 4px;")
        left.addWidget(title)
        self.model_status = QLabel("モデル未読み込み")
        self.model_status.setWordWrap(True)
        self.model_status.setStyleSheet("background: #e4ebf2; padding: 10px; border-radius: 4px;")
        left.addWidget(self.model_status)
        self.model_button = QPushButton("モデルを読み込む")
        self.model_button.clicked.connect(self.load_models)
        left.addWidget(self.model_button)
        source_group = QGroupBox("入力ソース選択")
        source_layout = QVBoxLayout(source_group)
        self.source_button = QPushButton("画像 / 動画ファイルを選択")
        self.source_button.clicked.connect(self.choose_source)
        source_layout.addWidget(self.source_button)
        self.source_name = QLabel("ファイル未選択")
        self.source_name.setWordWrap(True)
        source_layout.addWidget(self.source_name)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.machine = QComboBox()
        self.machine.addItems(["DS10", "DS1000"])
        self.machine.setMinimumContentsLength(8)
        self.machine.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.machine.setMinimumWidth(160)
        self.machine.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form.addRow("機種", self.machine)
        self.stride = QSpinBox()
        self.stride.setRange(1, 1000)
        self.stride.setValue(10)
        self.stride.setSuffix(" フレームごと")
        form.addRow("動画の解析間隔", self.stride)
        source_layout.addLayout(form)
        source_note = QLabel("画像・動画ファイル対応\nカメラ / DLL / SAM連携は次の開発範囲")
        source_note.setStyleSheet("color: #617488; font-size: 11px;")
        source_layout.addWidget(source_note)
        left.addWidget(source_group)
        controls = QGroupBox("処理操作")
        control_layout = QVBoxLayout(controls)
        self.start_button = QPushButton("処理開始")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start_analysis)
        self.pause_button = QPushButton("一時停止")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button = QPushButton("処理停止")
        self.stop_button.clicked.connect(self.stop_analysis)
        for button in (self.start_button, self.pause_button, self.stop_button):
            control_layout.addWidget(button)
        left.addWidget(controls)
        setting_group = QGroupBox("検出・分類設定")
        setting_form = QFormLayout(setting_group)
        setting_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.mode = QComboBox()
        self.mode.addItem("物体検知 + 細胞分類", "detection_classification")
        self.mode.addItem("物体検知のみ", "detection_only")
        self.mode.addItem("細胞分類のみ（画像全体）", "classification_only")
        setting_form.addRow("モード", self.mode)
        self.det_threshold = QDoubleSpinBox()
        self.det_threshold.setRange(0, 1)
        self.det_threshold.setDecimals(2)
        self.det_threshold.setSingleStep(.01)
        self.det_threshold.setValue(.5)
        setting_form.addRow("検出しきい値", self.det_threshold)
        self.class_threshold = QDoubleSpinBox()
        self.class_threshold.setRange(0, 100)
        self.class_threshold.setDecimals(1)
        self.class_threshold.setSingleStep(.1)
        self.class_threshold.setValue(.5)
        setting_form.addRow("分類しきい値", self.class_threshold)
        self.threshold_note = QLabel("赤枠 = class1スコアが分類しきい値を超過\n設定変更は次回の解析に適用されます\n学習ラベルの意味は確認待ちです")
        self.threshold_note.setWordWrap(True)
        self.threshold_note.setStyleSheet("color: #6b7c8f; font-size: 11px;")
        setting_form.addRow(self.threshold_note)
        self.calibration_button = QPushButton("DS1000の改善設定を適用")
        self.calibration_button.setCheckable(True)
        self.calibration_button.setEnabled(False)
        self.calibration_button.setToolTip("DS1000の正解画像で学習した分類補正です。比較用56枚で誤検出と見逃しが減少しました。初期設定の0.5／0.5には適用されません。")
        self.calibration_button.toggled.connect(self.toggle_calibration)
        setting_form.addRow(self.calibration_button)
        left.addWidget(setting_group)
        self.export_button = QPushButton("検索結果CSVをダウンロード")
        self.export_button.clicked.connect(self.save_results)
        left.addWidget(self.export_button)
        self.export_feedback = QLabel("解析後にCSVをダウンロードできます。")
        self.export_feedback.setWordWrap(True)
        self.export_feedback.setOpenExternalLinks(True)
        self.export_feedback.setStyleSheet("color: #617488; font-size: 11px;")
        left.addWidget(self.export_feedback)
        model_group = QGroupBox("モデルファイル設定")
        model_layout = QVBoxLayout(model_group)
        self.detector_path = QLineEdit()
        self.classifier_path = QLineEdit()
        self.model_path_buttons = []
        for label, line in (("物体検知モデル", self.detector_path), ("細胞分類モデル", self.classifier_path)):
            line.textChanged.connect(self.invalidate_models)
            model_layout.addWidget(QLabel(label))
            row = QHBoxLayout()
            row.addWidget(line)
            button = QPushButton("選択")
            self.model_path_buttons.append(button)
            button.clicked.connect(lambda checked=False, target=line: self.choose_model(target))
            row.addWidget(button)
            model_layout.addLayout(row)
        self.save_config_button = QPushButton("このモデル設定を保存")
        self.save_config_button.clicked.connect(self.save_config)
        model_layout.addWidget(self.save_config_button)
        left.addWidget(model_group)
        left.addStretch()
        scroll.setWidget(panel)
        scroll.setMinimumWidth(320)
        split.addWidget(scroll)
        right = QWidget()
        layout = QVBoxLayout(right)
        heading = QHBoxLayout()
        title = QLabel("動画・画像処理エリア")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        heading.addWidget(title)
        heading.addStretch()
        layout.addLayout(heading)
        self.preview = Preview()
        zoom_controls = QHBoxLayout()
        zoom_controls.addWidget(QLabel("表示倍率"))
        for text, callback in (("−", self.preview.zoom_out), ("＋", self.preview.zoom_in), ("全体表示", self.preview.fit)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            zoom_controls.addWidget(button)
        self.zoom_label = QLabel("—")
        self.zoom_label.setMinimumWidth(60)
        self.preview.scale_changed.connect(lambda value: self.zoom_label.setText(f"{value:.0f}%"))
        zoom_controls.addWidget(self.zoom_label)
        zoom_controls.addStretch()
        zoom_controls.addWidget(QLabel("Ctrl / ⌘ + ホイールで拡大・縮小"))
        layout.addLayout(zoom_controls)
        layout.addWidget(self.preview, 1)
        timeline = QHBoxLayout()
        self.play_button = QPushButton("▶ プレビュー再生")
        self.play_button.clicked.connect(self.toggle_preview)
        timeline.addWidget(self.play_button)
        self.seek = QSlider(Qt.Horizontal)
        self.seek.valueChanged.connect(self.seek_frame)
        timeline.addWidget(self.seek, 1)
        self.position = QLabel("0 / 0")
        timeline.addWidget(self.position)
        layout.addLayout(timeline)
        layout.addWidget(self._build_result_cards())
        # Retain diagnostics for verification without occupying preview space.
        self.details_panel = QWidget(right)
        details_layout = QVBoxLayout(self.details_panel)
        self.summary = QLabel("検出枠: —　閾値超え表示: —", self.details_panel)
        self.media_info = QLabel("ファイルを選択してください。処理開始後に実際のモデル結果を表示します。")
        details_layout.addWidget(self.media_info)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        details_layout.addWidget(self.progress_bar)
        lower = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "検出スコア", "class1スコア", "しきい値超過"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMaximumHeight(180)
        lower.addWidget(self.table)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setMaximumHeight(180)
        lower.addWidget(self.log)
        details_layout.addWidget(lower)
        self.details_panel.hide()
        split.addWidget(right)
        split.setSizes([380, 1120])
        self.mode.currentIndexChanged.connect(self.refresh_pending_counts)
        self.det_threshold.valueChanged.connect(self.refresh_pending_counts)
        self.class_threshold.valueChanged.connect(self.refresh_pending_counts)
        self.machine.currentIndexChanged.connect(self.sync_calibration_controls)
        self.mode.currentIndexChanged.connect(self.sync_calibration_controls)
        self.update_result_counts()
        self.statusBar().showMessage("ファイル入力モック · 推論前の件数は未測定")

    def _build_result_cards(self):
        panel = QWidget()
        panel.setObjectName("resultCards")
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("解析結果")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        header.addWidget(title)
        self.result_state = QLabel("解析待ち")
        self.result_state.setStyleSheet("background: #e4ebf2; color: #526578; border-radius: 4px; padding: 3px 8px; font-size: 11px;")
        header.addWidget(self.result_state)
        header.addStretch()
        column.addLayout(header)
        cards = QHBoxLayout()
        cards.setSpacing(12)
        for name, text, accent, background, border in (
            ("detection", "物体検知した数", "#1769aa", "#ffffff", "#d4dce5"),
            ("candidate", "分類しきい値を超えた数", "#bd3d3d", "#fff7f6", "#f0d5d2"),
        ):
            card = QWidget()
            card.setObjectName(f"{name}Card")
            card.setStyleSheet(f"QWidget#{name}Card {{ background: {background}; border: 1px solid {border}; border-left: 3px solid {accent}; border-radius: 8px; }} QLabel {{ background: transparent; border: none; }}")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 10, 14, 10)
            card_layout.setSpacing(4)
            label = QLabel(text)
            label.setWordWrap(True)
            label.setStyleSheet(f"color: {accent}; font-size: 13px; font-weight: 600;")
            card_layout.addWidget(label)
            number_row = QHBoxLayout()
            number_row.setSpacing(6)
            value = QLabel("—")
            value.setStyleSheet(f"color: {accent}; font-size: 32px; font-weight: 700;")
            number_row.addWidget(value)
            unit = QLabel("件")
            unit.setStyleSheet("color: #617488; font-size: 12px; padding-bottom: 5px;")
            number_row.addWidget(unit, 0, Qt.AlignBottom)
            number_row.addStretch()
            card_layout.addLayout(number_row)
            note = QLabel()
            note.setWordWrap(True)
            note.setStyleSheet("color: #617488; font-size: 11px;")
            card_layout.addWidget(note)
            setattr(self, f"{name}_count", value)
            setattr(self, f"{name}_unit", unit)
            setattr(self, f"{name}_note", note)
            cards.addWidget(card, 1)
        column.addLayout(cards)
        self.result_context = QLabel()
        self.result_context.setWordWrap(True)
        self.result_context.setStyleSheet("color: #617488; font-size: 11px;")
        column.addWidget(self.result_context)
        return panel

    def update_result_counts(self, state="waiting"):
        settings = self.run_settings if self.run_settings and (self.result is not None or state != "waiting") else self.current_settings()
        detected = self.result.detection_count if self.result is not None and settings.mode != "classification_only" else None
        candidates = self.result.candidate_count if self.result is not None and settings.mode != "detection_only" else None
        for name, count in (("detection", detected), ("candidate", candidates)):
            getattr(self, f"{name}_count").setText("—" if count is None else f"{count:,}")
            getattr(self, f"{name}_unit").setVisible(count is not None)
        self.detection_note.setText("分類のみのため、物体検知は行いません" if settings.mode == "classification_only" else f"検出しきい値 {settings.detection_threshold:.2f} 以上")
        self.candidate_note.setText("物体検知のみのため、分類は行いません" if settings.mode == "detection_only" else f"分類スコアが {settings.classification_threshold:.1f} を超えた数 · 赤枠で表示")
        self.result_state.setText({"waiting": "解析待ち", "analyzing": "解析中", "completed": "解析完了", "paused": "一時停止", "stopping": "停止処理中", "stopped": "停止", "error": "解析エラー"}[state])
        if self.result is None:
            context = "結果はまだありません。画像・動画を選び、「処理開始」を押してください。"
            if state == "analyzing":
                context = "解析しています。フレームの処理が終わると件数を表示します。"
            elif state in ("stopped", "error"):
                context = "解析結果はありません。画面下部の状態を確認してください。"
            elif self.media and self.media.is_video:
                context = f"フレーム {self.frame_index + 1} は未解析です。このフレームを解析すると件数を表示します。"
        else:
            context = (f"フレーム {self.frame_index + 1} / {self.media.frame_count} の結果です（動画全体の累計ではありません）。"
                       if self.media and self.media.is_video else "表示中の画像の解析結果です。")
            if settings.mode == "classification_only":
                context += " 画像全体を1件として分類しています。"
            if self.result.skipped:
                context += f" 検出数には分類対象外の {self.result.skipped} 件を含みます。"
        self.result_context.setText(context)

    def refresh_pending_counts(self, *args):
        if self.result is None and not (self.loader or self.worker):
            self.update_result_counts()

    def message(self, text):
        self.log.appendPlainText(f"[{datetime.now():%H:%M:%S}] {text}")

    def _read_config(self):
        try:
            data = json.loads(Path(self.config_path).read_text(encoding="utf-8"))
            for key, line in (("detection", self.detector_path), ("classification", self.classifier_path)):
                path = Path(data[key]["path"]).expanduser()
                if not path.is_absolute():
                    path = self.base_dir / path
                line.setText(str(path))
        except (OSError, ValueError, KeyError, TypeError):
            self.message("モデル設定を選択してください。設定なしで推論は開始しません。")

    def choose_model(self, line):
        path, _ = QFileDialog.getOpenFileName(self, "提供されたモデルファイルを選択", str(self.base_dir), "モデル (*.pt *.pth);;すべて (*)")
        if path:
            line.setText(path)

    def invalidate_models(self):
        self.engine = None
        self.result = None
        self.history = []
        self.run_settings = self.run_metadata = None
        self.table.setRowCount(0)
        self.summary.setText("検出枠: —　閾値超え表示: —")
        self.update_result_counts()
        self.model_status.setText("モデル設定変更 · 読み込みが必要")
        self.refresh_preview()
        self.set_busy(bool(self.loader or self.worker))

    def save_config(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"schema_version": 1, "detection": {"path": self.detector_path.text()}, "classification": {"path": self.classifier_path.text()}}
            self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.message(f"モデル設定保存: {self.config_path}")
        except OSError as error:
            self.operation_failed(str(error))

    def load_models(self):
        if self.loader or self.worker:
            return
        self.preview_timer.stop()
        self.engine = None
        self.result = None
        self.history = []
        self.run_settings = self.run_metadata = None
        self.table.setRowCount(0)
        self.summary.setText("検出枠: —　閾値超え表示: —")
        self.update_result_counts()
        self.refresh_preview()
        self.model_status.setText("モデル読み込み中…")
        self.loader = ModelWorker(Path(self.detector_path.text()), Path(self.classifier_path.text()), self.device, self,
                                  calibration_path=self.config_path.parent / "cell-calibration.local.json")
        self.loader.message.connect(self.message)
        self.loader.ready.connect(self.models_ready)
        self.loader.failed.connect(self.operation_failed)
        self.loader.finished.connect(self.model_finished)
        self.set_busy(True)
        self.loader.start()

    def models_ready(self, engine):
        self.engine = engine
        self.model_status.setText(f"✓ 物体検知 / 細胞分類モデル読み込み済み\n実行デバイス: {engine.device}")
        self.message("提供モデルの読み込みが完了しました。")
        self.statusBar().showMessage("モデル読み込み完了 · ファイルを選んで処理開始")

    def model_finished(self):
        loader, self.loader = self.loader, None
        loader.deleteLater()
        self.set_busy(False)
        if self.close_pending:
            self.close()

    def choose_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "画像 / 動画ファイルを選択", "", "画像・動画 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.mp4 *.avi *.mov *.mkv);;すべて (*)")
        if path:
            self.open_source(Path(path))

    def open_source(self, path):
        self.preview_timer.stop()
        media = None
        try:
            media = MediaSource(path)
            image = media.read(0)
        except Exception as error:
            if media is not None:
                media.close()
            self.operation_failed(str(error))
            return
        if self.media:
            self.media.close()
        self.media, self.source_path = media, Path(path).resolve()
        self.original_image, self.result, self.frame_index = image, None, 0
        self.history = []
        self.run_settings = None
        self.run_metadata = None
        self.table.setRowCount(0)
        self.summary.setText("検出枠: —　閾値超え表示: —")
        self.source_name.setText(self.source_path.name)
        self.export_feedback.setText("解析が完了するとCSVをダウンロードできます。")
        self.source_name.setToolTip(str(self.source_path))
        self.seek.blockSignals(True)
        self.seek.setRange(0, media.frame_count - 1)
        self.seek.setValue(0)
        self.seek.blockSignals(False)
        details = f"{media.width} × {media.height}"
        if media.is_video:
            details += f" · {media.fps:.2f} FPS · {media.frame_count} フレーム"
        self.media_info.setText(details + " · 未解析")
        self.position.setText(f"1 / {media.frame_count}")
        self.progress_bar.setValue(0)
        self.update_result_counts()
        self.refresh_preview(reset_view=True)
        self.set_busy(bool(self.loader or self.worker))
        self.message(f"ファイル読み込み: {self.source_path.name}")

    def current_settings(self):
        return InferenceSettings(detection_threshold=self.det_threshold.value(), classification_threshold=self.class_threshold.value(), machine=self.machine.currentText(), mode=self.mode.currentData(), use_calibration=self.calibration_button.isChecked())

    def sync_calibration_controls(self):
        data = getattr(self.engine, "calibration", None)
        available = bool(data and "DS1000" in data["machines"] and self.machine.currentText() == "DS1000"
                         and self.mode.currentData() == "detection_classification")
        if not available and self.calibration_button.isChecked():
            self.calibration_button.setChecked(False)
        self.calibration_button.setEnabled(available and not self.loader and not self.worker)

    def toggle_calibration(self, enabled):
        if enabled:
            data = getattr(self.engine, "calibration", None)
            if not data or self.machine.currentText() != "DS1000" or self.mode.currentData() != "detection_classification":
                self.calibration_button.setChecked(False)
                return
            head = data["machines"]["DS1000"]
            self.det_threshold.setValue(head["detection_threshold"])
            self.class_threshold.setValue(head["classification_threshold"])
            self.threshold_note.setText("赤枠 = 補正後の陽性スコアが分類しきい値を超過\nDS1000の正解画像を使った分類補正\n別の画像での精度は未確認です")
            self.calibration_button.setText("DS1000改善設定を使用中（押すと標準）")
        else:
            self.det_threshold.setValue(.5)
            self.class_threshold.setValue(.5)
            self.threshold_note.setText("赤枠 = class1スコアが分類しきい値を超過\n設定変更は次回の解析に適用されます\n学習ラベルの意味は確認待ちです")
            self.calibration_button.setText("DS1000の改善設定を適用")
        self.refresh_pending_counts()

    def start_analysis(self):
        if not self.engine or not self.media or self.worker or self.loader:
            return
        self.preview_timer.stop()
        self.play_button.setText("▶ プレビュー再生")
        self.run_settings = self.current_settings()
        self.run_metadata = {"is_video": self.media.is_video, "fps": self.media.fps,
                             "frame_count": self.media.frame_count,
                             "stride": self.stride.value() if self.media.is_video else 1,
                             "start_frame": self.frame_index, "stop_reason": "running"}
        self.history = []
        self.result = None
        self.export_feedback.setText("解析が完了するとCSVをダウンロードできます。")
        self.table.setRowCount(0)
        self.summary.setText("解析中 · 検出枠: —　閾値超え表示: —")
        self.update_result_counts("analyzing")
        self.refresh_preview()
        self.paused = False
        self.pause_button.setText("一時停止")
        self.progress_bar.setValue(0)
        self.worker = AnalysisWorker(self.engine, self.source_path, self.run_settings, self.frame_index, self.stride.value(), self)
        self.worker.frame_ready.connect(self.receive_result)
        self.worker.progress.connect(lambda current, total: self.progress_bar.setValue(round(100 * current / total)))
        self.worker.failed.connect(self.operation_failed)
        self.worker.completed.connect(self.analysis_completed)
        self.worker.finished.connect(self.analysis_finished)
        self.set_busy(True)
        self.message(f"解析開始: {self.run_settings.machine} / {self.mode.currentText()} / 検出 {self.run_settings.detection_threshold} / 分類 {self.run_settings.classification_threshold}")
        self.worker.start()

    def analysis_completed(self, cancelled):
        self.run_metadata["stop_reason"] = "cancelled" if cancelled else "completed"
        self.update_result_counts("stopped" if cancelled else "completed")
        self.message("停止しました。保存対象は処理済みフレームです。" if cancelled else "解析が完了しました。")
        self.statusBar().showMessage(f"{'停止' if cancelled else '解析完了'} · 検出しきい値 {self.run_settings.detection_threshold} / 分類しきい値 {self.run_settings.classification_threshold}")

    def receive_result(self, index, image, result):
        self.frame_index, self.original_image, self.result = index, image, result
        self.history.append((index, result))
        self.seek.blockSignals(True)
        self.seek.setValue(index)
        self.seek.blockSignals(False)
        self.position.setText(f"{index + 1} / {self.media.frame_count}")
        self.summary.setText(f"検出枠: {result.detection_count}　閾値超え表示: {result.candidate_count}")
        self.update_result_counts("paused" if self.paused else "analyzing" if self.worker else "completed")
        self.media_info.setText(f"フレーム {index + 1} · 処理 {result.elapsed_seconds:.2f} 秒 · {result.device} · スキップ {result.skipped}")
        self.table.setRowCount(len(result.cells))
        for row, cell in enumerate(result.cells):
            values = (str(cell.id), "—" if cell.confidence is None else f"{cell.confidence:.4f}", "—" if cell.score is None else f"{cell.score:.3f}", "超過" if cell.candidate else "—")
            for column, text in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(text))
        self.refresh_preview()

    def refresh_preview(self, *args, reset_view=False):
        if self.original_image is None:
            return
        if self.result is not None:
            display = render_result(self.original_image, self.result, show_all=False, show_labels=False, box_color=CANDIDATE_COLOR)
        else:
            from .domain import FrameResult
            display = render_result(self.original_image, FrameResult(cells=[]))
        self.preview.set_image(display, reset_view=reset_view)

    def toggle_pause(self):
        if self.worker:
            self.paused = not self.paused
            self.worker.pause() if self.paused else self.worker.resume()
            self.pause_button.setText("再開" if self.paused else "一時停止")
            self.update_result_counts("paused" if self.paused else "analyzing")
            self.message("一時停止要求（実行中の1フレーム処理後に停止）" if self.paused else "再開しました。")

    def stop_analysis(self):
        if self.worker:
            self.worker.cancel()
            self.stop_button.setEnabled(False)
            self.update_result_counts("stopping")
            self.message("停止要求（実行中の1フレーム処理の終了を待ちます）")

    def analysis_finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.set_busy(False)
        if self.close_pending:
            self.close()

    def operation_failed(self, text):
        self.message(f"エラー: {text}")
        reason = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "原因不明のエラー")
        self.statusBar().showMessage(f"処理に失敗しました: {reason}")
        self.statusBar().setToolTip(text)
        if self.loader:
            self.model_status.setText(f"モデル読み込み失敗\n{reason}")
            self.model_status.setToolTip(text)
        if self.worker and self.run_metadata:
            self.run_metadata["stop_reason"] = "error"
        if self.loader or self.worker:
            self.update_result_counts("error")

    def set_busy(self, busy):
        self.model_button.setEnabled(not busy)
        self.source_button.setEnabled(not busy)
        for control in (self.machine, self.stride, self.mode, self.det_threshold, self.class_threshold, self.detector_path, self.classifier_path, self.save_config_button, *self.model_path_buttons):
            control.setEnabled(not busy)
        self.start_button.setEnabled(not busy and self.engine is not None and self.media is not None)
        self.sync_calibration_controls()
        self.pause_button.setEnabled(self.worker is not None)
        self.stop_button.setEnabled(self.worker is not None)
        self.export_button.setEnabled(not busy and self.result is not None)
        self.seek.setEnabled(not busy and self.media is not None and self.media.is_video)
        self.play_button.setEnabled(not busy and self.media is not None and self.media.is_video)

    def toggle_preview(self):
        if self.preview_timer.isActive():
            self.preview_timer.stop()
            self.play_button.setText("▶ プレビュー再生")
        elif self.media and self.media.is_video:
            if self.frame_index >= self.media.frame_count - 1:
                self.seek.setValue(0)
            self.preview_timer.start(max(20, round(1000 / self.media.fps)))
            self.play_button.setText("❚❚ プレビュー停止")

    def advance_preview(self):
        if self.frame_index >= self.media.frame_count - 1:
            self.preview_timer.stop()
            self.play_button.setText("▶ プレビュー再生")
        else:
            self.seek.setValue(self.frame_index + 1)

    def seek_frame(self, index):
        if not self.media or self.worker:
            return
        try:
            self.original_image = self.media.read(index)
            self.frame_index, self.result = index, None
            self.table.setRowCount(0)
            self.summary.setText("検出枠: —　閾値超え表示: —（未解析フレーム）")
            self.position.setText(f"{index + 1} / {self.media.frame_count}")
            self.update_result_counts()
            self.export_button.setEnabled(False)
            self.refresh_preview()
        except Exception as error:
            self.preview_timer.stop()
            self.operation_failed(str(error))

    def save_results(self):
        if self.result is None or self.run_settings is None:
            self.export_feedback.setText("まず画像・動画を解析してください。")
            return
        self.preview_timer.stop()
        output = self.export_directory / f"{self.source_path.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        try:
            rendered = render_result(self.original_image, self.result, show_all=False, show_labels=False, box_color=CANDIDATE_COLOR)
            display_settings = {"show_all": False, "show_labels": False, "box_color": CANDIDATE_COLOR, "grayscale": False, "brightness": 0}
            paths = export_bundle(output, self.source_path, self.frame_index, self.original_image, self.result, self.run_settings, self.engine.get_model_metadata(), rendered_image=rendered, history=self.history if self.media.is_video else None, display_settings=display_settings, input_metadata=self.run_metadata)
            download = CsvDownload(paths["results_csv"])
            self.downloads.append(download)
            opened = self.download_opener(QUrl(download.url))
            csv_url = escape(QUrl.fromLocalFile(str(paths["results_csv"])).toString(), quote=True)
            folder_url = escape(QUrl.fromLocalFile(str(output)).toString(), quote=True)
            self.export_feedback.setText(f"{'ブラウザーでダウンロードを開始しました。' if opened else 'CSVを保存しました。ブラウザーを開けませんでした。'}<br><a href='{csv_url}'>保存済みCSVを開く</a> · <a href='{folder_url}'>保存フォルダーを開く</a>")
            self.message(f"保存完了: {output} ({len(paths)} ファイル)")
            self.statusBar().showMessage(f"保存完了: {output}")
        except Exception as error:
            self.export_feedback.setText("保存またはダウンロードに失敗しました。画面下部の原因を確認してください。")
            self.operation_failed(str(error))

    def closeEvent(self, event):
        if self.loader or self.worker:
            self.close_pending = True
            if self.worker:
                self.worker.cancel()
            self.setEnabled(False)
            self.message("モデル処理の終了を待ってから閉じます。")
            event.ignore()
            return
        self.preview_timer.stop()
        for download in self.downloads:
            download.close()
        if self.media:
            self.media.close()
        if self.engine and hasattr(self.engine, "close"):
            self.engine.close()
        event.accept()
