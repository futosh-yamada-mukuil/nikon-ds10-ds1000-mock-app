"""Preview geometry and native control regressions for the reported display issue."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest

from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QStyle, QStyleFactory, QStyleOptionComboBox

from app.domain import FrameResult
from app.ui import MainWindow, Preview


APP = QApplication.instance() or QApplication([])


def settle():
    # Layout, deferred resize and viewport updates can be separate queued events.
    for _ in range(4):
        APP.processEvents()


def image_rectangle(preview):
    return preview.mapFromScene(preview.item.sceneBoundingRect()).boundingRect()


def wheel(preview, angle=(0, 0), pixel=(0, 0), modifiers=Qt.NoModifier):
    point = QPointF(preview.viewport().rect().center())
    event = QWheelEvent(point, point, QPoint(*pixel), QPoint(*angle), Qt.NoButton,
                        modifiers, Qt.NoScrollPhase, False)
    APP.sendEvent(preview.viewport(), event)
    settle()


class PreviewGeometryTests(unittest.TestCase):
    def setUp(self):
        self.preview = Preview()
        self.preview.resize(850, 550)
        self.preview.show()
        settle()
        self.image = Image.new("RGB", (2928, 2928), (100, 120, 140))
        self.preview.set_image(self.image)
        settle()

    def tearDown(self):
        self.preview.close()
        settle()

    def assert_filled_short_side(self):
        projected = image_rectangle(self.preview)
        viewport = self.preview.viewport().rect()
        self.assertGreaterEqual(min(projected.width(), projected.height()),
                                .8 * min(viewport.width(), viewport.height()),
                                "A square input must occupy the preview's available short side")
        self.assertLessEqual(projected.width(), viewport.width() + 4)
        self.assertLessEqual(projected.height(), viewport.height() + 4)

    def test_square_image_fills_preview_and_fit_restores_it_after_explicit_zoom(self):
        self.assert_filled_short_side()
        for _ in range(3):
            self.preview.zoom_in()
        self.assertGreater(image_rectangle(self.preview).height(), self.preview.viewport().height() * 1.5)
        self.preview.fit()
        settle()
        self.assert_filled_short_side()

    def test_scroll_trackpad_zero_and_horizontal_events_do_not_shrink_image(self):
        initial = self.preview.transform().m11()
        for angle, pixel in (((0, -120), (0, 0)), ((120, 0), (0, 0)),
                             ((0, 0), (0, -30)), ((0, 0), (0, 0))):
            for _ in range(20):
                wheel(self.preview, angle, pixel)
            self.assertAlmostEqual(self.preview.transform().m11(), initial)
            self.assertTrue(self.preview.auto_fit)
        self.assert_filled_short_side()

    def test_modified_vertical_wheel_zooms_but_zero_or_horizontal_do_not(self):
        initial = self.preview.transform().m11()
        wheel(self.preview, (0, 120), modifiers=Qt.ControlModifier)
        self.assertGreater(self.preview.transform().m11(), initial)
        self.assertFalse(self.preview.auto_fit)
        zoomed = self.preview.transform().m11()
        wheel(self.preview, (0, 0), modifiers=Qt.ControlModifier)
        wheel(self.preview, (120, 0), modifiers=Qt.ControlModifier)
        self.assertAlmostEqual(self.preview.transform().m11(), zoomed)
        wheel(self.preview, (0, -120), modifiers=Qt.MetaModifier)
        self.assertLess(self.preview.transform().m11(), zoomed)
        self.preview.fit()
        self.assert_filled_short_side()

    def test_repeated_same_size_frames_preserve_explicit_zoom_and_auto_fit_resizes(self):
        self.preview.zoom_in()
        zoomed = self.preview.transform().m11()
        for _ in range(10):
            self.preview.set_image(self.image)
        self.assertAlmostEqual(self.preview.transform().m11(), zoomed)
        self.preview.fit()
        self.preview.resize(1050, 700)
        settle()
        self.assert_filled_short_side()

    def test_zoom_buttons_are_bounded_and_fit_remains_usable(self):
        for _ in range(150):
            self.preview.zoom_in()
        self.assertGreater(self.preview.transform().m11(), 0)
        self.assertLessEqual(self.preview.transform().m11(), 30.0001)
        for _ in range(300):
            self.preview.zoom_out()
        self.assertGreater(self.preview.transform().m11(), 0)
        self.assert_filled_short_side()
        self.preview.fit()
        self.assert_filled_short_side()


class PreviewFileAndControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.first = self.directory / "first.png"
        self.second = self.directory / "second.png"
        Image.new("RGB", (1200, 1200), (100, 120, 140)).save(self.first)
        Image.new("RGB", (1600, 1600), (140, 120, 100)).save(self.second)
        self.window = MainWindow(config=self.directory / "missing.json", autoload=False)
        self.window.show()
        settle()
        self.window.open_source(self.first)
        settle()

    def tearDown(self):
        self.window.close()
        settle()
        self.temporary.cleanup()

    def assert_large(self):
        rect = image_rectangle(self.window.preview)
        viewport = self.window.preview.viewport().rect()
        self.assertGreaterEqual(min(rect.width(), rect.height()), .8 * min(viewport.width(), viewport.height()))

    def test_new_file_resets_old_zoom_out_and_fills_preview(self):
        # Reproduce the legacy screenshot's tiny persisted transform. The new
        # zoom controls themselves should no longer permit this state.
        self.window.preview.auto_fit = False
        self.window.preview.resetTransform()
        self.window.preview.scale(.02, .02)
        old_height = image_rectangle(self.window.preview).height()
        self.window.open_source(self.second)
        settle()
        self.assertGreater(image_rectangle(self.window.preview).height(), old_height * 2)
        self.assert_large()
        self.assertTrue(self.window.preview.auto_fit)

    def test_repeated_result_refresh_never_shrinks_fit(self):
        initial = image_rectangle(self.window.preview).height()
        for _ in range(4):
            self.window.refresh_preview()
            settle()
            self.assert_large()
        self.window.receive_result(0, self.window.original_image, FrameResult([]))
        settle()
        self.assertGreaterEqual(image_rectangle(self.window.preview).height(), initial * .95)

    def test_explicit_zoom_survives_result_updates(self):
        self.window.preview.zoom_in()
        initial = self.window.preview.transform().m11()
        self.window.receive_result(0, self.window.original_image, FrameResult([]))
        settle()
        self.assertAlmostEqual(self.window.preview.transform().m11(), initial)

    def assert_machine_text_fits(self):
        self.window.machine.setCurrentText("DS1000")
        settle()
        combo = self.window.machine
        option = QStyleOptionComboBox()
        combo.initStyleOption(option)
        edit_rect = combo.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, combo)
        text_width = combo.fontMetrics().horizontalAdvance("DS1000")
        self.assertGreaterEqual(combo.width(), 160)
        self.assertGreaterEqual(edit_rect.width(), text_width + 4,
                                "The displayed DS1000 text must fit before the native combo arrow")

    def test_machine_combobox_displays_full_ds1000_at_supported_window_sizes(self):
        for width, height in ((1500, 940), (1000, 700)):
            self.window.resize(width, height)
            self.assert_machine_text_fits()

    def test_native_macos_combo_text_fits_if_aqua_style_is_available(self):
        names = QStyleFactory.keys()
        name = next((name for name in names if name.lower() == "macos"), None)
        if name is None:
            self.skipTest("Native macOS style is unavailable on this platform")
        self.native_style = QStyleFactory.create(name)
        self.window.machine.setStyle(self.native_style)
        self.assert_machine_text_fits()


if __name__ == "__main__":
    unittest.main()
