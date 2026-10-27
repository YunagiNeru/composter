import unittest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from desktop_pet.behavior import PetMetrics, compose_behavior
from desktop_pet.settings import DEFAULTS
from desktop_pet.window import PetWindow, display_side_for_area


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_window_is_transparent_frameless_and_topmost_by_default(self):
        window = PetWindow(DEFAULTS.copy())
        flags = window.windowFlags()
        self.assertTrue(flags & Qt.WindowType.FramelessWindowHint)
        self.assertTrue(flags & Qt.WindowType.WindowStaysOnTopHint)
        self.assertTrue(window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground))
        window.close()

    def test_alpha_mask_excludes_clear_corners_and_keeps_cat_body(self):
        window = PetWindow(DEFAULTS.copy())
        self.assertFalse(window.mask().contains(QPoint(0, 0)))
        self.assertTrue(window.mask().contains(QPoint(window.width() // 2, window.height() // 2)))
        window.close()

    def test_gpu_size_keeps_max_area_inside_selected_display(self):
        window = PetWindow(DEFAULTS.copy())
        metrics = PetMetrics("test", 5, 80, 100, "LOW", "LOW", 0, 0, 50, None)
        behavior = compose_behavior(metrics, DEFAULTS["pet"], DEFAULTS, {}, rule_only=True)
        window.set_sensor_snapshot(metrics, behavior, DEFAULTS)
        screen = window._selected_screen()
        if screen is not None:
            bounds = screen.availableGeometry()
            self.assertLessEqual(window.geometry().right(), bounds.right())
            self.assertLessEqual(window.geometry().bottom(), bounds.bottom())
            area_percent = (window.width() * window.height() /
                            (bounds.width() * bounds.height()) * 100)
            self.assertLessEqual(area_percent, 3.05)
        window.close()

    def test_minimum_area_setting_is_not_raised_to_a_64px_sprite(self):
        self.assertEqual(display_side_for_area(800, 600, 0.5), 48)
        self.assertLessEqual((48 * 48) / (800 * 600) * 100, 0.5)

    def test_display_area_rejects_invalid_geometry_or_percentage(self):
        for arguments in ((0, 600, 3), (800, 0, 3), (800, 600, 0),
                          (800, 600, float("nan"))):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                display_side_for_area(*arguments)

    def test_ai_status_clears_pid_after_worker_exit(self):
        window = PetWindow(DEFAULTS.copy())
        window.set_ai_status("利用中", pid=43127)
        self.assertEqual(window.worker_pid, 43127)
        window.set_ai_status("AI利用不可", pid=None)
        self.assertIsNone(window.worker_pid)
        window.close()


if __name__ == "__main__":
    unittest.main()
