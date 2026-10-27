import unittest

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

from desktop_pet.sprite_art import ANIMATION_FRAMES, SpritePainter


class SpriteTests(unittest.TestCase):
    def test_every_animation_frame_is_rgba_64_square_with_transparency(self):
        for animation, frame_count in ANIMATION_FRAMES.items():
            for frame in range(frame_count):
                with self.subTest(animation=animation, frame=frame):
                    image = SpritePainter.render(animation, frame)
                    self.assertEqual((image.width(), image.height()), (64, 64))
                    self.assertEqual(image.format(), QImage.Format.Format_RGBA8888)
                    self.assertEqual(image.pixelColor(0, 0).alpha(), 0)
                    self.assertGreater(sum(image.pixelColor(x, y).alpha() > 0
                                           for x in range(64) for y in range(64)), 500)

    def test_nearest_neighbor_scaling_preserves_hard_pixel_edges(self):
        image = SpritePainter.render("normal_idle", 0)
        enlarged = image.scaled(128, 128, Qt.AspectRatioMode.IgnoreAspectRatio,
                                Qt.TransformationMode.FastTransformation)
        for y in range(64):
            for x in range(64):
                color = image.pixelColor(x, y)
                self.assertEqual(enlarged.pixelColor(x * 2, y * 2), color)
                self.assertEqual(enlarged.pixelColor(x * 2 + 1, y * 2 + 1), color)


if __name__ == "__main__":
    unittest.main()
