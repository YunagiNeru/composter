import unittest
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

from desktop_pet.sprite_art import ANIMATION_FRAMES, SpritePainter


class SpriteTests(unittest.TestCase):
    def test_all_frames_and_dynamic_variants_use_only_black_white_or_transparent(self):
        allowed = {bytes((0, 0, 0, 0)), bytes((0, 0, 0, 255)), bytes((255, 255, 255, 255))}
        for animation, count in ANIMATION_FRAMES.items():
            for frame in range(count):
                for fullness in (0.0, 1.0):
                    with self.subTest(animation=animation, frame=frame, fullness=fullness):
                        image = SpritePainter.render(animation, frame, fullness,
                                                     expression="SLEEPY", ear_frame=frame)
                        data = bytes(image.constBits())
                        colors = {data[i:i + 4] for i in range(0, len(data), 4)}
                        self.assertEqual(colors, allowed)
                        for edge in range(64):
                            for x, y in ((edge, 0), (edge, 63), (0, edge), (63, edge)):
                                self.assertEqual(image.pixelColor(x, y).alpha(), 0)

    def test_saved_frames_match_current_renderer(self):
        assets = Path(__file__).resolve().parents[1] / "assets" / "sprites"
        self.assertEqual(len(list(assets.glob("*/frame-*.png"))), sum(ANIMATION_FRAMES.values()))
        for animation, count in ANIMATION_FRAMES.items():
            for frame in range(count):
                with self.subTest(animation=animation, frame=frame):
                    saved = QImage(str(assets / animation / f"frame-{frame:02d}.png"))
                    self.assertEqual(saved.convertToFormat(QImage.Format.Format_RGBA8888),
                                     SpritePainter.render(animation, frame))

    def test_poses_and_runtime_controls_are_visibly_distinct(self):
        for name in ("walk", "drowsy_walk", "run", "eat", "play", "dig", "body_size"):
            frames = set()
            for i in range(ANIMATION_FRAMES[name]):
                image = SpritePainter.render(name, i)
                frames.add(bytes(image.constBits()))
            self.assertGreater(len(frames), 1, name)
        idle = SpritePainter.render("normal_idle")
        for kwargs in ({"expression": "SLEEPY"}, {"expression": "CHEERFUL"},
                       {"social_response": "GENTLE_WITHDRAW"}, {"ear_frame": 1}):
            self.assertNotEqual(idle, SpritePainter.render("normal_idle", **kwargs))
        self.assertNotEqual(SpritePainter.render("walk"), SpritePainter.render("run"))

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
