from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPolygon

from .theme import COLORS


ANIMATION_FRAMES = {
    "normal_idle": 2, "sleepy_idle": 2, "walk": 4, "drowsy_walk": 4,
    "run": 4, "sleep_yawn": 4, "meow": 3, "sleepy_meow": 3,
    "ears": 3, "pet_happy": 3, "pet_dislike": 3, "eat": 4,
    "play": 4, "dig": 4, "body_size": 3,
}
BODY_LEVELS = (0.12, 0.5, 0.9)


class SpritePainter:
    """Draws one consistent 64x64 tabby cat with integer pixel coordinates."""

    @staticmethod
    def render(animation: str, frame: int = 0, fullness: float = 0.5, *,
               expression: str | None = None, social_response: str | None = None,
               ear_frame: int | None = None) -> QImage:
        if animation not in ANIMATION_FRAMES:
            animation = "normal_idle"
        count = ANIMATION_FRAMES[animation]
        frame %= count
        if animation == "body_size":
            fullness = BODY_LEVELS[min(2, frame)]
        image = QImage(64, 64, QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.setPen(Qt.PenStyle.NoPen)
        SpritePainter._cat(painter, animation, frame, fullness,
                           expression, social_response, ear_frame)
        painter.end()
        return image

    @staticmethod
    def _cat(p: QPainter, animation: str, frame: int, fullness: float,
             expression: str | None, social_response: str | None,
             ear_frame: int | None) -> None:
        c = {key: QColor(value) for key, value in COLORS.items()}
        moving = animation in {"walk", "drowsy_walk", "run", "play"}
        body_y = 30 + (1 if moving and frame % 2 else 0)
        body_width = 34 + round(8 * min(1.0, max(0.0, fullness)))
        body_x = (64 - body_width) // 2 - 1
        leg_shift = [0, 2, -1, 1][frame % 4] if moving else 0
        if animation == "sleep_yawn":
            body_y = 32

        # Tail is the same base shape in every pose; only its tip changes by a pixel.
        tail_points = [QPoint(46, 42), QPoint(53, 38), QPoint(56, 31),
                       QPoint(58, 26 + frame % 2), QPoint(55, 23 + frame % 2)]
        SpritePainter._polyline(p, tail_points, c["fur_outline"], 4)
        SpritePainter._polyline(p, tail_points[1:], c["fur"], 2)
        p.setBrush(c["fur_light"])
        p.drawEllipse(QRect(body_x - 1, body_y, body_width + 2, 23))
        p.setBrush(c["fur"])
        p.drawEllipse(QRect(body_x, body_y + 1, body_width, 21))
        p.setBrush(c["cream"])
        p.drawEllipse(QRect(body_x + body_width // 3, body_y + 10, body_width // 2, 11))

        # Shared legs maintain the same body anchor across idle and interaction poses.
        leg_y = 47 + (1 if animation in {"sleep_yawn", "sleepy_idle"} else 0)
        if animation in {"walk", "drowsy_walk", "run", "play"}:
            offsets = (leg_shift, -leg_shift, -leg_shift, leg_shift)
        else:
            offsets = (0, 0, 0, 0)
        legs = ((body_x + 3, 43, 9, 15), (body_x + 12, 45, 8, 13),
                (body_x + body_width - 17, 44, 8, 14),
                (body_x + body_width - 8, 42, 9, 16))
        for (x, y, w, h), shift in zip(legs, offsets):
            SpritePainter._outlined_rect(p, QRect(x + shift, y, w, h), c["fur_outline"], c["fur"])
            p.setBrush(c["cream"])
            p.drawRect(QRect(x + shift + 2, y + h - 3, max(2, w - 4), 2))

        # Ears are part of one head silhouette; a dislike pose only lowers their tips.
        head_y = 11
        ear_drop = (5 if animation == "pet_dislike" or social_response == "GENTLE_WITHDRAW"
                    else 1 if (animation == "ears" and frame == 1)
                    or (ear_frame is not None and ear_frame % 3 == 1) else 0)
        head = QRect(17, head_y + ear_drop, 30, 29)
        p.setBrush(c["fur_outline"])
        p.drawPolygon(QPolygon([QPoint(17, 18 + ear_drop), QPoint(19, 3 + ear_drop),
                                QPoint(31, 15 + ear_drop), QPoint(35, 14 + ear_drop),
                                QPoint(47, 4 + ear_drop), QPoint(48, 21 + ear_drop)]))
        p.setBrush(c["pink"])
        p.drawPolygon(QPolygon([QPoint(21, 14 + ear_drop), QPoint(21, 8 + ear_drop),
                                QPoint(27, 15 + ear_drop)]))
        p.drawPolygon(QPolygon([QPoint(39, 15 + ear_drop), QPoint(45, 9 + ear_drop),
                                QPoint(44, 17 + ear_drop)]))
        p.setBrush(c["fur"])
        p.drawEllipse(head)

        # Tabby stripes use the same forehead/cheek locations in every frame.
        p.fillRect(QRect(28, 15 + ear_drop, 3, 7), c["fur_dark"])
        p.fillRect(QRect(33, 13 + ear_drop, 3, 8), c["fur_dark"])
        p.fillRect(QRect(38, 15 + ear_drop, 3, 7), c["fur_dark"])
        p.fillRect(QRect(20, 28 + ear_drop, 7, 2), c["fur_dark"])
        p.fillRect(QRect(38, 29 + ear_drop, 7, 2), c["fur_dark"])

        # Eyes and face remain fixed to the head; sleepy poses narrow the eye pixels.
        sleepy = (expression == "SLEEPY" or animation in {"sleepy_idle", "sleep_yawn", "sleepy_meow"}
                  or animation == "normal_idle" and frame == 1
                  or social_response == "SLOW_BLINK")
        cheerful = expression == "CHEERFUL" or animation == "pet_happy"
        if cheerful:
            SpritePainter._eye(p, 25, 23 + ear_drop, c["eye"], happy=True)
            SpritePainter._eye(p, 38, 23 + ear_drop, c["eye"], happy=True)
        elif sleepy:
            p.fillRect(QRect(23, 25 + ear_drop, 7, 2), c["eye"])
            p.fillRect(QRect(36, 25 + ear_drop, 7, 2), c["eye"])
        else:
            eye_h = 7 if animation in {"pet_dislike", "sleep_yawn"} else 8
            p.setBrush(c["white"])
            p.drawEllipse(QRect(23, 22 + ear_drop, 8, eye_h))
            p.drawEllipse(QRect(36, 22 + ear_drop, 8, eye_h))
            p.setBrush(c["eye"])
            p.drawEllipse(QRect(26, 24 + ear_drop, 4, 5))
            p.drawEllipse(QRect(38, 24 + ear_drop, 4, 5))
            p.fillRect(QRect(27, 24 + ear_drop, 1, 2), c["white"])
            p.fillRect(QRect(39, 24 + ear_drop, 1, 2), c["white"])

        p.setBrush(c["pink"])
        p.drawPolygon(QPolygon([QPoint(31, 31 + ear_drop), QPoint(34, 31 + ear_drop),
                                QPoint(32, 33 + ear_drop)]))
        p.setPen(c["fur_outline"])
        p.drawLine(32, 33 + ear_drop, 32, 35 + ear_drop)
        p.drawLine(32, 35 + ear_drop, 29, 36 + ear_drop)
        p.drawLine(32, 35 + ear_drop, 35, 36 + ear_drop)
        p.setPen(Qt.PenStyle.NoPen)

        mouth_open = (animation in {"meow", "sleepy_meow"} and frame % 3 == 1) or (
            animation == "sleep_yawn" and frame in {2, 3}
        )
        if mouth_open:
            p.setBrush(c["fur_outline"])
            p.drawEllipse(QRect(29, 35 + ear_drop, 7, 8 if animation == "sleep_yawn" else 6))
            p.setBrush(c["pink"])
            p.drawEllipse(QRect(31, 39 + ear_drop, 3, 3))
        elif animation in {"meow", "sleepy_meow"}:
            p.fillRect(QRect(31, 36 + ear_drop, 4, 2), c["fur_outline"])

        if animation == "sleep_yawn":
            SpritePainter._pixel_text(p, 50, 18 - (frame % 2) * 2, "Z", c["accent"])
            SpritePainter._pixel_text(p, 55, 11, "Z", c["accent"])
        elif animation == "pet_happy":
            SpritePainter._heart(p, 49, 13, c["pink"])
        elif animation == "pet_dislike":
            p.fillRect(QRect(12, 20, 3, 2), c["danger"])
        elif animation == "ears":
            p.fillRect(QRect(15 + (frame % 2), 8, 2, 5), c["accent"])
            p.fillRect(QRect(49 - (frame % 2), 9, 2, 5), c["accent"])
        elif animation == "eat":
            p.setBrush(c["fur_outline"])
            p.drawRect(QRect(3, 50, 17, 5))
            p.setBrush(c["pink"])
            p.drawRect(QRect(6, 48, 11, 4))
            if frame % 2 == 0:
                p.fillRect(QRect(24, 41, 3, 2), c["cream"])
        elif animation == "play":
            p.setBrush(c["fur_outline"])
            p.drawEllipse(QRect(49 - frame % 2 * 3, 50, 9, 9))
            p.setBrush(c["accent"])
            p.drawEllipse(QRect(51 - frame % 2 * 3, 52, 5, 5))
        elif animation == "dig":
            p.fillRect(QRect(3, 56, 58, 2), c["ground"])
            p.fillRect(QRect(10 + frame % 2 * 4, 52, 4, 3), c["ground"])
            p.fillRect(QRect(18 + frame % 2 * 3, 54, 3, 2), c["ground"])

        if ear_frame is not None and animation != "ears" and ear_frame % 4 in {1, 2}:
            p.fillRect(QRect(13, 8 + ear_frame % 2, 2, 4), c["accent"])
            p.fillRect(QRect(49, 8 + ear_frame % 2, 2, 4), c["accent"])

    @staticmethod
    def _eye(p: QPainter, x: int, y: int, color: QColor, happy: bool = False) -> None:
        p.setPen(color)
        p.drawLine(x, y + 2, x + 2, y)
        p.drawLine(x + 2, y, x + 4, y + 2)
        p.setPen(Qt.PenStyle.NoPen)

    @staticmethod
    def _outlined_rect(p: QPainter, rect: QRect, outline: QColor, fill: QColor) -> None:
        p.setBrush(outline)
        p.drawRect(rect)
        p.setBrush(fill)
        p.drawRect(rect.adjusted(1, 0, -1, -1))

    @staticmethod
    def _polyline(p: QPainter, points: list[QPoint], color: QColor, width: int) -> None:
        path = QPainterPath(QPointF(points[0]))
        for point in points[1:]:
            path.lineTo(QPointF(point))
        pen = p.pen()
        pen.setColor(color)
        pen.setWidth(width)
        pen.setCapStyle(Qt.PenCapStyle.SquareCap)
        p.setPen(pen)
        p.drawPath(path)
        p.setPen(Qt.PenStyle.NoPen)

    @staticmethod
    def _pixel_text(p: QPainter, x: int, y: int, letter: str, color: QColor) -> None:
        # Tiny block Z, not a system-font glyph, so it stays pixel sharp.
        p.fillRect(QRect(x, y, 4, 1), color)
        p.fillRect(QRect(x + 2, y + 1, 1, 1), color)
        p.fillRect(QRect(x + 1, y + 2, 1, 1), color)
        p.fillRect(QRect(x, y + 3, 4, 1), color)

    @staticmethod
    def _heart(p: QPainter, x: int, y: int, color: QColor) -> None:
        p.fillRect(QRect(x + 1, y, 2, 2), color)
        p.fillRect(QRect(x + 5, y, 2, 2), color)
        p.fillRect(QRect(x, y + 1, 8, 2), color)
        p.fillRect(QRect(x + 1, y + 3, 6, 2), color)
        p.fillRect(QRect(x + 2, y + 5, 4, 2), color)
        p.fillRect(QRect(x + 3, y + 7, 2, 1), color)


def generate_sprites(destination: Path) -> list[Path]:
    from PySide6.QtGui import QImage

    generated: list[Path] = []
    for animation, count in ANIMATION_FRAMES.items():
        target = destination / animation
        target.mkdir(parents=True, exist_ok=True)
        for frame in range(count):
            image = SpritePainter.render(animation, frame)
            path = target / f"frame-{frame:02d}.png"
            if not image.save(str(path), "PNG"):
                raise OSError(f"sprite save failed: {path.name}")
            generated.append(path)

    columns = 5
    rows = (len(ANIMATION_FRAMES) + columns - 1) // columns
    sheet = QImage(columns * 64, rows * 64, QImage.Format.Format_RGBA8888)
    sheet.fill(Qt.GlobalColor.transparent)
    painter = QPainter(sheet)
    for index, (animation, _) in enumerate(ANIMATION_FRAMES.items()):
        x, y = index % columns * 64, index // columns * 64
        painter.drawImage(QPoint(x, y), SpritePainter.render(animation, 0))
    painter.end()
    sheet_path = destination.parent / "sprite-preview-1x.png"
    sheet.save(str(sheet_path), "PNG")
    for scale in (2, 4):
        preview = sheet.scaled(sheet.width() * scale, sheet.height() * scale,
                               Qt.AspectRatioMode.IgnoreAspectRatio,
                               Qt.TransformationMode.FastTransformation)
        preview_path = destination.parent / f"sprite-preview-{scale}x.png"
        preview.save(str(preview_path), "PNG")
        generated.append(preview_path)
    generated.append(sheet_path)
    return generated
