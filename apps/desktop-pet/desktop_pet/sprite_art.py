from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygon

from .theme import SPRITE_COLORS


ANIMATION_FRAMES = {
    "normal_idle": 2, "sleepy_idle": 2, "walk": 4, "drowsy_walk": 4,
    "run": 4, "sleep_yawn": 4, "meow": 3, "sleepy_meow": 3,
    "ears": 3, "pet_happy": 3, "pet_dislike": 3, "eat": 4,
    "play": 4, "dig": 4, "body_size": 3,
}
BODY_LEVELS = (0.12, 0.5, 0.9)
INK = QColor(SPRITE_COLORS["ink"])
PAPER = QColor(SPRITE_COLORS["paper"])


class SpritePainter:
    """白黒の短足猫。輪郭・顔の座標を共有し、移動は四足姿勢で描く。"""

    @staticmethod
    def render(animation: str, frame: int = 0, fullness: float = 0.5, *,
               expression: str | None = None, social_response: str | None = None,
               ear_frame: int | None = None) -> QImage:
        if animation not in ANIMATION_FRAMES:
            animation = "normal_idle"
        frame %= ANIMATION_FRAMES[animation]
        if animation == "body_size":
            fullness = BODY_LEVELS[frame]
        image = QImage(64, 64, QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        p = QPainter(image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(Qt.PenStyle.NoPen)
        SpritePainter._cat(p, animation, frame, min(1.0, max(0.0, fullness)),
                           expression, social_response, ear_frame)
        p.end()

        # 暗い壁紙でも輪郭が消えない1pxの白縁。半透明色は生成しない。
        silhouette = image.copy()
        p = QPainter(silhouette)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        p.fillRect(silhouette.rect(), PAPER)
        p.end()
        result = QImage(64, 64, QImage.Format.Format_RGBA8888)
        result.fill(Qt.GlobalColor.transparent)
        p = QPainter(result)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            p.drawImage(dx, dy, silhouette)
        p.drawImage(0, 0, image)
        p.end()
        return result

    @staticmethod
    def _cat(p: QPainter, name: str, frame: int, fullness: float,
             expression: str | None, social: str | None, ear_frame: int | None) -> None:
        moving = name in {"walk", "drowsy_walk", "run", "play"}
        crouching = name in {"eat", "dig"}
        sleeping = name == "sleep_yawn" and frame < 2
        width = 22 + round(fullness * 7)
        bob = (0, 1, 0, -1)[frame % 4] if moving else 0
        head_x, head_y = (21, 29 + bob) if moving else (28, 29)
        if name == "run":
            head_x, head_y = 20, 31 + bob
        elif crouching:
            head_x, head_y = 21, 36 + frame % 2
        elif sleeping:
            head_x, head_y = 23, 39 + frame % 2
        elif name == "sleepy_idle":
            head_y += 2
        elif name == "pet_happy" or social == "LEAN_IN":
            head_x -= frame % 2
        elif name == "pet_dislike" or social == "GENTLE_WITHDRAW":
            head_x += 2

        # 尻尾→胴→短い足→頭の順。黒い面をつなげ、関節の箱形を残さない。
        tail_tip = (0, -1, 1, 0)[frame % 4]
        if sleeping:
            SpritePainter._stroke(p, [(39, 48), (47, 49), (50, 46), (50, 40),
                                      (47, 38)], INK, 5)
            p.setBrush(INK)
            p.drawEllipse(QRect(20, 36, width + 6, 18))
        elif moving or crouching:
            SpritePainter._stroke(p, [(43, 41 + bob), (51, 39 + bob), (54, 35),
                                      (54, 28 + tail_tip), (51, 25 + tail_tip)], INK, 5)
            p.setBrush(INK)
            p.drawEllipse(QRect(22, 31 + bob, width + 3, 19))
            stride = (2, -1, -2, 1)[frame % 4] if moving else frame % 2
            lift = (0, 2, 0, 1)[frame % 4] if moving else 0
            if name == "run":
                stride *= 2
                lift += 2
            for x, shift, raised in ((19, stride, lift), (26, -stride, 0),
                                      (39, -stride, 0), (45, stride, lift)):
                SpritePainter._paw(p, x + shift, 44 + bob, 5, 11 - raised - bob)
            if name == "play":
                SpritePainter._stroke(p, [(21, 44), (16, 49), (9 + frame % 2, 51)], INK, 5)
            elif name == "dig":
                SpritePainter._stroke(p, [(23, 45), (17 - frame % 2 * 3, 53)], INK, 5)
        else:
            SpritePainter._stroke(p, [(40, 51), (50, 50), (55, 46), (56, 39),
                                      (54, 35 + tail_tip), (51, 35 + tail_tip)], INK, 5)
            p.setBrush(INK)
            p.drawEllipse(QRect(18, 33, width, 22))
            SpritePainter._paw(p, 22, 44, 7, 11)
            SpritePainter._paw(p, 33, 44, 7, 11)

        sleepy = (expression == "SLEEPY" or name in {
            "sleepy_idle", "drowsy_walk", "sleep_yawn", "sleepy_meow"}
            or social == "SLOW_BLINK" or name == "normal_idle" and frame == 1)
        happy = expression == "CHEERFUL" or name == "pet_happy"
        flattened = name == "pet_dislike" or social == "GENTLE_WITHDRAW"
        twitch = (ear_frame if ear_frame is not None else frame if name == "ears" else 0) % 4
        SpritePainter._head(p, head_x, head_y, sleepy, happy, flattened, twitch,
                            sleeping, name, frame)

        # 座った前足も横へ張り出さず、真下へ下りてU字に閉じる。
        if not moving and not crouching and not sleeping:
            for x in (22, 33):
                SpritePainter._stroke(p, [(x, 46), (x, 52), (x + 1, 54),
                                          (x + 5, 54), (x + 6, 52), (x + 6, 46)], PAPER)
        if sleeping:
            SpritePainter._stroke(p, [(35, 49), (40, 51), (46, 50)], PAPER)
            SpritePainter._symbol(p, 47, 20 - frame, ("1111", "0010", "0100", "1111"))
        elif name == "pet_happy":
            SpritePainter._symbol(p, 49, 16 - frame % 2,
                                  ("0110110", "1111111", "1111111", "0111110", "0011100", "0001000"))
        elif name == "eat":
            SpritePainter._polygon(p, [(10, 54), (33, 54), (30, 59), (13, 59)], INK)
            p.fillRect(QRect(12, 54, 19, 1), PAPER)
            p.fillRect(QRect(15, 52 + frame % 2, 3, 2), INK)
            p.fillRect(QRect(23, 52, 3, 2), INK)
        elif name == "play":
            p.setBrush(INK)
            p.drawEllipse(QRect(4 + frame % 2 * 2, 53, 7, 7))
            p.fillRect(QRect(6 + frame % 2 * 2, 55, 2, 2), PAPER)
        elif name == "dig":
            for x, y in ((9, 55), (14, 58), (5, 51 - frame % 2 * 3)):
                p.fillRect(QRect(x + frame % 2, y, 2, 2), INK)

    @staticmethod
    def _head(p: QPainter, x: int, y: int, sleepy: bool, happy: bool,
              flattened: bool, twitch: int, resting: bool, name: str, frame: int) -> None:
        cheek = 9 if resting else 12
        # 頭頂と頬は全ポーズ共通。耳だけ独立して傾ける。
        SpritePainter._polygon(p, [(x - 15, y - 7), (x - 11, y - 11),
            (x + 10, y - 11), (x + 15, y - 7), (x + 17, y - 1),
            (x + 17, y + 4), (x + 14, y + 8), (x + 8, y + cheek),
            (x - 8, y + cheek), (x - 14, y + 8), (x - 17, y + 4),
            (x - 17, y - 1)], INK)
        left_tip = (x - 18, y - 10) if flattened else (x - 13 - (twitch == 1), y - 19 + (twitch == 1) * 3)
        right_tip = (x + 18, y - 10) if flattened else (x + 13 + (twitch == 2), y - 19 + (twitch == 2) * 3)
        SpritePainter._polygon(p, [(x - 15, y - 2), left_tip, (x - 3, y - 9)], INK)
        SpritePainter._polygon(p, [(x + 3, y - 9), right_tip, (x + 15, y - 2)], INK)
        if not flattened:
            SpritePainter._stroke(p, [(left_tip[0] + 2, left_tip[1] + 5), (x - 9, y - 9)], PAPER)
            SpritePainter._stroke(p, [(right_tip[0] - 2, right_tip[1] + 5), (x + 9, y - 9)], PAPER)

        eye_y = y + (1 if resting else 0)
        for eye_x in (x - 8, x + 5):
            if happy:
                SpritePainter._stroke(p, [(eye_x - 1, eye_y + 2), (eye_x + 1, eye_y),
                                          (eye_x + 3, eye_y + 2)], PAPER)
            elif sleepy:
                SpritePainter._stroke(p, [(eye_x - 1, eye_y + 1), (eye_x + 1, eye_y + 2),
                                          (eye_x + 4, eye_y + 1)], PAPER)
            else:
                p.fillRect(QRect(eye_x, eye_y - 2, 4, 6), PAPER)
                p.fillRect(QRect(eye_x - 1, eye_y - 1, 6, 4), PAPER)
        mouth_open = name in {"meow", "sleepy_meow"} and frame == 1
        yawn = name == "sleep_yawn" and frame >= 2
        if mouth_open or yawn:
            p.fillRect(QRect(x - 1, y + 8, 3, 3 if mouth_open else 4), PAPER)
            p.fillRect(QRect(x, y + 9, 1, 1 if mouth_open else 2), INK)
        elif not resting:
            SpritePainter._stroke(p, [(x - 3, y + 8), (x - 1, y + 9), (x, y + 8),
                                      (x + 1, y + 9), (x + 3, y + 8)], PAPER)

    @staticmethod
    def _paw(p: QPainter, x: int, y: int, width: int, height: int) -> None:
        p.fillRect(QRect(x, y, width, height - 1), INK)
        p.fillRect(QRect(x + 1, y + height - 1, width - 2, 1), INK)

    @staticmethod
    def _polygon(p: QPainter, points: list[tuple[int, int]], color: QColor) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawPolygon(QPolygon([QPoint(x, y) for x, y in points]))

    @staticmethod
    def _stroke(p: QPainter, points: list[tuple[int, int]], color: QColor, width: int = 1) -> None:
        p.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap,
                      Qt.PenJoinStyle.MiterJoin))
        p.drawPolyline(QPolygon([QPoint(x, y) for x, y in points]))
        p.setPen(Qt.PenStyle.NoPen)

    @staticmethod
    def _symbol(p: QPainter, x: int, y: int, rows: tuple[str, ...]) -> None:
        for dy, row in enumerate(rows):
            for dx, bit in enumerate(row):
                if bit == "1":
                    p.fillRect(QRect(x + dx, y + dy, 1, 1), INK)


def generate_sprites(destination: Path) -> list[Path]:
    generated: list[Path] = []
    for animation, count in ANIMATION_FRAMES.items():
        target = destination / animation
        target.mkdir(parents=True, exist_ok=True)
        for frame in range(count):
            path = target / f"frame-{frame:02d}.png"
            if not SpritePainter.render(animation, frame).save(str(path), "PNG"):
                raise OSError(f"sprite save failed: {path.name}")
            generated.append(path)

    sheet = QImage(5 * 64, 3 * 64, QImage.Format.Format_RGBA8888)
    sheet.fill(Qt.GlobalColor.transparent)
    painter = QPainter(sheet)
    for index, animation in enumerate(ANIMATION_FRAMES):
        painter.drawImage(index % 5 * 64, index // 5 * 64, SpritePainter.render(animation))
    painter.end()
    for scale in (1, 2, 4):
        path = destination.parent / f"sprite-preview-{scale}x.png"
        sheet.scaled(sheet.width() * scale, sheet.height() * scale,
                     Qt.AspectRatioMode.IgnoreAspectRatio,
                     Qt.TransformationMode.FastTransformation).save(str(path), "PNG")
        generated.append(path)

    # 行順はANIMATION_FRAMES、列はフレーム順。全コマをレビューできる一覧。
    frames = QImage(4 * 64, len(ANIMATION_FRAMES) * 64, QImage.Format.Format_RGBA8888)
    frames.fill(Qt.GlobalColor.transparent)
    painter = QPainter(frames)
    for row, (animation, count) in enumerate(ANIMATION_FRAMES.items()):
        for frame in range(count):
            painter.drawImage(frame * 64, row * 64, SpritePainter.render(animation, frame))
    painter.end()
    path = destination.parent / "sprite-frames-2x.png"
    frames.scaled(frames.width() * 2, frames.height() * 2,
                  Qt.AspectRatioMode.IgnoreAspectRatio,
                  Qt.TransformationMode.FastTransformation).save(str(path), "PNG")
    generated.append(path)
    return generated
