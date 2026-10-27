"""Shared palette and typography tokens for the desktop UI and sprite renderer."""

COLORS = {
    "window": "#17201d",
    "surface": "#202b27",
    "surface_hover": "#2b3a34",
    "border": "#44534c",
    "text": "#f4f0e8",
    "muted": "#b6c0b8",
    "accent": "#e3ad66",
    "danger": "#e17d71",
    "fur_outline": "#382015",
    "fur": "#ca7937",
    "fur_light": "#f4c583",
    "fur_dark": "#873e24",
    "cream": "#ffe3bb",
    "pink": "#e98b82",
    "eye": "#34251d",
    "white": "#fff6e8",
    "ground": "#684432",
}
FONT_FAMILY = "Segoe UI"
FONT_SIZE = 10
CORNER_RADIUS = 7
SPACING = 8


def application_stylesheet() -> str:
    return f"""
        QWidget {{ background: {COLORS['window']}; color: {COLORS['text']};
                  font-family: '{FONT_FAMILY}'; font-size: {FONT_SIZE}pt; }}
        QDialog {{ border: 1px solid {COLORS['border']}; border-radius: {CORNER_RADIUS}px; }}
        QMenu {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']};
                 padding: 4px; }}
        QMenu::item {{ padding: 7px 24px 7px 12px; border-radius: 4px; }}
        QMenu::item:selected {{ background: {COLORS['surface_hover']}; }}
        QMenu::separator {{ height: 1px; background: {COLORS['border']}; margin: 4px 8px; }}
        QPushButton, QComboBox, QDoubleSpinBox {{ background: {COLORS['surface']};
            border: 1px solid {COLORS['border']}; border-radius: 5px; padding: 5px 8px; }}
        QPushButton:hover {{ background: {COLORS['surface_hover']}; }}
        QCheckBox {{ spacing: 8px; }}
        QCheckBox::indicator {{ width: 16px; height: 16px; }}
        QToolTip {{ background: {COLORS['surface']}; color: {COLORS['text']};
                   border: 1px solid {COLORS['border']}; }}
    """
