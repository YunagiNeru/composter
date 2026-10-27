import sys
from pathlib import Path

app_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(app_root))

from desktop_pet.sprite_art import generate_sprites


if __name__ == "__main__":
    generated = generate_sprites(app_root / "assets" / "sprites")
    print(f"generated {len(generated)} sprite assets")
