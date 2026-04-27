from pathlib import Path
import random

from PIL import Image


def image_index(path: Path) -> int:
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    return 10**9


def main() -> None:
    folder = Path(__file__).resolve().parent
    output_path = folder / "total.png"
    image_paths = sorted(folder.glob("*.tif"), key=image_index)
    if len(image_paths) != 400:
        raise ValueError(f"Expected 400 tif images, found {len(image_paths)}")
    random.shuffle(image_paths)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    tile_w, tile_h = images[0].size
    rows, cols = 20, 20
    max_canvas_w, max_canvas_h = 4000, 4000
    scale = min(1.0, max_canvas_w / (cols * tile_w), max_canvas_h / (rows * tile_h))
    tile_w = max(1, int(tile_w * scale))
    tile_h = max(1, int(tile_h * scale))
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x, y = col * tile_w, row * tile_h
        canvas.paste(img.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (x, y))
    canvas.save(output_path, optimize=True, compress_level=9)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
