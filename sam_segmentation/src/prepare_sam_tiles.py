#!/usr/bin/env python3
"""Create traceable 512x512, original-resolution crops for SAM_training.py.

Edit INPUT_ROOT and OUTPUT_DIR. Run: python -u prepare_sam_tiles.py
The script recursively searches INPUT_ROOT, skips tiles that are almost entirely
outside the specimen (optional), and writes a CSV mapping each tile to its
source image and original pixel coordinates. Originals are never modified.
Only load tiles from OUTPUT_DIR into SAM_training.py's IMAGE_FOLDER; its main()
reads a flat directory, not nested sample folders.
"""

import csv
import hashlib
import math
import re
from pathlib import Path

import cv2
import numpy as np

INPUT_ROOT = Path("/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/Computations/CFRP-microstructure-segmentation/data/large_real_images")
OUTPUT_DIR = Path("/home/BeeGFS/ExtLabs/LEME/marabkoohichetabi/Computations/CFRP-microstructure-segmentation/data/real_images_sam_tiles")

TILE_SIZE = 512
OVERLAP = 64  # 0 for non-overlapping tiles; 64 provides overlap near boundaries
MIN_SPECIMEN_FRACTION = 0.80  # set to 0.0 to keep all tiles
SKIP_OUTSIDE_SPECIMEN = False  # first run False; inspect contact sheet/tiles before enabling
# Outside specimen in the provided panorama appears relatively uniform and dark;
# the check below is only a heuristic, not a reliable specimen segmentation.
OUTSIDE_STD_THRESHOLD = 9.0
OUTSIDE_MEAN_THRESHOLD = 110.0

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MANIFEST_NAME = "sam_tiles_manifest.csv"


def positions(length, tile_size, overlap):
    if length < tile_size:
        return []
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("OVERLAP must be smaller than TILE_SIZE")
    points = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if points[-1] != last:
        points.append(last)
    return points


def slug(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "root"


def read_image(path):
    # imdecode supports non-ASCII paths on platforms where cv2.imread may not.
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def keep_tile(tile):
    if not SKIP_OUTSIDE_SPECIMEN:
        return True, 1.0
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    # Fixed coarse blocks: preserve large specimen regions and ignore isolated pits.
    block = 32
    gh, gw = gray.shape
    usable = gray[: gh // block * block, : gw // block * block]
    blocks = usable.reshape(gh // block, block, gw // block, block).transpose(0, 2, 1, 3)
    std = blocks.std(axis=(2, 3))
    mean = blocks.mean(axis=(2, 3))
    fraction = float(np.mean((std >= OUTSIDE_STD_THRESHOLD) | (mean >= OUTSIDE_MEAN_THRESHOLD)))
    return fraction >= MIN_SPECIMEN_FRACTION, fraction


def main():
    src = INPUT_ROOT.resolve()
    dst = OUTPUT_DIR.resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {src}")
    if dst == src or src in dst.parents:
        raise ValueError("OUTPUT_DIR must be outside INPUT_ROOT to avoid re-tiling outputs")
    dst.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    if not images:
        raise RuntimeError(f"No supported image files found under {src}")
    manifest = dst / MANIFEST_NAME
    if manifest.exists():
        raise FileExistsError(f"Manifest already exists: {manifest}; use a new OUTPUT_DIR for each run")

    fields = ["tile_filename", "source_path", "source_relative_path", "tube_id", "sample_id",
              "source_width", "source_height", "x", "y", "tile_width", "tile_height",
              "estimated_specimen_fraction"]
    total = 0
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for path in images:
            rel = path.relative_to(src)
            parts = rel.parts
            # For INPUT_ROOT/tube/sample/image.ext, keep those identifiers.
            # Otherwise IDs default to the first parent and its subfolder (or 'unknown').
            tube_id = parts[0] if len(parts) >= 3 else "unknown"
            sample_id = parts[1] if len(parts) >= 3 else (parts[0] if len(parts) >= 2 else "unknown")
            image = read_image(path)
            if image is None:
                print(f"SKIP unreadable: {path}", flush=True)
                continue
            h, w = image.shape[:2]
            if min(h, w) < TILE_SIZE:
                print(f"SKIP too small ({w}x{h}): {path}", flush=True)
                continue
            xs, ys = positions(w, TILE_SIZE, OVERLAP), positions(h, TILE_SIZE, OVERLAP)
            image_key = slug(str(rel.with_suffix("")).replace("/", "__"))
            digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:8]
            written = 0
            for y in ys:
                for x in xs:
                    tile = image[y:y + TILE_SIZE, x:x + TILE_SIZE]
                    accepted, fraction = keep_tile(tile)
                    if not accepted:
                        continue
                    name = f"{image_key}__{digest}__x{x:06d}_y{y:06d}.png"
                    out = dst / name
                    if out.exists():
                        raise FileExistsError(f"Refusing to overwrite: {out}")
                    if not cv2.imwrite(str(out), tile):
                        raise OSError(f"Could not write: {out}")
                    writer.writerow({"tile_filename": name, "source_path": str(path),
                                     "source_relative_path": str(rel), "tube_id": tube_id,
                                     "sample_id": sample_id, "source_width": w, "source_height": h,
                                     "x": x, "y": y, "tile_width": TILE_SIZE,
                                     "tile_height": TILE_SIZE,
                                     "estimated_specimen_fraction": round(fraction, 4)})
                    written += 1
                    total += 1
            f.flush()
            print(f"{rel}: {w}x{h}, {len(xs) * len(ys)} possible tiles, {written} written", flush=True)
    print(f"Done: {total} tiles in {dst}; coordinate manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
