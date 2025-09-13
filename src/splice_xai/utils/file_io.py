from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Iterable, Set

import numpy as np
from PIL import Image, ImageOps, ImageFile

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate partial files


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image as RGB with EXIF orientation corrected."""
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    img = Image.open(p)
    img = ImageOps.exif_transpose(img)  # honor orientation tag
    return img.convert("RGB")


def load_mask(mask_path: str | Path) -> Image.Image:
    """
    Load a mask as L and binarize to {0,255} using threshold 128.
    """
    p = Path(mask_path)
    if not p.is_file():
        raise FileNotFoundError(f"Mask not found: {p}")
    m = Image.open(p).convert("L")
    arr = (np.array(m, dtype=np.uint8) > 128).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def _union_fieldnames(dicts: Iterable[Dict[str, Any]]) -> List[str]:
    """
    Build a stable field order: keys present in the first dict first,
    then the remaining keys in sorted order.
    """
    it = iter(dicts)
    try:
        first = next(it)
    except StopIteration:
        return []
    head = list(first.keys())
    rest_keys: Set[str] = set(head)
    for d in it:
        rest_keys.update(d.keys())
    tail = sorted(k for k in rest_keys if k not in head)
    return head + tail


def save_results_to_csv(results: List[Dict[str, Any]], csv_path: str | Path) -> None:
    """Save list of result dicts to CSV, tolerating heterogeneous keys."""
    if not results:
        logger.warning("No results to save.")
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = _union_fieldnames(results)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logger.info(f"Results saved to {csv_path}")
