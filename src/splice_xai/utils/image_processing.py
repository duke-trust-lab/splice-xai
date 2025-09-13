from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


def _to_L(mask: Image.Image) -> Image.Image:
    return mask.convert("L") if mask.mode != "L" else mask


def invert_mask(mask: Image.Image) -> Image.Image:
    """Invert a mask (0↔255). Works for any input mode."""
    m = _to_L(mask)
    arr = np.array(m, dtype=np.uint8)
    inv = 255 - arr
    return Image.fromarray(inv, mode="L")


def dilate_mask(mask: Image.Image, pixels: int = 8) -> Image.Image:
    """Binary-ish dilation using MaxFilter; idempotent for pixels<=0."""
    if pixels <= 0:
        return _to_L(mask)
    m = _to_L(mask)
    k = 2 * pixels + 1
    return m.filter(ImageFilter.MaxFilter(size=k))


def feather_mask(mask: Image.Image, radius: int = 3) -> Image.Image:
    """Feather/smooth mask edges with Gaussian blur on L channel."""
    if radius <= 0:
        return _to_L(mask)
    m = _to_L(mask)
    return m.filter(ImageFilter.GaussianBlur(radius=radius))


def resize_for_model(
    image: Image.Image,
    mask: Image.Image,
    model_name: str,
    default_sizes: Dict[str, Optional[Tuple[int, int]]],
) -> tuple[Image.Image, Image.Image]:
    """
    Resize image/mask pair for the target model.
    If size is None, return originals.
    """
    size = default_sizes.get(model_name)
    if size is None:
        return image, _to_L(mask)

    w, h = size
    if not isinstance(w, int) or not isinstance(h, int):
        logger.warning(f"Invalid target size for '{model_name}': {size}. Returning originals.")
        return image, _to_L(mask)

    img_resized = image.convert("RGB").resize((w, h), Image.LANCZOS)
    mask_resized = _to_L(mask).resize((w, h), Image.NEAREST)
    return img_resized, mask_resized
