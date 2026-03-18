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


def letterbox_image(
    image: Image.Image,
    target_size: Tuple[int, int],
    fill_color: int | Tuple[int, int, int] = 0,
    resample=Image.LANCZOS,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Resize image to target_size while preserving aspect ratio using padding.
    Returns: (padded_image, (left, top, right, bottom) padding offsets)
    """
    iw, ih = image.size
    tw, th = target_size
    scale = min(tw / iw, th / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image_resized = image.resize((nw, nh), resample=resample)

    # Create canvas with fill color (0 for masks/black, (128,128,128) for images)
    new_image = Image.new(image.mode, target_size, fill_color)

    # Calculate offsets to center the image
    left = (tw - nw) // 2
    top = (th - nh) // 2
    new_image.paste(image_resized, (left, top))

    return new_image, (left, top, left + nw, top + nh)


def resize_for_model(
    image: Image.Image,
    mask: Image.Image,
    model_name: str,
    default_sizes: Dict[str, Optional[Tuple[int, int]]],
) -> tuple[Image.Image, Image.Image]:
    """
    Resize image/mask pair using letterboxing to preserve aspect ratio.
    """
    size = default_sizes.get(model_name)
    if size is None:
        return image, _to_L(mask)

    w, h = size
    if not isinstance(w, int) or not isinstance(h, int):
        logger.warning(
            f"Invalid target size for '{model_name}': {size}. Returning originals."
        )
        return image, _to_L(mask)

    # 1. Letterbox the Image (using gray padding for natural background)
    img_resized, _ = letterbox_image(
        image.convert("RGB"), (w, h), fill_color=(128, 128, 128), resample=Image.LANCZOS
    )

    # 2. Letterbox the Mask (using black padding for the mask)
    mask_resized, _ = letterbox_image(
        _to_L(mask), (w, h), fill_color=0, resample=Image.NEAREST
    )

    return img_resized, mask_resized
