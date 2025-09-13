from PIL import Image
import logging

logger = logging.getLogger(__name__)

def validate_image_input(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be PIL Image")
    if image.mode != "RGB":
        logger.warning(f"Converting image from {image.mode} to RGB")
        return image.convert("RGB")
    return image

def validate_mask_input(mask: Image.Image) -> Image.Image:
    if not isinstance(mask, Image.Image):
        raise TypeError("mask must be PIL Image")
    if mask.mode not in ["L", "1"]:
        logger.warning(f"Converting mask from {mask.mode} to L")
        return mask.convert("L")
    return mask
