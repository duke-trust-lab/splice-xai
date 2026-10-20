"""Traditional removal baselines using the existing animal mask."""

import math

from PIL import Image, ImageFilter


def apply_traditional_perturbation(
    image,
    mask,
    method,
    blur_radius=15.0,
):
    """
    Apply a traditional perturbation through the existing animal mask.

    White mask pixels receive the perturbation.
    Black mask pixels preserve the original image.
    Intermediate mask values blend the two images.

    No resizing or inpainting is performed.
    """
    image = image.convert("RGB")

    if image.size != mask.size:
        raise ValueError("Mask and image must have the same dimensions.")

    mask = mask.convert("L")

    if method == "black_mask":
        edited = Image.new(
            "RGB",
            image.size,
            (0, 0, 0),
        )

    elif method == "blurring":
        if not math.isfinite(blur_radius) or blur_radius <= 0:
            raise ValueError("blur_radius must be finite and greater than zero.")

        edited = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    else:
        raise ValueError(f"Unsupported traditional perturbation: {method}")

    return Image.composite(
        edited,
        image,
        mask,
    )
