from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Iterable, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from splice_xai.core.results import CounterfactualResult
from splice_xai.utils.file_io import load_image

logger = logging.getLogger(__name__)


def _iter_boxes(boxes: Optional[Iterable]) -> Iterable[Tuple[float, float, float, float]]:
    if not boxes:
        return []
    for b in boxes:
        # tolerate tensors/arrays/tuples/lists
        if hasattr(b, "detach"):
            b = b.detach().cpu().numpy()
        b = np.asarray(b).astype(float).tolist()
        if len(b) == 4:
            x1, y1, x2, y2 = b
            # Normalize any reversed corners
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            yield float(x1), float(y1), float(x2), float(y2)


def _clamp_box(box, w: int, h: int):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(x1, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    x2 = max(0.0, min(x2, w - 1))
    y2 = max(0.0, min(y2, h - 1))
    return x1, y1, x2, y2


def create_comparison_plot(
    original_path: str,
    result: CounterfactualResult,
    save_path: Optional[str] = None,
) -> Optional[str]:
    """
    Create a side-by-side visualization: original, mask, and inpainted image with boxes.

    Returns the save_path if provided, else None. Always closes the figure.
    """
    # Load with EXIF orientation handling
    orig = load_image(original_path)
    W, H = orig.size

    # Prepare figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    try:
        # Panel 1: Original
        axes[0].imshow(orig)
        axes[0].set_title("Original")
        axes[0].axis("off")

        for box in _iter_boxes(result.original_boxes):
            x1, y1, x2, y2 = _clamp_box(box, W, H)
            axes[0].add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none",
                )
            )

        # Panel 2: Mask (force L mode, show as grayscale)
        if result.mask is not None:
            mask_L = result.mask.convert("L")
            axes[1].imshow(np.array(mask_L), cmap="gray", vmin=0, vmax=255)
            axes[1].set_title("Mask")
            axes[1].axis("off")
        else:
            axes[1].axis("off")
            axes[1].set_title("Mask (none)")

        # Panel 3: Result
        if result.image is not None:
            # If result size differs, we still display it; boxes are drawn in its own coordinate space
            res_img = result.image
            axes[2].imshow(res_img)
            axes[2].set_title("Inpainted")
            axes[2].axis("off")

            w2, h2 = res_img.size
            for box in _iter_boxes(result.result_boxes):
                x1, y1, x2, y2 = _clamp_box(box, w2, h2)
                axes[2].add_patch(
                    patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                )
        else:
            axes[2].axis("off")
            axes[2].set_title("Inpainted (none)")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Visualization saved to {save_path}")
            return save_path
        return None
    finally:
        plt.close(fig)
