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


def _iter_boxes(
    boxes: Optional[Iterable],
) -> Iterable[Tuple[float, float, float, float]]:
    if not boxes:
        return []
    for b in boxes:
        if hasattr(b, "detach"):
            b = b.detach().cpu().numpy()
        b = np.asarray(b).astype(float).tolist()
        if len(b) == 4:
            x1, y1, x2, y2 = b
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            yield float(x1), float(y1), float(x2), float(y2)


def _clamp_box(box, w: int, h: int):
    x1, y1, x2, y2 = box
    return (
        max(0, min(x1, w - 1)),
        max(0, min(y1, h - 1)),
        max(0, min(x2, w - 1)),
        max(0, min(y2, h - 1)),
    )


def create_comparison_plot(
    original_path: str,
    result: CounterfactualResult,
    save_path: Optional[str] = None,
) -> Optional[str]:
    orig = load_image(original_path)
    W, H = orig.size
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Check if we should draw the 15% margin
    is_sam = getattr(result, "use_sam", False)

    try:
        axes[0].imshow(orig)
        axes[0].set_title(f"Original {'(SAM)' if is_sam else '(+15% Area)'}")
        axes[0].axis("off")

        for box in _iter_boxes(result.original_boxes):
            x1, y1, x2, y2 = _clamp_box(box, W, H)
            # Solid Lime = Model Prediction
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

            # ONLY Draw 15% Area Dashed Box if NOT SAM
            if not is_sam:
                bw, bh = x2 - x1, y2 - y1
                ex1, ey1 = max(0, x1 - (bw * 0.036)), max(0, y1 - (bh * 0.036))
                ex2, ey2 = min(W, x2 + (bw * 0.036)), min(H, y2 + (bh * 0.036))
                axes[0].add_patch(
                    patches.Rectangle(
                        (ex1, ey1),
                        ex2 - ex1,
                        ey2 - ey1,
                        linewidth=1.5,
                        edgecolor="cyan",
                        linestyle="--",
                        facecolor="none",
                    )
                )

        # Panel 2: Mask
        if result.mask is not None:
            axes[1].imshow(
                np.array(result.mask.convert("L")), cmap="gray", vmin=0, vmax=255
            )
            axes[1].set_title("Inpainting Mask")
            axes[1].axis("off")

        # Panel 3: Inpainted Result
        if result.image is not None:
            axes[2].imshow(result.image)
            axes[2].set_title("Result")
            axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
