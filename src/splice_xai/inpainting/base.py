from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any
from PIL import Image


class InpaintingBackend(ABC):
    """
    Backend interface for inpainting engines (e.g., Replicate, local SD, etc.).
    Implementations should be stateless w.r.t. a single call.
    """

    @abstractmethod
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        model_type: str = "stable_diffusion",
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Returns a new RGB image with the masked region inpainted.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image (any mode). Will be converted to RGB.
        mask : PIL.Image.Image
            Grayscale or binary mask (white=edit, black=keep). Will be converted to L.
        model_type : str
            Logical model key (e.g., 'lama', 'stable_diffusion', 'flux', 'sdxl').
        prompt : Optional[str]
            Positive prompt (ignored for some models like LaMa).
        **kwargs : Any
            Backend-specific parameters (e.g., num_inference_steps, guidance_scale, negative_prompt).
        """
        raise NotImplementedError
