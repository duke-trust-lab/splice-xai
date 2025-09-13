from __future__ import annotations

import base64
import logging
import os
import random
import time
from io import BytesIO
from typing import Optional, Any, Dict, Iterable, Union

import requests
import replicate
from PIL import Image, ImageOps

from .base import InpaintingBackend

logger = logging.getLogger(__name__)


def _b64_png(img: Image.Image) -> str:
    """Encode PIL image as data URL PNG."""
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _ensure_mask_l(mask: Image.Image) -> Image.Image:
    # Convert to single-channel L; normalize if boolean-ish
    m = mask.convert("L")
    return m


def _extract_first_url(output: Union[str, Any, Iterable[Any]]) -> str:
    """
    Replicate outputs can be:
      - a single URL string,
      - an object with .url,
      - an iterable of either strings or objects with .url.
    """
    if isinstance(output, str):
        return output
    if hasattr(output, "url"):
        return output.url  # type: ignore[attr-defined]
    if isinstance(output, Iterable):
        for item in output:
            if isinstance(item, str):
                return item
            if hasattr(item, "url"):
                return item.url  # type: ignore[attr-defined]
    raise ValueError(f"Cannot extract URL from output of type {type(output)}")


class ReplicateInpainter(InpaintingBackend):
    """
    Replicate-backed inpainting.

    Notes
    -----
    - Model slugs are not hardcoded; we read them from `config.inpainting_models`
      if provided. This avoids drift across files.
    - Respects `config.max_retries` and `config.timeout_seconds` when present.
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Any] = None):
        token = api_key or os.getenv("REPLICATE_API_TOKEN")
        # replicate.Client() uses env var automatically; explicit is clearer for users.
        self.client = replicate.Client(api_token=token) if token else replicate.Client()
        self.config = config

    # -----------------------------
    # Public API
    # -----------------------------
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        model_type: str = "stable_diffusion",
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Image.Image:
        img = _ensure_rgb(image)
        m = _ensure_mask_l(mask)

        # Resolve model slug from config (preferred)
        model_slug = None
        if self.config and hasattr(self.config, "inpainting_models"):
            model_slug = self.config.inpainting_models.get(model_type)
        if not model_slug:
            # Fallback slugs (kept here as a safety net)
            model_slug = {
                "lama": "allenhooo/lama:cdac78a1bec5b23c07fd29692fb70baa513ea403a39e643c48ec5edadb15fe72",
                "stable_diffusion": "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                "flux": "zsxkib/flux-dev-inpainting:ca8350ff748d56b3ebbd5a12bd3436c2214262a4ff8619de9890ecc41751a008",
                "sdxl": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
            }.get(model_type)

        if not model_slug:
            raise ValueError(f"Unknown model_type '{model_type}'")

        payload = self._build_inputs(model_type, img, m, prompt, **kwargs)

        max_retries = getattr(self.config, "max_retries", 3) if self.config else 3
        timeout_seconds = getattr(self.config, "timeout_seconds", 60) if self.config else 60

        # Exponential backoff with jitter
        for attempt in range(1, max_retries + 1):
            try:
                output = self.client.run(model_slug, input=payload)
                url = _extract_first_url(output)
                content = self._download(url, timeout_seconds)
                out = Image.open(BytesIO(content)).convert("RGB")
                return out
            except Exception as e:
                if attempt >= max_retries:
                    logger.error(f"Replicate inpaint failed after {attempt} attempts: {e}")
                    raise
                sleep = min(8, 2 ** (attempt - 1)) + random.random() * 0.3
                logger.warning(f"Replicate attempt {attempt} failed: {e}. Retrying in {sleep:.1f}s.")
                time.sleep(sleep)

        # Unreachable, but satisfies type checker
        raise RuntimeError("Replicate inpaint failed unexpectedly.")

    # -----------------------------
    # Internals
    # -----------------------------
    def _build_inputs(
        self,
        model_type: str,
        image: Image.Image,
        mask: Image.Image,
        prompt: Optional[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Construct Replicate input dict with model-specific parameters.
        We standardize to PNG data URLs for both image and mask.
        """
        inputs: Dict[str, Any] = {
            "image": _b64_png(image),
            "mask": _b64_png(mask),
            # Some models want explicit size fields; harmless for others
            "height": image.height,
            "width": image.width,
        }

        neg = kwargs.get("negative_prompt")
        if model_type != "lama":
            if prompt:
                inputs["prompt"] = prompt
            if neg:
                inputs["negative_prompt"] = neg

        # Sensible defaults; callers can override via kwargs
        if model_type == "stable_diffusion":
            inputs.update(
                {
                    "num_outputs": 1,
                    "num_inference_steps": kwargs.get("num_inference_steps", 40),
                    "guidance_scale": kwargs.get("guidance_scale", 7.5),
                    # optional: "seed": kwargs.get("seed"),
                }
            )
        elif model_type == "flux":
            inputs.update(
                {
                    "strength": kwargs.get("strength", 0.9),
                    "num_inference_steps": kwargs.get("num_inference_steps", 36),
                    "guidance_scale": kwargs.get("guidance_scale", 10.0),
                }
            )
        elif model_type == "sdxl":
            # Many SDXL runners accept similar knobs; keep conservative defaults.
            inputs.update(
                {
                    "num_outputs": 1,
                    "num_inference_steps": kwargs.get("num_inference_steps", 35),
                    "guidance_scale": kwargs.get("guidance_scale", 6.5),
                }
            )
        elif model_type == "lama":
            # LaMa typically ignores text prompts; keep payload minimal.
            pass

        # Allow passthrough for any extra model-specific keys
        for k, v in kwargs.items():
            if k not in inputs and v is not None:
                inputs[k] = v

        return inputs

    def _download(self, url: str, timeout_seconds: int) -> bytes:
        headers = {
            "User-Agent": "splice-xai/0.1 (+https://github.com/your-org/splice-xai)"
        }
        resp = requests.get(url, timeout=timeout_seconds, headers=headers)
        resp.raise_for_status()
        return resp.content
