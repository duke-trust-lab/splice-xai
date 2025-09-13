from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal


@dataclass
class InpaintingConfig:
    # Detection / segmentation
    yolo_conf_threshold: float = 0.4
    mask_union_threshold: Optional[float] = None

    # Execution
    max_retries: int = 3
    timeout_seconds: int = 60
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # Model sizing for backends
    default_model_sizes: Dict[str, Optional[Tuple[int, int]]] = field(
        default_factory=lambda: {
            "stable_diffusion": (512, 512),
            "flux": (1024, 1024),
            "sdxl": (1024, 1024),
            "lama": None,
        }
    )

    # Prompts
    default_removal_prompt: str = "seamless natural background, consistent lighting"
    default_replacement_prompt: str = "photorealistic object in natural setting"
    default_negative_prompt: str = "duplicate, distortion, artifacts, low quality"

    # Background presets
    available_backgrounds: Dict[str, str] = field(
        default_factory=lambda: {
            "forest": "dense green forest with tall trees",
            "mountain": "majestic mountain landscape",
            "beach": "tropical beach with white sand",
            "city": "modern urban cityscape",
            "desert": "vast sandy desert",
            "countryside": "peaceful rural countryside",
            "garden": "beautiful botanical garden",
            "winter": "snowy winter landscape",
            "rocky": "rugged rocky terrain",
        }
    )

    # SAM checkpoint management
    sam_checkpoint_url: str = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    sam_checkpoint_path: str = "data/models/sam_vit_b_01ec64.pth"

    def __post_init__(self):
        if self.mask_union_threshold is None:
            self.mask_union_threshold = self.yolo_conf_threshold

    @property
    def inpainting_models(self) -> Dict[str, str]:
        # Mapping for Replicate model slugs; keep these pinned to known-good revisions
        return {
            "lama": "allenhooo/lama:cdac78a1bec5b23c07fd29692fb70baa513ea403a39e643c48ec5edadb15fe72",
            "stable_diffusion": "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
            "flux": "zsxkib/flux-dev-inpainting:ca8350ff748d56b3ebbd5a12bd3436c2214262a4ff8619de9890ecc41751a008",
            "sdxl": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        }
