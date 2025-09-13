"""SPLICE-XAI: Semantically Plausible Localized Inpainting for Context-preserving Explanations"""

from .core.analyzer import SPLICEAnalyzer
from .core.config import InpaintingConfig
from .core.results import CounterfactualResult

__version__ = "0.1.0"
__all__ = ["SPLICEAnalyzer", "InpaintingConfig", "CounterfactualResult"]
