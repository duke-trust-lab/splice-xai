from __future__ import annotations

import gc
import logging

import torch

logger = logging.getLogger(__name__)


class GPUResourceManager:
    """
    Context manager to keep GPU memory pressure low between heavy ops.
    Cleans caches on exit; safe on CPU-only machines (no-ops).
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_gpu_memory()

    @staticmethod
    def cleanup_gpu_memory():
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Optional: free cached blocks on all devices
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"GPU cleanup skipped: {e}")
