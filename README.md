
# SPLICE: Semantically Plausible Localized Inpainting for Context-preserving Explanations in Ecology

This is a Python package for generating counterfactual explanations of object detections by combining **YOLO detection**, **SAM segmentation**, and **Replicate inpainting models**.  
---

## Features

- Remove objects from ecological images while preserving plausible backgrounds.
- Replace objects with semantically guided alternatives.
- Change backgrounds around detected objects.
- Batch processing of entire datasets.
- Visualize original vs. inpainted images with bounding boxes.
- Configurable backends (LaMa, Stable Diffusion, Flux, SDXL).

---

## Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/TRUST-Lab/splice-xai.git
cd splice-xai
pip install -e .
```

Dependencies: Python 3.9–3.11, PyTorch, Ultralytics YOLO, Replicate API client.  
For optional GPU acceleration, ensure CUDA is available.

---

## Quickstart

### 1. Single image via CLI

```bash
splice-xai   --image data/images/seal1.jpg   --yolo-model data/models/seal.pt   --mode remove   --save-images   --output results/
```

### 2. Replace object with prompt

```bash
splice-xai   --image data/images/seal1.jpg   --yolo-model data/models/seal.pt   --mode replace   --prompt "a wooden boat on water"   --target-label boat   --save-images
```

### 3. Batch removal

```bash
splice-xai   --batch data/images/   --yolo-model data/models/seal.pt   --mode remove   --save-images   --csv results/batch.csv
```

---

## Python API

```python
from splice_xai import SPLICEAnalyzer, InpaintingConfig

analyzer = SPLICEAnalyzer("data/models/model_name.pt", InpaintingConfig())

result = analyzer.remove_object("data/images/image_name.jpg")
before, after = result.confidence_before, result.confidence_after
print(result.outcome, before, after)

if result.image:
    result.image.save("output.png")
```

---

## Requirements

- **YOLO weights**: provide your own `.pt` file in `data/models/` (trained on your dataset).
- **SAM checkpoint**: will be auto-downloaded if not present in `data/models/`.
- **Replicate API**: set your token in the environment:

```bash
export REPLICATE_API_TOKEN=your_token_here
```

---

## Contributing

- Use feature branches + pull requests.
- Add tests under `tests/` for new features.
- Run `pytest` before submitting PRs.
- Follow PEP8/black for formatting.

---

## License

MIT License – see [LICENSE](LICENSE) for details.
