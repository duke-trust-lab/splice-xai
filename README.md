
# SPLICE: Semantically Plausible Localized Inpainting for Context-preserving Explanations in Ecology

This is a Python package for generating counterfactual explanations of object detections by combining **YOLO detection**, **SAM segmentation**, and **Replicate inpainting models**.  

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


### All CLI Arguments for Customization

#### Input Arguments (Required - Choose One)
- **`--image PATH`**: Path to a single image file for processing
- **`--batch PATH`**: Path to a directory containing multiple images for batch processing

#### Model Configuration (Required)
- **`--yolo-model PATH`**: Path to your trained YOLO model file (.pt format)
- **`--mask PATH`**: Path to a custom mask image (optional - if not provided, SAM will generate masks automatically)

#### Processing Mode
- **`--mode {remove,replace,background,all}`**: Counterfactual generation mode (default: remove)
  - `remove`: Remove detected objects from images
  - `replace`: Replace objects with new content using prompts
  - `background`: Change background around detected objects
  - `all`: Run all three modes on each image

#### Inpainting Configuration
- **`--inpaint-model {lama,stable_diffusion,flux,sdxl}`**: Choose inpainting backend (default: stable_diffusion)
  - `lama`: Fast, lightweight model good for simple object removal
  - `stable_diffusion`: Balanced quality and speed for most use cases
  - `flux`: High-quality results for complex scenes
  - `sdxl`: Premium quality for detailed inpainting tasks
- **`--prompt TEXT`**: Custom text prompt for object replacement or inpainting guidance (required for replace mode)
- **`--negative-prompt TEXT`**: Negative prompt to avoid unwanted elements in generated content
- **`--background TYPE`**: Background type for background change mode (default: rocky)
- **`--target-label CLASS`**: Specific object class to target for replacement (useful when multiple objects are detected)

#### Output Options
- **`--output DIR`**: Output directory for results (default: results)
- **`--save-images`**: Save the generated/modified images to disk
- **`--visualize`**: Create before/after comparison visualizations with bounding boxes
- **`--csv PATH`**: Save processing results and statistics to CSV file

#### Advanced Configuration
- **`--conf-threshold FLOAT`**: YOLO confidence threshold for object detection (default: 0.4)
  - Lower values detect more objects but may include false positives
  - Higher values are more selective but may miss objects
- **`--seed INT`**: Random seed for reproducible results (default: 42)
- **`--device {auto,cpu,cuda}`**: Computation device (default: auto)
  - `auto`: Automatically choose GPU if available, otherwise CPU
  - `cpu`: Force CPU usage
  - `cuda`: Force GPU usage (requires CUDA setup)

#### Usage Examples by Scenario
**High-quality object removal with visualization:**
```bash
splice-xai --image photo.jpg --yolo-model model.pt --mode remove --inpaint-model sdxl --save-images --visualize
```

**Batch processing with custom confidence threshold:**
```bash
splice-xai --batch ./dataset/ --yolo-model model.pt --mode all --conf-threshold 0.6 --csv results.csv
```

**Replace specific objects with custom prompts:**
```bash
splice-xai --image scene.jpg --yolo-model model.pt --mode replace --prompt "a red car" --target-label vehicle --negative-prompt "blurry, low quality"
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
## Adapting SPLICE for Different Detection Models

SPLICE currently uses YOLO for object detection, but can be adapted to work with other detection frameworks. To integrate a different model:

### Code Modifications Required

1. **Create a new detector class** following the interface pattern in `src/splice_xai/detection/yolo_detector.py`:
   ```python
   class YourDetector:
       def __init__(self, model_path: str, device: Optional[str] = None):
           # Initialize your model
           pass
       
       def detect(self, image: np.ndarray, conf_threshold: float = 0.4):
           # Return detection results
           pass
       
       def get_top_detection(self, image: np.ndarray, conf_threshold: float = 0.4) -> ObjectDetectionResult:
           # Return top detection in ObjectDetectionResult format
           pass
   ```

2. **Update the analyzer initialization** in `src/splice_xai/core/analyzer.py`:
   ```python
   # Replace this line:
   self.detector = YOLODetector(yolo_model, device=getattr(self.config, "device", "auto"))
   
   # With:
   self.detector = YourDetector(model_path, device=getattr(self.config, "device", "auto"))
   ```

3. **Ensure your detector returns the expected format**:
   - `bbox`: List of 4 coordinates [x1, y1, x2, y2]
   - `class_id`: Integer class ID
   - `confidence`: Float confidence score
   - `class_names`: Dictionary mapping class IDs to names

### Key Interface Requirements

Your detector must provide:
- Image input as `np.ndarray` (HWC RGB format)
- Bounding box output as `[x1, y1, x2, y2]` coordinates
- Confidence thresholding capability
- Class name mapping for interpretable results

For detailed implementation examples or assistance adapting specific models, please open an issue on the repository.

---

## Contributing

- Use feature branches + pull requests.
- Add tests under `tests/` for new features.
- Run `pytest` before submitting PRs.
- Follow PEP8/black for formatting.

---

## License

MIT License – see [LICENSE](LICENSE) for details.
