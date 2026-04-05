# Insegment

Interactive instance segmentation annotation tool. Load microscopy images (or any images), run a segmentation model, then correct the predictions with clicks -- add missed objects, remove false positives, reclassify, draw polygons.

Built for researchers who need fast, accurate annotations without the overhead of heavyweight labeling platforms.

## Features

- **Model-agnostic**: Plug in any instance segmentation model via a simple Python adapter
- **Click to correct**: Left-click to add, right-click to remove, 1/2/3 to reclassify
- **Polygon drawing**: Press P to draw freehand polygon annotations
- **Pan & zoom**: Shift+drag to pan, scroll to zoom, F to fit-to-screen
- **COCO JSON export**: Industry-standard format, compatible with most training pipelines
- **Semi-annotation mode**: Load existing COCO annotations and correct them
- **No model required**: Use without a model for pure manual annotation

## Install

```bash
pip install insegment
```

Or install from source:

```bash
git clone https://github.com/SergioCZ28/Insegment.git
cd Insegment
pip install -e .
```

## Quick Start

### Annotation-only mode (no model)

```bash
insegment serve --semiannotation-dir ./my_images
```

Where `my_images/` contains PNG files and an `_annotations.coco.json` file.

### With a model

```bash
insegment serve --model my_models:MySegmenter --checkpoint model.pth --tiff-dir ./data
```

Then open http://localhost:5000 in your browser.

## Writing a Model Adapter

Create a Python class that inherits from `BaseSegmenter`:

```python
from insegment.models import BaseSegmenter
from insegment.models.base import SegmentationResult
import numpy as np

class MySegmenter(BaseSegmenter):
    def __init__(self, checkpoint_path=None, **kwargs):
        # Load your model here
        self.model = load_my_model(checkpoint_path)

    def predict(self, image: np.ndarray) -> SegmentationResult:
        # Run inference -- return masks, boxes, classes, scores
        results = self.model(image)
        return SegmentationResult(
            masks=results.masks,       # (H, W) int array, 0=bg, 1..N=instances
            bboxes=results.boxes,      # (N, 4) array, [x1, y1, x2, y2]
            class_ids=results.classes,  # (N,) int array
            scores=results.scores,     # (N,) float array
        )

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "cell", 1: "debris"}
```

Then run:

```bash
insegment serve --model my_module:MySegmenter --checkpoint model.pth
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **1 / 2 / 3** | Set active class or reclassify selected annotation |
| **P** | Toggle polygon drawing mode |
| **G** | Toggle grid overlay |
| **F** | Fit image to screen |
| **Escape** | Deselect / cancel polygon |
| **Ctrl+S** | Save annotations |
| **Scroll** | Zoom in/out |
| **Shift+drag** | Pan |
| **Left-click** | Add annotation or select existing |
| **Right-click** | Remove nearest annotation |

## Export Formats

Currently exports to **COCO JSON** format. More formats (YOLO, CSV, Pascal VOC) coming soon.

## Roadmap

- [ ] Customizable labels (rename, recolor, add/remove)
- [ ] Multiple export formats (YOLO, CSV, VOC)
- [ ] Annotation shapes (circle, rectangle, point, polygon)
- [ ] Progress bar for model inference
- [ ] Auto-save / persistence
- [ ] Dark mode

## License

MIT
