"""Base class for segmentation model adapters.

To use your own model with Insegment, create a class that inherits from
BaseSegmenter and implement the `predict()` method.

Example:
    class MyModel(BaseSegmenter):
        def __init__(self, checkpoint_path, **kwargs):
            self.model = load_my_model(checkpoint_path)

        def predict(self, image):
            results = self.model(image)
            return SegmentationResult(
                masks=results.masks,          # (H, W) int array, 0=background, 1..N=instances
                bboxes=results.boxes,          # (N, 4) float array, each row [x1, y1, x2, y2]
                class_ids=results.classes,     # (N,) int array, class index per detection
                scores=results.confidences,    # (N,) float array, confidence per detection
            )

        @property
        def class_names(self):
            return {0: "cell", 1: "debris"}

Then run:
    insegment serve --model my_module:MyModel --checkpoint model.pth
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationResult:
    """Container for model predictions.

    Attributes:
        masks: Instance segmentation mask, shape (H, W). Integer array where
               0 = background and each positive integer is a unique instance ID.
        bboxes: Bounding boxes, shape (N, 4). Each row is [x1, y1, x2, y2]
                in pixel coordinates.
        class_ids: Class label per detection, shape (N,). Integer indices that
                   map to the keys in `class_names`.
        scores: Confidence score per detection, shape (N,). Float values
                between 0.0 and 1.0.
    """

    masks: np.ndarray
    bboxes: np.ndarray
    class_ids: np.ndarray
    scores: np.ndarray


class BaseSegmenter(ABC):
    """Abstract base class for segmentation models.

    Subclass this and implement `predict()` and `class_names` to plug any
    instance segmentation model into Insegment.
    """

    @abstractmethod
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """Run inference on a single image.

        Args:
            image: Grayscale or RGB image as a numpy array.
                   Shape is (H, W) for grayscale or (H, W, 3) for RGB.
                   Pixel values are uint8 (0-255) or float32 (0.0-1.0).

        Returns:
            SegmentationResult with masks, bboxes, class_ids, and scores.
        """
        ...

    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]:
        """Mapping from class index to human-readable name.

        Example: {0: "single-cell", 1: "clump", 2: "debris"}
        """
        ...
