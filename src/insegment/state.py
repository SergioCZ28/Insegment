"""Module-level state and constants shared by the Flask routes.

Everything here is a single-process global — a deliberate trade-off for a
local annotation tool. If Insegment is ever deployed multi-process, this
is the module to replace with a proper session/DB store.
"""

# A palette of 10 visually distinct colors. When a user adds a new class,
# it automatically gets the next color from this list (cycling back to the
# start after 10). Users can override individual colors via the UI.
DEFAULT_COLORS = [
    "#4caf50",  # green
    "#f44336",  # red
    "#ffeb3b",  # yellow
    "#2196f3",  # blue
    "#9c27b0",  # purple
    "#ff9800",  # orange
    "#00bcd4",  # cyan
    "#e91e63",  # pink
    "#8bc34a",  # lime
    "#795548",  # brown
]

# Supported image extensions (case-insensitive).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Global state -- shared across requests (single-process server).
STATE = {
    "segmenter": None,           # Model instance (BaseSegmenter subclass) or None
    "image_dir": None,           # Path to directory with images
    "images": [],                # [{path: str, filename: str}, ...] sorted
    "output_dir": None,          # Path for exported annotation files
    "annotations": {},           # int index -> annotation data for that image
    "cell_radius": 4,            # default radius for manually added cells (pixels)
    "class_names": {0: "single-cell", 1: "clump", 2: "debris"},  # default labels
    "class_colors": {0: "#4caf50", 1: "#f44336", 2: "#ffeb3b"},  # default colors
}

# Inference job tracking for SSE progress
_inference_jobs = {}
