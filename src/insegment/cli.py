"""Command-line interface for Insegment.

This is what runs when you type `insegment` in the terminal.
The `main()` function is registered as an "entry point" in pyproject.toml,
meaning pip creates a small script that calls main() when you type `insegment`.

Usage:
    insegment serve                          # annotation-only mode (no model)
    insegment serve --image-dir ./images     # load a folder of images
    insegment serve --model mymod:MyModel    # with a custom model
    insegment serve --port 8080              # custom port
"""

import argparse
import importlib
import logging
import sys

from insegment import __version__

logger = logging.getLogger(__name__)


def load_model_class(model_string):
    """Load a model class from a 'module:ClassName' string.

    This is how users tell Insegment which model to use. The format is:
        module_path:ClassName

    For example:
        my_models.bacdetr:BacDETRAdapter

    This means: "import the module my_models.bacdetr, then get the class
    BacDETRAdapter from it."

    It's the same pattern used by tools like uvicorn (a web server) and
    gunicorn (another web server) for loading apps.
    """
    if ":" not in model_string:
        logger.error("Model must be in 'module:ClassName' format, got '%s'", model_string)
        logger.error("Example: insegment serve --model my_models:MySegmenter")
        sys.exit(1)

    module_path, class_name = model_string.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        logger.error("Could not import module '%s': %s", module_path, e)
        logger.error("Make sure the module is installed or on your Python path.")
        sys.exit(1)

    try:
        cls = getattr(module, class_name)
    except AttributeError:
        logger.error("Module '%s' has no class '%s'", module_path, class_name)
        available = [x for x in dir(module) if not x.startswith("_")]
        logger.error("Available names: %s", ", ".join(available))
        sys.exit(1)

    return cls


def cmd_serve(args):
    """Start the annotation server."""
    from insegment.app import app, configure_app

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )

    # Handle deprecated --tiff-dir
    image_dir = args.image_dir
    if args.tiff_dir:
        logger.warning("--tiff-dir is deprecated. Use --image-dir instead.")
        if not image_dir:
            image_dir = args.tiff_dir

    # Load model if specified
    segmenter = None
    if args.model:
        model_cls = load_model_class(args.model)
        # Pass checkpoint and any extra kwargs to the model constructor
        kwargs = {}
        if args.checkpoint:
            kwargs["checkpoint_path"] = args.checkpoint
        logger.info("Loading model: %s", args.model)
        segmenter = model_cls(**kwargs)
        logger.info("Model loaded. Classes: %s", segmenter.class_names)

    configure_app(
        segmenter=segmenter,
        image_dir=image_dir,
        output_dir=args.output_dir,
        cell_radius=args.cell_radius,
        semiannotation_dir=args.semiannotation_dir,
    )

    if segmenter is None:
        logger.info("No model loaded -- running in annotation-only mode.")
        logger.info("Use --model module:ClassName to enable model inference.")

    if not image_dir:
        logger.info("No --image-dir specified -- use Browse Folder in the UI to load images.")

    logger.info("Starting Insegment v%s on http://localhost:%d", __version__, args.port)
    app.run(host="0.0.0.0", port=args.port, debug=False)


def main():
    """Main entry point for the `insegment` CLI command."""
    parser = argparse.ArgumentParser(
        prog="insegment",
        description="Insegment -- Interactive instance segmentation annotation tool",
    )
    parser.add_argument(
        "--version", action="version", version=f"insegment {__version__}"
    )

    # Subcommands: currently just "serve", but we can add more later
    # (like "export", "convert", etc.)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- serve command ---
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the annotation server",
        description="Launch the web-based annotation interface.",
    )
    serve_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model adapter in 'module:ClassName' format (e.g., my_models:MySegmenter)",
    )
    serve_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint file (passed to model constructor)",
    )
    serve_parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to directory with image files (PNG, JPEG, TIFF, etc.)",
    )
    serve_parser.add_argument(
        "--tiff-dir",
        type=str,
        default=None,
        help=argparse.SUPPRESS,  # Deprecated, hidden from help
    )
    serve_parser.add_argument(
        "--output-dir",
        type=str,
        default="./annotations_output",
        help="Output directory for exported annotations (default: ./annotations_output)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )
    serve_parser.add_argument(
        "--cell-radius",
        type=int,
        default=4,
        help="Radius in pixels for manually added circle annotations (default: 4)",
    )
    serve_parser.add_argument(
        "--semiannotation-dir",
        type=str,
        default=None,
        help="Path to semi-annotation directory (PNGs + _annotations.coco.json)",
    )
    serve_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG-level) logging output",
    )
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
