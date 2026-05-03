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
    from insegment.profiles import apply_profile

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )

    # Resolve profile values into any unset args BEFORE we apply hard
    # defaults below. CLI flags always win over profile values.
    # If --profile wasn't passed, look for a profile literally named
    # "default" and apply it silently if it exists.
    if args.profile:
        apply_profile(args, args.profile)
    else:
        from insegment.profiles import load_profiles
        if "default" in load_profiles():
            apply_profile(args, "default")

    # Apply hard fallbacks for args that argparse left as None so that
    # profiles get a chance to populate them first.
    if args.output_dir is None:
        args.output_dir = "./annotations_output"
    if args.port is None:
        args.port = 5000
    if args.cell_radius is None:
        args.cell_radius = 4

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
        # Pass checkpoint and any extra kwargs to the model constructor.
        # Optional kwargs are only added when the user explicitly passes them,
        # so they don't clash with adapters that don't accept them.
        kwargs = {}
        if args.checkpoint:
            kwargs["checkpoint_path"] = args.checkpoint
        if args.min_area is not None:
            kwargs["min_area"] = args.min_area
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
        default=None,
        help="Output directory for exported annotations (default: ./annotations_output)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on (default: 5000)",
    )
    serve_parser.add_argument(
        "--cell-radius",
        type=int,
        default=None,
        help="Radius in pixels for manually added circle annotations (default: 4)",
    )
    serve_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=(
            "Named profile from ~/.insegment/profiles.json to fill in "
            "defaults (e.g. model, checkpoint, image-dir). CLI flags "
            "override profile values. If a profile literally named "
            "'default' exists, it is applied automatically when --profile "
            "is omitted."
        ),
    )
    serve_parser.add_argument(
        "--min-area",
        type=int,
        default=None,
        help=(
            "Minimum mask area in pixels for model-predicted instances. "
            "Passed to the adapter as min_area kwarg. BacDETR adapter "
            "default is 20. Set to 0 to disable. Only takes effect if the "
            "adapter accepts a min_area parameter."
        ),
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

    # --- profile subcommand: list/show profiles ----------------------------
    profile_parser = subparsers.add_parser(
        "profile",
        help="Inspect Insegment CLI profiles",
        description=(
            "List or print profiles defined in ~/.insegment/profiles.json. "
            "Profiles are bundles of `serve` defaults (model, image-dir, "
            "etc.) recalled via `insegment serve --profile <name>`. The "
            "file is plain JSON -- create or edit it with any text editor."
        ),
    )
    profile_subs = profile_parser.add_subparsers(
        dest="profile_action", help="profile actions",
    )
    profile_subs.add_parser("list", help="List all profile names")
    show_p = profile_subs.add_parser("show", help="Print one profile's contents")
    show_p.add_argument("name", help="Profile name to show")
    profile_parser.set_defaults(func=cmd_profile)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


def cmd_profile(args):
    """Implementation of the `insegment profile ...` subcommand."""
    from insegment.profiles import load_profiles, profiles_path

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    profiles = load_profiles()
    path = profiles_path()

    action = getattr(args, "profile_action", None)
    if action is None or action == "list":
        if not profiles:
            print(f"No profiles found. Edit {path} to create one.")
            return
        print(f"Profiles in {path}:")
        for name in sorted(profiles.keys()):
            data = profiles[name]
            keys = ", ".join(sorted(data.keys())) if isinstance(data, dict) else "(invalid)"
            print(f"  {name}  [{keys}]")
        return

    if action == "show":
        if args.name not in profiles:
            available = ", ".join(sorted(profiles.keys())) or "(none)"
            print(f"Profile '{args.name}' not found. Available: {available}")
            sys.exit(1)
        import json as _json
        print(_json.dumps(profiles[args.name], indent=2, sort_keys=True))
        return


if __name__ == "__main__":
    main()
