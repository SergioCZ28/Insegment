"""CLI profiles -- named bundles of `insegment serve` defaults.

Why this exists
---------------
The most-common `insegment serve` invocation has many flags (model,
checkpoint, image-dir, output-dir, port). Re-typing them every session is
error-prone. Profiles let you save a bundle once and recall it as
`insegment serve --profile <name>`.

Storage
-------
Profiles live in a single JSON file at `~/.insegment/profiles.json`. The
file is plain JSON, hand-editable, no third-party TOML parser needed
(insegment supports Python 3.9, where `tomllib` is not in the stdlib).

Schema:

    {
      "default": {
        "model": "insegment.models.bacdetr:BacDETRSegmenter",
        "checkpoint": "C:/path/to/checkpoint.pth",
        "image_dir": "C:/path/to/images",
        "output_dir": "C:/path/to/saves",
        "port": 5000
      },
      "another-profile": { ... }
    }

Each value maps to a `serve` flag (with hyphens converted to underscores
to keep JSON keys clean). Keys not understood by `serve` are ignored.

Resolution
----------
`apply_profile(args, profile_name)` mutates `args` in-place: for each
attribute that is currently None / unset, fill in the profile value if
present. CLI arguments always win -- the profile only fills gaps. That
way you can override a single value without rewriting the profile, e.g.
`insegment serve --profile default --port 5001`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def profiles_path() -> Path:
    """Where the profiles JSON lives. Honors $INSEGMENT_PROFILES_PATH for tests."""
    override = os.environ.get("INSEGMENT_PROFILES_PATH")
    if override:
        return Path(override)
    return Path.home() / ".insegment" / "profiles.json"


def load_profiles() -> dict[str, dict[str, Any]]:
    """Read the profiles file. Returns {} if it doesn't exist or is invalid.

    Invalid JSON is logged but not fatal -- a broken profiles file should
    never block a user from running `insegment serve`.
    """
    path = profiles_path()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read profiles file %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "Profiles file %s: expected a JSON object at the top level, got %s",
            path, type(data).__name__,
        )
        return {}
    return data


# Keys that the serve subcommand accepts as profile values. Any other keys
# in a profile dict are silently ignored (forward-compat: a profile written
# by a newer Insegment can still be read by an older one).
_PROFILE_KEYS = {
    "model",
    "checkpoint",
    "image_dir",
    "output_dir",
    "semiannotation_dir",
    "port",
    "cell_radius",
    "min_area",
}


def apply_profile(args, profile_name: str) -> None:
    """Fill any unset attributes on `args` from the named profile.

    CLI arguments always win: a profile value is only applied when the
    corresponding attribute on `args` is None (i.e. the user did not pass
    that flag). This means you can override one value at a time:

        insegment serve --profile default --port 5001

    Args:
        args: The argparse Namespace from `serve`.
        profile_name: Name of the profile to load. If it doesn't exist,
            log a warning and continue without applying anything (the run
            proceeds with whatever CLI args were given).
    """
    profiles = load_profiles()
    if profile_name not in profiles:
        if profiles:
            available = ", ".join(sorted(profiles.keys())) or "(none)"
            logger.warning(
                "Profile '%s' not found in %s. Available: %s",
                profile_name, profiles_path(), available,
            )
        else:
            logger.warning(
                "Profile '%s' requested but no profiles file at %s",
                profile_name, profiles_path(),
            )
        return

    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        logger.warning(
            "Profile '%s' is not a JSON object, ignoring", profile_name
        )
        return

    applied = []
    for key, value in profile.items():
        if key not in _PROFILE_KEYS:
            logger.debug("Profile '%s': unknown key '%s' ignored", profile_name, key)
            continue
        if not hasattr(args, key):
            continue
        if getattr(args, key) is None:
            setattr(args, key, value)
            applied.append(f"{key}={value!r}")
    if applied:
        logger.info("Applied profile '%s': %s", profile_name, ", ".join(applied))
