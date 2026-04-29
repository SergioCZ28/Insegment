"""Unit tests for insegment.profiles.

Profiles are JSON bundles of `serve` defaults recalled by name with
`insegment serve --profile <name>`. The contract these tests pin down:

  * a missing profiles file is fine (returns {})
  * malformed JSON is fine (returns {}, logs a warning)
  * apply_profile only fills attrs that are currently None on `args`
  * unknown keys in a profile are ignored, not propagated to args
  * INSEGMENT_PROFILES_PATH overrides the default path (used by tests so
    we don't write to the user's real ~/.insegment/profiles.json).
"""

from __future__ import annotations

import argparse
import json

import pytest

from insegment import profiles as profiles_mod


@pytest.fixture
def profiles_file(tmp_path, monkeypatch):
    """Point the profiles module at a fresh JSON in tmp_path."""
    path = tmp_path / "profiles.json"
    monkeypatch.setenv("INSEGMENT_PROFILES_PATH", str(path))
    return path


def _ns(**kw):
    """Build an argparse Namespace with the same attrs `serve` uses."""
    base = dict(
        model=None, checkpoint=None, image_dir=None, output_dir=None,
        semiannotation_dir=None, port=None, cell_radius=None, min_area=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


class TestLoadProfiles:
    def test_missing_file_returns_empty(self, profiles_file):
        assert not profiles_file.exists()
        assert profiles_mod.load_profiles() == {}

    def test_malformed_json_returns_empty(self, profiles_file):
        profiles_file.write_text("{not valid json")
        assert profiles_mod.load_profiles() == {}

    def test_top_level_not_object_returns_empty(self, profiles_file):
        profiles_file.write_text(json.dumps(["just", "a", "list"]))
        assert profiles_mod.load_profiles() == {}

    def test_well_formed_file_returns_dict(self, profiles_file):
        profiles_file.write_text(json.dumps({
            "default": {"model": "foo:Bar", "port": 5001},
        }))
        loaded = profiles_mod.load_profiles()
        assert "default" in loaded
        assert loaded["default"]["model"] == "foo:Bar"


class TestApplyProfile:
    def test_fills_unset_attrs(self, profiles_file):
        profiles_file.write_text(json.dumps({
            "p1": {
                "model": "mymod:Adapter",
                "image_dir": "/tmp/imgs",
                "output_dir": "/tmp/saves",
                "port": 5050,
            },
        }))
        args = _ns()
        profiles_mod.apply_profile(args, "p1")
        assert args.model == "mymod:Adapter"
        assert args.image_dir == "/tmp/imgs"
        assert args.output_dir == "/tmp/saves"
        assert args.port == 5050

    def test_cli_value_wins_over_profile(self, profiles_file):
        """If the user passed --port 9999 on the CLI, the profile must NOT
        clobber it. Profile only fills holes (None values)."""
        profiles_file.write_text(json.dumps({
            "p1": {"port": 5050, "model": "mymod:Adapter"},
        }))
        args = _ns(port=9999)
        profiles_mod.apply_profile(args, "p1")
        assert args.port == 9999            # CLI wins
        assert args.model == "mymod:Adapter"  # gap filled

    def test_unknown_keys_ignored(self, profiles_file):
        """Forward-compat: a profile written by a newer Insegment with new
        keys must not blow up an older client. Unknown keys are ignored."""
        profiles_file.write_text(json.dumps({
            "p1": {"model": "x:Y", "future_flag": "abc", "another": 42},
        }))
        args = _ns()
        profiles_mod.apply_profile(args, "p1")
        assert args.model == "x:Y"
        assert not hasattr(args, "future_flag")
        assert not hasattr(args, "another")

    def test_missing_profile_logs_but_does_not_crash(self, profiles_file, caplog):
        profiles_file.write_text(json.dumps({"existing": {}}))
        args = _ns(model="cli-value")
        with caplog.at_level("WARNING"):
            profiles_mod.apply_profile(args, "does-not-exist")
        assert args.model == "cli-value"  # untouched
        assert any("not found" in rec.message for rec in caplog.records)

    def test_no_profiles_file_logs_but_does_not_crash(self, profiles_file, caplog):
        # File doesn't exist at all
        assert not profiles_file.exists()
        args = _ns()
        with caplog.at_level("WARNING"):
            profiles_mod.apply_profile(args, "anything")
        assert args.model is None

    def test_non_dict_profile_value_ignored(self, profiles_file, caplog):
        profiles_file.write_text(json.dumps({"p1": "this should be a dict"}))
        args = _ns()
        with caplog.at_level("WARNING"):
            profiles_mod.apply_profile(args, "p1")
        assert args.model is None
