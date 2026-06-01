"""
src/zones/__init__.py
=====================
Zone detection package initialiser.

Provides ZoneConfig — a single object that loads zone_config.yaml
and exposes every parameter as a clean attribute. All zone modules
import this instead of reading the YAML themselves.

Usage from any zone module:
    from src.zones import ZoneConfig
    cfg = ZoneConfig()
    lookback = cfg.swing.lookback
    merge_dist = cfg.merger.merge_distance_atr

Usage from a notebook or script:
    from src.zones import ZoneConfig, detect_swing_zones, merge_zones, score_zones
"""

from pathlib import Path
import yaml


# ── Path resolution ────────────────────────────────────────────
_PROJECT_ROOT   = Path(__file__).resolve().parents[2]
_ZONE_CFG_PATH  = _PROJECT_ROOT / "config" / "zone_config.yaml"


class _Section:
    """Turns a dict into dot-accessible attributes (cfg.swing.lookback)."""
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _Section(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        items = {k: v for k, v in self.__dict__.items()}
        return f"Section({items})"


class ZoneConfig:
    """
    Loads config/zone_config.yaml and exposes each section as a
    dot-accessible object.

    Sections available:
        .swing          — swing high/low detection params
        .merger         — zone merging params
        .scorer         — zone scoring params
        .nearby_filter  — nearby zone filter params
        .cluster        — cluster zone params (future)
        .volume_profile — volume profile params (future)
        .supply_demand  — supply/demand zone params (future)

    Example
    -------
    >>> cfg = ZoneConfig()
    >>> cfg.swing.lookback
    5
    >>> cfg.merger.merge_distance_atr
    0.5
    >>> cfg.scorer.weights.move
    0.35
    """

    def __init__(self, path: Path = _ZONE_CFG_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"zone_config.yaml not found at {path}. "
                "Make sure the file exists in config/."
            )
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        for section_name, section_data in raw.items():
            if isinstance(section_data, dict):
                setattr(self, section_name, _Section(section_data))
            else:
                setattr(self, section_name, section_data)

    def __repr__(self):
        sections = [k for k in self.__dict__.keys()]
        return f"ZoneConfig(sections={sections})"


# ── Re-export for clean imports ────────────────────────────────
from src.zones.swing_zones  import detect_swing_zones, get_nearby_zones, save_zones, load_zones, zone_summary  # noqa: E402
from src.zones.zone_merger  import merge_zones   # noqa: E402
from src.zones.zone_scorer  import score_zones   # noqa: E402

__all__ = [
    "ZoneConfig",
    "detect_swing_zones",
    "merge_zones",
    "score_zones",
    "get_nearby_zones",
    "save_zones",
    "load_zones",
    "zone_summary",
]
