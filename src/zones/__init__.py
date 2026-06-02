"""
src/zones/__init__.py
=====================
Zone detection package.

Exports the main detect() function and I/O helpers from zone_detector.
"""

from src.zones.zone_detector import detect, save_zones, load_zones

__all__ = ["detect", "save_zones", "load_zones"]
