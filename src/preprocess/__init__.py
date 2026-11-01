"""Preprocessing package for data preparation and metadata building."""
from __future__ import annotations

from .constants import LANGUAGE, SEED, SRC_SUBDIR, TARGET_SAMPLE_RATE
from .normalization import normalize

__all__ = [
    "normalize",
    "LANGUAGE",
    "SEED",
    "SRC_SUBDIR",
    "TARGET_SAMPLE_RATE",
]
