"""Preprocessing package for data preparation and metadata building."""
from __future__ import annotations

from .constants import (
    SEED,
    SRC_SUBDIR,
    CORPUS_AUDIO_EXT,
    CORPUS_LANGUAGE,
    CORPUS_KIND,
    TARGET_SAMPLE_RATE,
)
from .normalization import normalize

__all__ = [
    "normalize",
    "SEED",
    "SRC_SUBDIR",
    "CORPUS_AUDIO_EXT",
    "CORPUS_LANGUAGE",
    "CORPUS_KIND",
    "TARGET_SAMPLE_RATE",
]
