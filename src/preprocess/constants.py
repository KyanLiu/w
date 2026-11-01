"""
Preprocessing constants used
"""
from __future__ import annotations

LANGUAGE = "kk"
SEED = 42
TARGET_SAMPLE_RATE = 16000

SRC_SUBDIR = {
    "ksc2": "ksc2/ISSAI_KSC2",
    "kazemotts": "kazemotts/EmoKaz",
}

CORPUS_AUDIO_EXT = {
    "ksc2": ".flac",
    "kazemotts": ".wav",
}
