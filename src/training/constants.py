"""
Training constants used
"""
from __future__ import annotations

from preprocess.constants import TARGET_SAMPLE_RATE

MODEL_ID = "openai/whisper-large-v3-turbo"
LANGUAGE = "kazakh"
TASK = "transcribe"

LANGUAGE_NAMES = {
    "kk": "kazakh",
    "ru": "russian",
    "en": "english",
}

TARGET_SR = TARGET_SAMPLE_RATE
MAX_AUDIO_SECONDS = 30.0
# whisper hard limit
MAX_LABEL_TOKENS = 448
# Skipped pad values
LABEL_IGNORE_INDEX = -100
