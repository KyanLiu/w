"""
Training constants used
"""
from __future__ import annotations

from preprocess.constants import TARGET_SAMPLE_RATE

MODEL_ID = "openai/whisper-large-v3-turbo"
LANGUAGE = "kazakh"
TASK = "transcribe"
TARGET_SR = TARGET_SAMPLE_RATE
MAX_AUDIO_SECONDS = 30.0
