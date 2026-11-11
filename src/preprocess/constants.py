"""
Preprocessing constants used
"""
from __future__ import annotations

SEED = 42
TARGET_SAMPLE_RATE = 16000

SRC_SUBDIR = {
    "ksc2": "ksc2/ISSAI_KSC2",
    "kazemotts": "kazemotts/EmoKaz",
    "cve": "cve/cv-corpus-27.0-2026-09-11/en",
    "cvr": "cvr/cv-corpus-27.0-2026-09-11/ru",
    "golos": "golos",
    "kazmedspeech": "kazmedspeech/data",
}

CORPUS_AUDIO_EXT = {
    "ksc2": ".flac",
    "kazemotts": ".wav",
    "cve": ".mp3",
    "cvr": ".mp3",
    "golos": ".opus",
    "kazmedspeech": ".wav",
}

CORPUS_LANGUAGE = {
    "ksc2": "kk",
    "kazemotts": "kk",
    "cve": "en",
    "cvr": "ru",
    "golos": "ru",
    "kazmedspeech": "kk",
}

# Corpus pairs audio with its transcript.
# txt_tree : audio tree + per-file .txt sidecar (ksc2, kazemotts)
# cv_tsv   : Common Voice clips/*.mp3 + validated.tsv (cve, cvr)
# manifest : audio tree + a JSONL manifest mapping id -> text (golos)
# parquet  : parquet shards with embedded audio bytes + native splits and a per-row duration column (kazmedspeech)
CORPUS_KIND = {
    "ksc2": "txt_tree",
    "kazemotts": "txt_tree",
    "cve": "cv_tsv",
    "cvr": "cv_tsv",
    "golos": "manifest",
    "kazmedspeech": "parquet",
}

# Column names for KazMedSpeech parquet rows
KAZMEDSPEECH_TEXT_COL = "transcription"
KAZMEDSPEECH_DURATION_COL = "duration"
KAZMEDSPEECH_AUDIO_COL = "audio"
