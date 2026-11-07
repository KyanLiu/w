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
}

CORPUS_AUDIO_EXT = {
    "ksc2": ".flac",
    "kazemotts": ".wav",
    "cve": ".mp3",
    "cvr": ".mp3",
    "golos": ".opus",
}

CORPUS_LANGUAGE = {
    "ksc2": "kk",
    "kazemotts": "kk",
    "cve": "en",
    "cvr": "ru",
    "golos": "ru",
}

# Corpus pairs audio with its transcript.
# txt_tree : audio tree + per-file .txt sidecar (ksc2, kazemotts)
# cv_tsv   : Common Voice clips/*.mp3 + validated.tsv (cve, cvr)
# manifest : audio tree + a JSONL manifest mapping id -> text (golos)
CORPUS_KIND = {
    "ksc2": "txt_tree",
    "kazemotts": "txt_tree",
    "cve": "cv_tsv",
    "cvr": "cv_tsv",
    "golos": "manifest",
}
