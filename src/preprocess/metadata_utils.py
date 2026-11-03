"""
Shared helpers for building JSONL metadata files.
"""
from __future__ import annotations

import json
import soundfile as sf
from pathlib import Path
from typing import Iterator

from preprocess.constants import LANGUAGE
from preprocess.normalization import normalize


def audio_duration_seconds(wav: Path) -> float:
    """Return duration in seconds by reading the WAV header only (no decode)."""
    info = sf.info(str(wav))
    return info.frames / float(info.samplerate)


def read_transcript(txt_path: Path) -> str | None:
    """Read and normalize a transcript file. Returns None if missing/empty."""
    if not txt_path.exists():
        return None
    raw = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    norm = normalize(raw)
    return norm or None


def match_wavs_with_text(wav_root: Path, src_root: Path) -> Iterator[tuple[Path, str]]:
    """
    Yield (wav_path, text) for every WAV that matches .txt
    wav_root : converted 16k WAVs (data/wav16k/<corpus>)
    src_root : extracted source tree holding the original .txt files
    """
    missing_txt = 0
    empty_txt = 0
    total = 0
    for wav in wav_root.rglob("*.wav"):
        total += 1
        rel = wav.relative_to(wav_root).with_suffix(".txt")
        txt = src_root / rel
        text = read_transcript(txt)
        if text is None:
            if txt.exists():
                empty_txt += 1
            else:
                missing_txt += 1
            continue
        yield wav, text
    if missing_txt or empty_txt:
        print(f"[warn] {missing_txt} WAVs had no corresponding .txt, "
              f"{empty_txt} had empty/normalized-empty text (skipped), "
              f"of {total} total WAVs")


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write a list of dict rows to a JSONL file (one JSON object per line)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {len(rows):>7} rows -> {path}")


def build_entry(wav: Path, text: str, corpus: str, split: str, base_dir: Path) -> dict:
    """Build one metadata row"""
    try:
        rel = wav.resolve().relative_to(base_dir).as_posix()
    except ValueError:
        rel = wav.as_posix()
    return {
        "audio_path": rel,
        "text": text,
        "language": LANGUAGE,
        "corpus": corpus,
        "split": split,
        "duration": round(audio_duration_seconds(wav), 3),
    }
