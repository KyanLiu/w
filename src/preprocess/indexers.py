"""Unified source indexing for different audio preprocessing pipeline"""
from __future__ import annotations

import csv
import sys
import random
import soundfile as sf
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from preprocess.constants import (
    SEED,
    SRC_SUBDIR,
    CORPUS_AUDIO_EXT,
    CORPUS_KIND,
    PARQUET_COLUMNS,
    TARGET_SAMPLE_RATE,
)
from preprocess.normalization import normalize


@dataclass
class Entry:
    src_audio: str          # abs path to source audio file ("" if byte-sourced)
    rel_key: str            # relative output path, .wav suffix
    text: str               # normalized transcript ("" if unknown here)
    duration: float | None  # seconds, or None if not yet known
    split: str = "train"    # native split: train/dev/test (default train)
    audio_bytes: bytes | None = None  # raw WAV bytes for byte-sourced corpora
    notations: str | None = None      # optional per-clip notation term list (KazMedSpeech -> medical term list)
    domain: str | None = None         # optional domain/specialty label (KazMedSpeech -> per-domain WER)
    recording: str | None = None      # optional recording/episode id for the bootstrap grouping


# Duration probing (header-only, no decode)
def _probe_one(path: str) -> tuple[str, float | None]:
    try:
        info = sf.info(path)
        return path, info.frames / float(info.samplerate)
    except Exception:
        return path, None


def probe_durations(paths: list[str], workers: int) -> dict[str, float]:
    """Read duration from each audio header in parallel. Used if audio durations are not stored elsewhere"""
    out: dict[str, float] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path, dur in ex.map(_probe_one, paths, chunksize=256):
            if dur is not None:
                out[path] = dur
    return out


# Indexers: one per CORPUS_KIND

def _index_txt_tree(src_root: Path, ext: str) -> Iterator[Entry]:
    """
    ksc2 / kazemotts: audio files with a sibling .txt transcript of the same stem. rel_key mirrors the source tree layout.
    """
    missing = 0
    total = 0
    for audio in src_root.rglob(f"*{ext}"):
        total += 1
        txt = audio.with_suffix(".txt")
        if not txt.exists():
            missing += 1
            continue
        raw = txt.read_text(encoding="utf-8", errors="replace").strip()
        text = normalize(raw)
        if not text:
            continue
        rel_key = audio.relative_to(src_root).with_suffix(".wav").as_posix()
        yield Entry(str(audio), rel_key, text, None)
    if missing:
        print(f"[index] txt_tree: {missing}/{total} audio had no .txt (skipped)")




def _pick(fieldnames: list[str], cands: tuple[str, ...], where: str) -> str:
    for c in cands:
        if c in fieldnames:
            return c
    sys.exit(f"[error] {where}: none of {cands} in columns {fieldnames}")


def _load_cv_durations(lang_dir: Path) -> dict[str, float]:
    """clip_durations.tsv (duration[ms]) -> {clip_filename: seconds}, or {}."""
    cand = lang_dir / "clip_durations.tsv"
    if not cand.exists():
        return {}
    dmap: dict[str, float] = {}
    with open(cand, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames or []
        clip_col = next((c for c in ("clip", "path") if c in fields), None)
        dur_col = next((c for c in ("duration[ms]", "duration_ms", "duration") if c in fields), None)
        if not clip_col or not dur_col:
            return {}
        for row in reader:
            try:
                dmap[row[clip_col].strip()] = float(row[dur_col]) / 1000.0
            except (KeyError, TypeError, ValueError):
                continue
    print(f"[index] cv_tsv: loaded {len(dmap)} durations from clip_durations.tsv")
    return dmap

_PATH_COLS = ("path", "clip", "filename")
_TEXT_COLS = ("sentence", "text")

def _index_cv_tsv(src_root: Path, tsv_name: str, workers: int) -> Iterator[Entry]:
    """
    Common Voice: clips/*.mp3 with transcripts in <tsv_name> (default
    validated.tsv). Durations from clip_durations.tsv if present, else probed
    from mp3 headers. rel_key = clips/<stem>.wav.
    """
    lang_dir = src_root
    clips_dir = lang_dir / "clips"
    tsv_path = lang_dir / tsv_name
    if not tsv_path.exists():
        sys.exit(f"[error] tsv not found: {tsv_path}")

    rows: list[tuple[str, str]] = []  # (clip_filename, text)
    skipped = 0
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            sys.exit(f"[error] {tsv_path}: empty/unreadable")
        path_col = _pick(reader.fieldnames, _PATH_COLS, str(tsv_path))
        text_col = _pick(reader.fieldnames, _TEXT_COLS, str(tsv_path))
        for row in reader:
            clip = (row.get(path_col) or "").strip()
            text = normalize(row.get(text_col) or "")
            if not clip or not text:
                skipped += 1
                continue
            rows.append((clip, text))
    print(f"[index] cv_tsv {tsv_name}: {len(rows)} usable rows "
          f"(path={path_col!r} text={text_col!r}, {skipped} skipped)")

    dmap = _load_cv_durations(lang_dir)
    if not dmap:
        print(f"[index] cv_tsv: no durations file; probing {len(rows)} headers")
        dmap = probe_durations([str(clips_dir / c) for c, _ in rows], workers)
        dmap = {Path(k).name: v for k, v in dmap.items()}

    for clip, text in rows:
        src = clips_dir / clip
        rel_key = f"clips/{Path(clip).stem}.wav"
        yield Entry(str(src), rel_key, text, dmap.get(clip))


def _index_manifest(src_root: Path, ext: str, workers: int) -> Iterator[Entry]:
    """
    Used for Golos: JSONL manifest mapping an audio path/id -> transcription.
    Searches src_root for a manifest, pairs each entry with its audio file.
    rel_key mirrors the audio's path under src_root.
    """
    import json

    # Find a manifest file (jsonl/json) under the source root.
    manifests = [p for p in src_root.rglob("*.json*")
                 if p.suffix in (".json", ".jsonl")]
    if not manifests:
        sys.exit(f"[error] manifest: no .json/.jsonl under {src_root}")
    manifest = manifests[0]
    print(f"[index] manifest: using {manifest}")

    audio_keys = ("audio_filepath", "audio", "path", "wav", "file")
    text_keys = ("text", "transcription", "transcript", "sentence")

    n = 0
    missing = 0
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            apath = next((obj[k] for k in audio_keys if k in obj), None)
            atext = next((obj[k] for k in text_keys if k in obj), None)
            if apath is None or atext is None:
                continue
            # audio path in manifest may be relative to src_root or absolute
            src = Path(apath)
            if not src.is_absolute():
                src = src_root / apath
            if not src.exists():
                missing += 1
                continue
            text = normalize(atext)
            if not text:
                continue
            try:
                rel = src.relative_to(src_root)
            except ValueError:
                rel = Path(src.name)
            rel_key = rel.with_suffix(".wav").as_posix()
            n += 1
            yield Entry(str(src), rel_key, text,
                        float(obj["duration"]) if "duration" in obj else None)
    if missing:
        print(f"[index] manifest: {missing} entries had missing audio (skipped)")
    print(f"[index] manifest: {n} usable entries")


def _index_parquet(src_root: Path, corpus: str) -> Iterator[Entry]:
    """
    Parquet corpora with embedded audio bytes (KazMedSpeech, FLEURS Kazakh)
    Split into train/dev/test
    where each rows hold:
      - audio: struct {bytes, path} -- bytes are already 16 kHz mono PCM-16 WAV
      - transcription: the transcript (Kazakh)
      - duration: seconds (float)
    Other additional columns are also parsed as part of the corpus
    """
    spec = PARQUET_COLUMNS[corpus]
    text_col = spec["text"]
    audio_col = spec["audio"]
    dur_col = spec["duration"]
    nsamp_col = spec["num_samples"]
    notations_col = spec["notations"]
    domain_col = spec.get("domain")
    recording_col = spec.get("recording")

    shards = sorted(src_root.glob("*.parquet"))
    if not shards:
        sys.exit(f"[error] parquet: no *.parquet under {src_root}")

    def _split_of(name: str) -> str:
        low = name.lower()
        if low.startswith("train"):
            return "train"
        if low.startswith("dev") or low.startswith("valid"):
            return "dev"
        if low.startswith("test"):
            return "test"
        return "train"  # fallback

    n = 0
    skipped = 0
    per_split: dict[str, int] = {}
    for shard in shards:
        split = _split_of(shard.name)
        stem = shard.stem
        pf = pq.ParquetFile(str(shard))
        names = set(pf.schema_arrow.names)
        row_idx = 0
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg)
            audio_list = table.column(audio_col).to_pylist()
            text_list = table.column(text_col).to_pylist()
            dur_list = table.column(dur_col).to_pylist() if dur_col and dur_col in names else None
            nsamp_list = table.column(nsamp_col).to_pylist() if nsamp_col and nsamp_col in names else None
            notations_list = (table.column(notations_col).to_pylist()
                              if notations_col and notations_col in names else None)
            domain_list = (table.column(domain_col).to_pylist()
                           if domain_col and domain_col in names else None)
            recording_list = (table.column(recording_col).to_pylist()
                              if recording_col and recording_col in names else None)

            for j in range(len(text_list)):
                i = row_idx
                row_idx += 1
                audio = audio_list[j]
                text = normalize(text_list[j] or "")
                audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                if not text or audio_bytes is None:
                    skipped += 1
                    continue
                # duration: prefer a direct column, else derive from num_samples
                if dur_list is not None and dur_list[j] is not None:
                    duration = float(dur_list[j])
                elif nsamp_list is not None and nsamp_list[j] is not None:
                    duration = float(nsamp_list[j]) / float(TARGET_SAMPLE_RATE)
                else:
                    duration = None
                notations = notations_list[j] if notations_list is not None else None
                domain = domain_list[j] if domain_list is not None else None
                recording = recording_list[j] if recording_list is not None else None
                rel_key = f"{split}/{stem}_{i:06d}.wav"
                n += 1
                per_split[split] = per_split.get(split, 0) + 1
                yield Entry(
                    src_audio="",
                    rel_key=rel_key,
                    text=text,
                    duration=duration,
                    split=split,
                    audio_bytes=audio_bytes,
                    notations=notations,
                    domain=domain,
                    recording=recording,
                )
    print(f"[index] parquet ({corpus}): {n} usable rows across {len(shards)} shards "
          f"(by split: {per_split}, {skipped} skipped)")


def index_corpus(corpus: str, src_root: Path, workers: int, cv_tsv_name: str = "validated.tsv") -> list[Entry]:
    """Dispatch to the right indexer by CORPUS_KIND; returns a materialized list."""
    kind = CORPUS_KIND[corpus]
    ext = CORPUS_AUDIO_EXT[corpus]
    if kind == "txt_tree":
        return list(_index_txt_tree(src_root, ext))
    if kind == "cv_tsv":
        return list(_index_cv_tsv(src_root, cv_tsv_name, workers))
    if kind == "manifest":
        return list(_index_manifest(src_root, ext, workers))
    if kind == "parquet":
        return list(_index_parquet(src_root, corpus))
    sys.exit(f"[error] unknown corpus kind {kind!r} for {corpus}")


# Generic hour-capping based on Entry Durations-- works on any list[Entry]
def cap_hours(entries: list[Entry], target_hours: float | None,
              workers: int) -> list[Entry]:
    """
    Seeded, duration-aware subsample to ~target_hours. If target_hours is None,
    return all entries.
    """
    if not entries:
        return entries

    # Fill any missing durations (header probe) so capping/metadata have them.
    need = [e.src_audio for e in entries if e.duration is None]
    if need:
        print(f"[cap] probing {len(need)} missing durations")
        dmap = probe_durations(need, workers)
        for e in entries:
            if e.duration is None:
                e.duration = dmap.get(e.src_audio)

    usable = [e for e in entries if e.duration is not None]
    dropped = len(entries) - len(usable)
    if dropped:
        print(f"[cap] dropped {dropped} entries with no readable duration")

    if target_hours is None:
        total_h = sum(e.duration for e in usable) / 3600.0
        print(f"[cap] no cap: {len(usable)} entries = {total_h:.1f} h")
        return sorted(usable, key=lambda e: e.rel_key)

    target_s = target_hours * 3600.0
    rng = random.Random(SEED)
    order = list(range(len(usable)))
    rng.shuffle(order)
    chosen: list[Entry] = []
    total = 0.0
    for i in order:
        e = usable[i]
        chosen.append(e)
        total += e.duration
        if total >= target_s:
            break
    chosen.sort(key=lambda e: e.rel_key)
    print(f"[cap] selected {len(chosen)} entries = {total/3600.0:.1f} h "
          f"(target {target_hours} h)")
    if total < target_s:
        print(f"[warn] only {total/3600.0:.1f} h available, below target "
              f"{target_hours} h -- using all")
    return chosen
