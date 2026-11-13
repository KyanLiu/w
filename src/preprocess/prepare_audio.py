"""
Convert source audio to 16 kHz mono PCM-16 WAV, with optional hour-capping.
Uses:
1. index_corpus to return Entry
2. cap_hours to sample data
"""
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from preprocess.constants import SRC_SUBDIR, TARGET_SAMPLE_RATE, OPTIONAL_CLIP_FIELDS
from preprocess.indexers import index_corpus, cap_hours


def _convert_audio_file(src: str, dst: str, audio_bytes: bytes | None = None) -> tuple[str, str | None]:
    """
    Convert one source audio file to a 16 kHz mono PCM-16 WAV at dst.
    If audio_bytes exist, return it as is
    Returns (src, None) on success (or skip), (src, error_message) on failure.
    Produce a 16 kHz mono PCM-16 WAV at dst.
    """
    key = src or dst
    try:
        import soundfile as sf
        import numpy as np

        if Path(dst).exists():
            return (key, None)

        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        tmp = dst + ".tmp"

        if audio_bytes is not None:
            # Already 16 kHz mono PCM-16 WAV bytes -> write as-is, atomically.
            with open(tmp, "wb") as fh:
                fh.write(audio_bytes)
            Path(tmp).replace(dst)
            return (key, None)

        audio, sr = sf.read(src, always_2d=True)
        audio = audio.mean(axis=1)  # downmix to mono
        if sr != TARGET_SAMPLE_RATE:
            import soxr
            audio = soxr.resample(audio, sr, TARGET_SAMPLE_RATE)
        sf.write(tmp, np.clip(audio, -1.0, 1.0).astype(np.float32),
                 TARGET_SAMPLE_RATE, subtype="PCM_16", format="WAV")
        Path(tmp).replace(dst)
        return (key, None)
    except Exception as e:
        return (key, f"{type(e).__name__}: {e}")


def _convert_job(job: tuple) -> tuple[str, str | None]:
    """Unpack a (src, dst[, audio_bytes]) job so the worker can be driven by map."""
    if len(job) == 3:
        src, dst, audio_bytes = job
    else:
        src, dst = job
        audio_bytes = None
    return _convert_audio_file(src, dst, audio_bytes)


def _write_selection(entries, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            row = {"rel_key": e.rel_key, "text": e.text,
                   "duration": e.duration, "split": e.split}
            # optional fields
            for key in OPTIONAL_CLIP_FIELDS:
                val = getattr(e, key, None)
                if val is not None:
                    row[key] = val
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[write] {len(entries)} selection rows -> {path}")


def prepare_corpus(
    corpus: str, extracted_root: Path, out_root: Path,
    metadata_dir: Path, target_hours: float | None,
    workers: int, missing_ok: bool = False
) -> None:
    """Index -> cap -> convert selected -> write selection, for one corpus."""
    src_root = extracted_root / SRC_SUBDIR[corpus]
    dst_root = out_root / corpus
    marker = dst_root / ".convert.done"
    selection_path = metadata_dir / f"{corpus}_selection.jsonl"

    if marker.exists():
        print(f"[skip] {corpus}: already converted in {dst_root}")
        return
    if not src_root.exists():
        if missing_ok:
            print(f"[skip] {corpus}: source not found: {src_root}")
            return
        sys.exit(f"[error] source not found: {src_root}")

    print(f"[index] {corpus}: scanning {src_root}")
    entries = index_corpus(corpus, src_root, workers)
    if not entries:
        sys.exit(f"[error] {corpus}: indexer found no usable entries")

    entries = cap_hours(entries, target_hours, workers)
    if not entries:
        sys.exit(f"[error] {corpus}: no entries after capping")

    # Persist the exact selected set (text + duration) for build_metadata.
    _write_selection(entries, selection_path)

    jobs = [
        (e.src_audio, str(dst_root / e.rel_key), e.audio_bytes)
        for e in entries
    ]
    print(f"[convert] {corpus}: converting {len(jobs)} clips, {workers} workers")

    done = 0
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for src, err in executor.map(_convert_job, jobs, chunksize=64):
            done += 1
            if err:
                errors.append(f"{src}: {err}")
            if done % 2000 == 0:
                print(f"{done}/{len(jobs)} done ({len(errors)} errors)")

    print(f"{done}/{len(jobs)} done ({len(errors)} errors)")
    if errors:
        print(f"\n[warn] {len(errors)} files failed - not writing .done marker.")
        for line in errors[:20]:
            print(line)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more")
        sys.exit(1)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")
    print(f"[ok] {corpus}: {done} clips converted -> {dst_root}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus",
        choices=["ksc2", "kazemotts", "cve", "cvr", "golos", "kazmedspeech", "fleurs_kk", "all"],
        default="all")
    parser.add_argument("--extracted-root", default="data/extracted")
    parser.add_argument("--out-root", default="data/wav16k")
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--workers", type=int, default=8)
    # Per-corpus hour caps. Omit / set <=0 to keep all audio for that corpus.
    parser.add_argument("--cve-target-hours", type=float, default=200.0)
    parser.add_argument("--cvr-target-hours", type=float, default=250.0)
    parser.add_argument("--golos-target-hours", type=float, default=0.0,
                        help="<=0 means keep all Golos")
    args = parser.parse_args()

    extracted_root = Path(args.extracted_root)
    out_root = Path(args.out_root)
    metadata_dir = Path(args.metadata_dir)

    all_corpora = ["ksc2", "kazemotts", "cve", "cvr", "golos", "kazmedspeech", "fleurs_kk"]
    sweep = args.corpus == "all"
    targets = all_corpora if sweep else [args.corpus]

    # None = no cap (keep all). Kazakh corpora are never capped.
    def _cap(v: float) -> float | None:
        return v if v and v > 0 else None

    caps = {
        "ksc2": None,
        "kazemotts": None,
        "cve": _cap(args.cve_target_hours),
        "cvr": _cap(args.cvr_target_hours),
        "golos": _cap(args.golos_target_hours),
        "kazmedspeech": None,
        "fleurs_kk": None,   # eval set, used in full
    }

    for corpus in targets:
        prepare_corpus(corpus, extracted_root, out_root, metadata_dir, caps[corpus], args.workers, missing_ok=sweep)


if __name__ == "__main__":
    main()
