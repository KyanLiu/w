"""
Convert
"""
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from preprocess.constants import SRC_SUBDIR, CORPUS_AUDIO_EXT, TARGET_SAMPLE_RATE

def _convert_single_corpus(src: str, dst: str) -> tuple[str, str | None]:
    try:
        import soundfile as sf
        import numpy as np

        if Path(dst).exists():
            return (src, None)
        audio, sr = sf.read(src, always_2d=True)
        # averaging audio channels to mono-channel
        audio = audio.mean(axis=1)

        if sr != TARGET_SAMPLE_RATE:
            import soxr
            audio = soxr.resample(audio, sr, TARGET_SAMPLE_RATE)

        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        tmp = dst + ".tmp"
        sf.write(tmp, np.clip(audio, -1.0, 1.0).astype(np.float32), TARGET_SAMPLE_RATE, subtype="PCM_16", format="WAV")
        Path(tmp).replace(dst)
        return (src, None)
    except Exception as e:
        return (src, f"{type(e).__name__}: {e}")


def _convert_one(job: tuple[str, str]) -> tuple[str, str | None]:
    """Single-arg wrapper so the job generator can be streamed via executor.map."""
    src, dst = job
    return _convert_single_corpus(src, dst)


def _iter_search_jobs(src_root: Path, ext: str, out_root: Path):
    # Return all searches
    for src in src_root.rglob(f"*{ext}"):
        rel = src.relative_to(src_root).with_suffix(".wav")
        yield str(src), str(out_root / rel)

def convert_corpus(corpus: str, extracted_root: Path, out_root: Path, workers: int,
                   missing_ok: bool = False) -> None:
    sub = SRC_SUBDIR[corpus]
    ext = CORPUS_AUDIO_EXT[corpus]
    src_root = extracted_root / sub
    dst_root = out_root / corpus

    marker = dst_root / ".convert.done"

    if marker.exists():
        print(f"[skip] {corpus} skipped: already converted in {dst_root}")
        return
    if not src_root.exists():
        # For an explicit single corpus this is a hard error
        if missing_ok:
            print(f"[skip] {corpus} skipped: source not found: {src_root}")
            return
        sys.exit(f"[error] source not found: {src_root}")


    # Stream jobs lazily through the pool now
    jobs = _iter_search_jobs(src_root, ext, dst_root)

    print(f"[convert] {corpus} converting (streaming), {workers} workers, ext={ext}")

    done = 0
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for src, err in executor.map(_convert_one, jobs, chunksize=64):
            done += 1
            if err:
                errors.append(f"{src}: {err}")
            if done % 2000 == 0:
                print(f"{done} done ({len(errors)} errors)")

    if done == 0:
        sys.exit(f"[error] no {ext} files under {src_root}")

    print(f"{done} done ({len(errors)} errors)")

    if errors:
        print(f"\n[warn] {len(errors)} files failed - not writing .done marker.")
        for line in errors[:20]:
            print(line)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more")
        sys.exit(1)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")
    print(f"[ok] {corpus}: all {done} files converted -> {dst_root}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Corpus choice
    parser.add_argument(
        "--corpus",
        choices=["ksc2", "kazemotts", "cve", "cvr", "golos", "all"],
        default="all"
    )
    # Storing extracted data
    parser.add_argument(
        "--extracted-root",
        default="data/extracted"
    )
    # WAVs location written
    parser.add_argument(
        "--out-root",
        default="data/wav16k"
    )
    # Parallel worker processes
    parser.add_argument(
        "--workers",
        type=int,
        default=8
    )
    args = parser.parse_args()

    extracted_root = Path(args.extracted_root)
    out_root = Path(args.out_root)
    all_corpora = ["ksc2", "kazemotts", "cve", "cvr", "golos"]
    sweep = args.corpus == "all"
    targets = all_corpora if sweep else [args.corpus]

    for corpus in targets:
        convert_corpus(corpus, extracted_root, out_root, args.workers,
                       missing_ok=sweep)


if __name__ == "__main__":
    main()
