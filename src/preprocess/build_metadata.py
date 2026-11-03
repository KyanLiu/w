"""
Build the JSONL data based on WAVs and .txt transcripts
"""
import sys
import argparse
import random
from pathlib import Path

from preprocess.constants import SEED, SRC_SUBDIR
from preprocess.metadata_utils import (
    build_entries_parallel,
    match_wavs_with_text,
    write_jsonl,
)


def build_ksc2(wav_root: Path, src_root: Path, metadata_dir: Path,
               eval_source: str, eval_count: int, base_dir: Path,
               workers: int) -> None:
    """Build the JSONL for KSC2 train and eval"""
    train_split_dir = wav_root / "Train"
    eval_split_dir = wav_root / eval_source

    if not train_split_dir.exists():
        sys.exit(f"[error] expected KSC2 Train dir: {train_split_dir}")
    if not eval_split_dir.exists():
        sys.exit(f"[error] expected KSC2 {eval_source} dir: {eval_split_dir}")

    print(f"[ksc2] collecting all training data from {train_split_dir}")
    train_pairs = list(match_wavs_with_text(train_split_dir, src_root / "Train"))
    train_rows = build_entries_parallel(train_pairs, "ksc2", "train", base_dir, workers)

    print(f"[ksc2] collecting all eval data {eval_source} from {eval_split_dir}")
    eval_pool = [
        (wav, txt) for wav, txt in match_wavs_with_text(eval_split_dir, src_root / eval_source)
    ]
    eval_pool.sort(key=lambda wt: wt[0].as_posix())

    if len(eval_pool) < eval_count:
        sys.exit(f"[error] {eval_source} has only {len(eval_pool)} usable data entries out of {eval_count} for eval")

    # Generate the selected eval set
    rng = random.Random(SEED)
    idxs = sorted(rng.sample(range(len(eval_pool)), eval_count))
    eval_set = set(idxs)

    eval_rows = build_entries_parallel(
        [eval_pool[i] for i in idxs], "ksc2", "eval", base_dir, workers
    )
    # Remaining data is combined with training rows
    leftover_pairs = [
        (wav, txt) for i, (wav, txt) in enumerate(eval_pool) if i not in eval_set
    ]
    leftover_rows = build_entries_parallel(
        leftover_pairs, "ksc2", "train", base_dir, workers
    )
    train_rows.extend(leftover_rows)

    write_jsonl(train_rows, metadata_dir / "ksc2_train.jsonl")
    write_jsonl(eval_rows, metadata_dir / f"ksc2_eval_{eval_count}.jsonl")
    print(f"[ksc2] written jsonl, train={len(train_rows)} "
          f"(Train + {len(leftover_rows)} leftover {eval_source}), "
          f"eval={len(eval_rows)} (held out, disjoint)")



def build_kazemotts(wav_root: Path, src_root: Path, metadata_dir: Path,
                    base_dir: Path, workers: int) -> None:
    """Build the JSONL for all KazEmoTTS train/eval audio"""
    print(f"[kazEmoTTS] collecting all data from {wav_root}")
    pairs = list(match_wavs_with_text(wav_root, src_root))
    rows = build_entries_parallel(pairs, "kazemotts", "train", base_dir, workers)
    if not rows:
        sys.exit(f"[error] no KazEmoTTS WAVs found in {wav_root}")

    write_jsonl(rows, metadata_dir / "kazemotts_train.jsonl")
    print(f"[kazEmoTTS] written jsonl, train={len(rows)}")

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Corpus choice
    parser.add_argument(
        "--corpus",
        choices=["ksc2", "kazemotts", "all"],
        default="all"
    )
    # Location of converted WAVs
    parser.add_argument(
        "--wav-root",
        default="data/wav16k"
    )
    # Storing extracted data
    parser.add_argument(
        "--extracted-root",
        default="data/extracted"
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/metadata"
    )
    parser.add_argument(
        "--ksc2-eval-source",
        choices = ["Test", "Dev"],
        default="Test"
    )
    parser.add_argument(
        "--ksc2-eval-count",
        type=int,
        default=597
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="directory that audio_path is made relative to (default: cwd)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel processes for reading audio durations"
    )

    args = parser.parse_args()

    wav_root = Path(args.wav_root)
    extracted_root = Path(args.extracted_root)
    metadata_dir = Path(args.metadata_dir)
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    targets = ["ksc2", "kazemotts"] if args.corpus == "all" else [args.corpus]

    for corpus in targets:
        src_root = extracted_root / SRC_SUBDIR[corpus]
        corpus_wav_root = wav_root / corpus
        if not corpus_wav_root.exists():
            sys.exit(f"[error] converted WAVs not found: {corpus_wav_root}")
        if corpus == "ksc2":
            build_ksc2(
                corpus_wav_root,
                src_root,
                metadata_dir,
                args.ksc2_eval_source,
                args.ksc2_eval_count,
                base_dir,
                args.workers
            )
        elif corpus == "kazemotts":
            build_kazemotts(corpus_wav_root, src_root, metadata_dir, base_dir, args.workers)




if __name__ == "__main__":
    main()
