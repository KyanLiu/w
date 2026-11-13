"""Build JSONL metadata from the selection files written by prepare_audio"""
import sys
import json
import random
import argparse
from pathlib import Path
from preprocess.constants import SEED, CORPUS_LANGUAGE, OPTIONAL_CLIP_FIELDS
from preprocess.metadata_utils import write_jsonl


def _load_selection(corpus: str, metadata_dir: Path, missing_ok: bool = False) -> list[dict] | None:
    path = metadata_dir / f"{corpus}_selection.jsonl"
    if not path.exists():
        if missing_ok:
            print(f"[skip] {corpus}: no selection file yet ({path})")
            return None
        sys.exit(f"[error] selection not found: {path} "
                 f"(run prepare_audio --corpus {corpus} first)")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _row(rel_key: str, text: str, duration, corpus: str, split: str,
         wav_root: Path, base_dir: Path, extra: dict | None = None) -> dict | None:
    wav = wav_root / rel_key
    if not wav.exists():
        return None
    try:
        rel = wav.resolve().relative_to(base_dir).as_posix()
    except ValueError:
        rel = wav.as_posix()
    row = {
        "audio_path": rel,
        "text": text,
        "language": CORPUS_LANGUAGE[corpus],
        "corpus": corpus,
        "split": split,
        "duration": round(float(duration), 3),
    }
    # optional eval-only fields (notations -> MER, domain -> per-domain WER,
    # recording -> bootstrap grouping)
    if extra:
        for key in OPTIONAL_CLIP_FIELDS:
            if extra.get(key) is not None:
                row[key] = extra[key]
    return row


def build_corpus_native_splits(
    corpus: str, wav_root: Path, metadata_dir: Path,
    base_dir: Path, missing_ok: bool = False,
) -> None:
    """Build one JSONL per NATIVE split (train/dev/test) for corpora that ship
    their own splits (KazMedSpeech). Unlike build_corpus, this does NOT
    draw a random holdout — it groups selection rows by their recorded `split`
    so the corpus's own train/dev/test are preserved. The `test` split is built
    but reserved (not referenced by the training config)."""
    sel = _load_selection(corpus, metadata_dir, missing_ok)
    if sel is None:
        return

    by_split: dict[str, list[dict]] = {}
    missing = 0
    for r in sel:
        split = r.get("split", "train")
        extra = {k: r.get(k) for k in OPTIONAL_CLIP_FIELDS if k in r}
        row = _row(r["rel_key"], r["text"], r["duration"], corpus, split,
                   wav_root, base_dir, extra=extra)
        if row is None:
            missing += 1
        else:
            by_split.setdefault(split, []).append(row)
    if not by_split:
        sys.exit(f"[error] {corpus}: no converted WAVs found (run prepare_audio --corpus {corpus})")
    if missing:
        print(f"[warn] {corpus}: {missing} selected clips had no WAV (skipped)")

    lang = CORPUS_LANGUAGE[corpus]
    for split, rows in sorted(by_split.items()):
        rows.sort(key=lambda r: r["audio_path"])
        out = metadata_dir / f"{corpus}_{split}.jsonl"
        write_jsonl(rows, out)
        total_h = sum(r["duration"] for r in rows) / 3600.0
        note = " (reserved for eval, not used in training)" if split == "test" else ""
        print(f"[{corpus}] {split}={len(rows)} ({total_h:.1f} h, language={lang}){note}")


def build_corpus(
    corpus: str, wav_root: Path, metadata_dir: Path,
    base_dir: Path, eval_count: int = 0, missing_ok: bool = False
) -> None:
    """Build train/eval JSONL for one corpus from its selection"""
    sel = _load_selection(corpus, metadata_dir, missing_ok)
    if sel is None:
        return

    rows, missing = [], 0
    for r in sel:
        row = _row(r["rel_key"], r["text"], r["duration"], corpus, "train", wav_root, base_dir)
        if row is None:
            missing += 1
        else:
            rows.append(row)
    if not rows:
        sys.exit(f"[error] {corpus}: no converted WAVs found "
                 f"(run prepare_audio --corpus {corpus})")
    if missing:
        print(f"[warn] {corpus}: {missing} selected clips had no WAV (skipped)")

    lang = CORPUS_LANGUAGE[corpus]

    if eval_count <= 0:
        write_jsonl(rows, metadata_dir / f"{corpus}_train.jsonl")
        total_h = sum(r["duration"] for r in rows) / 3600.0
        print(f"[{corpus}] train={len(rows)} ({total_h:.1f} h, language={lang})")
        return

    # Seeded, disjoint eval set
    rows.sort(key=lambda r: r["audio_path"])
    if len(rows) < eval_count:
        sys.exit(f"[error] {corpus}: only {len(rows)} clips, need {eval_count} for eval")
    rng = random.Random(SEED)
    eval_idx = set(rng.sample(range(len(rows)), eval_count))

    eval_rows, train_rows = [], []
    for i, row in enumerate(rows):
        if i in eval_idx:
            r2 = dict(row); r2["split"] = "eval"
            eval_rows.append(r2)
        else:
            train_rows.append(row)

    write_jsonl(train_rows, metadata_dir / f"{corpus}_train.jsonl")
    write_jsonl(eval_rows, metadata_dir / f"{corpus}_eval_{eval_count}.jsonl")
    train_h = sum(r["duration"] for r in train_rows) / 3600.0
    eval_h = sum(r["duration"] for r in eval_rows) / 3600.0
    print(f"[{corpus}] train={len(train_rows)} ({train_h:.1f} h), "
          f"eval={len(eval_rows)} ({eval_h:.1f} h, held out, disjoint), "
          f"language={lang}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus",
        choices=["ksc2", "kazemotts", "cve", "cvr", "golos", "kazmedspeech", "fleurs_kk", "all"],
        default="all")
    parser.add_argument("--wav-root", default="data/wav16k")
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--base-dir", default=None, help="dir that audio_path is relative to (default: cwd)")
    # Per-corpus eval holdout sizes (0 = no eval split, pure train).
    parser.add_argument("--ksc2-eval-count", type=int, default=597)
    parser.add_argument("--cve-eval-count", type=int, default=400)
    parser.add_argument("--cvr-eval-count", type=int, default=400)
    args = parser.parse_args()

    wav_root = Path(args.wav_root)
    metadata_dir = Path(args.metadata_dir)
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()

    eval_count = {
        "ksc2": args.ksc2_eval_count,
        "kazemotts": 0,
        "cve": args.cve_eval_count,
        "cvr": args.cvr_eval_count,
        "golos": 0,
    }

    # Corpora that ship their own train/dev/test splits
    native_split_corpora = {"kazmedspeech", "fleurs_kk"}

    all_corpora = ["ksc2", "kazemotts", "cve", "cvr", "golos", "kazmedspeech", "fleurs_kk"]
    sweep = args.corpus == "all"
    targets = all_corpora if sweep else [args.corpus]

    for corpus in targets:
        if corpus in native_split_corpora:
            build_corpus_native_splits(
                corpus, wav_root / corpus, metadata_dir, base_dir, missing_ok=sweep)
        else:
            build_corpus(
                corpus, wav_root / corpus, metadata_dir, base_dir,
                eval_count=eval_count[corpus], missing_ok=sweep)


if __name__ == "__main__":
    main()
