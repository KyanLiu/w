"""
Build the dataset from JSONL metadata
1. Pair each row's audio with its transcript and loads the audio from audio path lazily
2. Filter out audio lengths longer than max samples
3. Convert audio to log-mel spectrogram and text to tokens
"""
from __future__ import annotations
import argparse
from typing import Any

import numpy as np
import soundfile as sf
from datasets import load_dataset
from transformers import WhisperProcessor
from training.constants import (
    LANGUAGE,
    LANGUAGE_NAMES,
    MAX_AUDIO_SECONDS,
    MODEL_ID,
    TARGET_SR,
    TASK,
)


def load_processor(model_id: str = MODEL_ID):
    """Load the WhisperProcessor"""
    processor = WhisperProcessor.from_pretrained(model_id, language=LANGUAGE, task=TASK)
    return processor


def build_dataset(
    train_metadata: list[str],
    eval_metadata: str | dict[str, str],
    processor: Any,
    max_audio_seconds: float = MAX_AUDIO_SECONDS,
    num_proc: int = 1):
    """Build a DatasetDict with lazily-featurized splits"""
    if isinstance(eval_metadata, dict):
        data_files = {"train": train_metadata}
        for name, path in eval_metadata.items():
            data_files[f"eval_{name}"] = path
    else:
        data_files = {"train": train_metadata, "eval": eval_metadata}

    raw = load_dataset("json", data_files=data_files)

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    def _not_overlength(example: dict) -> bool:
        """Keep clips no longer than max_audio_seconds"""
        return example["duration"] <= max_audio_seconds

    raw = raw.filter(_not_overlength, num_proc=num_proc)

    def _load_audio(path: str) -> np.ndarray:
        """Read a WAV to a mono float32 array - assert it is at TARGET_SR"""
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != TARGET_SR:
            raise RuntimeError(
                f"{path}: sample rate {sr} != expected {TARGET_SR} (audio must be pre-converted to 16k)"
            )
        return audio.mean(axis=1)  # mono-channel

    def _prepare(batch: dict) -> dict:
        """Lazy per-batch transform: audio path -> log-Mel, text -> token ids"""
        paths = batch["audio_path"]
        if isinstance(paths, list):
            arrays = [_load_audio(p) for p in paths]
            texts = batch["text"]
            langs = batch["language"]
        else:
            arrays = [_load_audio(paths)]
            texts = [batch["text"]]
            langs = [batch["language"]]

        feats = feature_extractor(
            arrays, sampling_rate=TARGET_SR
        ).input_features

        # Tokenize each label with its own language prefix to get tokens
        labels = []
        current_lang = None
        for text, lang in zip(texts, langs):
            full_lang = LANGUAGE_NAMES.get(lang, lang)
            if full_lang != current_lang:
                tokenizer.set_prefix_tokens(language=full_lang, task=TASK)
                current_lang = full_lang
            labels.append(tokenizer(text).input_ids)

        return {"input_features": feats, "labels": labels}

    raw = raw.with_transform(_prepare)
    return raw

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--train-metadata", 
        action="append", 
        required=True,
        dest="train_metadata",
        help="path to train JSONL"
    )
    parser.add_argument(
        "--eval-metadata",
        required=True,
        help="path to the eval JSONL"
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float, default=MAX_AUDIO_SECONDS
    )
    parser.add_argument(
        "--num-proc",
        type=int, default=1,
        help="processes for the over-length filter pass"
    )
    args = parser.parse_args()

    print(f"[dataset] loading processor: {args.model_id}")
    processor = load_processor(args.model_id)

    print(f"[dataset] building from: train={args.train_metadata}, eval={args.eval_metadata}")
    dataset = build_dataset(
        args.train_metadata,
        args.eval_metadata,
        processor,
        max_audio_seconds=args.max_audio_seconds,
        num_proc=args.num_proc,
    )
    eval_names = [k for k in dataset.keys() if k.startswith("eval")]
    sizes = {k: dataset[k].num_rows for k in ["train"] + eval_names}
    print(f"[dataset] sizes: {sizes}")

    print("[dataset] ready (lazy)")


if __name__ == "__main__":
    main()
