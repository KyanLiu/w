"""
Build the dataset from JSONL metadata
1. Pair each row's audio with its transcript and loads the audio from audio path lazily
2. Filter out audio lengths longer than max samples
3. Convert audio to log-mel spectrogram and text to tokens
"""
from __future__ import annotations
import argparse
from typing import Any
from datasets import Audio, load_dataset
from transformers import WhisperProcessor
from training.constants import (
    LANGUAGE,
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
    eval_metadata: str,
    processor: Any,
    max_audio_seconds: float = MAX_AUDIO_SECONDS,
    num_proc: int = 1):
    """Build a DatasetDict with lazily-featurized `train` and `eval` splits"""
    raw = load_dataset(
        "json",
        data_files={"train": train_metadata, "eval": eval_metadata},
    )

    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    def _not_overlength(example: dict) -> bool:
        """Keep clips no longer than max_audio_seconds"""
        return example["duration"] <= max_audio_seconds

    raw = raw.filter(_not_overlength, num_proc=num_proc)

    # Lazily load the data, "audio_path" -> audio data (decoded on access)
    raw = raw.cast_column("audio_path", Audio(sampling_rate=TARGET_SR))

    def _prepare(batch: dict) -> dict:
        """Lazy per-batch transform: audio -> log-Mel, text -> token ids"""
        audio = batch["audio_path"]
        if isinstance(audio, list):
            arrays = [a["array"] for a in audio]
            texts = batch["text"]
        else:
            arrays = [audio["array"]]
            texts = [batch["text"]]

        feats = feature_extractor(
            arrays, sampling_rate=TARGET_SR
        ).input_features
        labels = tokenizer(texts).input_ids

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
    print(f"[dataset] sizes: train={dataset['train'].num_rows}, eval={dataset['eval'].num_rows}")

    print("[dataset] ready (lazy)")


if __name__ == "__main__":
    main()
