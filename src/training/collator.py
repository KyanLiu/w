"""Whisper speech-seq2seq data collator to pad into a loss-masked batch"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from training.constants import LABEL_IGNORE_INDEX


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        # pad the log-Mel features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # pad the label token sequences
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # mask padding from the loss with the ignore index
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), LABEL_IGNORE_INDEX
        )

        # strip the start-of-transcript token
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
