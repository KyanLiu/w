"""
WER computation metrics for evaluation, compute_metrics used by Seq2SeqTrainer calls
WER calculated via jiwer as a percent
"""
from __future__ import annotations
from typing import Any, Callable
from preprocess import normalize
from training.constants import LABEL_IGNORE_INDEX
import jiwer


def build_compute_metrics(processor: Any) -> Callable:
    """Return a compute_metrics(pred) closure bound to this processor"""
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id

    def compute_metrics(pred) -> dict:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        pred_ids[pred_ids == LABEL_IGNORE_INDEX] = pad_id
        label_ids[label_ids == LABEL_IGNORE_INDEX] = pad_id
        # Convert back to text
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [normalize(s) for s in pred_str]
        label_str = [normalize(s) for s in label_str]

        refs, hyps = [], []
        for r, h in zip(label_str, pred_str):
            if r.strip():
                refs.append(r)
                hyps.append(h)

        wer = jiwer.wer(refs, hyps) if refs else 0.0
        return {"wer": 100.0 * wer}

    return compute_metrics
