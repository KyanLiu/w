"""Helpers for train.py."""
from __future__ import annotations
from training.constants import LABEL_IGNORE_INDEX, LANGUAGE_NAMES
import torch


def compute_grad_accum(effective_batch: int, per_device: int) -> tuple[int, int]:
    """Derive gradient_accumulation_steps from the visible GPU count"""
    num_gpus = max(1, torch.cuda.device_count())
    denom = per_device * num_gpus
    if effective_batch % denom != 0:
        grad_accum = max(1, round(effective_batch / denom))
        actual = per_device * num_gpus * grad_accum
        print(f"[train][warn] effective batch {effective_batch} not divisible by per_device({per_device}) * num_gpus({num_gpus}) = {denom}; using grad_accum={grad_accum} -> actual effective batch {actual}")
    else:
        grad_accum = effective_batch // denom
    return num_gpus, grad_accum


def preflight_probe(dataset, collator, processor) -> None:
    """Featurize one small batch through the collator to validate shape + kk token."""
    print("[train] pre-flight probe: featurizing one batch through the collator")
    n = min(2, dataset["train"].num_rows)
    examples = [dataset["train"][i] for i in range(n)]

    # each dataset row (with_transform) yields batched 1-element arrays; unwrap
    unwrapped = []
    for ex in examples:
        feat = ex["input_features"]
        lab = ex["labels"]
        feat0 = feat[0] if (hasattr(feat, "__len__") and len(feat) == 1) else feat
        lab0 = lab[0] if (lab and isinstance(lab[0], list)) else lab
        unwrapped.append({"input_features": feat0, "labels": lab0})

    batch = collator(unwrapped)
    feats = batch["input_features"]
    labels = batch["labels"]
    print(f"input_features tensor: {tuple(feats.shape)} (expect [*, 128, 3000])")
    print(f"labels tensor: {tuple(labels.shape)}")

    n_mels = feats.shape[-2]
    if n_mels != 128:
        raise RuntimeError(
            f"pre-flight: expected 128 mel bins, got {n_mels} — feature extractor / model mismatch."
        )

    # decode first label row
    pad_id = processor.tokenizer.pad_token_id
    first = labels[0].clone()
    first[first == LABEL_IGNORE_INDEX] = pad_id
    decoded = processor.tokenizer.decode(first, skip_special_tokens=False)
    print(f"decoded[0]: {decoded[:140]}")

    # A valid Whisper language prefix token must be present
    expected = [f"<|{code}|>" for code in LANGUAGE_NAMES]
    ids = {tok: processor.tokenizer.convert_tokens_to_ids(tok) for tok in expected}
    present = [tok for tok, tid in ids.items()
               if (labels == tid).any().item() or tok in decoded]
    if not present:
        raise RuntimeError(
            f"pre-flight: none of {expected} found in labels — processor language is misconfigured. Aborting before training."
        )

    print(f"[ok] language token(s) present: {present}")
    print("[train] pre-flight probe passed.")
