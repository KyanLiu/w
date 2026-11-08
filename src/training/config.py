""" Typed config loader """
from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
import yaml


@dataclass
class DataConfig:
    train_metadata: list[str]
    eval_metadata: str | dict[str, str]
    max_audio_seconds: float = 30.0
    num_proc: int = 8


@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints/new_phase"

    effective_batch_size: int = 160
    per_device_train_batch_size: int = 10
    per_device_eval_batch_size: int = 10

    learning_rate: float = 1.0e-5
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.01
    num_train_epochs: int = 3
    bf16: bool = True
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42

    eval_steps: int = 1112
    save_steps: int = 1112
    save_total_limit: int = 3

    predict_with_generate: bool = True
    generation_max_length: int = 225

    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True

    dataloader_num_workers: int = 8
    report_to: str = "tensorboard"
    logging_steps: int = 50


@dataclass
class PhaseConfig:
    phase: str
    init_model: str
    data: DataConfig
    training: TrainingConfig
    language: str | None = "kk"   # None = Whisper auto-detects language at generation
    task: str = "transcribe"
    timestamps: bool = False


def _filter_known(cls, d: dict) -> dict:
    """Keep only keys that are fields of the dataclass"""
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    return {k: v for k, v in d.items() if k in known}


def load_config(path: str | Path) -> PhaseConfig:
    """Load a phase YAML into a typed PhaseConfig."""
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    data = DataConfig(**_filter_known(DataConfig, raw.pop("data", {})))
    training = TrainingConfig(**_filter_known(TrainingConfig, raw.pop("training", {})))
    return PhaseConfig(data=data, training=training,
        **_filter_known(PhaseConfig,
        {k: v for k, v in raw.items()
        if k not in ("data", "training")})
    )
