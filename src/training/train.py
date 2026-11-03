"""
Train for phase-a.yaml
Will resume automatically from latest checkpoint
"""
from __future__ import annotations
import argparse
from pathlib import Path
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    set_seed,
)
from training.collator import DataCollatorSpeechSeq2SeqWithPadding
from training.config import load_config
from training.dataset import build_dataset, load_processor
from training.metrics import build_compute_metrics
from training.train_utils import compute_grad_accum, preflight_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to the config YAML"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="do not resume from the latest checkpoint"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="cap training at N steps"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    t = cfg.training
    print(f"[train] phase={cfg.phase} init={cfg.init_model} seed={t.seed}")
    set_seed(t.seed)

    # ---- data ----
    processor = load_processor(cfg.init_model)
    dataset = build_dataset(
        cfg.data.train_metadata,
        cfg.data.eval_metadata,
        processor,
        max_audio_seconds=cfg.data.max_audio_seconds,
        num_proc=cfg.data.num_proc,
    )
    print(f"[train] data sizes: train={dataset['train'].num_rows}, eval={dataset['eval'].num_rows}")

    # ---- model
    print(f"[train] loading model: {cfg.init_model}")
    model = WhisperForConditionalGeneration.from_pretrained(cfg.init_model)
    # configure generation
    model.generation_config.language = cfg.language
    model.generation_config.task = cfg.task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    compute_metrics = build_compute_metrics(processor)

    # Probe check before
    preflight_probe(dataset, collator, processor)

    # ---- effective batch math ----
    num_gpus, grad_accum = compute_grad_accum(
        t.effective_batch_size, t.per_device_train_batch_size
    )
    eff = t.per_device_train_batch_size * num_gpus * grad_accum
    print(f"[train] GPUs={num_gpus} per_device={t.per_device_train_batch_size}, grad_accum={grad_accum} -> effective_batch={eff}")

    # ---- test override ----
    smoke = args.max_steps > 0
    if smoke:
        num_train_epochs = 1  # ignored once max_steps is set, but keep valid
        max_steps = args.max_steps
        eval_steps = args.max_steps
        save_steps = args.max_steps
        print(f"[train] TEST: max_steps={max_steps}, eval+save at step {max_steps}")
    else:
        num_train_epochs = t.num_train_epochs
        max_steps = -1  # HF default: epoch-driven
        eval_steps = t.eval_steps
        save_steps = t.save_steps

    # ---- training args ----
    targs = Seq2SeqTrainingArguments(
        output_dir=t.output_dir,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_ratio=t.warmup_ratio,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        bf16=t.bf16,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        seed=t.seed,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=t.save_total_limit,
        predict_with_generate=t.predict_with_generate,
        generation_max_length=t.generation_max_length,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        load_best_model_at_end=t.load_best_model_at_end,
        dataloader_num_workers=t.dataloader_num_workers,
        report_to=[t.report_to] if t.report_to else [],
        logging_steps=t.logging_steps,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    # ---- resume logic ----
    resume = None
    out = Path(t.output_dir)
    if not smoke and not args.no_resume and out.exists() and any(out.glob("checkpoint-*")):
        resume = True
        print(f"[train] resuming from latest checkpoint in {out}")

    print("[train] starting training...")
    trainer.train(resume_from_checkpoint=resume)

    # ---- save final/best ----
    trainer.save_model(str(out / "best"))
    processor.save_pretrained(str(out / "best"))
    print(f"[train] done. best model (lowest WER) saved under: {out / 'best'}")


if __name__ == "__main__":
    main()
