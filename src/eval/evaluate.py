"""
Evaluation of an ASR model
Loads one or more models, greedy-decodes each eval set, and reports WER + CER
(+ MER for KazMedSpeech) with 2,000-resample bootstrap confidence intervals,
plus an optional per-medical-domain WER breakdown. 
Writes a JSON + markdown report
"""
from __future__ import annotations

import argparse
import json
import time
import torch
import yaml
from datetime import datetime
from pathlib import Path
from transformers import WhisperForConditionalGeneration

from training.collator import DataCollatorSpeechSeq2SeqWithPadding
from training.dataset import build_dataset, load_processor
from eval.metrics import (
    MetricCounts,
    wer_counts,
    cer_counts,
    mer_counts,
    parse_medical_terms,
    aggregate,
)
from eval.bootstrap import bootstrap_ci


def _load_eval_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _decode_set(model, processor, collator, jsonl_path: str, cfg: dict, device) -> list[dict]:
    """Greedy-decode one eval set. Returns a list of per-clip result dicts with hyp, ref, and the original metadata row"""
    # Build a single-split eval dataset via the training path 
    ds = build_dataset(
        train_metadata=[jsonl_path],
        eval_metadata=jsonl_path,
        processor=processor,
        max_audio_seconds=cfg.get("max_audio_seconds", 30),
        num_proc=cfg.get("num_proc", 8),
    )
    eval_ds = ds["eval"]

    # Also load the raw rows (for notations/domain/recording), aligned by order.
    raw_rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_rows.append(json.loads(line))

    dec = cfg.get("decoding", {})
    batch_size = dec.get("per_device_eval_batch_size", 16)
    gen_max_len = dec.get("generation_max_length", 225)
    num_beams = dec.get("num_beams", 1)  # greedy default

    results: list[dict] = []
    model.eval()
    n = eval_ds.num_rows
    idx = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_rows = [eval_ds[i] for i in range(start, min(start + batch_size, n))]
            # unwrap with_transform's 1-element batching
            feats = []
            for ex in batch_rows:
                f = ex["input_features"]
                f0 = f[0] if (hasattr(f, "__len__") and len(f) == 1 and not isinstance(f[0], float)) else f
                feats.append({"input_features": f0, "labels": ex["labels"][0]
                              if (ex["labels"] and isinstance(ex["labels"][0], list)) else ex["labels"]})
            collated = collator(feats)
            input_features = collated["input_features"].to(device)
            gen_ids = model.generate(
                input_features=input_features,
                num_beams=num_beams,
                do_sample=False,
                max_length=gen_max_len,
                language=cfg.get("language", "kk"),
                task=cfg.get("task", "transcribe"),
            )
            hyps = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for h in hyps:
                row = raw_rows[idx]
                results.append({"hyp": h, "ref": row["text"], "row": row})
                idx += 1
    return results


def _metrics_for_set(set_name: str, results: list[dict], cfg: dict) -> dict:
    """Compute WER + CER (+ MER if notations present) + bootstrap CIs + optional per-domain WER for one decoded eval set."""
    metrics_cfg = cfg.get("metrics", {})
    n_boot = metrics_cfg.get("bootstrap_resamples", 2000)
    seed = metrics_cfg.get("seed", 42)

    wer_comps: list[MetricCounts] = []
    cer_comps: list[MetricCounts] = []
    groups: list[str] = []
    mer_comps: list[MetricCounts] = []
    mer_groups: list[str] = []
    has_medical = False
    domain_comps: dict[str, list[MetricCounts]] = {}

    for i, r in enumerate(results):
        row = r["row"]
        wc = wer_counts(r["ref"], r["hyp"])
        wer_comps.append(wc)
        cer_comps.append(cer_counts(r["ref"], r["hyp"]))
        # bootstrap grouping
        g = row.get("recording") or f"__clip_{i}"
        groups.append(str(g))

        # MER (only if this row carries medical notations)
        notations = row.get("notations")
        if notations is not None:
            has_medical = True
            terms = parse_medical_terms(notations)
            mc = mer_counts(r["ref"], r["hyp"], terms)
            mer_comps.append(mc)
            mer_groups.append(str(g))

        # per-domain WER (optional)
        domain = row.get("domain")
        if domain:
            domain_comps.setdefault(domain, []).append(wc)

    out: dict = {"n_clips": len(results)}
    wer_ci = bootstrap_ci(wer_comps, groups, resamples=n_boot, seed=seed)
    out["wer"] = {"point": wer_ci.point, "ci_lo": wer_ci.lo, "ci_hi": wer_ci.hi}

    cer_ci = bootstrap_ci(cer_comps, groups, resamples=n_boot, seed=seed)
    out["cer"] = {"point": cer_ci.point, "ci_lo": cer_ci.lo, "ci_hi": cer_ci.hi}

    if has_medical and any(c.length > 0 for c in mer_comps):
        mer_ci = bootstrap_ci(mer_comps, mer_groups, resamples=n_boot, seed=seed)
        out["mer"] = {"point": mer_ci.point, "ci_lo": mer_ci.lo, "ci_hi": mer_ci.hi}

    if domain_comps:
        out["wer_by_domain"] = {
            d: round(aggregate(cs), 3) for d, cs in sorted(domain_comps.items())
        }
    return out


def _write_report(report: dict, output_dir: str) -> tuple[Path, Path]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tag = report["model_tag"]
    ts = report["timestamp"]
    base = Path(output_dir) / f"{tag}_{ts}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # markdown
    lines = [f"# Evaluation: {report['model']}", "",
             f"Run: {ts}", ""]
    for set_name, m in report["sets"].items():
        lines.append(f"## {set_name}  (n={m['n_clips']})")
        w = m["wer"]
        lines.append(f"- **WER**: {w['point']:.2f}%  (95% CI {w['ci_lo']:.2f}–{w['ci_hi']:.2f})")
        if "cer" in m:
            c = m["cer"]
            lines.append(f"- **CER**: {c['point']:.2f}%  (95% CI {c['ci_lo']:.2f}–{c['ci_hi']:.2f})")
        if "mer" in m:
            me = m["mer"]
            lines.append(f"- **MER**: {me['point']:.2f}%  (95% CI {me['ci_lo']:.2f}–{me['ci_hi']:.2f})")
        if "wer_by_domain" in m:
            lines.append("- WER by medical domain:")
            for d, v in m["wer_by_domain"].items():
                lines.append(f"    - {d}: {v:.2f}%")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return json_path, md_path


def _evaluate_one_model(model_path: str, cfg: dict, device) -> dict:
    print(f"[eval] loading model: {model_path}")
    processor = load_processor(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    model.generation_config.language = cfg.get("language", "kk")
    model.generation_config.task = cfg.get("task", "transcribe")
    model.generation_config.forced_decoder_ids = None
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    sets_out: dict = {}
    for set_name, jsonl_path in cfg["eval_sets"].items():
        if not Path(jsonl_path).exists():
            print(f"[eval][skip] {set_name}: {jsonl_path} not found")
            continue
        t0 = time.time()
        print(f"[eval] decoding {set_name} ({jsonl_path}) ...")
        results = _decode_set(model, processor, collator, jsonl_path, cfg, device)
        sets_out[set_name] = _metrics_for_set(set_name, results, cfg)
        dt = time.time() - t0
        w = sets_out[set_name]["wer"]
        cer_str = f" | CER {sets_out[set_name]['cer']['point']:.2f}%" if "cer" in sets_out[set_name] else ""
        extra = f" | MER {sets_out[set_name]['mer']['point']:.2f}%" if "mer" in sets_out[set_name] else ""
        print(f"[eval] {set_name}: WER {w['point']:.2f}%{cer_str}{extra}  ({dt:.0f}s, n={sets_out[set_name]['n_clips']})")

    tag = Path(model_path.rstrip("/")).name or "model"
    return {
        "model": model_path,
        "model_tag": tag,
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "sets": sets_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="path to eval YAML")
    args = parser.parse_args()

    cfg = _load_eval_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    # `model` may be a single path or `models` a list (forgetting comparison).
    model_paths = cfg.get("models") or [cfg["model"]]
    output_dir = cfg.get("output_dir", "results/eval")

    for mp in model_paths:
        report = _evaluate_one_model(mp, cfg, device)
        jp, md = _write_report(report, output_dir)
        print(f"[eval] wrote {jp}")
        print(f"[eval] wrote {md}")


if __name__ == "__main__":
    main()
