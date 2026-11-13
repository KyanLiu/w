"""
Evaluation metrics: word-error rate (WER), character-error rate (CER), medical-term error rate (MER).
Metrics are computed per clip, micro-averaged
"""
from __future__ import annotations
from dataclasses import dataclass
from preprocess import normalize
import jiwer


@dataclass
class MetricCounts:
    """Per-clip metrics"""
    errors: int # substitutions, deletions, insertion errors
    length: int # length of the reference length


# WER
def wer_counts(ref: str, hyp: str) -> MetricCounts:
    """Per-clip WER components, ref/hyp are the corresponding texts"""
    ref_n = normalize(ref)
    hyp_n = normalize(hyp)
    ref_words = ref_n.split()
    if not ref_words:
        return MetricCounts(errors=0, length=0)
    out = jiwer.process_words([ref_n], [hyp_n])
    errors = out.substitutions + out.deletions + out.insertions
    return MetricCounts(errors=errors, length=len(ref_words))


# CER
def cer_counts(ref: str, hyp: str) -> MetricCounts:
    """Per-clip CER components, whitespaces are included in eval, ref/hyp are the corresponding texts"""
    ref_n = normalize(ref)
    hyp_n = normalize(hyp)
    if not ref_n:
        return MetricCounts(errors=0, length=0)
    out = jiwer.process_characters([ref_n], [hyp_n])
    errors = out.substitutions + out.deletions + out.insertions
    return MetricCounts(errors=errors, length=len(ref_n))


def parse_medical_terms(medical_notations: str | None) -> list[str]:
    """Parse comma-separated medical-term list"""
    if not medical_notations:
        return []
    terms = []
    for raw in medical_notations.split(","):
        t = normalize(raw)
        if t:
            terms.append(t)
    return terms

# MER
def mer_counts(ref: str, hyp: str, medical_terms: list[str]) -> MetricCounts:
    """Per-clip MER components by token, handles misspell and incorrect ordering"""
    ref_n = normalize(ref)
    hyp_n = normalize(hyp)

    # Build the set of medical TOKENS (flatten multi-word terms to word tokens)
    medical_tokens: set[str] = set()
    for term in medical_terms:
        for tok in term.split():
            medical_tokens.add(tok)
    if not medical_tokens:
        return MetricCounts(errors=0, length=0)

    ref_words = ref_n.split()
    # Only medical tokens that actually appear in the reference count toward MER
    # Thie handles misspell errors
    ref_medical_positions = [i for i, w in enumerate(ref_words) if w in medical_tokens]
    denom = len(ref_medical_positions)
    if denom == 0:
        return MetricCounts(errors=0, length=0)

    # For each aligned chunk, find which reference positions did not matched correctly and count only the medical reference positions among them
    # This handles the incorrectly ordered or placed medical terms
    out = jiwer.process_words([ref_n], [hyp_n])
    bad_ref_positions: set[int] = set()
    # jiwer alignment chunks reference span [ref_start_idx, ref_end_idx)
    for chunk in out.alignments[0]:
        if chunk.type in ("substitute", "delete"):
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                bad_ref_positions.add(i)

    errors = sum(1 for i in ref_medical_positions if i in bad_ref_positions)
    return MetricCounts(errors=errors, length=denom)


# Aggregation
def aggregate(components: list[MetricCounts]) -> float:
    """Micro-averaged metric as a percentage (total summed and divided instead of averaging the ratios)"""
    total_len = sum(c.length for c in components)
    if total_len == 0:
        return 0.0
    total_err = sum(c.errors for c in components)
    return 100.0 * total_err / total_len
