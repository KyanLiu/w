"""Grouped bootstrap confidence intervals for metrics"""
from __future__ import annotations

import random
from dataclasses import dataclass
from eval.metrics import MetricCounts


@dataclass
class CIResult:
    point: float        # metric on the full set (percentage)
    lo: float           # 2.5th percentile (95% CI lower)
    hi: float           # 97.5th percentile (95% CI upper)
    resamples: int

    def __str__(self) -> str:
        return f"{self.point:.2f}% (95% CI {self.lo:.2f}-{self.hi:.2f}, n_boot={self.resamples})"


def _micro(components: list[MetricCounts]) -> float:
    total_len = sum(c.length for c in components)
    if total_len == 0:
        return 0.0
    return 100.0 * sum(c.errors for c in components) / total_len


def bootstrap_ci(
    components: list[MetricCounts],
    groups: list[str],
    resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> CIResult:
    """
    Grouped bootstrap confidence interval for a micro-averaged metric
    Given components (MetricCounts) and grouping
    Returns confidence interval result with the full-set point estimate and percentile confidence interval
    """
    if len(components) != len(groups):
        raise ValueError("components and groups must be the same length")

    point = _micro(components)
    if not components:
        return CIResult(point=0.0, lo=0.0, hi=0.0, resamples=resamples)

    # Bucket clip-components by group so a drawn group contributes all its clips.
    by_group: dict[str, list[MetricCounts]] = {}
    for comp, g in zip(components, groups):
        by_group.setdefault(g, []).append(comp)
    group_keys = list(by_group.keys())
    n_groups = len(group_keys)

    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(resamples):
        # draw n_groups recordings with replacement
        drawn: list[MetricCounts] = []
        for _ in range(n_groups):
            g = group_keys[rng.randrange(n_groups)]
            drawn.extend(by_group[g])
        stats.append(_micro(drawn))

    stats.sort()
    lo_idx = int((alpha / 2) * resamples)
    hi_idx = int((1 - alpha / 2) * resamples) - 1
    lo_idx = max(0, min(lo_idx, resamples - 1))
    hi_idx = max(0, min(hi_idx, resamples - 1))
    return CIResult(point=point, lo=stats[lo_idx], hi=stats[hi_idx], resamples=resamples)
