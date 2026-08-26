"""Deterministic stratified sampling plan for S4 human audit."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def build_sample(
    reviews: list[dict[str, Any]],
    *,
    minimum: int = 3,
    rate: float = 0.05,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Queue every BLOCK and proportionally sample WARN/PASS.

    RETRYABLE_ERROR belongs to automatic endpoint retry and is intentionally
    excluded from the human decision surface.
    """
    if rate < 0:
        raise ValueError("rate must be non-negative")
    eligible = [
        item for item in reviews
        if str(item.get("verdict") or "") != "RETRYABLE_ERROR"
    ]
    if not eligible:
        return []

    blocks = [item for item in eligible if str(item.get("verdict") or "") == "BLOCK"]
    warns = [item for item in eligible if str(item.get("verdict") or "") == "WARN"]
    passes = [
        item
        for item in eligible
        if str(item.get("verdict") or "PASS") not in {"BLOCK", "WARN"}
    ]
    sample_count = min(
        len(warns) + len(passes),
        max(minimum, round(len(eligible) * rate)),
    )
    rng = random.Random(seed)
    rng.shuffle(warns)
    rng.shuffle(passes)
    chosen = list(blocks)
    if not warns:
        warn_count = 0
        pass_count = sample_count
    elif not passes:
        warn_count = sample_count
        pass_count = 0
    elif sample_count == 1:
        # Preserve deterministic proportional behavior when one card cannot
        # represent both strata.
        warn_count = int(len(warns) >= len(passes))
        pass_count = 1 - warn_count
    else:
        pass_count = max(
            1,
            round(sample_count * len(passes) / (len(warns) + len(passes))),
        )
        pass_count = min(len(passes), pass_count)
        warn_count = min(len(warns), sample_count - pass_count)
        # Both strata exist and the budget permits both. Fill a depleted
        # stratum from the other without starving PASS audit coverage.
        if warn_count == 0:
            warn_count = 1
            pass_count = min(len(passes), sample_count - warn_count)
        remaining = sample_count - warn_count - pass_count
        if remaining:
            extra_warn = min(len(warns) - warn_count, remaining)
            warn_count += extra_warn
            pass_count += min(len(passes) - pass_count, remaining - extra_warn)
    chosen.extend(warns[:warn_count])
    chosen.extend(passes[:pass_count])
    return sorted(chosen, key=lambda review: str(review.get("segment_id", "")))


def materialize_sample(
    *,
    reviews_path: Path,
    output_path: Path,
    minimum: int = 3,
    rate: float = 0.05,
    seed: int = 0,
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for line in reviews_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            reviews.append(item)
    queue = build_sample(reviews, minimum=minimum, rate=rate, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(queue, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize typed S4 review queue")
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=3)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    queue = materialize_sample(
        reviews_path=args.reviews,
        output_path=args.output,
        minimum=args.minimum,
        rate=args.rate,
        seed=args.seed,
    )
    print(json.dumps({"n_queue": len(queue)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
