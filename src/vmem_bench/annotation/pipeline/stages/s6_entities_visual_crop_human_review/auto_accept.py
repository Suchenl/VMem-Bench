"""Automation-only S6 substitute used to smoke-test non-human pipeline stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def materialize_automation_review(
    *,
    proposals_path: Path,
    out_dir: Path,
    accept_all: bool = False,
) -> list[dict[str, Any]]:
    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    accepted = [
        proposal for proposal in proposals
        if proposal.get("crop_path") and (accept_all or proposal.get("accepted", False))
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "accepted_crops.json").write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "review_audit.json").write_text(
        json.dumps(
            {
                "mode": "automation_smoke_only",
                "human_review_skipped": True,
                "accepted_count": len(accepted),
                "proposal_count": len(proposals),
                "accept_all": accept_all,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--accept-all", action="store_true")
    args = parser.parse_args()
    materialize_automation_review(
        proposals_path=args.proposals, out_dir=args.out_dir, accept_all=args.accept_all
    )


if __name__ == "__main__":
    main()
