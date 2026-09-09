"""Read-only audit of frozen loader epoch/cursor accounting.

Uses the same planner as production resume. Never accesses permanent TEST.
CPU-only; does not build production modalities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trencher.world_model.resumable_loader import audit_frozen_loader_cursor
from trencher.world_model.split_guard import PermanentTestAccessError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit frozen world-model loader epoch batch accounting"
    )
    parser.add_argument("--world-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--population", default="broad", choices=["broad", "rich"])
    parser.add_argument("--mode", default="final_train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--expected-batches",
        type=int,
        default=None,
        help="Fail closed if computed epoch batch count differs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON only",
    )
    args = parser.parse_args(argv)

    try:
        payload = audit_frozen_loader_cursor(
            world_root=args.world_root,
            split_root=args.split_root,
            mode=args.mode,
            population=args.population,
            seed=args.seed,
            num_workers=args.workers,
            batch_size=args.batch_size,
            shuffle=not args.no_shuffle,
        )
    except PermanentTestAccessError as exc:
        print(f"REFUSED permanent TEST access: {exc}", file=sys.stderr)
        return 2

    expected = int(payload["expected_dataloader_batches"])
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for i in range(int(payload["num_workers"])):
            print(f"worker_{i}_accepted_rows={payload[f'worker_{i}_accepted_rows']}")
            print(f"worker_{i}_batches={payload[f'worker_{i}_batches']}")
        print(f"accepted_rows_total={payload['accepted_rows_total']}")
        print(f"expected_dataloader_batches={expected}")

    if args.expected_batches is not None:
        if expected != int(args.expected_batches):
            print(
                f"BROAD_RAW_CURSOR_MATCHES_OBSERVED_EPOCH_FAIL "
                f"got={expected} expected={args.expected_batches}",
                file=sys.stderr,
            )
            return 1
        print("BROAD_RAW_CURSOR_MATCHES_OBSERVED_EPOCH_PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
