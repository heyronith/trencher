"""Preregistered TEST scorer — intentionally unimplemented until protocol is frozen.

This module must not read q85+ TEST rows until a later explicit phase.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score q85 TEST once under TEST_PROTOCOL_V1 (not enabled yet)."
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preregistration-sha", required=True)
    parser.add_argument("--out", required=True)
    parser.parse_args(argv)
    print(
        "REFUSED: score_test_once is preregistered but not executable in Phase 6. "
        "No q85+ TEST rows were read.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
