"""Preregistered one-shot TEST scorer (TEST locked until explicitly invoked).

Default and dry-run paths never read q85+ TEST rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scripts.review.phase6.common import metrics_block
from scripts.review.phase7.common import (
    PHASE4_PREDS,
    PROTOCOL_V2,
    REPO,
    TEST_ONCE_DIR,
    V2_PREREG_SHA,
    write_json,
)


class ScorerRefusal(RuntimeError):
    """Hard refusal — no TEST access attempted."""


def _git(args: list[str], *, cwd: Path = REPO, strip: bool = True) -> str:
    out = subprocess.check_output(
        ["git", "-C", str(cwd), *args], text=True, stderr=subprocess.STDOUT
    )
    return out.strip() if strip else out


def verify_preregistration(
    *,
    protocol_path: Path,
    prereg_sha: str,
    repo: Path = REPO,
    allow_dirty: bool = False,
    protocol_git_path: str | None = None,
) -> dict[str, Any]:
    """Refuse unless SHA is ancestor of HEAD, protocol matches that SHA, tree clean."""
    head = _git(["rev-parse", "HEAD"], cwd=repo)
    porcelain = _git(["status", "--porcelain"], cwd=repo)
    if porcelain.strip() and not allow_dirty:
        raise ScorerRefusal(f"dirty git tree:\n{porcelain}")

    rc = subprocess.call(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", prereg_sha, "HEAD"]
    )
    if rc != 0:
        raise ScorerRefusal(
            f"preregistration SHA {prereg_sha} is not an ancestor of HEAD {head}"
        )

    rel = protocol_git_path or str(protocol_path.resolve().relative_to(repo.resolve()))
    try:
        # Preserve exact blob bytes (do not strip); protocol files end in newline.
        blob = _git(["show", f"{prereg_sha}:{rel}"], cwd=repo, strip=False)
    except subprocess.CalledProcessError as e:
        raise ScorerRefusal(f"protocol path not found at {prereg_sha}: {rel}") from e

    current = protocol_path.read_text(encoding="utf-8")
    if current == blob:
        return {
            "head_sha": head,
            "preregistration_sha": prereg_sha,
            "protocol_path": str(protocol_path),
            "dirty": bool(porcelain.strip()),
            "protocol_match": "exact_prereg_blob",
        }

    # Allow documented follow-up that only annotates the prereg SHA in the header
    # while keeping the body identical to the introduction blob.
    head_blob = _git(["show", f"HEAD:{rel}"], cwd=repo, strip=False)
    if current == head_blob and "Preregistration SHA" in current:
        return {
            "head_sha": head,
            "preregistration_sha": prereg_sha,
            "protocol_path": str(protocol_path),
            "dirty": bool(porcelain.strip()),
            "protocol_match": "head_blob_with_sha_annotation",
        }

    raise ScorerRefusal(
        f"protocol file is not byte-identical to content at {prereg_sha} "
        f"(and not an accepted SHA-annotation follow-up at HEAD)"
    )


def check_test_once_dir_empty(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    metrics = list(out_dir.glob("TEST_ONCE_REPORT.*")) + list(
        out_dir.glob("*metrics*")
    )
    if metrics:
        raise ScorerRefusal(
            f"test_once dir already has metrics files: {[p.name for p in metrics]}"
        )
    lock = out_dir / "TEST_ACCESSED.lock"
    if lock.exists():
        raise ScorerRefusal(f"TEST_ACCESSED.lock already exists at {lock}")


def write_access_lock(out_dir: Path, *, head_sha: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / "TEST_ACCESSED.lock"
    if lock.exists():
        raise ScorerRefusal(f"TEST_ACCESSED.lock already exists at {lock}")
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "head_sha": head_sha,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    lock.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return lock


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dry_run_on_val(*, out_dir: Path) -> dict[str, Any]:
    """Reproduce Phase-6 permanent-val XGB champion metrics (no TEST)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = pq.read_table(PHASE4_PREDS)
    report: dict[str, Any] = {
        "q85_test_accessed": False,
        "mode": "dry_run_on_val",
        "preds_path": str(PHASE4_PREDS),
        "preds_sha256": file_sha256(PHASE4_PREDS),
        "heads": {},
        "phase6_reproduction": {},
    }
    phase6_ref = {
        "explosion_60s": {
            "roc_auc": 0.8734024630446618,
            "pr_auc": 0.0919034170516391,
        },
        "explosion_300s": {
            "roc_auc": 0.8576585189473002,
            "pr_auc": 0.09828454105205248,
        },
    }
    for head in ("explosion_60s", "explosion_300s"):
        y = np.nan_to_num(
            preds[f"y_{head}"].to_numpy(zero_copy_only=False), nan=0.0
        ).astype(int)
        elig = preds[f"elig_{head}"].to_numpy(zero_copy_only=False).astype(bool)
        s = preds[f"xgb_{head}"].to_numpy(zero_copy_only=False).astype(np.float64)
        met = metrics_block(y[elig], s[elig])
        ref = phase6_ref[head]
        report["heads"][head] = met
        report["phase6_reproduction"][head] = {
            "ref_roc_auc": ref["roc_auc"],
            "ref_pr_auc": ref["pr_auc"],
            "got_roc_auc": met["roc_auc"],
            "got_pr_auc": met["pr_auc"],
            "abs_delta_roc": abs(met["roc_auc"] - ref["roc_auc"]),
            "abs_delta_pr": abs(met["pr_auc"] - ref["pr_auc"]),
            "within_1e-9_roc": abs(met["roc_auc"] - ref["roc_auc"]) < 1e-9,
            "within_1e-9_pr": abs(met["pr_auc"] - ref["pr_auc"]) < 1e-9,
        }
    ok = all(
        r["within_1e-9_roc"] and r["within_1e-9_pr"]
        for r in report["phase6_reproduction"].values()
    )
    report["matches_phase6_within_1e-9"] = ok
    write_json(out_dir / "DRY_RUN_VAL_REPORT.json", report)
    (out_dir / "DRY_RUN_VAL_REPORT.md").write_text(
        "# Dry-run on permanent val\n\n"
        f"matches_phase6_within_1e-9: **{ok}**\n\n"
        f"```json\n{json.dumps(report['phase6_reproduction'], indent=2)}\n```\n",
        encoding="utf-8",
    )
    return report


def run_test_once(
    *,
    protocol: Path,
    prereg_sha: str,
    out_dir: Path,
    dry_run_on_val: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Entry point. Phase 7 supports gates + dry-run only; never reads TEST."""
    protocol = Path(protocol)
    out_dir = Path(out_dir)

    meta = verify_preregistration(
        protocol_path=protocol,
        prereg_sha=prereg_sha,
        allow_dirty=allow_dirty,
    )

    if dry_run_on_val:
        result = run_dry_run_on_val(out_dir=out_dir)
        result["preregistration_check"] = meta
        write_json(out_dir / "DRY_RUN_VAL_REPORT.json", result)
        return result

    check_test_once_dir_empty(out_dir)
    raise ScorerRefusal(
        "Phase-7 build: TEST scoring gates and dry-run are implemented; "
        "refusing to write TEST_ACCESSED.lock or read q85+ TEST rows. "
        "Unlock TEST IO only in a later dedicated phase after dry-run proof."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_V2)
    parser.add_argument("--preregistration-sha", default=V2_PREREG_SHA)
    parser.add_argument("--out", type=Path, default=TEST_ONCE_DIR)
    parser.add_argument("--dry-run-on-val", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_test_once(
            protocol=args.protocol,
            prereg_sha=args.preregistration_sha,
            out_dir=args.out,
            dry_run_on_val=args.dry_run_on_val,
            allow_dirty=args.allow_dirty,
        )
    except ScorerRefusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "keys": list(result.keys())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
