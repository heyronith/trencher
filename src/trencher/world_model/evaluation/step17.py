"""CLI entry: ``python -m trencher.world_model.evaluation.step17 ...``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trencher.world_model.evaluation.ablations import ablation_launch_commands
from trencher.world_model.evaluation.baselines import (
    compare_world_model_to_baselines,
    load_baseline_core,
)
from trencher.world_model.evaluation.gru_control import (
    EventGRUControl,
    count_control_parameters,
    gru_control_plan,
)
from trencher.world_model.evaluation.inference import run_permanent_val_inference
from trencher.world_model.evaluation.lineage import stage_b_lineage, stage_c_lineage
from trencher.world_model.evaluation.masking import diagnostic_configs
from trencher.world_model.evaluation.modes import assert_step17_mode
from trencher.world_model.evaluation.preflight import write_preflight
from trencher.world_model.evaluation.report import placeholder_report_payload, write_reports
from trencher.world_model.split_guard import PermanentTestAccessError


def _cmd_preflight(args: argparse.Namespace) -> int:
    payload = write_preflight(Path(args.out), repo_root=Path(args.repo_root))
    print(json.dumps({"wrote": args.out, "status": payload["status"]}, indent=2))
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    summary = run_permanent_val_inference(
        checkpoint=Path(args.checkpoint),
        config=Path(args.config),
        world_root=Path(args.world_root),
        split_root=Path(args.split_root),
        out_dir=Path(args.out_dir),
        population=args.population,
        mode=args.mode,
        phase=args.phase,
        device=args.device,
        git_sha=args.git_sha,
        omit_modalities=[x for x in (args.omit or "").split(",") if x],
        inference_mask_omit=[x for x in (args.inference_mask or "").split(",") if x],
        run_id=args.run_id,
        max_batches=args.max_batches,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_lineage(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {"stage_b": stage_b_lineage(), "stage_c": stage_c_lineage()},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_ablation_plan(_: argparse.Namespace) -> int:
    print(json.dumps(ablation_launch_commands(), indent=2, sort_keys=True))
    return 0


def _cmd_diagnose_plan(_: argparse.Namespace) -> int:
    print(json.dumps(diagnostic_configs(), indent=2, sort_keys=True))
    return 0


def _cmd_gru_plan(_: argparse.Namespace) -> int:
    print(json.dumps(gru_control_plan(), indent=2, sort_keys=True))
    return 0


def _cmd_gru_smoke(_: argparse.Namespace) -> int:
    """CPU smoke constructing the GRU control (no training)."""
    import torch

    from trencher.world_model.config import tiny_smoke_config

    cfg = tiny_smoke_config()
    model = EventGRUControl(cfg)
    b = 2
    t = 8
    batch = {
        "event_x": torch.randn(b, t, cfg.event.n_channels),
        "event_mask": torch.ones(b, t),
        "curve_x": torch.randn(b, 4, cfg.curve.n_channels),
        "curve_mask": torch.ones(b, 4),
        "global_buckets": torch.randn(b, 4, cfg.global_temporal.n_bucket_channels),
        "global_scalars": torch.randn(b, cfg.global_temporal.n_scalars),
        "global_mask": torch.ones(b, 4),
    }
    out = model(batch, phase="rich")
    assert "explosion_logits" in out
    print(
        json.dumps(
            {
                "parameters": count_control_parameters(cfg),
                "explosion_shape": list(out["explosion_logits"].shape),
                "q85_test_accessed": False,
            },
            indent=2,
        )
    )
    return 0


def _cmd_refuse_demo(args: argparse.Namespace) -> int:
    try:
        assert_step17_mode(args.mode)
    except PermanentTestAccessError as exc:
        print(json.dumps({"refused": True, "mode": args.mode, "error": str(exc)}))
        return 0
    print(json.dumps({"refused": False, "mode": args.mode}))
    return 1


def _cmd_report_placeholder(args: argparse.Namespace) -> int:
    paths = write_reports(Path(args.out_dir), placeholder_report_payload())
    print(json.dumps(paths, indent=2))
    return 0


def _cmd_baseline_compare(args: argparse.Namespace) -> int:
    wm = json.loads(Path(args.wm_metrics).read_text(encoding="utf-8")) if args.wm_metrics else {}
    # Accept either Stage-C native schema or summarize_prediction schema.
    if "explosion" not in wm and "explosion_metrics_by_horizon_sec" in wm:
        wm = {
            "explosion": {
                k: {"auroc": v.get("auroc"), "support": v.get("support")}
                for k, v in wm["explosion_metrics_by_horizon_sec"].items()
            }
        }
    if args.baseline_core:
        core = load_baseline_core(Path(args.baseline_core))
    else:
        core = load_baseline_core()
    print(json.dumps(compare_world_model_to_baselines(wm, baseline_core=core), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trencher.world_model.evaluation.step17",
        description="Step-17 scientific evaluation / ablation framework",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preflight", help="Write STEP17_PREFLIGHT.json")
    sp.add_argument("--out", default="manifests/world_model_data/evaluation/STEP17_PREFLIGHT.json")
    sp.add_argument("--repo-root", default=".")
    sp.set_defaults(func=_cmd_preflight)

    sp = sub.add_parser("infer", help="Permanent-val inference (GPU)")
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--config", default="configs/world_model/world_model_candidate_v2.yaml")
    sp.add_argument("--world-root", required=True)
    sp.add_argument("--split-root", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--population", default="rich")
    sp.add_argument("--mode", default="permanent_val_eval")
    sp.add_argument("--phase", default="rich")
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--git-sha", default="unknown")
    sp.add_argument("--run-id", default="permanent_val")
    sp.add_argument("--omit", default="", help="Comma-separated train-time modality omits")
    sp.add_argument(
        "--inference-mask",
        default="",
        help="Comma-separated inference-time availability masks (diagnostic)",
    )
    sp.add_argument("--max-batches", type=int, default=None)
    sp.add_argument("--expected-checkpoint-sha256", default=None)
    sp.set_defaults(func=_cmd_infer)

    sub.add_parser("lineage", help="Print verified Stage-B/C lineage").set_defaults(
        func=_cmd_lineage
    )
    sub.add_parser("ablation-plan", help="Print scientific ablation launch plan").set_defaults(
        func=_cmd_ablation_plan
    )
    sub.add_parser("diagnose-plan", help="Print inference-mask diagnostic configs").set_defaults(
        func=_cmd_diagnose_plan
    )
    sub.add_parser("gru-plan", help="Print GRU control plan (do not auto-launch)").set_defaults(
        func=_cmd_gru_plan
    )
    sub.add_parser("gru-smoke", help="CPU construct GRU control").set_defaults(func=_cmd_gru_smoke)

    sp = sub.add_parser("refuse-demo", help="Show TEST alias refusal")
    sp.add_argument("--mode", required=True)
    sp.set_defaults(func=_cmd_refuse_demo)

    sp = sub.add_parser("report-placeholder", help="Write pending scientific report skeleton")
    sp.add_argument("--out-dir", default="manifests/world_model_data/evaluation")
    sp.set_defaults(func=_cmd_report_placeholder)

    sp = sub.add_parser("baseline-compare", help="Compare WM metrics to Baseline V2")
    sp.add_argument("--wm-metrics", default=None)
    sp.add_argument("--baseline-core", default=None)
    sp.set_defaults(func=_cmd_baseline_compare)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
