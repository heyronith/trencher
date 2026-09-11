"""Step-17 scientific report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trencher.data.world_model.io import write_json
from trencher.world_model.evaluation.modes import provenance_q85_flag


def build_summary_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def render_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Step-17 Scientific Ablation Report",
        "",
        f"Contract: `{payload.get('contract')}`",
        f"q85 TEST accessed: `{payload.get('q85_test_accessed')}`",
        "",
        "## Summary table",
        "",
        "| Candidate | Explosion | Survival | Barriers | Returns | Δ vs Stage B | Compute |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("summary_table") or []:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                row.get("candidate", ""),
                row.get("explosion", "..."),
                row.get("survival", "..."),
                row.get("barriers", "..."),
                row.get("returns", "..."),
                row.get("delta_vs_stage_b", "..."),
                row.get("compute", "..."),
            )
        )
    lines.extend(["", "## Explicit answers", ""])
    answers = payload.get("answers") or {}
    questions = [
        ("stage_c_improved_over_stage_b", "1. Did Stage C improve over Stage B?"),
        (
            "improvements_statistically_defensible",
            "2. Are those improvements statistically defensible?",
        ),
        ("rich_modality_most", "3. Which rich modality contributes the most?"),
        ("rich_modality_least", "4. Which rich modality contributes the least?"),
        (
            "removable_without_meaningful_degradation",
            "5. Can any modality be removed without meaningful degradation?",
        ),
        (
            "outperforms_conventional_controls",
            "6. Does the full model materially outperform the frozen "
            "conventional controls?",
        ),
        (
            "outperforms_simple_temporal_control",
            "7. Does it materially outperform the simple temporal deep control?",
        ),
        (
            "full_stage_c_justified",
            "8. Is full Stage C justified for the final candidate?",
        ),
    ]
    for key, title in questions:
        lines.append(f"### {title}")
        lines.append("")
        default = "PENDING — awaiting GPU ablation / paired rich inference runs"
        lines.append(str(answers.get(key, default)))
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- No final candidate selection using q85.",
            "- Inference-time modality masks are diagnostic only.",
            "- Scientific ablations require retraining from the frozen Stage-B checkpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(out_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.update(provenance_q85_flag(accessed=False))
    json_path = out_dir / "STEP17_SCIENTIFIC_ABLATION_REPORT.json"
    md_path = out_dir / "STEP17_SCIENTIFIC_ABLATION_REPORT.md"
    write_json(json_path, payload)
    md_path.write_text(render_markdown_report(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def placeholder_report_payload() -> dict[str, Any]:
    """Skeleton report used until GPU ablations complete."""
    return {
        "contract": "STEP17_SCIENTIFIC_EVALUATION_V1",
        "status": "preflight_complete_awaiting_gpu_runs",
        "q85_test_accessed": False,
        "summary_table": [
            {
                "candidate": "Baseline controls",
                "explosion": "see BASELINE_CORE_METRICS_V2",
                "survival": "partial / family-specific",
                "barriers": "family-specific",
                "returns": "family-specific",
                "delta_vs_stage_b": "n/a",
                "compute": "frozen",
            },
            {
                "candidate": "Simple temporal control",
                "explosion": "PENDING",
                "survival": "PENDING",
                "barriers": "PENDING",
                "returns": "PENDING",
                "delta_vs_stage_b": "PENDING",
                "compute": "~1 A100-hour (not launched)",
            },
            {
                "candidate": "Stage B",
                "explosion": "broad permanent-val only",
                "survival": "broad permanent-val only",
                "barriers": "broad permanent-val only",
                "returns": "broad permanent-val only",
                "delta_vs_stage_b": "reference",
                "compute": "complete",
            },
            {
                "candidate": "Stage C full",
                "explosion": "rich permanent-val complete",
                "survival": "rich permanent-val complete",
                "barriers": "rich permanent-val complete",
                "returns": "rich permanent-val complete",
                "delta_vs_stage_b": "PENDING paired rich inference",
                "compute": "complete",
            },
            {
                "candidate": "− graph",
                "explosion": "PENDING",
                "survival": "PENDING",
                "barriers": "PENDING",
                "returns": "PENDING",
                "delta_vs_stage_b": "PENDING",
                "compute": "~1.5 A100-hours (not launched)",
            },
            {
                "candidate": "− wallet prior",
                "explosion": "PENDING",
                "survival": "PENDING",
                "barriers": "PENDING",
                "returns": "PENDING",
                "delta_vs_stage_b": "PENDING",
                "compute": "~1.5 A100-hours (not launched)",
            },
            {
                "candidate": "− ownership",
                "explosion": "PENDING",
                "survival": "PENDING",
                "barriers": "PENDING",
                "returns": "PENDING",
                "delta_vs_stage_b": "PENDING",
                "compute": "~1.5 A100-hours (not launched)",
            },
            {
                "candidate": "− microstructure",
                "explosion": "PENDING",
                "survival": "PENDING",
                "barriers": "PENDING",
                "returns": "PENDING",
                "delta_vs_stage_b": "PENDING",
                "compute": "~1.5 A100-hours (not launched)",
            },
        ],
        "answers": {
            "stage_c_improved_over_stage_b": (
                "PENDING: Stage-B permanent-val is broad-population; Stage-C is rich. "
                "Generate Stage-B rich permanent_val predictions before answering."
            ),
            "improvements_statistically_defensible": (
                "PENDING paired bootstrap on shared rich observations"
            ),
            "rich_modality_most": (
                "PENDING scientific retraining ablations "
                "(diagnostics may prioritize)"
            ),
            "rich_modality_least": "PENDING",
            "removable_without_meaningful_degradation": "PENDING",
            "outperforms_conventional_controls": (
                "Partial: overlapping explosion_60s/300s cells can be compared from frozen "
                "Baseline V2 vs Stage-C rich metrics; full suite pending protocol alignment notes."
            ),
            "outperforms_simple_temporal_control": "PENDING GRU control training",
            "full_stage_c_justified": "PENDING — no q85 selection",
        },
    }


def dump_placeholder(path: Path) -> None:
    write_reports(path, placeholder_report_payload())
    # also keep JSON pretty for debugging
    _ = json.dumps
