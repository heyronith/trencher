# Step-17 Scientific Ablation Report

Contract: `STEP17_SCIENTIFIC_EVALUATION_V1`
q85 TEST accessed: `False`

## Summary table

| Candidate | Explosion | Survival | Barriers | Returns | Δ vs Stage B | Compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline controls | see BASELINE_CORE_METRICS_V2 | partial / family-specific | family-specific | family-specific | n/a | frozen |
| Simple temporal control | PENDING | PENDING | PENDING | PENDING | PENDING | ~1 A100-hour (not launched) |
| Stage B | broad permanent-val only | broad permanent-val only | broad permanent-val only | broad permanent-val only | reference | complete |
| Stage C full | rich permanent-val complete | rich permanent-val complete | rich permanent-val complete | rich permanent-val complete | PENDING paired rich inference | complete |
| − graph | PENDING | PENDING | PENDING | PENDING | PENDING | ~1.5 A100-hours (not launched) |
| − wallet prior | PENDING | PENDING | PENDING | PENDING | PENDING | ~1.5 A100-hours (not launched) |
| − ownership | PENDING | PENDING | PENDING | PENDING | PENDING | ~1.5 A100-hours (not launched) |
| − microstructure | PENDING | PENDING | PENDING | PENDING | PENDING | ~1.5 A100-hours (not launched) |

## Explicit answers

### 1. Did Stage C improve over Stage B?

PENDING: Stage-B permanent-val is broad-population; Stage-C is rich. Generate Stage-B rich permanent_val predictions before answering.

### 2. Are those improvements statistically defensible?

PENDING paired bootstrap on shared rich observations

### 3. Which rich modality contributes the most?

PENDING scientific retraining ablations (diagnostics may prioritize)

### 4. Which rich modality contributes the least?

PENDING

### 5. Can any modality be removed without meaningful degradation?

PENDING

### 6. Does the full model materially outperform the frozen conventional controls?

Partial: overlapping explosion_60s/300s cells can be compared from frozen Baseline V2 vs Stage-C rich metrics; full suite pending protocol alignment notes.

### 7. Does it materially outperform the simple temporal deep control?

PENDING GRU control training

### 8. Is full Stage C justified for the final candidate?

PENDING — no q85 selection

## Notes

- No final candidate selection using q85.
- Inference-time modality masks are diagnostic only.
- Scientific ablations require retraining from the frozen Stage-B checkpoint.
