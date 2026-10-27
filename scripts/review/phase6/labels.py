"""Section 3: label sensitivity, 2× target, definition variants."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scripts.review.phase5.common import fit_xgb_deterministic
from scripts.review.phase5.matrices import load_head_bundle
from scripts.review.phase6.common import (
    Q85,
    SCRATCH,
    SEED,
    SPLIT,
    WORLD,
    metrics_block,
    write_json,
)


def _stream_obs_target_counts(split: str) -> dict[str, Any]:
    """Counts for reaches_plus_100pct and upside>=1 on train/val obs only."""
    assign_dir = SPLIT / "observation_assignments_broad"
    oid_split: dict[str, str] = {}
    for ap in sorted(assign_dir.glob("part-*.parquet")):
        t = pq.read_table(ap, columns=["observation_id", "base_split"])
        for oid, sp in zip(
            t["observation_id"].to_pylist(), t["base_split"].to_pylist(), strict=True
        ):
            if sp in ("train", "val"):
                oid_split[str(oid)] = sp

    cols = [
        "observation_id",
        "explosion_reaches_plus_100pct",
        "remaining_upside_to_explosion_peak",
        "time_to_explosion_peak_sec",
        "explosion_duration_onset_to_peak_sec",
        "explosion_within_300s",
        "explosion_within_300s_status",
        "time_to_graduation_sec",
        "graduated_at_or_before_t",
    ]
    root = WORLD / "broad" / "broad_targets_v1" / "observation_targets"
    n = {"train": 0, "val": 0}
    n_reach = {"train": 0, "val": 0}
    n_ups_ge1 = {"train": 0, "val": 0}
    n_ups_defined = {"train": 0, "val": 0}
    n_agree = {"train": 0, "val": 0}
    n_disagree = {"train": 0, "val": 0}
    n_peak_cap = {"train": 0, "val": 0}  # onset_to_peak ~1800
    n_both_defined = {"train": 0, "val": 0}

    for part in sorted(root.glob("part-*.parquet")):
        schema = set(pq.read_schema(part).names)
        use = [c for c in cols if c in schema]
        t = pq.read_table(part, columns=use)
        for row in t.to_pylist():
            oid = str(row["observation_id"])
            sp = oid_split.get(oid)
            if sp is None:
                continue
            n[sp] += 1
            reach = row.get("explosion_reaches_plus_100pct")
            ups = row.get("remaining_upside_to_explosion_peak")
            if reach is True:
                n_reach[sp] += 1
            if ups is not None:
                n_ups_defined[sp] += 1
                if float(ups) >= 1.0:
                    n_ups_ge1[sp] += 1
            if reach is not None and ups is not None:
                n_both_defined[sp] += 1
                r = bool(reach)
                u = float(ups) >= 1.0
                if r == u:
                    n_agree[sp] += 1
                else:
                    n_disagree[sp] += 1
            dur = row.get("explosion_duration_onset_to_peak_sec")
            if dur is not None and float(dur) >= 1799.0:
                n_peak_cap[sp] += 1

    out = {}
    for sp in ("train", "val"):
        out[sp] = {
            "n_rows": n[sp],
            "explosion_reaches_plus_100pct": {
                "n_true": n_reach[sp],
                "prevalence": n_reach[sp] / n[sp] if n[sp] else None,
            },
            "remaining_upside_ge_1": {
                "n_defined": n_ups_defined[sp],
                "n_true": n_ups_ge1[sp],
                "prevalence_among_defined": (
                    n_ups_ge1[sp] / n_ups_defined[sp] if n_ups_defined[sp] else None
                ),
                "prevalence_among_all_rows": n_ups_ge1[sp] / n[sp] if n[sp] else None,
            },
            "agreement_reach_vs_upside_ge1": {
                "n_both_defined": n_both_defined[sp],
                "n_agree": n_agree[sp],
                "n_disagree": n_disagree[sp],
                "agree_rate": (
                    n_agree[sp] / n_both_defined[sp] if n_both_defined[sp] else None
                ),
            },
            "n_onset_to_peak_ge_1799s": n_peak_cap[sp],
            "peak_search_cap_sec": 1800.0,
        }
    return {**Q85, "by_split": out, "note": "train+val only; q85+ TEST never read"}


def _load_alt_labels_for_val_train() -> dict[str, Any]:
    """Build arrays aligned to phase5 matrix oids for alt targets."""
    # Load matrix oids
    tr = load_head_bundle("explosion_60s", "train")
    va = load_head_bundle("explosion_60s", "val")

    def _index(oids: np.ndarray) -> dict[str, int]:
        return {str(o): i for i, o in enumerate(oids)}

    tr_i = _index(tr["observation_id"])
    va_i = _index(va["observation_id"])

    n_tr, n_va = len(tr["observation_id"]), len(va["observation_id"])
    fields = {
        "cons_onset": (np.full(n_tr, np.nan), np.full(n_va, np.nan)),
        "agg_onset": (np.full(n_tr, np.nan), np.full(n_va, np.nan)),
        "obs_t": (np.full(n_tr, np.nan), np.full(n_va, np.nan)),
        "within_60": (np.zeros(n_tr, dtype=np.int8), np.zeros(n_va, dtype=np.int8)),
        "within_300": (np.zeros(n_tr, dtype=np.int8), np.zeros(n_va, dtype=np.int8)),
        "elig_60": (np.zeros(n_tr, dtype=bool), np.zeros(n_va, dtype=bool)),
        "elig_300": (np.zeros(n_tr, dtype=bool), np.zeros(n_va, dtype=bool)),
        "upside": (np.full(n_tr, np.nan), np.full(n_va, np.nan)),
        "reach100": (np.full(n_tr, -1, dtype=np.int8), np.full(n_va, -1, dtype=np.int8)),
    }

    # Fill obs_t / within / elig from existing head bundles + side npz for val upside
    fields["obs_t"] = (tr["times"].copy(), va["times"].copy())
    fields["within_60"] = (
        tr["y"].astype(np.int8),
        va["y"].astype(np.int8),
    )
    fields["elig_60"] = (tr["elig"].copy(), va["elig"].copy())
    tr300 = load_head_bundle("explosion_300s", "train")
    va300 = load_head_bundle("explosion_300s", "val")
    fields["within_300"] = (tr300["y"].astype(np.int8), va300["y"].astype(np.int8))
    fields["elig_300"] = (tr300["elig"].copy(), va300["elig"].copy())

    # upside from training_cache
    for _split, idx_map, slot in (
        ("train", tr_i, 0),
        ("val", va_i, 1),
    ):
        cache = SPLIT / "training_cache_v1" / "broad" / "by_mint_partition"
        for part in sorted(cache.glob("*/joined.parquet")):
            t = pq.read_table(
                part,
                columns=["observation_id", "remaining_upside_to_explosion_peak"],
            )
            for oid, ups in zip(
                t["observation_id"].to_pylist(),
                t["remaining_upside_to_explosion_peak"].to_pylist(),
                strict=True,
            ):
                j = idx_map.get(str(oid))
                if j is None:
                    continue
                if ups is not None:
                    fields["upside"][slot][j] = float(ups)

    # conservative/aggressive onset from broad obs targets
    root = WORLD / "broad" / "broad_targets_v1" / "observation_targets"
    cols = [
        "observation_id",
        "explosion_conservative_onset_unix",
        "explosion_aggressive_onset_unix",
        "explosion_reaches_plus_100pct",
    ]
    for part in sorted(root.glob("part-*.parquet")):
        schema = set(pq.read_schema(part).names)
        use = [c for c in cols if c in schema]
        t = pq.read_table(part, columns=use)
        for row in t.to_pylist():
            oid = str(row["observation_id"])
            for idx_map, slot in ((tr_i, 0), (va_i, 1)):
                j = idx_map.get(oid)
                if j is None:
                    continue
                co = row.get("explosion_conservative_onset_unix")
                ao = row.get("explosion_aggressive_onset_unix")
                if co is not None:
                    fields["cons_onset"][slot][j] = float(co)
                if ao is not None:
                    fields["agg_onset"][slot][j] = float(ao)
                r = row.get("explosion_reaches_plus_100pct")
                if r is not None:
                    fields["reach100"][slot][j] = 1 if r else 0

    return {
        "train": tr,
        "val": va,
        "train_300": tr300,
        "val_300": va300,
        "fields": fields,
        **Q85,
    }


def _within_from_onset(
    obs_t: np.ndarray, onset: np.ndarray, horizon: float
) -> tuple[np.ndarray, np.ndarray]:
    """y, elig for explosion within horizon from definition onset."""
    y = np.zeros(len(obs_t), dtype=np.int8)
    elig = np.zeros(len(obs_t), dtype=bool)
    for i in range(len(obs_t)):
        T = obs_t[i]
        o = onset[i]
        if not np.isfinite(T):
            continue
        if not np.isfinite(o):
            # no explosion under this definition → negative if we can evaluate
            y[i] = 0
            elig[i] = True
            continue
        if o <= T:
            # already started → negative for imminence (match primary protocol)
            y[i] = 0
            elig[i] = True
            continue
        y[i] = 1 if o <= T + horizon else 0
        elig[i] = True
    return y, elig


def section_labels() -> dict[str, Any]:
    print("label data check (train/val obs targets)...", flush=True)
    data_check = _stream_obs_target_counts("unused")
    print("loading alt labels...", flush=True)
    pack = _load_alt_labels_for_val_train()
    fields = pack["fields"]
    train = pack["train"]
    val = pack["val"]
    tr300 = pack["train_300"]
    va300 = pack["val_300"]

    # Fixed iters from CORE xgb
    import json
    from pathlib import Path

    core = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "manifests/world_model_data/baseline_v2/BASELINE_CORE_METRICS_V2.json"
        ).read_text(encoding="utf-8")
    )
    fixed_60 = int(core["xgboost"]["heads"]["explosion_60s"]["fixed_iterations"])
    fixed_300 = int(core["xgboost"]["heads"]["explosion_300s"]["fixed_iterations"])

    targets: dict[str, Any] = {}
    scored_log = []

    def _train_eval(name: str, y_tr, elig_tr, y_va, elig_va, fixed: int, X_tr, X_va, times_tr):
        print(f"  train XGB target={name} n_pos_tr={int(y_tr[elig_tr].sum())}...", flush=True)
        preds = fit_xgb_deterministic(
            X_train=X_tr,
            y_train=y_tr.astype(np.float64),
            elig_train=elig_tr,
            times_train=times_tr,
            X_val=X_va,
            fixed_iters=fixed,
            seed=SEED,
            params_override={"nthread": 8, "seed": SEED},
            use_platt=True,
        )["calibrated"]
        yy = y_va[elig_va].astype(int)
        met = metrics_block(yy, preds[elig_va])
        scored_log.append({"target": name, "fixed_iterations": fixed, "permanent_val": True})
        return {
            "fixed_iterations": fixed,
            "n_train_elig": int(elig_tr.sum()),
            "n_train_pos": int(y_tr[elig_tr].sum()),
            "n_val_elig": int(elig_va.sum()),
            "n_pos": int(yy.sum()),
            "prevalence": float(yy.mean()) if yy.size else None,
            "metrics": met,
            **Q85,
        }

    X_tr = np.asarray(train["X"])
    X_va = np.asarray(val["X"])
    times_tr = train["times"]

    # conservative / aggressive 60s
    y_c_tr, e_c_tr = _within_from_onset(fields["obs_t"][0], fields["cons_onset"][0], 60.0)
    y_c_va, e_c_va = _within_from_onset(fields["obs_t"][1], fields["cons_onset"][1], 60.0)
    targets["explosion_60s_conservative"] = _train_eval(
        "explosion_60s_conservative",
        y_c_tr, e_c_tr, y_c_va, e_c_va, fixed_60, X_tr, X_va, times_tr,
    )

    y_a_tr, e_a_tr = _within_from_onset(fields["obs_t"][0], fields["agg_onset"][0], 60.0)
    y_a_va, e_a_va = _within_from_onset(fields["obs_t"][1], fields["agg_onset"][1], 60.0)
    targets["explosion_60s_aggressive"] = _train_eval(
        "explosion_60s_aggressive",
        y_a_tr, e_a_tr, y_a_va, e_a_va, fixed_60, X_tr, X_va, times_tr,
    )

    # 2× target: within 300s AND upside >= 1.0; unknown peak with onset in horizon → exclude
    def _twox(slot: int, within, elig_base):
        y = np.zeros(len(within), dtype=np.int8)
        elig = np.zeros(len(within), dtype=bool)
        ups = fields["upside"][slot]
        for i in range(len(within)):
            if not elig_base[i]:
                continue
            if within[i] == 1:
                if not np.isfinite(ups[i]):
                    # onset in horizon but peak unknown → excluded
                    continue
                if ups[i] >= 1.0:
                    y[i] = 1
                    elig[i] = True
                else:
                    y[i] = 0
                    elig[i] = True
            else:
                y[i] = 0
                elig[i] = True
        return y, elig

    y2_tr, e2_tr = _twox(0, fields["within_300"][0], fields["elig_300"][0])
    y2_va, e2_va = _twox(1, fields["within_300"][1], fields["elig_300"][1])
    targets["explosion_300s_and_upside_ge_1"] = _train_eval(
        "explosion_300s_and_upside_ge_1",
        y2_tr, e2_tr, y2_va, e2_va, fixed_300,
        np.asarray(tr300["X"]), np.asarray(va300["X"]), tr300["times"],
    )

    # Also primary 60/300 reference retrain for comparison (same protocol)
    targets["explosion_60s_primary_retrain"] = _train_eval(
        "explosion_60s_primary_retrain",
        train["y"], train["elig"], val["y"], val["elig"], fixed_60, X_tr, X_va, times_tr,
    )
    targets["explosion_300s_primary_retrain"] = _train_eval(
        "explosion_300s_primary_retrain",
        tr300["y"], tr300["elig"], va300["y"], va300["elig"], fixed_300,
        np.asarray(tr300["X"]), np.asarray(va300["X"]), tr300["times"],
    )

    result = {
        **Q85,
        "data_check": data_check,
        "targets": targets,
        "permanent_val_configs_scored": scored_log,
        "peak_cap_note": (
            "peak_search_sec=1800 in explosion detector; "
            "n_onset_to_peak_ge_1799s counts events hitting the search cap. "
            "Graduation/migration clipping is not separately flagged on remaining_upside; "
            "peak may end at migration when bonding-curve path ends."
        ),
    }
    write_json(SCRATCH / "cache" / "section_labels.json", result)
    return result
