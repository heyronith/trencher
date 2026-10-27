# Phase 3 — Rich-feature look-ahead leak audit

`q85_test_accessed: false` · branch `review/phase3-leak-v1` · CPU only

## Summary

**Yes — microstructure (and ownership via the same signature targets) has look-ahead leakage.** `n_sel_before_T` matches frozen `micro_n` at rate 1.0000 on rich val. Single-feature AUROC of `micro_n` on explosion_60s is 0.1251; the leak canary `n_sel_before_T/n_sel_total` is 0.1033.

Of the LightGBM arm-A → arm-B gain on explosion_60s (0.8348 → 0.9168, Δ=0.0821):
- Dropping all microstructure columns → ROC **0.8811** (Δ vs B -0.0357).
- Replacing `micro_n` with leak-free `n_trades_before_T` → ROC **0.8860**.
- Adding the leak canary feature → ROC **0.9208**.
- Arm B with only CLEAN rich groups (graph + wallet_prior) → ROC **0.8407** (vs A 0.8348; Δ=0.0059).

**Bottom line:** Most of the 0.835→0.917-style rich gain is microstructure-driven and contaminated by selection leak; CLEAN rich groups alone retain only a small lift over arm A.

## 1. Direct leak tests

### micro_n vs n_sel_before_T match

- train match rate: 1.000000
- val match rate: 1.000000

### Boolean prevalence (val)

| flag | fraction |
|---|---:|
| largest_after_T | 0.2610 |
| max_curve_after_T | 0.2166 |
| last_after_T | 0.4170 |

### Single-feature AUROC (rich val, eligible)

| feature | explosion_60s ROC | explosion_300s ROC |
|---|---:|---:|
| micro_n | 0.1251 | 0.1301 |
| frac_sel_before | 0.1033 | 0.1067 |
| largest_after_T | 0.8254 | 0.8249 |
| max_curve_after_T | 0.8094 | 0.8152 |
| last_after_T | 0.7434 | 0.7439 |
| n_trades_before_T | 0.3054 | 0.3303 |

### LightGBM variants (explosion_60s)

| variant | ROC-AUC | PR-AUC | ΔROC vs B | ΔROC vs A |
|---|---:|---:|---:|---:|
| full B | 0.9168 | 0.1143 | 0.0000 | 0.0821 |
| (a) −micro_n,mask_micro | 0.8924 | 0.1065 | -0.0244 | 0.0577 |
| (b) −all micro | 0.8811 | 0.1083 | -0.0357 | 0.0464 |
| (c) micro_n→n_trades_before_T | 0.8860 | 0.1065 | -0.0309 | 0.0512 |
| (d) +frac canary | 0.9208 | 0.1203 | 0.0039 | 0.0860 |
| arm A | 0.8348 | 0.0744 | — | 0 |

## 2. Group audit

### graph: **CLEAN**
- `src/trencher/data/world_model/sequences.py` :: `_build_graph_token_store` (815-848)
- Graph-token edges are derived from the full event sequence store for the mint, then observation refs slice by observation_time_t. Selection of which event rows exist is not driven by lifetime max/last stratified RPC targets. Causal filter at T is enforced in packs_for_observation.

### microstructure: **LEAK**
- `src/trencher/data/world_model/acquire/local_corpus.py` :: `select_stratified_trades_for_mint` (131-177)
- extract_signature_targets scans full-lifetime trades into a per-mint reservoir, then select_stratified_trades_for_mint picks first/last/median buy-sell/largest/max-curve/temporal grid. Only those signatures receive getTransaction RPC. At observation T, micro_n counts selected txs with time<T, but which txs were selected depends on post-T extremes.

### ownership: **LEAK**
- `src/trencher/data/world_model/ownership.py` :: `build_ownership_evidence / trade_wallets from signature_targets` (395-405)
- Ownership RPC token-balance recovery and trade-wallet universe are keyed off the same lifetime-stratified signature_targets, so which ownership edges exist inherits the microstructure selection leak. Per-observation slicing still uses t<T, but record membership is future-dependent.

### wallet_prior: **CLEAN**
- `src/trencher/baselines/rich_index.py` :: `_wallet_prior_pack / CausalRichSequenceIndex` (184-221, 248-252)
- Wallet set comes from graph-token edges available at T; each wallet's prior events are truncated to event_time < T. Forbidden lifetime edges_wallet_token aggregates are explicitly not used.

### Broad features

- **creator_history_n**: CLEAN — Same creator-history path as creator_prior_*; pre-T creator stats.
- **creator_prior_rank**: CLEAN — Creator priors are launch-time / history-before-mint fields in broad feature construction; not selected via post-T trade extremes.
- **daily_network_***: CLEAN — Daily aggregates: last fully completed UTC day strictly before T (never the incomplete calendar day containing T).
- **global_launches_1h_24h**: CLEAN — hi = searchsorted(arr_t, last_sec, side='right'); window back from last_completed_second(T)
- **other_broad_window_past_T**: CLEAN — Broad event/curve windows are causal sequence prefixes / last-N before T in baseline feature_cache; no additional lifetime-stratified RPC selection.
- **token_age_seconds**: CLEAN — T - detected_at; no future dependence.

## 3. CLEAN-only LGBM + RPC estimate

CLEAN groups: ['graph', 'wallet_prior']

| head | CLEAN-only ROC | arm A ROC | ΔROC (CLEAN−A) 95% CI |
|---|---:|---:|---|
| explosion_60s | 0.8407 | 0.8348 | 0.0059 [-0.0160, 0.0260] |
| explosion_300s | 0.8173 | 0.8242 | -0.0069 [-0.0261, 0.0114] |

Leak-free micro refetch (K=16, not executed): **54144** unique sigs; wall time 1.50h @10 rps, 0.30h @50, 0.08h @200.

## 4. Stage-C inference-time masks

World-model rich phase loads sequence_dataset_v2 modalities via make_frozen_dataloader / ProductionExampleBuilder (src/trencher/world_model/v4_loader.py, src/trencher/data/world_model/training/sequences_batch.py). Same frozen micro/ownership/graph/wallet_prior sequences as baselines.

| mask | ROC@60 | ΔROC@60 | ROC@300 | ΔROC@300 |
|---|---:|---:|---:|---:|
| baseline | 0.8500 | 0 | 0.8399 | 0 |
| no_graph | 0.8249 | -0.0251 | 0.8249 | -0.0150 |
| no_microstructure | 0.7451 | -0.1049 | 0.7230 | -0.1168 |
| no_ownership | 0.8292 | -0.0209 | 0.8232 | -0.0167 |
| no_wallet_prior | 0.8401 | -0.0099 | 0.8134 | -0.0265 |

## 5. `_safe_auc` before/after (Stage-C rich val)

| horizon | buggy | fixed | binary_metrics |
|---|---:|---:|---:|
| 300s | 0.5178 | 0.8750 | 0.8750 |
| 60s | 0.5353 | 0.8945 | 0.8945 |

## Could not complete

- n_trades_before_T uses sequence_events_v2, not a full pumpfun corpus rescan.
