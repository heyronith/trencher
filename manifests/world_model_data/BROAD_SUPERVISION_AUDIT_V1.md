# Broad Supervision Audit (`BROAD_SUPERVISION_AUDIT_V1`)

Generated at: `2026-09-05T04:14:08.859221+00:00`
Freeze: `world_model_freeze_v1`

## Population

```json
{'launch_population': 798430, 'event_curve_population': 622303, 'price_target_population': 596338, 'graduation_event_population': 798430, 'has_migration_evidence_metadata': 5697}
```

## Counts

- sequences: `{'event_rows': 26886634, 'curve_rows': 26886634, 'creator_rows': 55686561, 'excluded_system_program_rows': 'not_recorded', 'exact_duplicate_groups_removed': 0, 'signatures_with_heterogeneous_event_fingerprints': 'not_recorded', 'heterogeneous_same_signature_events_preserved': 'not_recorded', 'invalid_nonfinite_rows_excluded': 'not_recorded'}`
- sequence_quality: `{'excluded_system_program_rows': 'not_recorded', 'exact_duplicate_groups_removed': 0, 'signatures_with_heterogeneous_event_fingerprints': 'not_recorded', 'heterogeneous_same_signature_events_preserved': 'not_recorded', 'invalid_nonfinite_rows_excluded': 'not_recorded', 'source': 'broad_sequence_events_v1/manifest.json', 'note': 'Counts taken only from existing manifests; missing fields are marked not_recorded (resume finalize may omit exclusion tallies).'}`
- observations: `{'tokens_n': 622303, 'observations_n': 8449949, 'projected': {'tokens': 622303, 'fixed_age_obs_if_all_emitted': 6223030, 'event_count_obs_upper_bound': 6223030, 'total_upper_bound_before_dedupe': 12446060, 'note': 'Actual count is lower after coincident-time dedupe and when tokens have fewer than 512 trades.'}, 'actual': {'tokens': 622303, 'observations': 8449949, 'duplicate_time_collisions_removed': 0}}`
- targets: `{'observations_processed': 8449949, 'primary_explosion_events': 49836, 'mints_with_primary_explosion': 49836, 'explosion_definition_rows': 1789191}`
- disk MB: **32837.949**
- timings_sec: `{'population': 8.99655670800712, 'sequences': 147.5605618749978, 'observations': 312.90213095900253, 'targets': 5507.716352957999, 'audit': 614.5539309169981}`

## Explosion scale vs rich 4k

```json
{'rich_4k_primary_explosion_events': 222, 'broad_primary_explosion_events': 49836, 'broad_distinct_tokens_with_primary': 49836, 'scale_factor_vs_rich': 224.48648648648648, 'imminence_positives_by_horizon': {'10': 77297, '30': 109114, '60': 120638, '120': 128367, '300': 140230}, 'imminence_valid_tokens_by_horizon': {'10': 49836, '30': 49836, '60': 49836, '120': 49836, '300': 49836}, 'pre_explosion_observations': 159768, 'observations_scanned_for_sufficiency': 8449949, 'price_target_available_observations': 7124293}
```

## Graduation decision

- retain_conservative_v3: **True**
- tracking_expires proven end: **False**

## Rich vs broad capability matrix

- `event_history_sequences`: rich=`True` broad=`True` — Broad uses full Pump trades; rich uses atomic subset
- `curve_reserve_trajectory`: rich=`True` broad=`True` — Broad from trade reserves; rich also has execution_curve_states
- `creator_prior_launches`: rich=`graph_edges_partial` broad=`True` — Broad reconstructs from tokens timeline (no past_rugs leakage)
- `microstructure_rpc`: rich=`True` broad=`False` — RPC sampled on rich subset only
- `ownership_ledger_bq`: rich=`partial` broad=`False` — BQ transfers limited; broad ownership targets sparse/unavailable
- `wallet_prior_graph`: rich=`True` broad=`False` — Not in broad event/curve package
- `pit_model_state_json`: rich=`True` broad=`False` — Broad stores index + sequence refs only
- `price_return_barrier_targets`: rich=`True` broad=`True` — Same compute_* semantics
- `graduation_targets`: rich=`True` broad=`True` — Conservative v3 absence censoring retained
- `explosion_hazard`: rich=`marginal` broad=`scale_audit` — Broad increases independent explosion events

## Supervision sufficiency

- **returns_h300**: valid_tokens=122695, pos_tokens=63726, events=122695, rec=`sufficient_on_broad`
- **barrier_u100_l30_h3600**: valid_tokens=269007, pos_tokens=72786, events=72786, rec=`sufficient_on_broad`
- **survival_24h**: valid_tokens=592798, pos_tokens=4412, events=592798, rec=`sufficient_on_broad`
- **graduation_time_to_event**: valid_tokens=622303, pos_tokens=4150, events=4150, rec=`sufficient_on_broad`
- **graduated_within_h86400_binary**: valid_tokens=4150, pos_tokens=4018, events=4018, rec=`marginal_on_broad`
- **explosion_within_10s**: valid_tokens=49836, pos_tokens=43694, events=49836, rec=`sufficient_on_broad`
- **explosion_within_30s**: valid_tokens=49836, pos_tokens=47031, events=49836, rec=`sufficient_on_broad`
- **explosion_within_60s**: valid_tokens=49836, pos_tokens=47887, events=49836, rec=`sufficient_on_broad`
- **explosion_within_120s**: valid_tokens=49836, pos_tokens=48556, events=49836, rec=`sufficient_on_broad`
- **explosion_within_300s**: valid_tokens=49836, pos_tokens=49226, events=49836, rec=`sufficient_on_broad`
- **explosion_onset_primary**: valid_tokens=49836, pos_tokens=47887, events=49836, rec=`sufficient_on_broad`
- **holder_state_targets**: valid_tokens=6, pos_tokens=6, events=6, rec=`still_sparse_ownership`
