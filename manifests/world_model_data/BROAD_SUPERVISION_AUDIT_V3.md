# Broad Supervision Audit (`BROAD_SUPERVISION_AUDIT_V3`)

Generated at: `2026-09-05T16:33:14.215540+00:00`
Freeze: `world_model_freeze_v1`
Supersedes: `BROAD_SUPERVISION_AUDIT_V2`

## Detector row breakdown

- `{'event_curve_mints': 622303, 'price_capable_mints': 596338, 'mints_with_stored_primary_detector_row': 596397, 'mints_without_detector_row': 25906, 'primary_exists_true': 49836, 'primary_exists_false': 546561, 'price_capable_without_primary': 0, 'event_curve_without_primary': 25906}`

## Explosion imminence by horizon (v3)

- **10s**: valid=8104230, neg=8026933, pos=77297, censored=5509, unavailable=340210, prev_obs=0.009537858624446739, prev_tok=0.07326327932568406
- **30s**: valid=8104190, neg=7995076, pos=109114, censored=5549, unavailable=340210, prev_obs=0.013463899538386934, prev_tok=0.07885854556612458
- **60s**: valid=8104131, neg=7983493, pos=120638, censored=5608, unavailable=340210, prev_obs=0.014885988392833236, prev_tok=0.08029383112255763
- **120s**: valid=8103930, neg=7975563, pos=128367, censored=5809, unavailable=340210, prev_obs=0.01584009239961352, prev_tok=0.0814166592609857
- **300s**: valid=8103129, neg=7962899, pos=140230, censored=6610, unavailable=340210, prev_obs=0.017305660566430573, prev_tok=0.08254894562528822

- invariants: `{'positive_subseteq_valid_v3': True, 'primary_event_count_unchanged': True, 'no_detector_row_never_valid_explosion_negative': True}`
- invariant_scan: `{'horizon_sec': 60, 'observations_scanned': 8449949, 'observations_without_primary_row': 340210, 'valid_negative_violations': 0, 'passed': True}`

## Population / counts

- population: `{'launch_population': 798430, 'event_curve_population': 622303, 'price_target_population': 596338, 'graduation_event_population': 798430, 'has_migration_evidence_metadata': 5697}`
- sequences: `{'event_rows': 26886634, 'curve_rows': 26886634, 'creator_rows': 55686561, 'excluded_system_program_rows': 'not_recorded', 'exact_duplicate_groups_removed': 0, 'signatures_with_heterogeneous_event_fingerprints': 'not_recorded', 'invalid_nonfinite_rows_excluded': 'not_recorded'}`
- observations: `{'tokens_n': 622303, 'observations_n': 8449949, 'projected': {'tokens': 622303, 'fixed_age_obs_if_all_emitted': 6223030, 'event_count_obs_upper_bound': 6223030, 'total_upper_bound_before_dedupe': 12446060, 'note': 'Actual count is lower after coincident-time dedupe and when tokens have fewer than 512 trades.'}, 'actual': {'tokens': 622303, 'observations': 8449949, 'duplicate_time_collisions_removed': 0}}`
- targets: `{'v1_preserved': True, 'v2_preserved': True, 'v3_observations_processed': 8449949, 'primary_explosion_events': 49836, 'mints_with_primary_definition_row': 596397, 'mints_without_primary_definition_row': 25906, 'explosion_repair': {'method': 'copy_v2_explosion_cols_when_primary_row_exists;recompute_with_primary_None_when_absent', 'primary_semantics': 'stored primary row (exists true/false) kept; missing primary → modality_unavailable (no _stub_primary_absent)', 'non_explosion_fields': 'copied_unchanged_from_v2', 'modality_unavailable_60s_after': 340210, 'valid_negative_60s_after': 7983493, 'valid_positive_60s_after': 120638, 'horizons_sec': [10, 30, 60, 120, 300], 'invariant_no_detector_row_never_valid_negative': True}}`
- timings: `{'population': 8.99655670800712, 'sequences': 147.5605618749978, 'observations': 312.90213095900253, 'targets': 5507.716352957999, 'audit': 614.5539309169981, 'targets_v2': 5406.331738124994, 'audit_v2': 1261.7452203339926, 'targets_v3': 3425.8987888339907}`
- disk MB: **49230.192**
