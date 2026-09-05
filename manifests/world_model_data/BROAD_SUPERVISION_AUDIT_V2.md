# Broad Supervision Audit (`BROAD_SUPERVISION_AUDIT_V2`)

Generated at: `2026-09-05T14:50:21.111302+00:00`
Freeze: `world_model_freeze_v1`
Supersedes: `BROAD_SUPERVISION_AUDIT_V1`

## Explosion imminence comparison (60s)

- **rich_4k**: valid_tok=2987, pos_tok=211, neg_tok=2776, prev_tok=0.07063943756277201, indep=222
- **broad_v1_broken**: valid_tok=49836, pos_tok=47887, neg_tok=49836, prev_tok=0.9608917248575327, indep=49836
- **broad_v2_repaired**: valid_tok=622303, pos_tok=47887, neg_tok=622303, prev_tok=0.07695126007748637, indep=49836

- invariants: `{'positive_subseteq_valid_v2': True, 'valid_negative_tokens_substantial_v2': True, 'primary_event_count_unchanged': True}`

## Population / counts

- population: `{'launch_population': 798430, 'event_curve_population': 622303, 'price_target_population': 596338, 'graduation_event_population': 798430, 'has_migration_evidence_metadata': 5697}`
- sequences: `{'event_rows': 26886634, 'curve_rows': 26886634, 'creator_rows': 55686561, 'excluded_system_program_rows': 'not_recorded', 'exact_duplicate_groups_removed': 0, 'signatures_with_heterogeneous_event_fingerprints': 'not_recorded', 'invalid_nonfinite_rows_excluded': 'not_recorded'}`
- observations: `{'tokens_n': 622303, 'observations_n': 8449949, 'projected': {'tokens': 622303, 'fixed_age_obs_if_all_emitted': 6223030, 'event_count_obs_upper_bound': 6223030, 'total_upper_bound_before_dedupe': 12446060, 'note': 'Actual count is lower after coincident-time dedupe and when tokens have fewer than 512 trades.'}, 'actual': {'tokens': 622303, 'observations': 8449949, 'duplicate_time_collisions_removed': 0}}`
- targets: `{'v1_preserved': True, 'v2_observations_processed': 8449949, 'primary_explosion_events': 49836, 'mints_with_primary_explosion': 49836, 'explosion_repair': {'method': 'recompute_compute_explosion_obs_from_stored_explosion_events', 'primary_semantics': 'store every primary_event(events) whether explosion_exists true or false (match target_dataset_v3)', 'non_explosion_fields': 'copied_unchanged_from_v1', 'modality_unavailable_60s_before': 7600130, 'modality_unavailable_60s_after': 0, 'valid_negative_60s_after': 8323607, 'valid_positive_60s_after': 120638, 'horizons_sec': [10, 30, 60, 120, 300]}}`
- timings: `{'population': 8.99655670800712, 'sequences': 147.5605618749978, 'observations': 312.90213095900253, 'targets': 5507.716352957999, 'audit': 614.5539309169981, 'targets_v2': 5406.331738124994}`
- disk MB: **41033.365**

## Supervision sufficiency (v2 repaired)

- **graduation_time_to_event**: valid_tok=622303, pos_tok=4150, events=4150, rec=`sufficient_on_broad`
- **graduated_within_h86400_binary**: valid_tok=4150, pos_tok=4018, events=4018, rec=`marginal_on_broad`
- **explosion_within_10s**: valid_tok=622303, pos_tok=43694, events=49836, rec=`sufficient_on_broad`
- **explosion_within_30s**: valid_tok=622303, pos_tok=47031, events=49836, rec=`sufficient_on_broad`
- **explosion_within_60s**: valid_tok=622303, pos_tok=47887, events=49836, rec=`sufficient_on_broad`
- **explosion_within_120s**: valid_tok=622295, pos_tok=48556, events=49836, rec=`sufficient_on_broad`
- **explosion_within_300s**: valid_tok=622230, pos_tok=49226, events=49836, rec=`sufficient_on_broad`
- **explosion_onset_primary**: valid_tok=622303, pos_tok=47887, events=49836, rec=`sufficient_on_broad`
