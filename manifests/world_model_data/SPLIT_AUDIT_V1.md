# Split Audit (`SPLIT_AUDIT_V1`)

Generated at: `2026-09-05T17:17:57.203512+00:00`
Freeze: `world_model_freeze_v1`

## Chronological cutoffs

- method: `global_quantiles_of_event_curve_detected_at_unix`
- quantiles_iso: `{'q40': '2026-06-20T03:33:03.368926+00:00', 'q55': '2026-06-26T16:09:54.867074+00:00', 'q70': '2026-07-01T01:34:01.017306+00:00', 'q85': '2026-07-07T21:19:00.308182+00:00'}`
- launch range: `2026-06-05T09:12:26.343743+00:00` → `2026-07-14T15:01:41.919140+00:00`
- tie_policy: tokens sharing an exact detected_at_unix stay on the same side of each boundary; report actual resulting percentages

## Actual split percentages (after ties)

- `{'broad_event_curve': {'n': 622303, 'counts': {'train': 435612, 'val': 93345, 'test': 93346}, 'fractions': {'train': 0.6999998393065757, 'val': 0.1499992768795908, 'test': 0.15000088381383345}}, 'rich_subset': {'n': 4000, 'counts': {'train': 2824, 'val': 799, 'test': 377}, 'fractions': {'train': 0.706, 'val': 0.19975, 'test': 0.09425}}}`

## Token / observation counts

- tokens: `{'broad_event_curve': {'train': 435612, 'val': 93345, 'test': 93346}, 'rich_subset': {'train': 2824, 'val': 799, 'test': 377}}`
- observations: `{'broad': {'train': 5879063, 'val': 1275120, 'test': 1295766}, 'broad_total': 8449949, 'broad_late_state_purged': 17713, 'rich': {'train': 36228, 'val': 10159, 'test': 4966}, 'rich_total': 51353, 'rich_late_state_purged': 169}`

## Leakage invariants

- all_passed: **True**
- details: `{'token_disjoint': {'passed': True, 'counts': {'train': 558286, 'val': 124117, 'test': 116027}, 'overlap_samples': {'train_val': [], 'train_test': [], 'val_test': []}}, 'chronological_ordering': {'passed': True, 'max_train_launch': 1782869635.119335, 'min_val_launch': 1782869649.864263, 'max_val_launch': 1783459137.742453, 'min_test_launch': 1783459141.407781, 'q70': 1782869641.017306, 'q85': 1783459140.3081825, 'reasons': []}, 'walkforward_excludes_test': {'passed': True, 'test_tokens_in_folds': 0}, 'broad_eligibility_strict': {'passed': True, 'rows_checked': 99915, 'violations': 0, 'parts_sampled': 3}, 'rich_eligibility_strict': {'passed': True, 'rows_checked': 51353, 'violations': 0, 'parts_sampled': 1}, 'all_passed': True}`

## Walk-forward

- `[{'fold_id': 'wf_fold_1', 'train': {'start_unix': 1780650746.343743, 'end_unix_exclusive': 1781926383.3689256}, 'val': {'start_unix': 1781926383.3689256, 'end_unix_exclusive': 1782490194.8670735}, 'rule': 'train=[min,q40), val=[q40,q55)'}, {'fold_id': 'wf_fold_2', 'train': {'start_unix': 1780650746.343743, 'end_unix_exclusive': 1782490194.8670735}, 'val': {'start_unix': 1782490194.8670735, 'end_unix_exclusive': 1782869641.017306}, 'rule': 'train=[min,q55), val=[q55,q70)'}, {'fold_id': 'wf_fold_3', 'train': {'start_unix': 1780650746.343743, 'end_unix_exclusive': 1782869641.017306}, 'val': {'start_unix': 1782869641.017306, 'end_unix_exclusive': 1783459140.3081825}, 'rule': 'train=[min,q70), val=[q70,q85)'}]`
