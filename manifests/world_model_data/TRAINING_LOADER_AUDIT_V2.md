# Training Loader Audit (`TRAINING_LOADER_AUDIT_V2`)

Generated: `2026-09-05T17:54:14.015835+00:00`

## TTE overlays

`{'broad_rows': 8449949, 'broad_grad_admin_censored': 8382533, 'broad_expl_admin_censored': 7600147, 'rich_rows': 51353, 'rich_grad_admin_censored': 51060, 'rich_expl_admin_censored': 47602}`

## Smoke

- peak_rss_mb: **8091.78125**
- broad: `{'examples': 320, 'batches': 20, 'examples_per_sec': 7.689491804445362, 'batches_per_sec': 0.48059323777783514, 'elapsed_sec': 41.61523389816284, 'sequence_shapes': {'creator': [16, 256, 2], 'curve': [16, 42, 5], 'events': [16, 42, 6]}, 'length_means': {'creator': 76.040625, 'curve': 16.009375, 'events': 16.009375}, 'truncation_rate_mean': {'events': 0.00625, 'curve': 0.00625, 'creator': 0.0, 'micro': 0.0, 'ownership': 0.0, 'graph': 0.0}, 'modality_availability_mean': {'modality_mask_creator': 0.775, 'modality_mask_curve': 0.88125, 'modality_mask_events': 0.88125}, 'target_status_example': {'available': True, 'returns_h60': {'is_missing': False, 'frozen_status': 'valid_negative'}, 'graduation_tte': {'effective_event_observed': False, 'split_boundary_censored': True, 'frozen_tte_sec': None}, 'explosion_tte': {'effective_event_observed': False, 'split_boundary_censored': True}}, 'causal_failures': 0, 'production': True}`
- rich: `{'examples': 320, 'batches': 20, 'examples_per_sec': 2.5054350726463683, 'batches_per_sec': 0.15658969204039802, 'elapsed_sec': 127.72232794761658, 'sequence_shapes': {'curve': [16, 41, 4], 'events': [16, 13, 6], 'graph_token': [16, 14, 2], 'micro': [16, 6, 2], 'ownership': [16, 2, 2], 'wallet_prior': [16, 1, 2]}, 'length_means': {'curve': 15.375, 'events': 12.765625, 'graph_token': 14.0, 'micro': 2.996875, 'ownership': 2.01875, 'wallet_prior': 1.0}, 'truncation_rate_mean': {'events': 0.0125, 'curve': 0.0, 'creator': 0.0, 'micro': 0.0, 'ownership': 0.0, 'graph': 0.015625}, 'modality_availability_mean': {'modality_mask_curve': 0.75625, 'modality_mask_events': 0.7625, 'modality_mask_graph': 1.0, 'modality_mask_micro': 0.771875, 'modality_mask_ownership': 0.734375, 'modality_mask_wallet_prior': 1.0}, 'target_status_example': {'available': True, 'returns_h60': {'is_missing': True, 'frozen_status': 'right_censored'}, 'graduation_tte': {'effective_event_observed': False, 'split_boundary_censored': True, 'frozen_tte_sec': None}, 'explosion_tte': {'effective_event_observed': False, 'split_boundary_censored': True}}, 'causal_failures': 0, 'production': True}`
