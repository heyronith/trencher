# Training Loader Audit (`TRAINING_LOADER_AUDIT_V1`)

Generated at: `2026-09-05T17:21:09.897900+00:00`

## Transforms

- `{'path': 'data/external/world_model/splits/split_v1/transforms_train_v1/sequence_numeric_stats.json', 'rows_seen': 2195762, 'train_mints_n': 435612, 'columns': ['sol_amount', 'token_amount'], 'fit_scope': 'unique_underlying_training_sequence_rows'}`

## Smoke benchmark

- peak_rss_mb: **1655.515625**
- broad: `{'examples': 1280, 'batches': 20, 'examples_per_sec': 9083.75667275216, 'batches_per_sec': 141.9336980117525, 'feature_shapes': {'modality_mask_creator': [64], 'modality_mask_curve': [64], 'modality_mask_events': [64], 'modality_mask_graph': [64], 'modality_mask_micro': [64], 'modality_mask_ownership': [64], 'observation_time_t': [64], 'time_delta_from_launch': [64], 'token_age_seconds': [64]}, 'modality_availability_mean': {'modality_mask_creator': 1.0, 'modality_mask_curve': 1.0, 'modality_mask_events': 1.0, 'modality_mask_graph': 0.0, 'modality_mask_micro': 0.0, 'modality_mask_ownership': 0.0}, 'elapsed_sec': 0.14091086387634277}`
- rich: `{'examples': 1280, 'batches': 20, 'examples_per_sec': 6018.151971895087, 'batches_per_sec': 94.03362456086073, 'feature_shapes': {'modality_mask_creator': [64], 'modality_mask_curve': [64], 'modality_mask_events': [64], 'modality_mask_graph': [64], 'modality_mask_micro': [64], 'modality_mask_ownership': [64], 'observation_time_t': [64], 'time_delta_from_launch': [64], 'token_age_seconds': [64]}, 'modality_availability_mean': {'modality_mask_creator': 1.0, 'modality_mask_curve': 1.0, 'modality_mask_events': 1.0, 'modality_mask_graph': 1.0, 'modality_mask_micro': 1.0, 'modality_mask_ownership': 1.0}, 'elapsed_sec': 0.21268987655639648}`
- mixed: `{'batches': 20, 'batches_per_sec': 60.88261784991359, 'elapsed_sec': 0.32850098609924316}`
