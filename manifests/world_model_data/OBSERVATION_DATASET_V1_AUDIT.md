# Observation Dataset v1 (`observation_dataset_v1`)

Generated at: `2026-09-04T17:22:10.434680+00:00`
Freeze: `world_model_freeze_v1`
PIT engine: `pit_v4`

- tokens: **4000**
- observations: **51353**
- by trigger: `{'fixed_age': 40000, 'event_count': 11353}`
- obs/token: `{'min': 10, 'p50': 13.0, 'p90': 16.0, 'p99': 20.0, 'max': 20, 'mean': 12.83825}`
- duplicate collisions removed: 0
- disk MB: 88.552
- leakage all_passed: **True**

## Known limitations

- Model states store JSON payloads; wallet graphs can be large for high-activity mints.
- Event-driven points use atomic Pump trades (curve fallback only if atomic trades absent).
- Global trade context lags by <1 completed second by design.
- Ownership reconstructable remains rare under partition-coverage rules.
- No prediction labels or outcome fields are attached.
