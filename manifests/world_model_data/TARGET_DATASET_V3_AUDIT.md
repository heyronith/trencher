# Target Dataset v3 (`target_dataset_v3`)

Generated at: `2026-09-05T00:37:57.369108+00:00`
Freeze: `world_model_freeze_v1`
Supersedes: `target_dataset_v2`
Preserves: `['target_dataset_v1', 'target_dataset_v2']`
Observations: **51353**
Unique tokens: **4000**
Reference-price coverage: **35609** (0.693)
Disk MB: **76.5**

## Survival (activity >= T+H; negatives need H..H+30m coverage)
- positive counts monotonic nonincreasing: **True**
- by horizon: `{'30': 17658, '60': 14919, '120': 11741, '300': 7401, '600': 5271, '1800': 2488, '3600': 1562, '21600': 403, '86400': 242}`

## Isolation
- input modules do not import targets: **True**

## Explosion (primary; detector unchanged)
- independent events: **222**
- tokens with explosion: **222**
- meaningful pre-onset warning: **82**
- 4k sufficiency: **marginal**

## Supervision sufficiency
- `returns_h300`: tokens=613, pos_tokens=836, events=613, **marginal**
- `barrier_u100_l30_h3600`: tokens=1369, pos_tokens=334, events=334, **sufficient_on_rich_4k**
- `survival_24h`: tokens=3724, pos_tokens=17, events=3724, **sufficient_on_rich_4k**
- `graduation`: tokens=20, pos_tokens=20, events=20, **requires_broader_population**
- `explosion_onset_primary`: tokens=222, pos_tokens=211, events=222, **marginal**
- `holder_state_targets`: tokens=6, pos_tokens=6, events=6, **requires_broader_population**

## Graduation
- distinct graduated tokens: **20**

