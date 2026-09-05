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
- `returns_h300`: valid_obs=4717, valid_tokens=613, pos_obs=1326, pos_tokens=310, events=613, prev_valid=0.2811108755564978, **marginal**
- `barrier_u100_l30_h3600`: valid_obs=7473, valid_tokens=1369, pos_obs=994, pos_tokens=334, events=334, prev_valid=0.13301217717114946, **sufficient_on_rich_4k**
- `survival_24h`: valid_obs=47665, valid_tokens=3724, pos_obs=242, pos_tokens=17, events=3724, prev_valid=0.005077100597923005, **sufficient_on_rich_4k**
- `graduation`: valid_obs=293, valid_tokens=20, pos_obs=165, pos_tokens=20, events=20, prev_valid=0.5631399317406144, **requires_broader_population**
- `explosion_onset_primary`: valid_obs=40507, valid_tokens=222, pos_obs=555, pos_tokens=211, events=222, prev_valid=0.013701335571629595, **marginal**
- `holder_state_targets`: valid_obs=85, valid_tokens=6, pos_obs=85, pos_tokens=6, events=6, prev_valid=1.0, **requires_broader_population**

## Graduation
- distinct graduated tokens: **20**

