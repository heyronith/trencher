# Target Dataset v1 (`target_dataset_v1`)

Generated at: `2026-09-04T23:31:42.883318+00:00`
Freeze: `world_model_freeze_v1`
Observations: **51353**
Unique tokens: **4000**
Reference-price coverage: **35609** (0.693)
Disk MB: **74.6**

## Isolation
- input modules do not import targets: **True**

## Explosion (primary)
- independent events: **222**
- tokens with explosion: **222**
- meaningful pre-onset warning: **82** (0.369)
- zero measurable precursor: **130**
- 4k sufficiency: **marginal**

## Supervision sufficiency
- `returns_h300`: tokens=613, pos_tokens=836, events=613, **marginal**
- `barrier_u100_l30_h3600`: tokens=1369, pos_tokens=334, events=334, **sufficient_on_rich_4k**
- `survival_24h`: tokens=2987, pos_tokens=2983, events=2987, **sufficient_on_rich_4k**
- `graduation`: tokens=37, pos_tokens=20, events=20, **requires_broader_population**
- `explosion_onset_primary`: tokens=222, pos_tokens=211, events=222, **marginal**
- `holder_state_targets`: tokens=6, pos_tokens=6, events=6, **requires_broader_population**

## Graduation
- distinct graduated tokens: **20**

