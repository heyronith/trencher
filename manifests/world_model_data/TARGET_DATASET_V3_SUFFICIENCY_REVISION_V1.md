# Target Dataset v3 Sufficiency Revision

Revision: `target_dataset_v3_sufficiency_revision_v1`
Generated: `2026-09-05T04:03:17.196868+00:00`
Target semantics unchanged: **True**

Invariant: `distinct_positive_tokens ⊆ unique_tokens_with_valid_supervision`

## Notes

Corrects returns_h300 positives to require valid status; corrects explosion imminence so valid tokens include status-gated valid_negatives (not merely eventual explosion tokens); splits graduation into time-to-event vs binary within-H sufficiency.

## Diffs vs prior audit

- `returns_h300`: pos_tokens 310→310; valid_tokens 613→613; subset_ok=True
- `barrier_u100_l30_h3600`: pos_tokens 334→334; valid_tokens 1369→1369; subset_ok=True
- `survival_24h`: pos_tokens 17→17; valid_tokens 3724→3724; subset_ok=True
- `graduation_time_to_event`: pos_tokens None→20; valid_tokens None→4000; subset_ok=True
- `graduated_within_h86400_binary`: pos_tokens None→20; valid_tokens None→20; subset_ok=True
- `explosion_within_10s`: pos_tokens None→192; valid_tokens None→2987; subset_ok=True
- `explosion_within_30s`: pos_tokens None→208; valid_tokens None→2987; subset_ok=True
- `explosion_within_60s`: pos_tokens None→211; valid_tokens None→2987; subset_ok=True
- `explosion_within_120s`: pos_tokens None→215; valid_tokens None→2987; subset_ok=True
- `explosion_within_300s`: pos_tokens None→218; valid_tokens None→2986; subset_ok=True
- `explosion_onset_primary`: pos_tokens 211→211; valid_tokens 222→2987; subset_ok=True
- `holder_state_targets`: pos_tokens 6→6; valid_tokens 6→6; subset_ok=True

## Corrected sufficiency

- `returns_h300`: valid_tok=613, pos_tok=310, events=613, prev_tok=0.5057096247960848, **marginal**
- `barrier_u100_l30_h3600`: valid_tok=1369, pos_tok=334, events=334, prev_tok=0.24397370343316288, **sufficient_on_rich_4k**
- `survival_24h`: valid_tok=3724, pos_tok=17, events=3724, prev_tok=0.004564983888292159, **sufficient_on_rich_4k**
- `graduation_time_to_event`: valid_tok=4000, pos_tok=20, events=20, prev_tok=0.005, **requires_broader_population**
- `graduated_within_h86400_binary`: valid_tok=20, pos_tok=20, events=20, prev_tok=1.0, **requires_broader_population**
- `explosion_within_10s`: valid_tok=2987, pos_tok=192, events=222, prev_tok=0.06427854034147974, **marginal**
- `explosion_within_30s`: valid_tok=2987, pos_tok=208, events=222, prev_tok=0.06963508536993639, **marginal**
- `explosion_within_60s`: valid_tok=2987, pos_tok=211, events=222, prev_tok=0.07063943756277201, **marginal**
- `explosion_within_120s`: valid_tok=2987, pos_tok=215, events=222, prev_tok=0.07197857381988618, **marginal**
- `explosion_within_300s`: valid_tok=2986, pos_tok=218, events=222, prev_tok=0.07300736771600803, **marginal**
- `explosion_onset_primary`: valid_tok=2987, pos_tok=211, events=222, prev_tok=0.07063943756277201, **marginal**
- `holder_state_targets`: valid_tok=6, pos_tok=6, events=6, prev_tok=1.0, **requires_broader_population**
