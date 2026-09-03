# Historical Data Feasibility

Generated at: `2026-09-03T04:12:11.240375+00:00`

## Verdict

**Can we build the intended research-grade model from these datasets?** YES

### Fully supported

- pre-graduation trade/event history at arbitrary T
- bonding-curve price/reserve trajectory
- wallet participation history (recomputed from trades)
- graduation / time-to-graduation labels
- self-generated survival, barrier, return, MFE/MAE, inactivity labels
- buyer/activity persistence labels

### Partially supported

- Post-graduation trajectory only for ~5.7k graduated tokens
- RED-PUMP overlaps PumpFun only partially (different observation windows/populations)
- RED-COHORT is a short June 11-25 window coordination catalogue, not full-history wallet graph

### Exclude / recompute

- wallet_stats activity/volume aggregates
- uncorrected concentration percentages; top10_pct_suspect==True rows for concentration features
- System Program wallet trades for wallet features
- trades with null/inconsistent sol/price for SOL P&L features
- synthetic/backfilled migration pool_address values when needing real pools
- source outcome_label taxonomies as canonical labels
- entry_price_20s/30s/1m_usd as predictive features (graduation leakage)

### Solarchive supplementation

**Required:** False

Not required for the intended research-grade bonding-curve/graduation model. Solarchive would only be needed for raw instruction/account reconstruction beyond what trades/snapshots already encode, or for chain periods outside this corpus.

## Major quality findings

- PumpFun tokens: 798,430 rows; 5,689 graduated; 109,561 mayhem; 1,486 concentration-suspect.
- Trades: 33,581,765 rows; null price rate=0.0703; system-program rate=0.1992; sol inconsistency rate=0.0364; Jul 3 trades=0.
- Snapshots curve OOB rate=0.0271.
- Migrations synthetic/backfilled=1380/69.
- Postgard outcome_label has `rug`? False.

## Usable overlap / coverage

- PumpFun ∩ RED-PUMP mints: 108,629 (coverage of RED-PUMP=0.126)
- Graduated ∩ migrations: 5,689 (coverage of graduated=1.000)
- Cohort mints ∩ PumpFun: 4,216 (coverage of cohort mints=0.706)
- Cohort wallets ∩ trade wallets: 2,874 (coverage of cohort wallets=0.969)
- PumpFun graduated ∩ RED-PUMP graduated: 1 (near-disjoint graduation populations)

## Unsupported / unreliable / leakage

See `unreliable_fields.json`, `leakage_risk_fields.json`, and `column_concept_map.json` in this directory.

## Bottom line

These datasets are sufficient to build a research-grade pre-/peri-graduation model with self-generated labels, provided documented quality filters are applied and source-specific outcome taxonomies are not treated as canonical.
