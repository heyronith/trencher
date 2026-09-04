# Fusion Expansion Cost Plan

Generated at: `2026-09-04T04:24:29.662126+00:00`

## BigQuery account state

- Bytes already consumed this month: **901.051 GiB**
- Free-tier remaining: **122.949 GiB**
- Emergency reserve: **80.0 GiB**
- On-demand price assumption: **$6.25/TiB**

## Fusion target

- Current multimodal overlap: **287**
- Target: **2000**
- Gap: **1713**
- Limiting modality: **ownership_spl** (2846 tokens need ownership only)

## Recommended path (non-BigQuery first)

2,846 tokens already have event+graph+micro+execution and lack only ownership. Launch-day BQ partitions were already scanned. Re-parsing archival getTransaction pre/post token balances for existing Pump trade signatures yields genuine SPL ownership edges without BQ cost.

- Expected overlap if RPC ownership succeeds for all targets: **3133**
- Projected BQ processing $: **$0.00**

## BigQuery-only alternative (dry-run ranked)

- Unqueried partitions: **20** days / **301.185 GiB**
- Expected new ownership mints (conservative / base / optimistic): **137 / 290 / 725**
- Projected overlap after all unqueried days: **424 / 577 / 1012**
- Paid bytes if executed: **178.236 GiB**
- Projected paid USD: **$1.09**
- BQ alone reaches 2000 at base rate: **False**

### Partition ranking (tokens unlocked / GiB, base)

| day | est GiB | eligible | unlock base | tokens/GiB | cum overlap |
| --- | ------: | -------: | ----------: | ---------: | ----------: |
| 2026-07-11 | 13.5203 | 2750 | 22 | 1.6272 | 309 |
| 2026-07-13 | 16.6348 | 2750 | 22 | 1.3225 | 331 |
| 2026-07-08 | 17.1195 | 2586 | 21 | 1.2267 | 352 |
| 2026-07-09 | 17.1368 | 2586 | 21 | 1.2254 | 373 |
| 2026-07-05 | 15.8008 | 2386 | 19 | 1.2025 | 392 |
| 2026-07-06 | 17.8249 | 2386 | 19 | 1.0659 | 411 |
| 2026-06-27 | 15.2605 | 1838 | 15 | 0.9829 | 426 |
| 2026-06-21 | 12.2027 | 1390 | 11 | 0.9014 | 437 |
| 2026-06-30 | 18.4455 | 2016 | 16 | 0.8674 | 453 |
| 2026-06-20 | 12.7875 | 1390 | 11 | 0.8602 | 464 |
| 2026-06-19 | 12.997 | 1390 | 11 | 0.8463 | 475 |
| 2026-06-24 | 15.4392 | 1607 | 13 | 0.842 | 488 |
| 2026-06-23 | 15.475 | 1607 | 13 | 0.8401 | 501 |
| 2026-06-25 | 16.1848 | 1607 | 13 | 0.8032 | 514 |
| 2026-06-17 | 13.7649 | 1390 | 11 | 0.7991 | 525 |
| 2026-06-28 | 19.0794 | 1838 | 15 | 0.7862 | 540 |
| 2026-06-16 | 14.39 | 1390 | 11 | 0.7644 | 551 |
| 2026-06-14 | 11.8997 | 1170 | 9 | 0.7563 | 560 |
| 2026-06-13 | 12.0069 | 1170 | 9 | 0.7496 | 569 |
| 2026-06-11 | 13.2149 | 960 | 8 | 0.6054 | 577 |

## Decision

- Proceed with RPC ownership: **True**
- Proceed with BQ before review: **False**
- Forced-all-days BQ USD (under $50?): **$1.09** (True)

BQ unqueried-day scans are cheap in USD but cannot reach ~2000 overlap at observed Token Transfers hit rates; execute RPC token-balance ownership recovery first (projected $0 BQ).

Dry-runs do not incur BigQuery data-processing charges.

## Execution actuals

- BQ bytes processed/billed this expansion: **0 / 0**
- BQ USD charged (est.): **$0.00**
- RPC ownership mints unlocked: **2845**
- Final fusion overlap: **3132**
- Target (>=2000) met: **True**
