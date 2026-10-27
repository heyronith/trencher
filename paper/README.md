# Paper: Into the Trenches

LaTeX source for the arXiv manuscript *Into the Trenches: Predicting Memecoin Pumps Seconds Before They Start*.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | Preamble and section order |
| `sections/` | One file per section, plus the appendix |
| `tables/` | Generated LaTeX tables, plus `facts_*.json` with the values quoted in the text |
| `figures/` | Generated figures (PDF) |
| `references.bib` | Bibliography. Every entry was checked against its source on 2026-10-27 |
| `scripts/make_assets.py` | Regenerates every table and figure from the committed evaluation reports |

## Rebuild

From the repository root:

```bash
python paper/scripts/make_assets.py           # tables/ and figures/ from manifests/world_model_data/evaluation/
cd paper && latexmk -pdf main.tex              # builds main.pdf
```

`make_assets.py` needs only `matplotlib`. It reads these reports:

- `manifests/world_model_data/evaluation/test_once_v1/` (preregistered test run, commit `704bd70`);
- the phase 3–7 review reports;
- `SPLIT_AUDIT_V1.json` and `BROAD_SUPERVISION_AUDIT_V3.json`.

No number in a table or figure is typed by hand.

## arXiv upload

Run `make arxiv` to produce `arxiv_upload.zip`. It contains:

- `main.tex`, `sections/`, `tables/*.tex`, `figures/*.pdf` and `references.bib`;
- the compiled `main.bbl`, which arXiv expects.
