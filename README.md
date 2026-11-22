# NBA Shot Outcome Modeling

An independent portfolio project that estimates make probabilities from publicly available NBA shot chart data. It uses only shot chart context such as shot distance, court location, shot type, and game clock. It does not use or reproduce any internship assessment data, code, results, or materials.

## Scope

The model estimates make probability conditional on a recorded shot's location and type. This is a retrospective shot quality analysis, not a pre-shot decision model. The public shot chart fields do not include defender tracking or contest distance, so this project makes no claims about those factors.

## Data

The scripts accept season-specific CSVs for the `shotdetail` data described by [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data), whose repository describes its shot details as sourced from NBA Stats. The repository lists an Apache-2.0 license, but that should not be taken as a license from the NBA for underlying records. Review the source and applicable terms before use. This project does not redistribute any dataset.

Download and extract two or more regular-season `shotdetail_YYYY` CSV files into `data/raw/`, retaining the year in each filename, for example `shotdetail_2022.csv`. The data directory is ignored by Git.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/evaluate.py --validation-season 2024
```

`prepare_data.py` combines the local season files and adds a season column from each filename. `evaluate.py` holds out the selected season, trains on earlier seasons, compares a logistic regression baseline with a random forest, and reports log loss and Brier score. It writes a metrics file and a calibration chart under `reports/`; these outputs are local and ignored by Git.

## Validation and limitations

Holding out a complete season gives a more realistic check of transfer across time than randomly splitting individual shots. The dataset is observational and its public shot chart fields are limited. Results can be affected by rule changes, season composition, player and team mix, and changes in how shots are recorded. This project reports validation results only; it does not claim performance on future seasons.

## Project structure

```text
scripts/
  prepare_data.py   # combine local season CSVs
  evaluate.py       # season holdout and probability evaluation
data/raw/           # local source files; not committed
reports/            # local metrics and chart; not committed
```
