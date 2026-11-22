"""Combine locally downloaded shotdetail season CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


REQUIRED_COLUMNS = {
    "GAME_ID",
    "PLAYER_ID",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "SHOT_MADE_FLAG",
    "SHOT_TYPE",
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "ACTION_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/shots.csv")
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("shotdetail_*.csv"))
    files += sorted(args.input_dir.glob("shotdetail_*.csv.gz"))
    if not files:
        raise SystemExit(
            f"No shotdetail_*.csv or shotdetail_*.csv.gz files found in {args.input_dir}"
        )

    seasons = []
    seen_seasons: set[int] = set()
    for file_path in files:
        match = re.search(r"shotdetail_(\d{4})\.csv(?:\.gz)?$", file_path.name)
        if match is None:
            raise SystemExit(f"Could not read a season year from {file_path.name}")

        season = int(match.group(1))
        if season in seen_seasons:
            raise SystemExit(f"More than one input file found for season {season}")
        seen_seasons.add(season)

        frame = pd.read_csv(file_path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise SystemExit(
                f"{file_path.name} is missing required columns: {sorted(missing)}"
            )
        frame["SEASON"] = season
        seasons.append(frame)

    combined = pd.concat(seasons, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined):,} shots across {len(seasons)} seasons to {args.output}")


if __name__ == "__main__":
    main()
