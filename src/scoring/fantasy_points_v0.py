from pathlib import Path
import pandas as pd
import numpy as np


INPUT_PATH = Path("data/processed/cleaned_master_dataset.csv")
OUTPUT_PATH = Path("output/player_match_fantasy_v0.csv")


REQUIRED_COLUMNS = [
    "match_id",
    "player_name",
    "runs_scored",
    "balls_faced",
    "fours",
    "sixes",
    "batting_strike_rate",
    "balls_bowled",
    "runs_conceded",
    "wickets",
    "bowling_economy",
    "catches",
    "stumpings",
    "runouts",
    "team",
    "match_date",
    "season",
    "venue",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_fantasy_points_v0(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "runs_scored",
        "balls_faced",
        "fours",
        "sixes",
        "batting_strike_rate",
        "balls_bowled",
        "runs_conceded",
        "wickets",
        "bowling_economy",
        "catches",
        "stumpings",
        "runouts",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["overs_bowled"] = df["balls_bowled"] / 6.0

    df["points_batting_runs"] = df["runs_scored"]
    df["points_boundary_bonus"] = df["fours"] * 1
    df["points_six_bonus"] = df["sixes"] * 2

    df["points_batting_milestone_bonus"] = np.select(
        [
            df["runs_scored"] >= 100,
            (df["runs_scored"] >= 50) & (df["runs_scored"] < 100),
        ],
        [16, 8],
        default=0,
    )

    df["points_strike_rate_adjustment"] = np.select(
        [
            (df["balls_faced"] >= 10)
            & (df["batting_strike_rate"] >= 60)
            & (df["batting_strike_rate"] < 70),
            (df["balls_faced"] >= 10)
            & (df["batting_strike_rate"] >= 50)
            & (df["batting_strike_rate"] < 60),
            (df["balls_faced"] >= 10)
            & (df["batting_strike_rate"] < 50),
        ],
        [-2, -4, -6],
        default=0,
    )

    df["points_duck_penalty"] = 0
    df["points_lineup"] = 0

    df["points_wickets"] = df["wickets"] * 25

    df["points_wicket_haul_bonus"] = np.select(
        [
            df["wickets"] >= 5,
            (df["wickets"] >= 4) & (df["wickets"] < 5),
        ],
        [16, 8],
        default=0,
    )

    df["points_maiden_bonus"] = 0

    df["points_economy_adjustment"] = np.select(
        [
            (df["balls_bowled"] >= 12)
            & (df["bowling_economy"] >= 4)
            & (df["bowling_economy"] < 5),
            (df["balls_bowled"] >= 12)
            & (df["bowling_economy"] >= 5)
            & (df["bowling_economy"] < 6),
        ],
        [4, 2],
        default=0,
    )

    df["points_catches"] = df["catches"] * 8
    df["points_stumpings"] = df["stumpings"] * 12
    df["points_runouts_proxy"] = df["runouts"] * 6

    score_cols = [
        "points_batting_runs",
        "points_boundary_bonus",
        "points_six_bonus",
        "points_batting_milestone_bonus",
        "points_strike_rate_adjustment",
        "points_duck_penalty",
        "points_lineup",
        "points_wickets",
        "points_wicket_haul_bonus",
        "points_maiden_bonus",
        "points_economy_adjustment",
        "points_catches",
        "points_stumpings",
        "points_runouts_proxy",
    ]

    df["fantasy_points_v0"] = df[score_cols].sum(axis=1)

    return df


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    validate_columns(df)
    out = add_fantasy_points_v0(df)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("\nTop 10 fantasy_points_v0 rows:")
    preview_cols = [
        "match_id",
        "player_name",
        "team",
        "runs_scored",
        "wickets",
        "catches",
        "stumpings",
        "runouts",
        "fantasy_points_v0",
    ]
    print(out.sort_values("fantasy_points_v0", ascending=False)[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()