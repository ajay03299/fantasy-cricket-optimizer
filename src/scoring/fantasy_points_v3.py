from pathlib import Path
import pandas as pd
import numpy as np

INPUT_PATH = Path("output/v3_dataset_with_opponent.csv")
OUTPUT_PATH = Path("output/player_match_fantasy_v3.csv")

REQUIRED_COLUMNS = [
    "match_id",
    "player_name",
    "team",
    "opponent",
    "player_role_platform",
    "in_announced_lineup",
    "runs_scored",
    "balls_faced",
    "fours",
    "sixes",
    "batting_strike_rate",
    "dismissed",
    "is_duck",
    "balls_bowled",
    "runs_conceded",
    "wickets",
    "maidens",
    "bowling_economy",
    "catches",
    "stumpings",
    "runout_direct",
    "runout_assist",
    "match_date",
    "season",
    "venue",
]


def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_fantasy_points_v3(df):
    df = df.copy()

    numeric_cols = [
        "in_announced_lineup",
        "runs_scored",
        "balls_faced",
        "fours",
        "sixes",
        "batting_strike_rate",
        "dismissed",
        "is_duck",
        "balls_bowled",
        "runs_conceded",
        "wickets",
        "maidens",
        "bowling_economy",
        "catches",
        "stumpings",
        "runout_direct",
        "runout_assist",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Batting
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

    df["points_duck_penalty"] = np.where(
        (df["is_duck"] == 1) & (df["player_role_platform"].isin(["BAT", "WK", "AR"])),
        -2,
        0,
    )

    df["points_strike_rate_adjustment"] = np.select(
        [
            (df["player_role_platform"] != "BOWL") & (df["balls_faced"] >= 10) & (df["batting_strike_rate"] >= 60) & (df["batting_strike_rate"] < 70),
            (df["player_role_platform"] != "BOWL") & (df["balls_faced"] >= 10) & (df["batting_strike_rate"] >= 50) & (df["batting_strike_rate"] < 60),
            (df["player_role_platform"] != "BOWL") & (df["balls_faced"] >= 10) & (df["batting_strike_rate"] < 50),
        ],
        [-2, -4, -6],
        default=0,
    )

    # Bowling
    df["points_wickets"] = df["wickets"] * 25
    df["points_wicket_haul_bonus"] = np.select(
        [
            df["wickets"] >= 5,
            (df["wickets"] >= 4) & (df["wickets"] < 5),
        ],
        [16, 8],
        default=0,
    )
    df["points_maiden_bonus"] = df["maidens"] * 8

    df["points_economy_adjustment"] = np.select(
        [
            (df["balls_bowled"] >= 12) & (df["bowling_economy"] >= 4) & (df["bowling_economy"] < 5),
            (df["balls_bowled"] >= 12) & (df["bowling_economy"] >= 5) & (df["bowling_economy"] < 6),
        ],
        [4, 2],
        default=0,
    )

    # Fielding
    df["points_catches"] = df["catches"] * 8
    df["points_stumpings"] = df["stumpings"] * 12
    df["points_runout_direct"] = df["runout_direct"] * 12
    df["points_runout_assist"] = df["runout_assist"] * 6

    # Lineup
    df["points_lineup"] = df["in_announced_lineup"] * 4

    score_cols = [
        "points_batting_runs",
        "points_boundary_bonus",
        "points_six_bonus",
        "points_batting_milestone_bonus",
        "points_duck_penalty",
        "points_strike_rate_adjustment",
        "points_wickets",
        "points_wicket_haul_bonus",
        "points_maiden_bonus",
        "points_economy_adjustment",
        "points_catches",
        "points_stumpings",
        "points_runout_direct",
        "points_runout_assist",
        "points_lineup",
    ]

    df["fantasy_points_v3"] = df[score_cols].sum(axis=1)
    return df


def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    validate_columns(df)
    out = add_fantasy_points_v3(df)
    out.to_csv(OUTPUT_PATH, index=False)

    print("Input rows:", len(df))
    print("Saved:", OUTPUT_PATH)
    print("\nTop 15 rows:")
    print(
        out.sort_values("fantasy_points_v3", ascending=False)[
            ["match_id", "player_name", "team", "player_role_platform", "runs_scored", "wickets", "maidens", "catches", "stumpings", "runout_direct", "runout_assist", "fantasy_points_v3"]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()