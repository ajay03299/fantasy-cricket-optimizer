import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("output/player_match_fantasy_v1_with_opponent.csv")
OUTPUT_PATH = Path("output/model_feature_table_v1.csv")


def add_group_history_features(df, group_cols, value_col, feature_prefix):
    df = df.copy()

    shifted = df.groupby(group_cols)[value_col].shift(1)

    df[f"{feature_prefix}_last_1"] = shifted
    df[f"{feature_prefix}_last_3_avg"] = (
        shifted.groupby([df[c] for c in group_cols])
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    df[f"{feature_prefix}_last_5_avg"] = (
        shifted.groupby([df[c] for c in group_cols])
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )

    return df


def add_expanding_mean_feature(df, group_cols, value_col, out_col):
    df = df.copy()

    shifted = df.groupby(group_cols)[value_col].shift(1)
    df[out_col] = (
        shifted.groupby([df[c] for c in group_cols])
        .expanding()
        .mean()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )

    return df


def main():
    df = pd.read_csv(INPUT_PATH)

    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["player_name", "match_date", "match_id"]).reset_index(drop=True)

    # Matches played before current one
    df["matches_played_before"] = df.groupby("player_name").cumcount()

    # Career expanding mean of fantasy points before match
    df = add_expanding_mean_feature(
        df,
        ["player_name"],
        "fantasy_points_v1",
        "career_avg_points_before"
    )

    # Recent player history
    df = add_group_history_features(df, ["player_name"], "fantasy_points_v1", "points")
    df = add_group_history_features(df, ["player_name"], "runs_scored", "runs")
    df = add_group_history_features(df, ["player_name"], "wickets", "wickets")
    df = add_group_history_features(df, ["player_name"], "balls_faced", "balls_faced")
    df = add_group_history_features(df, ["player_name"], "balls_bowled", "balls_bowled")

    # Venue history before match
    df = add_expanding_mean_feature(
        df,
        ["player_name", "venue"],
        "fantasy_points_v1",
        "avg_points_at_venue_before"
    )

    df = add_expanding_mean_feature(
        df,
        ["player_name", "venue"],
        "runs_scored",
        "avg_runs_at_venue_before"
    )

    # Opponent history before match
    df = add_expanding_mean_feature(
        df,
        ["player_name", "opponent"],
        "fantasy_points_v1",
        "avg_points_vs_opponent_before"
    )

    df = add_expanding_mean_feature(
        df,
        ["player_name", "opponent"],
        "wickets",
        "avg_wickets_vs_opponent_before"
    )

    # Optional fill strategy for early-career rows
    feature_cols = [
        "career_avg_points_before",
        "points_last_1",
        "points_last_3_avg",
        "points_last_5_avg",
        "runs_last_1",
        "runs_last_3_avg",
        "runs_last_5_avg",
        "wickets_last_1",
        "wickets_last_3_avg",
        "wickets_last_5_avg",
        "balls_faced_last_1",
        "balls_faced_last_3_avg",
        "balls_faced_last_5_avg",
        "balls_bowled_last_1",
        "balls_bowled_last_3_avg",
        "balls_bowled_last_5_avg",
        "avg_points_at_venue_before",
        "avg_runs_at_venue_before",
        "avg_points_vs_opponent_before",
        "avg_wickets_vs_opponent_before",
    ]

    for col in feature_cols:
        df[col] = df[col].fillna(0)

    # Keep a clean model-ready subset
    model_cols = [
        "match_id",
        "match_date",
        "season",
        "player_name",
        "team",
        "opponent",
        "venue",
        "player_role_platform",
        "matches_played_before",
        "career_avg_points_before",
        "points_last_1",
        "points_last_3_avg",
        "points_last_5_avg",
        "runs_last_1",
        "runs_last_3_avg",
        "runs_last_5_avg",
        "wickets_last_1",
        "wickets_last_3_avg",
        "wickets_last_5_avg",
        "balls_faced_last_1",
        "balls_faced_last_3_avg",
        "balls_faced_last_5_avg",
        "balls_bowled_last_1",
        "balls_bowled_last_3_avg",
        "balls_bowled_last_5_avg",
        "avg_points_at_venue_before",
        "avg_runs_at_venue_before",
        "avg_points_vs_opponent_before",
        "avg_wickets_vs_opponent_before",
        "fantasy_points_v1",
    ]

    model_df = df[model_cols].copy()
    model_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("\nShape:")
    print(model_df.shape)

    print("\nSample rows:")
    print(model_df.head(10).to_string(index=False))

    print("\nMissing values in feature table:")
    print(model_df.isna().sum())


if __name__ == "__main__":
    main()