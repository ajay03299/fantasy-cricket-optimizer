import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("output/player_match_fantasy_v1_with_opponent.csv")
OUTPUT_PATH = Path("output/model_feature_table_v2.csv")


def add_shifted_group_roll_mean(df, group_col, value_col, window, out_col):
    shifted = df.groupby(group_col)[value_col].shift(1)
    df[out_col] = (
        shifted.groupby(df[group_col])
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def add_shifted_group_roll_std(df, group_col, value_col, window, out_col):
    shifted = df.groupby(group_col)[value_col].shift(1)
    df[out_col] = (
        shifted.groupby(df[group_col])
        .rolling(window, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )
    return df


def add_shifted_group_roll_max(df, group_col, value_col, window, out_col):
    shifted = df.groupby(group_col)[value_col].shift(1)
    df[out_col] = (
        shifted.groupby(df[group_col])
        .rolling(window, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )
    return df


def add_shifted_group_roll_rate(df, group_col, condition_series, window, out_col):
    shifted = condition_series.groupby(df[group_col]).shift(1)
    df[out_col] = (
        shifted.groupby(df[group_col])
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df


def add_expanding_mean(df, group_cols, value_col, out_col):
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

    df["matches_played_before"] = df.groupby("player_name").cumcount()

    # Existing core features
    df = add_expanding_mean(df, ["player_name"], "fantasy_points_v1", "career_avg_points_before")

    for w in [1, 3, 5, 10]:
        if w == 1:
            df = add_shifted_group_roll_mean(df, "player_name", "fantasy_points_v1", 1, "points_last_1")
            df = add_shifted_group_roll_mean(df, "player_name", "runs_scored", 1, "runs_last_1")
            df = add_shifted_group_roll_mean(df, "player_name", "wickets", 1, "wickets_last_1")
            df = add_shifted_group_roll_mean(df, "player_name", "balls_faced", 1, "balls_faced_last_1")
            df = add_shifted_group_roll_mean(df, "player_name", "balls_bowled", 1, "balls_bowled_last_1")
        else:
            df = add_shifted_group_roll_mean(df, "player_name", "fantasy_points_v1", w, f"points_last_{w}_avg")
            df = add_shifted_group_roll_mean(df, "player_name", "runs_scored", w, f"runs_last_{w}_avg")
            df = add_shifted_group_roll_mean(df, "player_name", "wickets", w, f"wickets_last_{w}_avg")
            df = add_shifted_group_roll_mean(df, "player_name", "balls_faced", w, f"balls_faced_last_{w}_avg")
            df = add_shifted_group_roll_mean(df, "player_name", "balls_bowled", w, f"balls_bowled_last_{w}_avg")

    # New volatility / ceiling features
    df = add_shifted_group_roll_std(df, "player_name", "fantasy_points_v1", 5, "points_last_5_std")
    df = add_shifted_group_roll_std(df, "player_name", "fantasy_points_v1", 10, "points_last_10_std")
    df = add_shifted_group_roll_max(df, "player_name", "fantasy_points_v1", 5, "points_last_5_max")
    df = add_shifted_group_roll_max(df, "player_name", "fantasy_points_v1", 10, "points_last_10_max")

    # Fielding history
    df = add_shifted_group_roll_mean(df, "player_name", "catches", 5, "catches_last_5_avg")
    df = add_shifted_group_roll_mean(df, "player_name", "stumpings", 5, "stumpings_last_5_avg")
    df = add_shifted_group_roll_mean(df, "player_name", "runouts", 5, "runouts_last_5_avg")

    # Maiden history
    df = add_shifted_group_roll_mean(df, "player_name", "maidens", 5, "maidens_last_5_avg")

    # Activity rates
    batting_active = (df["balls_faced"] > 0).astype(float)
    bowling_active = (df["balls_bowled"] > 0).astype(float)
    dismissed_flag = (df["dismissed"] > 0).astype(float)
    duck_flag = (df["is_duck"] > 0).astype(float)

    df = add_shifted_group_roll_rate(df, "player_name", batting_active, 5, "batting_active_rate_last_5")
    df = add_shifted_group_roll_rate(df, "player_name", bowling_active, 5, "bowling_active_rate_last_5")
    df = add_shifted_group_roll_rate(df, "player_name", dismissed_flag, 5, "dismissal_rate_last_5")
    df = add_shifted_group_roll_rate(df, "player_name", duck_flag, 10, "duck_rate_last_10")

    # Venue / opponent history
    df = add_expanding_mean(df, ["player_name", "venue"], "fantasy_points_v1", "avg_points_at_venue_before")
    df = add_expanding_mean(df, ["player_name", "venue"], "runs_scored", "avg_runs_at_venue_before")
    df = add_expanding_mean(df, ["player_name", "opponent"], "fantasy_points_v1", "avg_points_vs_opponent_before")
    df = add_expanding_mean(df, ["player_name", "opponent"], "wickets", "avg_wickets_vs_opponent_before")

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
        "points_last_10_avg",
        "points_last_5_std",
        "points_last_10_std",
        "points_last_5_max",
        "points_last_10_max",
        "runs_last_1",
        "runs_last_3_avg",
        "runs_last_5_avg",
        "runs_last_10_avg",
        "wickets_last_1",
        "wickets_last_3_avg",
        "wickets_last_5_avg",
        "wickets_last_10_avg",
        "balls_faced_last_1",
        "balls_faced_last_3_avg",
        "balls_faced_last_5_avg",
        "balls_faced_last_10_avg",
        "balls_bowled_last_1",
        "balls_bowled_last_3_avg",
        "balls_bowled_last_5_avg",
        "balls_bowled_last_10_avg",
        "catches_last_5_avg",
        "stumpings_last_5_avg",
        "runouts_last_5_avg",
        "maidens_last_5_avg",
        "batting_active_rate_last_5",
        "bowling_active_rate_last_5",
        "dismissal_rate_last_5",
        "duck_rate_last_10",
        "avg_points_at_venue_before",
        "avg_runs_at_venue_before",
        "avg_points_vs_opponent_before",
        "avg_wickets_vs_opponent_before",
        "fantasy_points_v1",
    ]

    model_df = df[model_cols].copy()
    model_df = model_df.fillna(0)
    model_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", model_df.shape)
    print("\nMissing values:")
    print(model_df.isna().sum().sum())
    print("\nSample rows:")
    print(model_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()