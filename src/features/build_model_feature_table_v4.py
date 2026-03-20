import pandas as pd
from pathlib import Path

INPUT_PATH = Path("output/player_match_fantasy_v4.csv")
OUTPUT_PATH = Path("output/model_feature_table_v4.csv")


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
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["player_name", "match_date", "match_id"]).reset_index(drop=True)

    df["matches_played_before"] = df.groupby("player_name").cumcount()

    # Core score history
    df = add_expanding_mean(df, ["player_name"], "fantasy_points_v4", "career_avg_points_before")
    df = add_shifted_group_roll_mean(df, "player_name", "fantasy_points_v4", 1, "points_last_1")
    df = add_shifted_group_roll_mean(df, "player_name", "fantasy_points_v4", 3, "points_last_3_avg")
    df = add_shifted_group_roll_mean(df, "player_name", "fantasy_points_v4", 5, "points_last_5_avg")
    df = add_shifted_group_roll_mean(df, "player_name", "fantasy_points_v4", 10, "points_last_10_avg")
    df = add_shifted_group_roll_std(df, "player_name", "fantasy_points_v4", 5, "points_last_5_std")

    # Role / usage
    df = add_shifted_group_roll_mean(df, "player_name", "batting_position_actual", 5, "bat_pos_last_5_avg")
    df = add_shifted_group_roll_rate(df, "player_name", df["is_opener"].astype(float), 5, "opener_rate_last_5")
    df = add_shifted_group_roll_rate(df, "player_name", df["is_top_order"].astype(float), 5, "top_order_rate_last_5")
    df = add_shifted_group_roll_rate(df, "player_name", df["is_finisher"].astype(float), 5, "finisher_rate_last_5")
    df = add_shifted_group_roll_rate(df, "player_name", df["actually_batted"].astype(float), 5, "actually_batted_rate_last_5")
    df = add_shifted_group_roll_rate(df, "player_name", df["actually_bowled"].astype(float), 5, "actually_bowled_rate_last_5")
    df = add_shifted_group_roll_rate(df, "player_name", df["did_bowl_full_quota"].astype(float), 5, "full_quota_rate_last_5")

    # Player raw-enriched rolling features
    player_cols = [
        "runs_scored",
        "balls_faced",
        "powerplay_runs",
        "middle_overs_runs",
        "death_overs_runs",
        "powerplay_balls_faced",
        "middle_overs_balls_faced",
        "death_overs_balls_faced",
        "dot_ball_percentage_batting",
        "boundary_ball_percentage",
        "strike_rotation_rate",
        "runs_share_of_team",
        "balls_faced_share_of_team",
        "wickets",
        "balls_bowled",
        "overs_bowled",
        "maidens",
        "powerplay_balls_bowled",
        "middle_overs_balls_bowled",
        "death_overs_balls_bowled",
        "powerplay_runs_conceded",
        "middle_overs_runs_conceded",
        "death_overs_runs_conceded",
        "powerplay_wickets",
        "middle_overs_wickets",
        "death_overs_wickets",
        "dot_ball_percentage_bowling",
        "boundary_conceded_rate",
        "overs_bowled_share_of_team",
        "catches",
        "stumpings",
        "runout_direct",
        "runout_assist",
        "catch_and_bowled_flag",
    ]

    for col in player_cols:
        df = add_shifted_group_roll_mean(df, "player_name", col, 5, f"{col}_last_5_avg")

    # Opponent / venue player history
    df = add_expanding_mean(df, ["player_name", "venue"], "fantasy_points_v4", "avg_points_at_venue_before")
    df = add_expanding_mean(df, ["player_name", "opponent"], "fantasy_points_v4", "avg_points_vs_opponent_before")
    df = add_expanding_mean(df, ["player_name", "opponent"], "wickets", "avg_wickets_vs_opponent_before")
    df = add_expanding_mean(df, ["player_name", "venue"], "runs_scored", "avg_runs_at_venue_before")

    # Keep match-context columns directly
    context_cols = [
        "venue_avg_total_runs_before",
        "venue_avg_wickets_before",
        "venue_avg_run_rate_before",
        "venue_avg_powerplay_runs_before",
        "venue_avg_middle_overs_runs_before",
        "venue_avg_death_overs_runs_before",
        "venue_avg_powerplay_wickets_before",
        "venue_avg_middle_overs_wickets_before",
        "venue_avg_death_overs_wickets_before",
        "team_avg_total_runs_last_5",
        "team_avg_wickets_lost_last_5",
        "team_avg_run_rate_last_5",
        "team_avg_powerplay_runs_last_5",
        "team_avg_middle_overs_runs_last_5",
        "team_avg_death_overs_runs_last_5",
        "team_avg_powerplay_wickets_lost_last_5",
        "team_avg_middle_overs_wickets_lost_last_5",
        "team_avg_death_overs_wickets_lost_last_5",
        "opponent_avg_wickets_taken_last_5",
        "opponent_avg_economy_last_5",
        "opponent_avg_powerplay_runs_conceded_last_5",
        "opponent_avg_middle_overs_runs_conceded_last_5",
        "opponent_avg_death_overs_runs_conceded_last_5",
        "opponent_avg_powerplay_wickets_last_5",
        "opponent_avg_middle_overs_wickets_last_5",
        "opponent_avg_death_overs_wickets_last_5",
        "opponent_avg_total_runs_last_5",
        "opponent_avg_wickets_lost_last_5",
        "opponent_avg_powerplay_runs_last_5",
        "opponent_avg_middle_overs_runs_last_5",
        "opponent_avg_death_overs_runs_last_5",
    ]

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
        "bat_pos_last_5_avg",
        "opener_rate_last_5",
        "top_order_rate_last_5",
        "finisher_rate_last_5",
        "actually_batted_rate_last_5",
        "actually_bowled_rate_last_5",
        "full_quota_rate_last_5",
    ] + [f"{c}_last_5_avg" for c in player_cols] + [
        "avg_points_at_venue_before",
        "avg_points_vs_opponent_before",
        "avg_wickets_vs_opponent_before",
        "avg_runs_at_venue_before",
    ] + context_cols + [
        "fantasy_points_v4",
    ]

    model_df = df[model_cols].copy()
    model_df = model_df.fillna(0)
    model_df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Shape:", model_df.shape)
    print("\nMissing values total:", model_df.isna().sum().sum())
    print("\nSample rows:")
    print(model_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()