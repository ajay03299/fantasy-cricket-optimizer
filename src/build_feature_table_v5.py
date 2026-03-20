from pathlib import Path
import pandas as pd
import numpy as np

INPUT_PATH  = Path("output/player_match_fantasy_v5.csv")
OUTPUT_PATH = Path("output/model_feature_table_v5.csv")


# ── Rolling helpers (always shift(1) to avoid leakage) ────────────────────────

def roll_mean(df, group_col, value_col, window, out_col):
    shifted = df.groupby(group_col)[value_col].shift(1)
    df[out_col] = (
        shifted.groupby(df[group_col])
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df

def roll_std(df, group_col, value_col, window, out_col):
    shifted = df.groupby(group_col)[value_col].shift(1)
    df[out_col] = (
        shifted.groupby(df[group_col])
        .rolling(window, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )
    return df

def roll_rate(df, group_col, condition_series, window, out_col):
    shifted = condition_series.groupby(df[group_col]).shift(1)
    df[out_col] = (
        shifted.groupby(df[group_col])
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df

def expanding_mean(df, group_cols, value_col, out_col):
    shifted = df.groupby(group_cols)[value_col].shift(1)
    df[out_col] = (
        shifted.groupby([df[c] for c in group_cols])
        .expanding()
        .mean()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    return df


# ── Smart imputation — replaces the broken fillna(0) ─────────────────────────

def smart_impute(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Cold-start rows get role-based median instead of 0.
    fillna(0) was making debutants look like players who always score 0,
    which tanks the model's ability to differentiate unknowns from known bad players.
    """
    print("  Running smart imputation...")
    experienced = df[df["matches_played_before"] >= 5]
    role_medians = (
        experienced
        .groupby("player_role_platform")[feature_cols]
        .median()
    )
    for col in feature_cols:
        for role in role_medians.index:
            if role in role_medians.index and col in role_medians.columns:
                med = role_medians.loc[role, col]
                mask = df[col].isna() & (df["player_role_platform"] == role)
                df.loc[mask, col] = med

    # Any truly remaining NaNs → global median
    remaining_nulls = df[feature_cols].isna().sum().sum()
    if remaining_nulls > 0:
        print(f"  Filling {remaining_nulls} remaining NaNs with global median")
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df


# ── NEW: Form slope — is the player trending up or down? ─────────────────────

def add_form_slope(df: pd.DataFrame) -> pd.DataFrame:
    def _slope(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr), dtype=float)
        return float(np.polyfit(x, arr, 1)[0])

    shifted = df.groupby("player_name")["fantasy_points_v5"].shift(1)
    df["form_slope_last5"] = (
        shifted.groupby(df["player_name"])
        .rolling(5, min_periods=2)
        .apply(_slope, raw=True)
        .reset_index(level=0, drop=True)
    )
    return df


# ── NEW: Days since last match — freshness/rust signal ───────────────────────

def add_days_since_last_match(df: pd.DataFrame) -> pd.DataFrame:
    prev_date = df.groupby("player_name")["match_date"].shift(1)
    df["days_since_last_match"] = (df["match_date"] - prev_date).dt.days
    df["days_since_last_match"] = df["days_since_last_match"].clip(upper=180)
    return df


# ── NEW: Points above role average — relative form signal ────────────────────

def add_relative_form(df: pd.DataFrame) -> pd.DataFrame:
    role_med = df.groupby("player_role_platform")["points_last_5_avg"].transform("median")
    df["pts_above_role_avg"] = df["points_last_5_avg"] - role_med
    return df


def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["player_name", "match_date", "match_id"]).reset_index(drop=True)

    print(f"Rows: {len(df)} | Unique players: {df['player_name'].nunique()}")

    # ── Match count before this match ────────────────────────────────────────
    df["matches_played_before"] = df.groupby("player_name").cumcount()

    # ── Days since last match ─────────────────────────────────────────────────
    df = add_days_since_last_match(df)

    # ── Core fantasy points history ───────────────────────────────────────────
    print("Building rolling fantasy point features...")
    df = expanding_mean(df, ["player_name"], "fantasy_points_v5", "career_avg_points_before")
    df = roll_mean(df, "player_name", "fantasy_points_v5", 1,  "points_last_1")
    df = roll_mean(df, "player_name", "fantasy_points_v5", 3,  "points_last_3_avg")
    df = roll_mean(df, "player_name", "fantasy_points_v5", 5,  "points_last_5_avg")
    df = roll_mean(df, "player_name", "fantasy_points_v5", 10, "points_last_10_avg")
    df = roll_std( df, "player_name", "fantasy_points_v5", 5,  "points_last_5_std")
    df = roll_std( df, "player_name", "fantasy_points_v5", 10, "points_last_10_std")

    # ── Form slope (NEW) ──────────────────────────────────────────────────────
    print("Building form slope...")
    df = add_form_slope(df)

    # ── Role / usage rates ────────────────────────────────────────────────────
    print("Building role/usage features...")
    df = roll_mean(df, "player_name", "batting_position_actual", 5, "bat_pos_last_5_avg")
    for flag_col, out_col in [
        ("is_opener",         "opener_rate_last_5"),
        ("is_top_order",      "top_order_rate_last_5"),
        ("is_finisher",       "finisher_rate_last_5"),
        ("actually_batted",   "actually_batted_rate_last_5"),
        ("actually_bowled",   "actually_bowled_rate_last_5"),
        ("did_bowl_full_quota","full_quota_rate_last_5"),
    ]:
        if flag_col in df.columns:
            df = roll_rate(df, "player_name", df[flag_col].astype(float), 5, out_col)

    # ── Batting phase features ─────────────────────────────────────────────────
    print("Building batting phase features...")
    bat_cols = [
        "runs_scored", "balls_faced",
        "powerplay_runs", "middle_overs_runs", "death_overs_runs",
        "powerplay_balls_faced", "middle_overs_balls_faced", "death_overs_balls_faced",
        "dot_ball_percentage_batting", "boundary_ball_percentage",
        "strike_rotation_rate", "runs_share_of_team", "balls_faced_share_of_team",
    ]
    for col in bat_cols:
        if col in df.columns:
            df = roll_mean(df, "player_name", col, 5, f"{col}_last_5_avg")

    # ── Bowling phase features ─────────────────────────────────────────────────
    print("Building bowling phase features...")
    bowl_cols = [
        "wickets", "balls_bowled", "overs_bowled", "maidens",
        "powerplay_balls_bowled", "middle_overs_balls_bowled", "death_overs_balls_bowled",
        "powerplay_runs_conceded", "middle_overs_runs_conceded", "death_overs_runs_conceded",
        "powerplay_wickets", "middle_overs_wickets", "death_overs_wickets",
        "dot_ball_percentage_bowling", "boundary_conceded_rate", "overs_bowled_share_of_team",
        "lbw_wickets", "bowled_wickets",
    ]
    for col in bowl_cols:
        if col in df.columns:
            df = roll_mean(df, "player_name", col, 5, f"{col}_last_5_avg")

    # ── Fielding features ─────────────────────────────────────────────────────
    print("Building fielding features...")
    field_cols = ["catches", "stumpings", "runout_direct", "runout_assist", "catch_and_bowled_flag"]
    for col in field_cols:
        if col in df.columns:
            df = roll_mean(df, "player_name", col, 5, f"{col}_last_5_avg")

    # ── Venue / opponent contextual history ───────────────────────────────────
    print("Building venue/opponent history...")
    df = expanding_mean(df, ["player_name", "venue"],    "fantasy_points_v5", "avg_points_at_venue_before")
    df = expanding_mean(df, ["player_name", "venue"],    "runs_scored",        "avg_runs_at_venue_before")
    df = expanding_mean(df, ["player_name", "opponent"], "fantasy_points_v5", "avg_points_vs_opponent_before")
    df = expanding_mean(df, ["player_name", "opponent"], "wickets",            "avg_wickets_vs_opponent_before")

    # ── Relative form (NEW — added after points_last_5_avg exists) ────────────
    df = add_relative_form(df)

    # ── Assemble feature list ─────────────────────────────────────────────────
    context_cols = [
        "venue_avg_total_runs_before", "venue_avg_wickets_before",
        "venue_avg_run_rate_before", "venue_avg_powerplay_runs_before",
        "venue_avg_middle_overs_runs_before", "venue_avg_death_overs_runs_before",
        "venue_avg_powerplay_wickets_before", "venue_avg_middle_overs_wickets_before",
        "venue_avg_death_overs_wickets_before",
        "team_avg_total_runs_last_5", "team_avg_wickets_lost_last_5",
        "team_avg_run_rate_last_5", "team_avg_powerplay_runs_last_5",
        "team_avg_middle_overs_runs_last_5", "team_avg_death_overs_runs_last_5",
        "team_avg_powerplay_wickets_lost_last_5", "team_avg_middle_overs_wickets_lost_last_5",
        "team_avg_death_overs_wickets_lost_last_5",
        "opponent_avg_wickets_taken_last_5", "opponent_avg_economy_last_5",
        "opponent_avg_powerplay_runs_conceded_last_5", "opponent_avg_middle_overs_runs_conceded_last_5",
        "opponent_avg_death_overs_runs_conceded_last_5", "opponent_avg_powerplay_wickets_last_5",
        "opponent_avg_middle_overs_wickets_last_5", "opponent_avg_death_overs_wickets_last_5",
        "opponent_avg_total_runs_last_5", "opponent_avg_wickets_lost_last_5",
        "opponent_avg_powerplay_runs_last_5", "opponent_avg_middle_overs_runs_last_5",
        "opponent_avg_death_overs_runs_last_5",
    ]

    id_cols = ["match_id", "match_date", "season", "player_name", "team",
               "opponent", "venue", "player_role_platform"]

    rolling_feature_cols = (
        ["matches_played_before", "days_since_last_match",
         "career_avg_points_before",
         "points_last_1", "points_last_3_avg", "points_last_5_avg",
         "points_last_10_avg", "points_last_5_std", "points_last_10_std",
         "form_slope_last5", "pts_above_role_avg",
         "bat_pos_last_5_avg", "opener_rate_last_5", "top_order_rate_last_5",
         "finisher_rate_last_5", "actually_batted_rate_last_5",
         "actually_bowled_rate_last_5", "full_quota_rate_last_5"]
        + [f"{c}_last_5_avg" for c in bat_cols  if c in df.columns]
        + [f"{c}_last_5_avg" for c in bowl_cols if c in df.columns]
        + [f"{c}_last_5_avg" for c in field_cols if c in df.columns]
        + ["avg_points_at_venue_before", "avg_runs_at_venue_before",
           "avg_points_vs_opponent_before", "avg_wickets_vs_opponent_before"]
        + [c for c in context_cols if c in df.columns]
    )

    # Deduplicate while preserving order
    seen = set()
    rolling_feature_cols = [c for c in rolling_feature_cols
                             if c in df.columns and not (c in seen or seen.add(c))]

    # ── Smart imputation (replaces fillna(0)) ─────────────────────────────────
    df = smart_impute(df, rolling_feature_cols)

    # ── Save ─────────────────────────────────────────────────────────────────
    model_cols = id_cols + rolling_feature_cols + ["fantasy_points_v5"]
    model_df = df[[c for c in model_cols if c in df.columns]].copy()
    model_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Saved: {OUTPUT_PATH}")
    print(f"   Shape: {model_df.shape}")
    print(f"   Features: {len(rolling_feature_cols)}")
    print(f"   Remaining NaNs: {model_df.isna().sum().sum()}")
    print(f"\nTarget distribution:")
    print(model_df["fantasy_points_v5"].describe().round(2))


if __name__ == "__main__":
    main()
