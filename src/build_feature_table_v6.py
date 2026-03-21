"""
Feature table v6: adds 6 high-signal features missing from v4/v5.
Builds on top of model_feature_table_v5.csv (already has 98 features).

New features added:
  1. is_home_ground        — player's team playing at home city
  2. venue_bat_bowl_index  — is this venue historically batter/bowler friendly
  3. season_stage          — league (0) vs playoff (1)
  4. team_win_streak_last5 — team momentum (wins in last 5 matches)
  5. player_pts_vs_this_venue_count — how many times played here before
  6. career_high_score_before — player's personal best before this match
"""
from pathlib import Path
import pandas as pd
import numpy as np

INPUT_FEATURES = Path("output/model_feature_table_v5.csv")
INPUT_RAW      = Path("data/processed/v4_dataset.csv")
OUTPUT_PATH    = Path("output/model_feature_table_v6.csv")

# IPL home city mapping (team → home venue keywords)
HOME_VENUE_MAP = {
    "Mumbai Indians":            ["wankhede", "mumbai"],
    "Chennai Super Kings":       ["chepauk", "chennai", "ma chidambaram"],
    "Royal Challengers Bengaluru":["chinnaswamy", "bengaluru", "bangalore"],
    "Royal Challengers Bangalore":["chinnaswamy", "bengaluru", "bangalore"],
    "Kolkata Knight Riders":     ["eden gardens", "kolkata"],
    "Delhi Capitals":            ["arun jaitley", "feroz shah", "delhi"],
    "Delhi Daredevils":          ["arun jaitley", "feroz shah", "delhi"],
    "Rajasthan Royals":          ["sawai mansingh", "jaipur"],
    "Sunrisers Hyderabad":       ["rajiv gandhi", "hyderabad", "uppal"],
    "Punjab Kings":              ["punjab cricket", "mohali", "chandigarh"],
    "Kings XI Punjab":           ["punjab cricket", "mohali", "chandigarh"],
    "Gujarat Titans":            ["narendra modi", "ahmedabad"],
    "Lucknow Super Giants":      ["atal bihari", "lucknow", "ekana"],
    "Rising Pune Supergiant":    ["maharashtra cricket", "pune"],
    "Pune Warriors":             ["maharashtra cricket", "pune"],
}

def is_home(team: str, venue: str) -> int:
    keywords = HOME_VENUE_MAP.get(team, [])
    venue_lower = str(venue).lower()
    return int(any(kw in venue_lower for kw in keywords))


def add_is_home_ground(df: pd.DataFrame) -> pd.DataFrame:
    df["is_home_ground"] = df.apply(
        lambda r: is_home(r["team"], r["venue"]), axis=1
    )
    return df


def add_venue_bat_bowl_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    venue_bat_bowl_index: rolling avg total runs at venue before this match.
    Higher = batter-friendly venue. Already available as venue_avg_total_runs_before.
    Convert to a z-score so it's scale-free.
    """
    if "venue_avg_total_runs_before" in df.columns:
        mean = df["venue_avg_total_runs_before"].mean()
        std  = df["venue_avg_total_runs_before"].std()
        df["venue_bat_index"] = (df["venue_avg_total_runs_before"] - mean) / (std + 1e-6)
    else:
        df["venue_bat_index"] = 0.0
    return df


def add_season_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    IPL playoff matches are typically the last 4 matches of the season.
    We flag them based on match_date being in May (IPL playoffs historically in May).
    Simple heuristic: match_date month >= 5 AND day >= 20.
    """
    df["match_date"] = pd.to_datetime(df["match_date"])
    df["season_stage"] = (
        (df["match_date"].dt.month >= 5) & (df["match_date"].dt.day >= 20)
    ).astype(int)
    return df


def add_team_win_streak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling win rate of team in last 5 matches before this one.
    Requires match_result per team — approximate from score data:
    if team scored more runs than opponent in that match.
    We use a simpler proxy: count of matches where team_avg_total_runs > opponent_avg_total_runs.
    """
    if "team_avg_total_runs_last_5" in df.columns and "opponent_avg_total_runs_last_5" in df.columns:
        df["team_form_advantage"] = (
            df["team_avg_total_runs_last_5"] - df["opponent_avg_total_runs_last_5"]
        )
    else:
        df["team_form_advantage"] = 0.0
    return df


def add_career_high_before(df: pd.DataFrame) -> pd.DataFrame:
    """
    Player's personal best score in any match before this one.
    Captures ceiling potential — a player who has hit 120 before can do it again.
    """
    df = df.sort_values(["player_name", "match_date", "match_id"]).reset_index(drop=True)
    shifted = df.groupby("player_name")["fantasy_points_v5"].shift(1)
    df["career_high_before"] = (
        shifted.groupby(df["player_name"])
        .expanding()
        .max()
        .reset_index(level=0, drop=True)
    )
    return df


def add_venue_experience(df: pd.DataFrame) -> pd.DataFrame:
    """
    How many times has the player played at this venue before?
    More experience at a ground = less uncertainty.
    """
    df = df.sort_values(["player_name", "match_date", "match_id"]).reset_index(drop=True)
    df["venue_experience"] = df.groupby(["player_name", "venue"]).cumcount()
    return df


def main():
    print("Loading feature table v5...")
    df = pd.read_csv(INPUT_FEATURES, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["player_name", "match_date", "match_id"]).reset_index(drop=True)

    print(f"  Input shape: {df.shape}")

    print("Adding new features...")
    df = add_is_home_ground(df)
    df = add_venue_bat_bowl_index(df)
    df = add_season_stage(df)
    df = add_team_win_streak(df)
    df = add_career_high_before(df)
    df = add_venue_experience(df)

    # Smart fill for new features
    new_cols = ["is_home_ground", "venue_bat_index", "season_stage",
                "team_form_advantage", "career_high_before", "venue_experience"]
    for col in new_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"  Filled {null_count} NaNs in {col}")

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Saved: {OUTPUT_PATH}")
    print(f"   Shape: {df.shape}")
    print(f"   New cols: {new_cols}")
    print(f"\nNew feature stats:")
    print(df[new_cols].describe().round(3))

    print(f"\nHome ground distribution:")
    print(df["is_home_ground"].value_counts())

    print(f"\nSeason stage (playoff) rows: {df['season_stage'].sum()}")
    print(f"Career high before — mean: {df['career_high_before'].mean():.1f}, max: {df['career_high_before'].max():.1f}")
    print(f"Venue experience — mean: {df['venue_experience'].mean():.1f}")


if __name__ == "__main__":
    main()
