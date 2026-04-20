"""
Feature Table v7: Ranking-first approach
========================================
Key changes from v6:
  1. Removes 22+ redundant features (|correlation| > 0.90 pairs)
  2. Adds 6 interaction features that capture matchup-specific signal
  3. Adds consistency + momentum features
  4. Keeps only features that carry independent signal

Reads:  output/model_feature_table_v6.csv
Writes: output/model_feature_table_v7.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np

INPUT_PATH  = Path("output/model_feature_table_v6.csv")
OUTPUT_PATH = Path("output/model_feature_table_v7.csv")

TARGET = "fantasy_points_v5"
EXCLUDE = {"match_id", "match_date", "player_name", "team", TARGET,
           "fantasy_points_v4", "fantasy_points_v3", "fantasy_points_v1"}
CAT_COLS = ["season", "venue", "opponent", "player_role_platform"]


def remove_redundant_features(df: pd.DataFrame) -> list:
    all_feats = [c for c in df.columns
                 if c not in EXCLUDE
                 and c not in ["match_id", "match_date", "player_name", "team"]
                 and c not in CAT_COLS]
    numeric_feats = [c for c in all_feats
                     if df[c].dtype in ["float64", "int64", "float32"]]

    print(f"  Computing correlation matrix for {len(numeric_feats)} numeric features...")
    corr_matrix = df[numeric_feats].corr()

    to_drop = set()
    pairs_found = 0
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.90:
                ci = corr_matrix.columns[i]
                cj = corr_matrix.columns[j]
                ri = abs(df[ci].corr(df[TARGET]))
                rj = abs(df[cj].corr(df[TARGET]))
                drop = cj if ri >= rj else ci
                to_drop.add(drop)
                pairs_found += 1

    print(f"  Found {pairs_found} redundant pairs -> dropping {len(to_drop)} features:")
    for c in sorted(to_drop):
        print(f"    x {c}")

    return list(to_drop)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # venue_bat_index may have been dropped as redundant — use source column
    if "venue_bat_index" in df.columns:
        venue_signal = df["venue_bat_index"].fillna(0)
    elif "venue_avg_total_runs_before" in df.columns:
        v = df["venue_avg_total_runs_before"]
        venue_signal = ((v - v.mean()) / (v.std() + 1e-6)).fillna(0)
    else:
        venue_signal = 0

    df["form_x_venue"] = df["points_last_5_avg"].fillna(0) * venue_signal

    df["bat_form_x_opp_weakness"] = (
        df["runs_scored_last_5_avg"].fillna(0)
        * df["opponent_avg_wickets_lost_last_5"].fillna(0)
    )

    df["bowl_form_x_opp_runs"] = (
        df["wickets_last_5_avg"].fillna(0)
        * df["opponent_avg_total_runs_last_5"].fillna(0)
    )

    df["consistency_score"] = (
        df["points_last_5_avg"].fillna(0)
        / (df["points_last_5_std"].fillna(1) + 1)
    )

    df["momentum"] = (
        df["points_last_1"].fillna(0)
        - df["points_last_5_avg"].fillna(0)
    )

    bat_rate = df["actually_batted_rate_last_5"].fillna(0)
    bowl_rate = df.get("actually_bowled_rate_last_5", pd.Series(0, index=df.index)).fillna(0)
    df["dual_threat"] = ((bat_rate > 0.5) & (bowl_rate > 0.5)).astype(int)

    return df


def main():
    print("=" * 60)
    print("  BUILD FEATURE TABLE v7")
    print("=" * 60)

    print(f"\nLoading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["match_date", "match_id", "player_name"]).reset_index(drop=True)
    print(f"  Input: {df.shape[0]} rows x {df.shape[1]} cols")

    print("\n[1/3] Removing redundant features...")
    dropped = remove_redundant_features(df)
    df = df.drop(columns=[c for c in dropped if c in df.columns])
    print(f"  After cleanup: {df.shape[1]} cols")

    print("\n[2/3] Adding interaction features...")
    df = add_interaction_features(df)

    new_feats = ["form_x_venue", "bat_form_x_opp_weakness", "bowl_form_x_opp_runs",
                 "consistency_score", "momentum", "dual_threat"]

    for col in new_feats:
        nans = df[col].isna().sum()
        if nans > 0:
            df[col] = df[col].fillna(0)
            print(f"  Filled {nans} NaNs in {col}")

    print("\n[3/3] Saving...")
    df.to_csv(OUTPUT_PATH, index=False)

    feature_cols = [c for c in df.columns
                    if c not in EXCLUDE
                    and c not in ["match_id", "match_date", "player_name", "team"]]

    print(f"\n  Saved: {OUTPUT_PATH}")
    print(f"   Shape: {df.shape}")
    print(f"   Feature count: {len(feature_cols)}")
    print(f"   Dropped (redundant): {len(dropped)}")
    print(f"   Added (interactions): {len(new_feats)}")

    print(f"\nNew feature stats:")
    print(df[new_feats].describe().round(3).to_string())

    print(f"\nNew feature correlations with {TARGET}:")
    for f in new_feats:
        r = df[f].corr(df[TARGET])
        print(f"  {r:+.4f}  {f}")


if __name__ == "__main__":
    main()
