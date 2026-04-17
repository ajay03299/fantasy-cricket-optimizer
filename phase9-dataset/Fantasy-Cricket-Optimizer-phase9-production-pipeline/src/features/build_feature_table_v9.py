"""
Feature Table v9: Phase x bowler type + v7 cleanup + opponent quality
=====================================================================
Three stacked upgrades:
  A) Apply v7 redundancy removal (drop 22 features with |r|>0.90)
  B) Add 6 phase x bowler type features from ball-by-ball
  C) Add opponent bowling quality index

Reads:  output/model_feature_table_v8.csv + ball-by-ball data
Writes: output/model_feature_table_v9.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

INPUT_FT  = Path("output/model_feature_table_v8.csv")
INPUT_BBB = Path("data/all_ball_by_ball_data.csv")
OUTPUT    = Path("output/model_feature_table_v9.csv")

TARGET = "fantasy_points_v5"
EXCLUDE = {"match_id", "match_date", "player_name", "team", TARGET,
           "fantasy_points_v4", "fantasy_points_v3", "fantasy_points_v1"}
CAT_COLS = ["season", "venue", "opponent", "player_role_platform"]


def classify_bowler(bt):
    bt = str(bt).lower()
    if any(k in bt for k in ['fast', 'medium fast', 'fast medium']):
        return 'PACE'
    elif any(k in bt for k in ['offbreak', 'legbreak', 'googly', 'orthodox', 'wrist spin']):
        return 'SPIN'
    return 'MEDIUM'


def remove_redundant_features(df):
    """Same v7 cleanup — drop features with |r|>0.90 within each pair."""
    print("  [A] Removing redundant features...")
    all_feats = [c for c in df.columns
                 if c not in EXCLUDE and c not in ["match_id", "match_date", "player_name", "team"]
                 and c not in CAT_COLS]
    numeric_feats = [c for c in all_feats if df[c].dtype in ["float64", "int64", "float32"]]

    corr_matrix = df[numeric_feats].corr()
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.90:
                ci, cj = corr_matrix.columns[i], corr_matrix.columns[j]
                ri = abs(df[ci].corr(df[TARGET]))
                rj = abs(df[cj].corr(df[TARGET]))
                to_drop.add(cj if ri >= rj else ci)

    df = df.drop(columns=[c for c in to_drop if c in df.columns])
    print(f"  Dropped {len(to_drop)} redundant features -> {df.shape[1]} cols remain")
    return df


def build_phase_bowler_features(bbb, ft):
    """
    Phase x bowler type SR for each batter — BEFORE each match.
    6 features: PP/MID/DEATH x PACE/SPIN
    """
    print("  [B] Building phase x bowler type features...")
    bbb = bbb.copy()
    bbb['bowler_cat'] = bbb['bowler_type'].apply(classify_bowler)
    bbb['phase'] = pd.cut(bbb['over_number'], bins=[-1, 5, 15, 20], labels=['pp', 'mid', 'death'])

    match_dates = ft[['match_id', 'match_date']].drop_duplicates()
    bbb = bbb.merge(match_dates, on='match_id', how='left')
    bbb['match_date'] = pd.to_datetime(bbb['match_date'])

    # Only PACE and SPIN (MEDIUM has too little data for reliable splits)
    results = []
    for phase in ['pp', 'mid', 'death']:
        for btype in ['PACE', 'SPIN']:
            col_name = f'batter_sr_{btype.lower()}_in_{phase}_before'
            sub = bbb[(bbb['phase'] == phase) & (bbb['bowler_cat'] == btype)]

            per_match = sub.groupby(['match_id', 'match_date', 'batter']).agg(
                balls=('batter_runs', 'count'),
                runs=('batter_runs', 'sum'),
            ).reset_index()
            per_match = per_match.sort_values(['batter', 'match_date', 'match_id'])

            # Cumulative BEFORE this match
            per_match['cum_balls'] = per_match.groupby('batter')['balls'].cumsum().shift(1)
            per_match['cum_runs'] = per_match.groupby('batter')['runs'].cumsum().shift(1)
            per_match[col_name] = per_match['cum_runs'] / per_match['cum_balls'].replace(0, np.nan) * 100

            results.append(
                per_match[['match_id', 'batter', col_name]]
                .rename(columns={'batter': 'player_name'})
            )

    # Merge all 6 features
    merged = results[0]
    for r in results[1:]:
        merged = merged.merge(r, on=['match_id', 'player_name'], how='outer')

    print(f"  Built 6 phase x bowler type features for {merged['player_name'].nunique()} batters")
    return merged


def build_opponent_bowling_quality(bbb, ft):
    """
    Opponent bowling quality index: avg economy of bowlers in the opponent team
    over their last 5 matches. Lower = tougher attack.
    """
    print("  [C] Building opponent bowling quality index...")
    match_dates = ft[['match_id', 'match_date']].drop_duplicates()
    bbb_m = bbb.merge(match_dates, on='match_id', how='left')
    bbb_m['match_date'] = pd.to_datetime(bbb_m['match_date'])

    # Per bowler per match: economy rate
    bowler_match = bbb_m.groupby(['match_id', 'match_date', 'team_bowling', 'bowler']).agg(
        balls=('total_runs', 'count'),
        runs_conceded=('total_runs', 'sum'),
        wickets=('is_wicket', 'sum'),
    ).reset_index()
    bowler_match['overs'] = bowler_match['balls'] / 6
    bowler_match['economy'] = bowler_match['runs_conceded'] / bowler_match['overs'].replace(0, np.nan)
    bowler_match['bowling_sr'] = bowler_match['balls'] / bowler_match['wickets'].replace(0, np.nan)

    # Per team per match: avg economy of their bowling attack
    team_bowling = bowler_match.groupby(['match_id', 'match_date', 'team_bowling']).agg(
        avg_economy=('economy', 'mean'),
        avg_bowling_sr=('bowling_sr', 'mean'),
        num_bowlers=('bowler', 'nunique'),
    ).reset_index()

    team_bowling = team_bowling.sort_values(['team_bowling', 'match_date'])

    # Rolling last 5 matches
    for col in ['avg_economy', 'avg_bowling_sr']:
        team_bowling[f'{col}_rolling5'] = (
            team_bowling.groupby('team_bowling')[col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

    result = team_bowling[['match_id', 'team_bowling',
                           'avg_economy_rolling5', 'avg_bowling_sr_rolling5']].copy()
    result = result.rename(columns={
        'team_bowling': 'opponent_team',
        'avg_economy_rolling5': 'opp_bowl_quality_eco',
        'avg_bowling_sr_rolling5': 'opp_bowl_quality_sr',
    })

    print(f"  Built bowling quality index for {result['opponent_team'].nunique()} teams")
    return result


def main():
    print("=" * 60)
    print("  BUILD FEATURE TABLE v9")
    print("  Phase x bowler type + redundancy cleanup + opp quality")
    print("=" * 60)

    ft = pd.read_csv(INPUT_FT, low_memory=False)
    ft['match_date'] = pd.to_datetime(ft['match_date'])
    ft = ft.sort_values(['match_date', 'match_id', 'player_name']).reset_index(drop=True)
    # Remove any duplicates from v8
    ft = ft.drop_duplicates(subset=['match_id', 'player_name'], keep='first')
    print(f"\n  Input: {ft.shape[0]} rows x {ft.shape[1]} cols")

    bbb = pd.read_csv(INPUT_BBB, low_memory=False)
    print(f"  Ball-by-ball: {len(bbb)} deliveries")

    # A) Remove redundant features
    ft = remove_redundant_features(ft)

    # B) Phase x bowler type features
    phase_feats = build_phase_bowler_features(bbb, ft)
    ft = ft.merge(phase_feats, on=['match_id', 'player_name'], how='left')
    print(f"  After phase features merge: {ft.shape}")

    # C) Opponent bowling quality
    opp_quality = build_opponent_bowling_quality(bbb, ft)

    # Need to map opponent to team_id
    teams = pd.read_csv(Path("data/all_teams_data.csv"))
    aliases = pd.read_csv(Path("data/all_team_aliases.csv"))
    name_to_id = dict(zip(teams['team_name'], teams['team_id']))
    for _, row in aliases.iterrows():
        name_to_id[row['alias']] = row['team_id']

    if 'opponent' in ft.columns:
        ft['opp_team_id'] = ft['opponent'].map(name_to_id)
        ft = ft.merge(opp_quality, left_on=['match_id', 'opp_team_id'],
                       right_on=['match_id', 'opponent_team'], how='left')
        ft = ft.drop(columns=['opp_team_id', 'opponent_team'], errors='ignore')

    print(f"  After opponent quality merge: {ft.shape}")

    # Fill NaNs
    new_cols = [c for c in ft.columns if 'batter_sr_' in c and '_in_' in c] + \
               ['opp_bowl_quality_eco', 'opp_bowl_quality_sr']
    existing = [c for c in new_cols if c in ft.columns]
    for col in existing:
        nans = ft[col].isna().sum()
        if nans > 0:
            ft[col] = ft[col].fillna(ft[col].median() if pd.notna(ft[col].median()) else 0)
            print(f"  Filled {nans} NaNs in {col}")

    # Remove any new duplicates
    ft = ft.drop_duplicates(subset=['match_id', 'player_name'], keep='first')

    ft.to_csv(OUTPUT, index=False)

    print(f"\n{'='*60}")
    print(f"  Saved: {OUTPUT}")
    print(f"  Shape: {ft.shape}")
    print(f"  New features: {existing}")
    print(f"{'='*60}")

    print(f"\nNew feature correlations with {TARGET}:")
    for f in existing:
        if f in ft.columns:
            r = ft[f].corr(ft[TARGET])
            print(f"  {r:+.4f}  {f}")


if __name__ == "__main__":
    main()
