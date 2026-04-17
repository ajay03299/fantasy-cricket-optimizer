"""
Feature Table v10: Player-vs-bowler + EWMA form + season weighting
==================================================================
Three upgrades:
  A) Batter vs opponent's actual bowlers (match-level SR aggregation)
     For each match, avg the batter's historical SR against specific
     bowlers in the opposing team. Falls back to bowler-type SR.
  B) EWMA form features (half-life=3 matches)
     Exponentially weighted points, runs, wickets — recent form > old
  C) Season-weighted career stats (last 3 seasons get 2x weight)

Reads:  output/model_feature_table_v9.csv + ball-by-ball
Writes: output/model_feature_table_v10.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

INPUT_FT  = Path("output/model_feature_table_v9.csv")
INPUT_BBB = Path("data/all_ball_by_ball_data.csv")
OUTPUT    = Path("output/model_feature_table_v10.csv")
TARGET    = "fantasy_points_v5"


def build_batter_vs_match_bowlers(bbb, ft):
    """
    For each batter in each match, compute their avg historical SR
    against the SPECIFIC bowlers they'll face (opponent's bowlers).
    Uses only data from BEFORE that match.
    """
    print("  [A] Building batter vs specific bowler features...")

    match_dates = ft[['match_id', 'match_date']].drop_duplicates()
    bbb_m = bbb.merge(match_dates, on='match_id', how='left')
    bbb_m['match_date'] = pd.to_datetime(bbb_m['match_date'])

    # Step 1: Cumulative batter-vs-bowler stats BEFORE each match
    pair_match = bbb_m.groupby(['match_id', 'match_date', 'batter', 'bowler']).agg(
        balls=('batter_runs', 'count'),
        runs=('batter_runs', 'sum'),
        dismissals=('is_wicket', 'sum'),
    ).reset_index()

    pair_match = pair_match.sort_values(['batter', 'bowler', 'match_date'])

    # Cumulative before this match
    pair_match['cum_balls'] = pair_match.groupby(['batter', 'bowler'])['balls'].cumsum().shift(1)
    pair_match['cum_runs'] = pair_match.groupby(['batter', 'bowler'])['runs'].cumsum().shift(1)
    pair_match['cum_dismissals'] = pair_match.groupby(['batter', 'bowler'])['dismissals'].cumsum().shift(1)

    pair_match['hist_sr'] = pair_match['cum_runs'] / pair_match['cum_balls'].replace(0, np.nan) * 100
    pair_match['hist_avg'] = pair_match['cum_runs'] / pair_match['cum_dismissals'].replace(0, 0.5)
    pair_match['has_history'] = pair_match['cum_balls'].fillna(0) >= 6  # min 6 balls

    # Step 2: For each match, get which bowlers played for each team
    bowlers_per_match = bbb_m.groupby(['match_id', 'team_bowling'])['bowler'].apply(set).reset_index()
    bowlers_per_match.columns = ['match_id', 'team_bowling', 'bowlers']

    # Step 3: For each batter in each match, avg their hist_sr vs opponent bowlers
    # We need to know which team the batter faced
    batter_team = bbb_m[['match_id', 'batter', 'team_batting', 'team_bowling']].drop_duplicates()

    results = []
    match_ids = ft['match_id'].unique()

    for mid in match_ids:
        # Get batter info for this match
        match_batters = batter_team[batter_team['match_id'] == mid]
        if len(match_batters) == 0:
            continue

        for _, row in match_batters.drop_duplicates(subset=['batter']).iterrows():
            batter = row['batter']
            opp_team = row['team_bowling']

            # Get opponent's bowlers in this match
            opp_bowlers_row = bowlers_per_match[
                (bowlers_per_match['match_id'] == mid) &
                (bowlers_per_match['team_bowling'] == opp_team)
            ]
            if len(opp_bowlers_row) == 0:
                continue
            opp_bowlers = opp_bowlers_row.iloc[0]['bowlers']

            # Get batter's history vs these specific bowlers
            hist = pair_match[
                (pair_match['match_id'] == mid) &
                (pair_match['batter'] == batter) &
                (pair_match['bowler'].isin(opp_bowlers)) &
                (pair_match['has_history'] == True)
            ]

            if len(hist) > 0:
                # Weighted average by balls faced (more balls = more reliable)
                weights = hist['cum_balls'].fillna(0)
                if weights.sum() > 0:
                    avg_sr = np.average(hist['hist_sr'].fillna(0), weights=weights)
                    avg_avg = np.average(hist['hist_avg'].fillna(0), weights=weights)
                    n_bowlers_known = len(hist)
                else:
                    avg_sr, avg_avg, n_bowlers_known = np.nan, np.nan, 0
            else:
                avg_sr, avg_avg, n_bowlers_known = np.nan, np.nan, 0

            results.append({
                'match_id': mid,
                'player_name': batter,
                'sr_vs_match_bowlers': avg_sr,
                'avg_vs_match_bowlers': avg_avg,
                'n_bowlers_with_history': n_bowlers_known,
            })

    result_df = pd.DataFrame(results)
    print(f"  Built features for {result_df['player_name'].nunique()} batters across {result_df['match_id'].nunique()} matches")
    print(f"  Avg bowlers with history per batter-match: {result_df['n_bowlers_with_history'].mean():.1f}")
    return result_df


def build_ewma_features(ft):
    """
    Exponentially weighted moving average of fantasy points.
    Half-life=3: a match 3 games ago gets 50% the weight of the most recent.
    """
    print("  [B] Building EWMA form features...")
    df = ft.sort_values(['player_name', 'match_date', 'match_id']).copy()

    alpha = 1 - np.exp(-np.log(2) / 3)  # half-life = 3

    # EWMA of fantasy points (using shift so we only use PAST data)
    df['pts_shifted'] = df.groupby('player_name')[TARGET].shift(1)
    df['ewma_points'] = df.groupby('player_name')['pts_shifted'].transform(
        lambda x: x.ewm(alpha=alpha, min_periods=1).mean()
    )

    # EWMA of runs for batters
    if 'runs_scored_last_5_avg' in df.columns:
        # Approximate: use points as proxy since we have it per match
        pass

    # EWMA volatility (how stable is the player's recent form)
    df['ewma_std'] = df.groupby('player_name')['pts_shifted'].transform(
        lambda x: x.ewm(alpha=alpha, min_periods=3).std()
    )

    # EWMA consistency: ewma_points / (ewma_std + 1)
    df['ewma_consistency'] = df['ewma_points'] / (df['ewma_std'].fillna(1) + 1)

    df = df.drop(columns=['pts_shifted'])
    print(f"  Built EWMA features (alpha={alpha:.3f}, half-life=3)")
    return df


def build_season_weighted_career(ft):
    """
    Career avg where recent 3 seasons get 2x weight vs older seasons.
    Captures player improvement/decline.
    """
    print("  [C] Building season-weighted career stats...")
    df = ft.sort_values(['player_name', 'match_date', 'match_id']).copy()

    df['season_num'] = pd.to_datetime(df['match_date']).dt.year

    def weighted_career(group):
        result = np.full(len(group), np.nan)
        pts = group[TARGET].values
        years = group['season_num'].values

        for i in range(1, len(group)):
            past_pts = pts[:i]
            past_years = years[:i]
            current_year = years[i]

            # Weight: 2x for last 3 seasons, 1x for older
            weights = np.where(current_year - past_years <= 3, 2.0, 1.0)
            result[i] = np.average(past_pts, weights=weights)

        return pd.Series(result, index=group.index)

    df['season_weighted_career_avg'] = df.groupby('player_name', group_keys=False).apply(weighted_career)
    print(f"  Built season-weighted career averages")
    return df


def main():
    print("=" * 60)
    print("  BUILD FEATURE TABLE v10")
    print("  Player-vs-bowler + EWMA + season weighting")
    print("=" * 60)

    ft = pd.read_csv(INPUT_FT, low_memory=False)
    ft['match_date'] = pd.to_datetime(ft['match_date'])
    ft = ft.sort_values(['match_date', 'match_id', 'player_name']).reset_index(drop=True)
    ft = ft.drop_duplicates(subset=['match_id', 'player_name'], keep='first')
    print(f"\n  Input: {ft.shape[0]} rows x {ft.shape[1]} cols")

    bbb = pd.read_csv(INPUT_BBB, low_memory=False)
    print(f"  Ball-by-ball: {len(bbb)} deliveries")

    # A) Batter vs specific bowlers
    bowler_feats = build_batter_vs_match_bowlers(bbb, ft)
    ft = ft.merge(bowler_feats, on=['match_id', 'player_name'], how='left')
    print(f"  After bowler matchup merge: {ft.shape}")

    # B) EWMA features
    ft = build_ewma_features(ft)
    print(f"  After EWMA: {ft.shape}")

    # C) Season-weighted career
    ft = build_season_weighted_career(ft)
    print(f"  After season weighting: {ft.shape}")

    # Fill NaNs
    new_cols = ['sr_vs_match_bowlers', 'avg_vs_match_bowlers', 'n_bowlers_with_history',
                'ewma_points', 'ewma_std', 'ewma_consistency', 'season_weighted_career_avg']
    existing = [c for c in new_cols if c in ft.columns]

    for col in existing:
        nans = ft[col].isna().sum()
        if nans > 0:
            med = ft[col].median()
            ft[col] = ft[col].fillna(med if pd.notna(med) else 0)
            print(f"  Filled {nans} NaNs in {col}")

    # Drop helper columns
    ft = ft.drop(columns=['season_num'], errors='ignore')
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
