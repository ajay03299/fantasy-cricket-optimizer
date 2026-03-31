"""
Feature Table v8: Ball-by-ball matchup features + toss context
==============================================================
New features from ball-by-ball data (278K deliveries):
  1-2. batter_sr_vs_pace_before / batter_sr_vs_spin_before
  3-4. batter_avg_vs_pace_before / batter_avg_vs_spin_before
  5-6. opponent_pace_pct / opponent_spin_pct (attack profile)
  7.   matchup_advantage (batter SR vs opponent's dominant type)
  8.   bowler_death_eco_before (death overs economy)
  9.   batter_death_sr_before (death overs strike rate)

New features from match data:
  10.  batting_first (1st innings vs chasing)
  11.  toss_won (did player's team win toss)
  12.  toss_chose_field (toss winner chose to field)

Reads:  output/model_feature_table_v6.csv + ball-by-ball + match data
Writes: output/model_feature_table_v8.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

INPUT_FT     = Path("output/model_feature_table_v6.csv")
INPUT_BBB    = Path("data/all_ball_by_ball_data.csv")
INPUT_MATCHES= Path("data/all_ipl_matches_data.csv")
OUTPUT_PATH  = Path("output/model_feature_table_v8.csv")

TARGET = "fantasy_points_v5"


def classify_bowler(bt):
    bt = str(bt).lower()
    if any(k in bt for k in ['fast', 'medium fast', 'fast medium']):
        return 'PACE'
    elif any(k in bt for k in ['offbreak', 'legbreak', 'googly', 'orthodox', 'wrist spin', 'spin']):
        return 'SPIN'
    elif 'medium' in bt:
        return 'MEDIUM'
    return 'UNKNOWN'


def build_batter_vs_type_features(bbb, ft):
    """
    For each batter, compute rolling career SR and avg
    against PACE, SPIN, and MEDIUM bowling — BEFORE each match.
    """
    print("  Computing batter vs bowler-type matchup stats...")
    bbb = bbb.copy()
    bbb['bowler_cat'] = bbb['bowler_type'].apply(classify_bowler)

    # Get match dates for ordering
    match_dates = ft[['match_id', 'match_date']].drop_duplicates()
    bbb = bbb.merge(match_dates, on='match_id', how='left')
    bbb['match_date'] = pd.to_datetime(bbb['match_date'])

    # Per-match per-batter stats vs each bowler category
    batter_match = bbb.groupby(['match_id', 'match_date', 'batter', 'bowler_cat']).agg(
        balls=('batter_runs', 'count'),
        runs=('batter_runs', 'sum'),
        dismissals=('is_wicket', 'sum'),
    ).reset_index()

    results = []
    for cat in ['PACE', 'SPIN']:
        cat_data = batter_match[batter_match['bowler_cat'] == cat].copy()
        cat_data = cat_data.sort_values(['batter', 'match_date', 'match_id'])

        # Cumulative stats BEFORE this match (shift by 1)
        cat_data['cum_balls'] = cat_data.groupby('batter')['balls'].cumsum().shift(1)
        cat_data['cum_runs'] = cat_data.groupby('batter')['runs'].cumsum().shift(1)
        cat_data['cum_dismissals'] = cat_data.groupby('batter')['dismissals'].cumsum().shift(1)

        # Fill first match with NaN (no history)
        cat_data[f'batter_sr_vs_{cat.lower()}_before'] = (
            cat_data['cum_runs'] / cat_data['cum_balls'].replace(0, np.nan) * 100
        )
        cat_data[f'batter_avg_vs_{cat.lower()}_before'] = (
            cat_data['cum_runs'] / (cat_data['cum_dismissals'].replace(0, 0.5))
        )

        keep_cols = ['match_id', 'batter',
                     f'batter_sr_vs_{cat.lower()}_before',
                     f'batter_avg_vs_{cat.lower()}_before']
        results.append(cat_data[keep_cols].rename(columns={'batter': 'player_name'}))

    # Merge all
    merged = results[0]
    for r in results[1:]:
        merged = merged.merge(r, on=['match_id', 'player_name'], how='outer')

    print(f"  Built matchup features for {merged['player_name'].nunique()} batters")
    return merged


def build_opponent_attack_profile(bbb, ft):
    """
    For each team, compute the % of PACE vs SPIN balls they bowl
    (rolling average of last 5 matches).
    """
    print("  Computing opponent attack profiles...")
    bbb = bbb.copy()
    bbb['bowler_cat'] = bbb['bowler_type'].apply(classify_bowler)

    match_dates = ft[['match_id', 'match_date']].drop_duplicates()
    bbb = bbb.merge(match_dates, on='match_id', how='left')

    # Per match per bowling team
    team_match = bbb.groupby(['match_id', 'match_date', 'team_bowling', 'bowler_cat']).size().reset_index(name='balls')
    pivot = team_match.pivot_table(
        index=['match_id', 'match_date', 'team_bowling'],
        columns='bowler_cat', values='balls', fill_value=0
    ).reset_index()

    total = pivot[['PACE', 'SPIN', 'MEDIUM']].sum(axis=1)
    pivot['pace_pct'] = pivot['PACE'] / total
    pivot['spin_pct'] = pivot['SPIN'] / total

    pivot = pivot.sort_values(['team_bowling', 'match_date'])

    # Rolling last-5 average
    for col in ['pace_pct', 'spin_pct']:
        pivot[f'{col}_rolling'] = (
            pivot.groupby('team_bowling')[col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

    result = pivot[['match_id', 'team_bowling', 'pace_pct_rolling', 'spin_pct_rolling']].copy()
    result = result.rename(columns={
        'team_bowling': 'opponent_team',
        'pace_pct_rolling': 'opponent_pace_pct',
        'spin_pct_rolling': 'opponent_spin_pct',
    })

    print(f"  Built attack profiles for {result['opponent_team'].nunique()} teams")
    return result


def build_death_overs_features(bbb, ft):
    """
    Death overs (16-20) specialist stats — SR for batters, economy for bowlers.
    Rolling career stats BEFORE each match.
    """
    print("  Computing death overs specialist features...")
    death = bbb[bbb['over_number'] >= 16].copy()

    match_dates = ft[['match_id', 'match_date']].drop_duplicates()
    death = death.merge(match_dates, on='match_id', how='left')
    death['match_date'] = pd.to_datetime(death['match_date'])

    # Batter death SR
    bat_death = death.groupby(['match_id', 'match_date', 'batter']).agg(
        death_balls=('batter_runs', 'count'),
        death_runs=('batter_runs', 'sum'),
    ).reset_index()
    bat_death = bat_death.sort_values(['batter', 'match_date'])
    bat_death['cum_balls'] = bat_death.groupby('batter')['death_balls'].cumsum().shift(1)
    bat_death['cum_runs'] = bat_death.groupby('batter')['death_runs'].cumsum().shift(1)
    bat_death['batter_death_sr_before'] = (
        bat_death['cum_runs'] / bat_death['cum_balls'].replace(0, np.nan) * 100
    )

    # Bowler death economy
    bowl_death = death.groupby(['match_id', 'match_date', 'bowler']).agg(
        death_balls=('total_runs', 'count'),
        death_runs_conceded=('total_runs', 'sum'),
    ).reset_index()
    bowl_death = bowl_death.sort_values(['bowler', 'match_date'])
    bowl_death['cum_balls'] = bowl_death.groupby('bowler')['death_balls'].cumsum().shift(1)
    bowl_death['cum_runs'] = bowl_death.groupby('bowler')['death_runs_conceded'].cumsum().shift(1)
    bowl_death['bowler_death_eco_before'] = (
        bowl_death['cum_runs'] / (bowl_death['cum_balls'].replace(0, np.nan) / 6)
    )

    bat_result = bat_death[['match_id', 'batter', 'batter_death_sr_before']].rename(
        columns={'batter': 'player_name'}
    )
    bowl_result = bowl_death[['match_id', 'bowler', 'bowler_death_eco_before']].rename(
        columns={'bowler': 'player_name'}
    )

    # Merge — a player can be both batter and bowler
    result = bat_result.merge(bowl_result, on=['match_id', 'player_name'], how='outer')
    print(f"  Built death features for {result['player_name'].nunique()} players")
    return result


def build_toss_features(matches, ft):
    """
    Toss context: batting first, toss won, toss decision.
    """
    print("  Computing toss context features...")
    bbb_raw = pd.read_csv(Path("data/all_ball_by_ball_data.csv"),
                          usecols=['match_id', 'team_batting', 'innings'], low_memory=False)

    # Get team batting in innings 1 (= batting first)
    inn1 = bbb_raw[bbb_raw['innings'] == 1][['match_id', 'team_batting']].drop_duplicates()
    inn1 = inn1.rename(columns={'team_batting': 'team_batting_first'})

    m = matches[['match_id', 'toss_winner', 'toss_decision', 'team1', 'team2']].copy()
    m = m.merge(inn1, on='match_id', how='left')

    result = ft[['match_id', 'player_name', 'team']].copy()

    # Need to map team names to team IDs used in bbb
    # For now, create features at match level
    result = result.merge(
        m[['match_id', 'toss_winner', 'toss_decision', 'team1', 'team2', 'team_batting_first']],
        on='match_id', how='left'
    )

    # batting_first: is player's team batting first?
    # We need to map team names to team IDs — use the teams data
    teams = pd.read_csv(Path("data/all_teams_data.csv"))
    aliases = pd.read_csv(Path("data/all_team_aliases.csv"))

    # Build team name -> team_id mapping
    name_to_id = dict(zip(teams['team_name'], teams['team_id']))
    for _, row in aliases.iterrows():
        name_to_id[row['alias_name']] = row['team_id']

    # Map player team names to IDs
    result['team_id'] = result['team'].map(name_to_id)

    result['batting_first'] = (result['team_id'] == result['team_batting_first']).astype(int)
    result['toss_won'] = (result['team_id'] == result['toss_winner']).astype(int)
    result['toss_chose_field'] = (result['toss_decision'] == 'field').astype(int)

    keep = ['match_id', 'player_name', 'batting_first', 'toss_won', 'toss_chose_field']
    print(f"  Toss features built. Batting first rate: {result['batting_first'].mean():.2%}")
    return result[keep]


def main():
    print("=" * 60)
    print("  BUILD FEATURE TABLE v8")
    print("  Ball-by-ball matchup + toss context features")
    print("=" * 60)

    # Load base
    print(f"\nLoading {INPUT_FT}...")
    ft = pd.read_csv(INPUT_FT, low_memory=False)
    ft['match_date'] = pd.to_datetime(ft['match_date'])
    print(f"  Input: {ft.shape[0]} rows x {ft.shape[1]} cols")

    print(f"\nLoading ball-by-ball data...")
    bbb = pd.read_csv(INPUT_BBB, low_memory=False)
    print(f"  {len(bbb)} deliveries across {bbb['match_id'].nunique()} matches")

    matches = pd.read_csv(INPUT_MATCHES, low_memory=False)

    # Build features
    print("\n[1/4] Batter vs bowler-type matchups...")
    matchup_feats = build_batter_vs_type_features(bbb, ft)

    print("\n[2/4] Opponent attack profiles...")
    attack_feats = build_opponent_attack_profile(bbb, ft)

    print("\n[3/4] Death overs specialist features...")
    death_feats = build_death_overs_features(bbb, ft)

    print("\n[4/4] Toss context features...")
    toss_feats = build_toss_features(matches, ft)

    # Merge all into feature table
    print("\nMerging features...")
    ft = ft.merge(matchup_feats, on=['match_id', 'player_name'], how='left')
    print(f"  After matchup merge: {ft.shape}")

    # For attack profile, need to map opponent team
    if 'opponent' in ft.columns:
        # Map opponent name to team_id
        teams = pd.read_csv(Path("data/all_teams_data.csv"))
        aliases = pd.read_csv(Path("data/all_team_aliases.csv"))
        name_to_id = dict(zip(teams['team_name'], teams['team_id']))
        for _, row in aliases.iterrows():
            name_to_id[row['alias_name']] = row['team_id']

        ft['opponent_team_id'] = ft['opponent'].map(name_to_id)
        ft = ft.merge(attack_feats, left_on=['match_id', 'opponent_team_id'],
                       right_on=['match_id', 'opponent_team'], how='left')
        ft = ft.drop(columns=['opponent_team_id', 'opponent_team'], errors='ignore')
    print(f"  After attack profile merge: {ft.shape}")

    ft = ft.merge(death_feats, on=['match_id', 'player_name'], how='left')
    print(f"  After death overs merge: {ft.shape}")

    ft = ft.merge(toss_feats, on=['match_id', 'player_name'], how='left')
    print(f"  After toss merge: {ft.shape}")

    # Matchup advantage: batter SR vs opponent's dominant bowling type
    # If opponent is pace-heavy, use batter_sr_vs_pace, else vs_spin
    if 'opponent_pace_pct' in ft.columns and 'batter_sr_vs_pace_before' in ft.columns:
        pace_heavy = ft['opponent_pace_pct'].fillna(0.5) > 0.5
        ft['matchup_advantage'] = np.where(
            pace_heavy,
            ft['batter_sr_vs_pace_before'].fillna(0),
            ft['batter_sr_vs_spin_before'].fillna(0)
        )
        # Normalize to z-score
        mean_ma = ft['matchup_advantage'].mean()
        std_ma = ft['matchup_advantage'].std()
        ft['matchup_advantage'] = (ft['matchup_advantage'] - mean_ma) / (std_ma + 1e-6)

    # Fill NaNs in new features
    new_cols = [
        'batter_sr_vs_pace_before', 'batter_sr_vs_spin_before',
        'batter_avg_vs_pace_before', 'batter_avg_vs_spin_before',
        'opponent_pace_pct', 'opponent_spin_pct',
        'matchup_advantage',
        'batter_death_sr_before', 'bowler_death_eco_before',
        'batting_first', 'toss_won', 'toss_chose_field',
    ]
    existing_new = [c for c in new_cols if c in ft.columns]

    for col in existing_new:
        nans = ft[col].isna().sum()
        if nans > 0:
            ft[col] = ft[col].fillna(ft[col].median() if ft[col].median() == ft[col].median() else 0)
            print(f"  Filled {nans} NaNs in {col}")

    # Save
    ft.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"  Saved: {OUTPUT_PATH}")
    print(f"  Shape: {ft.shape}")
    print(f"  New features: {existing_new}")
    print(f"{'='*60}")

    # Show correlations
    print(f"\nNew feature correlations with {TARGET}:")
    for f in existing_new:
        if f in ft.columns:
            r = ft[f].corr(ft[TARGET])
            print(f"  {r:+.4f}  {f}")


if __name__ == "__main__":
    main()
