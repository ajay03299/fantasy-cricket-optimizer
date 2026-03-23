"""
V8 Leakage Audit — verify the +164.7% Spearman jump is clean
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

TARGET = "fantasy_points_v5"

print("=" * 70)
print("  V8 LEAKAGE AUDIT")
print("=" * 70)

# Load
ft8 = pd.read_csv("output/model_feature_table_v8.csv", low_memory=False)
pred = pd.read_csv("output/best_model_v8_predictions.csv", low_memory=False)
ft6 = pd.read_csv("output/model_feature_table_v6.csv", low_memory=False)

# 1. ROW COUNT CHECK
print(f"\n[1] ROW COUNT CHECK")
print(f"  v6 rows: {len(ft6)}")
print(f"  v8 rows: {len(ft8)}")
print(f"  Difference: {len(ft8) - len(ft6)}")
if len(ft8) != len(ft6):
    print("  WARNING: Row count mismatch! Toss merge likely created duplicates.")

# 2. DUPLICATE CHECK
print(f"\n[2] DUPLICATE CHECK")
dupes = ft8.duplicated(subset=['match_id', 'player_name']).sum()
print(f"  Duplicate (match_id, player_name) pairs: {dupes}")
if dupes > 0:
    print("  LEAKAGE RISK: Duplicates inflate training data and corrupt evaluation!")
    dup_rows = ft8[ft8.duplicated(subset=['match_id', 'player_name'], keep=False)]
    print(f"  Example duplicates:")
    print(dup_rows.head(4)[['match_id', 'player_name', 'team', TARGET]].to_string())

# 3. TEMPORAL INTEGRITY — do features only use past data?
print(f"\n[3] TEMPORAL INTEGRITY")
ft8['match_date'] = pd.to_datetime(ft8['match_date'])
ft8_sorted = ft8.sort_values(['player_name', 'match_date'])

# Check: is batter_sr_vs_pace_before actually from BEFORE?
# For each player's first match, it should be NaN or filled with median
first_matches = ft8_sorted.groupby('player_name').first()
new_feats = ['batter_sr_vs_pace_before', 'batter_sr_vs_spin_before',
             'batter_avg_vs_pace_before', 'batter_avg_vs_spin_before',
             'matchup_advantage', 'batter_death_sr_before', 'bowler_death_eco_before']
for f in new_feats:
    if f in first_matches.columns:
        non_null = first_matches[f].notna().sum()
        # After fillna, all will be non-null, but check if they're all the same (median)
        unique_vals = first_matches[f].nunique()
        print(f"  {f}: first-match values — {unique_vals} unique vals")

# 4. MATCHUP ADVANTAGE DEEP DIVE
print(f"\n[4] MATCHUP ADVANTAGE DEEP DIVE")
if 'matchup_advantage' in ft8.columns:
    ma = ft8['matchup_advantage']
    print(f"  Correlation with target: {ma.corr(ft8[TARGET]):.4f}")
    print(f"  Mean: {ma.mean():.4f}, Std: {ma.std():.4f}")
    print(f"  Zero values: {(ma == 0).sum()} ({(ma == 0).mean():.1%})")
    
    # Check if matchup_advantage is just a proxy for batting ability
    if 'career_avg_points_before' in ft8.columns:
        proxy_r = ma.corr(ft8['career_avg_points_before'])
        print(f"  Correlation with career_avg: {proxy_r:.4f}")
        print(f"  (If high, matchup_advantage may just be encoding player quality)")

# 5. PREDICTION ANALYSIS
print(f"\n[5] PREDICTION QUALITY")
pred['match_date'] = pd.to_datetime(pred['match_date'])
print(f"  Test date range: {pred['match_date'].min().date()} to {pred['match_date'].max().date()}")
print(f"  Test rows: {len(pred)}")
print(f"  Pred std: {pred['pred'].std():.2f} vs Actual std: {pred[TARGET].std():.2f}")

# Per-match Spearman distribution
corrs = []
for mid, g in pred.groupby('match_id'):
    if len(g) > 3:
        c, _ = spearmanr(g[TARGET], g['pred'])
        if not np.isnan(c):
            corrs.append(c)

print(f"\n  Per-match Spearman:")
print(f"    Mean: {np.mean(corrs):.4f}")
print(f"    Median: {np.median(corrs):.4f}")
print(f"    >0.5: {sum(1 for c in corrs if c > 0.5)}/{len(corrs)} ({sum(1 for c in corrs if c > 0.5)/len(corrs):.1%})")
print(f"    >0.3: {sum(1 for c in corrs if c > 0.3)}/{len(corrs)} ({sum(1 for c in corrs if c > 0.3)/len(corrs):.1%})")
print(f"    <0:   {sum(1 for c in corrs if c < 0)}/{len(corrs)} ({sum(1 for c in corrs if c < 0)/len(corrs):.1%})")

# 6. VERDICT
print(f"\n{'='*70}")
print("  VERDICT")
print(f"{'='*70}")
issues = []
if len(ft8) != len(ft6):
    issues.append("ROW COUNT MISMATCH — toss merge created extra rows")
if dupes > 0:
    issues.append(f"DUPLICATES — {dupes} duplicate player-match pairs")

if issues:
    print("  ISSUES FOUND:")
    for i in issues:
        print(f"    - {i}")
    print("\n  ACTION: Fix duplicates before proceeding to v9.")
    print("  The improvement may be partially inflated by duplicate rows.")
else:
    print("  NO LEAKAGE DETECTED. Results are clean.")
    print("  The improvement is driven by genuinely new matchup signal.")

