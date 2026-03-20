"""Quick sanity check comparing v4 vs v5 scoring and feature quality."""
import pandas as pd
import numpy as np

print("=" * 55)
print("FANTASY POINTS: v4 vs v5 comparison")
print("=" * 55)

v5 = pd.read_csv("output/player_match_fantasy_v5.csv", low_memory=False)

# Reload v4 scores from feature table (has fantasy_points_v4)
v4 = pd.read_csv("output/model_feature_table_v4.csv", low_memory=False)

print(f"\nv4 points  — mean: {v4['fantasy_points_v4'].mean():.2f} | std: {v4['fantasy_points_v4'].std():.2f}")
print(f"v5 points  — mean: {v5['fantasy_points_v5'].mean():.2f} | std: {v5['fantasy_points_v5'].std():.2f}")

print(f"\nv5 SR bonus (pts_sr > 0):        {(v5['pts_sr'] > 0).sum()} rows")
print(f"v5 SR penalty (pts_sr < 0):      {(v5['pts_sr'] < 0).sum()} rows")
print(f"v5 LBW/bowled bonus > 0:         {(v5['pts_lbw_bowled'] > 0).sum()} rows")
print(f"v5 3-catch bonus > 0:            {(v5['pts_3catch'] > 0).sum()} rows")
print(f"v5 30-run milestone bonus:       {((v5['pts_milestone'] == 4)).sum()} rows")
print(f"v5 economy bonus (eco > 0, <7):  {(v5['pts_eco'] > 0).sum()} rows")
print(f"v5 economy penalty:              {(v5['pts_eco'] < 0).sum()} rows")

print("\n" + "=" * 55)
print("FEATURE TABLE v5 quality check")
print("=" * 55)
ft = pd.read_csv("output/model_feature_table_v5.csv", low_memory=False)
print(f"Rows: {len(ft)} | Columns: {len(ft.columns)}")
print(f"NaN count: {ft.isna().sum().sum()}")
print(f"\nform_slope_last5  — mean: {ft['form_slope_last5'].mean():.3f} | non-zero: {(ft['form_slope_last5'] != 0).sum()}")
print(f"pts_above_role_avg — mean: {ft['pts_above_role_avg'].mean():.3f} | std: {ft['pts_above_role_avg'].std():.3f}")
print(f"days_since_last_match — median: {ft['days_since_last_match'].median():.0f} | max: {ft['days_since_last_match'].max():.0f}")

print("\nRole distribution in feature table:")
print(ft["player_role_platform"].value_counts())

print("\nCold-start rows (matches_played_before == 0):", (ft["matches_played_before"] == 0).sum())
print("Check imputation — career_avg for cold-start rows (should NOT all be 0):")
cold = ft[ft["matches_played_before"] == 0]
print(cold["career_avg_points_before"].describe().round(2))
