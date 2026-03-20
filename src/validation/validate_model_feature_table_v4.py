import pandas as pd

df = pd.read_csv("output/model_feature_table_v4.csv", low_memory=False)

print("Shape:")
print(df.shape)

print("\nMissing values total:")
print(df.isna().sum().sum())

print("\nTarget summary:")
print(df["fantasy_points_v4"].describe())

print("\nRole distribution:")
print(df["player_role_platform"].value_counts())

print("\nContext feature summaries:")
for col in [
    "venue_avg_total_runs_before",
    "venue_avg_wickets_before",
    "team_avg_total_runs_last_5",
    "team_avg_death_overs_runs_last_5",
    "opponent_avg_economy_last_5",
    "opponent_avg_powerplay_wickets_last_5",
]:
    print(f"\n{col}:")
    print(df[col].describe())

print("\nSample rows:")
print(df.head(10).to_string(index=False))