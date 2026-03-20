import pandas as pd

df = pd.read_csv("output/model_feature_table_v3.csv", low_memory=False)

print("Shape:")
print(df.shape)

print("\nMissing values total:")
print(df.isna().sum().sum())

print("\nTarget summary:")
print(df["fantasy_points_v3"].describe())

print("\nRole distribution:")
print(df["player_role_platform"].value_counts())

print("\nKey feature summaries:")
for col in [
    "career_avg_points_before",
    "points_last_5_avg",
    "points_last_10_avg",
    "points_last_5_std",
    "bat_pos_last_5_avg",
    "opener_rate_last_5",
    "death_overs_runs_last_5_avg",
    "death_overs_balls_bowled_last_5_avg",
    "dot_ball_percentage_bowling_last_5_avg",
    "boundary_ball_percentage_last_5_avg",
    "full_quota_rate_last_5",
]:
    print(f"\n{col}:")
    print(df[col].describe())

print("\nSample rows:")
print(df.head(10).to_string(index=False))