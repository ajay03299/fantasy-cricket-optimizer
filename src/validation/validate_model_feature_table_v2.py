import pandas as pd

df = pd.read_csv("output/model_feature_table_v2.csv")

print("Shape:")
print(df.shape)

print("\nMissing values total:")
print(df.isna().sum().sum())

print("\nTarget summary:")
print(df["fantasy_points_v1"].describe())

print("\nNew feature summaries:")
for col in [
    "points_last_10_avg",
    "points_last_5_std",
    "points_last_10_std",
    "points_last_5_max",
    "points_last_10_max",
    "batting_active_rate_last_5",
    "bowling_active_rate_last_5",
    "dismissal_rate_last_5",
    "duck_rate_last_10",
    "maidens_last_5_avg",
]:
    print(f"\n{col}:")
    print(df[col].describe())

print("\nSample rows:")
print(df.head(10).to_string(index=False))