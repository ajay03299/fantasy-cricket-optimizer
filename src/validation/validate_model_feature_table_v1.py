import pandas as pd

df = pd.read_csv("output/model_feature_table_v1.csv")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nTarget summary:")
print(df["fantasy_points_v1"].describe())

print("\nMatches played before summary:")
print(df["matches_played_before"].describe())

print("\nRows with all-zero core history features:")
zero_history = df[
    (df["points_last_1"] == 0) &
    (df["points_last_3_avg"] == 0) &
    (df["points_last_5_avg"] == 0) &
    (df["career_avg_points_before"] == 0)
]
print(len(zero_history))

print("\nUnique roles:")
print(df["player_role_platform"].value_counts())

print("\nSample rows:")
print(df.head(15).to_string(index=False))
