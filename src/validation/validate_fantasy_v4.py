import pandas as pd

df = pd.read_csv("output/player_match_fantasy_v4.csv", low_memory=False)

print("Shape:")
print(df.shape)

print("\nFantasy points v4 summary:")
print(df["fantasy_points_v4"].describe())

print("\nNegative score rows:")
print((df["fantasy_points_v4"] < 0).sum())

print("\nLineup points applied:")
print((df["points_lineup"] > 0).sum())

print("\nRunout direct points applied:")
print((df["points_runout_direct"] > 0).sum())

print("\nRunout assist points applied:")
print((df["points_runout_assist"] > 0).sum())

print("\nTop 20 rows:")
print(
    df.sort_values("fantasy_points_v4", ascending=False)[
        ["match_id", "player_name", "team", "player_role_platform", "runs_scored", "wickets", "maidens", "fantasy_points_v4"]
    ].head(20).to_string(index=False)
)