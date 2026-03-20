import pandas as pd

df = pd.read_csv("output/player_match_fantasy_v3.csv", low_memory=False)

print("Shape:")
print(df.shape)

print("\nFantasy points v3 summary:")
print(df["fantasy_points_v3"].describe())

print("\nNegative score rows:")
print((df["fantasy_points_v3"] < 0).sum())

print("\nDuck penalties applied:")
print((df["points_duck_penalty"] < 0).sum())

print("\nMaiden bonuses applied:")
print((df["points_maiden_bonus"] > 0).sum())

print("\nRunout direct points applied:")
print((df["points_runout_direct"] > 0).sum())

print("\nRunout assist points applied:")
print((df["points_runout_assist"] > 0).sum())

print("\nLineup points applied:")
print((df["points_lineup"] > 0).sum())

print("\nTop 20 rows:")
print(
    df.sort_values("fantasy_points_v3", ascending=False)[
        ["match_id", "player_name", "team", "player_role_platform", "runs_scored", "wickets", "maidens", "runout_direct", "runout_assist", "fantasy_points_v3"]
    ].head(20).to_string(index=False)
)