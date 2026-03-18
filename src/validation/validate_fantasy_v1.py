import pandas as pd

df = pd.read_csv("output/player_match_fantasy_v1.csv")

print("Shape:")
print(df.shape)

print("\nFantasy points v1 summary:")
print(df["fantasy_points_v1"].describe())

print("\nTop 20 rows:")
print(
    df.sort_values("fantasy_points_v1", ascending=False)[
        [
            "match_id",
            "player_name",
            "team",
            "player_role_platform",
            "runs_scored",
            "wickets",
            "maidens",
            "catches",
            "stumpings",
            "runouts",
            "fantasy_points_v1",
        ]
    ].head(20).to_string(index=False)
)

print("\nNegative score rows:")
print((df["fantasy_points_v1"] < 0).sum())

print("\nDuck penalties applied:")
print((df["points_duck_penalty"] < 0).sum())

print("\nMaiden bonuses applied:")
print((df["points_maiden_bonus"] > 0).sum())

print("\nRows with balls_bowled < 12 but economy bonus > 0:")
bad_eco = df[
    (df["balls_bowled"] < 12) &
    (df["points_economy_adjustment"] > 0)
]
print(len(bad_eco))

print("\nRows with balls_faced < 10 but strike rate adjustment != 0:")
bad_sr = df[
    (df["balls_faced"] < 10) &
    (df["points_strike_rate_adjustment"] != 0)
]
print(len(bad_sr))