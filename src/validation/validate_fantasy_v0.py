import pandas as pd

df = pd.read_csv("output/player_match_fantasy_v0.csv")

print("\nShape:")
print(df.shape)

print("\nFantasy points summary:")
print(df["fantasy_points_v0"].describe())

print("\nTop 20 rows by fantasy_points_v0:")
print(
    df.sort_values("fantasy_points_v0", ascending=False)[
        [
            "match_id",
            "player_name",
            "team",
            "runs_scored",
            "wickets",
            "catches",
            "stumpings",
            "runouts",
            "fantasy_points_v0",
        ]
    ].head(20).to_string(index=False)
)

print("\nRows with negative scores:")
print((df["fantasy_points_v0"] < 0).sum())

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