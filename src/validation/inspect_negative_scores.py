import pandas as pd

df = pd.read_csv("output/player_match_fantasy_v0.csv")

neg = df[df["fantasy_points_v0"] < 0].copy()

print("Negative score rows:", len(neg))
print("\nLowest 30 fantasy scores:")
print(
    neg.sort_values("fantasy_points_v0")[
        [
            "match_id",
            "player_name",
            "team",
            "runs_scored",
            "balls_faced",
            "batting_strike_rate",
            "points_strike_rate_adjustment",
            "fantasy_points_v0",
        ]
    ].head(30).to_string(index=False)
)