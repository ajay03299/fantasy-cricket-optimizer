import pandas as pd

df = pd.read_csv("output/best_model_v3_predictions.csv")

df["error"] = df["predicted_fantasy_points_v3"] - df["fantasy_points_v3"]
df["abs_error"] = df["error"].abs()

print("Top 25 biggest misses:\n")
print(
    df.sort_values("abs_error", ascending=False)[
        [
            "match_date",
            "player_name",
            "team",
            "opponent",
            "player_role_platform",
            "fantasy_points_v3",
            "predicted_fantasy_points_v3",
            "error",
            "abs_error",
            "points_last_1",
            "points_last_3_avg",
            "points_last_5_avg",
            "career_avg_points_before",
            "bat_pos_last_5_avg",
            "opener_rate_last_5",
            "death_overs_runs_last_5_avg",
            "death_overs_balls_bowled_last_5_avg",
            "full_quota_rate_last_5",
        ]
    ].head(25).to_string(index=False)
)

print("\nAverage absolute error by role:\n")
print(df.groupby("player_role_platform")["abs_error"].mean().sort_values())

print("\nAverage actual vs predicted:")
print("Actual mean:", df["fantasy_points_v3"].mean())
print("Predicted mean:", df["predicted_fantasy_points_v3"].mean())