import pandas as pd

df = pd.read_csv("output/baseline_model_v1_predictions.csv")

df["error"] = df["predicted_fantasy_points_v1"] - df["fantasy_points_v1"]
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
            "fantasy_points_v1",
            "predicted_fantasy_points_v1",
            "error",
            "abs_error",
            "points_last_1",
            "points_last_3_avg",
            "points_last_5_avg",
            "career_avg_points_before",
        ]
    ].head(25).to_string(index=False)
)

print("\nAverage absolute error by role:\n")
print(df.groupby("player_role_platform")["abs_error"].mean().sort_values())

print("\nAverage absolute error by season:\n")
print(df.groupby("season")["abs_error"].mean().sort_index())

print("\nAverage actual vs predicted:")
print("Actual mean:", df["fantasy_points_v1"].mean())
print("Predicted mean:", df["predicted_fantasy_points_v1"].mean())