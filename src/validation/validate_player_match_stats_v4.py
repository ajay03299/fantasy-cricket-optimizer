import pandas as pd

df = pd.read_csv("data/processed/v4_dataset.csv", low_memory=False)

print("Shape:")
print(df.shape)

print("\nMissing values total:")
print(df.isna().sum().sum())

print("\nRole distribution:")
print(df["player_role_platform"].value_counts())

print("\nOpponent missing:")
print(df["opponent"].isna().sum())

print("\nLineup distribution:")
print(df["in_announced_lineup"].value_counts())

print("\nActually batted:")
print(df["actually_batted"].value_counts())

print("\nActually bowled:")
print(df["actually_bowled"].value_counts())

print("\nContext feature checks:")
context_cols = [
    "venue_avg_total_runs_before",
    "venue_avg_wickets_before",
    "venue_avg_run_rate_before",
    "team_avg_total_runs_last_5",
    "team_avg_wickets_lost_last_5",
    "opponent_avg_wickets_taken_last_5",
    "opponent_avg_economy_last_5",
]
print(df[context_cols].describe())

print("\nPotential duck inconsistencies:")
duck_issues = df[(df["is_duck"] == 1) & (df["runs_scored"] != 0)]
print(len(duck_issues))

print("\nPotential dismissal inconsistencies:")
dismissal_issues = df[(df["dismissed"] == 0) & (df["dismissal_kind"] != "Not Out")]
print(len(dismissal_issues))

print("\nPotential lineup inconsistencies:")
lineup_issues = df[(df["in_announced_lineup"] == 0) & ((df["actually_batted"] == 1) | (df["actually_bowled"] == 1))]
print(len(lineup_issues))

print("\nSample rows:")
print(df.head(10).to_string(index=False))