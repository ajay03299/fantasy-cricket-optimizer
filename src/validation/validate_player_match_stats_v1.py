import pandas as pd

df = pd.read_csv("data/processed/v1_dataset.csv")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nRole distribution:")
print(df["player_role_platform"].value_counts())

print("\nDismissal distribution:")
print(df["dismissal_kind"].value_counts().head(15))

print("\nDuck count:")
print(df["is_duck"].value_counts())

print("\nDismissed count:")
print(df["dismissed"].value_counts())

print("\nMaiden summary:")
print(df["maidens"].describe())

print("\nPotential duck inconsistencies:")
duck_issues = df[(df["is_duck"] == 1) & (df["runs_scored"] != 0)]
print(len(duck_issues))

print("\nPotential dismissal inconsistencies:")
dismissal_issues = df[(df["dismissed"] == 0) & (df["dismissal_kind"] != "Not Out")]
print(len(dismissal_issues))

print("\nPotential batting SR inconsistencies:")
sr_issues = df[(df["balls_faced"] == 0) & (df["batting_strike_rate"] != 0)]
print(len(sr_issues))

print("\nPotential bowling economy inconsistencies:")
eco_issues = df[(df["balls_bowled"] == 0) & (df["bowling_economy"] != 0)]
print(len(eco_issues))