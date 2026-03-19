import pandas as pd

df = pd.read_csv("data/processed/v3_dataset.csv", low_memory=False)

print("Shape:")
print(df.shape)

print("\nMissing values total:")
print(df.isna().sum().sum())

print("\nColumns:")
print(df.columns.tolist())

print("\nRole distribution:")
print(df["player_role_platform"].value_counts())

print("\nLineup distribution:")
print(df["in_announced_lineup"].value_counts())

print("\nActually batted:")
print(df["actually_batted"].value_counts())

print("\nActually bowled:")
print(df["actually_bowled"].value_counts())

print("\nBatting position summary:")
print(df["batting_position_actual"].describe())

print("\nEntry over summary:")
print(df["entry_over"].describe())

print("\nRunout direct summary:")
print(df["runout_direct"].describe())

print("\nRunout assist summary:")
print(df["runout_assist"].describe())

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