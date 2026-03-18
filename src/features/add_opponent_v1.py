import pandas as pd
from pathlib import Path

INPUT_PATH = Path("output/player_match_fantasy_v1.csv")
OUTPUT_PATH = Path("output/player_match_fantasy_v1_with_opponent.csv")


def main():
    df = pd.read_csv(INPUT_PATH)

    teams_per_match = df.groupby("match_id")["team"].unique().to_dict()

    def get_opponent(row):
        teams = teams_per_match.get(row["match_id"], [])
        if len(teams) != 2:
            return None
        return teams[0] if row["team"] == teams[1] else teams[1]

    df["opponent"] = df.apply(get_opponent, axis=1)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("\nMissing opponent rows:", df["opponent"].isna().sum())
    print("\nSample rows:")
    print(df[["match_id", "team", "opponent"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()