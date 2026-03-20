from pathlib import Path
import pandas as pd
import numpy as np

INPUT_PATH  = Path("data/processed/v4_dataset.csv")
OUTPUT_PATH = Path("output/player_match_fantasy_v5.csv")

REQUIRED_COLUMNS = [
    "match_id", "player_name", "team", "opponent", "player_role_platform",
    "in_announced_lineup", "runs_scored", "balls_faced", "fours", "sixes",
    "batting_strike_rate", "dismissed", "is_duck", "balls_bowled",
    "runs_conceded", "wickets", "maidens", "bowling_economy",
    "lbw_wickets", "bowled_wickets",
    "catches", "stumpings", "runout_direct", "runout_assist",
    "match_date", "season", "venue",
]

def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def add_fantasy_points_v5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "in_announced_lineup", "runs_scored", "balls_faced", "fours", "sixes",
        "batting_strike_rate", "dismissed", "is_duck", "balls_bowled",
        "runs_conceded", "wickets", "maidens", "bowling_economy",
        "lbw_wickets", "bowled_wickets",
        "catches", "stumpings", "runout_direct", "runout_assist",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # ── BATTING ──────────────────────────────────────────────────────────────
    df["pts_runs"]      = df["runs_scored"]
    df["pts_4s"]        = df["fours"] * 1
    df["pts_6s"]        = df["sixes"] * 2

    df["pts_milestone"] = np.select(
        [df["runs_scored"] >= 100,
         df["runs_scored"] >= 50,
         df["runs_scored"] >= 30],
        [16, 8, 4],
        default=0,
    )

    df["pts_duck"] = np.where(
        (df["is_duck"] == 1) & df["player_role_platform"].isin(["BAT", "WK", "AR"]),
        -2, 0,
    )

    # SR bonus AND penalty — v0-v4 were missing the bonus side entirely
    sr_eligible = (df["balls_faced"] >= 10) & (df["player_role_platform"] != "BOWL")
    df["pts_sr"] = np.select(
        [sr_eligible & (df["batting_strike_rate"] > 170),
         sr_eligible & (df["batting_strike_rate"] > 150),
         sr_eligible & (df["batting_strike_rate"] > 130),
         sr_eligible & (df["batting_strike_rate"] < 50),
         sr_eligible & (df["batting_strike_rate"] < 60),
         sr_eligible & (df["batting_strike_rate"] < 70)],
        [6, 4, 2, -6, -4, -2],
        default=0,
    )

    # ── BOWLING ───────────────────────────────────────────────────────────────
    df["pts_wickets"] = df["wickets"] * 25

    df["pts_haul"] = np.select(
        [df["wickets"] >= 5,
         df["wickets"] >= 4,
         df["wickets"] >= 3],
        [16, 8, 4],
        default=0,
    )

    # LBW + bowled bonus: 8 pts each (separate columns in your dataset)
    df["pts_lbw_bowled"] = (df["lbw_wickets"] + df["bowled_wickets"]) * 8

    df["pts_maidens"] = df["maidens"] * 8

    # Economy: full bonus AND penalty scale (v0-v4 only had partial penalty)
    eco_eligible = df["balls_bowled"] >= 12
    df["pts_eco"] = np.select(
        [eco_eligible & (df["bowling_economy"] < 5),
         eco_eligible & (df["bowling_economy"] < 6),
         eco_eligible & (df["bowling_economy"] < 7),
         eco_eligible & (df["bowling_economy"] > 10),
         eco_eligible & (df["bowling_economy"] > 9),
         eco_eligible & (df["bowling_economy"] > 8)],
        [6, 4, 2, -6, -4, -2],
        default=0,
    )

    # ── FIELDING ──────────────────────────────────────────────────────────────
    df["pts_catches"]   = df["catches"] * 8
    df["pts_3catch"]    = np.where(df["catches"] >= 3, 4, 0)
    df["pts_stumpings"] = df["stumpings"] * 12
    df["pts_ro_direct"] = df["runout_direct"] * 12
    df["pts_ro_assist"] = df["runout_assist"] * 6

    # ── PLAYING XI ────────────────────────────────────────────────────────────
    df["pts_playing_xi"] = df["in_announced_lineup"] * 4

    score_cols = [
        "pts_runs", "pts_4s", "pts_6s", "pts_milestone", "pts_duck", "pts_sr",
        "pts_wickets", "pts_haul", "pts_lbw_bowled", "pts_maidens", "pts_eco",
        "pts_catches", "pts_3catch", "pts_stumpings", "pts_ro_direct",
        "pts_ro_assist", "pts_playing_xi",
    ]
    df["fantasy_points_v5"] = df[score_cols].sum(axis=1)
    return df

def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    validate_columns(df)
    out = add_fantasy_points_v5(df)
    out.to_csv(OUTPUT_PATH, index=False)

    print("✅ Saved:", OUTPUT_PATH)
    print("\nfantasy_points_v5 distribution:")
    print(out["fantasy_points_v5"].describe().round(2))

    print("\nTop 15 scores:")
    preview = ["match_id","player_name","team","player_role_platform",
               "runs_scored","wickets","catches","pts_lbw_bowled","pts_sr","fantasy_points_v5"]
    print(out.sort_values("fantasy_points_v5", ascending=False)[preview].head(15).to_string(index=False))

    print("\nComparison v4 vs v5 (same rows):")
    if "fantasy_points_v4" in out.columns:
        diff = out["fantasy_points_v5"] - out["fantasy_points_v4"]
        print(f"  Mean diff: {diff.mean():.2f}")
        print(f"  Rows where v5 > v4: {(diff > 0).sum()}")
        print(f"  Rows where v5 < v4: {(diff < 0).sum()}")

if __name__ == "__main__":
    main()
