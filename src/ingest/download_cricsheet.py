"""
Cricsheet IPL Data Ingestion
============================
Downloads latest IPL data from Cricsheet, parses it into our schema,
and appends only NEW matches to data/all_ball_by_ball_data.csv and
data/all_ipl_matches_data.csv.

Usage:
  python src/ingest/download_cricsheet.py --dry-run   # preview only
  python src/ingest/download_cricsheet.py             # actually write
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

CRICKSHEET_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"
DATA_DIR = Path("data")
BBB_PATH = DATA_DIR / "all_ball_by_ball_data.csv"
MATCHES_PATH = DATA_DIR / "all_ipl_matches_data.csv"
TEAMS_PATH = DATA_DIR / "all_teams_data.csv"
ALIASES_PATH = DATA_DIR / "all_team_aliases.csv"


def build_team_map():
    """Map team name -> team_id using both teams data and aliases."""
    teams = pd.read_csv(TEAMS_PATH)
    aliases = pd.read_csv(ALIASES_PATH)

    team_map = {}
    for _, row in teams.iterrows():
        team_map[row["team_name"].strip().lower()] = int(row["team_id"])
    for _, row in aliases.iterrows():
        team_map[row["alias"].strip().lower()] = int(row["team_id"])
    return team_map


def lookup_team_id(name, team_map):
    if not name:
        return None
    return team_map.get(str(name).strip().lower())


def parse_info_csv(content):
    """Parse a Cricsheet _info.csv into a dict of metadata."""
    info = {"teams": [], "players": {}}
    for line in content.split("\n"):
        parts = line.split(",")
        if len(parts) < 3 or parts[0] != "info":
            continue
        key = parts[1].strip()
        val = parts[2].strip()
        if key == "team":
            info["teams"].append(val)
        elif key == "toss_winner":
            info["toss_winner"] = val
        elif key == "toss_decision":
            info["toss_decision"] = val
        elif key == "winner":
            info["match_winner"] = val
        elif key == "winner_runs":
            info["win_by_runs"] = val
        elif key == "winner_wickets":
            info["win_by_wickets"] = val
        elif key == "player_of_match":
            info["player_of_match"] = val
        elif key == "venue":
            info["venue"] = val
        elif key == "city":
            info["city"] = val
        elif key == "date":
            info["match_date"] = val
        elif key == "season":
            info["season"] = val
        elif key == "event":
            info["event_name"] = val
        elif key == "match_number":
            info["match_number"] = val
        elif key == "gender":
            info["gender"] = val
        elif key == "match_type":
            info["match_type"] = val
        elif key == "overs":
            info["overs"] = val
        elif key == "balls_per_over":
            info["balls_per_over"] = val
    return info


def parse_match_csv(content, match_id, season_id, team_map):
    """Parse a Cricsheet match deliveries CSV into our ball-by-ball schema."""
    import csv as _csv
    from io import StringIO
    reader = _csv.DictReader(StringIO(content))
    rows = []

    for rec in reader:

        team_batting_id = lookup_team_id(rec.get("batting_team", ""), team_map)
        if team_batting_id is None:
            continue

        # Cricsheet has: match_id, season, start_date, venue, innings, ball, batting_team,
        # bowling_team, striker, non_striker, bowler, runs_off_bat, extras, wides, noballs,
        # byes, legbyes, penalty, wicket_type, player_dismissed, other_wicket_type, other_player_dismissed
        try:
            ball_str = rec.get("ball", "0.0")
            over_num = int(float(ball_str))
            ball_num = int(round((float(ball_str) - over_num) * 10))
        except (ValueError, TypeError):
            over_num, ball_num = 0, 0

        runs_off_bat = int(rec.get("runs_off_bat", 0) or 0)
        extras = int(rec.get("extras", 0) or 0)
        wides = int(rec.get("wides", 0) or 0)
        noballs = int(rec.get("noballs", 0) or 0)
        byes = int(rec.get("byes", 0) or 0)
        legbyes = int(rec.get("legbyes", 0) or 0)
        penalty = int(rec.get("penalty", 0) or 0)

        bowling_team = rec.get("bowling_team", "")
        team_bowling_id = lookup_team_id(bowling_team, team_map)
        if team_bowling_id is None:
            continue

        wicket_kind = rec.get("wicket_type", "") or ""
        is_wicket = bool(wicket_kind.strip())

        rows.append({
            "season_id": season_id,
            "match_id": match_id,
            "batter": rec.get("striker", ""),
            "bowler": rec.get("bowler", ""),
            "non_striker": rec.get("non_striker", ""),
            "team_batting": team_batting_id,
            "team_bowling": team_bowling_id,
            "over_number": over_num,
            "ball_number": ball_num,
            "batter_runs": runs_off_bat,
            "extras": extras,
            "total_runs": runs_off_bat + extras,
            "batsman_type": "",  # not in cricsheet, leave blank
            "bowler_type": "",
            "player_out": rec.get("player_dismissed", ""),
            "fielders_involved": "",
            "is_wicket": is_wicket,
            "is_wide_ball": wides > 0,
            "is_no_ball": noballs > 0,
            "is_leg_bye": legbyes > 0,
            "is_bye": byes > 0,
            "is_penalty": penalty > 0,
            "wide_ball_runs": wides,
            "no_ball_runs": noballs,
            "leg_bye_runs": legbyes,
            "bye_runs": byes,
            "penalty_runs": penalty,
            "wicket_kind": wicket_kind,
            "is_super_over": False,
            "innings": int(rec.get("innings", 1) or 1),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Download and parse but don't write")
    args = parser.parse_args()

    print("Building team name lookup...")
    team_map = build_team_map()
    print(f"  Loaded {len(team_map)} team name mappings")

    print(f"\nLoading existing data...")
    existing_bbb = pd.read_csv(BBB_PATH, low_memory=False)
    existing_matches = pd.read_csv(MATCHES_PATH, low_memory=False)
    existing_match_ids = set(existing_bbb["match_id"].unique())
    print(f"  Existing: {len(existing_match_ids)} matches, {len(existing_bbb)} balls")

    print(f"\nDownloading from {CRICKSHEET_URL}...")
    r = requests.get(CRICKSHEET_URL, timeout=120)
    r.raise_for_status()
    print(f"  Got {len(r.content)} bytes")

    z = zipfile.ZipFile(io.BytesIO(r.content))
    info_files = sorted([n for n in z.namelist() if n.endswith("_info.csv")])
    print(f"  Found {len(info_files)} matches in pack")

    new_bbb_rows = []
    new_match_rows = []
    skipped = 0

    for info_fname in info_files:
        match_id_str = info_fname.replace("_info.csv", "")
        try:
            match_id = int(match_id_str)
        except ValueError:
            continue

        if match_id in existing_match_ids:
            continue

        # Parse info
        with z.open(info_fname) as f:
            info = parse_info_csv(f.read().decode("utf-8"))

        if len(info["teams"]) != 2:
            skipped += 1
            continue

        team1_id = lookup_team_id(info["teams"][0], team_map)
        team2_id = lookup_team_id(info["teams"][1], team_map)
        if team1_id is None or team2_id is None:
            print(f"  WARN: unknown team in match {match_id}: {info['teams']}")
            skipped += 1
            continue

        season_id = int(str(info.get("season", "0"))[:4]) if info.get("season") else 0

        # Parse deliveries
        deliveries_fname = f"{match_id}.csv"
        if deliveries_fname not in z.namelist():
            skipped += 1
            continue

        with z.open(deliveries_fname) as f:
            ball_rows = parse_match_csv(
                f.read().decode("utf-8"), match_id, season_id, team_map
            )

        if not ball_rows:
            skipped += 1
            continue

        new_bbb_rows.extend(ball_rows)

        # Match metadata row
        new_match_rows.append({
            "match_id": match_id,
            "season_id": season_id,
            "balls_per_over": int(info.get("balls_per_over", 6) or 6),
            "city": info.get("city", ""),
            "match_date": info.get("match_date", ""),
            "event_name": info.get("event_name", "Indian Premier League"),
            "match_number": info.get("match_number", ""),
            "gender": info.get("gender", "male"),
            "match_type": info.get("match_type", "T20"),
            "format": "T20",
            "overs": int(info.get("overs", 20) or 20),
            "season": info.get("season", ""),
            "team_type": "club",
            "venue": info.get("venue", ""),
            "toss_winner": lookup_team_id(info.get("toss_winner"), team_map),
            "team1": team1_id,
            "team2": team2_id,
            "toss_decision": info.get("toss_decision", ""),
            "match_winner": lookup_team_id(info.get("match_winner"), team_map),
            "win_by_runs": info.get("win_by_runs", ""),
            "win_by_wickets": info.get("win_by_wickets", ""),
            "player_of_match": info.get("player_of_match", ""),
            "result": "win",
        })

    print(f"\n  New matches: {len(new_match_rows)}")
    print(f"  New ball rows: {len(new_bbb_rows)}")
    print(f"  Skipped: {skipped}")

    if new_match_rows:
        latest_date = max(m["match_date"] for m in new_match_rows if m["match_date"])
        print(f"  Latest new match: {latest_date}")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return

    if not new_bbb_rows:
        print("\nNothing new to write.")
        return

    print("\nWriting updates...")
    new_bbb_df = pd.DataFrame(new_bbb_rows)
    new_match_df = pd.DataFrame(new_match_rows)

    # Align columns to existing schema
    new_bbb_df = new_bbb_df.reindex(columns=existing_bbb.columns)
    new_match_df = new_match_df.reindex(columns=existing_matches.columns)

    combined_bbb = pd.concat([existing_bbb, new_bbb_df], ignore_index=True)
    combined_matches = pd.concat([existing_matches, new_match_df], ignore_index=True)

    combined_bbb.to_csv(BBB_PATH, index=False)
    combined_matches.to_csv(MATCHES_PATH, index=False)

    print(f"  Updated {BBB_PATH}: {len(existing_bbb)} -> {len(combined_bbb)} rows")
    print(f"  Updated {MATCHES_PATH}: {len(existing_matches)} -> {len(combined_matches)} rows")
    print("\nDone.")


if __name__ == "__main__":
    main()
