"""
Feature Refresh Orchestrator
============================
Rebuilds feature tables (v8 -> v9 -> v10) when new ball-by-ball data
has been ingested. Validates each step for duplicates and logs results.

Usage:
  python src/features/refresh_features.py            # rebuild if stale
  python src/features/refresh_features.py --force    # always rebuild
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_BBB = Path("data/all_ball_by_ball_data.csv")
FT_V10 = Path("output/model_feature_table_v10.csv")
LOG_PATH = Path("output/.feature_refresh_log.json")

STEPS = [
    ("v8 build",       ["python", "src/features/build_feature_table_v8.py"], "output/model_feature_table_v8.csv"),
    ("v8 dedup",       ["python", "src/features/fix_v8_duplicates.py"],      "output/model_feature_table_v8.csv"),
    ("v9 build",       ["python", "src/features/build_feature_table_v9.py"], "output/model_feature_table_v9.csv"),
    ("v10 build",      ["python", "src/features/build_feature_table_v10.py"],"output/model_feature_table_v10.csv"),
]


def needs_rebuild():
    if not FT_V10.exists():
        return True, "v10 feature table missing"
    bbb_mtime = DATA_BBB.stat().st_mtime
    v10_mtime = FT_V10.stat().st_mtime
    if bbb_mtime > v10_mtime:
        return True, f"ball-by-ball is newer than v10 ({bbb_mtime - v10_mtime:.0f}s)"
    return False, "features are current"


def validate(csv_path, step_name):
    df = pd.read_csv(csv_path, low_memory=False)
    n_rows = len(df)
    n_dups = df.duplicated(subset=["match_id", "player_name"]).sum() if "player_name" in df.columns else 0
    if n_dups > 0:
        print(f"  WARN: {n_dups} duplicates in {csv_path.name} after {step_name}")
    return n_rows, int(n_dups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild even if current")
    args = parser.parse_args()

    print("=" * 60)
    print("  FEATURE REFRESH")
    print("=" * 60)

    rebuild, reason = needs_rebuild()
    print(f"\n  Status: {reason}")

    if not rebuild and not args.force:
        print("  Nothing to do. Use --force to rebuild anyway.")
        return

    if args.force:
        print("  --force flag: rebuilding regardless")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "trigger": "force" if args.force else reason,
        "steps": []
    }

    for step_name, cmd, output_csv in STEPS:
        print(f"\n[{step_name}] running...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  FAILED")
            print(result.stdout[-500:])
            print(result.stderr[-500:])
            sys.exit(1)

        n_rows, n_dups = validate(Path(output_csv), step_name)
        print(f"  OK: {n_rows} rows, {n_dups} dups")

        log_entry["steps"].append({
            "name": step_name,
            "rows": n_rows,
            "duplicates": n_dups,
            "output": output_csv,
        })

    # Append to log
    log = []
    if LOG_PATH.exists():
        log = json.loads(LOG_PATH.read_text())
    log.append(log_entry)
    LOG_PATH.write_text(json.dumps(log, indent=2))

    print(f"\n{'='*60}")
    print(f"  All steps complete. Final v10: {log_entry['steps'][-1]['rows']} rows")
    print(f"  Log: {LOG_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
