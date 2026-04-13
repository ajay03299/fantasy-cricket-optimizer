"""
Production Retraining Pipeline
==============================
Refreshes features, backs up production model, retrains, and rolls back
if the new model is worse than the old one.

Usage:
  python src/training/retrain_pipeline.py            # normal run
  python src/training/retrain_pipeline.py --dry-run  # simulate, no changes
  python src/training/retrain_pipeline.py --force    # keep new model regardless
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
import numpy as np

PROD_MODEL_DIR = Path("models/v11")
BACKUP_BASE = Path("models")
PRED_PATH = Path("output/best_model_v11_predictions.csv")
LOG_PATH = Path("logs/training_log.json")
TARGET = "fantasy_points_v5"
SPEARMAN_TOLERANCE = 0.005  # new must be >= old - 0.005


def per_match_spearman(df):
    corrs = []
    for mid, g in df.groupby("match_id"):
        if len(g) > 3:
            c, _ = spearmanr(g[TARGET], g["pred"])
            if not np.isnan(c):
                corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


def top_k_overlap(df, k=11):
    overlaps = []
    for mid, g in df.groupby("match_id"):
        if len(g) >= k:
            actual = set(g.nlargest(k, TARGET)["player_name"])
            pred = set(g.nlargest(k, "pred")["player_name"])
            overlaps.append(len(actual & pred) / k)
    return float(np.mean(overlaps)) if overlaps else 0.0


def load_metrics(pred_path):
    if not pred_path.exists():
        return None
    df = pd.read_csv(pred_path, low_memory=False)
    return {
        "spearman": per_match_spearman(df),
        "top11": top_k_overlap(df),
        "n_test_rows": len(df),
        "n_matches": df["match_id"].nunique(),
    }


def append_log(entry):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = []
    if LOG_PATH.exists():
        log = json.loads(LOG_PATH.read_text())
    log.append(entry)
    LOG_PATH.write_text(json.dumps(log, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="simulate everything but don't replace files")
    parser.add_argument("--force", action="store_true",
                        help="keep new model regardless of metrics")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  RETRAIN PIPELINE  ({'DRY RUN' if args.dry_run else 'LIVE'})")
    print("=" * 60)

    started = datetime.now()
    backup_dir = BACKUP_BASE / f"v11_backup_{started.strftime('%Y%m%d_%H%M%S')}"

    # Step 1: Refresh features
    print("\n[1/5] Refreshing features...")
    result = subprocess.run(
        ["python", "src/features/refresh_features.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("  FAILED")
        print(result.stderr[-500:])
        sys.exit(1)
    print("  OK")

    # Step 2: Capture old metrics
    print("\n[2/5] Capturing baseline metrics...")
    old_metrics = load_metrics(PRED_PATH)
    if old_metrics is None:
        print("  No baseline predictions found — first training, will keep new model")
    else:
        print(f"  Old: Spearman={old_metrics['spearman']:.4f}  Top11={old_metrics['top11']:.2%}")

    # Step 3: Backup current model
    print(f"\n[3/5] Backing up to {backup_dir}...")
    if PROD_MODEL_DIR.exists():
        if args.dry_run:
            print(f"  [DRY RUN] would copy {PROD_MODEL_DIR} -> {backup_dir}")
        else:
            shutil.copytree(PROD_MODEL_DIR, backup_dir)
            print(f"  Backed up.")
    else:
        print(f"  No existing production model to backup")

    # Step 4: Retrain
    print("\n[4/5] Retraining v11...")
    if args.dry_run:
        print("  [DRY RUN] would run: python src/models/train_v11.py")
    else:
        result = subprocess.run(
            ["python", "src/models/train_v11.py"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("  TRAINING FAILED")
            print(result.stderr[-800:])
            if backup_dir.exists():
                print("  Backup is intact, no rollback needed.")
            sys.exit(1)
        print("  OK")

    # Step 5: Evaluate and decide
    print("\n[5/5] Evaluating new model...")
    if args.dry_run:
        print("  [DRY RUN] would compare new vs old metrics here")
        decision = "dry_run"
        new_metrics = None
    else:
        new_metrics = load_metrics(PRED_PATH)
        if new_metrics is None:
            print("  No new predictions found — something went wrong, rolling back")
            decision = "rolled_back_no_predictions"
            if backup_dir.exists():
                shutil.rmtree(PROD_MODEL_DIR, ignore_errors=True)
                shutil.copytree(backup_dir, PROD_MODEL_DIR)
        else:
            print(f"  New: Spearman={new_metrics['spearman']:.4f}  Top11={new_metrics['top11']:.2%}")

            if old_metrics is None or args.force:
                decision = "kept_forced" if args.force else "kept_first_run"
                print(f"  Decision: KEPT ({decision})")
            elif new_metrics['spearman'] >= old_metrics['spearman'] - SPEARMAN_TOLERANCE:
                decision = "kept"
                delta = new_metrics['spearman'] - old_metrics['spearman']
                print(f"  Decision: KEPT (delta {delta:+.4f}, within tolerance)")
            else:
                decision = "rolled_back"
                delta = new_metrics['spearman'] - old_metrics['spearman']
                print(f"  Decision: ROLLED BACK (delta {delta:+.4f}, below tolerance)")
                shutil.rmtree(PROD_MODEL_DIR, ignore_errors=True)
                shutil.copytree(backup_dir, PROD_MODEL_DIR)

    # Log
    entry = {
        "timestamp": started.isoformat(),
        "duration_seconds": (datetime.now() - started).total_seconds(),
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "decision": decision,
        "backup_dir": str(backup_dir),
        "dry_run": args.dry_run,
    }
    append_log(entry)

    print(f"\n{'='*60}")
    print(f"  Done in {entry['duration_seconds']:.0f}s. Decision: {decision}")
    print(f"  Log: {LOG_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
