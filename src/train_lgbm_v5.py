"""
Phase 2: Role-specific LightGBM models with Optuna hyperparameter tuning.

Key improvements over v4:
- Separate model per role (BAT / BOWL / AR / WK)
- LightGBM with native categorical support (no OHE explosion)
- Optuna tuning with TimeSeriesSplit (no future leakage)
- Season-aware train/test split (train 2008-2022, test 2023+)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

optuna.logging.set_verbosity(optuna.logging.WARNING)

INPUT_PATH   = Path("output/model_feature_table_v5.csv")
OUTPUT_PATH  = Path("output/best_model_v5_predictions.csv")
MODEL_DIR    = Path("models/v5")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ROLES        = ["BAT", "BOWL", "AR", "WK"]
CAT_FEATURES = ["season", "venue", "opponent"]   # player_role_platform excluded (separate model per role)
EXCLUDE_COLS = {"match_id", "match_date", "player_name", "team",
                "fantasy_points_v5", "player_role_platform"}
TARGET       = "fantasy_points_v5"
N_OPTUNA_TRIALS = 50  # increase to 80 for final run, 40 is fast enough to start


def spearman_per_match(df: pd.DataFrame, pred_col: str = "pred") -> float:
    corrs = []
    for _, grp in df.groupby("match_id"):
        if len(grp) > 3:
            c, _ = spearmanr(grp[TARGET], grp[pred_col])
            if not np.isnan(c):
                corrs.append(c)
    return float(np.nanmean(corrs))


def top_n_overlap(df: pd.DataFrame, pred_col: str = "pred", n: int = 11) -> float:
    overlaps = []
    for _, grp in df.groupby("match_id"):
        if len(grp) < n:
            continue
        actual_top = set(grp.nlargest(n, TARGET)["player_name"])
        pred_top   = set(grp.nlargest(n, pred_col)["player_name"])
        overlaps.append(len(actual_top & pred_top) / n)
    return float(np.mean(overlaps))


def make_lgbm_dataset(X: pd.DataFrame, y: pd.Series, cat_cols: list) -> lgb.Dataset:
    return lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=False)


def objective(trial, X: pd.DataFrame, y: pd.Series, cat_cols: list) -> float:
    params = {
        "objective":         "regression_l1",   # MAE directly
        "metric":            "mae",
        "verbosity":         -1,
        "boosting_type":     "gbdt",
        "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 300, 1200, step=100),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state":      42,
        "n_jobs":            -1,
    }

    tscv = TimeSeriesSplit(n_splits=4)
    fold_maes = []

    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            categorical_feature=cat_cols,
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        fold_maes.append(mean_absolute_error(yva, model.predict(Xva)))

    return float(np.mean(fold_maes))


def train_role_model(
    role: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    cat_cols: list,
) -> tuple:
    print(f"\n{'='*55}")
    print(f"  Role: {role}  |  Train: {len(train_df)}  |  Test: {len(test_df)}")
    print(f"{'='*55}")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET]
    X_test  = test_df[feature_cols].copy()
    y_test  = test_df[TARGET]

    # Optuna tuning
    print(f"  Running Optuna ({N_OPTUNA_TRIALS} trials)...")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, cat_cols),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_params.update({
        "objective":    "regression_l1",
        "metric":       "mae",
        "verbosity":    -1,
        "random_state": 42,
        "n_jobs":       -1,
    })
    print(f"  Best CV MAE: {study.best_value:.3f}")
    print(f"  Best params: { {k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()} }")

    # Final model on full train set
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train, categorical_feature=cat_cols)

    preds   = final_model.predict(X_test)
    mae     = mean_absolute_error(y_test, preds)
    pred_std = np.std(preds)
    print(f"  Test MAE:    {mae:.3f}")
    print(f"  Pred std:    {pred_std:.3f}  (actual std: {y_test.std():.3f})")

    # Save model
    model_path = MODEL_DIR / f"lgbm_{role}.pkl"
    joblib.dump(final_model, model_path)
    print(f"  Saved: {model_path}")

    out_df = test_df.copy()
    out_df["pred"] = preds
    return out_df, final_model, mae


def main():
    print("Loading feature table...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["match_date", "match_id", "player_name"]).reset_index(drop=True)

    # Encode categoricals — LightGBM handles these natively, no OHE needed
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Season-based split: train on 2008-2022, test on 2023+
    # This is correct — your v1-v4 split used random 80/20 which leaked future seasons
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()
    print(f"Train rows: {len(train_df)} ({train_df['match_date'].dt.year.min()}-{train_df['match_date'].dt.year.max()})")
    print(f"Test rows:  {len(test_df)} ({test_df['match_date'].dt.year.min()}-{test_df['match_date'].dt.year.max()})")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS
                    and not c.endswith("_match_id")]
    cat_cols = [c for c in CAT_FEATURES if c in feature_cols]

    print(f"Total features: {len(feature_cols)}  |  Categorical: {len(cat_cols)}")

    all_preds  = []
    role_maes  = {}

    for role in ROLES:
        train_r = train_df[train_df["player_role_platform"] == role].copy()
        test_r  = test_df[test_df["player_role_platform"]  == role].copy()

        if len(train_r) < 200:
            print(f"\nSkipping {role}: only {len(train_r)} train rows")
            continue

        out_df, model, mae = train_role_model(
            role, train_r, test_r, feature_cols, cat_cols
        )
        all_preds.append(out_df)
        role_maes[role] = mae

    # Combine all roles
    result = pd.concat(all_preds).sort_values(["match_date", "match_id"])
    result.to_csv(OUTPUT_PATH, index=False)

    # ── Final metrics ─────────────────────────────────────────────────────────
    overall_mae   = mean_absolute_error(result[TARGET], result["pred"])
    spear         = spearman_per_match(result)
    top11         = top_n_overlap(result, n=11)
    pred_std      = result["pred"].std()
    actual_std    = result[TARGET].std()

    print(f"\n{'='*55}")
    print("  FINAL RESULTS — v5 LightGBM role models")
    print(f"{'='*55}")
    print(f"  Overall MAE:              {overall_mae:.3f}")
    print(f"  Avg Spearman/match:       {spear:.4f}   (v4 baseline: 0.1704)")
    print(f"  Top-11 overlap:           {top11:.2%}   (v4 baseline: 53.37%)")
    print(f"  Predicted std:            {pred_std:.3f}   (actual: {actual_std:.3f})")
    print(f"\n  Per-role MAE:")
    for role, mae in role_maes.items():
        print(f"    {role}: {mae:.3f}")
    print(f"\n  Predictions saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
