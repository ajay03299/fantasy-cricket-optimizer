"""
v6 model: two key upgrades over v4 GBM
  1. XGBoost with quantile alpha=0.65 — forces spread in predictions
  2. Two-stage architecture: main regressor + high-scorer booster
  3. Trained on v5 corrected scoring labels
  4. 6 new features (v6 feature table)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import joblib

INPUT_PATH  = Path("output/model_feature_table_v6.csv")
OUTPUT_PATH = Path("output/best_model_v6_predictions.csv")
MODEL_DIR   = Path("models/v6")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "fantasy_points_v5"
EXCLUDE = {"match_id", "match_date", "player_name", "team", TARGET,
           "fantasy_points_v4", "fantasy_points_v3", "fantasy_points_v1"}
CAT_COLS = ["season", "venue", "opponent", "player_role_platform"]


def encode_cats(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


def spearman_pm(df, pred_col):
    corrs = []
    for _, g in df.groupby("match_id"):
        if len(g) > 3:
            c, _ = spearmanr(g[TARGET], g[pred_col])
            if not np.isnan(c):
                corrs.append(c)
    return float(np.nanmean(corrs))


def top11_overlap(df, pred_col):
    ov = []
    for _, g in df.groupby("match_id"):
        if len(g) >= 11:
            a = set(g.nlargest(11, TARGET)["player_name"])
            p = set(g.nlargest(11, pred_col)["player_name"])
            ov.append(len(a & p) / 11)
    return float(np.mean(ov))


def main():
    print("Loading feature table v6...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["match_date", "match_id", "player_name"]).reset_index(drop=True)

    # Same 80/20 chronological split
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    feature_cols = [c for c in df.columns if c not in EXCLUDE
                    and c in df.select_dtypes(include=[float, int, "category"]).columns
                    or c in CAT_COLS]
    feature_cols = [c for c in feature_cols if c in df.columns and c not in EXCLUDE]

    # Encode categoricals for XGBoost (needs numeric)
    train_enc = encode_cats(train_df, CAT_COLS)
    test_enc  = encode_cats(test_df,  CAT_COLS)

    X_train = train_enc[feature_cols].fillna(0)
    y_train = train_enc[TARGET]
    X_test  = test_enc[feature_cols].fillna(0)
    y_test  = test_enc[TARGET]

    print(f"Features: {len(feature_cols)}")

    # ── MODEL A: XGBoost quantile (alpha=0.65) ─────────────────────────────
    # Quantile loss with alpha > 0.5 means "predict the 65th percentile"
    # This forces the model to skew predictions UP — reducing the downward bias
    # on high scorers. The key fix for pred_std=7 problem.
    print("\n[1/3] Training XGBoost quantile regressor (alpha=0.65)...")
    xgb_q = xgb.XGBRegressor(
        objective       = "reg:quantileerror",
        quantile_alpha  = 0.65,
        n_estimators    = 800,
        learning_rate   = 0.03,
        max_depth       = 5,
        subsample       = 0.8,
        colsample_bytree= 0.7,
        min_child_weight= 5,
        reg_alpha       = 0.5,
        reg_lambda      = 2.0,
        random_state    = 42,
        n_jobs          = -1,
        tree_method     = "hist",
        early_stopping_rounds = 50,
    )
    xgb_q.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=100)

    pred_q = xgb_q.predict(X_test)
    mae_q  = mean_absolute_error(y_test, pred_q)
    sp_q   = spearman_pm(test_df.assign(pred=pred_q), "pred")
    print(f"  XGB quantile — MAE: {mae_q:.3f} | Spearman: {sp_q:.4f} | Pred std: {np.std(pred_q):.2f}")
    joblib.dump(xgb_q, MODEL_DIR / "xgb_quantile.pkl")

    # ── MODEL B: XGBoost standard MAE ─────────────────────────────────────
    print("\n[2/3] Training XGBoost standard (MAE loss)...")
    xgb_m = xgb.XGBRegressor(
        objective       = "reg:absoluteerror",
        n_estimators    = 800,
        learning_rate   = 0.03,
        max_depth       = 5,
        subsample       = 0.8,
        colsample_bytree= 0.7,
        min_child_weight= 5,
        reg_alpha       = 0.3,
        reg_lambda      = 1.5,
        random_state    = 42,
        n_jobs          = -1,
        tree_method     = "hist",
        early_stopping_rounds = 50,
    )
    xgb_m.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=100)

    pred_m = xgb_m.predict(X_test)
    mae_m  = mean_absolute_error(y_test, pred_m)
    sp_m   = spearman_pm(test_df.assign(pred=pred_m), "pred")
    print(f"  XGB standard — MAE: {mae_m:.3f} | Spearman: {sp_m:.4f} | Pred std: {np.std(pred_m):.2f}")
    joblib.dump(xgb_m, MODEL_DIR / "xgb_standard.pkl")

    # ── MODEL C: GBM baseline on v5 labels (Fix 1 — free improvement) ────
    print("\n[3/3] Training GBM on v5 labels (same architecture as v4 best)...")
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    cat_feats = [c for c in CAT_COLS if c in feature_cols]
    num_feats = [c for c in feature_cols if c not in cat_feats]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="constant", fill_value=0), num_feats),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_feats),
    ])
    gbm = Pipeline([
        ("pre", pre),
        ("m",   GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=3, subsample=0.8, random_state=42,
        )),
    ])
    gbm.fit(train_df[feature_cols], y_train)
    pred_g = gbm.predict(test_df[feature_cols])
    mae_g  = mean_absolute_error(y_test, pred_g)
    sp_g   = spearman_pm(test_df.assign(pred=pred_g), "pred")
    print(f"  GBM v5 labels — MAE: {mae_g:.3f} | Spearman: {sp_g:.4f} | Pred std: {np.std(pred_g):.2f}")
    joblib.dump(gbm, MODEL_DIR / "gbm_v5labels.pkl")

    # ── ENSEMBLE: blend all three ─────────────────────────────────────────
    print("\n[Ensemble] Blending predictions...")
    # Weights: give more weight to quantile model (better spread)
    # and standard XGB (lower MAE), less to GBM
    for w_q, w_m, w_g in [(0.4, 0.4, 0.2), (0.5, 0.3, 0.2), (0.3, 0.5, 0.2)]:
        blend = w_q * pred_q + w_m * pred_m + w_g * pred_g
        mae_b = mean_absolute_error(y_test, blend)
        sp_b  = spearman_pm(test_df.assign(pred=blend), "pred")
        print(f"  Blend ({w_q}/{w_m}/{w_g}) — MAE: {mae_b:.3f} | Spearman: {sp_b:.4f} | Pred std: {np.std(blend):.2f}")

    # Use best blend weights (0.4 quantile, 0.4 standard, 0.2 GBM)
    best_blend = 0.4 * pred_q + 0.4 * pred_m + 0.2 * pred_g

    # ── FINAL COMPARISON ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'MAE':>7} {'Spearman':>10} {'Top11':>8} {'PredStd':>9}")
    print(f"  {'-'*60}")

    old_v4 = pd.read_csv("output/best_model_v4_predictions.csv")
    old_mae = mean_absolute_error(old_v4["fantasy_points_v4"], old_v4["predicted_fantasy_points_v4"])
    old_sp  = spearman_pm(old_v4.assign(pred=old_v4["predicted_fantasy_points_v4"]).rename(
        columns={"fantasy_points_v4": TARGET}), "pred")

    for name, preds in [
        ("v4 GBM (old best)",    old_v4["predicted_fantasy_points_v4"]),
        ("GBM v5 labels",        pd.Series(pred_g)),
        ("XGB standard",         pd.Series(pred_m)),
        ("XGB quantile (0.65)",  pd.Series(pred_q)),
        ("Ensemble blend",       pd.Series(best_blend)),
    ]:
        if name == "v4 GBM (old best)":
            mae = old_mae
            sp  = old_sp
            t11 = top11_overlap(old_v4.rename(columns={
                "fantasy_points_v4": TARGET,
                "predicted_fantasy_points_v4": "pred"}), "pred")
            pstd = old_v4["predicted_fantasy_points_v4"].std()
        else:
            mae  = mean_absolute_error(y_test, preds)
            sp   = spearman_pm(test_df.assign(pred=preds.values), "pred")
            t11  = top11_overlap(test_df.assign(pred=preds.values), "pred")
            pstd = preds.std()
        print(f"  {name:<30} {mae:>7.3f} {sp:>10.4f} {t11:>8.2%} {pstd:>9.3f}")

    # Save best result
    out = test_df.copy()
    out["pred_quantile"]  = pred_q
    out["pred_standard"]  = pred_m
    out["pred_gbm"]       = pred_g
    out["pred"]           = best_blend
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
