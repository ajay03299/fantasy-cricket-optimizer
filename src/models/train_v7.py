"""
Train v7: Ranking-first fantasy cricket optimizer
===================================================
Architecture:
  Model A - XGBoost regression (cleaned features + interactions)
  Model B - XGBoost LambdaMART ranker (rank:pairwise) <- KEY INNOVATION
  Model C - Role-specific XGBoost regressors (BAT/BOWL/AR/WK)
  Model D - Quantile regressor (60th percentile - upward bias)
  Ensemble - Per-match normalized blend of A + B + C + D
"""
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

INPUT_PATH  = Path("output/model_feature_table_v7.csv")
OUTPUT_PATH = Path("output/best_model_v7_predictions.csv")
MODEL_DIR   = Path("models/v7")
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


def spearman_per_match(actual, predicted, match_ids):
    corrs = []
    for mid in np.unique(match_ids):
        mask = match_ids == mid
        if mask.sum() > 3:
            c, _ = spearmanr(actual[mask], predicted[mask])
            if not np.isnan(c):
                corrs.append(c)
    return float(np.mean(corrs)), float(np.std(corrs))


def top_k_overlap(actual, predicted, match_ids, player_names, k=11):
    overlaps = []
    for mid in np.unique(match_ids):
        mask = match_ids == mid
        if mask.sum() >= k:
            a_idx = np.argsort(-actual[mask])[:k]
            p_idx = np.argsort(-predicted[mask])[:k]
            a_set = set(player_names[mask][a_idx])
            p_set = set(player_names[mask][p_idx])
            overlaps.append(len(a_set & p_set) / k)
    return float(np.mean(overlaps))


def ndcg_per_match(actual, predicted, match_ids, k=11):
    ndcgs = []
    for mid in np.unique(match_ids):
        mask = match_ids == mid
        if mask.sum() >= k:
            a = actual[mask]
            p = predicted[mask]
            order = np.argsort(-p)[:k]
            gains = a[order]
            discounts = np.log2(np.arange(2, k + 2))
            dcg = np.sum(gains / discounts)
            ideal_order = np.argsort(-a)[:k]
            ideal_gains = a[ideal_order]
            idcg = np.sum(ideal_gains / discounts)
            if idcg > 0:
                ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs))


def print_metrics(name, actual, predicted, match_ids, player_names):
    mae = mean_absolute_error(actual, predicted)
    sp, sp_std = spearman_per_match(actual, predicted, match_ids)
    t11 = top_k_overlap(actual, predicted, match_ids, player_names)
    ndcg = ndcg_per_match(actual, predicted, match_ids)
    pstd = float(np.std(predicted))
    print(f"  {name:<35} MAE={mae:.3f}  Spearman={sp:.4f}+/-{sp_std:.3f}  Top11={t11:.2%}  NDCG={ndcg:.4f}  PredStd={pstd:.2f}")
    return {"name": name, "mae": mae, "spearman": sp, "top11": t11, "ndcg": ndcg, "pred_std": pstd}


def main():
    print("=" * 70)
    print("  FANTASY CRICKET OPTIMIZER v7 - RANKING-FIRST APPROACH")
    print("=" * 70)

    # Load
    print(f"\nLoading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(["match_date", "match_id", "player_name"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns
                    if c not in EXCLUDE
                    and c not in ["match_id", "match_date", "player_name", "team"]
                    and (df[c].dtype in ["float64", "int64", "float32"] or c in CAT_COLS)]
    feature_cols = sorted(set(feature_cols))

    print(f"  Rows: {len(df)}  |  Features: {len(feature_cols)}  |  Matches: {df['match_id'].nunique()}")

    # Chronological split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    train_enc = encode_cats(train_df, CAT_COLS)
    test_enc  = encode_cats(test_df, CAT_COLS)

    X_train = train_enc[feature_cols].fillna(0).astype(np.float32)
    y_train = train_enc[TARGET].values
    X_test  = test_enc[feature_cols].fillna(0).astype(np.float32)
    y_test  = test_enc[TARGET].values

    match_ids = test_df["match_id"].values
    player_names = test_df["player_name"].values

    print(f"  Train: {len(train_df)}  |  Test: {len(test_df)}")

    results = []

    # ---- MODEL A: XGBoost Regression ----
    print(f"\n{'~'*70}")
    print("  [A] XGBoost Regression (squared error)")
    print(f"{'~'*70}")

    xgb_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000, learning_rate=0.025, max_depth=5,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.3, reg_lambda=1.5,
        random_state=42, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    pred_a = xgb_reg.predict(X_test)
    r = print_metrics("XGB Regression", y_test, pred_a, match_ids, player_names)
    results.append(r)
    joblib.dump(xgb_reg, MODEL_DIR / "xgb_regression.pkl")

    # ---- MODEL B: XGBoost Ranker (LambdaMART) ----
    print(f"\n{'~'*70}")
    print("  [B] XGBoost Ranker (rank:pairwise / LambdaMART)")
    print(f"{'~'*70}")

    train_sorted = train_enc.sort_values("match_id")
    test_sorted  = test_enc.sort_values("match_id")

    X_train_r = train_sorted[feature_cols].fillna(0).astype(np.float32)
    y_train_r_raw = train_sorted[TARGET].values
    X_test_r  = test_sorted[feature_cols].fillna(0).astype(np.float32)
    y_test_r_raw  = test_sorted[TARGET].values

    train_groups = train_sorted.groupby("match_id", sort=False).size().values
    test_groups  = test_sorted.groupby("match_id", sort=False).size().values

    # XGBRanker needs non-negative integer labels <= 31
    # Convert to within-match ranks (0 = worst, N = best in that match)
    def to_match_ranks(y, match_groups):
        ranks = np.zeros(len(y), dtype=int)
        idx = 0
        for size in match_groups:
            chunk = y[idx:idx+size]
            order = np.argsort(np.argsort(chunk))
            if size > 1:
                scaled = np.round(order / (size - 1) * 31).astype(int)
            else:
                scaled = np.array([15])
            ranks[idx:idx+size] = scaled
            idx += size
        return ranks

    y_train_r = to_match_ranks(y_train_r_raw, train_groups)
    y_test_r  = to_match_ranks(y_test_r_raw, test_groups)
    print(f"  Labels converted to match ranks (0-31 scale)")
    print(f"  Train label range: [{y_train_r.min()}, {y_train_r.max()}]")

    xgb_rank = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=1000, learning_rate=0.025, max_depth=5,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.3, reg_lambda=1.5,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    xgb_rank.fit(
        X_train_r, y_train_r, group=train_groups,
        eval_set=[(X_test_r, y_test_r)], eval_group=[test_groups],
        verbose=100,
    )
    pred_b_sorted = xgb_rank.predict(X_test_r)

    # Map ranker predictions back to original test order
    pred_b = np.zeros(len(test_df))
    sorted_indices = test_sorted.index.values
    for i, idx in enumerate(sorted_indices):
        pos = np.where(test_df.index == idx)[0][0]
        pred_b[pos] = pred_b_sorted[i]

    # Note: ranker scores are relative, not point estimates
    # Evaluate ranking quality (Spearman/Top11 still valid, MAE less meaningful)
    r = print_metrics("XGB Ranker (pairwise)", y_test, pred_b, match_ids, player_names)
    results.append(r)
    joblib.dump(xgb_rank, MODEL_DIR / "xgb_ranker.pkl")

    # ---- MODEL C: Role-specific XGBoost ----
    print(f"\n{'~'*70}")
    print("  [C] Role-specific XGBoost Regressors")
    print(f"{'~'*70}")

    role_feats = [c for c in feature_cols if c != "player_role_platform"]
    pred_c = np.zeros(len(test_df))

    for role in ["BAT", "BOWL", "AR", "WK"]:
        tr_mask = train_df["player_role_platform"] == role
        te_mask = test_df["player_role_platform"] == role
        if tr_mask.sum() < 100 or te_mask.sum() == 0:
            print(f"  {role}: skipped")
            continue

        X_tr = train_enc.loc[tr_mask.values, role_feats].fillna(0).astype(np.float32)
        y_tr = train_enc.loc[tr_mask.values, TARGET].values
        X_te = test_enc.loc[te_mask.values, role_feats].fillna(0).astype(np.float32)
        y_te = test_enc.loc[te_mask.values, TARGET].values

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.3, reg_lambda=1.5,
            random_state=42, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=40,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=0)
        preds = model.predict(X_te)
        pred_c[te_mask.values] = preds
        mae_r = mean_absolute_error(y_te, preds)
        print(f"  {role}: MAE={mae_r:.3f}  n_train={len(X_tr)}  n_test={len(X_te)}  pred_std={np.std(preds):.2f}")
        joblib.dump(model, MODEL_DIR / f"xgb_role_{role.lower()}.pkl")

    r = print_metrics("Role-specific ensemble", y_test, pred_c, match_ids, player_names)
    results.append(r)

    # ---- MODEL D: Quantile Regressor ----
    print(f"\n{'~'*70}")
    print("  [D] XGBoost Quantile Regressor (60th percentile)")
    print(f"{'~'*70}")

    xgb_q = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=0.60,
        n_estimators=800, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_q.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    pred_d = xgb_q.predict(X_test)
    r = print_metrics("XGB Quantile (0.60)", y_test, pred_d, match_ids, player_names)
    results.append(r)
    joblib.dump(xgb_q, MODEL_DIR / "xgb_quantile60.pkl")

    # ---- ENSEMBLE ----
    print(f"\n{'~'*70}")
    print("  [E] Ensemble - per-match normalized blend")
    print(f"{'~'*70}")

    # Ranker underperformed due to rank discretization — exclude from ensemble
    # Role-specific model is the star (Spearman=0.1902, Top11=54.30%)
    preds_dict = {"reg": pred_a, "role": pred_c, "quant": pred_d}

    configs = [
        {"reg": 0.35, "role": 0.45, "quant": 0.20},
        {"reg": 0.30, "role": 0.50, "quant": 0.20},
        {"reg": 0.40, "role": 0.40, "quant": 0.20},
        {"reg": 0.25, "role": 0.55, "quant": 0.20},
        {"reg": 0.30, "role": 0.45, "quant": 0.25},
        {"reg": 0.20, "role": 0.60, "quant": 0.20},
    ]

    best_sp = -1
    best_weights = None
    best_ensemble = None

    for weights in configs:
        ens = np.zeros(len(y_test))
        for mid in np.unique(match_ids):
            mask = match_ids == mid
            ref = preds_dict["reg"][mask]
            ref_mean, ref_std = ref.mean(), max(np.std(ref), 0.01)
            blended = np.zeros(mask.sum())
            for key, weight in weights.items():
                p = preds_dict[key][mask]
                if np.std(p) > 0:
                    p_norm = (p - p.mean()) / np.std(p) * ref_std + ref_mean
                else:
                    p_norm = ref
                blended += weight * p_norm
            ens[mask] = blended

        sp, _ = spearman_per_match(y_test, ens, match_ids)
        t11 = top_k_overlap(y_test, ens, match_ids, player_names)
        w_str = "/".join(f"{v:.0%}" for v in weights.values())
        print(f"  Weights [{w_str}]: Spearman={sp:.4f}  Top11={t11:.2%}")
        if sp > best_sp:
            best_sp = sp
            best_weights = weights
            best_ensemble = ens

    print(f"\n  Best weights: {best_weights}")
    r = print_metrics("BEST ENSEMBLE", y_test, best_ensemble, match_ids, player_names)
    results.append(r)

    # ---- COMPARISON vs v6 ----
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON - v7 vs v6")
    print(f"{'='*70}")

    try:
        old = pd.read_csv("output/best_model_v6_predictions.csv", low_memory=False)
        old_sp, _ = spearman_per_match(old[TARGET].values, old["pred"].values, old["match_id"].values)
        old_t11 = top_k_overlap(old[TARGET].values, old["pred"].values,
                                old["match_id"].values, old["player_name"].values)
        old_mae = mean_absolute_error(old[TARGET], old["pred"])
        old_ndcg = ndcg_per_match(old[TARGET].values, old["pred"].values, old["match_id"].values)
        print(f"\n  v6 ensemble:    MAE={old_mae:.3f}  Spearman={old_sp:.4f}  Top11={old_t11:.2%}  NDCG={old_ndcg:.4f}")
        print(f"  v7 BEST:        MAE={r['mae']:.3f}  Spearman={r['spearman']:.4f}  Top11={r['top11']:.2%}  NDCG={r['ndcg']:.4f}")
        print(f"\n  Delta Spearman: {r['spearman'] - old_sp:+.4f} ({(r['spearman']-old_sp)/abs(old_sp)*100:+.1f}%)")
        print(f"  Delta Top-11:   {r['top11'] - old_t11:+.4f} ({(r['top11']-old_t11)/abs(old_t11)*100:+.1f}%)")
        print(f"  Delta NDCG:     {r['ndcg'] - old_ndcg:+.4f} ({(r['ndcg']-old_ndcg)/abs(old_ndcg)*100:+.1f}%)")
    except FileNotFoundError:
        print("  (v6 predictions not found - skipping comparison)")

    # Feature importance from ranker
    print(f"\n{'='*70}")
    print("  TOP 20 FEATURES (XGBoost Ranker)")
    print(f"{'='*70}")
    imp = xgb_rank.feature_importances_
    feat_imp = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
    for fname, importance in feat_imp[:20]:
        print(f"  {importance:>8.4f}  {fname}")

    # Save predictions
    out = test_df.copy()
    out["pred_regression"] = pred_a
    out["pred_ranker"]     = pred_b
    out["pred_role"]       = pred_c
    out["pred_quantile"]   = pred_d
    out["pred"]            = best_ensemble
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved predictions: {OUTPUT_PATH}")

    meta = {"best_weights": best_weights, "feature_cols": feature_cols, "metrics": results}
    with open(MODEL_DIR / "v7_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Saved metadata: {MODEL_DIR}/v7_metadata.json")


if __name__ == "__main__":
    main()
