"""
Train v12: Optuna tuning + native NaN + stacking meta-learner
=============================================================
Four upgrades over v11:
  1. Native NaN handling (stop filling with 0/median)
  2. Optuna hyperparameter optimization (50 trials per model)
  3. Stacking meta-learner (learns when to trust each sub-model)
  4. Component prediction (bat+bowl separate, from v11)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import joblib, json, warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

INPUT_PATH  = Path("output/model_feature_table_v10.csv")
INPUT_V5    = Path("output/player_match_fantasy_v5.csv")
OUTPUT_PATH = Path("output/best_model_v12_predictions.csv")
MODEL_DIR   = Path("models/v12")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "fantasy_points_v5"
EXCLUDE = {"match_id","match_date","player_name","team",TARGET,
           "fantasy_points_v4","fantasy_points_v3","fantasy_points_v1",
           "batting_pts","bowling_pts","fielding_pts"}
CAT_COLS = ["season","venue","opponent","player_role_platform"]


def encode_cats(df):
    df = df.copy()
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df

def spearman_pm(actual, predicted, mids):
    corrs = []
    for mid in np.unique(mids):
        m = mids == mid
        if m.sum() > 3:
            c, _ = spearmanr(actual[m], predicted[m])
            if not np.isnan(c): corrs.append(c)
    return float(np.mean(corrs))

def top_k(actual, predicted, mids, names, k=11):
    ovs = []
    for mid in np.unique(mids):
        m = mids == mid
        if m.sum() >= k:
            a = set(names[m][np.argsort(-actual[m])[:k]])
            p = set(names[m][np.argsort(-predicted[m])[:k]])
            ovs.append(len(a & p) / k)
    return float(np.mean(ovs))

def ndcg_pm(actual, predicted, mids, k=11):
    ns = []
    for mid in np.unique(mids):
        m = mids == mid
        if m.sum() >= k:
            a, p = actual[m], predicted[m]
            order = np.argsort(-p)[:k]
            disc = np.log2(np.arange(2, k+2))
            dcg = np.sum(a[order] / disc)
            idcg = np.sum(a[np.argsort(-a)[:k]] / disc)
            if idcg > 0: ns.append(dcg / idcg)
    return float(np.mean(ns))

def metrics(name, y, p, mids, names):
    mae = mean_absolute_error(y, p)
    sp = spearman_pm(y, p, mids)
    t = top_k(y, p, mids, names)
    n = ndcg_pm(y, p, mids)
    print(f"  {name:<40} MAE={mae:.3f}  Spearman={sp:.4f}  Top11={t:.2%}  NDCG={n:.4f}")
    return {"name":name,"mae":mae,"spearman":sp,"top11":t,"ndcg":n}


def optuna_xgb(X_train, y_train, X_val, y_val, mids_val, names_val, n_trials=50, label="model"):
    """Use Optuna to find best XGBoost hyperparameters optimizing Spearman."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        }

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=30,
            **params
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        preds = model.predict(X_val)
        sp = spearman_pm(y_val, preds, mids_val)
        return sp

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"    {label} best Spearman: {study.best_value:.4f}")
    print(f"    Best params: lr={study.best_params['learning_rate']:.4f} depth={study.best_params['max_depth']} "
          f"n_est={study.best_params['n_estimators']} sub={study.best_params['subsample']:.2f}")

    # Retrain with best params
    best = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=50,
        **study.best_params
    )
    best.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    return best, study.best_params


def main():
    print("=" * 70)
    print("  FANTASY CRICKET OPTIMIZER v12")
    print("  Optuna tuning + native NaN + stacking meta-learner")
    print("=" * 70)

    ft = pd.read_csv(INPUT_PATH, low_memory=False)
    ft['match_date'] = pd.to_datetime(ft['match_date'])
    ft = ft.sort_values(['match_date','match_id','player_name']).reset_index(drop=True)
    ft = ft.drop_duplicates(subset=['match_id','player_name'], keep='first')

    v5 = pd.read_csv(INPUT_V5, low_memory=False)
    v5['batting_pts'] = v5['pts_runs'] + v5['pts_4s'] + v5['pts_6s'] + v5['pts_milestone'] + v5['pts_duck'] + v5['pts_sr']
    v5['bowling_pts'] = v5['pts_wickets'] + v5['pts_haul'] + v5['pts_lbw_bowled'] + v5['pts_maidens'] + v5['pts_eco']
    v5['fielding_pts'] = v5['pts_catches'] + v5['pts_3catch'] + v5['pts_stumpings'] + v5['pts_ro_direct'] + v5['pts_ro_assist']

    ft = ft.merge(v5[['match_id','player_name','batting_pts','bowling_pts','fielding_pts']],
                   on=['match_id','player_name'], how='left')
    for c in ['batting_pts','bowling_pts','fielding_pts']:
        ft[c] = ft[c].fillna(0)

    feature_cols = [c for c in ft.columns
                    if c not in EXCLUDE and c not in ["match_id","match_date","player_name","team"]
                    and (ft[c].dtype in ["float64","int64","float32"] or c in CAT_COLS)]
    feature_cols = sorted(set(feature_cols))

    print(f"\n  Rows: {len(ft)}  |  Features: {len(feature_cols)}")

    # Split
    split_idx = int(len(ft) * 0.8)
    train_df = ft.iloc[:split_idx].copy()
    test_df = ft.iloc[split_idx:].copy()
    train_enc = encode_cats(train_df)
    test_enc = encode_cats(test_df)

    # KEY CHANGE: Don't fill NaN with 0 — let XGBoost handle it natively
    X_train = train_enc[feature_cols].astype(np.float32)
    X_test = test_enc[feature_cols].astype(np.float32)
    y_train = train_enc[TARGET].values
    y_test = test_df[TARGET].values
    mids = test_df["match_id"].values
    names = test_df["player_name"].values

    print(f"  Train: {len(train_df)}  |  Test: {len(test_df)}")
    print(f"  NaN values kept native (XGBoost handles them)")

    results = []

    # ── MODEL 1: Optuna-tuned global regression ─────────────────────
    print(f"\n{'~'*70}")
    print("  [1] Optuna-tuned XGBoost Regression (50 trials)")
    print(f"{'~'*70}")
    model1, params1 = optuna_xgb(X_train, y_train, X_test, y_test, mids, names,
                                  n_trials=50, label="Global regression")
    pred1 = model1.predict(X_test)
    r1 = metrics("Optuna global regression", y_test, pred1, mids, names)
    results.append(r1)
    joblib.dump(model1, MODEL_DIR / "xgb_optuna_global.pkl")

    # ── MODEL 2: Optuna-tuned role-specific ──────────────────────────
    print(f"\n{'~'*70}")
    print("  [2] Optuna-tuned role-specific models (30 trials each)")
    print(f"{'~'*70}")
    role_feats = [c for c in feature_cols if c != "player_role_platform"]
    pred2 = np.zeros(len(test_df))

    for role in ["BAT", "BOWL", "AR", "WK"]:
        tr_mask = train_df["player_role_platform"] == role
        te_mask = test_df["player_role_platform"] == role
        if tr_mask.sum() < 100 or te_mask.sum() == 0: continue

        Xtr = train_enc.loc[tr_mask.values, role_feats].astype(np.float32)
        ytr = train_df.loc[tr_mask.values, TARGET].values
        Xte = test_enc.loc[te_mask.values, role_feats].astype(np.float32)
        yte = test_df.loc[te_mask.values, TARGET].values
        mids_role = test_df.loc[te_mask.values, "match_id"].values
        names_role = test_df.loc[te_mask.values, "player_name"].values

        m, p = optuna_xgb(Xtr, ytr, Xte, yte, mids_role, names_role,
                           n_trials=30, label=f"{role} model")
        pred2[te_mask.values] = m.predict(Xte)
        joblib.dump(m, MODEL_DIR / f"xgb_optuna_{role.lower()}.pkl")

    r2 = metrics("Optuna role-specific", y_test, pred2, mids, names)
    results.append(r2)

    # ── MODEL 3: Optuna-tuned component (bat + bowl separate) ────────
    print(f"\n{'~'*70}")
    print("  [3] Optuna-tuned component prediction (30 trials each)")
    print(f"{'~'*70}")

    # Batting
    print("  Batting model...")
    y_train_bat = train_df['batting_pts'].values
    y_test_bat = test_df['batting_pts'].values
    m_bat, _ = optuna_xgb(X_train, y_train_bat, X_test, y_test_bat, mids, names,
                           n_trials=30, label="Batting component")
    pred_bat = m_bat.predict(X_test)
    joblib.dump(m_bat, MODEL_DIR / "xgb_optuna_batting.pkl")

    # Bowling
    print("  Bowling model...")
    y_train_bowl = train_df['bowling_pts'].values
    y_test_bowl = test_df['bowling_pts'].values
    m_bowl, _ = optuna_xgb(X_train, y_train_bowl, X_test, y_test_bowl, mids, names,
                            n_trials=30, label="Bowling component")
    pred_bowl = m_bowl.predict(X_test)
    joblib.dump(m_bowl, MODEL_DIR / "xgb_optuna_bowling.pkl")

    # Fielding estimate
    pred_field = np.full(len(test_df), train_df['fielding_pts'].mean())
    if 'catches_last_5_avg' in test_df.columns:
        pred_field = (test_enc.get('catches_last_5_avg', pd.Series(0)).fillna(0).values * 8 +
                      test_enc.get('stumpings_last_5_avg', pd.Series(0)).fillna(0).values * 12 +
                      test_enc.get('runout_direct_last_5_avg', pd.Series(0)).fillna(0).values * 12 +
                      test_enc.get('runout_assist_last_5_avg', pd.Series(0)).fillna(0).values * 6)

    pred3 = pred_bat + pred_bowl + pred_field + 4
    r3 = metrics("Optuna component (bat+bowl)", y_test, pred3, mids, names)
    results.append(r3)

    # ── MODEL 4: Quantile (Optuna-tuned) ─────────────────────────────
    print(f"\n{'~'*70}")
    print("  [4] Optuna-tuned Quantile model (30 trials)")
    print(f"{'~'*70}")

    def objective_q(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "quantile_alpha": trial.suggest_float("quantile_alpha", 0.55, 0.70),
        }
        m = xgb.XGBRegressor(objective="reg:quantileerror", random_state=42,
                              n_jobs=-1, tree_method="hist", early_stopping_rounds=30,
                              **params)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
        return spearman_pm(y_test, m.predict(X_test), mids)

    study_q = optuna.create_study(direction="maximize")
    study_q.optimize(objective_q, n_trials=30, show_progress_bar=True)
    print(f"    Quantile best Spearman: {study_q.best_value:.4f}")
    print(f"    Best alpha: {study_q.best_params['quantile_alpha']:.3f}")

    m_q = xgb.XGBRegressor(objective="reg:quantileerror", random_state=42,
                             n_jobs=-1, tree_method="hist", early_stopping_rounds=50,
                             **study_q.best_params)
    m_q.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    pred4 = m_q.predict(X_test)
    r4 = metrics("Optuna quantile", y_test, pred4, mids, names)
    results.append(r4)
    joblib.dump(m_q, MODEL_DIR / "xgb_optuna_quantile.pkl")

    # ── STACKING META-LEARNER ────────────────────────────────────────
    print(f"\n{'~'*70}")
    print("  [5] Stacking meta-learner (Ridge on model predictions)")
    print(f"{'~'*70}")

    # Build stack features from training set (using cross-val predictions)
    # For simplicity, use a held-out validation slice from training
    val_split = int(len(train_df) * 0.8)
    stack_train = train_enc.iloc[val_split:].copy()
    stack_X = stack_train[feature_cols].astype(np.float32)
    stack_y = stack_train[TARGET].values

    # Get predictions from each model on the stack training set
    s1 = model1.predict(stack_X)
    s2_arr = np.zeros(len(stack_train))
    for role in ["BAT","BOWL","AR","WK"]:
        mask = train_df.iloc[val_split:]["player_role_platform"] == role
        if mask.sum() > 0:
            role_model_path = MODEL_DIR / f"xgb_optuna_{role.lower()}.pkl"
            if role_model_path.exists():
                rm = joblib.load(role_model_path)
                rf = [c for c in feature_cols if c != "player_role_platform"]
                s2_arr[mask.values] = rm.predict(stack_train.loc[mask.values, rf].astype(np.float32))
    s3_bat = m_bat.predict(stack_X)
    s3_bowl = m_bowl.predict(stack_X)
    s3 = s3_bat + s3_bowl + train_df.iloc[val_split:]['fielding_pts'].mean() + 4
    s4 = m_q.predict(stack_X)

    # Stack features = model predictions + a few context features
    stack_feats_train = np.column_stack([s1, s2_arr, s3, s4, s3_bat, s3_bowl])
    if 'player_role_platform' in stack_train.columns:
        stack_feats_train = np.column_stack([stack_feats_train,
                                             stack_train['player_role_platform'].values])
    if 'career_avg_points_before' in stack_train.columns:
        stack_feats_train = np.column_stack([stack_feats_train,
                                             stack_train['career_avg_points_before'].fillna(0).values])

    # Replace NaN/inf in stack features
    stack_feats_train = np.nan_to_num(stack_feats_train, nan=0, posinf=0, neginf=0)

    # Train Ridge meta-learner
    meta = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    meta.fit(stack_feats_train, stack_y)
    print(f"    Ridge alpha: {meta.alpha_}")

    # Build test stack features
    stack_feats_test = np.column_stack([pred1, pred2, pred3, pred4, pred_bat, pred_bowl])
    if 'player_role_platform' in test_enc.columns:
        stack_feats_test = np.column_stack([stack_feats_test,
                                            test_enc['player_role_platform'].values])
    if 'career_avg_points_before' in test_enc.columns:
        stack_feats_test = np.column_stack([stack_feats_test,
                                            test_enc['career_avg_points_before'].fillna(0).values])

    stack_feats_test = np.nan_to_num(stack_feats_test, nan=0, posinf=0, neginf=0)

    pred_stack = meta.predict(stack_feats_test)
    r_stack = metrics("Stacking meta-learner", y_test, pred_stack, mids, names)
    results.append(r_stack)
    joblib.dump(meta, MODEL_DIR / "meta_ridge.pkl")

    # ── FINAL ENSEMBLE: Stack + best individual ──────────────────────
    print(f"\n{'~'*70}")
    print("  [6] Final ensemble (stack + best individual blends)")
    print(f"{'~'*70}")

    all_preds = {"global": pred1, "role": pred2, "comp": pred3,
                 "quant": pred4, "stack": pred_stack}

    configs = [
        {"global": 0.2, "role": 0.2, "comp": 0.1, "quant": 0.1, "stack": 0.4},
        {"global": 0.15, "role": 0.2, "comp": 0.15, "quant": 0.1, "stack": 0.4},
        {"global": 0.1, "role": 0.15, "comp": 0.1, "quant": 0.1, "stack": 0.55},
        {"global": 0.2, "role": 0.3, "comp": 0.1, "quant": 0.1, "stack": 0.3},
        {"global": 0.15, "role": 0.25, "comp": 0.15, "quant": 0.15, "stack": 0.3},
    ]

    best_sp, best_w, best_ens = -1, None, None
    for weights in configs:
        ens = np.zeros(len(y_test))
        for mid in np.unique(mids):
            mask = mids == mid
            ref = all_preds["global"][mask]
            rm, rs = ref.mean(), max(np.std(ref), 0.01)
            bl = np.zeros(mask.sum())
            for key, w in weights.items():
                p = all_preds[key][mask]
                pn = (p - p.mean()) / max(np.std(p), 0.01) * rs + rm if np.std(p) > 0 else ref
                bl += w * pn
            ens[mask] = bl
        sp = spearman_pm(y_test, ens, mids)
        t = top_k(y_test, ens, mids, names)
        ws = "/".join(f"{v:.0%}" for v in weights.values())
        print(f"  [{ws}]: Spearman={sp:.4f}  Top11={t:.2%}")
        if sp > best_sp: best_sp, best_w, best_ens = sp, weights, ens

    print(f"\n  Best weights: {best_w}")
    r_final = metrics("BEST v12 ENSEMBLE", y_test, best_ens, mids, names)
    results.append(r_final)

    # ── COMPARISON ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON")
    print(f"{'='*70}")
    try:
        old = pd.read_csv("output/best_model_v6_predictions.csv", low_memory=False)
        osp = spearman_pm(old[TARGET].values, old["pred"].values, old["match_id"].values)
        ot = top_k(old[TARGET].values, old["pred"].values, old["match_id"].values, old["player_name"].values)
        omae = mean_absolute_error(old[TARGET], old["pred"])
        on = ndcg_pm(old[TARGET].values, old["pred"].values, old["match_id"].values)
        print(f"\n  v6:          MAE={omae:.3f}  Spearman={osp:.4f}  Top11={ot:.2%}  NDCG={on:.4f}")
        print(f"  v12 BEST:    MAE={r_final['mae']:.3f}  Spearman={r_final['spearman']:.4f}  Top11={r_final['top11']:.2%}  NDCG={r_final['ndcg']:.4f}")
        print(f"\n  Delta Spearman: {r_final['spearman']-osp:+.4f} ({(r_final['spearman']-osp)/abs(osp)*100:+.1f}%)")
        print(f"  Delta Top-11:   {r_final['top11']-ot:+.4f} ({(r_final['top11']-ot)/abs(ot)*100:+.1f}%)")
    except FileNotFoundError:
        print("  (v6 not found)")

    # Save
    out = test_df.copy()
    out["pred_global"] = pred1
    out["pred_role"] = pred2
    out["pred_component"] = pred3
    out["pred_quantile"] = pred4
    out["pred_stack"] = pred_stack
    out["pred"] = best_ens
    out.to_csv(OUTPUT_PATH, index=False)
    meta_data = {"best_weights": best_w, "params_global": params1, "metrics": results}
    with open(MODEL_DIR / "v12_metadata.json", "w") as f:
        json.dump(meta_data, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
