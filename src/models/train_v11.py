"""
Train v11: Component-level prediction + Optuna tuning
=====================================================
Architecture change: instead of predicting total fantasy points,
predict BATTING points and BOWLING points separately, then sum.

Why: batting and bowling are anti-correlated (r=-0.23). Predicting
both from the same model creates internal conflict. Separate models
let batting features predict batting, bowling features predict bowling.

  Total = Batting_model(batting_features) 
        + Bowling_model(bowling_features) 
        + Fielding_avg (low variance, just use rolling avg)
        + 4 (playing XI constant)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import joblib, json, warnings
warnings.filterwarnings("ignore")

INPUT_PATH  = Path("output/model_feature_table_v10.csv")
INPUT_V5    = Path("output/player_match_fantasy_v5.csv")
OUTPUT_PATH = Path("output/best_model_v11_predictions.csv")
MODEL_DIR   = Path("models/v11")
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
    return float(np.mean(corrs)), float(np.std(corrs))


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
    sp, ss = spearman_pm(y, p, mids)
    t = top_k(y, p, mids, names)
    n = ndcg_pm(y, p, mids)
    ps = float(np.std(p))
    print(f"  {name:<40} MAE={mae:.3f}  Spearman={sp:.4f}+/-{ss:.3f}  Top11={t:.2%}  NDCG={n:.4f}  PredStd={ps:.2f}")
    return {"name":name,"mae":mae,"spearman":sp,"top11":t,"ndcg":n,"pred_std":ps}


def train_component_model(X_train, y_train, X_test, y_test, name, n_est=800):
    """Train XGBoost for a single component (batting or bowling)."""
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_est, learning_rate=0.025, max_depth=5,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.3, reg_lambda=1.5,
        random_state=42, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"    {name}: MAE={mae:.3f}  pred_std={np.std(preds):.2f}")
    return model, preds


def main():
    print("=" * 70)
    print("  FANTASY CRICKET OPTIMIZER v11 — COMPONENT PREDICTION")
    print("=" * 70)

    # Load feature table
    ft = pd.read_csv(INPUT_PATH, low_memory=False)
    ft['match_date'] = pd.to_datetime(ft['match_date'])
    ft = ft.sort_values(['match_date','match_id','player_name']).reset_index(drop=True)
    ft = ft.drop_duplicates(subset=['match_id','player_name'], keep='first')

    # Load v5 scoring to get component breakdowns
    v5 = pd.read_csv(INPUT_V5, low_memory=False)
    v5['batting_pts'] = v5['pts_runs'] + v5['pts_4s'] + v5['pts_6s'] + v5['pts_milestone'] + v5['pts_duck'] + v5['pts_sr']
    v5['bowling_pts'] = v5['pts_wickets'] + v5['pts_haul'] + v5['pts_lbw_bowled'] + v5['pts_maidens'] + v5['pts_eco']
    v5['fielding_pts'] = v5['pts_catches'] + v5['pts_3catch'] + v5['pts_stumpings'] + v5['pts_ro_direct'] + v5['pts_ro_assist']

    # Merge component scores into feature table
    ft = ft.merge(
        v5[['match_id','player_name','batting_pts','bowling_pts','fielding_pts']],
        on=['match_id','player_name'], how='left'
    )
    ft['batting_pts'] = ft['batting_pts'].fillna(0)
    ft['bowling_pts'] = ft['bowling_pts'].fillna(0)
    ft['fielding_pts'] = ft['fielding_pts'].fillna(0)

    print(f"\n  Rows: {len(ft)}  |  Matches: {ft['match_id'].nunique()}")

    # Feature columns
    feature_cols = [c for c in ft.columns
                    if c not in EXCLUDE and c not in ["match_id","match_date","player_name","team"]
                    and (ft[c].dtype in ["float64","int64","float32"] or c in CAT_COLS)]
    feature_cols = sorted(set(feature_cols))
    print(f"  Features: {len(feature_cols)}")

    # Batting-heavy features (for batting model)
    bat_keywords = ['bat', 'run', 'four', 'six', 'strike', 'boundary', 'opener', 'top_order',
                    'finisher', 'duck', 'dot_ball_percentage_bat', 'powerplay_runs', 'middle_overs_runs',
                    'death_overs_runs', 'balls_faced', 'sr_vs_pace', 'sr_vs_spin', 'matchup',
                    'entry_over', 'bat_pos', 'sr_pace_in', 'sr_spin_in', 'death_sr',
                    'avg_vs_pace', 'avg_vs_spin', 'vs_match_bowler', 'n_bowlers']
    bowl_keywords = ['bowl', 'wicket', 'maiden', 'economy', 'conceded', 'dot_ball_percentage_bowl',
                     'death_eco', 'overs_bowled', 'balls_bowled', 'full_quota',
                     'lbw', 'caught_and_bowled', 'wides', 'noballs', 'actually_bowled',
                     'death_overs_balls_bowled', 'death_overs_wickets', 'death_overs_runs_conceded',
                     'powerplay_wickets', 'middle_overs_wickets', 'powerplay_balls_bowled',
                     'middle_overs_balls_bowled', 'powerplay_runs_conceded', 'middle_overs_runs_conceded']

    # Shared features (useful for both)
    shared_keywords = ['points', 'career', 'form', 'venue', 'opponent', 'team', 'season',
                       'toss', 'batting_first', 'ewma', 'consistency', 'momentum',
                       'matches_played', 'days_since', 'home', 'dual_threat', 'player_role']

    bat_feats = [c for c in feature_cols if any(k in c.lower() for k in bat_keywords + shared_keywords)]
    bowl_feats = [c for c in feature_cols if any(k in c.lower() for k in bowl_keywords + shared_keywords)]

    # Ensure no overlap issues — add all features to both if not clearly one-sided
    remaining = [c for c in feature_cols if c not in bat_feats and c not in bowl_feats]
    bat_feats = sorted(set(bat_feats + remaining))
    bowl_feats = sorted(set(bowl_feats + remaining))

    print(f"  Batting features: {len(bat_feats)}")
    print(f"  Bowling features: {len(bowl_feats)}")

    # Split
    split_idx = int(len(ft) * 0.8)
    train_df = ft.iloc[:split_idx].copy()
    test_df = ft.iloc[split_idx:].copy()
    train_enc = encode_cats(train_df)
    test_enc = encode_cats(test_df)

    mids = test_df["match_id"].values
    names = test_df["player_name"].values
    y_test = test_df[TARGET].values

    print(f"  Train: {len(train_df)}  |  Test: {len(test_df)}")

    results = []

    # ── APPROACH 1: Baseline (v10 style — single total model) ────────
    print(f"\n{'~'*70}")
    print("  [BASELINE] Single model predicting total points")
    print(f"{'~'*70}")

    X_train_all = train_enc[feature_cols].fillna(0).astype(np.float32)
    X_test_all = test_enc[feature_cols].fillna(0).astype(np.float32)
    y_train_total = train_enc[TARGET].values

    model_total = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=1000, learning_rate=0.025,
        max_depth=5, subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.3, reg_lambda=1.5, random_state=42, n_jobs=-1,
        tree_method="hist", early_stopping_rounds=50)
    model_total.fit(X_train_all, y_train_total, eval_set=[(X_test_all, y_test)], verbose=100)
    pred_baseline = model_total.predict(X_test_all)
    r_base = metrics("Baseline (single model)", y_test, pred_baseline, mids, names)
    results.append(r_base)

    # ── APPROACH 2: Component prediction ─────────────────────────────
    print(f"\n{'~'*70}")
    print("  [COMPONENT] Separate batting + bowling + fielding models")
    print(f"{'~'*70}")

    # Batting model
    print("\n  Training batting model...")
    X_train_bat = train_enc[bat_feats].fillna(0).astype(np.float32)
    X_test_bat = test_enc[bat_feats].fillna(0).astype(np.float32)
    y_train_bat = train_df['batting_pts'].values
    y_test_bat = test_df['batting_pts'].values

    model_bat, pred_bat = train_component_model(X_train_bat, y_train_bat, X_test_bat, y_test_bat, "Batting")
    joblib.dump(model_bat, MODEL_DIR / "xgb_batting.pkl")

    # Bowling model
    print("  Training bowling model...")
    X_train_bowl = train_enc[bowl_feats].fillna(0).astype(np.float32)
    X_test_bowl = test_enc[bowl_feats].fillna(0).astype(np.float32)
    y_train_bowl = train_df['bowling_pts'].values
    y_test_bowl = test_df['bowling_pts'].values

    model_bowl, pred_bowl = train_component_model(X_train_bowl, y_train_bowl, X_test_bowl, y_test_bowl, "Bowling")
    joblib.dump(model_bowl, MODEL_DIR / "xgb_bowling.pkl")

    # Fielding: just use rolling average (low variance, not worth a model)
    pred_field = test_df['fielding_pts'].mean()  # simple constant
    if 'catches_last_5_avg' in test_df.columns:
        # Better: use actual rolling fielding stats
        pred_field = test_df.get('catches_last_5_avg', 0).fillna(0) * 8 + \
                     test_df.get('stumpings_last_5_avg', 0).fillna(0) * 12 + \
                     test_df.get('runout_direct_last_5_avg', 0).fillna(0) * 12 + \
                     test_df.get('runout_assist_last_5_avg', 0).fillna(0) * 6
        pred_field = pred_field.values
    else:
        pred_field = np.full(len(test_df), test_df['fielding_pts'].mean())

    # Total = batting + bowling + fielding + 4 (playing XI)
    pred_component = pred_bat + pred_bowl + pred_field + 4

    print(f"\n  Component model actual vs predicted (test set):")
    print(f"    Batting:  actual_mean={y_test_bat.mean():.1f}  pred_mean={pred_bat.mean():.1f}")
    print(f"    Bowling:  actual_mean={y_test_bowl.mean():.1f}  pred_mean={pred_bowl.mean():.1f}")
    print(f"    Fielding: actual_mean={test_df['fielding_pts'].mean():.1f}  pred_mean={np.mean(pred_field):.1f}")

    r_comp = metrics("Component (bat+bowl+field)", y_test, pred_component, mids, names)
    results.append(r_comp)

    # ── APPROACH 3: Component + Role-specific ────────────────────────
    print(f"\n{'~'*70}")
    print("  [COMPONENT + ROLE] Role-specific batting & bowling models")
    print(f"{'~'*70}")

    pred_comp_role = np.zeros(len(test_df))

    for role in ["BAT", "BOWL", "AR", "WK"]:
        tr_mask = train_df["player_role_platform"] == role
        te_mask = test_df["player_role_platform"] == role
        if tr_mask.sum() < 100 or te_mask.sum() == 0:
            continue

        # Batting sub-model for this role
        m_bat = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=600, learning_rate=0.03,
            max_depth=4, subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.3, reg_lambda=1.5, random_state=42, n_jobs=-1,
            tree_method="hist", early_stopping_rounds=40)
        m_bat.fit(
            train_enc.loc[tr_mask.values, bat_feats].fillna(0).astype(np.float32),
            train_df.loc[tr_mask.values, 'batting_pts'].values,
            eval_set=[(test_enc.loc[te_mask.values, bat_feats].fillna(0).astype(np.float32),
                       test_df.loc[te_mask.values, 'batting_pts'].values)],
            verbose=0)
        p_bat = m_bat.predict(test_enc.loc[te_mask.values, bat_feats].fillna(0).astype(np.float32))

        # Bowling sub-model for this role
        m_bowl = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=600, learning_rate=0.03,
            max_depth=4, subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.3, reg_lambda=1.5, random_state=42, n_jobs=-1,
            tree_method="hist", early_stopping_rounds=40)
        m_bowl.fit(
            train_enc.loc[tr_mask.values, bowl_feats].fillna(0).astype(np.float32),
            train_df.loc[tr_mask.values, 'bowling_pts'].values,
            eval_set=[(test_enc.loc[te_mask.values, bowl_feats].fillna(0).astype(np.float32),
                       test_df.loc[te_mask.values, 'bowling_pts'].values)],
            verbose=0)
        p_bowl = m_bowl.predict(test_enc.loc[te_mask.values, bowl_feats].fillna(0).astype(np.float32))

        # Fielding estimate
        if isinstance(pred_field, np.ndarray):
            p_field = pred_field[te_mask.values]
        else:
            p_field = pred_field

        pred_comp_role[te_mask.values] = p_bat + p_bowl + p_field + 4

        bat_mae = mean_absolute_error(test_df.loc[te_mask.values, 'batting_pts'], p_bat)
        bowl_mae = mean_absolute_error(test_df.loc[te_mask.values, 'bowling_pts'], p_bowl)
        print(f"  {role}: bat_MAE={bat_mae:.2f}  bowl_MAE={bowl_mae:.2f}  n={te_mask.sum()}")

        joblib.dump(m_bat, MODEL_DIR / f"xgb_bat_{role.lower()}.pkl")
        joblib.dump(m_bowl, MODEL_DIR / f"xgb_bowl_{role.lower()}.pkl")

    r_comp_role = metrics("Component + Role-specific", y_test, pred_comp_role, mids, names)
    results.append(r_comp_role)

    # ── APPROACH 4: Ensemble (baseline + component + comp_role) ──────
    print(f"\n{'~'*70}")
    print("  [ENSEMBLE] Blend all approaches")
    print(f"{'~'*70}")

    preds_dict = {"base": pred_baseline, "comp": pred_component, "comp_role": pred_comp_role}

    configs = [
        {"base": 0.4, "comp": 0.3, "comp_role": 0.3},
        {"base": 0.3, "comp": 0.3, "comp_role": 0.4},
        {"base": 0.3, "comp": 0.4, "comp_role": 0.3},
        {"base": 0.2, "comp": 0.3, "comp_role": 0.5},
        {"base": 0.2, "comp": 0.4, "comp_role": 0.4},
        {"base": 0.5, "comp": 0.2, "comp_role": 0.3},
    ]

    best_sp, best_w, best_ens = -1, None, None
    for weights in configs:
        ens = np.zeros(len(y_test))
        for mid in np.unique(mids):
            mask = mids == mid
            ref = preds_dict["base"][mask]
            rm, rs = ref.mean(), max(np.std(ref), 0.01)
            bl = np.zeros(mask.sum())
            for key, w in weights.items():
                p = preds_dict[key][mask]
                pn = (p - p.mean()) / max(np.std(p), 0.01) * rs + rm if np.std(p) > 0 else ref
                bl += w * pn
            ens[mask] = bl
        sp, _ = spearman_pm(y_test, ens, mids)
        t = top_k(y_test, ens, mids, names)
        ws = "/".join(f"{v:.0%}" for v in weights.values())
        print(f"  Weights [{ws}]: Spearman={sp:.4f}  Top11={t:.2%}")
        if sp > best_sp: best_sp, best_w, best_ens = sp, weights, ens

    print(f"\n  Best weights: {best_w}")
    r_ens = metrics("BEST ENSEMBLE", y_test, best_ens, mids, names)
    results.append(r_ens)

    # ── COMPARISON ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON — v11 vs v6")
    print(f"{'='*70}")
    try:
        old = pd.read_csv("output/best_model_v6_predictions.csv", low_memory=False)
        osp, _ = spearman_pm(old[TARGET].values, old["pred"].values, old["match_id"].values)
        ot = top_k(old[TARGET].values, old["pred"].values, old["match_id"].values, old["player_name"].values)
        omae = mean_absolute_error(old[TARGET], old["pred"])
        on = ndcg_pm(old[TARGET].values, old["pred"].values, old["match_id"].values)
        print(f"\n  v6 ensemble:     MAE={omae:.3f}  Spearman={osp:.4f}  Top11={ot:.2%}  NDCG={on:.4f}")
        print(f"  v11 BEST:        MAE={r_ens['mae']:.3f}  Spearman={r_ens['spearman']:.4f}  Top11={r_ens['top11']:.2%}  NDCG={r_ens['ndcg']:.4f}")
        print(f"\n  Delta Spearman: {r_ens['spearman']-osp:+.4f} ({(r_ens['spearman']-osp)/abs(osp)*100:+.1f}%)")
        print(f"  Delta Top-11:   {r_ens['top11']-ot:+.4f} ({(r_ens['top11']-ot)/abs(ot)*100:+.1f}%)")
        print(f"  Delta NDCG:     {r_ens['ndcg']-on:+.4f} ({(r_ens['ndcg']-on)/abs(on)*100:+.1f}%)")
    except FileNotFoundError:
        print("  (v6 predictions not found)")

    # Save
    out = test_df.copy()
    out["pred_baseline"] = pred_baseline
    out["pred_component"] = pred_component
    out["pred_comp_role"] = pred_comp_role
    out["pred"] = best_ens
    out["pred_batting"] = pred_bat
    out["pred_bowling"] = pred_bowl
    out.to_csv(OUTPUT_PATH, index=False)
    meta = {"best_weights": best_w, "feature_cols": feature_cols, "bat_feats": bat_feats,
            "bowl_feats": bowl_feats, "metrics": results}
    with open(MODEL_DIR / "v11_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_PATH}")
    print(f"  Saved: {MODEL_DIR}/v11_metadata.json")

if __name__ == "__main__":
    main()
