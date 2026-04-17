# Fantasy Cricket Optimizer

## Project context
ML system that predicts IPL fantasy points and optimizes Dream11 teams.
Built by Ajay & Varshith. Data from Cricsheet (278K ball-by-ball deliveries
across 1,169 IPL matches, 2008-2025).

## Current state
- Production model: v11 (component prediction, separate batting + bowling models)
- Best metrics: Spearman 0.5156, Top-11 overlap 67.79% (~7.5/11 players correct)
- Architecture: 4-model ensemble (XGBoost regression + role-specific + quantile + component)
- Trained on 80/20 chronological split, evaluated per-match

## Repository structure
- `data/` — raw datasets (Cricsheet ball-by-ball, match metadata, players, teams)
- `output/` — feature tables (v1-v10) and model predictions
- `models/` — trained pickle files organized by version
- `src/features/` — feature engineering scripts (build_feature_table_v6.py through v10)
- `src/models/` — training scripts (train_v7.py through train_v11.py)
- `src/inference/` — production inference engine for upcoming matches
- `app/` — Streamlit dashboard

## Key conventions
- Target column is always `fantasy_points_v5`
- Features are STRICTLY historical (use cumsum + shift, never current-match data)
- Categorical columns: season, venue, opponent, player_role_platform
- Roles: BAT, BOWL, AR (all-rounder), WK (wicketkeeper)
- Train/test split: 80/20 chronological (always sort by match_date first)
- Use XGBoost with tree_method="hist" for speed
- Per-match Spearman is the primary metric (not MAE)

## Code style
- Python 3.11, pandas 3.x, xgboost 2.x
- Use pathlib.Path for all file paths
- Always include early_stopping_rounds when fitting
- Import warnings and suppress them at top of training scripts
- Print metrics in this format: MAE=X.XXX  Spearman=X.XXXX  Top11=XX.XX%

## What NOT to do
- Never fill NaN with 0 or median in v11+ (XGBoost handles NaN natively)
- Never use future data in feature computation (audit with audit_v8.py pattern)
- Never include `fantasy_points_v5` or component point columns (batting_pts, etc.) as features — that's leakage
- Never modify training script splits — always 80/20 chronological

## Active work
Building production inference pipeline for upcoming matches.
See src/inference/inference_engine.py for the FantasyPredictor class.
