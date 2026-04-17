"""
Fantasy Cricket Inference Engine
================================
Predict fantasy points for an UPCOMING match (one that hasn't happened yet).

Core insight: feature table rows already contain "state-before-this-match"
cumulative features. For an upcoming match, we grab each player's most recent
row, override match-context fields (opponent, venue, date), and predict.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

warnings.filterwarnings("ignore")

TARGET = "fantasy_points_v5"
CAT_COLS = ["season", "venue", "opponent", "player_role_platform"]

HOME_VENUE_MAP = {
    "Mumbai Indians": ["wankhede", "mumbai"],
    "Chennai Super Kings": ["chepauk", "chennai", "ma chidambaram"],
    "Royal Challengers Bengaluru": ["chinnaswamy", "bengaluru", "bangalore"],
    "Royal Challengers Bangalore": ["chinnaswamy", "bengaluru", "bangalore"],
    "Kolkata Knight Riders": ["eden gardens", "kolkata"],
    "Delhi Capitals": ["arun jaitley", "feroz shah", "delhi"],
    "Rajasthan Royals": ["sawai mansingh", "jaipur"],
    "Sunrisers Hyderabad": ["rajiv gandhi", "hyderabad", "uppal"],
    "Punjab Kings": ["punjab cricket", "mohali", "chandigarh"],
    "Gujarat Titans": ["narendra modi", "ahmedabad"],
    "Lucknow Super Giants": ["atal bihari", "lucknow", "ekana"],
}


class FantasyPredictor:
    """Production inference engine for fantasy cricket predictions."""

    def __init__(self,
                 feature_table_path="output/model_feature_table_v10.csv",
                 model_dir="models/v10"):
        ft_path = Path(feature_table_path)
        if not ft_path.exists():
            raise FileNotFoundError(f"Feature table not found: {ft_path}")

        print(f"Loading feature table: {ft_path}")
        self.ft = pd.read_csv(ft_path, low_memory=False)
        self.ft['match_date'] = pd.to_datetime(self.ft['match_date'])
        self.ft = self.ft.sort_values(['player_name', 'match_date']).reset_index(drop=True)

        self.all_players = sorted(self.ft['player_name'].dropna().unique().tolist())
        self.all_teams = sorted(self.ft['team'].dropna().unique().tolist())

        # Models are optional — engine still useful for fuzzy lookup without them
        self.models_loaded = False
        model_path = Path(model_dir)
        if model_path.exists():
            try:
                self._load_models(model_path)
                self.models_loaded = True
            except Exception as e:
                print(f"  Warning: could not load models from {model_dir}: {e}")
                print(f"  Predictions will not be available.")

        print(f"  Players in database: {len(self.all_players)}")
        print(f"  Teams in database: {len(self.all_teams)}")
        print(f"  Models loaded: {self.models_loaded}")

    def _load_models(self, model_dir):
        self.model_global = joblib.load(model_dir / "xgb_regression.pkl")
        self.role_models = {}
        for role in ['BAT', 'BOWL', 'AR', 'WK']:
            p = model_dir / f"xgb_role_{role.lower()}.pkl"
            if p.exists():
                self.role_models[role] = joblib.load(p)
        self.model_quantile = joblib.load(model_dir / "xgb_quantile60.pkl")

        meta_path = model_dir / "v10_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.feature_cols = sorted(set(meta['feature_cols']))
            self.weights = meta.get('best_weights', {'reg': 0.4, 'role': 0.4, 'quant': 0.2})
        else:
            self.feature_cols = [c for c in self.ft.columns
                                 if c not in {"match_id","match_date","player_name","team",TARGET}]
            self.weights = {'reg': 0.4, 'role': 0.4, 'quant': 0.2}

    # ─── Fuzzy lookup ────────────────────────────────────────────────
    def list_available_players(self, query: str, limit: int = 10) -> list[str]:
        """Return top N fuzzy matches for autocomplete."""
        if not query or len(query) < 2:
            return self.all_players[:limit]
        results = process.extract(query, self.all_players, scorer=fuzz.WRatio, limit=limit)
        return [r[0] for r in results if r[1] >= 50]

    def resolve_player_name(self, name: str) -> str | None:
        """Resolve a possibly-imprecise name to the canonical database name."""
        if name in self.ft['player_name'].values:
            return name
        match = process.extractOne(name, self.all_players, scorer=fuzz.WRatio, score_cutoff=70)
        return match[0] if match else None

    def get_team_recent_xi(self, team_name: str) -> list[str]:
        """Return the most recent 11 players that played for this team."""
        team_data = self.ft[self.ft['team'] == team_name]
        if len(team_data) == 0:
            return []
        latest_match = team_data.sort_values('match_date').iloc[-1]['match_id']
        latest_xi = team_data[team_data['match_id'] == latest_match]['player_name'].tolist()
        return latest_xi[:11]

    # ─── Validation ──────────────────────────────────────────────────
    def _validate_match_input(self, match_info):
        required = ['match_date', 'venue', 'team1', 'team2', 'team1_xi', 'team2_xi']
        missing = [k for k in required if k not in match_info]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        if len(match_info['team1_xi']) != 11:
            raise ValueError(f"team1_xi must have 11 players, got {len(match_info['team1_xi'])}")
        if len(match_info['team2_xi']) != 11:
            raise ValueError(f"team2_xi must have 11 players, got {len(match_info['team2_xi'])}")

        if match_info['team1'] not in self.all_teams:
            raise ValueError(f"Unknown team1: '{match_info['team1']}'. "
                             f"Valid teams: {self.all_teams[:5]}...")
        if match_info['team2'] not in self.all_teams:
            raise ValueError(f"Unknown team2: '{match_info['team2']}'. "
                             f"Valid teams: {self.all_teams[:5]}...")

    # ─── Prediction ──────────────────────────────────────────────────
    def predict_match(self, match_info):
        if not self.models_loaded:
            raise RuntimeError("Models not loaded — predictions unavailable")
        self._validate_match_input(match_info)

        print(f"\n{'='*60}")
        print(f"  {match_info['team1']} vs {match_info['team2']}")
        print(f"  {match_info['match_date']} @ {match_info['venue']}")
        print(f"{'='*60}")

        rows = []
        missing = []

        for player in match_info['team1_xi']:
            row = self._build_player_row(player, match_info['team1'], match_info['team2'], match_info)
            (rows if row is not None else missing).append(row if row is not None else player)

        for player in match_info['team2_xi']:
            row = self._build_player_row(player, match_info['team2'], match_info['team1'], match_info)
            (rows if row is not None else missing).append(row if row is not None else player)

        if missing:
            print(f"\n  Players not found ({len(missing)}):")
            for p in missing:
                suggestion = self.list_available_players(p, limit=3)
                print(f"    - {p}  (did you mean: {', '.join(suggestion)})")

        if not rows:
            raise ValueError("No valid players found in database")

        df = pd.DataFrame(rows).reset_index(drop=True)
        df['predicted_points'] = self._ensemble_predict(df)
        df = df.sort_values('predicted_points', ascending=False).reset_index(drop=True)
        return df

    def _build_player_row(self, player_name, player_team, opponent, match_info):
        canonical = self.resolve_player_name(player_name)
        if canonical is None:
            return None

        history = self.ft[self.ft['player_name'] == canonical]
        if len(history) == 0:
            return None

        row = history.iloc[-1].copy()
        row['team'] = player_team
        row['opponent'] = opponent
        row['venue'] = match_info['venue']
        row['match_date'] = pd.to_datetime(match_info['match_date'])
        row['season'] = pd.to_datetime(match_info['match_date']).year
        row['match_id'] = 9999999
        row['player_name'] = canonical

        self._recompute_home_ground(row, player_team, match_info['venue'])
        if 'toss_winner' in match_info:
            self._recompute_toss(row, player_team, match_info)

        return row

    def _recompute_home_ground(self, row, team, venue):
        if 'is_home_ground' not in row.index:
            return
        keywords = HOME_VENUE_MAP.get(team, [])
        venue_lower = str(venue).lower()
        row['is_home_ground'] = int(any(kw in venue_lower for kw in keywords))

    def _recompute_toss(self, row, team, match_info):
        toss_winner = match_info.get('toss_winner')
        toss_decision = match_info.get('toss_decision', 'field')

        if 'toss_won' in row.index:
            row['toss_won'] = 1 if toss_winner == team else 0
        if 'toss_chose_field' in row.index:
            row['toss_chose_field'] = 1 if toss_decision == 'field' else 0
        if 'batting_first' in row.index:
            if toss_winner == team:
                row['batting_first'] = 1 if toss_decision == 'bat' else 0
            else:
                row['batting_first'] = 1 if toss_decision == 'field' else 0

    def _ensemble_predict(self, df):
        df_enc = df.copy()
        for col in CAT_COLS:
            if col in df_enc.columns:
                df_enc[col] = df_enc[col].astype("category").cat.codes

        for col in self.feature_cols:
            if col not in df_enc.columns:
                df_enc[col] = 0

        X = df_enc[self.feature_cols].fillna(0).astype(np.float32)

        pred_global = self.model_global.predict(X)
        pred_quantile = self.model_quantile.predict(X)

        pred_role = np.zeros(len(df))
        role_feats = [c for c in self.feature_cols if c != "player_role_platform"]
        for role, model in self.role_models.items():
            mask = df['player_role_platform'].values == role
            if mask.any():
                X_role = df_enc.loc[mask, role_feats].fillna(0).astype(np.float32)
                pred_role[mask] = model.predict(X_role)

        ref_mean = pred_global.mean()
        ref_std = max(pred_global.std(), 0.01)

        def normalize(p):
            if np.std(p) > 0:
                return (p - p.mean()) / np.std(p) * ref_std + ref_mean
            return p

        return (
            self.weights.get('reg', 0.4) * normalize(pred_global) +
            self.weights.get('role', 0.4) * normalize(pred_role) +
            self.weights.get('quant', 0.2) * normalize(pred_quantile)
        )


def optimize_dream11_team(predictions_df,
                          max_per_team=7,
                          role_min={"WK": 1, "BAT": 3, "BOWL": 3, "AR": 1},
                          role_max={"WK": 4, "BAT": 6, "BOWL": 6, "AR": 4}):
    """Greedy Dream11 optimizer."""
    df = predictions_df.sort_values('predicted_points', ascending=False).reset_index(drop=True)

    selected = []
    role_counts = {"WK": 0, "BAT": 0, "BOWL": 0, "AR": 0}
    team_counts = {}

    for role, min_ct in role_min.items():
        for _, p in df[df['player_role_platform'] == role].iterrows():
            if role_counts[role] >= min_ct:
                break
            t = p['team']
            if team_counts.get(t, 0) < max_per_team:
                selected.append(p)
                role_counts[role] += 1
                team_counts[t] = team_counts.get(t, 0) + 1

    selected_names = {p['player_name'] for p in selected}
    for _, p in df.iterrows():
        if len(selected) >= 11:
            break
        if p['player_name'] in selected_names:
            continue
        role = p['player_role_platform']
        t = p['team']
        if role_counts.get(role, 0) < role_max.get(role, 4) and team_counts.get(t, 0) < max_per_team:
            selected.append(p)
            role_counts[role] = role_counts.get(role, 0) + 1
            team_counts[t] = team_counts.get(t, 0) + 1
            selected_names.add(p['player_name'])

    team = pd.DataFrame(selected).reset_index(drop=True)
    team['is_captain'] = False
    team['is_vice_captain'] = False
    team.loc[0, 'is_captain'] = True
    team.loc[1, 'is_vice_captain'] = True
    team['expected_points'] = team['predicted_points'].copy()
    team.loc[team['is_captain'], 'expected_points'] *= 2.0
    team.loc[team['is_vice_captain'], 'expected_points'] *= 1.5
    return team
