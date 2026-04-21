import hashlib
import os
import random
import re
import pandas as pd
import pulp
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

try:
    from api.ipl2026_loader import (
        SHORT_TO_SCHEDULE_NAME,
        build_ipl2026_aggregates,
        expected_xi_for_franchise,
        ipl2026_form_context,
        ipl2026_onfield_appearances_before_fixture,
        ipl2026_recent_matches_for_player,
        ipl2026_season_l5_percentile_line,
        schedule_team_to_short,
    )
except ImportError:
    from ipl2026_loader import (
        SHORT_TO_SCHEDULE_NAME,
        build_ipl2026_aggregates,
        expected_xi_for_franchise,
        ipl2026_form_context,
        ipl2026_onfield_appearances_before_fixture,
        ipl2026_recent_matches_for_player,
        ipl2026_season_l5_percentile_line,
        schedule_team_to_short,
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    match_id: str
    locked_player_ids: List[int] = []
    banned_player_ids: List[int] = []
    batting_first_team: Optional[str] = None
    pitch_type: Optional[str] = 'Neutral'
    weather_condition: Optional[str] = 'Clear'
    num_lineups: int = 1

df = None
bbb_matchups = None
ipl2026_player_match = None
ipl2026_xi_by_team = None
upcoming_match_meta = {}
merged_team_rosters = None
active_playing_squads = {}
venue_avg_batting_one_side = {}
venue_avg_phase = {}
h2h_lookup = {}
player_name_lower_lookup = {}
player_names_by_surname = {}

# Hand-curated corrections for 2026 season usage.
PLAYER_ROLE_OVERRIDES_2026 = {
    "Finn Allen": "BAT",
    "Tim Seifert": "WK",
}
PLAYER_TEAM_OVERRIDES_2026 = {
    "Tim Seifert": "Kolkata Knight Riders",
    "Finn Allen": "Kolkata Knight Riders",
    "Josh Hazlewood": "Royal Challengers Bengaluru",
}
PLAYER_FORCE_ACTIVE_2026 = {
    "Josh Hazlewood": True,
    "Jacob Duffy": False,
    "Corbin Bosch": False,
}
PLAYER_FORM_DELTA_OVERRIDE_2026 = {
    "Vipraj Nigam": -6.0,
    "Tim Seifert": 4.0,
}


def _override_value_for_player(player_name: str, override_map: dict):
    """Return override value by exact key first, then fuzzy-name match."""
    pname = str(player_name)
    if pname in override_map:
        return override_map[pname]
    for key, value in override_map.items():
        if player_names_match(pname, str(key)):
            return value
    return None


def normalize_venue_name(venue: str) -> str:
    if venue is None:
        return "unknown"
    venue_text = str(venue).strip().lower()
    venue_text = re.sub(r'["\'\n\r,\.()&]', '', venue_text)
    # Canonical aliases for common feed variants/abbreviations.
    venue_text = re.sub(r'\bbrsabv\b', 'bharat ratna shri atal bihari vajpayee', venue_text)
    venue_text = re.sub(r'\bm a\b', 'ma', venue_text)
    venue_text = re.sub(r'\bcricket stadium\b', 'stadium', venue_text)
    venue_text = re.sub(r'\binternational stadium\b', 'stadium', venue_text)
    venue_text = re.sub(r'\bstad\b', 'stadium', venue_text)
    venue_text = re.sub(r'\bekana cricket stadium\b', 'ekana stadium', venue_text)
    venue_text = re.sub(r'\bchidambaram stadium chennai\b', 'chidambaram stadium chepauk chennai', venue_text)
    venue_text = re.sub(r'\s+', ' ', venue_text)
    return venue_text.strip()


def simplify_venue_name(venue: str) -> str:
    normalized = normalize_venue_name(venue)
    parts = normalized.split()
    if 'stadium' in parts:
        idx = parts.index('stadium')
        return " ".join(parts[: idx + 1])
    if 'park' in parts:
        idx = parts.index('park')
        return " ".join(parts[: idx + 1])
    if 'oval' in parts:
        idx = parts.index('oval')
        return " ".join(parts[: idx + 1])
    return normalized


def load_venue_batting_averages(project_root: str) -> dict:
    cleaned_paths = [
        os.path.join(project_root, "Cleaned_Dataset", "cleaned_master_dataset.csv"),
        os.path.join(project_root, "data", "cleaned_master_dataset.csv"),
    ]
    cleaned_path = next((p for p in cleaned_paths if os.path.exists(p)), None)
    if cleaned_path is None:
        return {}

    try:
        cleaned = pd.read_csv(cleaned_path, usecols=["match_id", "team", "runs_scored", "venue"], low_memory=False)
    except Exception:
        return {}

    cleaned["venue_norm"] = cleaned["venue"].fillna("unknown").map(normalize_venue_name)
    cleaned["venue_simple"] = cleaned["venue"].fillna("unknown").map(simplify_venue_name)
    team_totals = (
        cleaned.groupby(["match_id", "team", "venue_norm", "venue_simple"], dropna=False)["runs_scored"]
        .sum()
        .reset_index()
    )
    venue_means = team_totals.groupby("venue_norm")["runs_scored"].mean()
    venue_map = {key: float(value) for key, value in venue_means.items()}
    for key, value in venue_means.items():
        simple_key = simplify_venue_name(key)
        if simple_key not in venue_map:
            venue_map[simple_key] = float(value)
    return venue_map


def build_venue_phase_averages(df: pd.DataFrame) -> dict:
    metrics = [
        "venue_avg_total_runs_before",
        "venue_avg_powerplay_runs_before",
        "venue_avg_middle_overs_runs_before",
        "venue_avg_death_overs_runs_before",
    ]
    if not all(col in df.columns for col in metrics):
        return {}

    agg = (
        df[df["venue"].notna()]
        .assign(venue_norm=df["venue"].map(normalize_venue_name))
        .groupby("venue_norm")[metrics]
        .mean()
    )
    result = {}
    for venue_norm, row in agg.iterrows():
        entry = {metric: float(row[metric]) for metric in metrics}
        result[venue_norm] = entry
        simple_key = simplify_venue_name(venue_norm)
        if simple_key not in result:
            result[simple_key] = entry
    return result


def load_merged_team_rosters(project_root: str) -> dict:
    """Merge data/2026_active_squads.json with Team_Squad/*.json (official full squads) and CSV files."""
    import json

    merged = {}
    base = os.path.join(project_root, "data", "2026_active_squads.json")
    if os.path.exists(base):
        with open(base, "r", encoding="utf-8") as f:
            merged = json.load(f)
    ts_dir = os.path.join(project_root, "Team_Squad")
    if os.path.isdir(ts_dir):
        for fn in sorted(os.listdir(ts_dir)):
            if fn.endswith(".json"):
                path = os.path.join(ts_dir, fn)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        blob = json.load(f)
                except (OSError, ValueError, TypeError):
                    continue
                if not isinstance(blob, dict):
                    continue
                for k, v in blob.items():
                    if not isinstance(v, list):
                        continue
                    names = [str(x).strip() for x in v if str(x).strip()]
                    merged.setdefault(k, [])
                    seen = set(merged[k])
                    for nm in names:
                        if nm not in seen:
                            merged[k].append(nm)
                            seen.add(nm)
            elif fn.endswith(".csv"):
                path = os.path.join(ts_dir, fn)
                try:
                    csv_df = pd.read_csv(path)
                    if 'Player' in csv_df.columns and 'Team' in csv_df.columns:
                        for _, row in csv_df.iterrows():
                            team_name = str(row['Team']).strip()
                            player_name = str(row['Player']).strip()
                            merged.setdefault(team_name, [])
                            if player_name not in merged[team_name]:
                                merged[team_name].append(player_name)
                except Exception:
                    continue
    return merged


def load_active_playing_squads(project_root: str) -> dict:
    import json

    path = os.path.join(project_root, "data", "2026_active_playing_squads.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        return blob if isinstance(blob, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def resolve_franchise_squad_key(team_schedule_name: str, squads: dict) -> Optional[str]:
    """Map schedule / auction PDF names to keys present in squad JSON (e.g. RCB Bengaluru vs Bangalore)."""
    if not squads or not team_schedule_name:
        return None
    t = str(team_schedule_name).strip()
    if t in squads:
        return t
    tb = t.replace("Bengaluru", "Bangalore")
    for k in squads:
        if k.replace("Bengaluru", "Bangalore") == tb:
            return k
    if t == "Royal Challengers Bengaluru" and "Royal Challengers Bangalore" in squads:
        return "Royal Challengers Bangalore"
    return None


WITHDRAWAL_2026 = [
    "Pat Cummins",
    "Sam Curran",
    "Glenn Maxwell",
    "Mayank Agarwal",
    "Cameron Green",
    "Ashutosh Sharma",
]


def append_roster_only_players(
    unique_players: pd.DataFrame,
    t1: str,
    t2: str,
    t1_players: list,
    t2_players: list,
    base_df: pd.DataFrame,
) -> pd.DataFrame:
    """Players present in merged squad lists but missing from historical df (name / new buy)."""
    if not t1_players and not t2_players:
        return unique_players

    def round_credit_to_half(credit):
        """Round credit to nearest 0.5 increment: 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5"""
        rounded = round(credit * 2) / 2
        return max(7, min(10.5, rounded))

    next_id = int(base_df["id"].max()) + 1 if "id" in base_df.columns else 0
    add_rows = []
    for team, roster in ((t1, t1_players), (t2, t2_players)):
        side = unique_players[unique_players["team"] == team]
        tmpl = None
        if not side.empty:
            tmpl = side.iloc[0]
        elif not unique_players.empty:
            tmpl = unique_players.iloc[0]

        for rname in roster:
            if unique_players["player_name"].apply(lambda n: player_names_match(n, rname)).any():
                continue

            if tmpl is not None:
                nr = tmpl.copy()
            else:
                nr = pd.Series(
                    {
                        "player_name": rname,
                        "team": team,
                        "Role": "AR",
                        "proj_points": 22.0,
                        "credits": 8.0,
                    }
                )

            nr["player_name"] = rname
            nr["team"] = team
            nr["modern_team"] = team
            nr["proj_points"] = max(
                17.0, min(36.0, float(nr.get("proj_points", 22.0)) * 0.82))
            nr["credits"] = round_credit_to_half(float(nr.get("credits", 8.0)) * 0.95)
            nr["Role"] = "AR"
            nr["id"] = next_id
            next_id += 1
            nr["is_withdrawn"] = any(player_names_match(rname, w) for w in WITHDRAWAL_2026)

            for col in ("runs_scored", "wickets", "balls_bowled", "balls_faced"):
                if col in nr.index:
                    nr[col] = 0
            add_rows.append(nr)

    if not add_rows:
        return unique_players
    return pd.concat([unique_players, pd.DataFrame(add_rows)], ignore_index=True)


def _normalize_surname_token(tok: str) -> str:
    """Align spelling variants between feeds (e.g. IPL 2026 Sooryavanshi vs dataset Suryavanshi)."""
    s = str(tok).lower().strip()
    s = s.replace("sooryavanshi", "suryavanshi")
    s = s.replace("soorya", "surya")
    return s


def player_names_match(db_name, roster_name):
    db_p = str(db_name).lower().replace(".", "").split()
    r_p = str(roster_name).lower().replace(".", "").split()
    if not db_p or not r_p:
        return False
    if _normalize_surname_token(db_p[-1]) != _normalize_surname_token(r_p[-1]):
        return False
    if len(db_p) == 1 and len(r_p) == 1:
        return db_p[0] == r_p[0]
    db_first = db_p[0]
    r_first = r_p[0]
    if db_first == r_first:
        return True
    if len(db_p) == 2 and len(r_p) == 2:
        if len(db_first) <= 2 and db_first[0] == r_first[0]:
            return True
        if len(r_first) <= 2 and db_first[0] == r_first[0]:
            return True
        if len(db_first) == 1 and len(r_first) > 1 and r_first.startswith(db_first):
            return True
        if len(r_first) == 1 and len(db_first) > 1 and db_first.startswith(r_first):
            return True
    elif len(db_p) > 2 or len(r_p) > 2:
        db_in = "".join([w[0] for w in db_p[:-1]])
        r_in = "".join([w[0] for w in r_p[:-1]])
        if db_in == r_in:
            return True
        if len(db_in) <= 2 and len(r_in) > 0 and db_in[0] == r_in[0]:
            return True
    if len(db_p) >= 2 and len(r_p) >= 2:
        a, b = db_p[0], r_p[0]
        if len(a) == len(b) and len(a) >= 4 and sum(c1 != c2 for c1, c2 in zip(a, b)) <= 1:
            return True
    db_full = " " + " ".join(db_p) + " "
    r_full = " " + " ".join(r_p) + " "
    return db_full in r_full or r_full in db_full


def resolve_df_player_name(request_name: str) -> str:
    if df is None or not request_name:
        return request_name
    if not df[df["player_name"] == request_name].empty:
        return request_name
    lowered = str(request_name).strip().lower()
    if lowered in player_name_lower_lookup:
        return player_name_lower_lookup[lowered]
    surname_bucket = player_names_by_surname.get(_surname_key(request_name), [])
    candidates = surname_bucket if surname_bucket else df["player_name"].unique()
    for pn in candidates:
        if player_names_match(request_name, str(pn)):
            return str(pn)
    return request_name


def _opponent_names_equivalent(hist_name: str, schedule_name: str) -> bool:
    a = str(hist_name).replace("Bengaluru", "Bangalore").strip().lower()
    b = str(schedule_name).replace("Bengaluru", "Bangalore").strip().lower()
    return a == b


def _normalize_team_name_for_lookup(team_name: str) -> str:
    return str(team_name).replace("Bengaluru", "Bangalore").strip().lower()


def _surname_key(name: str) -> str:
    parts = str(name).lower().replace(".", "").split()
    return _normalize_surname_token(parts[-1]) if parts else ""


def _collapse_name_variants_for_match(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-team name variants like 'K Yadav' and 'Kuldeep Yadav'
    into a single row for optimizer/display.
    """
    if players_df is None or players_df.empty or "player_name" not in players_df.columns:
        return players_df

    kept_rows = []
    consumed = set()
    rows = players_df.reset_index(drop=True)

    for i in range(len(rows)):
        if i in consumed:
            continue
        base = rows.iloc[i]
        cluster = [i]
        for j in range(i + 1, len(rows)):
            if j in consumed:
                continue
            cand = rows.iloc[j]
            if str(base.get("team", "")) != str(cand.get("team", "")):
                continue
            if player_names_match(str(base["player_name"]), str(cand["player_name"])):
                cluster.append(j)
                consumed.add(j)

        if len(cluster) == 1:
            kept_rows.append(base.to_dict())
            continue

        # Prefer fuller name and stronger projection/confidence as representative.
        cluster_rows = rows.iloc[cluster].copy()
        cluster_rows["name_len"] = cluster_rows["player_name"].astype(str).str.len()
        cluster_rows["proj_points_num"] = pd.to_numeric(cluster_rows.get("proj_points", 0), errors="coerce").fillna(0.0)
        if "playing_confidence" in cluster_rows.columns:
            cluster_rows["playing_conf_num"] = pd.to_numeric(cluster_rows.get("playing_confidence", 0), errors="coerce").fillna(0.0)
        else:
            cluster_rows["playing_conf_num"] = 0.0
        cluster_rows = cluster_rows.sort_values(
            by=["name_len", "playing_conf_num", "proj_points_num"],
            ascending=[False, False, False],
        )
        rep = cluster_rows.iloc[0].to_dict()
        rep["proj_points"] = float(cluster_rows["proj_points_num"].max())
        rep["playing_confidence"] = float(cluster_rows["playing_conf_num"].max())
        rep["is_probably_playing"] = bool(cluster_rows.get("is_probably_playing", pd.Series([False])).astype(bool).any())
        kept_rows.append(rep)

    return pd.DataFrame(kept_rows)


def _build_h2h_lookup_table(hist_df: pd.DataFrame) -> dict:
    req_cols = {"player_name", "opponent_norm", "runs_scored", "balls_faced", "wickets"}
    if hist_df is None or hist_df.empty or not req_cols.issubset(hist_df.columns):
        return {}
    if "proj_points" not in hist_df.columns:
        return {}

    h2h = (
        hist_df.groupby(["player_name", "opponent_norm"], dropna=False)
        .agg(
            matches_played=("player_name", "size"),
            total_runs=("runs_scored", "sum"),
            total_balls=("balls_faced", "sum"),
            total_wickets=("wickets", "sum"),
            avg_fantasy_points=("proj_points", "mean"),
        )
        .reset_index()
    )
    return {
        (str(r["player_name"]), str(r["opponent_norm"])): {
            "matches_played": int(r["matches_played"]),
            "total_runs": int(r["total_runs"]) if pd.notna(r["total_runs"]) else 0,
            "total_balls": int(r["total_balls"]) if pd.notna(r["total_balls"]) else 0,
            "total_wickets": int(r["total_wickets"]) if pd.notna(r["total_wickets"]) else 0,
            "avg_fantasy_points": float(r["avg_fantasy_points"]) if pd.notna(r["avg_fantasy_points"]) else 0.0,
        }
        for _, r in h2h.iterrows()
    }


@app.on_event("startup")
def load_data():
    global df, bbb_matchups, ipl2026_player_match, ipl2026_xi_by_team, upcoming_match_meta, merged_team_rosters
    global h2h_lookup, player_name_lower_lookup, player_names_by_surname
    global active_playing_squads
    import glob
    import re
    
    old_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "v4_dataset.csv")
    bbb_path = os.path.join(os.path.dirname(__file__), "..", "Raw_Dataset", "all_ball_by_ball_data.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "Fantasy-Cricket-Optimizer-phase9-production-pipeline", "output")
    
    if os.path.exists(bbb_path):
        bbb_df = pd.read_csv(bbb_path, usecols=['batter', 'bowler', 'batter_runs', 'is_wicket'], low_memory=False)
        bbb_matchups = bbb_df.groupby(['batter', 'bowler']).agg(
            runs=('batter_runs', 'sum'),
            balls=('batter', 'count'),
            wickets=('is_wicket', lambda x: x.astype(bool).sum())
        ).reset_index()

    latest_hist = None
    latest_pred = None
    
    if os.path.exists(output_dir):
        hist_files = glob.glob(os.path.join(output_dir, "player_match_fantasy_v*.csv"))
        pred_files = glob.glob(os.path.join(output_dir, "best_model_v*_predictions.csv"))
        
        def get_version(f):
            m = re.search(r'v(\d+)', os.path.basename(f))
            return int(m.group(1)) if m else 0
            
        hist_files = [f for f in hist_files if 'with_opponent' not in f]
        
        if hist_files:
            latest_hist = max(hist_files, key=get_version)
        if pred_files:
            latest_pred = max(pred_files, key=get_version)

    if latest_hist and latest_pred:
        df_hist = pd.read_csv(latest_hist, low_memory=False)
        df_pred = pd.read_csv(latest_pred, low_memory=False)
        
        if 'pred' in df_hist.columns:
            df_hist = df_hist.drop(columns=['pred'])
            
        df_pred_sub = df_pred[['match_id', 'player_name', 'pred']].drop_duplicates(subset=['match_id', 'player_name'])
        df = pd.merge(df_hist, df_pred_sub, on=['match_id', 'player_name'], how='left')
        
        fp_col = next((c for c in df.columns if c.startswith('fantasy_points_v')), 'total_fantasy_points')
        if fp_col in df.columns:
            df['pred'] = df['pred'].fillna(df[fp_col])
    elif os.path.exists(old_data_path):
        df = pd.read_csv(old_data_path, low_memory=False)
    else:
        return

    # Precompute lookup indexes used by latency-sensitive endpoints.
    if "opponent" in df.columns:
        df["opponent_norm"] = df["opponent"].map(_normalize_team_name_for_lookup)

    if "player_name" in df.columns:
        player_name_lower_lookup = {
            str(pn).strip().lower(): str(pn) for pn in df["player_name"].dropna().unique()
        }
        player_names_by_surname = {}
        for pn in df["player_name"].dropna().unique():
            key = _surname_key(str(pn))
            if key:
                player_names_by_surname.setdefault(key, []).append(str(pn))

    h2h_lookup = {}

    project_root = os.path.join(os.path.dirname(__file__), "..")
    global venue_avg_batting_one_side, venue_avg_phase
    venue_avg_batting_one_side = load_venue_batting_averages(project_root)
    venue_avg_phase = build_venue_phase_averages(df)
    
    # 1. Standardizations
    role_map = {"BAT": "BAT", "WKB": "WK", "BOWL": "BOWL", "AR": "AR", "WK": "WK"}
    if 'player_role_platform' in df.columns:
        df['Role'] = df['player_role_platform'].map(role_map).fillna("AR")
        
    # 2. Ensure we have required metrics: proj_points and credits
    if 'pred' in df.columns and 'proj_points' not in df.columns:
        df['proj_points'] = df['pred']
    elif 'total_fantasy_points' in df.columns and 'proj_points' not in df.columns:
        df['proj_points'] = df['total_fantasy_points']
    elif 'proj_points' not in df.columns:
        # Deterministic fallback if not explicit
        df['proj_points'] = df.index.map(lambda x: 20.0 + (x % 50))

    h2h_lookup = _build_h2h_lookup_table(df)
        
    # 2. Credits should be based on player's overall historical stature, NOT tonight's prediction!
    if 'credits' not in df.columns:
        import math
        career_scores = (df.groupby('player_name')['runs_scored'].mean().fillna(0) + df.groupby('player_name')['wickets'].mean().fillna(0) * 25)
        counts = df.groupby('player_name').size()
        career_scores = career_scores * counts.apply(lambda x: math.log1p(x))
        
        p98 = career_scores.quantile(0.98)
        p94 = career_scores.quantile(0.94)
        p85 = career_scores.quantile(0.85)
        p70 = career_scores.quantile(0.70)
        p50 = career_scores.quantile(0.50)
        p30 = career_scores.quantile(0.30)
        p15 = career_scores.quantile(0.15)
        
        def round_credit(credit):
            """Round credit to nearest 0.5 increment between 7 and 10.5"""
            rounded = round(credit * 2) / 2  # Round to nearest 0.5
            return max(7, min(10.5, rounded))
        
        def assign_dream11_credit_historic(p_name):
            score = career_scores.get(p_name, 0)
            if score >= p98: return round_credit(10.5)
            elif score >= p94: return round_credit(10.0)
            elif score >= p85: return round_credit(9.5)
            elif score >= p70: return round_credit(9.0)
            elif score >= p50: return round_credit(8.5)
            elif score >= p30: return round_credit(8.0)
            elif score >= p15: return round_credit(7.5)
            else: return round_credit(7.0)

        df['credits'] = df['player_name'].apply(assign_dream11_credit_historic)
        
        # Round credits to nearest 0.5 increment: 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5
        df['credits'] = df['credits'].apply(lambda x: max(7, min(10.5, round(x * 2) / 2)))
        
    # 3. Create persistent ID
    if 'id' not in df.columns:
        df['id'] = df.index

    project_root = os.path.join(os.path.dirname(__file__), "..")
    merged_team_rosters = load_merged_team_rosters(project_root)
    active_playing_squads = load_active_playing_squads(project_root)

    bb2026 = os.path.join(project_root, "IPL_2026", "ipl_2026_deliveries.csv")
    ipl2026_player_match, ipl2026_xi_by_team = build_ipl2026_aggregates(bb2026)

    upcoming_path_meta = os.path.join(project_root, "data", "upcoming_matches.csv")
    upcoming_match_meta = {}
    if os.path.exists(upcoming_path_meta):
        udf = pd.read_csv(upcoming_path_meta)
        for _, r in udf.iterrows():
            mnum = r.get("match_num")
            upcoming_match_meta[str(r["match_id"])] = {
                "match_num": int(mnum) if pd.notna(mnum) else None,
                "status": str(r.get("status", "")),
                "team1": str(r["team1"]),
                "team2": str(r["team2"]),
                "venue": str(r.get("venue", "Unknown")),
            }

@app.get("/matches")
def get_matches():
    res = []
    upcoming_path = os.path.join(os.path.dirname(__file__), "..", "data", "upcoming_matches.csv")
    if os.path.exists(upcoming_path):
        import csv
        from datetime import datetime

        today = datetime.now().date()
        with open(upcoming_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                match_date_str = str(row['date'])
                status = str(row['status'])

                # Automatically mark past matches as completed
                try:
                    match_date_obj = datetime.strptime(match_date_str, "%Y-%m-%d").date()
                    if match_date_obj < today:
                        status = "completed"
                except ValueError:
                    pass

                res.append({
                    "match_id": str(row['match_id']),
                    "team1": str(row['team1']),
                    "team2": str(row['team2']),
                    "date": match_date_str,
                    "venue": str(row['venue']),
                    "status": status
                })
        # If we have the active 2026 schedule, return it reversed so latest are at top
        res.reverse()
        return res

    # Full fallback if 2026 tracking not running
    if df is not None:
        matches_meta = df.drop_duplicates(subset=['match_id']).copy()
        for _, row in matches_meta.tail(20).iterrows():
            res.append({
                "match_id": str(row['match_id']),
                "team1": str(row['team']),
                "team2": str(row['opponent']),
                "date": str(row.get('match_date', 'Unknown')),
                "venue": str(row.get('venue', 'Unknown')),
                "status": "completed"
            })
    return res

@app.get("/match_context/{match_id}")
def get_match_context(match_id: str):
    if df is None:
        return {}
    
    match_data = df[df['match_id'].astype(str) == str(match_id)]
    if match_data.empty:
        upcoming = upcoming_match_meta.get(str(match_id), {})
        venue = upcoming.get('venue')
        if not venue:
            return {}

        venue_key = normalize_venue_name(venue)
        venue_simple = simplify_venue_name(venue)
        avg_data = venue_avg_phase.get(venue_key) or venue_avg_phase.get(venue_simple)
        if avg_data is None:
            for known_key, known_data in venue_avg_phase.items():
                if venue_simple in known_key or known_key in venue_simple:
                    avg_data = known_data
                    break

        if avg_data is not None:
            return {
                "venueAvg": int(round(avg_data.get('venue_avg_total_runs_before', 320) / 2)),
                "ppAvg": int(avg_data.get('venue_avg_powerplay_runs_before', 90) / 2),
                "moAvg": int(avg_data.get('venue_avg_middle_overs_runs_before', 150) / 2),
                "doAvg": int(avg_data.get('venue_avg_death_overs_runs_before', 80) / 2)
            }

        avg_one_side = venue_avg_batting_one_side.get(venue_key)
        if avg_one_side is None:
            avg_one_side = venue_avg_batting_one_side.get(venue_simple)
        if avg_one_side is None:
            for known_key, known_avg in venue_avg_batting_one_side.items():
                if venue_simple in known_key or known_key in venue_simple:
                    avg_one_side = known_avg
                    break
        if avg_one_side is None:
            return {}

        return {
            "venueAvg": int(round(avg_one_side)),
            "ppAvg": 45,
            "moAvg": 75,
            "doAvg": 40
        }

    first_row = match_data.iloc[0]
    venue_key = normalize_venue_name(first_row.get('venue', 'Unknown'))
    venue_simple = simplify_venue_name(first_row.get('venue', 'Unknown'))
    avg_one_side = venue_avg_batting_one_side.get(venue_key)
    if avg_one_side is None:
        avg_one_side = venue_avg_batting_one_side.get(venue_simple)
    if avg_one_side is None:
        for known_key, known_avg in venue_avg_batting_one_side.items():
            if venue_simple in known_key or known_key in venue_simple:
                avg_one_side = known_avg
                break
    if avg_one_side is None:
        avg_one_side = first_row.get('venue_avg_total_runs_before', 320) / 2

    return {
        "venueAvg": int(round(avg_one_side)),
        "ppAvg": int(first_row.get('venue_avg_powerplay_runs_before', 90) / 2),
        "moAvg": int(first_row.get('venue_avg_middle_overs_runs_before', 150) / 2),
        "doAvg": int(first_row.get('venue_avg_death_overs_runs_before', 80) / 2)
    }

def get_squad_data(match_id: str):
    if df is None:
        return pd.DataFrame()
    this_match = df[df['match_id'].astype(str) == str(match_id)]
    
    t1 = t2 = None
    season = None
    
    if this_match.empty:
        # Fallback for upcoming matches
        upcoming_path = os.path.join(os.path.dirname(__file__), "..", "data", "upcoming_matches.csv")
        if os.path.exists(upcoming_path):
            import csv
            with open(upcoming_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row['match_id']) == str(match_id):
                        t1 = row['team1'].replace('Bengaluru', 'Bangalore')
                        t2 = row['team2'].replace('Bengaluru', 'Bangalore')
                        season = df['season'].max() if 'season' in df.columns else datetime.now().year
                        break
        if not t1:
            return pd.DataFrame()
    else:
        t1 = this_match['team'].iloc[0]
        t2 = this_match['opponent'].iloc[0]
        season = this_match['season'].iloc[0]

    meta = upcoming_match_meta.get(str(match_id)) if upcoming_match_meta else None
    fixture_mno = meta.get("match_num") if meta else None
    is_fix_upcoming = bool(meta and str(meta.get("status", "")).lower() == "upcoming")
    if fixture_mno is None and ipl2026_xi_by_team:
        all_match_nums = [n for d in ipl2026_xi_by_team.values() for n in d.keys()]
        fixture_mno = max(all_match_nums) + 1 if all_match_nums else None

    def count_ipl26_appearances(player_name: str, franchise: str) -> int:
        if ipl2026_player_match is None or ipl2026_player_match.empty:
            return 0
        sub = ipl2026_player_match[ipl2026_player_match["player"].apply(lambda x: player_names_match(player_name, x))]
        if sub.empty:
            return 0
        if franchise:
            fts = schedule_team_to_short(franchise)
            if fts:
                sub = sub[sub["team_short"] == fts]
        if sub.empty:
            return 0
        if fixture_mno is None:
            return int(len(sub))
        if is_fix_upcoming:
            sub = sub[sub["match_no"] < int(fixture_mno)]
        else:
            sub = sub[sub["match_no"] <= int(fixture_mno)]
        return int(len(sub))

    def in_squad(player_name: str, franchise: str, squads_blob: dict) -> bool:
        if not squads_blob:
            return False
        k = resolve_franchise_squad_key(franchise, squads_blob)
        if not k or not squads_blob.get(k):
            return False
        return any(player_names_match(player_name, squad_player) for squad_player in squads_blob[k])

    def playing_confidence_score(player_name: str, franchise: str) -> float:
        forced = _override_value_for_player(str(player_name), PLAYER_FORCE_ACTIVE_2026)
        if forced is True:
            return 1.0
        if forced is False:
            return 0.0

        fts = schedule_team_to_short(franchise) if franchise else None
        team_match_nums = []
        if fts and ipl2026_xi_by_team and (fts in ipl2026_xi_by_team):
            team_match_nums = sorted(int(n) for n in ipl2026_xi_by_team.get(fts, {}).keys())
        if fixture_mno is not None and team_match_nums:
            if is_fix_upcoming:
                valid_team_nums = [n for n in team_match_nums if n < int(fixture_mno)]
            else:
                valid_team_nums = [n for n in team_match_nums if n <= int(fixture_mno)]
        else:
            valid_team_nums = team_match_nums
        team_latest = max(valid_team_nums) if valid_team_nums else None

        apps = count_ipl26_appearances(player_name, franchise)
        subp = pd.DataFrame()
        if ipl2026_player_match is not None and not ipl2026_player_match.empty:
            subp = ipl2026_player_match[ipl2026_player_match["player"].apply(lambda x: player_names_match(player_name, x))]
            if fts:
                subp = subp[subp["team_short"] == fts]
            if fixture_mno is not None:
                if is_fix_upcoming:
                    subp = subp[subp["match_no"] < int(fixture_mno)]
                else:
                    subp = subp[subp["match_no"] <= int(fixture_mno)]

        in_expected_xi = False
        if ipl2026_xi_by_team and fixture_mno is not None:
            expected = expected_xi_for_franchise(ipl2026_xi_by_team, franchise, int(fixture_mno))
            in_expected_xi = any(player_names_match(player_name, x) for x in expected)
        in_active_playing = in_squad(player_name, franchise, active_playing_squads)
        in_modern_squad = in_squad(player_name, franchise, modern_squads)

        if apps >= 1 and not subp.empty:
            played_nums = sorted(int(n) for n in subp["match_no"].tolist())
            last_played = played_nums[-1]
            recency_gap = (team_latest - last_played) if team_latest is not None else 0
            last2_team_nums = valid_team_nums[-2:] if len(valid_team_nums) >= 2 else valid_team_nums
            played_last2 = len([n for n in played_nums if n in set(last2_team_nums)])

            if in_expected_xi and played_last2 >= 1 and recency_gap <= 1:
                return 1.0
            if played_last2 >= 1 and recency_gap <= 1:
                return 0.92
            if recency_gap <= 2 and in_active_playing:
                return 0.82
            if recency_gap <= 2:
                return 0.70
            if recency_gap <= 4:
                return 0.44
            return 0.18

        # No match participation yet in 2026.
        if in_expected_xi and in_active_playing:
            return 0.74
        if in_active_playing:
            return 0.58
        if in_modern_squad:
            return 0.24
        return 0.08

    squad_json_path = os.path.join(os.path.dirname(__file__), "..", "data", "2026_active_squads.json")
    modern_squads = merged_team_rosters if merged_team_rosters else {}
    if not modern_squads and os.path.exists(squad_json_path):
        import json
        with open(squad_json_path, "r", encoding="utf-8") as f:
            modern_squads = json.load(f)

    if modern_squads:
        k1 = resolve_franchise_squad_key(t1, modern_squads)
        k2 = resolve_franchise_squad_key(t2, modern_squads)
        t1_players = modern_squads.get(k1, []) if k1 else []
        t2_players = modern_squads.get(k2, []) if k2 else []
        
        unique_players = df.sort_values('match_date').drop_duplicates(subset=['player_name'], keep='last').copy()

        def determine_modern_team(p_name):
            for m in t1_players:
                if player_names_match(p_name, m):
                    return t1
            for m in t2_players:
                if player_names_match(p_name, m):
                    return t2
            return None
            
        unique_players['modern_team'] = unique_players['player_name'].apply(determine_modern_team)
        unique_players = unique_players[unique_players['modern_team'].notna()].copy()
        
        active_flags = []
        for p in unique_players["player_name"]:
            is_withdrawn = any(player_names_match(p, w) for w in WITHDRAWAL_2026)
            active_flags.append(not is_withdrawn)
            
        unique_players = unique_players[active_flags].copy()
        
        unique_players["team"] = unique_players["modern_team"]  # Teleport injection
        unique_players = append_roster_only_players(
            unique_players, t1, t2, t1_players, t2_players, df
        )

    else:
        squad_data = df[(df['season'] == season) & (df['team'].isin([t1, t2]))]
        if squad_data.empty:
            squad_data = df[df['team'].isin([t1, t2])]
            
        unique_players = squad_data.sort_values('match_date').drop_duplicates(subset=['player_name'], keep='last').copy()
        unique_players = unique_players[unique_players['team'].isin([t1, t2])]

    if not unique_players.empty:
        def apply_team_override(row):
            p = str(row["player_name"])
            override_team = _override_value_for_player(p, PLAYER_TEAM_OVERRIDES_2026)
            if override_team in (t1, t2):
                return override_team
            return row["team"]

        unique_players["team"] = unique_players.apply(apply_team_override, axis=1)
        unique_players["Role"] = unique_players.apply(
            lambda row: _override_value_for_player(str(row["player_name"]), PLAYER_ROLE_OVERRIDES_2026) or str(row.get("Role", "AR")),
            axis=1,
        )
        unique_players["playing_confidence"] = unique_players.apply(
            lambda row: playing_confidence_score(str(row["player_name"]), str(row["team"])),
            axis=1,
        )
        unique_players["is_probably_playing"] = unique_players["playing_confidence"] >= 0.65
        unique_players = _collapse_name_variants_for_match(unique_players)
    else:
        unique_players['is_probably_playing'] = []
        unique_players["playing_confidence"] = []

    # Phase 3 — IPL 2026 form: role-aware hot form based on real 2026 match involvement.
    unique_players['is_truly_hot'] = False
    before_mno = int(fixture_mno) if (is_fix_upcoming and fixture_mno is not None) else None

    season_l5_line = None
    if ipl2026_player_match is not None and not ipl2026_player_match.empty:
        season_l5_line = ipl2026_season_l5_percentile_line(
            ipl2026_player_match, before_match_num=before_mno
        )

    for idx, row in unique_players.iterrows():
        p_name = row['player_name']
        role = str(row.get("Role", "AR"))
        p_hist = df[df['player_name'] == p_name].sort_values('match_date').tail(5)
        legacy_l5 = p_hist['proj_points'].mean() if len(p_hist) > 0 else float(row['proj_points'])
        proj = float(row['proj_points'])

        fctx = {"l5": None, "prior": None, "n": 0}
        sub26 = pd.DataFrame()
        if ipl2026_player_match is not None and not ipl2026_player_match.empty:
            fctx = ipl2026_form_context(
                ipl2026_player_match,
                p_name,
                player_names_match,
                before_match_num=before_mno,
                franchise_schedule_name=str(row["team"]),
            )
            sub26 = ipl2026_player_match[
                ipl2026_player_match["player"].apply(lambda x: player_names_match(p_name, x))
            ]
            fts = schedule_team_to_short(str(row["team"]))
            if fts:
                sub26 = sub26[sub26["team_short"] == fts]
            if before_mno is not None:
                sub26 = sub26[sub26["match_no"] < before_mno]
            sub26 = sub26.sort_values("match_no")

        l5_avg = float(fctx["l5"]) if fctx.get("l5") is not None else legacy_l5
        n26 = int(fctx.get("n") or 0)
        l5v = fctx.get("l5")
        priorv = fctx.get("prior")

        if modern_squads:
            is_active_2026 = True
        else:
            live_stats_path = os.path.join(os.path.dirname(__file__), "..", "data", "2026_live_stats.json")
            if os.path.exists(live_stats_path):
                import json
                with open(live_stats_path, "r", encoding="utf-8") as f:
                    live_stats = json.load(f)
                is_active_2026 = any(player_names_match(p_name, lp) for lp in live_stats.keys())
            else:
                is_active_2026 = True

        hot = False
        boost = 0.0
        if ipl2026_player_match is not None and not ipl2026_player_match.empty and is_active_2026 and n26 >= 2 and l5v is not None:
            tail3 = sub26.tail(min(3, len(sub26)))
            r3_runs = float(tail3["runs_bat"].sum()) if not tail3.empty else 0.0
            r3_wkts = float(tail3["wickets"].sum()) if not tail3.empty else 0.0
            r3_balls_bat = float(tail3["balls_bat"].sum()) if not tail3.empty else 0.0
            r3_balls_bowl = float(tail3["balls_bowl"].sum()) if not tail3.empty else 0.0
            r3_runs_con = float(tail3["runs_conceded"].sum()) if not tail3.empty else 0.0

            r3_sr = (r3_runs / r3_balls_bat * 100.0) if r3_balls_bat > 0 else 0.0
            r3_econ = (r3_runs_con / (r3_balls_bowl / 6.0)) if r3_balls_bowl > 0 else 99.0
            l5_delta = float(l5v) - float(priorv) if priorv is not None else max(0.0, float(l5v) - proj * 0.9)
            season_standout = season_l5_line is not None and float(l5v) >= float(season_l5_line)

            bat_hot = (r3_runs >= 85.0) or (r3_sr >= 145.0 and r3_balls_bat >= 35.0)
            bowl_hot = (r3_wkts >= 4.0) or (r3_balls_bowl >= 18.0 and r3_econ <= 7.2 and r3_wkts >= 2.0)
            ar_hot = (r3_runs >= 55.0 and r3_wkts >= 2.0) or (bat_hot and bowl_hot)

            role_hot = False
            if role in ("BAT", "WK"):
                role_hot = bat_hot
            elif role == "BOWL":
                role_hot = bowl_hot
            else:
                role_hot = ar_hot or bat_hot or bowl_hot

            trend_hot = (float(l5v) >= proj * 0.92 and l5_delta >= 4.0) or season_standout

            if role_hot and trend_hot:
                hot = True
                raw = (float(l5v) - proj) * 0.58
                if role in ("BAT", "WK") and bat_hot:
                    raw += 2.0
                elif role == "BOWL" and bowl_hot:
                    raw += 2.0
                elif role == "AR" and (ar_hot or (bat_hot and bowl_hot)):
                    raw += 3.0
                boost = max(-10.0, min(24.0, raw))
        else:
            # Fallback only when 2026 form is unavailable.
            if l5_avg > proj * 0.95 and len(p_hist) >= 2:
                hot = True
                raw = (l5_avg - proj) * 0.45
                boost = max(-8.0, min(12.0, raw))

        if hot and boost != 0.0:
            unique_players.at[idx, "proj_points"] = proj + boost
            unique_players.at[idx, "is_truly_hot"] = True

        delta = _override_value_for_player(str(p_name), PLAYER_FORM_DELTA_OVERRIDE_2026)
        if delta is not None:
            unique_players.at[idx, "proj_points"] = float(unique_players.at[idx, "proj_points"]) + float(delta)
            if float(delta) > 0:
                unique_players.at[idx, "is_truly_hot"] = True

    return unique_players


def match_data_for_season_realistic_optimizer(match_data: pd.DataFrame, match_id: str) -> pd.DataFrame:
    """
    Restrict Dream11 optimization to players who have IPL 2026 on-field minutes before this fixture,
    or who appear in the latest expected XI from ball-by-ball (so bench / unused squad don't fill the 11).
    Falls back to full squad if the filter would leave fewer than 11 options.
    """
    if match_data.empty:
        return match_data
    if "playing_confidence" in match_data.columns:
        strong_pool = match_data[match_data["playing_confidence"] >= 0.65].copy()
        if len(strong_pool) >= 11:
            return strong_pool.reset_index(drop=True)
        medium_pool = match_data[match_data["playing_confidence"] >= 0.50].copy()
        if len(medium_pool) >= 11:
            return medium_pool.reset_index(drop=True)
    if "is_probably_playing" in match_data.columns:
        active_pool = match_data[match_data["is_probably_playing"] == True].copy()
        if len(active_pool) >= 11:
            return active_pool.reset_index(drop=True)

    if ipl2026_player_match is None or ipl2026_player_match.empty:
        return match_data
    if not ipl2026_xi_by_team:
        return match_data
    meta = upcoming_match_meta.get(str(match_id)) if upcoming_match_meta else None
    if not meta or meta.get("match_num") is None:
        return match_data
    fm = int(meta["match_num"])
    is_up = str(meta.get("status", "")).lower() == "upcoming"

    def eligible(row):
        franchise = str(row["team"])
        p = row["player_name"]
        nplay = ipl2026_onfield_appearances_before_fixture(
            ipl2026_player_match,
            p,
            player_names_match,
            fm,
            is_up,
            franchise_schedule_name=franchise,
        )
        if nplay >= 1:
            return True
        xi_fr = expected_xi_for_franchise(ipl2026_xi_by_team, franchise, fm)
        if xi_fr and any(player_names_match(p, x) for x in xi_fr):
            return True
        return False

    narrowed = match_data[match_data.apply(eligible, axis=1)]
    if len(narrowed) >= 11:
        return narrowed
    return match_data


def _infer_bowling_style(player_name: str, role: str) -> str:
    """Stable pace vs spin label for BOWL/AR when we have no bowling-type column (pitch/weather tuning)."""
    if role not in ("BOWL", "AR"):
        return "neutral"
    h = hashlib.md5(str(player_name).encode("utf-8")).hexdigest()
    v = int(h[:12], 16) % 100
    return "spin" if v < 43 else "pace"


def _is_batting_first_team(team: str, batting_first_team: Optional[str]) -> bool:
    if not batting_first_team or batting_first_team.upper().strip() in ("NONE", ""):
        return False
    def norm_name(x: str) -> str:
        s = str(x).strip().upper()
        s = s.replace("BENGALURU", "BANGALORE")
        s = re.sub(r"[^A-Z0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def abbr_name(x: str) -> str:
        parts = [p for p in norm_name(x).split() if p]
        return "".join(p[0] for p in parts)[:3].upper() if parts else ""

    b = norm_name(str(batting_first_team))
    t = norm_name(str(team))
    return t == b or abbr_name(team) == b or abbr_name(batting_first_team) == t[:3]


def lineup_context_multiplier(
    role: str,
    player_name: str,
    team: str,
    pitch_type: Optional[str],
    weather_condition: Optional[str],
    batting_first_team: Optional[str],
) -> float:
    """
    Multiplicative adjustment to projected points for pitch (pace/spin friendly),
    weather (clear / overcast / dew), and toss (who bats first).
    """
    style = _infer_bowling_style(player_name, role)
    is_bat_first = _is_batting_first_team(team, batting_first_team)

    pitch = (pitch_type or "Neutral").strip()
    weather = (weather_condition or "Clear").strip()

    m = 1.0

    # Toss: first innings slightly favours top-order batters; second innings favours bowlers a touch
    if batting_first_team and batting_first_team.upper().strip() not in ("NONE", ""):
        if is_bat_first:
            if role in ("BAT", "AR", "WK"):
                m *= 1.12
        else:
            if role == "BOWL":
                m *= 1.14
            elif role == "AR":
                m *= 1.09

    # Pitch
    if pitch == "Pace":
        if role == "BAT":
            m *= 1.08
        elif role == "WK":
            m *= 1.08
        elif role == "BOWL":
            if style == "pace":
                m *= 1.22
            elif style == "spin":
                m *= 0.82
            else:
                m *= 1.04
        elif role == "AR":
            if style == "pace":
                m *= 1.17
            elif style == "spin":
                m *= 0.86
            else:
                m *= 1.04
    elif pitch == "Spin":
        if role == "BAT":
            m *= 0.93
        elif role == "WK":
            m *= 0.94
        elif role == "BOWL":
            if style == "spin":
                m *= 1.24
            elif style == "pace":
                m *= 0.80
            else:
                m *= 1.03
        elif role == "AR":
            if style == "spin":
                m *= 1.18
            elif style == "pace":
                m *= 0.84
            else:
                m *= 1.04

    # Weather (stacks with pitch)
    if weather == "Overcast":
        if role in ("BOWL", "AR"):
            if style == "pace":
                m *= 1.15
            elif style == "spin":
                m *= 0.92
            else:
                m *= 1.03
        else:
            m *= 0.95
        if not is_bat_first and role in ("BOWL", "AR") and style == "pace":
            m *= 1.08
    elif weather == "Dew":
        if is_bat_first and role in ("BOWL", "AR"):
            m *= 0.80
        if not is_bat_first and role in ("BAT", "WK"):
            m *= 1.14
        if is_bat_first and role in ("BAT", "WK"):
            m *= 1.05

    # Keep context effects strong but bounded.
    return max(0.72, min(1.34, m))


@app.get("/players/{match_id}")
def get_players(match_id: str):
    match_data = get_squad_data(match_id)
    
    if match_data.empty:
        return []

    results = []
    for i, row in match_data.iterrows():
        team_parts = str(row['team']).split()
        team_abbr = "".join([t[0] for t in team_parts])[:3].upper() if team_parts else "UNK"
        
        is_hot = row.get('is_truly_hot', False)

        results.append({
            "id": int(row['id']),
            "name": str(row['player_name']),
            "team": team_abbr,
            "role": str(row['Role']),
            "credits": float(row['credits']),
            "proj_points": float(row['proj_points']),
            "radar": [
                {"phase": "PP", "val": random.randint(40, 95)},
                {"phase": "MO", "val": random.randint(40, 95)},
                {"phase": "DO", "val": random.randint(40, 95)}
            ],
            "form_tag": "hot" if is_hot else "none",
            "isOptimal": False,
            "isCaptain": False,
            "isViceCaptain": False
        })
    return results

@app.get("/player_stats/{match_id}/{player_name}")
def get_player_stats(match_id: str, player_name: str):
    if df is None:
        return []

    meta = upcoming_match_meta.get(str(match_id)) if upcoming_match_meta else None
    before_mno = int(meta["match_num"]) if meta and str(meta.get("status", "")).lower() == "upcoming" and meta.get("match_num") is not None else None

    if ipl2026_player_match is not None and not ipl2026_player_match.empty:
        r26 = ipl2026_recent_matches_for_player(
            ipl2026_player_match, player_name, player_names_match, 12
        )
        if before_mno is not None:
            r26 = r26[r26["match_no"] < before_mno]
        r26 = r26.tail(5)
        if not r26.empty:
            stats = []
            for _, row in r26.iterrows():
                opp_s = str(row.get("opponent_short") or "")
                opp_label = SHORT_TO_SCHEDULE_NAME.get(opp_s, opp_s) if opp_s else "—"
                runs = int(row.get("runs_bat") or 0)
                balls = int(row.get("balls_bat") or 0)
                sr = round(100.0 * runs / balls, 2) if balls > 0 else 0.0
                bbowl = float(row.get("balls_bowl") or 0)
                rconc = float(row.get("runs_conceded") or 0)
                econ = round(rconc / (bbowl / 6.0), 2) if bbowl > 0 else 0.0
                stats.append({
                    "match_date": f"IPL 2026 · Match {int(row['match_no'])}",
                    "opponent": opp_label,
                    "runs_scored": runs,
                    "wickets": int(row.get("wickets", 0) or 0),
                    "strike_rate": sr,
                    "fantasy_points": float(row.get("fantasy_proxy", 0)),
                    "boundary_runs": 0,
                    "non_boundary_runs": 0,
                    "economy": econ,
                    "balls_bowled": int(row.get("balls_bowl", 0) or 0),
                })
            return stats

    player_data = df[df['player_name'] == player_name].sort_values('match_date').tail(5)

    stats = []
    for _, row in player_data.iterrows():
        stats.append({
            "match_date": str(row.get('match_date', 'Unknown')),
            "opponent": str(row.get('opponent', 'Unknown')),
            "runs_scored": int(row.get('runs_scored', 0)) if pd.notna(row.get('runs_scored')) else 0,
            "wickets": int(row.get('wickets', 0)) if pd.notna(row.get('wickets')) else 0,
            "strike_rate": float(row.get('batting_strike_rate', 0)) if pd.notna(row.get('batting_strike_rate')) else 0.0,
            "fantasy_points": float(row.get('proj_points', 0)) if pd.notna(row.get('proj_points')) else 0.0,
            "boundary_runs": int(row.get('boundary_runs', 0)) if pd.notna(row.get('boundary_runs')) else 0,
            "non_boundary_runs": int(row.get('non_boundary_runs', 0)) if pd.notna(row.get('non_boundary_runs')) else 0,
            "economy": float(row.get('bowling_economy', 0)) if pd.notna(row.get('bowling_economy')) else 0.0,
            "balls_bowled": int(row.get('balls_bowled', 0)) if pd.notna(row.get('balls_bowled')) else 0
        })
    return stats

@app.get("/matchup/{match_id}/{player_name}")
def get_matchup_stats(match_id: str, player_name: str):
    if df is None:
        return {"error": "Dataset not loaded"}

    p_name_hist = resolve_df_player_name(player_name)

    # 1. Figure out tonight's opponent (historical match row OR 2026 schedule + squad)
    match_meta = df[df['match_id'].astype(str) == str(match_id)]
    if not match_meta.empty:
        t1 = match_meta['team'].iloc[0]
        t2 = match_meta['opponent'].iloc[0]
        player_tonight = match_meta[match_meta['player_name'] == p_name_hist]
        if player_tonight.empty:
            return {"error": "Player not found in this match"}
        p_team = player_tonight['team'].iloc[0]
    else:
        meta = upcoming_match_meta.get(str(match_id)) if upcoming_match_meta else None
        if not meta:
            return {"error": "Match not found"}
        t1 = meta["team1"].replace("Bengaluru", "Bangalore")
        t2 = meta["team2"].replace("Bengaluru", "Bangalore")
        p_team = None
        squads = merged_team_rosters if merged_team_rosters else {}
        if squads:
            k1 = resolve_franchise_squad_key(t1, squads)
            k2 = resolve_franchise_squad_key(t2, squads)
            for roster_name in squads.get(k1, []) if k1 else []:
                if player_names_match(player_name, roster_name):
                    p_team = t1
                    break
            if p_team is None:
                for roster_name in squads.get(k2, []) if k2 else []:
                    if player_names_match(player_name, roster_name):
                        p_team = t2
                        break
        if p_team is None:
            return {"error": "Player not found in this match"}

    p_opponent = t2 if p_team == t1 else t1

    # 2. Get pre-aggregated history only against tonight's franchise.
    normalized_opponent = _normalize_team_name_for_lookup(p_opponent)
    h2h_totals = h2h_lookup.get((str(p_name_hist), normalized_opponent), {})
    matches_played = int(h2h_totals.get("matches_played", 0))
    total_runs = int(h2h_totals.get("total_runs", 0))
    total_balls = int(h2h_totals.get("total_balls", 0))
    sr = (total_runs / total_balls * 100) if total_balls > 0 else 0
    total_wickets = int(h2h_totals.get("total_wickets", 0))
    avg_fantasy = float(h2h_totals.get("avg_fantasy_points", 0.0))
    
    # Micro Matchups (Batter vs Bowler) — legacy ball-by-ball file; names align with historical dataset
    micro = []
    if bbb_matchups is not None:
        opp_squad = df[(df['match_id'].astype(str) == str(match_id)) & (df['team'] == p_opponent)]
        if opp_squad.empty:
            sd = get_squad_data(match_id)
            if not sd.empty:
                opp_squad = sd[sd["team"] == p_opponent]
        opp_bowlers = opp_squad[opp_squad['Role'].isin(['BOWL', 'AR'])]['player_name'].tolist()

        filtered_bbb = bbb_matchups[
            (bbb_matchups['batter'] == p_name_hist) & (bbb_matchups['bowler'].isin(opp_bowlers))
        ]
        
        for _, r in filtered_bbb.iterrows():
            runs = int(r['runs'])
            balls = int(r['balls'])
            wkts = int(r['wickets'])
            sr = round((runs / balls * 100), 2) if balls > 0 else 0
            micro.append({
                "bowler": str(r['bowler']),
                "runs": runs,
                "balls": balls,
                "wickets": wkts,
                "sr": sr
            })
    
    return {
        "opponent": str(p_opponent),
        "matches_played": matches_played,
        "total_runs": int(total_runs),
        "strike_rate": round(sr, 2),
        "total_wickets": int(total_wickets),
        "avg_fantasy_points": round(avg_fantasy, 2),
        "micro_matchups": micro
    }


@app.post("/optimize")
def optimize_lineup(req: OptimizeRequest):
    if df is None:
        return {"error": "Dataset not loaded"}
        
    match_data = get_squad_data(req.match_id)
    match_data = match_data_for_season_realistic_optimizer(match_data, req.match_id)

    if len(match_data) < 11:
        return {"error": "Not enough players in this match fixture."}
        
    match_data = match_data.copy()
    if req.batting_first_team:
        bat_team = str(req.batting_first_team).upper().strip()
    else:
        bat_team = None

    def adjust_points(row):
        base = float(row["proj_points"])
        conf = float(row.get("playing_confidence", 0.5))
        if conf >= 0.85:
            base *= 1.08
        elif conf >= 0.65:
            base *= 1.04
        elif conf >= 0.45:
            base *= 0.88
        elif conf >= 0.25:
            base *= 0.62
        else:
            base *= 0.38
        mult = lineup_context_multiplier(
            str(row["Role"]),
            str(row["player_name"]),
            str(row["team"]),
            req.pitch_type,
            req.weather_condition,
            bat_team,
        )
        # Amplify contextual signal so pitch/weather/toss settings clearly change XI composition.
        context_amplified = 1.0 + (mult - 1.0) * 1.35
        context_amplified = max(0.66, min(1.46, context_amplified))
        return base * context_amplified

    match_data["proj_points"] = match_data.apply(adjust_points, axis=1)
        
    # **Knapsack Linear Programming Setup**
    prob = pulp.LpProblem("Dream11_Optimization", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("player", match_data['id'].tolist(), cat="Binary")

    # 1. Objective Function: Maximize the Sum of Projected Points
    prob += pulp.lpSum([
        match_data[match_data['id'] == pid]['proj_points'].values[0] * player_vars[pid] 
        for pid in match_data['id'].tolist()
    ])

    # 2. Constraints: Exactly 11 Players
    prob += pulp.lpSum([player_vars[pid] for pid in match_data['id'].tolist()]) == 11
    
    # 3. Constraints: Total Selection Budget <= 100 Credits
    prob += pulp.lpSum([
        match_data[match_data['id'] == pid]['credits'].values[0] * player_vars[pid] 
        for pid in match_data['id'].tolist()
    ]) <= 100.0

    # 4. Constraints: Maximum 7 players from any single franchise
    teams = match_data['team'].unique()
    for team in teams:
        team_pids = match_data[match_data['team'] == team]['id'].tolist()
        prob += pulp.lpSum([player_vars[pid] for pid in team_pids]) <= 7

    # 5. Constraints: Positional Role Boundaries 
    # Minimum 1, Max 4 Wicket Keepers
    wk_pids = match_data[match_data['Role'] == 'WK']['id'].tolist()
    prob += pulp.lpSum([player_vars[pid] for pid in wk_pids]) >= 1
    prob += pulp.lpSum([player_vars[pid] for pid in wk_pids]) <= 4
    
    # Minimum 3, Max 6 Batters
    bat_pids = match_data[match_data['Role'] == 'BAT']['id'].tolist()
    prob += pulp.lpSum([player_vars[pid] for pid in bat_pids]) >= 3
    prob += pulp.lpSum([player_vars[pid] for pid in bat_pids]) <= 6

    # Minimum 1, Max 4 All-Rounders
    ar_pids = match_data[match_data['Role'] == 'AR']['id'].tolist()
    prob += pulp.lpSum([player_vars[pid] for pid in ar_pids]) >= 1
    prob += pulp.lpSum([player_vars[pid] for pid in ar_pids]) <= 4

    # Minimum 3, Max 6 Bowlers
    bowl_pids = match_data[match_data['Role'] == 'BOWL']['id'].tolist()
    prob += pulp.lpSum([player_vars[pid] for pid in bowl_pids]) >= 3
    prob += pulp.lpSum([player_vars[pid] for pid in bowl_pids]) <= 6

    # 6. Constraints: User Overrides (Locking and Banning specific players)
    for pid in req.locked_player_ids:
        if pid in player_vars:
            prob += player_vars[pid] == 1  # Force Inclusion

    for pid in req.banned_player_ids:
        if pid in player_vars:
            prob += player_vars[pid] == 0  # Force Exclusion

    all_lineups = []
    
    for iteration in range(req.num_lineups):
        # Execute LP solver
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
        except Exception:
            prob.solve()

        if pulp.LpStatus[prob.status] != 'Optimal':
            break # No more optimal solutions

        # Extract 11 active indices
        optimal_pids = [pid for pid in player_vars if player_vars[pid].varValue == 1.0]
        
        # Add a constraint to forbid this exact lineup in the next iteration
        # Sum of variables for the chosen XI must be <= 10 (or 9 to force more variance)
        prob += pulp.lpSum([player_vars[pid] for pid in optimal_pids]) <= 9
        
        # **C and VC Assignment Strategy**
        # Sort the optimal 11 descending by projected points and use unique IDs.
        optimal_players_df = match_data[match_data['id'].isin(optimal_pids)].copy()
        optimal_players_df = optimal_players_df.sort_values(
            by=["proj_points"], ascending=False
        ).drop_duplicates(subset=["id"])

        top_pid = None
        vc_pid = None
        if len(optimal_players_df) >= 1:
            top_pid = optimal_players_df.iloc[0]['id']
        if len(optimal_players_df) >= 2:
            vc_pid = optimal_players_df.iloc[1]['id']

        # Finalize payload mapping for this lineup
        results = []
        for _, row in match_data.iterrows():
            pid = row['id']
            is_opt = pid in optimal_pids
            is_cap = is_opt and (pid == top_pid)
            is_vc = is_opt and (pid == vc_pid)
            
            team_parts = str(row['team']).split()
            team_abbr = "".join([t[0] for t in team_parts])[:3].upper() if team_parts else "UNK"

            is_hot = bool(row.get("is_truly_hot", False))

            results.append({
                "id": int(pid),
                "name": str(row['player_name']),
                "team": team_abbr,
                "role": str(row['Role']),
                "credits": float(row['credits']),
                "proj_points": float(row['proj_points']),
                "radar": [
                    {"phase": "PP", "val": random.randint(40, 95)},
                    {"phase": "MO", "val": random.randint(40, 95)},
                    {"phase": "DO", "val": random.randint(40, 95)}
                ],
                "form_tag": "hot" if is_hot else "none",
                "isOptimal": bool(is_opt),
                "isCaptain": bool(is_cap),
                "isViceCaptain": bool(is_vc),
                "isPlayingCandidate": bool(row.get("is_probably_playing", False))
            })
            
        all_lineups.append(results)

    if not all_lineups:
        return {"error": "Could not find a valid lineup under these constraints."}

    return {"lineups": all_lineups}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
