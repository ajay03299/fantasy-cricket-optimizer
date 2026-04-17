import os
import random
import pandas as pd
import pulp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

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

@app.on_event("startup")
def load_data():
    global df, bbb_matchups
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
        
        def assign_dream11_credit_historic(p_name):
            score = career_scores.get(p_name, 0)
            if score >= p98: return 10.5
            elif score >= p94: return 10.0
            elif score >= p85: return 9.5
            elif score >= p70: return 9.0
            elif score >= p50: return 8.5
            elif score >= p30: return 8.0
            elif score >= p15: return 7.5
            else: return 7.0

        df['credits'] = df['player_name'].apply(assign_dream11_credit_historic)
        
    # 3. Create persistent ID
    if 'id' not in df.columns:
        df['id'] = df.index

@app.get("/matches")
def get_matches():
    res = []
    
    upcoming_path = os.path.join(os.path.dirname(__file__), "..", "data", "upcoming_matches.csv")
    if os.path.exists(upcoming_path):
        import csv
        with open(upcoming_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                res.append({
                    "match_id": str(row['match_id']),
                    "team1": str(row['team1']),
                    "team2": str(row['team2']),
                    "date": str(row['date']),
                    "venue": str(row['venue']),
                    "status": str(row['status'])
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
        return {}

    first_row = match_data.iloc[0]
    
    return {
        "venueAvg": int(first_row.get('venue_avg_total_runs_before', 320) / 2),
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
                        t1 = row['team1']
                        t2 = row['team2']
                        season = df['season'].max() if 'season' in df.columns else datetime.now().year
                        break
        if not t1:
            return pd.DataFrame()
    else:
        t1 = this_match['team'].iloc[0]
        t2 = this_match['opponent'].iloc[0]
        season = this_match['season'].iloc[0]
    
    squad_data = df[(df['season'] == season) & (df['team'].isin([t1, t2]))]
    if squad_data.empty:
        # Relax season constraint if not matching fully
        squad_data = df[df['team'].isin([t1, t2])]
        
    unique_players = squad_data.sort_values('match_date').drop_duplicates(subset=['player_name'], keep='last').copy()
    unique_players = unique_players[unique_players['team'].isin([t1, t2])]
    return unique_players

@app.get("/players/{match_id}")
def get_players(match_id: str):
    match_data = get_squad_data(match_id)
    
    if match_data.empty:
        return []

    results = []
    for i, row in match_data.iterrows():
        team_parts = str(row['team']).split()
        team_abbr = "".join([t[0] for t in team_parts])[:3].upper() if team_parts else "UNK"
        
        runs = row.get('runs_scored', 0)
        wkts = row.get('wickets', 0)
        is_hot = (pd.notna(runs) and float(runs) > 40) or (pd.notna(wkts) and float(wkts) >= 2)

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
    
    # 1. Figure out tonight's opponent
    match_meta = df[df['match_id'].astype(str) == str(match_id)]
    if match_meta.empty:
        return {"error": "Match not found"}
        
    t1 = match_meta['team'].iloc[0]
    t2 = match_meta['opponent'].iloc[0]
    
    # Check which team the player belongs to in tonight's match
    player_tonight = match_meta[match_meta['player_name'] == player_name]
    if player_tonight.empty:
        return {"error": "Player not found in this match"}
        
    p_team = player_tonight['team'].iloc[0]
    p_opponent = t2 if p_team == t1 else t1
    
    # 2. Get history ONLY against this specific franchise across all seasons
    h2h_data = df[(df['player_name'] == player_name) & (df['opponent'] == p_opponent)]
    
    total_runs = h2h_data['runs_scored'].sum()
    total_balls = h2h_data['balls_faced'].sum()
    sr = (total_runs / total_balls * 100) if total_balls > 0 else 0
    total_wickets = h2h_data['wickets'].sum()
    avg_fantasy = h2h_data['proj_points'].mean() if len(h2h_data) > 0 else 0
    
    # Micro Matchups (Batter vs Bowler)
    micro = []
    if bbb_matchups is not None:
        opp_squad = df[(df['match_id'].astype(str) == str(match_id)) & (df['team'] == p_opponent)]
        opp_bowlers = opp_squad[opp_squad['Role'].isin(['BOWL', 'AR'])]['player_name'].tolist()
        
        filtered_bbb = bbb_matchups[(bbb_matchups['batter'] == player_name) & (bbb_matchups['bowler'].isin(opp_bowlers))]
        
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
        "matches_played": len(h2h_data),
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
    
    if len(match_data) < 11:
        return {"error": "Not enough players in this match fixture."}
        
    match_data = match_data.copy()
    if req.batting_first_team:
        bat_team = str(req.batting_first_team).upper().strip()
    else:
        bat_team = None

    def adjust_points(row):
        pts = float(row['proj_points'])
        team_parts = str(row['team']).split()
        team_abbr = "".join([t[0] for t in team_parts])[:3].upper() if team_parts else "UNK"
        
        is_bat_first = False
        if bat_team and (team_abbr == bat_team or str(row['team']).upper().strip() == bat_team):
            is_bat_first = True

        role = row['Role']
        
        # Determine pseudo bowling style for Spin vs Pace modifiers deterministically
        # If ID is even -> Pace, If odd -> Spin (approximated for demo purposes)
        p_id = int(row['id'])
        is_spin = (p_id % 2 != 0) and role in ['BOWL', 'AR']
        is_pace = (p_id % 2 == 0) and role in ['BOWL', 'AR']

        # Batting First modifiers
        if bat_team:
            if is_bat_first:
                if role in ['BAT', 'AR', 'WK']:
                    pts *= 1.05
            else:
                if role == 'BOWL':
                    pts *= 1.05
                    
        # Feature 3: Contextual AI Pitch Adjustments
        if req.pitch_type in ['Pace', 'Spin']:
            if role in ['BOWL', 'AR']:
                pts *= 1.10
            elif role == 'BAT':
                pts *= 0.95
                
        # Feature 3: Contextual AI Weather Adjustments
        if req.weather_condition == 'Overcast':
            # Early swing advantage for Team Bowling First (which means they bat second)
            if not is_bat_first and role in ['BOWL', 'AR']:
                pts *= 1.10
        elif req.weather_condition == 'Dew':
            # Extreme disadvantage for Team Bowling Second (which means they batted first)
            if is_bat_first and role in ['BOWL', 'AR']:
                pts *= 0.85
            # Boost to Team Batting Second as the ball skids to the bat
            if not is_bat_first and role in ['BAT', 'WK']:
                pts *= 1.10

        return pts
    match_data['proj_points'] = match_data.apply(adjust_points, axis=1)
        
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
        # Sort the optimal 11 descending by their pure projected points.
        optimal_players_df = match_data[match_data['id'].isin(optimal_pids)].copy()
        optimal_players_df = optimal_players_df.sort_values('proj_points', ascending=False)

        top_pid = None
        vc_pid = None
        if len(optimal_players_df) >= 2:
            top_pid = optimal_players_df.iloc[0]['id']
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

            runs = row.get('runs_scored', 0)
            wkts = row.get('wickets', 0)
            is_hot = (pd.notna(runs) and float(runs) > 40) or (pd.notna(wkts) and float(wkts) >= 2)

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
                "isViceCaptain": bool(is_vc)
            })
            
        all_lineups.append(results)

    if not all_lineups:
        return {"error": "Could not find a valid lineup under these constraints."}

    return {"lineups": all_lineups}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
