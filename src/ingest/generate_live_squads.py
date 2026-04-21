import json
import os
import pandas as pd

def generate_2026_rosters():
    # Load historical dataset
    csv_path = r"C:\Users\Varshith M Gowda\Downloads\Fantasy-Cricket-Optimizer-main\phase9-dataset\Fantasy-Cricket-Optimizer-phase9-production-pipeline\output\player_match_fantasy_v5.csv"
    df = pd.read_csv(csv_path)
    
    latest_squads = {}
    teams = df['team'].unique()

    # Grab the latest 25 distinct names for each team as a baseline
    for team in teams:
        team_players = df[df['team'] == team].sort_values('match_date', ascending=False)
        players = team_players['player_name'].drop_duplicates().head(25).tolist()
        latest_squads[team] = players

    # The 2026 Teleportation Mega-Auction Trades:
    # Explicitly trading super-stars to demonstrate the ML engine's decoupled franchise recognition.
    trades = [
        ("Hardik Pandya", "Mumbai Indians"),
        ("Cameron Green", "Royal Challengers Bangalore"),
        ("Rohit Sharma", "Chennai Super Kings"),
        ("KL Rahul", "Royal Challengers Bangalore"),
        ("Rishabh Pant", "Chennai Super Kings"),
        ("Shreyas Iyer", "Delhi Capitals"),
        ("Mohammed Shami", "Kolkata Knight Riders"),
        ("Suryakumar Yadav", "Royal Challengers Bangalore"),
        ("Rashid Khan", "Mumbai Indians"),
        ("Jasprit Bumrah", "Gujarat Titans")
    ]

    # Explicit 2026 Releases/Bans to remove from specific historical lists
    released_players = {
        "Royal Challengers Bangalore": ["Maxwell", "C Green", "Cameron Green", "Glenn Maxwell", "Harshal Patel"],
        "Sunrisers Hyderabad": ["Agarwal", "Mayank Agarwal"]
    }

    for player, new_team in trades:
        # Strip from all historical rosters (Fuzzy Search)
        for t in latest_squads:
            latest_squads[t] = [p for p in latest_squads[t] if player.lower().replace(" ", "") not in p.lower().replace(" ", "")]
        
        # Inject aggressively into the new 2026 roster
        if new_team in latest_squads:
            latest_squads[new_team].append(player)
            
    # Purge released players from squads
    for team, releases in released_players.items():
        if team in latest_squads:
            for released in releases:
                latest_squads[team] = [p for p in latest_squads[team] if released.lower().replace(" ", "") not in p.lower().replace(" ", "")]

    # Save to data directory
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "2026_active_squads.json")
    with open(output_path, 'w') as f:
        json.dump(latest_squads, f, indent=4)
        
    print(f"Aggressively rebuilt 2026 squads. Teleported {len(trades)} super-stars.")

if __name__ == "__main__":
    generate_2026_rosters()
