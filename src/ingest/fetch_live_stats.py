import os
import json
import cloudscraper
from bs4 import BeautifulSoup
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def fetch_sportskeeda_stats():
    print("Scraping live 2026 IPL player stats from Sportskeeda...")
    scraper = cloudscraper.create_scraper()
    
    live_stats = {}
    
    # 1. Fetch Orange Cap (Batters & All-rounders)
    try:
        html_bat = scraper.get('https://www.sportskeeda.com/go/ipl/orange-cap').text
        soup_bat = BeautifulSoup(html_bat, 'html.parser')
        bat_rows = soup_bat.find_all('tr')
        
        for row in bat_rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 6:
                try:
                    p_name = cells[1].text.strip()
                    matches = int(cells[3].text.strip())
                    runs = int(cells[5].text.strip())
                    
                    live_stats[p_name] = {
                        "player_name": p_name,
                        "matches_played": matches,
                        "runs": runs,
                        "wickets": 0
                    }
                except ValueError:
                    continue  # Skip header rows
    except Exception as e:
        print(f"Error fetching batting stats: {e}")

    # 2. Fetch Purple Cap (Bowlers & All-rounders)
    try:
        html_bowl = scraper.get('https://www.sportskeeda.com/go/ipl/purple-cap').text
        soup_bowl = BeautifulSoup(html_bowl, 'html.parser')
        bowl_rows = soup_bowl.find_all('tr')
        
        for row in bowl_rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 7:
                try:
                    p_name = cells[1].text.strip()
                    matches = int(cells[3].text.strip())
                    wickets = int(cells[6].text.strip())
                    
                    if p_name in live_stats:
                        live_stats[p_name]["matches_played"] = max(live_stats[p_name]["matches_played"], matches)
                        live_stats[p_name]["wickets"] = wickets
                    else:
                        live_stats[p_name] = {
                            "player_name": p_name,
                            "matches_played": matches,
                            "runs": 0,
                            "wickets": wickets
                        }
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error fetching bowling stats: {e}")

    # Save to JSON database
    output_path = DATA_DIR / "2026_live_stats.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(live_stats, f, indent=4)
        
    print(f"Successfully automated {len(live_stats)} active 2026 playing rosters!")

if __name__ == "__main__":
    fetch_sportskeeda_stats()
