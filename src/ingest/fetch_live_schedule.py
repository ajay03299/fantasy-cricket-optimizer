import csv
import os
import re
from pathlib import Path
import cloudscraper
from datetime import datetime

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_SCHEDULE_PATH = DATA_DIR / "upcoming_matches.csv"

TEAMS_MAP = {
    'chennai-super-kings': 'Chennai Super Kings',
    'delhi-capitals': 'Delhi Capitals',
    'gujarat-titans': 'Gujarat Titans',
    'kolkata-knight-riders': 'Kolkata Knight Riders',
    'lucknow-super-giants': 'Lucknow Super Giants',
    'mumbai-indians': 'Mumbai Indians',
    'punjab-kings': 'Punjab Kings',
    'rajasthan-royals': 'Rajasthan Royals',
    'royal-challengers-bengaluru': 'Royal Challengers Bengaluru',
    'sunrisers-hyderabad': 'Sunrisers Hyderabad'
}

VENUES_MAP = {
    'Chennai Super Kings': 'M. A. Chidambaram Stadium, Chennai',
    'Delhi Capitals': 'Arun Jaitley Stadium, Delhi',
    'Gujarat Titans': 'Narendra Modi Stadium, Ahmedabad',
    'Kolkata Knight Riders': 'Eden Gardens, Kolkata',
    'Lucknow Super Giants': 'BRSABV Ekana Cricket Stadium, Lucknow',
    'Mumbai Indians': 'Wankhede Stadium, Mumbai',
    'Punjab Kings': 'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
    'Rajasthan Royals': 'Sawai Mansingh Stadium, Jaipur',
    'Royal Challengers Bengaluru': 'M. Chinnaswamy Stadium, Bengaluru',
    'Sunrisers Hyderabad': 'Rajiv Gandhi International Cricket Stadium, Hyderabad'
}

def fetch_sportskeeda_2026():
    print("Fetching REAL 2026 IPL schedule from Sportskeeda...")
    scraper = cloudscraper.create_scraper()
    
    html_sched = scraper.get('https://www.sportskeeda.com/go/ipl/schedule').text
    html_res = scraper.get('https://www.sportskeeda.com/go/ipl/results').text
    
    # regex matches: /live-cricket-score/team1-vs-team2-match-XX-DATE
    pattern = r'/live-cricket-score/([^/]+?)-vs-([^/]+?)-match-(\d+)-(\d{2}-[a-z]+-2026)'
    
    matches_sched = re.findall(pattern, html_sched)
    matches_res = re.findall(pattern, html_res)
    
    unique_matches = {}
    
    # Process completed
    for m in matches_res:
        t1, t2, m_num, date_str = m
        date_obj = datetime.strptime(date_str, "%d-%B-%Y")
        match_id = 1400000 + int(m_num)
        unique_matches[match_id] = {
            "match_id": match_id,
            "date": date_obj.strftime("%Y-%m-%d"),
            "team1": TEAMS_MAP.get(t1, t1.replace('-', ' ').title()),
            "team2": TEAMS_MAP.get(t2, t2.replace('-', ' ').title()),
            "venue": VENUES_MAP.get(TEAMS_MAP.get(t1, t1.replace('-', ' ').title()), "TBD"),
            "status": "completed",
            "match_num": int(m_num)
        }
    
    # Process upcoming
    for m in matches_sched:
        t1, t2, m_num, date_str = m
        match_id = 1400000 + int(m_num)
        if match_id not in unique_matches:
            date_obj = datetime.strptime(date_str, "%d-%B-%Y")
            unique_matches[match_id] = {
                "match_id": match_id,
                "date": date_obj.strftime("%Y-%m-%d"),
                "team1": TEAMS_MAP.get(t1, t1.replace('-', ' ').title()),
                "team2": TEAMS_MAP.get(t2, t2.replace('-', ' ').title()),
                "venue": VENUES_MAP.get(TEAMS_MAP.get(t1, t1.replace('-', ' ').title()), "TBD"),
                "status": "upcoming",
                "match_num": int(m_num)
            }
            
    final_list = list(unique_matches.values())
    final_list.sort(key=lambda x: x["match_num"])
    return final_list

def save_schedule(matches):
    if not matches:
        print("No matches to save.")
        return
        
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ["match_id", "date", "team1", "team2", "venue", "status", "match_num"]
    
    with open(OUTPUT_SCHEDULE_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)
    print(f"Successfully wrote {len(matches)} 2026 IPL matches to {OUTPUT_SCHEDULE_PATH}")

def main():
    print("--- Fantasy Optimizer: Live Schedule Updater ---")
    try:
        matches = fetch_sportskeeda_2026()
        save_schedule(matches)
    except Exception as e:
        print("Failed to scrape Sportskeeda:", e)

if __name__ == "__main__":
    main()
