import os
import sys
sys.path.insert(0, os.getcwd())
from api.main import load_data, get_squad_data
import pandas as pd

load_data()
upcoming = pd.read_csv(os.path.join(os.getcwd(), 'data', 'upcoming_matches.csv'))
misses = []
for mid in upcoming['match_id'].astype(str).tolist():
    squad = get_squad_data(mid)
    if len(squad) < 40:
        misses.append((mid, len(squad), sorted(set(squad['team'].astype(str).tolist()))))
with open('temp_squad_check.txt', 'w', encoding='utf-8') as f:
    f.write(str(misses))
