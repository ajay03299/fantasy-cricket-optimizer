import os
import sys
sys.path.insert(0, os.getcwd())
from api.main import load_data, get_squad_data
import pandas as pd
load_data()
up = pd.read_csv(os.path.join(os.getcwd(), 'data', 'upcoming_matches.csv'))
counts = []
for mid in up['match_id'].astype(str):
    squad = get_squad_data(mid)
    if squad.empty:
        counts.append((mid, 0, 0, 0))
        continue
    total = len(squad)
    active = int(squad['is_probably_playing'].sum()) if 'is_probably_playing' in squad else 0
    counts.append((mid, total, active, total-active))
print('min active', min(c[2] for c in counts), 'max active', max(c[2] for c in counts))
print([c for c in counts if c[2] < 11])
with open('temp_active_counts.txt','w', encoding='utf-8') as f:
    f.write(str(counts))
