import os
import sys
sys.path.insert(0, os.getcwd())
from api.main import load_data, get_squad_data
import pandas as pd
load_data()
up = pd.read_csv(os.path.join(os.getcwd(),'data','upcoming_matches.csv'))
players = ['Vipraj Nigam','C Bosch','Nitish Rana']
for mid in up['match_id'].astype(str):
    squad = get_squad_data(mid)
    found = squad[squad['player_name'].isin(players)]
    if not found.empty:
        print(mid, found[['player_name','team','is_probably_playing']].to_dict('records'))
