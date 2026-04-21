import os
import sys
sys.path.insert(0, os.getcwd())
from api.main import load_data, get_squad_data, match_data_for_season_realistic_optimizer
load_data()
for mid in ['1400002','1400068']:
    full = get_squad_data(mid)
    narrow = match_data_for_season_realistic_optimizer(full, mid)
    print(mid, 'full', len(full), 'narrow', len(narrow), 'active', int(full['is_probably_playing'].sum()))
    if len(full) != len(narrow):
        print('excluded', set(full['id']) - set(narrow['id']))
