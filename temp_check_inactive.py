import os
import sys
sys.path.insert(0, os.getcwd())
from api.main import load_data, get_squad_data
load_data()
for mid in ['1400068']:
    squad = get_squad_data(mid)
    print('MATCH', mid, 'LEN', len(squad))
    print(squad[squad['player_name'].isin(['Vipraj Nigam','C Bosch','Nitish Rana'])][['player_name','team','is_probably_playing']].to_dict('records'))
    print()
