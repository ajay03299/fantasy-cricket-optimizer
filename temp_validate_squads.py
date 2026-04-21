import os
import sys
sys.path.insert(0, os.getcwd())
from api.main import get_squad_data
mids = ['1400001','1400002','1400003','1400004','1400005','1400025','1400026','1400027']
for mid in mids:
    squad = get_squad_data(mid)
    print('MATCH', mid, 'LENGTH', len(squad), 'TEAMS', squad['team'].unique().tolist() if not squad.empty else [])
    if not squad.empty:
        print(squad[['player_name','team']].drop_duplicates().sort_values('team').to_string(index=False))
        print('---')
