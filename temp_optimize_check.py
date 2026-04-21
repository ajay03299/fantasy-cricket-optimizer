import os
import sys
sys.path.insert(0, os.getcwd())
from api.main import load_data, optimize_lineup, OptimizeRequest
load_data()
for mid in ['1400002', '1400068']:
    req = OptimizeRequest(match_id=mid, num_lineups=1)
    result = optimize_lineup(req)
    print('MATCH', mid, 'RESULT', result.get('error', 'OK'))
    if 'lineups' in result:
        lineup = result['lineups'][0]
        caps = [p for p in lineup if p.get('isCaptain') or p.get('isViceCaptain')]
        inactive = [p for p in lineup if not p.get('isPlayingCandidate', True)]
        print('C/VC', caps)
        print('INACTIVE', inactive)
        print('LEN', len(lineup))
        print([p['name'] for p in lineup])
        print()
