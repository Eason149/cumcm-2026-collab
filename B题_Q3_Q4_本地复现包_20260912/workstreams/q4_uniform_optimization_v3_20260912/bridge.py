"""Read-only imports of the two locked Q4 versions; local world only."""
from pathlib import Path
import sys,socket,urllib.request,json
sys.dont_write_bytecode=True
WORK=Path(__file__).resolve().parent
V2=WORK.parent/'q4_local_optimization_v2_20260912'
V1=WORK.parent/'q4_local_optimization_20260912'
def deny(*a,**k):raise RuntimeError('Q4 v3 is strictly local in memory')
socket.socket=deny;socket.create_connection=deny;urllib.request.OpenerDirector.open=deny
sys.path.insert(0,str(V2))
from base import *
import candidate as v2_strategy
import experiment as v2_experiment
WORK=Path(__file__).resolve().parent
V2=WORK.parent/'q4_local_optimization_v2_20260912'
V2_PARAMETERS=json.loads((V2/'results/selection_lock.json').read_text(encoding='utf-8'))['selected_parameters']
sys.path.insert(0,str(WORK))
