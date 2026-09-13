"""Read-only reference imports; all new artifacts stay in the v4 workstream."""
from pathlib import Path
import sys,socket,urllib.request,json
sys.dont_write_bytecode=True
import numba  # Load before the read-only v2 module named coverage reaches sys.path.
WORK=Path(__file__).resolve().parent
V3=WORK.parent/'q4_uniform_optimization_v3_20260912'
def deny(*a,**k):raise RuntimeError('Q4 v4 is local in memory only')
socket.socket=deny;socket.create_connection=deny;urllib.request.OpenerDirector.open=deny
sys.path.insert(0,str(V3))
from bridge import *
import strategy_v3 as v3_strategy
import experiments_v3 as v3_experiments
WORK=Path(__file__).resolve().parent
V3=WORK.parent/'q4_uniform_optimization_v3_20260912'
V3_PARAMETERS=json.loads((V3/'results/selection_lock.json').read_text(encoding='utf-8'))['selected_parameters']
sys.path.insert(0,str(WORK))
