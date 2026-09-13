"""Read-only access to the locked first-round Q4 implementation."""
from pathlib import Path
import importlib.util,json,socket,sys,urllib.request
sys.dont_write_bytecode=True
WORK=Path(__file__).resolve().parent
V1=WORK.parent/'q4_local_optimization_20260912'
def deny(*args,**kwargs):raise RuntimeError('Q4 v2 is strictly offline')
socket.socket=deny;socket.create_connection=deny;urllib.request.OpenerDirector.open=deny
sys.path.insert(0,str(V1))
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,V1/file);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
v1=load('q4_v1_strategy_reference','strategy.py')
reference=load('q4_v1_experiments_reference','experiments.py')
from local_geometry import (ERROR,DirectionBelief,bearing_halfplanes,clip_polygon,exclude_disk,
    initial_polygon,minimum_circle,open_tour,optical_cover,stations_and_triangles)
from crossbar import probe_pair
from offline_world import World
V1_PARAMETERS=json.loads((V1/'results/selection_lock.json').read_text(encoding='utf-8'))['selected_parameters']
sys.path.insert(0,str(WORK))
