"""Offline development, post-lock validation and deterministic reruns."""
from pathlib import Path
import argparse,datetime,hashlib,json,math,socket,sys,time,urllib.request
sys.dont_write_bytecode=True
import numpy as np
WORK=Path(__file__).resolve().parent

def deny(*args,**kwargs):raise RuntimeError('This Q4 entry is strictly offline')
socket.socket=deny;socket.create_connection=deny;urllib.request.OpenerDirector.open=deny
from offline_world import World
from strategy import solve
from local_geometry import stations_and_triangles

def write_json(path,data):
    def convert(value):
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,np.ndarray):return value.tolist()
        raise TypeError(type(value).__name__)
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=convert),encoding='utf-8')

def make_case(seed,kind):
    rng=np.random.default_rng(seed);count=int(rng.integers(10,17));channels=rng.choice(np.arange(1,21),size=count,replace=False).tolist()
    directions=int(rng.integers(1,count));types=['directional']*directions+['omni']*(count-directions);rng.shuffle(types)
    if kind in ('outward','tangent','boundary'):types=['directional']*(count-1)+['omni']
    sources=[]
    for j,(channel,typ) in enumerate(zip(channels,types)):
        a=float(rng.uniform(0,2*math.pi));r=1800*math.sqrt(float(rng.random()))
        if kind=='outward':r=float(rng.uniform(1550,1800))
        elif kind=='tangent':r=float(rng.uniform(200,1800))
        elif kind=='boundary':r=1800. if j%2==0 else 1799.999
        elif kind=='cluster':
            a=(j%3)*2*math.pi/3+float(rng.normal(0,.1));r=min(1800,max(0,float(rng.normal(1050,150))))
        elif kind=='threshold':
            r=[0.,4.999,5.,5.001,19.999,20.,20.001,999.999,1000.,1000.001,1499.999,1500.,1799.999,1800.][j%14]
        radius=float(rng.uniform(1000,1500))
        if kind in ('boundary','outward','threshold'):radius=[1000.,1000.00001,1250.,1500.][j%4]
        direction=float(rng.uniform(0,360))
        if kind in ('outward','boundary'):direction=math.degrees(a)%360
        elif kind in ('tangent','threshold'):direction=(math.degrees(a)+[90.,270.,90.+1e-7,270.-1e-7][j%4])%360
        sources.append({'channel':int(channel),'position':[r*math.cos(a),r*math.sin(a)],'radius':radius,'type':typ,'direction_deg':direction if typ=='directional' else None})
    return {'name':f'q4_{kind}_{seed}','seed':seed,'kind':kind,'error_mode':['zero','plus','minus','sine','hashed'][seed%5],'sources':sources}

def build_cases(start,n):
    kinds=['uniform','outward','tangent','cluster','boundary','threshold']
    return [make_case(start+i,kinds[i%6]) for i in range(n)]

def point_in_convex(point,poly,tol=3e-5):
    p=np.asarray(poly);x=np.asarray(point)
    if len(p)==1:return math.dist(x,p[0])<=tol
    if len(p)==2:
        v=p[1]-p[0];t=float((x-p[0])@v/max(v@v,1e-20));return -tol<=t<=1+tol and abs(np.cross(v,x-p[0]))<=tol*max(1.,np.linalg.norm(v))
    e=np.roll(p,-1,axis=0)-p;w=x-p
    return bool(np.all(e[:,0]*w[:,1]-e[:,1]*w[:,0]>=-tol*np.maximum(1,np.linalg.norm(e,axis=1))))

def inspect(case,result,ledger):
    checks={};sources={s['channel']:s for s in case['sources']};done=set();pos=(0.,0.);current=1;clock=0.;movement=0.;measure=0.;switches=0;failed=0;success=0
    for row in ledger:
        i=row['sequence'];path=row['path'];p=row['position'];c=row['channel'];reply=row['response'];increment=0.
        if path in ('/measure','/clear'):
            travel=math.hypot(p[0]-pos[0],p[1]-pos[1])/5;increment+=travel;movement+=travel;pos=p
            src=sources.get(c);d=math.inf if src is None or c in done else math.hypot(p[0]-src['position'][0],p[1]-src['position'][1])
            if path=='/measure':
                change=int(current!=c);current=c;switches+=change;measure+=5;increment+=5+change
                visible=d<=src['radius']+1e-10 if src is not None and c not in done else False
                if visible and src['type']=='directional' and d>1e-12:
                    angle=math.degrees(math.atan2(p[1]-src['position'][1],p[0]-src['position'][0]));gap=abs((angle-src['direction_deg']+180)%360-180)
                    visible=gap<=90+1e-9
                expected='no_signal' if not visible else 'near' if d<=5+1e-10 else 'direction'
                checks[f'action_{i}_radio']=expected==reply['measure_result']
                if expected=='direction' and reply['measure_result']=='direction':
                    bearing=math.degrees(math.atan2(src['position'][1]-p[1],src['position'][0]-p[0]));error=abs((reply['svd_deg']-bearing+180)%360-180)
                    checks[f'action_{i}_bearing_bound']=error<=1.005000001
            else:
                ok=d<=20+1e-10;checks[f'action_{i}_clear']=(reply['clear_result']=='success')==ok
                if ok:done.add(c);success+=5;increment+=5
                else:failed+=3;increment+=3
        clock+=increment;checks[f'action_{i}_clock']=abs(clock-row['virtual_after_s'])<=1e-7*max(1,clock)
    checks['all_sources_cleared']=done==set(sources)
    checks['output_count']=result['cleared_count']==len(sources)
    checks['end_exit']=ledger[-1]['path']=='/exit'
    checks['virtual_100h']=clock<360000
    checks['final_time']=abs(clock-result['virtual_time_s'])<1e-6
    for c,records in result['observations'].items():
        truth=sources.get(int(c))
        if truth is None:continue
        for obs in records:
            i=obs['sequence']
            if obs.get('posterior_vertices') is not None:
                checks[f'geometry_{i}_region']=point_in_convex(truth['position'],obs['posterior_vertices'])
                checks[f'geometry_{i}_circle']=math.dist(truth['position'],obs['circle_center'])<=obs['circle_radius_m']+3e-5
            if obs['within_min_reception_radius']:
                checks[f'geometry_{i}_guaranteed_distance']=max(math.dist(v,obs['position']) for v in obs['prior_vertices'])<=1000+1e-6
                if obs['response']['measure_result']=='no_signal':checks[f'geometry_{i}_directional_evidence']=truth['type']=='directional'
    for attempt in result['clear_attempts']:
        if attempt.get('posterior_vertices') is not None:
            checks[f"clear_geometry_{attempt['sequence']}"]=point_in_convex(sources[attempt['channel']]['position'],attempt['posterior_vertices'])
    for rec in result.get('crossbar_records',[]):
        k=rec['sequence_before'];cert=rec['certificate'];truth=sources[rec['channel']]['position']
        checks[f'crossbar_{k}_range']=max(math.dist(p,v) for p in rec['points'] for v in rec['prior_vertices'])<1000
        checks[f'crossbar_{k}_width']=cert['halfwidth_m']+1e-8>=cert['section_distance_m']*math.tan(math.radians(1.005))
        if 'posterior_vertices' in rec:checks[f'crossbar_{k}_posterior']=point_in_convex(truth,rec['posterior_vertices'])
    if not result['count_upper_bound_stop']:
        checks['channel_partition']=set(result['absent_channels'])|done==set(range(1,21)) and not(set(result['absent_channels'])&done)
        for c,indices in result['all_station_no_signal_channels'].items():checks[f'absence_{c}_all25']=indices==list(range(25))
    return {'passed':all(checks.values()),'checks':checks,'check_count':len(checks),'failures':[k for k,v in checks.items() if not v],
        'recomputed_time_s':clock,'components_s':{'movement':movement,'measurement':measure,'channel_switch':switches,'failed_optical':failed,'successful_optical':success}}

def run_comparison(cases,configs,name,trace_every=0):
    target=WORK/'results'/name;target.mkdir(parents=True,exist_ok=True);rows=[];checks=0;start=time.perf_counter()
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [WORK/'strategy.py',WORK/'local_geometry.py',WORK/'offline_world.py',WORK/'experiments.py',WORK/'crossbar.py']}
    snapshot=WORK/'snapshots'/name;snapshot.mkdir(parents=True,exist_ok=True)
    for key in hashes:(snapshot/key).write_bytes((WORK/key).read_bytes())
    for ci,case in enumerate(cases):
        for label,params in configs.items():
            world=World(case);world.client.act('/enter');t=time.perf_counter()
            try:
                result=solve(world.client,**params);world.client.act('/exit');runtime=time.perf_counter()-t
                audit=inspect(case,result,world.client.rows);checks+=audit['check_count']
                row={'case':case['name'],'kind':case['kind'],'strategy':label,'passed':audit['passed'],'cleared':len(world.cleared),
                    'source_count':len(case['sources']),'total_s':world.client.virtual,'per_source_s':world.client.virtual/len(case['sources']),
                    'actions':len(world.client.rows),'runtime_s':runtime,'optical_fallbacks':len(result['optical_fallbacks']),
                    'failed_optical_attempts':sum(x['response']['clear_result']!='success' for x in result['clear_attempts']),
                    'check_count':audit['check_count'],'failures':audit['failures'],'components_s':audit['components_s']}
                if not audit['passed'] or (trace_every and ci%trace_every==0):write_json(target/f"{case['name']}_{label}_trace.json",{'case':case,'parameters':params,'result':result,'ledger':world.client.rows,'audit':audit})
            except Exception as exc:
                row={'case':case['name'],'kind':case['kind'],'strategy':label,'passed':False,'error':str(exc),'cleared':len(world.cleared),'source_count':len(case['sources'])}
                write_json(target/f"{case['name']}_{label}_failure.json",{'case':case,'parameters':params,'row':row,'ledger':world.client.rows})
            rows.append(row)
        if (ci+1)%4==0:print(f'{name}: {ci+1}/{len(cases)} cases',flush=True)
    summary=[]
    for label in configs:
        group=[r for r in rows if r['strategy']==label];valid=[r for r in group if 'total_s' in r]
        s={'strategy':label,'cases':len(group),'passed_cases':sum(r['passed'] for r in group),'error_cases':sum('error' in r for r in group)}
        if valid:s.update(pooled_s=sum(r['total_s'] for r in valid)/sum(r['source_count'] for r in valid),mean_case_s=float(np.mean([r['per_source_s'] for r in valid])),median_s=float(np.median([r['per_source_s'] for r in valid])),p90_s=float(np.quantile([r['per_source_s'] for r in valid],.9)),max_s=max(r['per_source_s'] for r in valid),max_actions=max(r['actions'] for r in valid),mean_runtime_s=float(np.mean([r['runtime_s'] for r in valid])),max_runtime_s=max(r['runtime_s'] for r in valid),cleared_sources=sum(r['cleared'] for r in valid),source_count=sum(r['source_count'] for r in valid),failed_optical_attempts=sum(r['failed_optical_attempts'] for r in valid))
        summary.append(s)
    report={'name':name,'scope':'strictly local generated scenes, no simulator','configs':configs,'case_names':[c['name'] for c in cases],
        'rows':rows,'summary':summary,'passed':all(r['passed'] for r in rows),'check_count':checks,'source_sha256':hashes,'elapsed_s':time.perf_counter()-start}
    write_json(target/'comparison.json',report);print(json.dumps({'name':name,'summary':summary,'passed':report['passed'],'checks':checks},indent=2),flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='initial');ap.add_argument('--count',type=int,default=24);ap.add_argument('--seed',type=int,default=924000);args=ap.parse_args()
    cases=build_cases(args.seed,args.count);write_json(WORK/'data'/f'{args.name}_cases.json',cases)
    sites,triangles=stations_and_triangles();write_json(WORK/'data'/'coverage.json',{'stations':sites.tolist(),'triangles':triangles,'outer_apothem_m':1800.5,'maximum_triangle_edge_m':max(math.dist(sites[a],sites[b]) for t in triangles for a in t for b in t)})
    configs={'baseline_optical':{'route':'sequential','share':False,'directional':False,'trial_radius':0.,'radio_limit':3},
        'shared_joint':{'route':'joint','share':True,'directional':False},
        'orientation_joint':{'route':'joint','share':True,'directional':True},
        'crossbar_joint':{'route':'joint','share':True,'directional':False,'crossbar':True}}
    run_comparison(cases,configs,args.name,trace_every=6)
