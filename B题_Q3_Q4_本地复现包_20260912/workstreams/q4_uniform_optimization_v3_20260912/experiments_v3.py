"""Paired development/validation for unknown-type directional sources."""
import hashlib,json,time
import numpy as np
from bridge import WORK,V1,V2,V1_PARAMETERS,V2_PARAMETERS,World,v1,v2_strategy,v2_experiment,reference
from strategy_v3 import solve

def write(path,obj):return reference.write_json(path,obj)

def audit(case,result,ledger):
    a=v2_experiment.audit(case,result,ledger);checks=a['checks']
    for j,event in enumerate(result.get('discovery_upper_bound_stops',[])):
        prior=ledger[:event['sequence']]
        public={r['channel'] for r in prior if r['path']=='/measure' and r['response']['measure_result'] in ('near','direction')}
        done={r['channel'] for r in prior if r['path']=='/clear' and r['response']['clear_result']=='success'}
        checks[f'discovery_stop_{j}_16_distinct_public_channels']=len(public)==16 and public==set(event['cleared_channels'])|set(event['known_channels'])
        checks[f'discovery_stop_{j}_cleared_confirmed']=done==set(event['cleared_channels'])
        checks[f'discovery_stop_{j}_disjoint_unknown']=not(public&set(event['stopped_unknown_channels']))
        checks[f'discovery_stop_{j}_no_later_unknown_measurement']=all(r['path']!='/measure' or r['channel'] not in event['stopped_unknown_channels'] for r in ledger[event['sequence']:])
    a.update(passed=all(checks.values()),check_count=len(checks),failures=[k for k,v in checks.items() if not v]);return a

def all_development():
    return (json.loads((V1/'data/development72.json').read_text())+json.loads((V1/'data/fresh_928000_928143.json').read_text())+
        json.loads((V2/'data/fresh_941000_941167.json').read_text()))

def compare(cases,configs,name,trace_every=0):
    out=WORK/'results'/name;out.mkdir(parents=True,exist_ok=True);snap=WORK/'snapshots'/name;snap.mkdir(parents=True,exist_ok=True)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in WORK.glob('*.py')}
    for f in hashes:(snap/f).write_bytes((WORK/f).read_bytes())
    rows=[];check_count=0;t0=time.perf_counter()
    for i,case in enumerate(cases):
        for label,params in configs.items():
            world=World(case);c=world.client;c.act('/enter');start=time.perf_counter()
            try:
                result=v1.solve(c,**V1_PARAMETERS) if params=='v1' else v2_strategy.solve(c,**V2_PARAMETERS) if params=='v2' else solve(c,**params)
                c.act('/exit');runtime=time.perf_counter()-start;a=audit(case,result,c.rows);check_count+=a['check_count']
                row={'case':case['name'],'kind':case['kind'],'strategy':label,'source_count':len(case['sources']),'cleared':len(world.cleared),'passed':a['passed'],
                    'total_s':c.virtual,'per_source_s':c.virtual/len(case['sources']),'actions':len(c.rows),'runtime_s':runtime,'components_s':a['components_s'],
                    'checks':a['check_count'],'failures':a['failures'],'discovery_upper_bound_stops':len(result.get('discovery_upper_bound_stops',[]))}
                if not a['passed'] or trace_every and i%trace_every==0:write(out/f"{case['name']}_{label}_trace.json",{'case':case,'parameters':params,'result':result,'ledger':c.rows,'audit':a})
            except Exception as e:
                row={'case':case['name'],'kind':case['kind'],'strategy':label,'source_count':len(case['sources']),'cleared':len(world.cleared),'passed':False,'error':repr(e)}
                write(out/f"{case['name']}_{label}_failure.json",{'case':case,'parameters':params,'row':row,'ledger':c.rows})
            rows.append(row)
        if (i+1)%8==0:print(name,i+1,'/',len(cases),flush=True)
    summary=[]
    for label in configs:
        g=[r for r in rows if r['strategy']==label];valid=[r for r in g if 'total_s' in r]
        s={'strategy':label,'cases':len(g),'passed_cases':sum(r['passed'] for r in g),'errors':sum('error' in r for r in g)}
        if valid:s.update(pooled_s=sum(r['total_s'] for r in valid)/sum(r['source_count'] for r in valid),mean_case_s=float(np.mean([r['per_source_s'] for r in valid])),max_actions=max(r['actions'] for r in valid),max_s=max(r['per_source_s'] for r in valid),cleared_sources=sum(r['cleared'] for r in valid),source_count=sum(r['source_count'] for r in valid),max_runtime_s=max(r['runtime_s'] for r in valid))
        summary.append(s)
    report={'name':name,'scope':'in-memory local generated scenes','configs':configs,'rows':rows,'summary':summary,'passed':all(r['passed'] for r in rows),'check_count':check_count,'source_sha256':hashes,'elapsed_s':time.perf_counter()-t0}
    write(out/'comparison.json',report);print(json.dumps({'summary':summary,'passed':report['passed'],'checks':check_count},indent=2),flush=True);return report

if __name__=='__main__':
    all_cases=all_development();write(WORK/'data/development384.json',all_cases)
    cases=[c for c in all_cases if c['kind']=='uniform'];write(WORK/'data/development_uniform64.json',cases)
    p=dict(V2_PARAMETERS,stop_when_found16=True,scan_current_first=True)
    configs={'v1':'v1','v2':'v2','v1_found16':dict(V1_PARAMETERS,stop_when_found16=True,scan_current_first=True),'v2_found16':p,
        'action_quarter':dict(p,crossbar_fraction=.25),'action_midpoint':dict(p,crossbar_fraction=.5),
        'action_near35':dict(p,crossbar_fraction=.35),'first_quarter':dict(p,crossbar_fraction=.25,crossbar_reference='first'),
        'centre_quarter':dict(p,crossbar_fraction=.25,route_estimate=.5),'near_optical':dict(p,crossbar_fraction=.25,near_clear_distance=200,near_clear_radius=40)}
    compare(cases,configs,'uniform_round1',trace_every=11)
