"""Q4 v2 offline paired comparison with immutable v1 as reference."""
import hashlib,json,math,time
import numpy as np
from base import WORK,V1,V1_PARAMETERS,World,v1,reference
from candidate import solve
from coverage import triangle_certificate

def write(path,data):return reference.write_json(path,data)
def development_cases():
    return json.loads((V1/'data/development72.json').read_text())+json.loads((V1/'data/fresh_928000_928143.json').read_text())

def audit(case,result,ledger):
    base_result=dict(result);base_result['crossbar_records']=[]
    r=reference.inspect(case,base_result,ledger);checks={k:v for k,v in r['checks'].items() if not(k.startswith('absence_') and k.endswith('_all25'))}
    truth={s['channel']:s for s in case['sources']}
    for rec in result.get('crossbar_records',[]):
        k=rec['sequence_before'];cert=rec['certificate'];source=truth[rec['channel']];poly=np.asarray(rec['prior_vertices']);s=np.asarray(cert['reference_position'])
        certificates=cert.get('range_certificates')
        checks[f'crossbar_{k}_span']=cert['halfwidth_m']>=cert['section_distance_m']*math.tan(math.radians(1.005))-1e-8
        for j,p in enumerate(rec['points']):
            p=np.asarray(p)
            checks[f'crossbar_{k}_{j}_actual_reception_distance']=math.dist(p,source['position'])<=source['radius']+1e-7
            if certificates is None:
                valid=max(math.dist(p,g) for g in poly)<=1000+1e-7
            else:
                c=certificates[j];kind=c['kind']
                if kind=='minimum_radius':valid=max(math.dist(p,g) for g in poly)<=1000+1e-7
                elif kind=='closer_than_received_point':valid=max(math.dist(p,g)**2-math.dist(s,g)**2 for g in poly)<=1e-5
                else:
                    ang=math.radians(cert['reference_bearing_deg']);u=np.array([math.cos(ang),math.sin(ang)]);n=np.array([-u[1],u[0]]);a=float((p-s)@u);b=abs(float((p-s)@n));d=math.radians(1.005)
                    valid=max(math.hypot(a-r*math.cos(d),b+r*math.sin(d)) for r in (5,1000))<=1000+1e-7
            checks[f'crossbar_{k}_{j}_certificate']=bool(valid)
        if rec['outcomes']==['no_signal','no_signal']:
            checks[f'crossbar_{k}_cut_truth']=np.dot(cert['cut_normal'],source['position'])<=cert['cut_rhs']+1e-6
            checks[f'crossbar_{k}_posterior']=reference.point_in_convex(source['position'],rec['posterior_vertices'])
    required=result.get('required_station_indices',list(range(25)))
    coverage=triangle_certificate(np.asarray(result['stations'])[required],1000-1e-5)
    checks['final_directional_cover']=coverage['passed']
    if not result['count_upper_bound_stop']:
        for c,idx in result['all_station_no_signal_channels'].items():checks['absence_'+str(c)+'_required_sites']=set(idx)==set(required)
    for ri,rec in enumerate(result.get('negative_region_records',[])):
        source=truth[rec['channel']]
        checks[f'negative_region_{ri}_truth_preserved']=reference.point_in_convex(source['position'],rec['posterior_vertices'])
        for j,region in enumerate(rec['certified_regions']):
            t=region['triangle'];p=region['region']
            checks[f'negative_region_{ri}_{j}_contained_in_triangle']=all(reference.point_in_convex(v,t) for v in p)
            checks[f'negative_region_{ri}_{j}_range']=max(math.dist(v,s) for v in p for s in t)<=1000+1e-7
            checks[f'negative_region_{ri}_{j}_negative_points']=all(any(math.dist(v,s)<1e-7 for s in rec['negative_points']) for v in t)
    r.update(checks=checks,passed=all(checks.values()),check_count=len(checks),failures=[k for k,v in checks.items() if not v],coverage=coverage)
    return r

def compare(cases,configs,name,trace_every=0):
    folder=WORK/'results'/name;folder.mkdir(parents=True,exist_ok=True);rows=[];started=time.perf_counter();check_count=0
    files=['candidate.py','conditional.py','coverage.py','negative_regions.py','base.py','experiment.py']
    hashes={f:hashlib.sha256((WORK/f).read_bytes()).hexdigest() for f in files}
    snapshot=WORK/'snapshots'/name;snapshot.mkdir(parents=True,exist_ok=True)
    for f in files:(snapshot/f).write_bytes((WORK/f).read_bytes())
    for ci,case in enumerate(cases):
        for label,params in configs.items():
            world=World(case);client=world.client;client.act('/enter');t=time.perf_counter()
            try:
                result=v1.solve(client,**V1_PARAMETERS) if params is None else solve(client,**params)
                client.act('/exit');runtime=time.perf_counter()-t;check=audit(case,result,client.rows);check_count+=check['check_count']
                row={'case':case['name'],'kind':case['kind'],'strategy':label,'source_count':len(case['sources']),'cleared':len(world.cleared),
                    'passed':check['passed'],'total_s':client.virtual,'per_source_s':client.virtual/len(case['sources']),'actions':len(client.rows),
                    'runtime_s':runtime,'components_s':check['components_s'],'checks':check['check_count'],'failures':check['failures'],
                    'coverage_replacements':len(result.get('coverage_changes',[])),'required_stations':len(result.get('required_station_indices',list(range(25))))}
                if not check['passed'] or trace_every and ci%trace_every==0:write(folder/f"{case['name']}_{label}_trace.json",{'case':case,'parameters':params,'result':result,'ledger':client.rows,'audit':check})
            except Exception as exc:
                row={'case':case['name'],'kind':case['kind'],'strategy':label,'passed':False,'error':repr(exc),'cleared':len(world.cleared),'source_count':len(case['sources'])}
                write(folder/f"{case['name']}_{label}_failure.json",{'case':case,'parameters':params,'row':row,'ledger':client.rows})
            rows.append(row)
        if (ci+1)%6==0:print(name,ci+1,'/',len(cases),flush=True)
    summaries=[]
    for label in configs:
        group=[r for r in rows if r['strategy']==label];valid=[r for r in group if 'total_s' in r]
        s={'strategy':label,'cases':len(group),'passed_cases':sum(r['passed'] for r in group),'error_cases':sum('error' in r for r in group)}
        if valid:s.update(pooled_s=sum(r['total_s'] for r in valid)/sum(r['source_count'] for r in valid),mean_case_s=float(np.mean([r['per_source_s'] for r in valid])),median_s=float(np.median([r['per_source_s'] for r in valid])),p90_s=float(np.quantile([r['per_source_s'] for r in valid],.9)),max_s=max(r['per_source_s'] for r in valid),max_actions=max(r['actions'] for r in valid),cleared_sources=sum(r['cleared'] for r in valid),source_count=sum(r['source_count'] for r in valid),mean_runtime_s=float(np.mean([r['runtime_s'] for r in valid])),replacements=sum(r['coverage_replacements'] for r in valid),retired_sites=sum(25-r['required_stations'] for r in valid))
        summaries.append(s)
    report={'scope':'local generated scenes only','name':name,'configs':configs,'rows':rows,'summary':summaries,'check_count':check_count,
        'passed':all(r['passed'] for r in rows),'source_sha256':hashes,'elapsed_s':time.perf_counter()-started}
    write(folder/'comparison.json',report);print(json.dumps({'summary':summaries,'passed':report['passed'],'checks':check_count},indent=2),flush=True);return report

if __name__=='__main__':
    all_cases=development_cases();cases=all_cases[:12]+all_cases[72:96]
    write(WORK/'data/development216.json',all_cases);write(WORK/'data/initial36.json',cases)
    common=dict(V1_PARAMETERS)
    configurations={'v1':None,'quarter':dict(common,conditional_range=True,crossbar_fraction=.25),'near10':dict(common,conditional_range=True,crossbar_fraction=.1),'near35':dict(common,conditional_range=True,crossbar_fraction=.35),
        'quarter_near_estimate':dict(common,conditional_range=True,crossbar_fraction=.25,route_estimate=.25),
        'quarter_outer_first':dict(common,conditional_range=True,crossbar_fraction=.25,route_mode='outer_first'),
        'midpoint_outer_first':dict(common,route_mode='outer_first'),
        'quarter_sweep':dict(common,conditional_range=True,crossbar_fraction=.25,route_mode='sweep'),
        'quarter_nearest':dict(common,conditional_range=True,crossbar_fraction=.25,route_mode='nearest'),
        'quarter_opportunity':dict(common,conditional_range=True,crossbar_fraction=.25,opportunistic=True),
        'quarter_drop':dict(common,conditional_range=True,crossbar_fraction=.25,opportunistic=True,drop_sites=True),
        'quarter_selective_share':dict(common,conditional_range=True,crossbar_fraction=.25,share_ratio=.4)}
    compare(cases,configurations,'structural_round1',trace_every=12)
