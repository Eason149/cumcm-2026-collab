"""Q4 local prototypes. Decisions use public responses and known problem bounds only."""
import math
import numpy as np
from local_geometry import (ERROR,DirectionBelief,bearing_halfplanes,clip_polygon,exclude_disk,
    initial_polygon,minimum_circle,open_tour,optical_cover,stations_and_triangles)
from crossbar import probe_pair

def solve(client,route='sequential',share=False,directional=False,angle_bin=4.,trial_radius=40.,
          probe_offset=.25,radio_limit=8,search_first=False,source_bias=0.,crossbar=False,crossbar_offset=.2,crossbar_after_dark=False,
          station_layout='8_16',station_inner=990.):
    sites,triangles=stations_and_triangles(station_inner,layout=station_layout);unused=set(range(1,len(sites)))
    unknown=set(range(1,21));known={};cleared=[];absent=set();observations={c:[] for c in range(1,21)}
    negatives={c:set() for c in range(1,21)};visits=[];attempts=[];fallbacks=[];crossbar_records=[]
    def update(d,p):
        if not len(p):raise RuntimeError('Empty position region')
        d['polygon']=p;circ=minimum_circle(p);d['center']=circ['center'];d['radius']=circ['radius']
    def observe(c,pos,kind,site=None):
        pos=np.asarray(pos,dtype=float);d=known.get(c);prior=None if d is None else d['polygon']
        guaranteed=prior is not None and float(np.max(np.linalg.norm(prior-pos,axis=1)))<=1000-1e-6
        reply=client.act('/measure',pos,c)
        rec={'sequence':len(client.rows),'position':pos.tolist(),'kind':kind,'response':reply.copy(),
             'within_min_reception_radius':bool(guaranteed),'prior_vertices':None if prior is None else prior.tolist()}
        observations[c].append(rec)
        if reply['measure_result']=='no_signal':
            if site is not None:negatives[c].add(site)
            if d is not None:
                d['last_negative']=True
                if guaranteed and directional:
                    d['negative_points'].append(pos.tolist())
                    if d['belief'] is None:d['belief']=DirectionBelief(d['polygon'],d['positive_points'],d['negative_points'],angle_bin)
                    else:d['belief'].update_sign(pos,False)
                    update(d,d['belief'].polygon())
            elif len(negatives[c])==len(sites):absent.add(c);unknown.remove(c)
        else:
            if d is None:
                d={'polygon':None,'center':pos,'radius':5.,'positive_points':[],'negative_points':[],'belief':None,
                   'radios':0,'last_negative':False,'last_measure':pos,'trial_count':0,'stalled':0};known[c]=d;unknown.remove(c)
            d['positive_points'].append(pos.tolist());d['last_negative']=False
            if reply['measure_result']=='near':
                d['polygon']=None;d['center']=pos;d['radius']=5.
            else:
                A,b=bearing_halfplanes(pos,reply['svd_deg'],ERROR)
                p=initial_polygon(pos,reply['svd_deg']) if d['polygon'] is None else clip_polygon(d['polygon'],A,b)
                if d['belief'] is not None:
                    d['belief'].restrict(p);d['belief'].update_sign(pos,True);p=d['belief'].polygon()
                update(d,p)
            d['last_measure']=pos
        if c in known:
            d=known[c];d['last_measure']=pos;rec.update(posterior_vertices=None if d['polygon'] is None else d['polygon'].tolist(),
                circle_center=d['center'].tolist(),circle_radius_m=d['radius'],orientation_bins=0 if d['belief'] is None else len(d['belief'].parts))
        return reply
    def sharing():
        if not share:return
        pos=np.asarray(client.position)
        for c in sorted(known):
            d=known[c]
            if d['polygon'] is None or d['radius']<=20 or math.dist(pos,d['last_measure'])<120 or math.dist(pos,d['center'])>1500:continue
            v=d['center']-pos
            if np.linalg.norm(v)<1e-7:continue
            A,b=bearing_halfplanes(pos,math.degrees(math.atan2(v[1],v[0])),ERROR);p=clip_polygon(d['polygon'],A,b)
            if len(p) and minimum_circle(p)['radius']<.7*d['radius']:observe(c,pos,'shared_bearing')
    def scan(i):
        visits.append(i)
        for c in sorted(unknown):observe(c,sites[i],'search',i)
        sharing()
    def clear_at(c,point,kind,certificate=None):
        d=known[c];prior=d['polygon'];reply=client.act('/clear',point,c)
        bound=5. if prior is None else float(np.max(np.linalg.norm(prior-point,axis=1)))
        rec={'sequence':len(client.rows),'channel':c,'position':list(point),'kind':kind,'prior_bound_m':bound,
             'prior_vertices':None if prior is None else prior.tolist(),'response':reply.copy(),'certificate':certificate}
        attempts.append(rec)
        if reply['clear_result']=='success':
            cleared.append({'channel':c,'point':list(point),'bound_source':'prior_geometry' if bound<=20 else 'successful_clear_response',
                'prior_bound_m':bound,'virtual_after_s':client.virtual});known.pop(c);return True
        if bound<=20-1e-6:raise RuntimeError('Certified optical clear failed')
        p=exclude_disk(prior,point)
        if d['belief'] is not None:
            d['belief'].restrict(p);p=d['belief'].polygon()
        update(d,p);rec['posterior_vertices']=p.tolist();return False
    def optical_finish(c):
        d=known[c];poly=d['polygon'].copy();points,cert=optical_cover(poly,client.position)
        fallbacks.append({'channel':c,'sequence_before':len(client.rows),'prior_vertices':poly.tolist(),'certificate':cert,'point_count':len(points)})
        for p in points:
            if clear_at(c,p,'grid_fallback',cert):return
        raise RuntimeError('Exhausted proven optical cover')
    def local_step(c):
        d=known[c]
        if d['radius']<=20-1e-6:
            clear_at(c,d['center'],'certified');sharing();return
        if trial_radius and d['radius']<=trial_radius and d['trial_count']<12:
            p=d['polygon'];vertex=min(p,key=lambda x:math.dist(client.position,x));v=d['center']-vertex
            target=vertex+v*min(1.,18/max(np.linalg.norm(v),1e-12));d['trial_count']+=1
            clear_at(c,target,'early_optical');sharing();return
        if d['radios']>=radio_limit or (not directional and not crossbar and d['last_negative']) or d['stalled']>=3:
            optical_finish(c);sharing();return
        prior=d['radius'];poly=d['polygon'];target=d['center'].copy()
        if crossbar and (not crossbar_after_dark or d['last_negative']):
            ref=next(o for o in observations[c] if o['response']['measure_result']=='direction')
            points,cert=probe_pair(poly,ref['position'],ref['response']['svd_deg'],client.position,crossbar_offset)
            rec={'channel':c,'sequence_before':len(client.rows),'prior_vertices':poly.tolist(),'points':[p.tolist() for p in points],'certificate':cert,'outcomes':[]}
            for p in points:
                reply=observe(c,p,'crossbar');d['radios']+=1;rec['outcomes'].append(reply['measure_result'])
                if reply['measure_result']!='no_signal':break
            if rec['outcomes']==['no_signal','no_signal']:
                p=clip_polygon(d['polygon'],[cert['cut_normal']],[cert['cut_rhs']])
                if d['belief'] is not None:d['belief'].restrict(p);p=d['belief'].polygon()
                update(d,p);rec['posterior_vertices']=p.tolist()
            crossbar_records.append(rec)
            d['stalled']=d['stalled']+1 if d['radius']>.95*prior else 0
            return
        if d['last_negative'] and directional:
            delta=poly[:,None,:]-poly[None,:,:];i,j=np.unravel_index(np.argmax(np.sum(delta**2,axis=2)),delta.shape[:2]);u=poly[i]-poly[j];u/=max(np.linalg.norm(u),1e-12);v=np.array([-u[1],u[0]])
            if np.dot(u,np.asarray(d['positive_points'][-1])-target)<0:u=-u
            target=target+.35*prior*u+((-1)**d['radios'])*probe_offset*prior*v
        observe(c,target,'localization');d['radios']+=1
        d['stalled']=d['stalled']+1 if d['radius']>.95*prior else 0
    scan(0)
    while unknown or known:
        if len(client.rows)>3500:raise RuntimeError('Local development action guard exceeded')
        if len(cleared)==16:break
        ready=[c for c,d in known.items() if d['radius']<=20 and math.dist(client.position,d['center'])<=50]
        if ready:local_step(min(ready,key=lambda c:math.dist(client.position,known[c]['center'])));continue
        active_sites=sorted(unused) if unknown else []
        if not known or (search_first and active_sites):
            if not active_sites:raise RuntimeError('No action resolves remaining channels')
            order=open_tour(client.position,sites[active_sites])
            i=active_sites[order[0]] if route=='joint' else min(active_sites,key=lambda i:math.dist(client.position,sites[i]))
            unused.remove(i);scan(i);continue
        if route=='sequential':
            c=min(known,key=lambda c:math.dist(client.position,known[c]['center']));local_step(c)
        else:
            labels=[('source',c) for c in sorted(known)]+[('site',i) for i in active_sites]
            points=[known[c]['center'] for c in sorted(known)]+[sites[i] for i in active_sites]
            order=open_tour(client.position,points);kind,i=labels[order[0]]
            if source_bias:
                near=min(known,key=lambda c:math.dist(client.position,known[c]['center']))
                if math.dist(client.position,known[near]['center'])<=source_bias:kind,i='source',near
            if kind=='site':unused.remove(i);scan(i)
            else:local_step(i)
    if not 10<=len(cleared)<=16:raise RuntimeError('Q4 cleared count outside source-count bounds')
    return {'strategy':'q4_local_prototype','cleared_sources':cleared,'cleared_count':len(cleared),'absent_channels':sorted(absent),
        'count_upper_bound_stop':len(cleared)==16,'remaining_unknown_channels':sorted(unknown),'observations':observations,
        'scan_visits':visits,'stations':sites.tolist(),'coverage_triangles':triangles,'clear_attempts':attempts,'optical_fallbacks':fallbacks,
        'all_station_no_signal_channels':{c:sorted(negatives[c]) for c in absent},'crossbar_records':crossbar_records,'virtual_time_s':client.virtual}
