"""Continuous all-direction coverage checks, not sampled-scene performance."""
import math,time
import numpy as np
from bridge_v4 import WORK,reference
from coverage import directional_cover,triangle_certificate

def layout(inner_count,outer_count,inner_radius,outer_radius,offset=0):
    return np.array([[0.,0.]]+[[inner_radius*math.cos(2*math.pi*k/inner_count+offset),inner_radius*math.sin(2*math.pi*k/inner_count+offset)] for k in range(inner_count)]+
        [[outer_radius*math.cos(2*math.pi*k/outer_count),outer_radius*math.sin(2*math.pi*k/outer_count)] for k in range(outer_count)])

def grid_layout(spacing,axis_extent):
    points=[[0.,0.]]
    for i in range(-2,3):
        for j in range(-2,3):
            if i==0 and j==0 or abs(i)==2 and abs(j)==2:continue
            x=i*spacing;y=j*spacing
            if abs(i)==2 and j==0:x=math.copysign(axis_extent,i)
            if abs(j)==2 and i==0:y=math.copysign(axis_extent,j)
            points.append([x,y])
    return np.asarray(points)

def run():
    trials=[(8,12,950,1900,0),(8,12,990,1900,0),(8,12,950,1950,0),(8,12,990,1950,0),
        (8,11,950,1900,0),(8,11,990,1950,0),(8,10,950,1950,0),(8,10,990,2000,0),
        (7,12,950,1900,0),(7,12,990,1900,0),(9,12,950,1900,0),(8,13,950,1900,0)]
    rows=[]
    for params in trials:
        start=time.perf_counter();p=layout(*params);cert=directional_cover(p,max_depth=14,keep_cells=True)
        row={'parameters':list(params),'stations':p.tolist(),'site_count':len(p),'certificate':cert,'runtime_s':time.perf_counter()-start};rows.append(row)
        print(params,len(p),{k:v for k,v in cert.items() if k not in ('cells',)},flush=True)
        reference.write_json(WORK/'geometry/hand_layouts.json',rows)

if __name__=='__main__':run()
