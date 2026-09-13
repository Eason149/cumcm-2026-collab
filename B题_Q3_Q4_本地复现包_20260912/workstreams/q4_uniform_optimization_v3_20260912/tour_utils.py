"""Location-only heuristics; no scene fields or source count are used."""
import math
import numpy as np

def area_centroid(poly):
    p=np.asarray(poly);q=np.roll(p,-1,axis=0);cross=p[:,0]*q[:,1]-p[:,1]*q[:,0];area2=float(cross.sum())
    if len(p)<3 or abs(area2)<1e-9:return p.mean(axis=0)
    return np.sum((p+q)*cross[:,None],axis=0)/(3*area2)

def insert_sources(position,points,station_order,source_indices):
    """Keep stations in their prescribed order and insert all source representatives."""
    path=list(station_order);left=set(source_indices)
    while left:
        choices=[]
        for j in sorted(left):
            for slot in range(len(path)+1):
                before=position if slot==0 else points[path[slot-1]]
                extra=math.dist(before,points[j])
                if slot<len(path):extra+=math.dist(points[j],points[path[slot]])-math.dist(before,points[path[slot]])
                choices.append((extra,j,slot))
        _,j,slot=min(choices);path.insert(slot,j);left.remove(j)
    return path
