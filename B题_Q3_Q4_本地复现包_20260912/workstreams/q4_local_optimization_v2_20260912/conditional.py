"""Conditional-range transverse probes from publicly received bearings."""
import math
import numpy as np
from base import ERROR

def reception_certificate(poly,point,reference,theta):
    p=np.asarray(poly);q=np.asarray(point);s=np.asarray(reference)
    maximum=float(np.max(np.linalg.norm(p-q,axis=1)))
    if maximum<=1000-1e-5:return {'kind':'minimum_radius','maximum_vertex_distance_m':maximum,'residual':maximum-1000}
    difference=float(np.max((q@q-s@s)-2*p@(q-s)))
    if difference<=-1e-5:return {'kind':'closer_than_received_point','maximum_squared_distance_difference':difference,'maximum_vertex_distance_m':maximum}
    a=math.radians(theta);u=np.array([math.cos(a),math.sin(a)]);v=np.array([-u[1],u[0]])
    forward=float((q-s)@u);side=abs(float((q-s)@v));delta=math.radians(ERROR)
    one=(forward-5*math.cos(delta))**2+(side+5*math.sin(delta))**2-1000**2
    two=(forward-1000*math.cos(delta))**2+(side+1000*math.sin(delta))**2-1000**2
    if max(one,two)<=-1e-4:return {'kind':'full_bearing_conditional_range','two_disk_residuals_m2':[one,two],'maximum_vertex_distance_m':maximum}
    return None

def conditional_pair(poly,reference_position,reference_bearing,position,offset=.03,fraction=.25):
    p=np.asarray(poly);s=np.asarray(reference_position);a=math.radians(reference_bearing)
    u=np.array([math.cos(a),math.sin(a)]);v=np.array([-u[1],u[0]])
    projection=(p-s)@u;lower=float(projection.min());upper=float(projection.max())
    attempts=[]
    for f in (fraction,.5):
        d=lower+f*(upper-lower)
        for b in (max(d*math.tan(math.radians(ERROR))+1e-5,offset*(upper-lower)),d*math.tan(math.radians(ERROR))+1e-5):
            points=[s+d*u+sign*b*v for sign in (-1,1)]
            certs=[reception_certificate(p,q,s,reference_bearing) for q in points]
            if not all(certs):continue
            order=sorted(range(2),key=lambda i:math.dist(position,points[i]));points=[points[i] for i in order];certs=[certs[i] for i in order]
            return points,{'reference_position':s.tolist(),'reference_bearing_deg':reference_bearing,'error_bound_deg':ERROR,
                'unit_forward':u.tolist(),'unit_transverse':v.tolist(),'section_distance_m':d,'halfwidth_m':b,
                'range_certificates':certs,'cut_normal':u.tolist(),'cut_rhs':float(u@s+d)+1e-7,
                'fraction':f,'projection_lower_m':lower,'projection_upper_m':upper}
    raise RuntimeError('No certified conditional-range crossbar')
