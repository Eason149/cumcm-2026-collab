"""Certified station layouts, isolated from any case truths or outcomes."""
from functools import lru_cache
from scipy.spatial import Delaunay
from geometry_hand_layouts import layout,grid_layout
from cover_union import certificate

@lru_cache(maxsize=256)
def certified_layout(parameters):
    sites=grid_layout(*parameters[1:]) if parameters[0]=='grid' else layout(*parameters)
    cert=certificate(sites,keep_regions=False)
    if not cert['passed']:raise RuntimeError('Uncertified directional search layout')
    return sites,Delaunay(sites).simplices.tolist(),cert

@lru_cache(maxsize=256)
def certify_points(key):return certificate(key,keep_regions=False,receiver_radius=999.,target_expansion=.02)
