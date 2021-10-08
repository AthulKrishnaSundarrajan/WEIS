#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTQPy_MESH_pts
Create the mesh for disctretizing the dynamic optimization problem

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""
import numpy as np

def DTQPy_MESH_pts(internal,dt):
    
    # extract 
    t0 = internal.t0
    tf = internal.tf
    
    # mesh
    mesh = dt.mesh
    
    w = []; d = [];
    
    if mesh.upper() == "ED":
        t = np.linspace(internal.t0,internal.tf,dt.nt)
        t = t[None].T
    
    elif mesh.upper() == "USER":
        
        if dt.t == []:
            raise Exception("ERROR: opts.dt.t for mesh type USER has not been specified")
        else:
            t = dt.t
            t = t[None].T
            
    
    # return 
    return t,w,d
    
    
    
    

