#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTQPy_OLOC_mesh
Create the mesh for disctretizing the dynamic optimization problem specific 
to DLCs in the transition region

Contributor: Athul Krishna Sundarrajan (AthulKrishnaSundarrajan on Github)
Contributor: Daniel Zalkind (dzalkind on Github)
Primary Contributor: Daniel R. Herber (danielrherber on Github)
"""

import numpy as np 
from dtqpy.src.classes.DTQPy_CLASS_OPTS import *
from scipy.interpolate import interp1d,PchipInterpolator
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter

def DTQPy_OLOC_mesh(tt,Wind_speed,dtqp_options,t0,tf):
    
    # initiate class
    opts = options()
    
    # extract dtqp options and assign
    opts.dt.nt = dtqp_options['nt']
    opts.solver.tolerence = dtqp_options['tolerance']
    opts.solver.maxiters = dtqp_options['maxiters']
    opts.solver.function = 'pyoptsparse'
    
    nt = opts.dt.nt
    
    # get the mean wind speed of the distribution
    Wind_mean = np.mean(Wind_speed)
    
    # check if mean wind speed is in the rated region 
    mean_flag = (Wind_mean > 12.5)
    
    # check if there are wind speeds in the transition region 
    transition_flag = Wind_speed < 12.5
    
    mesh_flag = mean_flag and transition_flag.any()
    
    if mesh_flag:
        t = np.linspace(t0,tf,nt)
        
        t_index = np.arange(0,nt)
        
        WS_pp = PchipInterpolator(np.squeeze(tt),np.squeeze(Wind_speed))
        
        WS = WS_pp(t)
        
        transition_index = WS < 12.5
        
        t_transition = t[transition_index]
        
        t_index_transition = t_index[transition_index]
        
        n_transition = len(t_transition)
        
        ranges =[]

        for k,g in groupby(enumerate(t_index_transition),lambda x:x[0]-x[1]):
            group = (map(itemgetter(1),g))
            group = list(map(int,group))
            ranges.append((group[0],group[-1]))
            
        ranges = np.array(ranges)
        
        ind0 = ranges[0,0]
        
        t_new = np.array([])
        
        t_new = np.append(t_new,t[0:ind0])
        
        range_len = ranges[:,1] - ranges[:,0]
        
        for i in range(len(range_len)):
            if range_len[i] == 1 or range_len[i] == 0:
                ind = ranges[i,:]
                t_new = np.append(t_new,t[ind[0]])
                t_new = np.append(t_new,t[ind[1]])
            else:
                n_ind = range_len[i]
                ind = ranges[i,:]
                tx = np.linspace(t[ind[0]],t[ind[1]],1*n_ind)
                t_new = np.append(t_new,tx)
            
            try:
                ind1 = ranges[i,1]; ind2 = ranges[i+1,0]
                t_new = np.append(t_new,t[ind1:ind2])
            except:
                pass
                
        ind_f = ranges[-1,-1]
        t_new = np.append(t_new,t[ind_f:])
        
        opts.dt.mesh = 'USER';
        opts.dt.t = t_new
        
        
        
        
    return opts
        
        
        
        
    
    
    
    
    
