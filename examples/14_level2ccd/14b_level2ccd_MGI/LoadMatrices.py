#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:21:57 2022

@author: athulsun
"""

import os
import pickle
import openmdao.api as om
import numpy as np

def LoadMatrices(fol_name):
    
     # get path to pickle and sql files
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    opt_path = mydir + os.sep + "outputs" + os.sep + fol_name
    pkl_path = opt_path + os.sep +  "ABCD_matrices.pkl"
    
     # load pickle file
    with open(pkl_path, 'rb') as handle:
        ABCD_list = pickle.load(handle)
    
    # get state, input, output descriptions
    DescStates = ABCD_list[0]['DescStates']; DescCntrlInpt = ABCD_list[0]['DescCntrlInpt']; DescOutput = ABCD_list[0]['DescOutput']
    u_h = ABCD_list[0]['u_h']
    # load case reader
    cr = om.CaseReader(opt_path+ os.sep +"log_opt.sql")
    driver_cases = cr.get_cases('driver')
    
    # initialize
    DVs = []
    A = []; B = []; C= []; D = [];x = []; u = []; y = [];
    
    # get ABCD matrices
    for idx, case in enumerate(driver_cases):
        dvs = case.get_design_vars(scaled=False)
        A.append(ABCD_list[idx]['A'][0:5,0:5]);B.append(ABCD_list[idx]['B'][0:5,:]);C.append(ABCD_list[idx]['C'][:,0:5]);D.append(ABCD_list[idx]['D']);
        x.append(ABCD_list[idx]['x_ops'][0:5]);u.append(ABCD_list[idx]['u_ops']);y.append(ABCD_list[idx]['y_ops']);
        for key in dvs.keys():
            DVs.append(dvs[key])
    
    # set length 
    n_dv = len(dvs.keys())
    u_h = ABCD_list[0]['u_h']
    
    # reshape into array
    DV = np.reshape(DVs,(idx+1,n_dv),order = 'C')
    
    # change list to array
    A = np.array(A); B = np.array(B); C = np.array(C); D = np.array(D);
    x = np.array(x);u = np.array(u);y = np.array(y)
     
    return A,B,C,D,x,u,y,u_h,DV,DescStates,DescCntrlInpt,DescOutput