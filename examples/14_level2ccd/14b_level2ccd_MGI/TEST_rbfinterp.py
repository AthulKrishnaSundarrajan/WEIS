#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:14:24 2022

@author: athulsun


"""

import os
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt 
import openmdao.api as om 
from scipy.interpolate import RBFInterpolator,LinearNDInterpolator
from scipy.io import savemat 
from functools import partial
from scipy.sparse import find

class LinearModel:
    def __init__(self,A,B,C,D,x,u,y,DescStates,DescControl,DescOutput,nx,nu,ny,u_h,mtype = None):
        self.A_ops = A; self.B_ops = B; self.C_ops = C; self.D_ops = D;
        self.x_ops = x; self.u_ops = u; self.y_ops = y;
        self.DescStates = DescStates; self.DescCntrlInpt = DescControl;
        self.DescOutput = DescOutput; self.u_h = u_h;
        self.nx = nx; self.nu = nu; self.ny = ny
        self.mtype = mtype
 
# def BuildFunction(data1,values1):
#     return LinearNDInterpolator(data1, values1)
        
# surrogate model
def BuildSurrogate(data,lin_mat):
    
    # get the size of mat
    sizemat = len(np.shape(lin_mat))
    
    # initialize l
    l = 0
    
    # sizemat == 3 for operating points 
    if sizemat == 3:
        
        # get size
        n_sample,na,nw = np.shape(lin_mat)
        
        # initialize matrices
        values = np.zeros((n_sample*nw))
        mat_sm = np.zeros((na,1),dtype='O')
        
        # loop through and get values
        for a in range(na):
            for n in range(n_sample):
                for w in range(nw):
                    values[l] = lin_mat[n,a,w]
                    l+=1
            # interpolate
            l = 0
            def BuildFunction(data = data,values = values):
                return LinearNDInterpolator(data, values)
            
            mat_sm[a] = BuildFunction(data, values)
            #mat_sm[a] = RBFInterpolator(data,values)
    
    # sizemat == 4 for ABCD matrices
    elif sizemat == 4:
        
        # get size
        n_sample,na,nb,nw = np.shape(lin_mat)
        
        # initialize matrices
        values = np.zeros((n_sample*nw))
        mat_sm = np.zeros((na,nb),dtype = 'O')
        
        # loop through and get values
        for a in range(na):
            for b in range(nb):
                for n in range(n_sample):
                    for w in range(nw):
                        values[l] = lin_mat[n,a,b,w]
                        l += 1
                # interpolate
                l=0
                def BuildFunction(data = data,values = values):
                    return LinearNDInterpolator(data, values)
                mat_sm[a,b] = BuildFunction(data, values)
                #mat_sm[a,b] = RBFInterpolator(data,values)
           
                  
    return mat_sm


def evaluateLM(LM,data,nw):
    
    # extract matrices
    A_ = evaluateModel(LM.A_ops,data);B_ = evaluateModel(LM.B_ops,data); C_ = evaluateModel(LM.C_ops,data); D_ = evaluateModel(LM.D_ops,data)
    x_ = evaluateModel(LM.x_ops,data); u_ = evaluateModel(LM.u_ops,data); y_ = evaluateModel(LM.y_ops,data)
    
    # description
    if nw == 1:
        x_ = np.squeeze(x_,-1)
        u_ = np.squeeze(u_,-1)
        y_ = np.squeeze(y_,-1)
        
    else:
        x_ = np.squeeze(x_,1)
        u_ = np.squeeze(u_,1)
        y_ = np.squeeze(y_,1)
    
    # get sizes
    nx = len(LM.DescStates); nu = len(LM.DescCntrlInpt); ny = len(LM.DescOutput)
    
    LM_ = LinearModel(A_,B_,C_,D_,x_,u_,y_,LM.DescStates,LM.DescCntrlInpt,LM.DescOutput,nx,nu,ny,LM.u_h,mtype = 'ABCD')
    
    return LM_

def evaluateModel(X_sm,data):
    
    # get shape of matrices
    na,nb = np.shape(X_sm)
    
    # number of wind speeds
    nw = len(data);
    
    # initialize matrix
    X_mat = np.zeros((na,nb,nw))
    
    # loop through and evaluate matrix 
    for i in range(na):
        for j in range(nb):
            
            X_mat[i,j,:] = X_sm[i,j](data)
           
    # return data
    return X_mat

if __name__ == "__main__":
    
    # get directory 
    weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    
    # read modeling and analysis options
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    
    # Linear Model
    opt_path = mydir + os.sep + "outputs" + os.sep +"jmd_full"
    pkl_file = opt_path + os.sep + "ABCD_matrices.pkl" 
    
    # load linear model
    with open(pkl_file,"rb") as handle:
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
   
    # change list to array
    A = np.array(A); B = np.array(B); C = np.array(C); D = np.array(D); A1 = A[0,:,:,0]
    x = np.array(x);u = np.array(u);y = np.array(y)
    
    # get shape
    ns,nx,nu,nw = np.shape(B); ny = len(DescOutput)
    
    # A = np.transpose(A,[0,3,1,2]); A = np.reshape(A,(ns,nw*nx*nx),order = 'F')
    # B = np.transpose(B,[0,3,1,2]); B = np.reshape(B,(ns,nw*nx*nu),order = 'F')
    # C = np.transpose(C,[0,3,1,2]); C = np.reshape(C,(ns,nw*ny*nx),order = 'F')
    # D = np.transpose(D,[0,3,1,2]); D = np.reshape(D,(ns,nw*ny*nu),order = 'F')
    
    # x = np.transpose(x,[0,2,1]);x = np.reshape(x,(ns,nw*nx),order = 'F')
    # u = np.transpose(u,[0,2,1]);u = np.reshape(u,(ns,nw*nu),order = 'F')
    # y = np.transpose(y,[0,2,1]); y = np.reshape(y,(ns,nw*ny),order = 'F')
    
   
    
    
    # set length 
    n_dv = len(dvs.keys())
    u_h = ABCD_list[0]['u_h']
    
    # reshape into array
    DV = np.reshape(DVs,(idx+1,n_dv),order = 'C')
    n_samples = len(driver_cases)
    
    # number of linmat
    n_wind = len(u_h)
    
    n_case = n_samples*n_wind

    if n_wind == 1:
        
        # set data as design variables
        data = DV
        
        
    elif n_wind>1:
        
        # initialize storage
        data = np.zeros((n_case,3))
        
        # initialize index
        l = 0
        
        # loop through and get data
        for i in range(n_samples):
            for j in range(n_wind):
                data[l,0:2] = DV[i,:]
                data[l,2] = u_h[j]
                l+=1
                
    kernel = 'cubic'
    # construct surrogate models
    t0_con = time.time()
    A_sm = BuildSurrogate(data, A)
    B_sm = BuildSurrogate(data, B)
    C_sm = BuildSurrogate(data, C)
    D_sm = BuildSurrogate(data, D)
    
    x_sm = BuildSurrogate(data, x)
    u_sm = BuildSurrogate(data, u)
    y_sm = BuildSurrogate(data, y)
    tf_con = time.time()
    nx,nu = np.shape(B_sm); ny = len(DescOutput)
    
     # add linear models
    LM_sm = LinearModel(A_sm,B_sm,C_sm,D_sm,x_sm,u_sm,y_sm,DescStates,DescCntrlInpt,DescOutput,nx,nu,ny,u_h)
    
    ind = -1
          
    if n_wind == 1:
        DV_ = np.array([[36, 350]])
        
    elif n_wind >1:
        DV_ = np.zeros((n_wind,3))
        DV_[:,0:2] = np.array([[36., 350.]])
        DV_[:,2] = u_h
     
    t0_eval = time.time()               
    LM_ = evaluateLM(LM_sm,DV_,n_wind)
    tf_eval = time.time()
    # extract values
    A_rbf = LM_.A_ops
    B_rbf = LM_.B_ops
    C_rbf = LM_.C_ops 
    D_rbf = LM_.D_ops 

    x_rbf = LM_.x_ops
    u_rbf = LM_.u_ops 
    y_rbf = LM_.y_ops 

    u_h = LM_.u_h 
    breakpoint()
    opt_path = mydir + os.sep + "outputs" + os.sep + "jmd_single"
    #case_detail_path = opt_path + os.sep + 'case_detail.pkl'
    pkl_path = opt_path + os.sep + "ABCD_matrices.pkl"

    # load pickle file
    with open(pkl_path, 'rb') as handle:
        ABCD_lin = pickle.load(handle)[0]
    #ABCD_lin = ABCD_list[ind]

    A_lin = ABCD_lin['A'][0:5, 0:5]
    B_lin = ABCD_lin['B'][0:5, :]
    C_lin = ABCD_lin['C'][:, 0:5]
    D_lin = ABCD_lin['D']

    x_lin = ABCD_lin['x_ops'][0:5]
    u_lin = ABCD_lin['u_ops']
    y_lin = ABCD_lin['y_ops']

    fig1,ax1 = plt.subplots(1)
    ax1.set_ylabel('Ptfm Pitch [deg] ',fontsize = 16)
    ax1.set_xlabel('Wind Speed [m/s]',fontsize = 16)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    
    ax1.plot(u_h,np.rad2deg(x_rbf[0,:]),'o-',label = 'LIN-ND')
    ax1.plot(u_h,np.rad2deg(x_lin[0,:]),'*-',label = 'LIN')
    
    ax1.legend()
    
    fig2,ax2 = plt.subplots(1)
    ax2.set_ylabel('Bld Pitch [deg]',fontsize = 16)
    ax2.set_xlabel('Wind Speed [m/s]',fontsize = 16)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    
    ax2.plot(u_h,np.rad2deg(u_rbf[2,:]),'o-',label = 'LIN-ND')
    ax2.plot(u_h,np.rad2deg(u_lin[2,:]),'*-',label = 'LIN')
    
    ax2.legend()
    
    fig3,ax3 = plt.subplots(1)
    ax3.set_ylabel('Gen Speed [rad/s] ',fontsize = 16)
    ax3.set_xlabel('Wind Speed [m/s]',fontsize = 16)
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    
    ax3.plot(u_h,x_rbf[4,:],'o-',label = 'LIN-ND')
    ax3.plot(u_h,x_lin[4,:],'*-',label = 'LIN')
    
    ax3.legend()
    
    fig4,ax4 = plt.subplots(1)
    ax4.set_ylabel('Gen Torque [MWm] ',fontsize = 16)
    ax4.set_xlabel('Wind Speed [m/s]',fontsize = 16)
    ax4.tick_params(axis='x', labelsize=12)
    ax4.tick_params(axis='y', labelsize=12)
    
    ax4.plot(u_h,u_rbf[1,:],'o-',label = 'LIN-ND')
    ax4.plot(u_h,u_lin[1,:],'*-',label = 'LIN')
    
    ax4.legend()
    
    A_e = A_rbf-A_lin
    A_error = np.linalg.norm(A_e,axis = 2)
    
    B_error = np.linalg.norm(B_rbf-B_lin,axis = 2)
    
    C_error = np.linalg.norm(C_rbf-C_lin,axis = 2)
    
    D_error = np.linalg.norm(D_rbf-D_lin,axis = 2)
    
    x_error = np.linalg.norm(x_rbf-x_lin,axis = 1)
    u_error = np.linalg.norm(u_rbf-u_lin,axis = 1)
    y_error = np.linalg.norm(y_rbf-y_lin,axis = 1)

    
        
    
