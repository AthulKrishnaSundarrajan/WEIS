#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:13:18 2022

@author: athulsun
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import find

class LinearModel:
    def __init__(self,A,B,C,D,x,u,y,DescStates,DescControl,DescOutput,nx,nu,ny,u_h,mtype = None):
        self.A_ops = A; self.B_ops = B; self.C_ops = C; self.D_ops = D;
        self.x_ops = x; self.u_ops = u; self.y_ops = y;
        self.DescStates = DescStates; self.DescCntrlInpt = DescControl;
        self.DescOutput = DescOutput; self.u_h = u_h;
        self.nx = nx; self.nu = nu; self.ny = ny
        self.mtype = mtype

def EvaluateModel(LM,sp_,data,nw,nx,nu,ny):
    
    # extract matrices
    A_ = EvaluateMatrix(LM.A_ops,sp_["sp_A"],nw,nx,nx,data)
    B_ = EvaluateMatrix(LM.B_ops,sp_["sp_B"],nw,nx,nu,data)
    C_ = EvaluateMatrix(LM.C_ops,sp_["sp_C"],nw,ny,nx,data)
    D_ = EvaluateMatrix(LM.D_ops,sp_["sp_D"],nw,ny,nu,data)
    
    x_ = EvaluateMatrix(LM.x_ops,sp_["sp_X"],nw,nx,1,data)
    u_ = EvaluateMatrix(LM.u_ops,sp_["sp_U"],nw,nu,1,data)
    y_ = EvaluateMatrix(LM.y_ops,sp_["sp_Y"],nw,ny,1,data)
    
    
    # get sizes
    nx = len(LM.DescStates); nu = len(LM.DescCntrlInpt); ny = len(LM.DescOutput)
    
    LM_ = LinearModel(A_,B_,C_,D_,x_,u_,y_,LM.DescStates,LM.DescCntrlInpt,LM.DescOutput,nx,nu,ny,LM.u_h,mtype = 'ABCD')
    
    return LM_

def EvaluateMatrix(Xsm,sp_,nw,na,nb,DV_):
    
    # initialize matrix
    Xmat = np.zeros((na*nb,nw))
    
    # loop through and evaluate matrix
    for i in range(na*nb):
   
        if sp_[i]:
    
            values_ = Xsm[i](DV_)
            Xmat[i,:] = values_
            
        else:
            Xmat[i,:] = 0  
            
    # reshape matrix
    if nb == 1:
        Xmat = np.reshape(Xmat,(na,nw),order = 'F')
    else:
        Xmat = np.reshape(Xmat,(na,nb,nw),order = 'F')
    
    return Xmat

def BuildFunction(data,values):
    return RegularGridInterpolator(data,values,bounds_error = False,fill_value=None)


def GetSparsity(Xmat,n_x,tol):
    
    
    # get sparsity pattern
    Xind = find(np.abs(Xmat)>tol)
    
    # get dimensions
    dim = np.unique(Xind[1])
    
    # initialize
    sp_ = np.zeros(n_x)
    sp_[dim] = True
    
    return sp_
    

def BuildSurrogate(Xmat,data):
    
    # get shape
    X_shape = np.shape(Xmat)
    
    # ABCD matrices
    if len(X_shape) == 4:
        
        # get shape
        ns = X_shape[0];na = X_shape[1];nb = X_shape[2];nw = X_shape[3]
        
        # matrices
        Xmat = np.transpose(Xmat,[0,3,1,2]); Xmat = np.reshape(Xmat,(ns*nw,na*nb),order = 'F')
    
    # operationg points     
    elif len(X_shape) == 3:
         
         # get shape
         ns = X_shape[0];na = X_shape[1];nb = 1;nw = X_shape[2]
         
         # matrices
         Xmat = np.transpose(Xmat,[0,2,1]);Xmat = np.reshape(Xmat,(ns*nw,na),order = 'F')
            
    
    # initialize matrix
    Xsm = np.zeros((na*nb),dtype = 'O')
    
    # get sparsity pattern 
    sp_ = GetSparsity(Xmat,na*nb,1e-7)
    
    # loop through 
    for i in range(na*nb):   
        
        # construct model only if the value is nonzero
        if sp_[i]:
            
            # extract values
            values_ = Xmat[:,i]
            
            # reshape values
            values_ = np.reshape(values_,(6,6,nw),order = 'F')
            
            # extract values
            Xsm[i] = BuildFunction(data,values_)
    
    # return values
    return Xsm,sp_