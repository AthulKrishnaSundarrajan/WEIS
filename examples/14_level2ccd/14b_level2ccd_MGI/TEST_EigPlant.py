#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 12:56:07 2022

@author: athulsun
"""

import numpy as np 
from LoadMatrices import LoadMatrices 
from InterpolateLinearModels import GetSparsity,LinearModel,BuildSurrogate,EvaluateModel
from scipy.interpolate import RBFInterpolator,RegularGridInterpolator,PchipInterpolator
import os
import pickle
import time
import matplotlib.pyplot as plt
from numpy.linalg import eig

def GetIndices(nl):
    if nl % 2:
        FitInd = np.arange(0,nl+1,2)
        ValInd = np.arange(1,nl-1,2)
    else:
        FitInd = np.arange(0,nl,2)
        FitInd = np.append(FitInd,nl-1)
        ValInd = np.arange(1,nl-1,2)
        
    return FitInd,ValInd

def eval_eig(A,index):
    
    na = len(A)
    
    eigval = np.zeros((na,2))
    
    for i in range(na):
        eig_,null = eig(A[i,:,:])
        
        eigval[i,0] = eig_[index].real 
        eigval[i,1] = eig_[index].imag 
        
    return eigval

if __name__ == '__main__':
    
     # read modeling and analysis options
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    
    # load matrices
    fol_name = 'jmd_full'
    A,B,C,D,x,u,y,u_h,DVo,DescStates,DescCntrlInpt,DescOutput = LoadMatrices(fol_name)
    
    # get size of matrices
    ns,nx,nu,nw = np.shape(B)
    ny = len(DescOutput)
    
    cs = np.unique(DVo[:,0]); bv = np.unique(DVo[:,1])
    data  = (cs,bv,u_h)
    Asm,sp_A = BuildSurrogate(A,data)
    Bsm,sp_B = BuildSurrogate(B,data)
    Csm,sp_C = BuildSurrogate(C,data)
    Dsm,sp_D = BuildSurrogate(D,data)
    
    xsm,sp_X = BuildSurrogate(x,data) 
    usm,sp_U = BuildSurrogate(u,data)
    ysm,sp_Y = BuildSurrogate(y,data)
    
    fol_name = 'pv_bv'
    Aval,Bval,Cval,Dval,xval,uval,yval,wval,DVval,DescStates,DescCntrlInpt,DescOutput = LoadMatrices(fol_name)
    
    
    LM_sm = LinearModel(Asm,Bsm,Csm,Dsm,xsm,usm,ysm,DescStates,DescCntrlInpt,DescOutput,nx,nu,ny,u_h)
    sp_info = {"sp_A":sp_A,"sp_B":sp_B,"sp_C":sp_C,"sp_D":sp_D,"sp_X":sp_X,"sp_U":sp_U,"sp_Y":sp_Y}
    
    
    # get fitting and validation indices
    FitInd,ValInd = GetIndices(len(DVval))
    nfit = len(FitInd); nval = len(ValInd)
    xind = [True,False,False,False,True]
    uind = [False,True,True]
    val = np.array([1,1,1e7,1])
    
    
    Aval = np.squeeze(Aval[ValInd,:,:,:])
    XUval  = np.zeros((nval,4))
    XU_ = np.squeeze(xval[ValInd,:,0])
    XUval[:,0:2] = XU_[:,xind]
   
    XU_ = np.squeeze(uval[ValInd,:,0])
    XUval[:,2:] = XU_[:,uind]
    
    XUval = XUval/val
    
    indW = (u_h == wval)
    indP = (DVo[:,0] == 54)
    
    Afit = np.squeeze(A[indP,:,:,indW])
    
    
    XUfit  = np.zeros((nfit,4))
    XU_ = np.squeeze(x[indP,:,indW])
    XUfit[:,0:2] = XU_[:,xind]
    XU_ = np.squeeze(u[indP,:,indW])
    XUfit[:,2:] = XU_[:,uind]
    XUfit = XUfit/val
    
    ninterp = 1000
    
    X_ = np.zeros((ninterp,3))
    X_[:,0] = 54
    X_[:,1] = np.linspace(bv[0],bv[-1],ninterp)
    X_[:,2] = wval
    
    Ainterp = np.zeros((ninterp,nx,nx))
    Einterp = np.zeros((ninterp,2))
    
    XUinterp = np.zeros((ninterp,4))
    
    
    for i in range(ninterp):
        
        # evaluate LM
        LM_ = EvaluateModel(LM_sm,sp_info,X_[i,:],len(wval),nx,nu,ny)
        
        # extract A matrix
        A_ = np.transpose(LM_.A_ops,[2,0,1])
        x_ = LM_.x_ops 
        u_ = LM_.u_ops
        
        Einterp[i,:] = eval_eig(A_,0)
        XUinterp[i,0:2] = x_[xind,0]
        XUinterp[i,2:] = u_[uind,0]
        
    XUinterp = XUinterp/val
        
        
        
    
    #Einterp = eval_eig(Ainterp,0)
    Efit = eval_eig(Afit,0)
    Eval = eval_eig(Aval,0)
    
    fig1,ax1 = plt.subplots(1)
    ax1.set_ylabel('Imaginary part ',fontsize = 16)
    ax1.set_xlabel('Real Part',fontsize = 16)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    
    ax1.plot(Einterp[:,0],Einterp[:,1],'k')
    ax1.plot(Efit[:,0],Efit[:,1],'ro',markersize=4)
    ax1.plot(Eval[:,0],Eval[:,1],'ro',markersize=4,alpha=0.5)
    
    fig2,ax2 = plt.subplots(1)
    ax2.set_xlabel('CS',fontsize = 16)
    ax2.set_ylabel('XU',fontsize = 16)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    
    ax2.plot(X_[:,1],XUinterp,'k')
    ax2.plot(DVval[FitInd],XUfit,'ro',markersize=4)
    ax2.plot(DVval[ValInd],XUval,'ro',markersize=4,alpha=0.5)
    
    results = {'Einterp':Einterp,'Efit':Efit,'Eval':Eval,
               'XUinterp':XUinterp,'XUfit':XUfit,'XUval':XUval,
               'Xinterp':X_,'Xfit':DVval[FitInd],'Xval':DVval[ValInd]}
    
    pkl_name = 'pv_eigOP_bv.mat'
    
    saveflag = 1
    
    if saveflag:
        from scipy.io import savemat 
        
        savemat(pkl_name,results)
    
    
    
    
    
    