#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST_InterpApproach.py


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

def EvalEig(A,u_h,FitInd,ValInd):
    
    A = np.transpose(A,[2,0,1])
    
    # get length
    ninterp = 1000
    index = 0
    
    W_ = np.linspace(u_h[0],u_h[-1],ninterp)
    
    Afit = A[FitInd,:,:]
    Aval = A[ValInd,:,:]
    
    A_pp = PchipInterpolator(u_h,A,axis=0)
    A_op = lambda w: A_pp(w)
    
    Ainterp = A_op(W_)
    
    
    # initialize eigen value matrix
    eig_interp = eval_eig(Ainterp,index)
    eig_fit =  eval_eig(A,index)
    eig_val =  eval_eig(Aval,index)

    return eig_interp,eig_fit,eig_val,W_
    
def EvaluateModelW(LM_list,nx,nu,ny,ns,nw,u_h,DescStates,DescCntrlInpt,DescOutput,data):
    
    # initialize matrices
    A = np.zeros((nx,nx,nw))
    B = np.zeros((nx,nu,nw))
    C = np.zeros((ny,nx,nw))
    D = np.zeros((ny,nu,nw))
    
    x = np.zeros((nx,nw))
    u = np.zeros((nu,nw))
    y = np.zeros((ny,nw))
    
    # loop through and evaluate matrix
    for i in range(nw):
        A[:,:,i] = EvaluateMatrixW(LM_list[i]['Asm'],LM_list[i]['spA'],nx,nx,data)
        B[:,:,i] = EvaluateMatrixW(LM_list[i]['Bsm'],LM_list[i]['spB'],nx,nu,data)
        C[:,:,i] = EvaluateMatrixW(LM_list[i]['Csm'],LM_list[i]['spC'],ny,nx,data)
        D[:,:,i] = EvaluateMatrixW(LM_list[i]['Dsm'],LM_list[i]['spD'],ny,nu,data)
        
        
        x[:,i] = EvaluateMatrixW(LM_list[i]['Xsm'],LM_list[i]['spX'],nx,1,data)
        u[:,i] = EvaluateMatrixW(LM_list[i]['Usm'],LM_list[i]['spU'],nu,1,data)
        y[:,i] = EvaluateMatrixW(LM_list[i]['Ysm'],LM_list[i]['spY'],ny,1,data)
        
    # assign linear model
    LM = LinearModel(A,B,C,D,x,u,y,DescStates,DescCntrlInpt,DescOutput,nx,nu,ny,u_h,mtype = 'ABCD')
    
    return LM
        
        
        
        
def EvaluateMatrixW(Xsm,spX,na,nb,data):
    
    # initialize
    Xmat = np.zeros((na*nb))
    
    # loop through and evaluate surrogate model
    for i in range(na*nb):
        
        # evaluate matrix only if nonzero
        if spX[i]:
            
            Xmat[i] = Xsm[i](data)
            
        else:
            
            Xmat[i] = 0
           
    # if matrix reshape
    if nb > 1:
        Xmat = np.reshape(Xmat,(na,nb),order = 'F')
                
    return Xmat
            

def BuildSurrogateW(A,B,C,D,x,u,y,DV,ns,nx,nu,ny,nw,method):
    
    LM = []
    
    for i in range(nw):
        
         # extract matrices
        A_ = A[:,:,:,i]
        B_ = B[:,:,:,i]
        C_ = C[:,:,:,i]
        D_ = D[:,:,:,i]
        
        # operating points
        x_ = x[:,:,i]
        u_ = u[:,:,i]
        y_ = y[:,:,i]
        
        # construct surrogate for matrix
        Asm,spA = BuildSurrogateMatrixW(A_,DV,method)
        Bsm,spB = BuildSurrogateMatrixW(B_,DV,method)
        Csm,spC = BuildSurrogateMatrixW(C_,DV,method)
        Dsm,spD = BuildSurrogateMatrixW(D_,DV,method)
        
        Xsm,spX = BuildSurrogateMatrixW(x_,DV,method) 
        Usm,spU = BuildSurrogateMatrixW(u_,DV,method)
        Ysm,spY = BuildSurrogateMatrixW(y_,DV,method)
        
        # add to dict
        sm_dict = {'Asm':Asm,'spA':spA,
                   'Bsm':Bsm,'spB':spB,
                   'Csm':Csm,'spC':spC,
                   'Dsm':Dsm,'spD':spD,
                   'Xsm':Xsm,'spX':spX,
                   'Usm':Usm,'spU':spU,
                   'Ysm':Ysm,'spY':spY
                   }
        
        LM.append(sm_dict)
        
    return LM
        


def BuildFunctionW(data,values,method):
    
    if method == 'RBF':
        return RBFInterpolator(data,values)
    
    elif method == 'RGI':
        return RegularGridInterpolator(data,values)

def BuildSurrogateMatrixW(Xmat,DV,method):
    
    # get shape
    Xsize = np.shape(Xmat)
    
    # depending on the method get the data
    
    if method == 'RBF':
        
        data = DV
        
    elif method == 'RGI':
        
        # extract unique values of design variables
        cs = np.unique(DV[:,0])
        bv = np.unique(DV[:,1])
        
        ndv = len(cs)
        
        data = (cs,bv)
   
    if len(Xsize) == 3:
        
        # ABCD matrix
        ns = Xsize[0];na = Xsize[1]; nb = Xsize[2]
        
        # reshape matrix
        Xmat = np.reshape(Xmat,(ns,na*nb),order = 'F')
        
    elif len(Xsize) == 2:
        
        # ABCD matrix
        ns = Xsize[0];na = Xsize[1]; nb = 1
        
        
    # initialize matrix
    Xsm = np.zeros((na*nb),dtype = 'O')
    
    # get sparsity pattern 
    sp_ = GetSparsity(Xmat,na*nb,1e-7)
    
    # loop through and build surrogate
    for i in range(na*nb):
        
        # build surrogate only if the element is nonzero
        if sp_[i]:
                      
            values_ = Xmat[:,i]
            
            # reshape values if using RGI
            if method == 'RGI':
                
                # reshape values
                values_ = np.reshape(values_,(ndv,ndv),order = 'F')
            
                # construct surrogate model
            Xsm[i] = BuildFunctionW(data,values_,method)
               
    
    return Xsm,sp_        
        
        


if __name__ == '__main__':
    
     # read modeling and analysis options
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    
    # load matrices
    fol_name = 'jmd_full'
    A,B,C,D,x,u,y,u_h,DV,DescStates,DescCntrlInpt,DescOutput = LoadMatrices(fol_name)
    
    # get size of matrices
    ns,nx,nu,nw = np.shape(B)
    ny = len(DescOutput)
    
    # initialize storage array
    t0 = time.time()
    LM_list = BuildSurrogateW(A,B,C,D,x,u,y,DV,ns,nx,nu,ny,nw,'RGI')
    tf = time.time()
    t_build1 = (tf-t0)
    
    data = np.array([[51.75,519]])
    t0 = time.time()
    LM = EvaluateModelW(LM_list,nx,nu,ny,ns,nw,u_h,DescStates,DescCntrlInpt,DescOutput,data)
    tf = time.time()
    t_eval1 = (tf-t0)
    
    A_int1 = LM.A_ops
    x_int = LM.x_ops
    
    #################################
    # construct surrogate model
    cs = np.unique(DV[:,0]); bv = np.unique(DV[:,1])
    data  = (cs,bv,u_h)
    t0 = time.time()
    Asm,sp_A = BuildSurrogate(A,data)
    Bsm,sp_B = BuildSurrogate(B,data)
    Csm,sp_C = BuildSurrogate(C,data)
    Dsm,sp_D = BuildSurrogate(D,data)
    
    xsm,sp_X = BuildSurrogate(x,data) 
    usm,sp_U = BuildSurrogate(u,data)
    ysm,sp_Y = BuildSurrogate(y,data)
    tf = time.time()
    t_build2 = (tf-t0)
    
    # add linear models
    LM_sm = LinearModel(Asm,Bsm,Csm,Dsm,xsm,usm,ysm,DescStates,DescCntrlInpt,DescOutput,nx,nu,ny,u_h)
    sp_info = {"sp_A":sp_A,"sp_B":sp_B,"sp_C":sp_C,"sp_D":sp_D,"sp_X":sp_X,"sp_U":sp_U,"sp_Y":sp_Y}
    
    DV = np.zeros((1000,3))
    DV[:,0:2] = np.array([[51.75,519]]) 
    DV[:,2] = np.linspace(u_h[0],u_h[-1],1000)
    
    t0 = time.time()
    LM_ = EvaluateModel(LM_sm,sp_info,DV,1000,nx,nu,ny)
    tf = time.time()
    t_eval2 = (tf-t0)
    A_ops2 = LM_.A_ops; A_ops2 = np.transpose(A_ops2,[2,0,1])
    I_int2 = eval_eig(A_ops2,0)
    
    DV = np.zeros((nw,3))
    DV[:,0:2] = np.array([[51.75,519]]) 
    DV[:,2] = u_h
    
    t0 = time.time()
    LM_ = EvaluateModel(LM_sm,sp_info,DV,nw,nx,nu,ny)
    tf = time.time()
    t_eval3 = (tf-t0)
    
    A_ops2 = LM_.A_ops; A_ops2 = np.transpose(A_ops2,[2,0,1])
    F_int2 = eval_eig(A_ops2,0)
    
    #################################################################################################
    
    opt_path = mydir + os.sep + "outputs" + os.sep + "jmd_single"
    pkl_path = opt_path + os.sep + "ABCD_matrices.pkl"

    # load pickle file
    with open(pkl_path, 'rb') as handle:
        ABCD_lin = pickle.load(handle)[0]
    #ABCD_lin = ABCD_list[ind]

    A_lin = ABCD_lin['A'][0:5,0:5]
    B_lin = ABCD_lin['B'][0:5, :]
    C_lin = ABCD_lin['C'][:, 0:5]
    D_lin = ABCD_lin['D']

    x_lin = ABCD_lin['x_ops'][0:5]
    u_lin = ABCD_lin['u_ops']
    y_lin = ABCD_lin['y_ops']
    
    FitInd,ValInd = GetIndices(nw)
    
    I_int1,F_int1,V_int1,W_ = EvalEig(LM.A_ops,u_h,FitInd,ValInd)
    I_int3,F_int3,V_int3,W_ = EvalEig(LM_.A_ops,u_h,FitInd,ValInd)
    I_lin,F_lin,V_lin,W_ = EvalEig(A_lin,u_h,FitInd,ValInd)
    
    fig1,ax1 = plt.subplots(1)
    ax1.set_ylabel('Imaginary part ',fontsize = 16)
    ax1.set_xlabel('Real Part',fontsize = 16)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    
    ax1.plot(I_int1[:,0],I_int1[:,1],'k-',label = 'Interpolated 1')
    ax1.plot(F_int1[:,0],F_int1[:,1],'ro',markersize=4)
    #ax1.plot(V_int1[:,0],V_int1[:,1],'ro',markersize=4,alpha=0.5)
    
    ax1.plot(I_int2[:,0],I_int2[:,1],'g-',label = 'Interpolated 2')
    ax1.plot(F_int2[:,0],F_int2[:,1],'ro',markersize=4)
    
    ax1.plot(I_int3[:,0],I_int3[:,1],'y-',label = 'Interpolated 3')
    ax1.plot(F_int3[:,0],F_int3[:,1],'ro',markersize=4)
    #ax1.plot(V_int2[:,0],V_int2[:,1],'ro',markersize=4,alpha=0.5)
    
    ax1.plot(I_lin[:,0],I_lin[:,1],'b-',label = 'Original')
    ax1.plot(F_lin[:,0],F_lin[:,1],'ro',markersize=4)
    #ax1.plot(V_lin[:,0],V_lin[:,1],'ro',markersize=4,alpha=0.5)
    
    ax1.legend()
    
    
    
        
       
        
        
        
        
    