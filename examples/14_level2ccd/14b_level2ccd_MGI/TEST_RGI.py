#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import openmdao.api as om
import pickle
import matplotlib.pyplot as plt
from scipy.sparse import find 
from scipy.interpolate import RegularGridInterpolator,LinearNDInterpolator,RBFInterpolator,NearestNDInterpolator

def BuildFunction(data,values):
    
    return RegularGridInterpolator(data,values)

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
        Xmat = np.transpose(Xmat,[0,3,1,2]); Xmat = np.reshape(Xmat,(ns*nw,nx*nx),order = 'F')
    
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
            values_ = np.reshape(values_,(6,6,23),order = 'F')
            
            # extract values
            Xsm[i] = BuildFunction(data,values_)
    
    # return values
    return Xsm,sp_

def EvaluateModel(Xsm,sp_,nw,na,nb,DV_):
    
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

# get path to pickle and sql files
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
opt_path = mydir + os.sep + "outputs" + os.sep +"jmd_full"

pkl_path = opt_path + os.sep+  "ABCD_matrices.pkl"

# load pickle file
with open(pkl_path, 'rb') as handle:
    ABCD_list = pickle.load(handle)

# get state, input, output descriptions
DescStates = ABCD_list[0]['DescStates']; DescCntrlInpt = ABCD_list[0]['DescCntrlInpt']; DescOutput = ABCD_list[0]['DescOutput']

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

x1 = x[0,:,:]; A1 = A[0,:,:,:]
# get shape
ns,nx,nu,nw = np.shape(B); ny = len(DescOutput)

# permute and reshape matrices
#A = np.transpose(A,[0,3,1,2]); A = np.reshape(A,(ns*nw,nx*nx),order = 'F')
# B = np.transpose(B,[0,3,1,2]); B = np.reshape(B,(ns*nw,nx*nu),order = 'F')
# C = np.transpose(C,[0,3,1,2]); C = np.reshape(C,(ns*nw,ny*nx),order = 'F')
# D = np.transpose(D,[0,3,1,2]); D = np.reshape(D,(ns*nw,ny*nu),order = 'F')

# x = np.transpose(x,[0,2,1]);x = np.reshape(x,(ns*nw,nx),order = 'F')
# u = np.transpose(u,[0,2,1]);u = np.reshape(u,(ns*nw,nu),order = 'F')
# y = np.transpose(y,[0,2,1]); y = np.reshape(y,(ns*nw,ny),order = 'F')

cs = np.unique(DV[:,0])
bv = np.unique(DV[:,1])
ws = u_h

DV_ = np.zeros((nw,3))
DV_[:,0:2] = np.array([[51.75, 519]])
DV_[:,2] = u_h   

Asm,sp_A = BuildSurrogate(A,(cs,bv,ws))
Xsm,sp_X = BuildSurrogate(x,(cs,bv,ws))
Usm,sp_U = BuildSurrogate(u,(cs,bv,ws))



A_rbf = EvaluateModel(Asm,sp_A,nw,nx,nx,DV_)
x_rbf = EvaluateModel(Xsm,sp_X,nw,nx,1,DV_)


opt_path = mydir + os.sep + "outputs" + os.sep +"jmd_single"
pkl_path = opt_path + os.sep+  "ABCD_matrices.pkl"

# load pickle file
with open(pkl_path, 'rb') as handle:
    ABCD_lin = pickle.load(handle)[0]

    
A_lin = ABCD_lin['A'][0:5,0:5]
B_lin = ABCD_lin['B'][0:5,:]
C_lin = ABCD_lin['C'][:,0:5]
D_lin = ABCD_lin['D']

x_lin = ABCD_lin['x_ops'][0:5]
u_lin = ABCD_lin['u_ops']
y_lin = ABCD_lin['y_ops']
    
A_error = A_lin-A_rbf

A_error0 = np.reshape(A_error,(nx*nx,nw),order = 'F')   

fig1,ax1 = plt.subplots(1)
ax1.set_ylabel('Ptfm Pitch [deg] ',fontsize = 16)
ax1.set_xlabel('Wind Speed [m/s]',fontsize = 16)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
#fig1.title('DV = [51.75,519]')

ax1.plot(u_h,np.rad2deg(x_rbf[0,:]),'o-',label = 'RGI')
ax1.plot(u_h,np.rad2deg(x_lin[0,:]),'*-',label = 'LIN')    

ax1.legend()

DV_ = np.zeros((nw,3))
DV_[:,0:2] = np.array([[cs[0],bv[0]]])
DV_[:,2] = u_h 

A_rbf1 = EvaluateModel(Asm,sp_A,nw,nx,nx,DV_)
x_rbf1 = EvaluateModel(Xsm,sp_X,nw,nx,1,DV_)

A_error = A1-A_rbf1

A_error1 = np.reshape(A_error,(nx*nx,nw),order = 'F')

fig2,ax2 = plt.subplots(1)
ax2.set_ylabel('Ptfm Pitch [deg] ',fontsize = 16)
ax2.set_xlabel('Wind Speed [m/s]',fontsize = 16)
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
#fig2.title('DV = [36,350]')
ax2.plot(u_h,np.rad2deg(x_rbf1[0,:]),'o-',label = 'RGI')
ax2.plot(u_h,np.rad2deg(x1[0,:]),'*-',label = 'LIN')    

ax2.legend()