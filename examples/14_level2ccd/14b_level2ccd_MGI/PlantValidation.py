"""
Different validation test
"""

import numpy as np
from control import ss
import matplotlib.pyplot as plt
from scipy.sparse import find
from scipy.interpolate import RegularGridInterpolator

import time
import os
import openmdao.api as om
import pickle
from scipy.io import savemat 


def Get_indices(nl):
    if nl % 2:
        FitInd = np.arange(0,nl+1,2)
        ValInd = np.arange(1,nl-1,2)
    else:
        FitInd = np.arange(0,nl,2)
        FitInd = np.append(FitInd,nl-1)
        ValInd = np.arange(1,nl-1,2)
        
    return FitInd,ValInd

class LinearModel:
    def __init__(self,A,B,C,D,x,u,y,DescStates,DescControl,DescOutput,DV,mtype = None):
        self.A = A; self.B = B; self.C = C; self.D = D;
        self.x = x; self.u = u; self.y = y;
        self.DescStates = DescStates; self.DescControl = DescControl;
        self.DescOutput = DescOutput; self.DV = DV;self.mtype = mtype
        

        
def EvalHinf(A_sm,B_sm,C_sm,D_sm,DescCntrlInpt,DescOutput,data):
    
    A = EvaluateModel(A_sm,data)
    B = EvaluateModel(B_sm,data)
    C = EvaluateModel(C_sm,data)
    D = EvaluateModel(D_sm,data)
    
    FR = Hinf(A,B,C,D,DescCntrlInpt,DescOutput)
    
    return FR

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


def Hinf(A,B,C,D,DescCntrlInpt,DescOutput):
    
    # output and input indices
    #Out_ind = DescOutput.index('ED GenSpeed, (rpm)')
    Out_ind = DescOutput.index('ED GenSpeed, (rpm)')
    In_ind = DescCntrlInpt.index('ED Extended input: collective blade-pitch command, rad')
    
    #
    omega = np.logspace(-3,2,1000)
   
    # construct state space
    sys = ss(A,B[:,In_ind],C[Out_ind,:],D[Out_ind,In_ind])
    
    # calculate frequency response
    mag,phase,om = sys.freqresp(omega)
    
    # frequency 
    FR = np.squeeze(mag)*np.cos(phase) + 1j*np.squeeze(mag)*np.sin(phase)
    
    return FR
        
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

    
if __name__ == '__main__':
    
    # load matrices and DV from the output folder
    Afit,Bfit,Cfit,Dfit,xfit,ufit,yfit,u_h,DV,DescStates,DescCntrlInpt,DescOutput = LoadMatrices('jmd_full')
    cs = np.unique(DV[:,0]); bv = np.unique(DV[:,1])
   
    # get shape
    ns,nx,nu,nw = np.shape(Bfit); ny = len(DescOutput)
    
    # number of linmat
    nw = len(u_h);
    
    # set data
    if nw == 1:
        data = (cs,bv)
    elif nw >1 :
        data  = (cs,bv,u_h)
     
    # construct surrogate model
    Asm,sp_A = BuildSurrogate(Afit,data)
    Bsm,sp_B = BuildSurrogate(Bfit,data)
    Csm,sp_C = BuildSurrogate(Cfit,data)
    Dsm,sp_D = BuildSurrogate(Dfit,data)
    
    Xsm,sp_X = BuildSurrogate(xfit,data) 
    Usm,sp_U = BuildSurrogate(ufit,data)
    Ysm,sp_Y = BuildSurrogate(yfit,data)
    
    # load validation matrices
    Aval,Bval,Cval,Dval,xval,uval,yval,u_h,sample,DescStates,DescCntrlInpt,DescOutput = LoadMatrices('pv_sample25')    

    # initialize plot
    fig1,ax1 = plt.subplots(1,1)
    ax1.set_ylabel('Column Spacing [m]',fontsize = 16)
    ax1.set_xlabel('Ballast Volume [m^3]',fontsize = 16)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.invert_yaxis()
    
    ax1.scatter(DV[:,0],DV[:,1],marker='*',label = 'Original Grid')
    ax1.scatter(sample[:,0],sample[:,1],marker = 'o',label = 'Validation Points')
    fig1.legend(bbox_to_anchor=(0.5,1), loc='upper center',ncol = 3,fontsize = 12) 
    
    w_ = 12
    w_ind = (u_h == w_)
    
    n_sample = len(sample)
    DV_ = np.zeros((n_sample,3))
    DV_[:,0:2] = sample
    DV_[:,2] = w_
    
    # extract fitting matrices
    Af = np.zeros((n_sample,nx,nx,1))
    Bf = np.zeros((n_sample,nx,nu,1))
    Cf = np.zeros((n_sample,ny,nx,1))
    Df = np.zeros((n_sample,ny,nu,1))
    
    for i in range(n_sample):
        Af[i,:,:,:] = EvaluateModel(Asm,sp_A,1,nx,nx,DV_[i,:])
        Bf[i,:,:,:] = EvaluateModel(Bsm,sp_B,1,nx,nu,DV_[i,:])
        Cf[i,:,:,:] = EvaluateModel(Csm,sp_C,1,ny,nx,DV_[i,:])
        Df[i,:,:,:] = EvaluateModel(Dsm,sp_D,1,ny,nu,DV_[i,:])
             
    CS,BV = np.meshgrid(cs,bv)
    
    HinfError = np.zeros((n_sample,))
    nval = 0
    
    for i in range(n_sample):
             
        Av = np.squeeze(Aval[i,:,:,w_ind]);Bv = np.squeeze(Bval[i,:,:,w_ind]);Cv = np.squeeze(Cval[i,:,:,w_ind]);Dv = np.squeeze(Dval[i,:,:,w_ind])
        A = np.squeeze(Af[i,:,:,0]);B = np.squeeze(Bf[i,:,:,0]);C = np.squeeze(Cf[i,:,:,0]);D = np.squeeze(Df[i,:,:,0]);
        
        FR_act = Hinf(Av,Bv,Cv,Dv,DescCntrlInpt,DescOutput)
        FR_int = Hinf(A,B,C,D,DescCntrlInpt,DescOutput)
        
        FR_diff = FR_act - FR_int
        
        HinfError[i] = np.nanmax(np.abs(FR_diff))/1000

    # save results to a dict    
    results_hinf = {'CS':CS,'BV':BV,'Hinf':HinfError,'DV':DV,'Samples':sample}
    matname = 'HinfResults.mat'
    
    
    saveflag = 1;
    
    if saveflag:        
        savemat(matname,results_hinf)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
