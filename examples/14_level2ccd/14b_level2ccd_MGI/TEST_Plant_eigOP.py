import numpy as np
from numpy.linalg import eig
import os
import openmdao.api as om
import pickle
import matplotlib.pyplot as plt
from scipy.io import savemat 
from scipy.interpolate import PchipInterpolator

def Get_indices(nl):
    if nl % 2:
        FitInd = np.arange(0,nl+1,2)
        ValInd = np.arange(1,nl-1,2)
    else:
        FitInd = np.arange(0,nl,2)
        FitInd = np.append(FitInd,nl-1)
        ValInd = np.arange(1,nl-1,2)
        
    return FitInd,ValInd
        

# get path to pickle and sql files
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
opt_path = mydir + os.sep + "outputs" + os.sep + 'pv_cs'
pkl_path = opt_path + os.sep +  "ABCD_matrices.pkl"

 # load pickle file
with open(pkl_path, 'rb') as handle:
    ABCD_list = pickle.load(handle)

# index
index = 0

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
DV = np.reshape(DVs,(idx+1,),order = 'C')

# get fitting and validation indices
FitInd,ValInd = Get_indices(len(DV))
nfit = len(FitInd); nval = len(ValInd)

# change list to array
A = np.array(A); B = np.array(B); C = np.array(C); D = np.array(D);
x = np.array(x);u = np.array(u);y = np.array(y)

# reduce dimensions of A
A = np.squeeze(A)
x = np.squeeze(x)
u = np.squeeze(u)

XUop = np.zeros((len(DV),4))

XUop[:,0] = x[:,0]
XUop[:,1] = x[:,4]
XUop[:,2] = u[:,1]/1e7
XUop[:,3] = np.rad2deg(u[:,2])

avg_ = np.max(XUop,axis=0)
#XUop = XUop/avg_

Afit = A[FitInd,:,:]
Aval = A[ValInd,:,:]

XUfit = XUop[FitInd,:]
XUval = XUop[ValInd,:]

# create interpolating function
A_pp = PchipInterpolator(DV[FitInd],Afit,axis = 0)
Aop = lambda x: A_pp(x)

XU_pp = PchipInterpolator(DV[FitInd],XUfit)
XUop = lambda x : XU_pp(x)

# find value of A
ninterp = 1000
X_ = np.linspace(DV[0],DV[-1],ninterp)

# find interpolating values
Ainterp = Aop(X_)
XUinterp = XUop(X_)

# initialize eigen value matrix
eig_interp = np.zeros((ninterp,2))
eig_fit = np.zeros((nfit,2))
eig_val = np.zeros((nval,2))


for i in range(ninterp):
    eig_,null = eig(Ainterp[i,:,:])
    
    eig_interp[i,0] = eig_[index].real 
    eig_interp[i,1] = eig_[index].imag 
    
# initialize plot
fig1,ax1 = plt.subplots(1,1)
ax1.set_ylabel('Imaginary',fontsize = 16)
ax1.set_xlabel('Real',fontsize = 16)
ax1.tick_params(axis='x', labelsize=8)
ax1.tick_params(axis='y', labelsize=8)

ax1.plot(eig_interp[:,0],eig_interp[:,1],'b-')


for i in range(nfit):
    eig_,null = eig(Afit[i,:,:])
    
    eig_fit[i,0] = eig_[index].real
    eig_fit[i,1] = eig_[index].imag 
    
ax1.plot(eig_fit[:,0],eig_fit[:,1],'ro',markersize=10)


for i in range(nval):
    eig_,null = eig(Aval[i,:,:])
    
    eig_val[i,0] = eig_[index].real
    eig_val[i,1] = eig_[index].imag 
    
ax1.plot(eig_val[:,0],eig_val[:,1],'ro',markersize=10,alpha = 0.5)

# initialize plot
fig2,ax2 = plt.subplots(1,1)
ax2.set_ylabel('DV',fontsize = 16)
ax2.set_xlabel('Operating Points',fontsize = 16)
ax2.tick_params(axis='x', labelsize=8)
ax2.tick_params(axis='y', labelsize=8)

ax2.plot(X_,XUinterp,'b-')
ax2.plot(DV[FitInd],XUfit,'ro',markersize=10)
ax2.plot(DV[ValInd],XUval,'ro',markersize=10,alpha = 0.5)