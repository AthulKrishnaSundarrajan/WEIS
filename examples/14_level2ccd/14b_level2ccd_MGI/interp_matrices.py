import os
import numpy as np
import openmdao.api as om
import pickle
import matplotlib.pyplot as plt
from scipy.sparse import find 
from scipy.interpolate import LinearNDInterpolator,RBFInterpolator,NearestNDInterpolator

def BuildSurrogate(data,values):
    
    return LinearNDInterpolator(data,values)

class LinearModel:
    def __init__(self,A,B,C,D,x,u,y,DescStates,DescControl,DescOutput,DV,mtype = None):
        self.A = A; self.B = B; self.C = C; self.D = D;
        self.x = x; self.u = u; self.y = y;
        self.DescStates = DescStates; self.DescControl = DescControl;
        self.DescOutput = DescOutput; self.DV = DV;self.mtype = mtype
        
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

# get shape
ns,nx,nu,nw = np.shape(B); ny = len(DescOutput)

A1 = np.reshape(A[0,:,:,0],(nx*nx),order = 'F')
B1 = np.reshape(B[0,:,:,0],(nx*nu),order = 'F')
C1 = np.reshape(C[0,:,:,0],(ny*nx),order = 'F')
D1 = np.reshape(D[0,:,:,0],(ny*nu),order = 'F')

Aind = find(np.abs(A1)>1e-6); Bind = find(np.abs(B1)>1e-6); Cind = find(np.abs(C1)>1e-6); Dind = find(np.abs(D1)>1e-6)
x1 = x[0,:,:]; u1 = u[0,:,0]; y1 = y[0,:,0]

# permute and reshape matrices
A = np.transpose(A,[0,3,1,2]); A = np.reshape(A,(ns*nw,nx*nx),order = 'F'); Amax = np.max(A,axis=0)
B = np.transpose(B,[0,3,1,2]); B = np.reshape(B,(ns*nw,nx*nu),order = 'F')
C = np.transpose(C,[0,3,1,2]); C = np.reshape(C,(ns*nw,ny*nx),order = 'F')
D = np.transpose(D,[0,3,1,2]); D = np.reshape(D,(ns*nw,ny*nu),order = 'F')

x = np.transpose(x,[0,2,1]);x = np.reshape(x,(ns*nw,nx),order = 'F')
u = np.transpose(u,[0,2,1]);u = np.reshape(u,(ns*nw,nu),order = 'F')
y = np.transpose(y,[0,2,1]); y = np.reshape(y,(ns*nw,ny),order = 'F')
  
# number of linmat
n_case = ns*nw

data = np.zeros((n_case,3))

l = 0
for i in range(ns):
    for j in range(nw):
        data[l,0:2] = DV[i,:]
        data[l,2] = u_h[j]
        l+=1

dvmax = np.max(data,axis=0)
data_ = data/dvmax
# construct mode

Asm = np.zeros((nx*nx),dtype='O')
xind = Aind[0]
yind = Aind[1]
tval = Aind[2]

sp_ind = np.zeros((nx*nx))
sp_ind[yind] = tval

for i in range(nx*nx):   
    
    if sp_ind[i]:

        values_ = A[:,i]
        
        Asm[i] = BuildSurrogate(data_,values_)

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
    
DV_ = np.zeros((nw,3))
DV_[:,0:2] = np.array([[51.75, 519]])
DV_[:,2] = u_h    

data_ = DV_/dvmax

x_lin = x1

A_rbf = np.zeros((nx*nx,nw))

for i in range(nx*nx):
   
    if sp_ind[i]:
        
        values_ = Asm[i](data_)
        A_rbf[i,:] = values_
        
    else:
        A_rbf[i,:] = 0    

A_lin = np.reshape(A_lin,(nx*nx,nw),order = 'F')   
# fig1,ax1 = plt.subplots(1)
# ax1.set_ylabel('Ptfm Pitch [deg] ',fontsize = 16)
# ax1.set_xlabel('Wind Speed [m/s]',fontsize = 16)
# ax1.tick_params(axis='x', labelsize=12)
# ax1.tick_params(axis='y', labelsize=12)

# ax1.plot(u_h,np.rad2deg(x_rbf[0,:]),'o-',label = 'LIN-ND')
# ax1.plot(u_h,np.rad2deg(x_lin[0,:]),'*-',label = 'LIN')    

# ax1.legend()
    
# fig3,ax3 = plt.subplots(1)
# ax3.set_ylabel('Gen Speed [rad/s] ',fontsize = 16)
# ax3.set_xlabel('Wind Speed [m/s]',fontsize = 16)
# ax3.tick_params(axis='x', labelsize=12)
# ax3.tick_params(axis='y', labelsize=12)

# ax3.plot(u_h,x_rbf[4,:],'o-',label = 'LIN-ND')
# ax3.plot(u_h,x_lin[4,:],'*-',label = 'LIN')

# ax3.legend()    
    
    
