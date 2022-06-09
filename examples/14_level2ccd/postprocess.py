import os
import numpy as np
import openmdao.api as om
import pickle
import matplotlib.pyplot as plt

# get path to pickle and sql files
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
opt_path = mydir + os.sep + "outputs" + os.sep +"jmd_single"
#case_detail_path = opt_path + os.sep + 'case_detail.pkl'
pkl_path = opt_path + os.sep+  "ABCD_matrices.pkl"

# load pickle file
with open(pkl_path, 'rb') as handle:
    ABCD_lin = pickle.load(handle)[0]
    
with open('LinModelRBF.pkl','rb')as handle:
    ABCD_rbf = pickle.load(handle)


x_rbf = ABCD_rbf['x_ops']
u_rbf = ABCD_rbf['u_ops']


x_lin = ABCD_lin['x_ops']
u_lin = ABCD_lin['u_ops']    

u_h = ABCD_lin['u_h']

# initialize plot
fig1,ax1 = plt.subplots(1,1)
ax1.set_ylabel('Controls',fontsize = 16)
ax1.set_xlabel('Wind Speed [m/s]',fontsize = 16)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

ax1.plot(u_h,np.rad2deg(x_rbf[0,:]),label = 'RBF')
ax1.plot(u_h,np.rad2deg(x_lin[0,:]),label = 'LIN')

ax1.legend()

# # load case reader
# cr = om.CaseReader(opt_path+ os.sep +"log_opt.sql")
# driver_cases = cr.get_cases('driver')

# # initialize
# DVs = []
# LCOE = []

# for idx, case in enumerate(driver_cases[:26]):
#     dvs = case.get_design_vars(scaled=False)
#     LCOE.append(case.get_objectives(scaled = False)['financese_post.lcoe'][0])
#     for key in dvs.keys():
#         DVs.append(dvs[key])

# # set length 

# n_dv = len(dvs.keys())
# u_h = ABCD_list[0]['u_h']
# # reshape into array
# DV = np.reshape(DVs,(idx+1,n_dv),order = 'C')

# DV_x = np.array([51.75,600,637.5])


# DV_mean = np.mean(DV,0)

# DV_avg = DV/DV_mean

# # iterations
# iterations = np.arange(idx+1)+1

# initialize plot
# fig1,ax1 = plt.subplots(1,1)
# ax1.set_ylabel('Controls',fontsize = 16)
# ax1.set_xlabel('Wind Speed [m/s]',fontsize = 16)
# ax1.tick_params(axis='x', labelsize=12)
# ax1.tick_params(axis='y', labelsize=12)

# ax1.plot(u_h,np.rad2deg(uw[2,:]),label = 'BldPitch')
# ax1.plot(u_h,uw[1,:]/1e6,label = 'GenTq')

# # get the name of design variables
# leg_names = list(dvs.keys())
# leg_names = ['Column Spacing','Platform Ballast Volume 0','Platform Ballast Volume 1']
# # plot each design variable 
# for i in range(n_dv):
#     ax1.plot(iterations,DV_avg[:,i],'-*',label = leg_names[i])
    
# # plot LCOE  
# #ax2 = ax1.twinx()
# fig2,ax2 = plt.subplots(1,1)
# ax2.plot(u_h,np.rad2deg(xw[0,:]),'*-',label = 'PtfmPitch')
# ax2.set_ylabel('PtfmPitch [deg]',fontsize = 16)
# ax2.set_xlabel('Wind Speed [m/s]',fontsize = 16)
# ax2.tick_params(axis='y', labelsize=12)



# #fig1.legend( loc = '',ncol = 1,fontsize = 12) # bbox_to_anchor=(1.5,1.0),
#fig1.legend(bbox_to_anchor=(0.8,0.15), loc = 'lower right',ncol = 1,fontsize = 12) 

# fig1.savefig('DV.svg',format = 'svg',dpi = 1200)
# fig2.savefig('LCOE.svg',format = 'svg',dpi = 1200)

