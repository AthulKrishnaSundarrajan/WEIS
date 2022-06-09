#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:20:35 2022

@author: athulsun
"""
import pickle 
import matplotlib.pyplot as plt
import numpy as np

pklname = 'q8_review.pkl'

with open(pklname,'rb') as handle:
    Results1 = pickle.load(handle)

#Results = [Results1[0],Results1[4],Results1[2]]
n_cases = len(Results1)


# initialize plot
fig1,ax1 = plt.subplots(1,1)
ax1.set_ylabel('Wind Speed [m/s]',fontsize = 16)
ax1.set_xlabel('Time [s]',fontsize = 16)
ax1.set_xlim([0,800])
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

# initialize plot
fig2,ax2 = plt.subplots(1,1)
ax2.set_ylabel('Generator Torque [MWm]',fontsize = 16)
ax2.set_xlabel('Time [s]',fontsize = 16)
ax2.set_xlim([0,800])
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)

# initialize plot
fig3,ax3 = plt.subplots(1,1)
ax3.set_ylabel('Blade Pitch [deg]',fontsize = 16)
ax3.set_xlabel('Time [s]',fontsize = 16)
ax3.set_xlim([0,800])
ax3.tick_params(axis='x', labelsize=12)
# initialize plot
fig4,ax4 = plt.subplots(1,1)
ax4.set_ylabel('Platform Pitch [deg]',fontsize = 16)
ax4.set_xlabel('Time [s]',fontsize = 16)
ax4.set_xlim([0,800])
ax4.tick_params(axis='x', labelsize=12)
ax4.tick_params(axis='y', labelsize=12)

# initialize plot
fig5,ax5 = plt.subplots(1,1)
ax5.set_ylabel('Generator Speed [rad/s]',fontsize = 16)
ax5.set_xlabel('Time [s]',fontsize = 16)
ax5.set_xlim([0,800])
ax5.tick_params(axis='x', labelsize=12)
ax5.tick_params(axis='y', labelsize=12)
ax5.tick_params(axis='y', labelsize=12)

fig6,ax6 = plt.subplots(1,1)
ax6.set_ylabel('Generator Power [MW]',fontsize = 16)
ax6.set_xlabel('Time [s]',fontsize = 16)
ax6.set_xlim([0,800])
ax6.tick_params(axis='x', labelsize=12)
ax6.tick_params(axis='y', labelsize=12)
ax6.tick_params(axis='y', labelsize=12)

fig7,ax7 = plt.subplots(1,1)
ax7.set_ylabel("ED TwrBsFxt, [MN]",fontsize = 16)
ax7.set_xlabel('Time [s]',fontsize = 16)
ax7.set_xlim([0,800])
ax7.tick_params(axis='x', labelsize=12)
ax7.tick_params(axis='y', labelsize=12)
ax7.tick_params(axis='y', labelsize=12)

fig8,ax8 = plt.subplots(1,1)
ax8.set_ylabel('ED TwrBsMxt, [MN-m]',fontsize = 16)
ax8.set_xlabel('Time [s]',fontsize = 16)
ax8.set_xlim([0,800])
ax8.tick_params(axis='x', labelsize=12)
ax8.tick_params(axis='y', labelsize=12)
ax8.tick_params(axis='y', labelsize=12)

labels = ['Below Rated','Transition Region','Rated']

OutputNames = Results1[0]['Outputnames']
indpwr = OutputNames.index('SrvD GenPwr, (kW)')
indFA = OutputNames.index("ED TwrBsFxt, (kN)")
indMA  = OutputNames.index("ED TwrBsMxt, (kN-m)")
for i in range(n_cases-1):
    
    X = Results1[i]['States']
    U = Results1[i]['Controls']
    Y = Results1[i]['Outputs']
    
    
    PtfmPitch = np.rad2deg(X[:,0])
    Wind = U[:,0]
    GenTorq = U[:,1]
    BldPitch = np.rad2deg(U[:,2])
    GenSpeed = X[:,4]
    
    GenPwr = Y[:,indpwr]
    TwrBsFxt = Y[:,indFA]
    TwrBsMxt = Y[:,indMA]

    Time = Results1[i]['Time']
    
    #ax1.plot(Time,Wind,linewidth = 2,markersize = 10,label = labels[i] )
    #ax2.plot(Time,GenTorq/1e7,linewidth = 2,markersize = 10,label =labels[i] )
    # ax3.plot(Time,BldPitch,linewidth = 2,markersize = 10,label =labels[i] )
    
    ax4.plot(Time,PtfmPitch,linewidth = 2,markersize = 10,label =labels[i] )
    ax4.hlines(4,xmin = 0, xmax = 800,color = 'k')
    # ax5.plot(Time,GenSpeed,linewidth = 2,markersize = 10,label =labels[i] )
    
    # ax6.plot(Time,GenPwr/1000,linewidth = 2,markersize = 10,label = labels[i] )
    # ax7.plot(Time,TwrBsFxt/1000,linewidth = 2,markersize = 10,label =labels[i] )
    # ax7.hlines(5,xmin = 0, xmax = 800,color = 'k')
    # ax8.plot(Time,TwrBsMxt/1000,linewidth = 2,markersize = 10,label =labels[i] )
    # ax8.hlines(35,xmin = 0, xmax = 800,color = 'k')
    

    
#fig1.legend(bbox_to_anchor=(0.5,1), loc='upper center',ncol = 3,fontsize = 12) 
# fig2.legend(bbox_to_anchor=(0.5,1), loc='upper center',ncol = 3,fontsize = 12) 
# fig3.legend(bbox_to_anchor=(0.5,1), loc='upper center',ncol = 3,fontsize = 12) 
fig4.legend(bbox_to_anchor=(0.5,1), loc='upper center',ncol = 3,fontsize = 12) 
# fig5.legend(bbox_to_anchor=(0.5,1), loc='upper center',ncol = 3,fontsize = 12) 
# fig6.legend(bbox_to_anchor=(0.5,1), loc='upper center',ncol = 3,fontsize = 12) 
# fig7.legend(bbox_to_anchor=(0.5,1), loc='upper center',ncol = 3,fontsize = 12) 
# fig8.legend(bbox_to_anchor=(0.5,1), loc='upper center',ncol = 3,fontsize = 12) 

# breakpoint()
    
#fig1.savefig('wind.svg',format = 'svg',dpi = 1200)
# fig2.savefig('GenToqr.svg',format = 'svg',dpi = 1200)
# fig3.savefig('BldPitch.svg',format = 'svg',dpi = 1200)
fig4.savefig('PtfmPitch.svg',format = 'svg',dpi = 1200)
# fig5.savefig('GenSpeed.svg',format = 'svg',dpi = 1200)
# fig6.savefig('GenPwr.svg',format = 'svg',dpi = 1200)
# fig7.savefig('TwrBsFxt.svg',format = 'svg',dpi = 1200)
# fig8.savefig('TwrBsMxt.svg',format = 'svg',dpi = 1200)